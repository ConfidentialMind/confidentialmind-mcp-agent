import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Literal, Optional

import typer
from confidentialmind_core.config_manager import load_environment

from agent.connectors import ConnectorConfigManager
from src.agent.agent import Agent
from src.agent.database import Database, DatabaseSettings, fetch_db_url
from src.agent.llm import LLMConnector
from src.agent.transport import TransportManager

logger = logging.getLogger("fastmcp_agent")

load_environment()

app = typer.Typer(
    name="fastmcp_agent",
    help="FastMCP agent for interacting with MCP servers",
    add_completion=False,
)


async def run_agent(
    query: str,
    session_id: Optional[str] = None,
    db_config_id: str = "DATABASE",
    llm_config_id: str = "LLM",
    mcp_servers: Optional[Dict[str, str]] = None,
    mode: Literal["cli", "api"] = "cli",
    use_module: bool = True,
    debug: bool = False,
):
    """Run the agent with the given query."""
    # Set default MCP servers if not provided
    if mcp_servers is None:
        # Default to streamable HTTP transport URL
        default_mcp_server = os.environ.get("AGENT_TOOLS_URL", "http://localhost:8080/mcp")

        # Handle legacy SSE URLs
        if default_mcp_server.endswith("/sse"):
            default_mcp_server = default_mcp_server.rsplit("/sse", 1)[0] + "/mcp"
            logger.info(f"Converting legacy SSE URL to streamable HTTP: {default_mcp_server}")

        mcp_servers = {"agentTools": default_mcp_server}

    # Initialize database
    db_url = await fetch_db_url(db_config_id)
    db_settings = DatabaseSettings()
    database = Database(db_settings)
    success = await database.connect(db_url)
    if not success:
        logger.error("Failed to connect to database")
        typer.secho(
            "ERROR: Failed to connect to database. Check configuration and try again.",
            fg=typer.colors.RED,
        )
        return

    # Initialize schema if needed
    success = await database.ensure_schema()
    if not success:
        logger.warning("Could not ensure database schema. Some operations might fail.")
        typer.secho(
            "WARNING: Could not ensure database schema. Some operations might fail.",
            fg=typer.colors.YELLOW,
        )

    # Initialize LLM connector
    llm_connector = LLMConnector(llm_config_id)
    success = await llm_connector.initialize()
    if not success:
        logger.error("Failed to initialize LLM connector")
        typer.secho(
            "ERROR: Failed to initialize LLM connector. Check configuration and try again.",
            fg=typer.colors.RED,
        )
        return

    # Create and configure transport manager
    transport_manager = TransportManager(mode=mode)

    # Configure transports based on mode
    for server_id, server_ref in mcp_servers.items():
        try:
            if mode == "cli":
                transport_manager.configure_transport(
                    server_id,
                    server_path=server_ref,
                    use_module=use_module,
                )
            else:  # api mode
                transport_manager.configure_transport(
                    server_id,
                    server_url=server_ref,
                )
        except Exception as e:
            logger.error(f"Error configuring transport for {server_id}: {e}")
            typer.secho(
                f"ERROR: Failed to configure transport for {server_id}. {e}", fg=typer.colors.RED
            )
            return

    # Generate a session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {session_id}")

    # Create and run agent
    agent = Agent(database, llm_connector, transport_manager, debug=debug)

    try:
        await agent.initialize()

        async with agent:
            # Execute the agent run
            state = await agent.run(query, session_id)

            # Print response
            if state.error:
                typer.secho(f"Error encountered: {state.error}", fg=typer.colors.RED)
                if state.response:
                    typer.echo(f"Response: {state.response}")
            elif state.response:
                typer.echo(f"Response: {state.response}")
            else:
                typer.secho("No response generated.", fg=typer.colors.YELLOW)

            # For debugging: print thoughts
            if debug and state.thoughts:
                typer.secho("\nAgent thoughts:", fg=typer.colors.BLUE)
                for i, thought in enumerate(state.thoughts):
                    typer.echo(f"{i + 1}. {thought}")

            # Return the final state
            return state

    finally:
        # Clean up resources
        await database.disconnect()
        await llm_connector.close()

        logger.info("Agent resources cleaned up")


def load_config_file(config_path: Optional[str]) -> Optional[Dict[str, Dict[str, str]]]:
    """Load configuration from a JSON file."""
    if not config_path:
        return None

    try:
        config_file = Path(config_path)
        if not config_file.exists():
            typer.secho(f"Config file not found: {config_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        with open(config_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        typer.secho(f"Invalid JSON in config file: {config_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"Error loading config file: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def setup_logging(debug: bool):
    """Set up logging with the appropriate level."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@app.command()
def query(
    query_text: str = typer.Argument(..., help="The query to run."),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID to use."),
    db: str = typer.Option("DATABASE", "--db", "-d", help="Database config ID."),
    llm: str = typer.Option("LLM", "--llm", "-l", help="LLM config ID."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
):
    """Run a query in CLI mode."""
    setup_logging(debug)

    # Load config file if provided
    config_data = load_config_file(config)
    mcp_servers = config_data.get("mcp_servers") if config_data else None

    try:
        asyncio.run(
            run_agent(
                query=query_text,
                session_id=session,
                db_config_id=db,
                llm_config_id=llm,
                mcp_servers=mcp_servers,
                mode="cli",
                debug=debug,
            )
        )
    except KeyboardInterrupt:
        typer.secho("Agent execution interrupted by user.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        typer.secho(f"ERROR: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="API server host."),
    port: int = typer.Option(8000, "--port", "-p", help="API server port."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file."),
    db: str = typer.Option("DATABASE", "--db", "-d", help="Database config ID."),
    llm: str = typer.Option("LLM", "--llm", "-l", help="LLM config ID."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
):
    """Run the agent as an API server."""
    setup_logging(debug)

    # Initialize connector configuration
    asyncio.run(ConnectorConfigManager().initialize())

    # Load config file if provided
    config_data = load_config_file(config)
    mcp_servers = config_data.get("mcp_servers") if config_data else None

    # Process MCP server URLs to convert any legacy SSE URLs
    if mcp_servers:
        updated_servers = {}
        for server_id, url in mcp_servers.items():
            if url.endswith("/sse"):
                updated_url = url.rsplit("/sse", 1)[0] + "/mcp"
                logger.info(
                    f"Converting legacy SSE URL to streamable HTTP for {server_id}: {updated_url}"
                )
                updated_servers[server_id] = updated_url
            else:
                updated_servers[server_id] = url

        # Set environment variables with updated URLs
        for server_id, url in updated_servers.items():
            os.environ[f"MCP_SERVER_{server_id.upper()}"] = url
    else:
        # Check if any environment variables need conversion
        for key, value in os.environ.items():
            if key.startswith("MCP_SERVER_") and value and value.endswith("/sse"):
                updated_url = value.rsplit("/sse", 1)[0] + "/mcp"
                logger.info(
                    f"Converting legacy SSE URL to streamable HTTP for {key}: {updated_url}"
                )
                os.environ[key] = updated_url

    # Set other environment variables
    if db:
        os.environ["DB_CONFIG_ID"] = db
    if llm:
        os.environ["LLM_CONFIG_ID"] = llm

    try:
        from src.agent.api import start_api_server

        typer.secho(f"Starting API server on {host}:{port}", fg=typer.colors.GREEN)
        start_api_server(host=host, port=port, log_level="debug" if debug else "info")
    except ImportError as e:
        typer.secho(f"Error starting API server: {e}", fg=typer.colors.RED)
        typer.secho("Make sure all required dependencies are installed.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"Error starting API server: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def clear_history(
    session: str = typer.Argument(..., help="Session ID to clear history for."),
    db: str = typer.Option("DATABASE", "--db", "-d", help="Database config ID."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
):
    """Clear the conversation history for a session."""
    setup_logging(debug)

    async def _clear_history():
        # Initialize database
        db_url = await fetch_db_url(db)
        db_settings = DatabaseSettings()
        database = Database(db_settings)
        success = await database.connect(db_url)
        if not success:
            typer.secho("ERROR: Failed to connect to database.", fg=typer.colors.RED)
            return

        try:
            success = await database.clear_history(session)
            if success:
                typer.secho(
                    f"Successfully cleared history for session {session}", fg=typer.colors.GREEN
                )
            else:
                typer.secho(f"Failed to clear history for session {session}", fg=typer.colors.RED)
        finally:
            await database.disconnect()

    try:
        asyncio.run(_clear_history())
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
