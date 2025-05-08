import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Dict, Optional

from confidentialmind_core.config_manager import load_environment

from .agent import Agent
from .database import Database, DatabaseSettings, fetch_db_url
from .llm import LLMConnector

logger = logging.getLogger("fastmcp_agent")

load_environment()


async def main(
    query: str,
    session_id: Optional[str] = None,
    db_config_id: str = "DATABASE",
    llm_config_id: str = "LLM",
    mcp_servers: Optional[Dict[str, str]] = None,
    debug: bool = False,
):
    """Run the agent with the given query."""
    # Set default MCP servers if not provided
    if mcp_servers is None:
        mcp_servers = {
            "agentTools": os.environ.get("AGENT_TOOLS_URL", "http://localhost:8000/sse"),
        }

    # Initialize database
    db_url = await fetch_db_url(db_config_id)
    db_settings = DatabaseSettings()
    database = Database(db_settings)
    success = await database.connect(db_url)
    if not success:
        logger.error("Failed to connect to database")
        print("ERROR: Failed to connect to database. Check configuration and try again.")
        return

    # Initialize schema if needed
    success = await database.ensure_schema()
    if not success:
        logger.warning("Could not ensure database schema. Some operations might fail.")

    # Initialize LLM connector
    llm_connector = LLMConnector(llm_config_id)
    success = await llm_connector.initialize()
    if not success:
        logger.error("Failed to initialize LLM connector")
        print("ERROR: Failed to initialize LLM connector. Check configuration and try again.")
        return

    # Generate a session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {session_id}")

    # Create and run agent
    agent = Agent(database, llm_connector, mcp_servers, debug=debug)

    try:
        await agent.initialize()

        async with agent:
            # Execute the agent run
            state = await agent.run(query, session_id)

            # Print response
            if state.error:
                print(f"Error encountered: {state.error}")
                if state.response:
                    print(f"Response: {state.response}")
            elif state.response:
                print(f"Response: {state.response}")
            else:
                print("No response generated.")

            # For debugging: print thoughts
            if debug and state.thoughts:
                print("\nAgent thoughts:")
                for i, thought in enumerate(state.thoughts):
                    print(f"{i + 1}. {thought}")

            # Return the final state
            return state

    finally:
        # Clean up resources
        await database.disconnect()
        await llm_connector.close()

        logger.info("Agent resources cleaned up")


def run_cli():
    """Run the agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the FastMCP agent.")
    parser.add_argument("query", help="The query to run.")
    parser.add_argument("--session", help="Session ID to use (optional).")
    parser.add_argument("--db", help="Database config ID.", default="DATABASE")
    parser.add_argument("--llm", help="LLM config ID.", default="LLM")
    parser.add_argument("--config", help="Path to config file (optional).")
    parser.add_argument("--debug", help="Enable debug logging.", action="store_true")
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load config file if provided
    mcp_servers = None
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
                mcp_servers = config.get("mcp_servers")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            print(f"Error loading config file: {e}")

    try:
        asyncio.run(
            main(
                query=args.query,
                session_id=args.session,
                db_config_id=args.db,
                llm_config_id=args.llm,
                mcp_servers=mcp_servers,
                debug=args.debug,
            )
        )
    except KeyboardInterrupt:
        print("Agent execution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
