import argparse
import atexit
import json
import logging
import os
import sys
from typing import Dict

from src.connectors.llm import LLMConnector
from src.core.agent import Agent
from src.mcp.mcp_client import MCPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("mcp_integration")

# Global client instances to ensure cleanup
mcp_client_instances: Dict[str, MCPClient] = {}


def cleanup_mcp_clients():
    """Function registered with atexit to stop all MCP servers."""
    global mcp_client_instances
    for server_id, client in mcp_client_instances.items():
        if client:
            logger.info(f"Running cleanup: Stopping {server_id} MCP server...")
            try:
                client.stop_server()
                logger.info(f"{server_id} MCP server stopped successfully during cleanup.")
            except Exception as e:
                logger.error(
                    f"Error stopping {server_id} MCP server during cleanup: {e}", exc_info=True
                )


# Register the cleanup function
atexit.register(cleanup_mcp_clients)


def create_postgres_mcp_client() -> MCPClient:
    """Creates and starts the MCPClient for the PostgreSQL server."""
    global mcp_client_instances
    if "postgres" in mcp_client_instances:
        logger.warning("PostgreSQL MCPClient instance already exists.")
        return mcp_client_instances["postgres"]

    postgres_conn_string = os.environ.get("PG_CONNECTION_STRING")
    if not postgres_conn_string:
        logger.error("FATAL: PG_CONNECTION_STRING environment variable is not set.")
        raise ValueError("PG_CONNECTION_STRING must be set in the environment.")

    # Define the command to run the MCP server
    # First check the expected path relative to this script
    server_script_path = os.path.join(
        os.path.dirname(__file__), "src", "mcp", "postgres_mcp_server.py"
    )
    if not os.path.exists(server_script_path):
        # Then check alternative paths
        for alt_path in ["src/mcp/postgres_mcp_server.py", "postgres_mcp_server.py"]:
            if os.path.exists(alt_path):
                server_script_path = alt_path
                break
        else:
            logger.warning(f"PostgreSQL MCP server script not found at any expected path")
            # Proceed with the original path as a last resort
            server_script_path = "postgres_mcp_server.py"

    server_command = f"{sys.executable} {server_script_path} {postgres_conn_string}"

    try:
        logger.info("Initializing PostgreSQL MCPClient...")
        # Create and store the client instance globally for cleanup
        client = MCPClient(server_command=server_command)
        mcp_client_instances["postgres"] = client
        logger.info("PostgreSQL MCPClient initialized and server process start initiated.")
        return client
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize PostgreSQL MCPClient: {e}", exc_info=True)
        # Ensure cleanup is attempted even if constructor fails partially
        cleanup_mcp_clients()
        raise  # Re-raise the exception


def create_rag_mcp_client() -> MCPClient:
    """Creates and starts the MCPClient for the RAG server."""
    global mcp_client_instances
    if "rag" in mcp_client_instances:
        logger.warning("RAG MCPClient instance already exists.")
        return mcp_client_instances["rag"]

    rag_api_url = os.environ.get("RAG_API_URL")
    if not rag_api_url:
        logger.error("FATAL: RAG_API_URL environment variable is not set.")
        raise ValueError("RAG_API_URL must be set in the environment.")

    # Get API key if available
    rag_api_key = os.environ.get("RAG_API_KEY", "")

    # Define the command to run the MCP server
    # First check the expected path relative to this script
    server_script_path = os.path.join(os.path.dirname(__file__), "src", "mcp", "rag_mcp_server.py")
    if not os.path.exists(server_script_path):
        # Then check alternative paths
        for alt_path in ["src/mcp/rag_mcp_server.py", "rag_mcp_server.py"]:
            if os.path.exists(alt_path):
                server_script_path = alt_path
                break
        else:
            logger.warning(f"RAG MCP server script not found at any expected path")
            # Proceed with the original path as a last resort
            server_script_path = "rag_mcp_server.py"

    # Include API key in command if available
    server_command = f"{sys.executable} {server_script_path} {rag_api_url}"
    if rag_api_key:
        server_command += f" {rag_api_key}"
        logger.info("RAG API Key provided for authentication")

    try:
        logger.info("Initializing RAG MCPClient...")
        # Create and store the client instance globally for cleanup
        client = MCPClient(server_command=server_command)
        mcp_client_instances["rag"] = client
        logger.info("RAG MCPClient initialized and server process start initiated.")
        return client
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize RAG MCPClient: {e}", exc_info=True)
        # Ensure cleanup is attempted even if constructor fails partially
        cleanup_mcp_clients()
        raise  # Re-raise the exception


def create_mcp_clients() -> Dict[str, MCPClient]:
    """Creates and starts MCPClients for all supported services."""
    clients = {}

    # Create PostgreSQL client if configured
    if os.environ.get("PG_CONNECTION_STRING"):
        try:
            pg_client = create_postgres_mcp_client()
            clients["postgres"] = pg_client
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL client: {e}")
    else:
        logger.warning("PG_CONNECTION_STRING not set, skipping PostgreSQL MCP client creation")

    # Create RAG client if configured
    if os.environ.get("RAG_API_URL"):
        try:
            rag_client = create_rag_mcp_client()
            clients["rag"] = rag_client
        except Exception as e:
            logger.error(f"Failed to create RAG client: {e}")
    else:
        logger.warning("RAG_API_URL not set, skipping RAG MCP client creation")

    if not clients:
        raise ValueError("No MCP clients could be created. Check your configuration.")

    return clients


def setup_with_env_vars() -> Agent:
    """Set up agent using environment variables for LLM and MCPClient for tools."""
    logger.info("Setting up Agent using environment variables for LLM connector.")

    # 1. LLM Connector Setup from Environment Variables
    llm_url = os.environ.get("LLM_URL", "http://localhost:8080/v1")
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_connector = LLMConnector(
        base_url=llm_url, headers={"Authorization": f"Bearer {llm_api_key}"}
    )
    logger.info(f"LLMConnector created from environment variables (URL: {llm_url}).")

    # 2. Get MCP Clients (manages the separate server processes)
    mcp_clients = create_mcp_clients()

    if not mcp_clients:
        raise ValueError("No MCP clients could be initialized. Cannot proceed.")

    # 3. Agent Initialization with dictionary of clients
    agent = Agent(llm_connector, mcp_clients, debug=True)
    logger.info(f"Agent initialized with ENV LLMConnector and {len(mcp_clients)} MCP clients.")
    return agent


def print_mcp_info(clients: Dict[str, MCPClient]) -> None:
    """Prints information discovered via the MCPClients."""
    for server_id, client in clients.items():
        print(f"\n--- MCP Server Information ('{server_id}') ---")
        try:
            print("Listing tools:")
            tools_resp = client.list_tools()
            print(json.dumps(tools_resp, indent=2))
        except Exception as e:
            print(f"  Error listing tools: {e}")
            logger.warning(f"Failed to list tools via {server_id} MCPClient: {e}", exc_info=True)

        try:
            print(f"\nListing resources ({server_id}):")
            res_resp = client.list_resources()
            print(json.dumps(res_resp, indent=2))
        except Exception as e:
            print(f"  Error listing resources: {e}")
            logger.warning(
                f"Failed to list resources via {server_id} MCPClient: {e}", exc_info=True
            )
        print(f"--- End {server_id} MCP Info ---")


def run_query(agent: Agent, query: str) -> None:
    """Run a query using the agent (which uses MCPClient)."""
    print(f"\nRunning query: {query}")
    logger.info("======== STARTING NEW QUERY ========")
    try:
        logger.info("Executing main agent workflow via agent.run()")
        result_state = agent.run(query)  # Pass the raw query

        # Print Debug info first
        print("\n--- Agent Thoughts ---")
        for i, thought in enumerate(result_state.thoughts):
            print(f"{i + 1}. {thought}")
        print("--- End Thoughts ---")

        print("\n--- MCP Results Collected ---")
        # Use default=str to handle non-serializable types like datetime
        print(json.dumps(result_state.mcp_results, indent=2, default=str))
        print("--- End MCP Results ---")

        # Print error if exists
        if result_state.error:
            print(f"\nAgent encountered an error state: {result_state.error}")
            logger.error(f"Agent workflow finished with error state: {result_state.error}")

        # Print the final response LAST
        print("\n--- Agent Final Response ---")
        print(result_state.response or "Agent did not produce a response.")
        print("--- End Agent Response ---")

    except Exception as e:
        print(f"\nFATAL ERROR during agent execution: {e}")
        logger.error(
            f"Unhandled exception in agent.run() or result processing: {e}",
            exc_info=True,
        )


def run_interactive_session(agent: Agent) -> None:
    """Run an interactive session with the agent"""
    print("\nStarting interactive session. Type 'exit' to quit.")
    while True:
        try:
            query = input("\nEnter your query: ")
            if query.lower() in ("exit", "quit"):
                break
            if not query.strip():
                continue
            run_query(agent, query)
        except EOFError:  # Handle Ctrl+D
            print("\nExiting.")
            break
        except KeyboardInterrupt:  # Handle Ctrl+C
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nError during interactive session: {str(e)}")
            logger.error("Error in interactive loop", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="MCP integration example")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--query", help="Query to run (non-interactive)")
    args = parser.parse_args()

    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    agent = None
    try:
        agent = setup_with_env_vars()

    except (ValueError, RuntimeError, ImportError) as setup_error:
        logger.error(f"Failed to set up agent: {setup_error}", exc_info=True)
        print(f"\nError during setup: {setup_error}")
        print(
            "Please check your configuration (environment variables, installed packages) and try again."
        )
        sys.exit(1)
    except Exception as e:  # Catch any other unexpected setup errors
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred during setup: {e}")
        sys.exit(1)

    # If agent setup was successful, proceed
    if agent and mcp_client_instances:
        # Print info fetched via MCPClient
        print_mcp_info(mcp_client_instances)

        # Run query or interactive session
        if args.query:
            run_query(agent, args.query)
        else:
            run_interactive_session(agent)
    else:
        logger.error("Agent or MCPClient initialization failed, cannot proceed.")
        print("\nAgent setup failed. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        if load_dotenv():
            logger.info("Loaded environment variables from .env file.")
        else:
            logger.debug(".env file not found or empty.")
    except ImportError:
        logger.warning("python-dotenv not found, .env file not loaded.")
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")

    main()
