import argparse
import atexit
import json
import logging
import os
import sys
from typing import Optional

from src.connectors.llm import LLMConnector
from src.core.agent import Agent
from src.mcp.mcp_client import MCPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("postgres_integration")

# Global client instance to ensure cleanup
mcp_client_instance: Optional[MCPClient] = None


def cleanup_mcp_client():
    """Function registered with atexit to stop the MCP server."""
    global mcp_client_instance
    if mcp_client_instance:
        logger.info("Running cleanup: Stopping MCP server...")
        try:
            mcp_client_instance.stop_server()
            logger.info("MCP server stopped successfully during cleanup.")
        except Exception as e:
            logger.error(f"Error stopping MCP server during cleanup: {e}", exc_info=True)


# Register the cleanup function
atexit.register(cleanup_mcp_client)


def create_mcp_client() -> MCPClient:
    """Creates and starts the MCPClient for the PostgreSQL server."""
    global mcp_client_instance
    if mcp_client_instance:
        logger.warning("MCPClient instance already exists.")
        return mcp_client_instance

    postgres_conn_string = os.environ.get("PG_CONNECTION_STRING")
    if not postgres_conn_string:
        logger.error("FATAL: PG_CONNECTION_STRING environment variable is not set.")
        raise ValueError("PG_CONNECTION_STRING must be set in the environment.")

    # Define the command to run the MCP server
    # Adjust path relative to this script if needed
    server_script_path = os.path.join(
        os.path.dirname(__file__), "src", "mcp", "postgres_mcp_server.py"
    )
    if not os.path.exists(server_script_path):
        logger.warning(f"MCP server script not found at default path: {server_script_path}")
        # Attempt fallback assuming it's in the same directory or PATH accessible
        server_script_path = "postgres_mcp_server.py"  # Adjust if necessary

    server_command = (
        f"{sys.executable} {server_script_path} {postgres_conn_string}"  # Do not quote connection string
    )

    try:
        logger.info("Initializing MCPClient...")
        # Create and store the client instance globally for cleanup
        mcp_client_instance = MCPClient(server_command=server_command)
        logger.info("MCPClient initialized and server process start initiated.")
        # Note: MCPClient constructor attempts to start the server. Error handling is inside MCPClient.
        return mcp_client_instance
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize MCPClient: {e}", exc_info=True)
        # Ensure cleanup is attempted even if constructor fails partially
        cleanup_mcp_client()
        raise  # Re-raise the exception


def setup_with_env_vars() -> Agent:
    """Set up agent using environment variables for LLM and MCPClient for tools."""
    logger.info("Setting up Agent using environment variables for LLM connector.")

    # 1. LLM Connector Setup from Environment Variables (Keep your multi-LLM logic if needed)
    # Simplified version:
    llm_url = os.environ.get("LLM_URL", "http://localhost:8080/v1")
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_connector = LLMConnector(
        base_url=llm_url, headers={"Authorization": f"Bearer {llm_api_key}"}
    )
    logger.info(f"LLMConnector created from environment variables (URL: {llm_url}).")

    # 2. Get MCP Client (manages the separate server process)
    mcp_client = create_mcp_client()

    # 3. Agent Initialization
    agent = Agent(llm_connector, mcp_client, debug=True)
    logger.info("Agent initialized with ENV LLMConnector and MCPClient.")
    return agent


def print_mcp_info(client: MCPClient) -> None:
    """Prints information discovered via the MCPClient."""
    print("\n--- MCP Server Information ('postgres') ---")
    try:
        print("Listing tools:")
        tools_resp = client.list_tools()
        print(json.dumps(tools_resp, indent=2))
    except Exception as e:
        print(f"  Error listing tools: {e}")
        logger.warning(f"Failed to list tools via MCPClient: {e}", exc_info=True)

    try:
        print("\nListing resources (schemas):")
        res_resp = client.list_resources()
        print(json.dumps(res_resp, indent=2))
    except Exception as e:
        print(f"  Error listing resources: {e}")
        logger.warning(f"Failed to list resources via MCPClient: {e}", exc_info=True)
    print("--- End MCP Info ---")


def run_query(agent: Agent, query: str) -> None:
    """Run a query using the agent (which uses MCPClient)."""
    print(f"\nRunning query: {query}")
    logger.info("======== STARTING NEW QUERY ========")
    try:
        logger.info("Executing main agent workflow via agent.run()")
        result_state = agent.run(query)  # Pass the raw query

        print("\n--- Agent Final Response ---")
        print(result_state.response or "Agent did not produce a response.")
        print("--- End Agent Response ---")

        if result_state.error:
            print(f"\nAgent encountered an error state: {result_state.error}")
            logger.error(f"Agent workflow finished with error state: {result_state.error}")

        # Optional: Print debug info even if main response exists
        if agent.debug or result_state.error:
            print("\n--- Agent Thoughts ---")
            for i, thought in enumerate(result_state.thoughts):
                print(f"{i + 1}. {thought}")
            print("--- End Thoughts ---")
            print("\n--- MCP Results Collected ---")
            # Use default=str to handle non-serializable types like datetime
            print(json.dumps(result_state.mcp_results, indent=2, default=str))
            print("--- End MCP Results ---")

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
    parser = argparse.ArgumentParser(description="PostgreSQL MCP integration example")
    parser.add_argument(
        "--mode",
        choices=["env", "cm"],
        default="env",
        help="Configuration mode: 'env' (use environment variables for LLM) or 'cm' (use ConfidentialMind for LLM).",
    )
    parser.add_argument("--query", help="Query to run (non-interactive)")
    args = parser.parse_args()

    agent = None
    try:
        agent = setup_with_env_vars()

    except (ValueError, RuntimeError, ImportError) as setup_error:
        logger.error(f"Failed to set up agent: {setup_error}", exc_info=True)
        print(f"\nError during setup: {setup_error}")
        print(
            "Please check your configuration (environment variables, CM setup, installed packages) and try again."
        )
        sys.exit(1)
    except Exception as e:  # Catch any other unexpected setup errors
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred during setup: {e}")
        sys.exit(1)

    # If agent setup was successful, proceed
    if agent and mcp_client_instance:
        # Print info fetched via MCPClient
        print_mcp_info(mcp_client_instance)

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
