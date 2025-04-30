# src/connectors/cm_mcp_connector.py
import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Dict

from confidentialmind_core import config
from confidentialmind_core.config_manager import ConfigManager, get_api_parameters

from mcp import ClientSession, McpError
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import INTERNAL_ERROR, INVALID_REQUEST, ErrorData

logger = logging.getLogger(__name__)


class CMMCPManager:
    """
    Manager for MCP ClientSessions using the confidentialmind SDK for connection details.
    """

    def __init__(self):
        """Initialize the MCP session manager."""
        self.sessions: Dict[str, ClientSession] = {}
        self._init_tasks: Dict[str, asyncio.Task] = {}
        self.config_manager = ConfigManager()
        self._lock = asyncio.Lock()
        # Use AsyncExitStack for proper context management
        self._exit_stack = AsyncExitStack()

    async def __aenter__(self):
        """Support async context manager pattern."""
        await self._exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of all resources."""
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def get_session(self, server_id: str) -> ClientSession:
        """
        Get an initialized MCP ClientSession for a specific server.

        Args:
            server_id: Unique identifier (config_id) for the server.

        Returns:
            An initialized ClientSession instance.

        Raises:
            McpError: If session creation or initialization fails.
        """
        async with self._lock:
            # Return existing session if available
            if server_id in self.sessions:
                return self.sessions[server_id]

            # Wait for existing initialization task if in progress
            if server_id in self._init_tasks and not self._init_tasks[server_id].done():
                try:
                    return await self._init_tasks[server_id]
                except Exception as e:
                    self._init_tasks.pop(server_id, None)
                    raise e

            init_task = asyncio.create_task(self._create_and_initialize_session(server_id))

            async with self._lock:
                self._init_tasks[server_id] = init_task

            try:
                session = await init_task
                async with self._lock:
                    self.sessions[server_id] = session
                    return session
            except Exception as e:
                error = (
                    e
                    if isinstance(e, McpError)
                    else McpError(
                        ErrorData(
                            code=INTERNAL_ERROR, message=f"Session initiialization error :{str(e)}"
                        )
                    )
                )
                async with self._lock:
                    self._init_tasks.pop(server_id, None)
                raise error

    async def get_all_sessions(self) -> Dict[str, ClientSession]:
        """
        Get all registered and potentially configurable MCP ClientSessions.

        Returns:
            Dictionary mapping server config_ids to initialized ClientSession instances.
        """
        # MCP server types we should initialize
        mcp_connector_types = {
            "agent_tool",  # for MCP servers
        }

        connector_ids_to_initialize = set()

        # TODO: DEBUG - delete
        logger.info(f"Connectors before loop: {self.config_manager.connectors}")
        logger.info(f"Array Connectors before loop: {self.config_manager.array_connectors}")

        # Process regular connectors
        if self.config_manager.connectors:
            for connector in self.config_manager.connectors:
                # TODO: DEBUG - delete
                logger.info(f"Checking regular connector type: {connector.type}")

                if connector.type in mcp_connector_types:
                    connector_ids_to_initialize.add(connector.config_id)

        # Process array connectors
        if self.config_manager.array_connectors:
            for array_connector in self.config_manager.array_connectors:
                # TODO: DEBUG - delete
                logger.info(f"Checking array connector type: {array_connector.type}")

                if array_connector.type in mcp_connector_types:
                    # Add the main connector ID
                    connector_ids_to_initialize.add(array_connector.config_id)

                    if array_connector.stack_ids:
                        # Add any stack IDs for multi-instance connectors
                        connector_ids_to_initialize.update(array_connector.stack_ids)

        # TODO: DEBUG - delete
        logger.info(f"Final connector IDs to initialize: {connector_ids_to_initialize}")

        if not connector_ids_to_initialize:
            logger.warning("ConfigManager has no MCP-related connectors registered.")
            return {}

        # Initialize sessions concurrently
        tasks = {
            server_id: asyncio.create_task(self.get_session(server_id))
            for server_id in connector_ids_to_initialize
        }

        results = {}
        for server_id, task in tasks.items():
            try:
                session = await task
                results[server_id] = session
            except Exception as e:
                logger.warning(
                    f"Failed to initialize session for '{server_id}': {e}", exc_info=True
                )
                # Skip this server but continue with others

        logger.info(f"Successfully initialized {len(results)} MCP sessions.")
        return results

    async def _create_and_initialize_session(self, server_id: str) -> ClientSession:
        """Create and initialize a session with proper context management."""
        server_to_module = {
            "agentTools": "src.mcp.postgres_mcp_server",
        }

        try:
            # Get connection parameters from SDK
            url_or_cmd, headers = get_api_parameters(server_id)
            if not url_or_cmd:
                raise McpError(
                    ErrorData(
                        code=INVALID_REQUEST,
                        message=f"Connection details for '{server_id}' not found",
                    )
                )

            # Determine transport based on deployment type
            if config.LOCAL_DEV:
                if server_id in server_to_module and server_to_module[server_id]:
                    # Use stdio transport for local development
                    command = "python"
                    args = ["-m", server_to_module[server_id]]
                    env_vars = {k.upper().replace("-", "_"): v for k, v in (headers or {}).items()}

                    env_vars["LOCAL_DEV"] = "True"

                    logger.info(f"Connecting to server {server_id} using stdio transport")
                    streams = await self._exit_stack.enter_async_context(
                        stdio_client(
                            StdioServerParameters(command=command, args=args, env=env_vars)
                        )
                    )
                else:
                    # For unknown servers, assume they're already running and use SSE
                    if not url_or_cmd.startswith("http"):
                        url_or_cmd = f"http://{url_or_cmd}"

                    logger.info(f"Connecting to server {server_id} at {url_or_cmd}/sse")
                    streams = await self._exit_stack.enter_async_context(
                        sse_client(f"{url_or_cmd}/sse", headers=headers or {})
                    )

            else:
                # Remote deployment with SSE transport
                if not isinstance(url_or_cmd, str) or not url_or_cmd.startswith("http"):
                    raise McpError(
                        ErrorData(
                            code=INVALID_REQUEST,
                            message=f"Invalid URL for '{server_id}': {url_or_cmd}",
                        )
                    )

                # Use the exit stack to manage the transport context
                streams = await self._exit_stack.enter_async_context(
                    sse_client(f"{url_or_cmd}/sse", headers=headers or {})
                )

            # Create and initialize the session
            read_stream, write_stream = streams
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            return session

        except McpError:
            # Pass through SDK errors
            raise
        except Exception as e:
            # Convert other exceptions to McpError
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"Failed to create session: {str(e)}")
            )

    async def close_all_sessions(self):
        """Closes all managed sessions and cleans up resources."""
        logger.info("Closing all MCP sessions...")

        async with self._lock:
            # Get a list of all session IDs before we start modifying the dictionary
            server_ids = list(self.sessions.keys())

        # Create tasks to close sessions concurrently
        close_tasks = []
        for server_id in server_ids:
            close_tasks.append(self._close_session(server_id))

        # Wait for all sessions to close
        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            # Log any errors, but don't raise them
            for server_id, result in zip(server_ids, results):
                if isinstance(result, Exception):
                    logger.error(f"Error closing session for '{server_id}': {result}")

        logger.info(f"Finished closing {len(server_ids)} MCP sessions.")

    async def _close_session(self, server_id: str):
        """Close a specific session and clean up its resources."""
        try:
            async with self._lock:
                session = self.sessions.pop(server_id, None)
                self._init_tasks.pop(server_id, None)  # Remove any initialization task

            if session:
                logger.debug(f"Closing session for '{server_id}'")
                # Let the AsyncExitStack handle the actual cleanup
                # Sessions added with enter_async_context will be closed automatically
                # when the exit stack exits

            return True
        except Exception as e:
            logger.error(f"Error during session closure for '{server_id}': {e}")
            return e
