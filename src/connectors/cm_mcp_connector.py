# src/connectors/cm_mcp_connector.py
import asyncio
import logging
from contextlib import AbstractContextManager
from typing import Dict, Optional

# Use ConfidentialMind SDK for configuration, DO NOT CHANGE CORE LOGIC
from confidentialmind_core import config
from confidentialmind_core.config_manager import ConfigManager, get_api_parameters

# Import official MCP SDK components
from mcp import ClientSession, McpError
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters

logger = logging.getLogger(__name__)


class CMMCPManager:
    """
    Manager for MCP ClientSessions using the confidentialmind SDK for connection details.
    Handles initializing connections via stdio (local dev) or SSE (remote).
    Relies on ConfigManager being initialized elsewhere (e.g., FastAPI lifespan).
    """

    def __init__(self):
        """Initialize the MCP session manager."""
        # Store ClientSession instances keyed by server_id (config_id)
        self.sessions: Dict[str, ClientSession] = {}
        # Store tasks responsible for initializing sessions
        self._initialization_tasks: Dict[str, asyncio.Task] = {}
        self.config_manager = ConfigManager()
        self._lock = asyncio.Lock()  # Lock for managing session creation/initialization
        # Store transport contexts that need explicit cleanup
        self.sessions_transport_ctx: Dict[str, AbstractContextManager] = {}

    async def get_session(self, server_id: str) -> ClientSession:
        """
        Get an initialized MCP ClientSession for a specific server.

        Retrieves connection parameters using ConfigManager, creates a session
        with the appropriate transport (stdio/sse), initializes it, and caches it.

        Args:
            server_id: Unique identifier (config_id) for the server.

        Returns:
            An initialized ClientSession instance.

        Raises:
            ValueError: If the server is not configured or connection fails.
            McpError: If session initialization fails.
        """
        async with self._lock:
            # Return existing, initialized session
            if server_id in self.sessions:
                session = self.sessions[server_id]
                # Ensure it's still connected (basic check)
                # TODO: Implement a more robust health check if needed via ping?
                return session
                # else: logger.warning(f"Session for '{server_id}' was closed. Re-initializing.")
                #     # Remove closed session and task
                #     self.sessions.pop(server_id, None)
                #     self._initialization_tasks.pop(server_id, None)

            # If initialization is already in progress, wait for it
            if server_id in self._initialization_tasks:
                logger.debug(f"Waiting for existing initialization task for '{server_id}'")
                return await self._initialization_tasks[server_id]

            # Start new initialization
            logger.info(f"Creating and initializing new session for server_id: '{server_id}'")
            # Create a future to represent the result of the initialization
            init_future = asyncio.Future[ClientSession]()
            # Store the task associated with this future
            self._initialization_tasks[server_id] = asyncio.create_task(init_future)

        # Perform connection and initialization outside the lock
        try:
            session = await self._create_and_initialize_session(server_id)
            async with self._lock:
                self.sessions[server_id] = session
                # Set the result for waiting tasks
                init_future.set_result(session)
                # Remove the completed task entry
                self._initialization_tasks.pop(server_id, None)
            logger.info(f"Successfully initialized session for '{server_id}'")
            return session
        except (ValueError, McpError, Exception) as e:
            logger.error(f"Failed to initialize session for '{server_id}': {e}", exc_info=True)
            async with self._lock:
                # Set exception for waiting tasks
                if not init_future.done():
                    init_future.set_exception(e)
                # Remove the failed task entry
                self._initialization_tasks.pop(server_id, None)
            # Re-raise the specific error type
            if isinstance(e, (ValueError, McpError)):
                raise e
            else:
                raise ValueError(f"Failed to initialize session for {server_id}: {e}") from e

    async def _create_and_initialize_session(self, server_id: str) -> ClientSession:
        """Handles the actual creation and initialization logic."""
        transport_context: Optional[AbstractContextManager] = None
        session: Optional[ClientSession] = None

        # 1. Get connection parameters from ConfigManager
        try:
            url_or_command_details, headers = get_api_parameters(server_id)
            # get_api_parameters returns tuple (url_or_None, headers_or_None)
            # We need to handle the case where URL might be None if not configured
            if not url_or_command_details:
                logger.error(f"MCP server {server_id} has no URL/connection details configured.")
                raise ValueError(f"URL/connection details for MCP server '{server_id}' not found.")
        except ValueError as ve:
            # Catch specific errors from get_api_parameters (like invalid API key format)
            logger.error(f"Configuration error for MCP server '{server_id}': {ve}")
            raise
        except Exception as e:
            logger.error(f"Failed to get connection parameters for '{server_id}': {e}")
            raise ValueError(f"Could not retrieve connection details for {server_id}") from e

        # 2. Determine transport based on LOCAL_DEV
        if config.LOCAL_DEV:
            # --- Stdio Transport (Local Development) ---
            logger.info(f"Using stdio transport for local dev server '{server_id}'")
            # Assumption: Command structure is 'uv run <config_id>'
            # We use server_id as config_id here.
            # TODO: Make command structure configurable if needed.
            command = "uv"
            args = ["run", server_id]

            # Pass environment variables from headers
            # Convert headers (potentially None) to a dictionary for env
            env_vars = {}
            if isinstance(headers, dict):
                for key, value in headers.items():
                    # Normalize header keys to typical env var format if needed
                    env_key = key.upper().replace("-", "_")
                    env_vars[env_key] = value

            # Add other necessary env vars if required by the server script
            # env_vars["PYTHONUNBUFFERED"] = "1" # Example

            logger.debug(f"Stdio command: {command} {' '.join(args)}")
            logger.debug(f"Stdio env vars: {env_vars}")

            stdio_params = StdioServerParameters(command=command, args=args, env=env_vars)
            # stdio_client is an async context manager
            try:
                read_stream, write_stream = await transport_context.__aenter__()  # type: ignore

                # Need to ensure transport_context.__aexit__() is called later!
                # Storing it seems necessary. We'll store it with the session.
                # This part is complex due to context management lifecycle.

                session = ClientSession(read_stream, write_stream)
                # Store the context manager for cleanup
                self.sessions_transport_ctx[server_id] = transport_context

            except Exception as e:
                logger.error(
                    f"Failed to start stdio server process for '{server_id}': {e}", exc_info=True
                )
                raise ValueError(f"Could not start stdio server {server_id}") from e

        else:
            # --- SSE Transport (Remote) ---
            logger.info(f"Using SSE transport for remote server '{server_id}'")
            base_url = url_or_command_details  # Should be a URL string here
            if not isinstance(base_url, str) or not base_url.startswith("http"):
                raise ValueError(
                    f"Invalid URL received for remote SSE server '{server_id}': {base_url}"
                )

            # sse_client is an async context manager
            try:
                # Same context management issue as stdio
                transport_context = sse_client(
                    f"{base_url}/sse", headers=headers or {}
                )  # Assuming /sse endpoint
                read_stream, write_stream = await transport_context.__aenter__()  # type: ignore

                session = ClientSession(read_stream, write_stream)
                # Store the context manager for cleanup
                self.sessions_transport_ctx[server_id] = transport_context  # Need to add this dict

            except Exception as e:
                logger.error(
                    f"Failed to connect SSE transport for '{server_id}': {e}", exc_info=True
                )
                raise ValueError(f"Could not connect SSE transport for {server_id}") from e

        # 3. Initialize the session
        try:
            await session.initialize()
        except McpError as e:
            logger.error(f"MCP initialization failed for '{server_id}': {e.error.message}")
            # Ensure transport context is cleaned up on failure
            if server_id in self.sessions_transport_ctx:
                await self._cleanup_transport_context(server_id)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during session initialization for '{server_id}': {e}")
            # Ensure transport context is cleaned up on failure if it was created
            if server_id in self.sessions_transport_ctx:
                await self._cleanup_transport_context(server_id)
            raise ValueError(f"Session initialization failed for {server_id}") from e

        return session

    async def get_all_sessions(self) -> Dict[str, ClientSession]:
        """
        Get all registered and potentially configurable MCP ClientSessions.

        Returns:
            Dictionary mapping server config_ids to initialized ClientSession instances.
        """
        # MCP server types we should initialize - adjust based on actual connector types used
        mcp_connector_types = {
            "agent_tool",
            "endpoint",
            "api",  # Assuming 'api' might be used for generic MCP servers
            # Add specific types like 'postgres_mcp', 'rag_mcp' if defined in ConnectorSchema
        }

        connector_ids_to_initialize = set()

        # Process regular connectors
        if self.config_manager.connectors:
            for connector in self.config_manager.connectors:
                if connector.type in mcp_connector_types:
                    connector_ids_to_initialize.add(connector.config_id)

        # Process array connectors
        if self.config_manager.array_connectors:
            for array_connector in self.config_manager.array_connectors:
                if array_connector.type in mcp_connector_types:
                    connector_ids_to_initialize.add(array_connector.config_id)

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
                    f"Failed to initialize session for '{server_id}' during get_all_sessions: {e}"
                )
                # Optionally store the error or skip the server

        logger.info(f"Successfully initialized {len(results)} MCP sessions.")
        return results

    async def _cleanup_transport_context(self, server_id: str):
        """Safely cleans up the transport context for a given server_id."""
        transport_ctx = self.sessions_transport_ctx.pop(server_id, None)
        if transport_ctx:
            try:
                logger.debug(f"Cleaning up transport context for '{server_id}'")
                await transport_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.error(
                    f"Error cleaning up transport context for '{server_id}': {e}", exc_info=True
                )

    async def close_session(self, server_id: str):
        """Closes a specific session and cleans up its transport context."""
        async with self._lock:
            session = self.sessions.pop(server_id, None)
            self._initialization_tasks.pop(server_id, None)  # Cancel ongoing init task if any

        if session:
            logger.info(f"Closing session for '{server_id}'")
            # ClientSession should handle closing its streams when its context manager exits
            # We rely on the cleanup of the transport context manager below

        # Cleanup transport context manager (stdio process or SSE connection)
        await self._cleanup_transport_context(server_id)

    async def close_all_sessions(self):
        """Closes all managed sessions and cleans up resources."""
        logger.info("Closing all MCP sessions...")
        # Create list of tasks to close sessions concurrently
        server_ids = list(self.sessions.keys())  # Get keys before iterating
        tasks = [self.close_session(server_id) for server_id in server_ids]
        if tasks:
            await asyncio.gather(
                *tasks, return_exceptions=True
            )  # Use return_exceptions to log errors

        # Ensure all transport contexts are cleaned up, even if session creation failed
        ctx_ids = list(self.sessions_transport_ctx.keys())
        cleanup_tasks = [self._cleanup_transport_context(server_id) for server_id in ctx_ids]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("Finished closing all MCP sessions.")
