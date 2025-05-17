import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional

from confidentialmind_core.config_manager import load_environment
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport, SSETransport

from src.agent.connectors import ConnectorConfigManager
from src.agent.module_transport import ModuleStdioTransport, path_to_module_path

logger = logging.getLogger(__name__)


class TransportManager:
    """Manages transport configurations for FastMCP clients."""

    def __init__(self, mode: Literal["cli", "api"] = "cli"):
        """
        Initialize transport manager with specified mode.

        Args:
            mode: Operating mode - "cli" for stdio transport or "api" for SSE
        """
        self.mode = mode
        self.transports = {}
        self.clients: Dict[str, Client] = {}
        self._background_fetch_task = None

        # Load environment variables
        load_environment()
        self._is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"TransportManager: Initialized in {'stack deployment' if self._is_stack_deployment else 'local config'} mode with {mode} mode"
        )

    async def _start_background_polling(self):
        """Start background polling for MCP server URLs if in stack deployment mode."""
        if self._background_fetch_task is not None and not self._background_fetch_task.done():
            logger.info("TransportManager: Background polling already started")
            return  # Already polling

        if self._is_stack_deployment:
            logger.info("TransportManager: Starting background polling for MCP server URLs")
            self._background_fetch_task = asyncio.create_task(
                self._poll_for_mcp_servers_in_background()
            )

    async def _poll_for_mcp_servers_in_background(self):
        """Continuously poll for MCP server URLs and configure transports when available."""
        connector_manager = ConnectorConfigManager()
        await connector_manager.initialize(register_connectors=False)

        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries
        known_servers = set()

        while True:
            try:
                servers = await connector_manager.fetch_mcp_servers()

                if servers:
                    # Check for new servers
                    for server_id, server_url in servers.items():
                        if server_id not in known_servers:
                            logger.info(f"TransportManager: Found new MCP server: {server_id}")
                            known_servers.add(server_id)
                            try:
                                self.configure_transport(server_id, server_url=server_url)
                                self.create_client(server_id)
                                logger.info(
                                    f"TransportManager: Successfully configured transport for {server_id}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"TransportManager: Error configuring transport for {server_id}: {e}"
                                )
            except Exception as e:
                # Log less frequently as retries increase
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.error(f"TransportManager: Error fetching MCP servers: {e}")

            retry_count += 1
            # Exponential backoff with a maximum wait time
            wait_time = min(30, 5 * (1.5 ** min(retry_count, 5)))
            await asyncio.sleep(wait_time)

    def configure_transport(
        self,
        server_id: str,
        server_path: Optional[str] = None,
        server_url: Optional[str] = None,
        use_module: bool = True,
    ) -> None:
        """
        Configure transport for a specific server.

        Args:
            server_id: Unique identifier for this server
            server_path: Path to Python script (for CLI mode)
            server_url: URL for SSE endpoint (for API mode)
            use_module: Whether to use module execution (-m flag) for CLI mode
        """
        if server_id in self.transports:
            logger.warning(
                f"TransportManager: Transport for {server_id} already configured. Reconfiguring."
            )

        if self.mode == "cli":
            if not server_path:
                raise ValueError(f"server_path required for CLI mode transport: {server_id}")

            # Validate that the path exists
            script_path = Path(server_path)
            if not script_path.exists():
                raise ValueError(f"Script not found: {os.path.abspath(server_path)}")

            if use_module:
                # Use ModuleStdioTransport for proper package imports
                try:
                    module_path = path_to_module_path(server_path)
                    self.transports[server_id] = ModuleStdioTransport(module_path=module_path)
                    logger.info(
                        f"TransportManager: Configured module transport for {server_id}: {module_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"TransportManager: Failed to create module transport for {server_id}: {e}. "
                        f"Falling back to script transport."
                    )
                    self.transports[server_id] = PythonStdioTransport(script_path=server_path)
                    logger.info(
                        f"TransportManager: Configured stdio transport for {server_id}: {server_path}"
                    )
            else:
                # Use regular PythonStdioTransport
                self.transports[server_id] = PythonStdioTransport(script_path=server_path)
                logger.info(
                    f"TransportManager: Configured stdio transport for {server_id}: {server_path}"
                )

        elif self.mode == "api":
            if not server_url:
                raise ValueError(f"server_url required for API mode transport: {server_id}")

            # Ensure the URL has the /sse path for SSE transport
            if not server_url.endswith("/sse"):
                # Append /sse to the URL if it doesn't already end with it
                server_url = server_url.rstrip("/") + "/sse"
                logger.info(f"TransportManager: Appended /sse to the URL: {server_url}")

            self.transports[server_id] = SSETransport(url=server_url)
            logger.info(f"TransportManager: Configured SSE transport for {server_id}: {server_url}")

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def configure_from_dict(self, config: Dict[str, str], use_module: bool = True) -> None:
        """
        Configure multiple transports from a dictionary.

        Args:
            config: Dictionary mapping server_id to path/url
            use_module: Whether to use module execution (-m flag) for CLI mode
        """
        for server_id, server_ref in config.items():
            if self.mode == "cli":
                self.configure_transport(server_id, server_path=server_ref, use_module=use_module)
            else:
                self.configure_transport(server_id, server_url=server_ref)

    async def configure_from_stack(self) -> None:
        """Configure transports using MCP servers from the stack."""
        connector_manager = ConnectorConfigManager()
        await connector_manager.initialize(register_connectors=True)

        # Start background polling for MCP server changes
        if self._is_stack_deployment:
            await self._start_background_polling()

        servers = await connector_manager.fetch_mcp_servers()
        if not servers:
            logger.warning("TransportManager: No MCP servers found from stack configuration")
            return

        # Configure transports based on mode
        for server_id, server_url in servers.items():
            try:
                logger.info(
                    f"TransportManager: Configuring transport for {server_id} with url {server_url} from stack"
                )
                self.configure_transport(server_id, server_url=server_url)
            except Exception as e:
                logger.error(f"TransportManager: Error configuring transport for {server_id}: {e}")

    def create_client(self, server_id: str) -> Optional[Client]:
        """
        Create a FastMCP client for a configured transport.

        Args:
            server_id: The server identifier to create a client for

        Returns:
            FastMCP Client instance or None if transport not configured
        """
        if server_id not in self.transports:
            logger.error(f"TransportManager: No transport configured for server_id: {server_id}")
            return None

        transport = self.transports[server_id]
        client = Client(transport)
        self.clients[server_id] = client
        return client

    def create_all_clients(self) -> Dict[str, Client]:
        """
        Create clients for all configured transports.

        Returns:
            Dictionary mapping server_id to Client instances
        """
        for server_id in self.transports:
            if server_id not in self.clients:
                self.create_client(server_id)

        return self.clients

    def get_client(self, server_id: str) -> Optional[Client]:
        """
        Get existing client by server_id or create if not exists.

        Args:
            server_id: The server identifier

        Returns:
            FastMCP Client instance or None if transport not configured
        """
        if server_id in self.clients:
            return self.clients[server_id]

        return self.create_client(server_id)

    def get_all_clients(self) -> Dict[str, Client]:
        """
        Get all configured clients.

        Returns:
            Dictionary mapping server_id to Client instances
        """
        return self.clients
