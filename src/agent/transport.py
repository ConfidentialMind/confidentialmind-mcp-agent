import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Set

from confidentialmind_core.config_manager import ArrayConnectorSchema, ConfigManager
from confidentialmind_core.config_manager import config as cm_config
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport, SSETransport

from .module_transport import ModuleStdioTransport, path_to_module_path

logger = logging.getLogger(__name__)


class TransportManager:
    """Manages transport configurations for FastMCP clients."""

    def __init__(self, mode: Literal["cli", "api"] = "cli"):
        """Initialize transport manager with specified mode.

        Args:
            mode: Operating mode - "cli" for stdio transport or "api" for SSE
        """
        self.mode = mode
        self.transports = {}
        self.clients: Dict[str, Client] = {}
        self._mcp_retry_task = None
        self._connected_mcp_ids: Set[str] = set()

    def configure_transport(
        self,
        server_id: str,
        server_path: Optional[str] = None,
        server_url: Optional[str] = None,
        use_module: bool = True,
    ) -> None:
        """Configure transport for a specific server.

        Args:
            server_id: Unique identifier for this server
            server_path: Path to Python script (for CLI mode)
            server_url: URL for SSE endpoint (for API mode)
            use_module: Whether to use module execution (-m flag) for CLI mode
        """
        if server_id in self.transports:
            logger.warning(f"Transport for {server_id} already configured. Reconfiguring.")

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
                    logger.info(f"Configured module transport for {server_id}: {module_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create module transport for {server_id}: {e}. "
                        f"Falling back to script transport."
                    )
                    self.transports[server_id] = PythonStdioTransport(script_path=server_path)
                    logger.info(f"Configured stdio transport for {server_id}: {server_path}")
            else:
                # Use regular PythonStdioTransport
                self.transports[server_id] = PythonStdioTransport(script_path=server_path)
                logger.info(f"Configured stdio transport for {server_id}: {server_path}")

        elif self.mode == "api":
            if not server_url:
                raise ValueError(f"server_url required for API mode transport: {server_id}")

            self.transports[server_id] = SSETransport(url=server_url)
            logger.info(f"Configured SSE transport for {server_id}: {server_url}")

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def configure_from_dict(self, config: Dict[str, str], use_module: bool = True) -> None:
        """Configure multiple transports from a dictionary.

        Args:
            config: Dictionary mapping server_id to path/url
            use_module: Whether to use module execution (-m flag) for CLI mode
        """
        for server_id, server_ref in config.items():
            if self.mode == "cli":
                self.configure_transport(server_id, server_path=server_ref, use_module=use_module)
            else:
                self.configure_transport(server_id, server_url=server_ref)

    def create_client(self, server_id: str) -> Optional[Client]:
        """Create a FastMCP client for a configured transport.

        Args:
            server_id: The server identifier to create a client for

        Returns:
            FastMCP Client instance or None if transport not configured
        """
        if server_id not in self.transports:
            logger.error(f"No transport configured for server_id: {server_id}")
            return None

        transport = self.transports[server_id]
        client = Client(transport)
        self.clients[server_id] = client
        return client

    def create_all_clients(self) -> Dict[str, Client]:
        """Create clients for all configured transports.

        Returns:
            Dictionary mapping server_id to Client instances
        """
        for server_id in self.transports:
            if server_id not in self.clients:
                self.create_client(server_id)

        return self.clients

    def get_client(self, server_id: str) -> Optional[Client]:
        """Get existing client by server_id or create if not exists.

        Args:
            server_id: The server identifier

        Returns:
            FastMCP Client instance or None if transport not configured
        """
        if server_id in self.clients:
            return self.clients[server_id]

        return self.create_client(server_id)

    def get_all_clients(self) -> Dict[str, Client]:
        """Get all configured clients.

        Returns:
            Dictionary mapping server_id to Client instances
        """
        return self.clients

    async def init_from_config_manager(self) -> None:
        """Initialize transports from ConfigManager for stack deployment.

        This method checks for MCP servers configured in the ConfigManager
        and sets up SSE transports for them.
        """
        if cm_config.LOCAL_CONFIGS:
            logger.info("Running in local mode, skipping ConfigManager MCP lookup")
            return

        if self.mode != "api":
            logger.warning("ConfigManager MCP lookup only supported in API mode")
            return

        # Start background polling task
        if self._mcp_retry_task is None or self._mcp_retry_task.done():
            self._mcp_retry_task = asyncio.create_task(self._poll_for_mcp_servers())

    async def _poll_for_mcp_servers(self) -> None:
        """Continuously poll for MCP servers in the ConfigManager."""
        poll_interval = 30  # seconds

        while True:
            try:
                config_manager = ConfigManager()

                # Check if ConfigManager is properly initialized
                if (
                    not hasattr(config_manager, "_ConfigManager__initialized")
                    or not config_manager._ConfigManager__initialized
                ):
                    logger.warning("ConfigManager not initialized yet for MCP polling")
                    await asyncio.sleep(poll_interval)
                    continue

                # Look for MCP array connector
                array_connectors = config_manager.array_connectors
                if array_connectors:
                    for connector in array_connectors:
                        if connector.config_id == "MCP_SERVERS":
                            self._process_mcp_connector(connector)
                            break

                # Sleep before next poll
                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error polling for MCP servers: {e}")
                await asyncio.sleep(poll_interval)

    def _process_mcp_connector(self, connector: ArrayConnectorSchema) -> None:
        """Process a discovered MCP connector and set up transports."""
        if not connector.stack_ids:
            logger.info("MCP_SERVERS connector exists but has no stack_ids configured")
            return

        config_manager = ConfigManager()

        # Get URLs for each stack ID
        for i, stack_id in enumerate(connector.stack_ids):
            # Skip already connected MCPs
            if stack_id in self._connected_mcp_ids:
                continue

            try:
                # Get the namespace from the connector type
                namespace = config_manager.getNamespaceForConnector("MCP_SERVERS")
                if not namespace:
                    namespace = "api-services"  # Default namespace

                # Construct the URL
                server_url = f"http://{stack_id}.{namespace}.svc.cluster.local:8080/sse"
                server_id = f"mcp-{stack_id}"

                # Configure the transport
                self.configure_transport(server_id, server_url=server_url)

                # Create the client
                self.create_client(server_id)

                # Remember that we've connected to this MCP
                self._connected_mcp_ids.add(stack_id)

                logger.info(f"Added MCP server from ConfigManager: {server_id} -> {server_url}")

            except Exception as e:
                logger.error(f"Error configuring MCP server {stack_id}: {e}")
