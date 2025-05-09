import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional

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
