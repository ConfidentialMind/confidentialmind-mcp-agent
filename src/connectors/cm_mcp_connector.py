import logging
import os
from typing import Dict, List, Set

from confidentialmind_core.config_manager import ConfigManager, get_api_parameters

from src.mcp.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class CMMCPManager:
    """
    Manager for MCP clients that uses the confidentialmind SDK to handle connections.
    Supports both HTTP-based connections to standalone servers and subprocess-based
    local development mode.
    """

    def __init__(self):
        """Initialize the MCP client manager"""
        self.clients: Dict[str, MCPClient] = {}
        self.registered_servers: Set[str] = set()
        self.config_manager = ConfigManager()

    def register_mcp_server(
        self, server_id: str, server_type: str, label: str, description: str = ""
    ) -> None:
        """
        Register an MCP server with the SDK

        Args:
            server_id: Unique identifier for the server
            server_type: Type of server (postgres, rag, obsidian, etc.)
            label: Human-readable label
            description: Detailed description
        """
        from confidentialmind_core.config_manager import ConnectorSchema

        # Register with the SDK
        connector_schema = ConnectorSchema(
            type=server_type,
            label=label,
            config_id=server_id,
            description=description or f"Connection for {server_id} services",
        )

        # Initialize or update connectors if needed
        if hasattr(self.config_manager, "connectors") and self.config_manager.connectors:
            self.config_manager.update_connectors([connector_schema])
        else:
            self.config_manager.init_manager(connectors=[connector_schema])

        # Track registered servers
        self.registered_servers.add(server_id)
        logger.info(f"Registered MCP server: {server_id} ({server_type})")

    def get_client(self, server_id: str) -> MCPClient:
        """
        Get an MCP client for a specific server

        Args:
            server_id: Unique identifier for the server

        Returns:
            MCPClient instance for the requested server

        Raises:
            ValueError: If the server is not registered or configured
        """
        # Return existing client if available
        if server_id in self.clients:
            return self.clients[server_id]

        # Get connection details from SDK
        base_url, headers = get_api_parameters(server_id)

        # Check if we got a valid URL
        if not base_url:
            raise ValueError(f"MCP server {server_id} not configured in SDK")

        # Create client based on URL
        client = MCPClient(base_url=base_url)
        self.clients[server_id] = client
        logger.info(f"Created MCP client for {server_id} using {base_url}")

        return client

    def get_all_clients(self) -> Dict[str, MCPClient]:
        """
        Get all registered and configured MCP clients

        Returns:
            Dictionary mapping server IDs to MCPClient instances
        """
        clients = {}

        # Try to initialize clients for all registered servers
        for server_id in self.registered_servers:
            try:
                clients[server_id] = self.get_client(server_id)
            except Exception as e:
                logger.warning(f"Failed to initialize MCP client for {server_id}: {e}")

        return clients

    def register_from_environment(self) -> List[str]:
        """
        Register MCP servers based on environment variables

        Returns:
            List of registered server IDs
        """
        registered = []

        # Check for PostgreSQL MCP server
        if os.environ.get("PG_MCP_URL"):
            self.register_mcp_server(
                "postgres", "postgres", "PostgreSQL MCP Server", "SQL query capabilities"
            )
            registered.append("postgres")

        # Check for RAG MCP server
        if os.environ.get("RAG_MCP_URL"):
            self.register_mcp_server(
                "rag", "rag", "RAG MCP Server", "Retrieval-augmented generation capabilities"
            )
            registered.append("rag")

        # Check for Obsidian MCP server
        if os.environ.get("OBSIDIAN_MCP_URL"):
            self.register_mcp_server(
                "obsidian",
                "obsidian",
                "Obsidian MCP Server",
                "Obsidian vault note access capabilities",
            )
            registered.append("obsidian")

        logger.info(f"Registered {len(registered)} MCP servers from environment variables")
        return registered
