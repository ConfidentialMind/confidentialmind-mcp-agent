import logging
from typing import Dict

from confidentialmind_core.config_manager import ConfigManager, get_api_parameters

from src.mcp.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class CMMCPManager:
    """
    Manager for MCP clients that uses the confidentialmind SDK to handle connections.
    Relies on ConfigManager being initialized elsewhere (e.g., FastAPI lifespan).
    """

    def __init__(self):
        """Initialize the MCP client manager"""
        self.clients: Dict[str, MCPClient] = {}
        self.config_manager = ConfigManager()

    def get_client(self, server_id: str) -> MCPClient:
        """
        Get an MCP client for a specific server using ConfigManager for connection details.

        Args:
            server_id: Unique identifier (config_id) for the server as registered in ConfigManager.

        Returns:
            MCPClient instance for the requested server.

        Raises:
            ValueError: If the server is not configured in the SDK or connection fails.
        """
        # Return existing client if available
        if server_id in self.clients:
            return self.clients[server_id]

        # Check if the connector exists in the config manager
        connector_exists = False
        if self.config_manager.connectors:
            connector_exists = any(c.config_id == server_id for c in self.config_manager.connectors)

        if not connector_exists:
            logger.error(
                f"MCP server config_id '{server_id}' not found in initialized ConfigManager connectors."
            )
            raise ValueError(f"MCP server {server_id} not found in configuration.")

        try:
            # Get connection details from SDK ConfigManager
            # get_api_parameters handles retrieval from env vars or portal based on SDK logic
            base_url, headers = get_api_parameters(server_id)

            # Check if we got a valid URL
            if not base_url:
                # This case should ideally be caught by the connector_exists check above,
                # but double-check for safety.
                logger.error(
                    f"MCP server {server_id} is registered but no URL could be retrieved via get_api_parameters."
                )
                raise ValueError(f"URL for MCP server {server_id} not configured in SDK.")

            # Create client based on URL and headers
            client = MCPClient(base_url=base_url, headers=headers)
            self.clients[server_id] = client
            logger.info(f"Created MCP client for '{server_id}' using URL: {base_url}")

            # Log if authentication is configured
            if headers and any(key.lower() in ("authorization", "x-api-key") for key in headers):
                logger.info(f"Authentication configured for MCP client '{server_id}'")

            return client

        except ValueError as ve:  # Catch specific configuration errors
            logger.error(f"Configuration error for MCP server '{server_id}': {ve}")
            raise  # Re-raise the specific error
        except Exception as e:
            logger.error(
                f"Failed to get connection details or create client for MCP server '{server_id}': {e}",
                exc_info=True,
            )
            raise ValueError(f"Failed to initialize MCP client for {server_id}: {e}")

    def get_all_clients(self) -> Dict[str, MCPClient]:
        """
        Get all registered and potentially configurable MCP clients based on ConfigManager.

        Returns:
            Dictionary mapping server config_ids to MCPClient instances.
        """
        clients = {}

        if not self.config_manager.connectors:
            logger.warning(
                "ConfigManager has no connectors registered, cannot retrieve MCP clients."
            )
            return clients

        # Iterate through connectors registered in ConfigManager
        # Assuming MCP servers are registered with type 'api' or specific types like 'postgres'
        mcp_connector_types = [
            "api",
            "postgres",
            "rag",
            "obsidian",
        ]  # Add other MCP types if needed
        for connector in self.config_manager.connectors:
            # Check if the connector type indicates an MCP server
            if connector.type in mcp_connector_types:
                server_id = connector.config_id
                try:
                    # Attempt to get/create the client for this connector
                    # This reuses existing clients if already created
                    clients[server_id] = self.get_client(server_id)
                except Exception as e:
                    # Log warning but continue trying to initialize other clients
                    logger.warning(f"Failed to initialize MCP client for '{server_id}': {e}")

        if not clients:
            logger.warning("No MCP clients could be successfully initialized.")

        return clients
