import logging
from typing import Dict

from confidentialmind_core.config_manager import ConfigManager, get_api_parameters

from src.mcp.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class CMMCPManager:
    """
    Manager for MCP clients that uses the confidentialmind SDK to handle connections.
    Supports both regular and array connectors for dynamic MCP server configuration.
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

        # Check if the connector exists in the config manager (either regular or array)
        connector_exists = False

        # Check regular connectors
        if self.config_manager.connectors:
            connector_exists = any(c.config_id == server_id for c in self.config_manager.connectors)

        # If not found in regular connectors, check array connectors
        if not connector_exists and self.config_manager.array_connectors:
            connector_exists = any(
                c.config_id == server_id for c in self.config_manager.array_connectors
            )

        if not connector_exists:
            logger.error(
                f"MCP server config_id '{server_id}' not found in ConfigManager connectors."
            )
            raise ValueError(f"MCP server {server_id} not found in configuration.")

        try:
            # Get connection details from SDK ConfigManager
            # get_api_parameters handles retrieval from env vars or portal based on SDK logic
            url_or_urls, headers = get_api_parameters(server_id)

            # Check if we got a valid URL or URLs
            if not url_or_urls:
                logger.error(f"MCP server {server_id} is registered but no URL could be retrieved.")
                raise ValueError(f"URL for MCP server {server_id} not configured in SDK.")

            # Handle both single URL and list of URLs
            if isinstance(url_or_urls, list):
                # For array connectors, we use the first URL as the primary
                # In the future, this could be enhanced to support load balancing or failover
                base_url = url_or_urls[0]
                logger.info(f"Using first URL from array connector '{server_id}': {base_url}")
                if len(url_or_urls) > 1:
                    logger.info(f"Additional URLs available for '{server_id}': {url_or_urls[1:]}")
            else:
                base_url = url_or_urls

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
        Supports both regular and array connectors.

        Returns:
            Dictionary mapping server config_ids to MCPClient instances.
        """
        clients = {}

        # Check if config manager has connectors registered
        has_connectors = (
            self.config_manager.connectors and len(self.config_manager.connectors) > 0
        ) or (
            self.config_manager.array_connectors and len(self.config_manager.array_connectors) > 0
        )

        if not has_connectors:
            logger.warning(
                "ConfigManager has no connectors registered, cannot retrieve MCP clients."
            )
            return clients

        # MCP server types we should initialize
        mcp_connector_types = [
            "api",
            "postgres",
            "rag",
            "obsidian",
            "agent_tool",  # New type from updated SDK
            "endpoint",  # New type from updated SDK
        ]

        # Process regular connectors
        if self.config_manager.connectors:
            for connector in self.config_manager.connectors:
                # Check if the connector type indicates an MCP server
                if connector.type in mcp_connector_types:
                    server_id = connector.config_id
                    try:
                        # Attempt to get/create the client for this connector
                        clients[server_id] = self.get_client(server_id)
                    except Exception as e:
                        # Log warning but continue trying to initialize other clients
                        logger.warning(f"Failed to initialize MCP client for '{server_id}': {e}")

        # Process array connectors
        if self.config_manager.array_connectors:
            for array_connector in self.config_manager.array_connectors:
                # Check if the connector type indicates an MCP server
                if array_connector.type in mcp_connector_types:
                    server_id = array_connector.config_id
                    try:
                        # Attempt to get/create the client for this array connector
                        clients[server_id] = self.get_client(server_id)
                    except Exception as e:
                        # Log warning but continue trying to initialize other clients
                        logger.warning(
                            f"Failed to initialize array MCP client for '{server_id}': {e}"
                        )

        if not clients:
            logger.warning("No MCP clients could be successfully initialized.")

        return clients
