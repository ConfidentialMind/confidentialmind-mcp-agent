import logging
import os
from typing import Dict, Optional, Tuple

from confidentialmind_core.config_manager import (
    ArrayConnectorSchema,
    ConfigManager,
    ConnectorSchema,
    get_api_parameters,
    load_environment,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Minimal config for the agent."""

    name: str = "confidentialmind-agent"
    description: str = "FastMCP agent for ConfidentialMind stack"


class ConnectorConfigManager:
    """Manages connector configurations for stack deployment and local development."""

    def __init__(self):
        """Initialize connector configuration manager."""
        self.initialized = False
        load_environment()
        self.is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )
        logger.info(
            f"ConnectorConfigManager: Initializing in {'stack deployment' if self.is_stack_deployment else 'local development'} mode"
        )
        self._background_tasks = []

    async def initialize(self, register_connectors=True):
        """
        Initialize connector configuration for the application.

        Args:
            register_connectors: Whether to register connectors with ConfigManager
        """
        if self.initialized:
            logger.debug("ConnectorConfigManager: Already initialized")
            return

        if register_connectors and self.is_stack_deployment:
            # Register connectors with the ConfigManager in stack deployment mode
            try:
                logger.info("ConnectorConfigManager: Registering connectors with ConfigManager")
                config_manager = ConfigManager()

                # Register connectors that the agent needs
                connectors = [
                    # Database connector for session management
                    ConnectorSchema(
                        type="database",
                        label="Session Management Database",
                        config_id="DATABASE",
                    ),
                    # LLM connector for language model integration
                    ConnectorSchema(
                        type="llm",
                        label="Language Model",
                        config_id="LLM",
                    ),
                ]

                # MCP servers connector (supports multiple servers)
                array_connectors = [
                    ArrayConnectorSchema(
                        type="agent_tool",
                        label="MCP Tool Servers",
                        config_id="agentTools",
                    )
                ]

                # Initialize the ConfigManager with our connectors
                config_manager.init_manager(
                    config_model=AgentConfig(),
                    connectors=connectors,
                    array_connectors=array_connectors,
                )

                logger.info(
                    f"ConnectorConfigManager: Registered {len(connectors)} connectors and {len(array_connectors)} array connectors"
                )
            except Exception as e:
                logger.error(f"ConnectorConfigManager: Error registering connectors: {e}")
                # Continue without raising - allow app to initialize without connectors

        self.initialized = True
        return True

    async def fetch_database_url(self, config_id: str = "DATABASE") -> Optional[str]:
        """
        Fetch database URL using the appropriate method based on deployment mode.

        Args:
            config_id: ConfigManager connector ID for the database

        Returns:
            Database URL or None if not available
        """
        try:
            url, _ = get_api_parameters(config_id)
            if url:
                logger.debug(
                    f"ConnectorConfigManager: Successfully retrieved database URL for {config_id}: {url}"
                )
                return url
            logger.debug(f"ConnectorConfigManager: No database URL found for {config_id}")
            return None
        except Exception as e:
            logger.error(
                f"ConnectorConfigManager: Error fetching database URL for {config_id}: {e}"
            )
            return None

    async def fetch_llm_url(self, config_id: str = "LLM") -> Tuple[Optional[str], Optional[Dict]]:
        """
        Fetch LLM URL and headers using the appropriate method based on deployment mode.

        Args:
            config_id: ConfigManager connector ID for the LLM

        Returns:
            Tuple of (URL, headers) or (None, None) if not available
        """
        try:
            url, headers = get_api_parameters(config_id)
            if url:
                logger.debug(
                    f"ConnectorConfigManager: Successfully retrieved LLM URL for {config_id}: {url}"
                )
                return url, headers
            logger.debug(f"ConnectorConfigManager: No LLM URL found for {config_id}")
            return None, None
        except Exception as e:
            logger.error(f"ConnectorConfigManager: Error fetching LLM URL for {config_id}: {e}")
            return None, None

    async def fetch_mcp_servers(self, config_id: str = "agentTools") -> Dict[str, str]:
        """
        Fetch MCP server URLs from stack or environment.

        Args:
            config_id: ConfigManager connector ID for the MCP servers

        Returns:
            Dictionary mapping server_id to server_url
        """
        if not self.is_stack_deployment:
            # Local development mode - collect from environment variables
            servers = {}

            # Look for MCP_SERVER_* environment variables
            for key, value in os.environ.items():
                if key.startswith("MCP_SERVER_") and value:
                    server_id = key[11:].lower()  # Remove "MCP_SERVER_" prefix
                    servers[server_id] = value

            # Add default if available
            default_mcp = os.environ.get("AGENT_TOOLS_URL")
            if default_mcp and "agenttools" not in servers:
                servers["agenttools"] = default_mcp

            if not servers:
                logger.warning(
                    "ConnectorConfigManager: No MCP servers found in environment variables"
                )

            return servers
        else:
            # Stack deployment mode - get from ArrayConnectorSchema
            try:
                config_manager = ConfigManager()

                # Get stack IDs from the array connector
                stack_ids = config_manager.getStackIdForConnector(config_id)
                if not stack_ids or not isinstance(stack_ids, list) or len(stack_ids) == 0:
                    logger.warning(f"ConnectorConfigManager: No stack IDs found for {config_id}")
                    return {}

                # Get URLs for each stack ID
                urls = config_manager.getUrlForConnector(config_id)
                if not urls or not isinstance(urls, list) or len(urls) == 0:
                    logger.warning(f"ConnectorConfigManager: No URLs found for {config_id}")
                    return {}

                # Create a dictionary of server_id -> url
                servers = {}
                for i, stack_id in enumerate(stack_ids):
                    if i < len(urls):
                        servers[stack_id.lower()] = urls[i]

                if servers:
                    logger.info(
                        f"ConnectorConfigManager: Retrieved {len(servers)} MCP server URLs from stack"
                    )
                    return servers
                else:
                    logger.warning(f"ConnectorConfigManager: No MCP servers found for {config_id}")
                    return {}
            except Exception as e:
                logger.error(f"ConnectorConfigManager: Error fetching MCP servers: {e}")
                return {}
