import asyncio
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
        self.initialized = False
        load_environment()
        self.is_stack_deployment = (
            not os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() == "true"
        )
        logger.info(
            f"Initializing in {'stack deployment' if self.is_stack_deployment else 'local development'} mode"
        )

    async def initialize(self, register_connectors=True):
        """Initialize connector configuration for the application."""
        if self.initialized:
            return

        config_manager = ConfigManager()

        if register_connectors and self.is_stack_deployment:
            # Register connectors with the ConfigManager in stack deployment mode
            try:
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
                    f"Registered {len(connectors)} connectors and {len(array_connectors)} array connectors"
                )
            except Exception as e:
                logger.error(f"Error registering connectors: {e}")
                # Continue without raising - allow app to initialize without connectors

        self.initialized = True
        return config_manager

    async def fetch_database_url(self, config_id: str = "DATABASE") -> Optional[str]:
        """Fetch database URL using the appropriate method based on deployment mode."""
        if not self.is_stack_deployment:
            # Local development mode - use get_api_parameters directly
            try:
                url, _ = get_api_parameters(config_id)
                if url:
                    logger.info(f"Successfully retrieved database URL from environment: {url}")
                    return url
                logger.warning(f"No database URL found in environment for {config_id}")
                return None
            except Exception as e:
                logger.error(f"Error fetching database URL from environment: {e}")
                return None
        else:
            # Stack deployment mode - use infinite polling loop
            return await self._fetch_url_with_polling(config_id)

    async def fetch_llm_url(self, config_id: str = "LLM") -> Tuple[Optional[str], Optional[Dict]]:
        """Fetch LLM URL and headers using the appropriate method based on deployment mode."""
        if not self.is_stack_deployment:
            # Local development mode - use get_api_parameters directly
            try:
                url, headers = get_api_parameters(config_id)
                if url:
                    logger.info(f"Successfully retrieved LLM URL from environment: {url}")
                    return url, headers
                logger.warning(f"No LLM URL found in environment for {config_id}")
                return None, None
            except Exception as e:
                logger.error(f"Error fetching LLM URL from environment: {e}")
                return None, None
        else:
            # Stack deployment mode - use infinite polling loop
            url = await self._fetch_url_with_polling(config_id)
            return url, {}  # In stack mode, headers are handled differently

    async def fetch_mcp_servers(self, config_id: str = "MCP_SERVERS") -> Dict[str, str]:
        """Fetch MCP server URLs from stack or environment."""
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
                logger.warning("No MCP servers found in environment variables")

            return servers
        else:
            # Stack deployment mode - get from ArrayConnectorSchema
            return await self._fetch_mcp_servers_with_polling(config_id)

    async def _fetch_url_with_polling(self, config_id: str) -> Optional[str]:
        """Poll for URL indefinitely until provided by the stack."""
        logger.info(f"Waiting for {config_id} URL from stack...")
        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries

        while True:
            try:
                url, _ = get_api_parameters(config_id)
                if url:
                    logger.info(f"Successfully retrieved {config_id} URL from stack: {url}")
                    return url
            except Exception as e:
                # Log less frequently as retries increase
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.debug(f"Attempt {retry_count}: Error fetching {config_id} URL: {e}")

            retry_count += 1
            # Exponential backoff with a maximum wait time
            wait_time = min(5, 1 * (1.5 ** min(retry_count, 10)))
            await asyncio.sleep(wait_time)

    async def _fetch_mcp_servers_with_polling(self, config_id: str) -> Dict[str, str]:
        """Poll for MCP server URLs indefinitely until provided by the stack."""
        logger.info(f"Waiting for {config_id} URLs from stack...")

        config_manager = ConfigManager()
        servers = {}
        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries

        while True:
            try:
                # Get stack IDs from the array connector
                stack_ids = config_manager.getStackIdForConnector(config_id)

                if stack_ids and isinstance(stack_ids, list) and len(stack_ids) > 0:
                    # Get URLs for each stack ID
                    urls = config_manager.getUrlForConnector(config_id)

                    if urls and isinstance(urls, list):
                        # Create a dictionary of server_id -> url
                        for i, stack_id in enumerate(stack_ids):
                            if i < len(urls):
                                servers[stack_id.lower()] = urls[i]

                        if servers:
                            logger.info(f"Retrieved {len(servers)} MCP server URLs from stack")
                            return servers
            except Exception as e:
                # Log less frequently as retries increase
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.debug(f"Attempt {retry_count}: Error fetching MCP server URLs: {e}")

            # Return empty dict if no servers found yet (allows application to start)
            if retry_count == 0:
                logger.warning(
                    "No MCP servers found yet, returning empty list. Will continue polling in background."
                )
                return {}

            retry_count += 1
            # Exponential backoff with a maximum wait time
            wait_time = min(5, 1 * (1.5 ** min(retry_count, 10)))
            await asyncio.sleep(wait_time)
