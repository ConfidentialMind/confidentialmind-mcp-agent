import asyncio
import logging
import os
from typing import Optional

from confidentialmind_core.config_manager import (
    ConfigManager,
    ConnectorSchema,
    get_api_parameters,
    load_environment,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseRAGConfig(BaseModel):
    """Minimal config for the BaseRAG MCP server."""

    name: str = "baserag-mcp-server"
    description: str = "BaseRAG API MCP Server"


class BaseRAGConnectorManager:
    """Manages connector configurations for BaseRAG MCP server."""

    def __init__(self, connector_id: str = "BASERAG_API"):
        """
        Initialize the connector manager.

        Args:
            connector_id: The connector ID to use for BaseRAG API access
        """
        self.initialized = False
        self.connector_id = connector_id

        load_environment()
        self.is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"BaseRAGConnectorManager: Initializing in {'stack deployment' if self.is_stack_deployment else 'local development'} mode"
        )
        self._background_fetch_task = None

    async def initialize(self, register_connectors=True):
        """
        Initialize connector configuration for the BaseRAG MCP server.

        Args:
            register_connectors: Whether to register connectors with ConfigManager
        """
        if self.initialized:
            logger.info("BaseRAGConnectorManager: Already initialized")
            return

        if register_connectors and self.is_stack_deployment:
            # Register BaseRAG API connector with the ConfigManager in stack deployment mode
            try:
                logger.info(
                    "BaseRAGConnectorManager: Registering BaseRAG API connector with ConfigManager"
                )
                config_manager = ConfigManager()
                connectors = [
                    # BaseRAG API connector
                    ConnectorSchema(
                        type="api",
                        label="BaseRAG API",
                        config_id=self.connector_id,
                    ),
                ]
                config_manager.init_manager(
                    config_model=BaseRAGConfig(),
                    connectors=connectors,
                )
                logger.info(
                    f"BaseRAGConnectorManager: Registered BaseRAG API connector with config_id={self.connector_id}"
                )
            except Exception as e:
                logger.error(f"BaseRAGConnectorManager: Error registering connectors: {e}")
                # Continue without raising - allow server to initialize without connectors

        self.initialized = True

        # Start background polling in stack deployment mode
        if self.is_stack_deployment:
            await self._start_background_polling()

    async def _start_background_polling(self):
        """Start background polling for BaseRAG API URL if in stack deployment mode."""
        if self._background_fetch_task is not None and not self._background_fetch_task.done():
            logger.info("BaseRAGConnectorManager: Background polling already started")
            return  # Already polling

        logger.info("BaseRAGConnectorManager: Starting background polling for BaseRAG API URL")
        self._background_fetch_task = asyncio.create_task(self._poll_for_url_in_background())

    async def _poll_for_url_in_background(self):
        """Monitor for URL changes in the background."""
        logger.info("BaseRAGConnectorManager: Background polling task started")
        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries

        while True:
            try:
                # Fetch URL on each iteration
                url, headers = await self.fetch_baserag_api_params()

                # Log at appropriate intervals
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    if url:
                        logger.debug(
                            f"BaseRAGConnectorManager: Poll {retry_count}: Found URL {url}"
                        )
                    else:
                        logger.debug(
                            f"BaseRAGConnectorManager: Poll {retry_count}: No URL available"
                        )

                # Exponential backoff with a maximum wait time
                wait_time = min(30, 5 * (1.5 ** min(retry_count, 5)))
                retry_count += 1
                await asyncio.sleep(wait_time)

            except Exception as e:
                # Log less frequently as retries increase
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.error(f"BaseRAGConnectorManager: Error in background polling: {e}")
                await asyncio.sleep(5)  # Simple backoff on error
                retry_count += 1

    async def fetch_baserag_api_params(self) -> tuple[Optional[str], Optional[dict]]:
        """Fetch BaseRAG API URL and headers using the appropriate method based on deployment mode."""
        try:
            url, headers = get_api_parameters(self.connector_id)
            if url:
                logger.debug(
                    f"BaseRAGConnectorManager: Successfully retrieved BaseRAG API URL: {url}"
                )
                return url, headers
            logger.debug(
                f"BaseRAGConnectorManager: No BaseRAG API URL found for {self.connector_id}"
            )
            return None, None
        except Exception as e:
            logger.error(f"BaseRAGConnectorManager: Error fetching BaseRAG API URL: {e}")
            return None, None
