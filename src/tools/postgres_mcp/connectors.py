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


class PostgresConfig(BaseModel):
    """Minimal config for the Postgres MCP server."""

    name: str = "postgres-mcp-server"
    description: str = "Read-only PostgreSQL MCP Server"


class PostgresConnectorManager:
    """Manages connector configurations for PostgreSQL MCP server."""

    def __init__(self, connector_id: str = "DATABASE"):
        """
        Initialize the connector manager.

        Args:
            connector_id: The connector ID to use for database access
        """
        self.initialized = False
        self.connector_id = connector_id

        load_environment()
        self.is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )
        logger.info(
            f"PostgresConnectorManager: Initializing in {'stack deployment' if self.is_stack_deployment else 'local development'} mode"
        )
        self._background_fetch_task = None
        self._last_fetch_time = 0

    async def initialize(self, register_connectors=True):
        """
        Initialize connector configuration for the PostgreSQL MCP server.

        Args:
            register_connectors: Whether to register connectors with ConfigManager
        """
        if self.initialized:
            logger.info("PostgresConnectorManager: Already initialized")
            return

        if register_connectors and self.is_stack_deployment:
            # Register database connector with the ConfigManager in stack deployment mode
            try:
                logger.info(
                    "PostgresConnectorManager: Registering database connector with ConfigManager"
                )
                config_manager = ConfigManager()
                connectors = [
                    # Database connector for PostgreSQL MCP server
                    ConnectorSchema(
                        type="database",
                        label="PostgreSQL Database",
                        config_id=self.connector_id,
                    ),
                ]
                config_manager.init_manager(
                    config_model=PostgresConfig(),
                    connectors=connectors,
                )
                logger.info(
                    f"PostgresConnectorManager: Registered database connector with config_id={self.connector_id}"
                )
            except Exception as e:
                logger.error(f"PostgresConnectorManager: Error registering connectors: {e}")
                # Continue without raising - allow server to initialize without connectors

        self.initialized = True

        # Start background polling in stack deployment mode
        if self.is_stack_deployment:
            await self._start_background_polling()

    async def _start_background_polling(self):
        """Start background polling for database URL if in stack deployment mode."""
        if self._background_fetch_task is not None and not self._background_fetch_task.done():
            logger.info("PostgresConnectorManager: Background polling already started")
            return  # Already polling

        logger.info("PostgresConnectorManager: Starting background polling for database URL")
        self._background_fetch_task = asyncio.create_task(self._poll_for_url_in_background())
        logger.info("PostgresConnectorManager: Background polling task created")

    async def _poll_for_url_in_background(self):
        """Monitor for URL changes in the background."""
        logger.info("PostgresConnectorManager: Background polling task started")
        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries

        try:
            while True:
                try:
                    # Fetch URL on each iteration
                    url = await self.fetch_database_url()

                    # Log at appropriate intervals
                    if retry_count < max_retry_log or retry_count % 10 == 0:
                        if url:
                            logger.debug(
                                f"PostgresConnectorManager: Poll {retry_count}: Found URL {url}"
                            )
                        else:
                            logger.debug(
                                f"PostgresConnectorManager: Poll {retry_count}: No URL available"
                            )

                    # Exponential backoff with a maximum wait time
                    wait_time = min(30, 5 * (1.5 ** min(retry_count, 5)))
                    retry_count += 1
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    # Log less frequently as retries increase
                    if retry_count < max_retry_log or retry_count % 10 == 0:
                        logger.error(f"PostgresConnectorManager: Error in background polling: {e}")
                    await asyncio.sleep(5)  # Simple backoff on error
                    retry_count += 1
        except asyncio.CancelledError:
            logger.info("PostgresConnectorManager: Background polling task cancelled")
        except Exception as e:
            logger.error(f"PostgresConnectorManager: Fatal error in background polling task: {e}")

    async def fetch_database_url(self) -> Optional[str]:
        """Fetch database URL using the appropriate method based on deployment mode."""
        try:
            url, _ = get_api_parameters(self.connector_id)
            if url:
                logger.debug(
                    f"PostgresConnectorManager: Successfully retrieved database URL: {url}"
                )
                return url
            logger.debug(f"PostgresConnectorManager: No database URL found for {self.connector_id}")
            return None
        except Exception as e:
            logger.error(f"PostgresConnectorManager: Error fetching database URL: {e}")
            return None
