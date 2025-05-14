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

    async def initialize(self, register_connectors=True):
        """Initialize connector configuration for the PostgreSQL MCP server."""
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

                # Initialize the ConfigManager with our connector
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
        # Just log that we're monitoring - the actual fetch happens in fetch_database_url
        try:
            while True:
                await asyncio.sleep(30)  # Just keep the task alive
        except asyncio.CancelledError:
            logger.info("PostgresConnectorManager: Background polling task cancelled")
        except Exception as e:
            logger.error(f"PostgresConnectorManager: Error in background polling task: {e}")

    async def fetch_database_url(self) -> Optional[str]:
        """Fetch database URL using the appropriate method based on deployment mode."""
        try:
            url, _ = get_api_parameters(self.connector_id)
            if url:
                logger.info(f"PostgresConnectorManager: Successfully retrieved database URL: {url}")
                return url
            logger.debug(f"PostgresConnectorManager: No database URL found for {self.connector_id}")
            return None
        except Exception as e:
            logger.error(f"PostgresConnectorManager: Error fetching database URL: {e}")
            return None
