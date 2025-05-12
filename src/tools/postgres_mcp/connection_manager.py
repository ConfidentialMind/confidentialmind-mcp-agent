import asyncio
import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import asyncpg
import backoff
from confidentialmind_core.config_manager import get_api_parameters, load_environment

from .settings import settings

logger = logging.getLogger(__name__)


def get_backoff_config() -> Dict[str, Any]:
    """Configuration for connection retry backoff."""
    return {
        "max_tries": 5,
        "max_time": 30,
        "on_backoff": lambda details: logger.info(
            f"Reconnection attempt {details['tries']} failed. Retrying in {details['wait']} seconds"
        ),
    }


class ConnectionManager:
    """Manages PostgreSQL connections with ConfidentialMind integration."""

    _pool: Optional[asyncpg.pool.Pool] = None
    _connection_error: Optional[str] = None
    _is_connecting: bool = False
    _reconnect_task: Optional[asyncio.Task] = None
    _current_url: Optional[str] = None

    @classmethod
    async def fetch_url(cls) -> Optional[str]:
        """Fetch database URL from ConfidentialMind SDK or environment."""
        try:
            url, _ = get_api_parameters(settings.connector_id)
            if url:
                logger.info(f"Retrieved database URL from ConfigManager: {url}")
                return url
        except Exception as e:
            logger.warning(f"Error fetching database URL from ConfigManager: {e}")

        # Fallback to environment variables is handled by settings
        return None

    @classmethod
    async def initialize(cls) -> bool:
        """Initialize connection manager and discover URLs if needed."""
        if settings.use_sdk_connector:
            load_environment()
            cls._current_url = await cls.fetch_url()
        return True

    @classmethod
    async def create_pool(cls) -> asyncpg.Pool:
        """Create and return connection pool with retry support."""
        if cls._pool is not None:
            return cls._pool

        cls._is_connecting = True
        try:
            connection_string = settings.get_connection_string(cls._current_url)

            logger.info(f"Creating database connection pool")

            cls._pool = await asyncpg.create_pool(
                dsn=connection_string, min_size=1, max_size=5, command_timeout=60.0
            )

            if not cls._pool:
                raise ConnectionError("Failed to create database pool")

            # Test connection
            async with cls._pool.acquire() as conn:
                await conn.execute("SELECT 1")

            logger.info("Database connection pool created successfully")
            cls._connection_error = None
            return cls._pool

        except Exception as e:
            cls._connection_error = f"Failed to connect to database: {str(e)}"
            logger.error(f"Database connection error: {e}")
            cls._pool = None
            cls._schedule_reconnect()
            raise ConnectionError(f"Failed to connect to database: {e}")
        finally:
            cls._is_connecting = False

    @classmethod
    def _schedule_reconnect(cls):
        """Schedule a reconnection attempt with exponential backoff."""
        if not cls._reconnect_task or cls._reconnect_task.done():
            cls._reconnect_task = asyncio.create_task(cls._reconnect_with_backoff())

    @classmethod
    @backoff.on_exception(backoff.expo, Exception, **get_backoff_config())
    async def _reconnect_with_backoff(cls):
        """Attempt to reconnect with exponential backoff."""
        try:
            await cls.create_pool()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            raise

    @classmethod
    async def close(cls):
        """Close the connection pool."""
        if cls._pool:
            logger.info("Closing database connection pool")
            if cls._reconnect_task and not cls._reconnect_task.done():
                cls._reconnect_task.cancel()
            await cls._pool.close()
            cls._pool = None
            logger.info("Database connection pool closed")
