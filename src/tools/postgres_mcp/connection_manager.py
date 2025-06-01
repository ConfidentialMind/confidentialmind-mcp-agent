import asyncio
import logging
import os
from typing import Any, Dict, Optional

import asyncpg
import backoff
from confidentialmind_core.config_manager import load_environment

from .connectors import PostgresConnectorManager
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
    _connector_manager: Optional[PostgresConnectorManager] = None
    _is_stack_deployment: bool = False
    _background_poll_task: Optional[asyncio.Task] = None
    _initialized: bool = False

    @classmethod
    async def initialize(cls) -> bool:
        """Initialize connection manager and discover URLs if needed."""
        if cls._initialized:
            logger.debug("ConnectionManager: Already initialized")
            return True

        # Determine deployment mode
        load_environment()
        cls._is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"ConnectionManager: Initializing in {'stack' if cls._is_stack_deployment else 'local'} mode"
        )

        # Check if we have an explicit DSN
        if settings.dsn:
            logger.info("ConnectionManager: Using explicit DSN from PG_DSN environment variable")
            cls._initialized = True
            # No need for connector manager or background polling with explicit DSN
            return True

        # Initialize the connector manager
        cls._connector_manager = PostgresConnectorManager(settings.connector_id)
        try:
            # Always register connectors in stack mode, but make it optional in local mode
            register_connectors = cls._is_stack_deployment
            await cls._connector_manager.initialize(register_connectors=register_connectors)

            # Try to get a URL initially
            cls._current_url = await cls._connector_manager.fetch_database_url()
            if cls._current_url:
                logger.info(f"ConnectionManager: Initial URL available: {cls._current_url}")
            else:
                logger.info("ConnectionManager: No initial URL available")

            # Start background polling for URL changes in stack mode
            if cls._is_stack_deployment:
                await cls._start_background_polling()

            cls._initialized = True
            return True
        except Exception as e:
            logger.error(f"ConnectionManager: Error during initialization: {e}")
            return False

    @classmethod
    async def _start_background_polling(cls):
        """Start background polling for database URL changes."""
        if cls._background_poll_task is not None and not cls._background_poll_task.done():
            logger.info("ConnectionManager: URL polling task already running")
            return

        logger.info("ConnectionManager: Starting background polling for URL changes")
        cls._background_poll_task = asyncio.create_task(cls._poll_for_url_changes())

    @classmethod
    async def _poll_for_url_changes(cls):
        """Poll for URL changes and update connections as needed."""
        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries

        while True:
            try:
                url = await cls._connector_manager.fetch_database_url()

                # Log at appropriate intervals
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    if url:
                        logger.debug(f"ConnectionManager: Poll {retry_count}: Found URL {url}")
                    else:
                        logger.debug(f"ConnectionManager: Poll {retry_count}: No URL available")

                # Handle URL changes
                if url and url != cls._current_url:
                    logger.info(
                        f"ConnectionManager: URL changed from {cls._current_url or 'None'} to {url}"
                    )
                    cls._current_url = url

                    # Update connection if URL changed
                    if cls._pool:
                        logger.info("ConnectionManager: Reconnecting with new URL")
                        await cls.close()

                    # Try to create a new pool
                    try:
                        await cls.create_pool()
                    except Exception as e:
                        logger.error(f"ConnectionManager: Failed to connect with new URL: {e}")
            except Exception as e:
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.error(f"ConnectionManager: Error polling for URL changes: {e}")

            # Increment and wait
            retry_count += 1
            # Exponential backoff with a maximum wait time
            wait_time = min(30, 5 * (1.5 ** min(retry_count, 5)))
            await asyncio.sleep(wait_time)

    @classmethod
    async def fetch_url(cls) -> Optional[str]:
        """Fetch database URL from connector manager."""
        if cls._connector_manager is None:
            cls._connector_manager = PostgresConnectorManager(settings.connector_id)
            await cls._connector_manager.initialize(register_connectors=False)

        return await cls._connector_manager.fetch_database_url()

    @classmethod
    async def create_pool(cls) -> Optional[asyncpg.Pool]:
        """Create and return connection pool with retry support."""
        if cls._is_connecting:
            logger.info("ConnectionManager: Connection attempt already in progress")
            return cls._pool

        if cls._pool is not None:
            logger.info("ConnectionManager: Pool already exists, reusing")
            return cls._pool

        cls._is_connecting = True
        try:
            # In stack deployment mode without URL, return None but don't raise
            if settings.dsn:
                logger.info(
                    "ConnectionManager: Using explicit DSN from PG_DSN environment variable"
                )
                connection_string = settings.get_connection_string()
            else:
                if cls._is_stack_deployment and not cls._current_url:
                    logger.info(
                        "ConnectionManager: No database URL available yet in stack deployment mode"
                    )
                    cls._connection_error = "No database URL available yet"
                    cls._is_connecting = False
                    return None

                # Get connection string using the URL
                connection_string = settings.get_connection_string(cls._current_url)
                logger.info(
                    f"ConnectionManager: Creating database connection pool with URL: {cls._current_url}"
                )

            # Create the pool
            cls._pool = await asyncpg.create_pool(
                dsn=connection_string,
                min_size=1,
                max_size=5,
                command_timeout=60.0,
                server_settings={
                    "statement_timeout": "10000",  # 10 seconds
                    "idle_in_transaction_session_timeout": "10000",  # 10 seconds
                    "lock_timeout": "2000",  # 2 seconds
                },
            )

            if not cls._pool:
                raise ConnectionError("Failed to create database pool")

            # Test connection
            async with cls._pool.acquire() as conn:
                await conn.execute("SELECT 1")

            logger.info("ConnectionManager: Database connection pool created successfully")
            cls._connection_error = None
            return cls._pool

        except Exception as e:
            cls._connection_error = f"Failed to connect to database: {str(e)}"
            logger.error(f"ConnectionManager: Database connection error: {e}")
            cls._pool = None
            cls._schedule_reconnect()

            # In stack deployment mode, return None instead of raising
            if cls._is_stack_deployment:
                logger.warning(
                    "ConnectionManager: Continuing without database connection in stack deployment mode"
                )
                cls._is_connecting = False
                return None

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
            logger.error(f"ConnectionManager: Reconnection failed: {e}")
            raise

    @classmethod
    async def close(cls):
        """Close the connection pool."""
        if cls._pool:
            logger.info("ConnectionManager: Closing database connection pool")
            if cls._reconnect_task and not cls._reconnect_task.done():
                cls._reconnect_task.cancel()
            await cls._pool.close()
            cls._pool = None
            logger.info("ConnectionManager: Database connection pool closed")

        # Also cancel background polling task
        if cls._background_poll_task and not cls._background_poll_task.done():
            logger.info("ConnectionManager: Cancelling background polling task")
            cls._background_poll_task.cancel()
            try:
                await cls._background_poll_task
            except asyncio.CancelledError:
                pass
            cls._background_poll_task = None

    @classmethod
    def get_pool(cls) -> Optional[asyncpg.Pool]:
        """Get the current connection pool."""
        return cls._pool

    @classmethod
    def is_connected(cls) -> bool:
        """Check if database is currently connected."""
        return cls._pool is not None and not cls._connection_error

    @classmethod
    def last_error(cls) -> Optional[str]:
        """Get the last connection error if any."""
        return cls._connection_error
