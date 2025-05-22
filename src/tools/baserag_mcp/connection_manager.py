import asyncio
import logging
import os
from typing import Dict, Optional

import aiohttp
from confidentialmind_core.config_manager import load_environment

from .connectors import BaseRAGConnectorManager
from .settings import settings

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages BaseRAG API connections with ConfidentialMind integration."""

    _session: Optional[aiohttp.ClientSession] = None
    _connection_error: Optional[str] = None
    _is_connecting: bool = False
    _reconnect_task: Optional[asyncio.Task] = None
    _current_url: Optional[str] = None
    _current_headers: Optional[Dict[str, str]] = None
    _connector_manager: Optional[BaseRAGConnectorManager] = None
    _is_stack_deployment: bool = False
    _background_poll_task: Optional[asyncio.Task] = None
    _initialized: bool = False
    _is_connected: bool = False

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

        # Initialize the connector manager
        cls._connector_manager = BaseRAGConnectorManager(settings.connector_id)
        try:
            # Always register connectors in stack mode, but make it optional in local mode
            register_connectors = cls._is_stack_deployment
            await cls._connector_manager.initialize(register_connectors=register_connectors)

            # Try to get a URL initially
            (
                cls._current_url,
                cls._current_headers,
            ) = await cls._connector_manager.fetch_baserag_api_params()
            if cls._current_url:
                logger.info(f"ConnectionManager: Initial URL available: {cls._current_url}")
                cls._is_connected = True
            else:
                logger.info("ConnectionManager: No initial URL available")
                cls._is_connected = False

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
        """Start background polling for BaseRAG API URL changes."""
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
                url, headers = await cls._connector_manager.fetch_baserag_api_params()

                # Log at appropriate intervals
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    if url:
                        logger.debug(f"ConnectionManager: Poll {retry_count}: Found URL {url}")
                    else:
                        logger.debug(f"ConnectionManager: Poll {retry_count}: No URL available")

                # Handle URL changes
                if url and (url != cls._current_url or headers != cls._current_headers):
                    logger.info(
                        f"ConnectionManager: URL or headers changed from {cls._current_url or 'None'} to {url}"
                    )
                    cls._current_url = url
                    cls._current_headers = headers or {}

                    # Update connection if URL changed
                    if cls._session:
                        await cls._session.close()
                        cls._session = None

                    # Create new session
                    cls._session = aiohttp.ClientSession(headers=cls._current_headers)
                    cls._is_connected = True
                    cls._connection_error = None
                    logger.info("ConnectionManager: Successfully connected to BaseRAG API")
            except Exception as e:
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.error(f"ConnectionManager: Error polling for URL changes: {e}")

            # Increment and wait
            retry_count += 1
            # Exponential backoff with a maximum wait time
            wait_time = min(30, 5 * (1.5 ** min(retry_count, 5)))
            await asyncio.sleep(wait_time)

    @classmethod
    async def create_session(cls) -> Optional[aiohttp.ClientSession]:
        """Create and return aiohttp session with proper headers."""
        if cls._is_connecting:
            logger.info("ConnectionManager: Connection attempt already in progress")
            return cls._session

        if cls._session is not None:
            logger.info("ConnectionManager: Session already exists, reusing")
            return cls._session

        cls._is_connecting = True
        try:
            # In stack deployment mode without URL, return None but don't raise
            if cls._is_stack_deployment and not cls._current_url:
                logger.info(
                    "ConnectionManager: No BaseRAG API URL available yet in stack deployment mode"
                )
                cls._connection_error = "No BaseRAG API URL available yet"
                cls._is_connecting = False
                return None

            # If no headers are available, use default settings
            if not cls._current_headers:
                cls._current_headers = {}
                if settings.api_key:
                    cls._current_headers["Authorization"] = f"Bearer {settings.api_key}"

            # Create the session
            logger.info(
                f"ConnectionManager: Creating BaseRAG API session with URL: {cls._current_url}"
            )
            cls._session = aiohttp.ClientSession(headers=cls._current_headers)

            # Test connection
            async with cls._session.get(f"{cls._current_url}/health") as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(
                        f"Failed to connect to BaseRAG API: Status {response.status}, Response: {response_text}"
                    )
                    raise ConnectionError(f"Failed to connect to BaseRAG API: {response.status}")

                response_data = await response.json()
                if response_data.get("status") != "ok":
                    raise ConnectionError(f"BaseRAG API health check failed: {response_data}")

            logger.info("ConnectionManager: BaseRAG API session created successfully")
            cls._is_connected = True
            cls._connection_error = None
            return cls._session

        except Exception as e:
            cls._connection_error = f"Failed to connect to BaseRAG API: {str(e)}"
            logger.error(f"ConnectionManager: BaseRAG API connection error: {e}")
            cls._session = None
            cls._is_connected = False
            cls._schedule_reconnect()

            # In stack deployment mode, return None instead of raising
            if cls._is_stack_deployment:
                logger.warning(
                    "ConnectionManager: Continuing without BaseRAG API connection in stack deployment mode"
                )
                cls._is_connecting = False
                return None

            raise ConnectionError(f"Failed to connect to BaseRAG API: {e}")
        finally:
            cls._is_connecting = False

    @classmethod
    def _schedule_reconnect(cls):
        """Schedule a reconnection attempt with exponential backoff."""
        if not cls._reconnect_task or cls._reconnect_task.done():
            cls._reconnect_task = asyncio.create_task(cls._reconnect_with_backoff())

    @classmethod
    async def _reconnect_with_backoff(cls):
        """Attempt to reconnect with exponential backoff."""
        retry_count = 0
        max_retries = 5

        while retry_count < max_retries:
            try:
                logger.info(
                    f"ConnectionManager: Reconnection attempt {retry_count + 1}/{max_retries}"
                )
                await cls.create_session()
                if cls._is_connected:
                    logger.info("ConnectionManager: Reconnection successful")
                    return
            except Exception as e:
                logger.error(f"ConnectionManager: Reconnection failed: {e}")

            # Exponential backoff
            retry_count += 1
            wait_time = min(30, 2**retry_count)
            await asyncio.sleep(wait_time)

        logger.error(f"ConnectionManager: Failed to reconnect after {max_retries} attempts")

    @classmethod
    async def close(cls):
        """Close the API session."""
        if cls._session:
            logger.info("ConnectionManager: Closing BaseRAG API session")
            if cls._reconnect_task and not cls._reconnect_task.done():
                cls._reconnect_task.cancel()
            await cls._session.close()
            cls._session = None
            cls._is_connected = False
            logger.info("ConnectionManager: BaseRAG API session closed")

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
    def get_session(cls) -> Optional[aiohttp.ClientSession]:
        """Get the current aiohttp session."""
        return cls._session

    @classmethod
    def is_connected(cls) -> bool:
        """Check if BaseRAG API is currently connected."""
        return cls._is_connected and cls._session is not None

    @classmethod
    def last_error(cls) -> Optional[str]:
        """Get the last connection error if any."""
        return cls._connection_error

    @classmethod
    def get_base_url(cls) -> Optional[str]:
        """Get the base URL for BaseRAG API."""
        return cls._current_url
