import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import asyncpg
import backoff
from confidentialmind_core.config_manager import load_environment
from pydantic_settings import BaseSettings

from src.agent.connectors import ConnectorConfigManager
from src.agent.state import Message

# Configure logging
logger = logging.getLogger(__name__)


def get_backoff_config() -> Dict[str, Any]:
    """Get backoff configuration for database reconnection attempts"""
    return {
        "max_tries": 5,
        "max_time": 30,  # Maximum time to spend on backoff retries (seconds)
        "on_backoff": lambda details: logger.info(
            f"Database reconnection attempt {details['tries']} failed. Retrying in {details['wait']} seconds..."
        ),
    }


class DatabaseSettings(BaseSettings):
    """Settings for PostgreSQL connection used by the agent for session management"""

    # Default connection settings (used for local development)
    database_host: str = "localhost"
    database_port: int = 5432
    database_user: str = "app"
    database_password: str = "testpass"
    database_name: str = "postgres"

    # Pool settings
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 30.0
    command_timeout: float = 60.0

    def get_connection_string(self, db_url: Optional[str] = None) -> str:
        """
        Generate PostgreSQL connection string.

        Args:
            db_url: Optional database URL from SDK. If provided, it's used for the host part.

        Returns:
            Connection string DSN
        """
        logger.debug(f"Generating connection string. SDK URL provided: {db_url}")

        is_local_config = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() == "true"
        )

        if is_local_config and db_url:
            # Local config mode with URL - use the SDK-provided URL as the host part
            host_part = db_url
            logger.info(f"Using SDK-provided DB host/endpoint in local config mode: {host_part}")

            # Construct DSN using potentially overridden user/pass/db from BaseSettings
            dsn = f"postgresql://{self.database_user}:{self.database_password}@{host_part}/{self.database_name}"
        elif is_local_config:
            # Local config mode with no URL - use default settings
            host_part = f"{self.database_host}:{self.database_port}"
            logger.info(f"Using default DB settings in local config mode: {host_part}")

            # Construct DSN using potentially overridden user/pass/db from BaseSettings
            dsn = f"postgresql://{self.database_user}:{self.database_password}@{host_part}/{self.database_name}"
        else:
            # Stack deployment mode - use the hostname from stack with credentials from settings
            if not db_url:
                raise ValueError("No database URL provided in stack deployment mode")

            dsn = f"postgresql://{self.database_user}:{self.database_password}@{db_url}/{self.database_name}"
            logger.info("Using stack configuration with settings credentials")

        logger.info(f"Constructed DSN: {self._get_safe_connection_string(dsn)}")
        return dsn

    def _get_safe_connection_string(self, connection_string: Optional[str]) -> str:
        """Mask password in connection string for safe logging"""
        if not connection_string:
            return "DSN not configured"
        try:
            parts = urlparse(connection_string)
            if "@" in parts.netloc:
                creds, host_port = parts.netloc.split("@", 1)
                if ":" in creds:
                    user, _ = creds.split(":", 1)
                    masked_netloc = f"{user}:****@{host_port}"
                    safe_parts = parts._replace(netloc=masked_netloc)
                    return urlunparse(safe_parts)
            return connection_string
        except Exception:
            return "postgresql://user:****@host:port/database"


class Database:
    """Database connection management for agent session history storage"""

    def __init__(self, settings: DatabaseSettings):
        """Initialize with database settings"""
        self.settings = settings
        self._pool: Optional[asyncpg.pool.Pool] = None
        self._connection_error: Optional[str] = None
        self._is_connecting: bool = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._current_db_url: Optional[str] = None
        self._config_id: str = os.environ.get("DB_CONFIG_ID", "DATABASE")
        self._background_fetch_task: Optional[asyncio.Task] = None

        # Load environment variables to set LOCAL_CONFIG flag
        load_environment()
        self._is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"Database: Initialized in {'stack deployment' if self._is_stack_deployment else 'local config'} mode"
        )

    async def _start_background_polling(self):
        """Start background polling for database URL if in stack deployment mode."""
        if self._background_fetch_task is not None and not self._background_fetch_task.done():
            logger.info("Database: Background polling already started")
            return  # Already polling

        if self._is_stack_deployment:
            logger.info("Database: Starting background polling for URL changes")
            self._background_fetch_task = asyncio.create_task(self._poll_for_url_in_background())

    async def _poll_for_url_in_background(self):
        """Continuously poll for database URL and attempt connection when available."""
        connector_manager = ConnectorConfigManager()
        await connector_manager.initialize(register_connectors=False)

        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries

        while True:
            try:
                if self.is_connected():
                    # Already connected, just sleep
                    await asyncio.sleep(30)
                    continue

                url = await connector_manager.fetch_database_url(self._config_id)

                # Only log at appropriate intervals
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    if url:
                        logger.debug(f"Database: Poll {retry_count}: Found URL {url}")
                    else:
                        logger.debug(f"Database: Poll {retry_count}: No URL available")

                if url and url != self._current_db_url:
                    logger.info("Database: Found new database URL, attempting connection")
                    self._current_db_url = url
                    await self.connect(url)
                    if self.is_connected():
                        logger.info("Database: Successfully connected in background")
            except Exception as e:
                # Log less frequently as retries increase
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.error(f"Database: Error in background polling: {e}")

            retry_count += 1
            # Exponential backoff with a maximum wait time
            wait_time = min(30, 5 * (1.5 ** min(retry_count, 5)))
            await asyncio.sleep(wait_time)

    async def connect(self, db_url: Optional[str] = None) -> bool:
        """
        Establish connection to database with proper locking and stack integration.

        This method handles both local and stack deployment modes, and won't raise exceptions
        if the database is not available in stack deployment mode.

        Args:
            db_url: Optional database URL override

        Returns:
            True if connection was successful, False otherwise
        """
        if self._is_connecting:
            logger.info("Database: Connection attempt already in progress")
            return self.is_connected()

        if self._pool is not None and self.is_connected():
            return True

        self._is_connecting = True
        try:
            # Fetch or use the provided database URL
            if db_url:
                self._current_db_url = db_url
            elif not self._current_db_url:
                # Use ConnectorConfigManager for consistency
                connector_manager = ConnectorConfigManager()

                # In stack mode, make sure connectors are registered
                if self._is_stack_deployment:
                    await connector_manager.initialize(register_connectors=True)
                else:
                    await connector_manager.initialize(register_connectors=False)

                self._current_db_url = await connector_manager.fetch_database_url(self._config_id)

                # Start background polling for URL changes
                if self._is_stack_deployment and not self._background_fetch_task:
                    await self._start_background_polling()

            if not self._current_db_url:
                if self._is_stack_deployment:
                    logger.warning(
                        "Database: No URL available yet. Agent will run in stateless mode until database is connected."
                    )
                    self._connection_error = "No database URL available yet"
                    self._is_connecting = False
                    return False
                else:
                    # In local mode, don't fail but warn
                    logger.warning(
                        "Database: No URL provided in local mode. Agent will run in stateless mode."
                    )
                    self._connection_error = "No database URL provided"
                    self._is_connecting = False
                    return False

            # Build connection string and connect
            connection_string = self.settings.get_connection_string(self._current_db_url)

            logger.info(
                f"Database: Connecting with connection string: {self.settings._get_safe_connection_string(connection_string)}"
            )

            self._pool = await asyncpg.create_pool(
                dsn=connection_string,
                min_size=self.settings.min_connections,
                max_size=self.settings.max_connections,
                timeout=self.settings.connection_timeout,
                command_timeout=self.settings.command_timeout,
                server_settings={
                    "statement_timeout": str(int(self.settings.command_timeout * 1000)),
                    "idle_in_transaction_session_timeout": "10000",
                    "lock_timeout": "2000",
                },
            )

            # Test the connection
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")

            logger.info("Database: Connected successfully")
            self._connection_error = None
            return True

        except Exception as e:
            self._connection_error = f"Failed to connect to database: {str(e)}"
            logger.error(f"Database: Connection error: {e}")
            self._pool = None
            self._schedule_reconnect()
            return False
        finally:
            self._is_connecting = False

    async def disconnect(self):
        """Disconnect from database and clean up resources"""
        if self._pool:
            logger.info("Database: Disconnecting")
            if self._reconnect_task and not self._reconnect_task.done():
                self._reconnect_task.cancel()
            await self._pool.close()
            self._pool = None
            logger.info("Database: Disconnected successfully")

        # Cancel background polling task
        if self._background_fetch_task and not self._background_fetch_task.done():
            logger.info("Database: Cancelling background polling task")
            self._background_fetch_task.cancel()
            try:
                await self._background_fetch_task
            except asyncio.CancelledError:
                pass
            self._background_fetch_task = None

    def is_connected(self) -> bool:
        """Check if database is currently connected"""
        return self._pool is not None and not self._connection_error

    def last_error(self) -> Optional[str]:
        """Get the last recorded connection error."""
        return self._connection_error

    def _schedule_reconnect(self):
        """Schedule a reconnection attempt with exponential backoff"""
        if not self._reconnect_task or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())

    @backoff.on_exception(backoff.expo, Exception, **get_backoff_config())
    async def _reconnect_with_backoff(self):
        """Attempt to reconnect with exponential backoff"""
        try:
            await self.connect(self._current_db_url)
        except Exception as e:
            logger.error(f"Database: Reconnection failed: {e}")
            raise

    async def ensure_connected(self):
        """Ensure database is connected, waiting for reconnection if necessary"""
        if not self.is_connected():
            success = await self.connect(self._current_db_url)
            if not success:
                raise ConnectionError(f"Failed to connect to database: {self.last_error()}")

    async def execute_query(self, query: str, *args, fetch_type: str = "all"):
        """
        Execute a database query with automatic connection management

        Args:
            query: SQL query string
            *args: Query parameters
            fetch_type: Type of fetch operation ('all', 'row', 'val', or 'none' for execute)

        Returns:
            Query results based on fetch_type
        """
        await self.ensure_connected()
        async with self._pool.acquire() as conn:
            try:
                if fetch_type == "all":
                    return await conn.fetch(query, *args)
                elif fetch_type == "row":
                    return await conn.fetchrow(query, *args)
                elif fetch_type == "val":
                    return await conn.fetchval(query, *args)
                elif fetch_type == "none":
                    return await conn.execute(query, *args)
                else:
                    raise ValueError(f"Unknown fetch_type: {fetch_type}")
            except Exception as e:
                logger.error(f"Database: Error executing query: {e}")
                raise

    # Session history methods

    async def load_history(self, session_id: str) -> List[Message]:
        """
        Load conversation history from the database for a session.

        If the database is not connected, returns an empty list,
        allowing the agent to operate in stateless mode.
        """
        if not self.is_connected():
            logger.warning("Database: Not connected. Running in stateless mode.")
            return []

        try:
            await self.ensure_connected()
            results = await self.execute_query(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE session_id = $1
                ORDER BY message_order
                """,
                session_id,
            )
            messages = [Message(role=row["role"], content=row["content"]) for row in results]
            logger.debug(f"Database: Loaded {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Database: Error loading history for session {session_id}: {e}")
            return []  # Return empty history on error

    async def save_message(self, session_id: str, message: Message) -> bool:
        """
        Save a message to the database for a session.

        If the database is not connected, returns False but doesn't raise an exception.
        """
        if not self.is_connected():
            logger.warning("Database: Not connected. Cannot save message.")
            return False

        try:
            await self.ensure_connected()
            max_order = await self.execute_query(
                "SELECT MAX(message_order) FROM conversation_messages WHERE session_id = $1",
                session_id,
                fetch_type="val",
            )
            message_order = 0 if max_order is None else max_order + 1
            await self.execute_query(
                """
                INSERT INTO conversation_messages
                (session_id, message_order, role, content, timestamp)
                VALUES ($1, $2, $3, $4, NOW())
                """,
                session_id,
                message_order,
                message.role,
                message.content,
                fetch_type="none",
            )
            logger.debug(f"Database: Saved message for session {session_id} order {message_order}")
            return True
        except Exception as e:
            if not self.is_connected():
                logger.warning(
                    f"Database: Not connected while saving msg for {session_id}: {self.last_error()}"
                )
            else:
                logger.error(f"Database: Error saving message for session {session_id}: {e}")
            return False

    async def clear_history(self, session_id: str) -> bool:
        """
        Clear the conversation history for a session in the database.

        If the database is not connected, returns False but doesn't raise an exception.
        """
        if not self.is_connected():
            logger.warning("Database: Not connected. Cannot clear history.")
            return False

        try:
            await self.ensure_connected()
            await self.execute_query(
                "DELETE FROM conversation_messages WHERE session_id = $1",
                session_id,
                fetch_type="none",
            )
            logger.info(f"Database: Cleared conversation history for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Database: Error clearing history for session {session_id}: {e}")
            return False

    async def ensure_schema(self) -> bool:
        """
        Ensure necessary database tables exist

        If the database is not connected, returns False but doesn't raise an exception.
        """
        if not self.is_connected():
            logger.warning("Database: Not connected. Cannot ensure schema.")
            return False

        try:
            await self.ensure_connected()

            # Check if table exists
            table_exists = await self.execute_query(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'conversation_messages')",
                fetch_type="val",
            )

            if not table_exists:
                # Create the table if it doesn't exist
                await self.execute_query(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        message_order INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_id 
                    ON conversation_messages(session_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_order 
                    ON conversation_messages(session_id, message_order);
                    """,
                    fetch_type="none",
                )
                logger.info("Database: Created conversation_messages table and indices")
                return True
            else:  # validate table structure
                expected_columns = {
                    "id": {"data_type": "integer", "is_nullable": "NO"},
                    "session_id": {"data_type": "text", "is_nullable": "NO"},
                    "message_order": {"data_type": "integer", "is_nullable": "NO"},
                    "role": {"data_type": "text", "is_nullable": "NO"},
                    "content": {"data_type": "text", "is_nullable": "NO"},
                    "timestamp": {"data_type": "timestamp with time zone", "is_nullable": "NO"},
                }

                # Get actual columns
                columns = await self.execute_query(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'conversation_messages'
                    """,
                    fetch_type="all",
                )

                actual_columns = {
                    col["column_name"]: {
                        "data_type": col["data_type"],
                        "is_nullable": col["is_nullable"],
                    }
                    for col in columns
                }

                # Check for missing columns
                missing_columns = set(expected_columns.keys()) - set(actual_columns.keys())
                if missing_columns:
                    logger.error(
                        f"Database: Missing columns in conversation_messages table: {missing_columns}"
                    )
                    return False

                # Check for columns with wrong type or nullability
                for col_name, expected_props in expected_columns.items():
                    if col_name in actual_columns:
                        actual_props = actual_columns[col_name]
                        if (
                            expected_props["data_type"] != actual_props["data_type"]
                            or expected_props["is_nullable"] != actual_props["is_nullable"]
                        ):
                            logger.error(
                                f"Database: Column {col_name} has incorrect properties. Expected: {expected_props}, Actual: {actual_props}"
                            )
                            return False

                # Check for indices
                expected_indices = [
                    "idx_conversation_messages_session_id",
                    "idx_conversation_messages_order",
                ]

                indices = await self.execute_query(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = 'conversation_messages'
                    """,
                    fetch_type="all",
                )

                actual_indices = [idx["indexname"] for idx in indices]
                missing_indices = set(expected_indices) - set(actual_indices)

                if missing_indices:
                    # Create missing indices
                    for idx in missing_indices:
                        if idx == "idx_conversation_messages_session_id":
                            await self.execute_query(
                                """
                                CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_id 
                                ON conversation_messages(session_id)
                                """,
                                fetch_type="none",
                            )
                        elif idx == "idx_conversation_messages_order":
                            await self.execute_query(
                                """
                                CREATE INDEX IF NOT EXISTS idx_conversation_messages_order 
                                ON conversation_messages(session_id, message_order)
                                """,
                                fetch_type="none",
                            )
                    logger.info(f"Database: Created missing indices: {missing_indices}")

                logger.info("Database: Validated conversation_messages table structure")
                return True

        except Exception as e:
            logger.error(f"Database: Error ensuring schema: {e}")
            return False


async def fetch_db_url(config_id: str = "DATABASE") -> Optional[str]:
    """
    Fetch the database URL using ConnectorConfigManager.

    This is a convenience function that can be used by other components to get
    the database URL without creating a Database instance.

    Args:
        config_id: The ConfigManager config_id for the database connector

    Returns:
        Database URL or None if not available
    """
    connector_manager = ConnectorConfigManager()
    await connector_manager.initialize(register_connectors=False)
    return await connector_manager.fetch_database_url(config_id)
