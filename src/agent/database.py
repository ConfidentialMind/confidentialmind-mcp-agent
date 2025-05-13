import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import asyncpg
import backoff
from confidentialmind_core.config_manager import ConfigManager, get_api_parameters, load_environment
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
            f"Reconnection attempt {details['tries']} failed. Retrying in {details['wait']} seconds..."
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
            # Stack deployment mode - use the full DSN from the stack without modifications
            # Note: db_url in this context is expected to be the full DSN
            if not db_url:
                raise ValueError("No database URL provided in stack deployment mode")

            dsn = f"postgresql://{self.database_user}:{self.database_password}@{db_url}/{self.database_name}"
            logger.info("Using the complete DSN from the stack configuration")

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

        # Load environment variables to set LOCAL_CONFIG flag
        load_environment()
        self._is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"Database initialized in {'stack deployment' if self._is_stack_deployment else 'local config'} mode"
        )

    async def fetch_db_url(self) -> Optional[str]:
        """
        Fetch database URL with infinite polling if in stack deployment mode.
        Similar to the baserag implementation pattern.

        Returns:
            Database URL from the stack or None if in local mode with no URL available
        """
        if not self._is_stack_deployment:
            # In local mode, just get the URL from environment once
            url, _ = get_api_parameters(self._config_id)
            if url:
                logger.info(f"Retrieved database URL from environment: {url}")
            else:
                logger.warning(f"No database URL found in environment for {self._config_id}")
            return url

        # In stack deployment mode, poll until URL is available
        logger.info(f"Waiting for {self._config_id} URL from stack...")
        while True:
            try:
                url, _ = get_api_parameters(self._config_id)
                if url:
                    logger.info(f"Successfully retrieved {self._config_id} URL from stack")

                    # If Confidentialmind SDK provides a full connection string via config manager,
                    # we get it here; otherwise, we get the base URL to build upon
                    if self._is_stack_deployment:
                        # In stack mode, try to get the full connection string from ConfigManager
                        try:
                            config_manager = ConfigManager()
                            full_url = config_manager.config.get_connection_string(url)
                            if full_url:
                                logger.info("Got full connection string from ConfigManager")
                                return full_url
                        except (AttributeError, Exception) as e:
                            logger.warning(
                                f"Could not get connection string from ConfigManager: {e}"
                            )
                            # Fall back to using the URL directly if config manager doesn't work

                    return url
            except Exception as e:
                logger.debug(f"Error fetching {self._config_id} URL, waiting: {e}")

            await asyncio.sleep(5)  # Wait before retrying

    async def connect(self, db_url: Optional[str] = None) -> bool:
        """Establish connection to database with proper locking and stack integration"""
        if self._is_connecting:
            logger.info("Connection attempt already in progress")
            return self.is_connected()

        if self._pool is not None:
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

            if not self._current_db_url and self._is_stack_deployment:
                logger.error("No database URL available in stack deployment mode")
                return False

            # Build connection string and connect
            connection_string = self.settings.get_connection_string(self._current_db_url)

            logger.debug(
                f"Connecting to database with connection string: {self.settings._get_safe_connection_string(connection_string)}"
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

            logger.info("Successfully connected to database")
            self._connection_error = None
            return True

        except Exception as e:
            self._connection_error = f"Failed to connect to database: {str(e)}"
            logger.error(self._connection_error)
            self._pool = None
            self._schedule_reconnect()
            return False
        finally:
            self._is_connecting = False

    async def disconnect(self):
        """Disconnect from database"""
        if self._pool:
            logger.info("Disconnecting from database")
            if self._reconnect_task and not self._reconnect_task.done():
                self._reconnect_task.cancel()
            await self._pool.close()
            self._pool = None
            logger.info("Successfully disconnected database connections")

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
        await self.connect(self._current_db_url)

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
                logger.error(f"Error executing query: {e}")
                raise

    # Session history methods

    async def load_history(self, session_id: str) -> List[Message]:
        """Load conversation history from the database for a session."""
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
            logger.debug(f"Loaded {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Error loading history for session {session_id}: {e}")
            return []

    async def save_message(self, session_id: str, message: Message) -> bool:
        """Save a message to the database for a session."""
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
            logger.debug(f"Saved message for session {session_id} order {message_order}")
            return True
        except Exception as e:
            if not self.is_connected():
                logger.warning(f"DB not connected saving msg for {session_id}: {self.last_error()}")
            else:
                logger.error(f"Error saving message for session {session_id}: {e}")
            return False

    async def clear_history(self, session_id: str) -> bool:
        """Clear the conversation history for a session in the database."""
        try:
            await self.ensure_connected()
            await self.execute_query(
                "DELETE FROM conversation_messages WHERE session_id = $1",
                session_id,
                fetch_type="none",
            )
            logger.info(f"Cleared conversation history for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing history for session {session_id}: {e}")
            return False

    async def ensure_schema(self) -> bool:
        """Ensure necessary database tables exist"""
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
                logger.info("Created conversation_messages table and indices")
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
                        f"Missing columns in conversation_messages table: {missing_columns}"
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
                                f"Column {col_name} has incorrect properties. Expected: {expected_props}, Actual: {actual_props}"
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
                    logger.info(f"Created missing indices: {missing_indices}")

                logger.info("Validated conversation_messages table structure")
                return True

        except Exception as e:
            logger.error(f"Error ensuring schema: {e}")
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
