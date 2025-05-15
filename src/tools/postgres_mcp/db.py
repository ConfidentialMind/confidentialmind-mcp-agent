import logging
import re
from typing import Any, Dict, List, Optional

import asyncpg

from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

# SQL validation regex patterns
READONLY_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|COMMENT|SET)\b",
    re.IGNORECASE | re.MULTILINE,
)
ALLOWED_START_KEYWORDS = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE | re.MULTILINE)


class DatabaseError(Exception):
    """Base exception for database-related errors."""

    pass


class QueryValidationError(ValueError):
    """Exception raised when a query fails validation."""

    pass


class DatabaseUnavailableError(DatabaseError):
    """Exception raised when database connection is not available."""

    pass


async def create_pool() -> Optional[asyncpg.Pool]:
    """
    Creates an asyncpg connection pool with ConfidentialMind support.

    This function will not raise an exception if the database is not
    available, allowing the server to start without a database connection.

    Returns:
        asyncpg.Pool or None if connection couldn't be established
    """
    await ConnectionManager.initialize()
    try:
        return await ConnectionManager.create_pool()
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        return None


class DatabaseClient:
    """Client for executing read-only PostgreSQL operations."""

    def __init__(self, pool: Optional[asyncpg.Pool]):
        """Initialize with a connection pool."""
        self.pool = pool

    async def get_table_schemas(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves schema information for tables accessible by the current user.

        Raises:
            DatabaseUnavailableError: If database connection is not available
            DatabaseError: For other database-related errors
        """
        if self.pool is None:
            raise DatabaseUnavailableError("Database connection is not available yet")

        schemas: Dict[str, List[Dict[str, Any]]] = {}

        try:
            async with self.pool.acquire() as conn:
                # Query to get table and column information for standard schemas
                rows = await conn.fetch(
                    """
                    SELECT
                        table_schema,
                        table_name,
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_schema, table_name, ordinal_position;
                    """
                )

                for row in rows:
                    table_key = f"{row['table_schema']}.{row['table_name']}"
                    if table_key not in schemas:
                        schemas[table_key] = []

                    schemas[table_key].append(
                        {
                            "column": row["column_name"],
                            "type": row["data_type"],
                            "nullable": row["is_nullable"] == "YES",
                            "default": row["column_default"],
                        }
                    )

            logger.info(f"Fetched schemas for {len(schemas)} tables.")
            return schemas
        except Exception as e:
            logger.error(f"Error fetching table schemas: {e}")
            raise DatabaseError(f"Failed to fetch schemas: {e}")

    async def execute_readonly_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        Executes a SQL query after validating it's likely read-only.

        Args:
            sql_query: The SQL query to execute

        Returns:
            A list of dictionaries representing the result rows

        Raises:
            QueryValidationError: If the query fails validation
            DatabaseError: For other database-related errors
            DatabaseUnavailableError: If database connection is not available
        """
        if self.pool is None:
            raise DatabaseUnavailableError("Database connection is not available yet")

        # Validate query starts with allowed keywords
        if not ALLOWED_START_KEYWORDS.match(sql_query):
            raise QueryValidationError("Query must start with SELECT, WITH, or EXPLAIN.")

        # Check for disallowed keywords
        if READONLY_KEYWORDS.search(sql_query):
            raise QueryValidationError(
                "Query contains disallowed keywords (potential write operation)."
            )

        logger.debug(f"Executing read-only query: {sql_query[:100]}...")

        try:
            async with self.pool.acquire() as conn:
                # Set a statement timeout for safety
                await conn.execute("SET statement_timeout = 5000")  # 5 seconds
                results = await conn.fetch(sql_query)
                # Convert asyncpg Record objects to dictionaries
                return [dict(row) for row in results]
        except asyncpg.PostgresError as e:
            logger.error(f"Database error executing query: {e}")
            # Provide a more user-friendly error message if possible
            raise DatabaseError(f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            raise DatabaseError(f"Error executing query: {str(e)}")

    async def check_health(self) -> bool:
        """Check database health by executing a simple query."""
        if self.pool is None:
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
