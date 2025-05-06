import logging
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from .settings import settings

logger = logging.getLogger(__name__)

# Basic Regex to check for disallowed SQL keywords (case-insensitive)
# WARNING: This is NOT foolproof security. A dedicated SQL parser or connection
# with read-only permissions is strongly recommended for production use.
READONLY_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|COMMENT|SET)\b",
    re.IGNORECASE | re.MULTILINE,
)
ALLOWED_START_KEYWORDS = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE | re.MULTILINE)


async def create_pool() -> asyncpg.Pool:
    """Creates an asyncpg connection pool."""
    logger.info(f"Creating database connection pool for {settings.database}...")
    try:
        pool = await asyncpg.create_pool(dsn=settings.effective_dsn, min_size=1, max_size=5)
        if pool:
            logger.info("Database connection pool created successfully.")
            return pool
        else:
            raise ConnectionError("Failed to create database pool (returned None).")
    except Exception as e:
        logger.error(f"Error creating database connection pool: {e}", exc_info=True)
        raise ConnectionError(f"Failed to connect to database: {e}")


@asynccontextmanager
async def db_connection_pool() -> AsyncIterator[asyncpg.Pool]:
    """Provides a database connection pool within an async context."""
    pool = None
    try:
        pool = await create_pool()
        yield pool
    finally:
        if pool:
            logger.info("Closing database connection pool.")
            await pool.close()


async def get_table_schemas(pool: asyncpg.Pool) -> dict[str, list[dict]]:
    """Retrieves schema information for tables accessible by the current user."""
    schemas: dict[str, list[dict]] = {}
    try:
        async with pool.acquire() as conn:
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
        logger.error(f"Error fetching table schemas: {e}", exc_info=True)
        raise


async def execute_readonly_query(pool: asyncpg.Pool, sql_query: str) -> list[dict]:
    """
    Executes a SQL query after validating it's likely read-only.

    WARNING: Basic validation only. Use with caution.
    """
    # 1. Validate query starts with allowed keywords
    if not ALLOWED_START_KEYWORDS.match(sql_query):
        raise ValueError("Query must start with SELECT, WITH, or EXPLAIN.")

    # 2. Check for disallowed keywords
    if READONLY_KEYWORDS.search(sql_query):
        raise ValueError("Query contains disallowed keywords (potential write operation).")

    logger.debug(f"Executing read-only query: {sql_query[:100]}...")
    try:
        async with pool.acquire() as conn:
            # Consider setting a statement timeout for safety
            # await conn.set_statement_timeout(5000) # 5 seconds
            results = await conn.fetch(sql_query)
            # Convert asyncpg Record objects to dictionaries
            return [dict(row) for row in results]
    except asyncpg.PostgresError as e:
        logger.error(f"Database error executing query: {e}", exc_info=False)
        # Provide a more user-friendly error message if possible
        raise ValueError(f"Database error: {e.message or e}")
    except Exception as e:
        logger.error(f"Unexpected error executing query: {e}", exc_info=True)
        raise
