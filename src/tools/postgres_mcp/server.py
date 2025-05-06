import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg
from fastmcp import Context, FastMCP

from .db import create_pool  # Import create_pool directly instead of db_connection_pool
from .db import (
    execute_readonly_query,
    get_table_schemas,
)

logger = logging.getLogger(__name__)


# Define the lifespan manager for the database pool
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Handles database pool setup and teardown."""
    pool = None
    state = {}
    try:
        # Create the pool directly without using another context manager
        pool = await create_pool()
        state["db_pool"] = pool
        logger.info("Database pool created and stored in application state.")
        yield state  # Provide the pool to the application state
    except ConnectionError as e:
        logger.critical(f"Database connection failed on startup: {e}")
        yield state  # Or yield empty state to allow server start but tools will fail
    finally:
        if pool:
            logger.info("Shutting down server, closing database connection pool.")
            await pool.close()


# Instantiate the FastMCP server with the lifespan manager
mcp_server = FastMCP(
    name="Postgres Reader MCP",
    instructions="Provides read-only access to a PostgreSQL database.",
    lifespan=lifespan,
)


@mcp_server.resource("postgres://schemas")
async def get_schemas(ctx: Context) -> dict[str, list[dict]]:
    """
    Retrieves the schemas (table names, columns, types) for all tables
    accessible by the connected database user. Excludes system tables.
    """
    pool = ctx.request_context.lifespan_context.get("db_pool")
    if not pool:
        logger.error("Database pool not available in context.")
        raise ConnectionError("Database connection is not available.")
    try:
        return await get_table_schemas(pool)
    except Exception as e:
        logger.error(f"Failed to get schemas: {e}")
        # Re-raise as a standard exception for MCP error handling
        raise RuntimeError(f"Could not retrieve database schemas: {e}")


@mcp_server.tool()
async def execute_sql(sql_query: str, ctx: Context) -> list[dict]:
    """
    Executes a read-only SQL query (must start with SELECT, WITH, or EXPLAIN
    and contain no modification keywords like INSERT, UPDATE, DELETE, CREATE, etc.)
    and returns the results. Use the 'postgres://schemas' resource to see
    available tables and columns first.

    WARNING: Only executes queries verified as read-only. Complex queries or
    those calling functions with side effects might be blocked or behave unexpectedly.

    Args:
        sql_query: The read-only SQL query string to execute.
        ctx: The MCP context (automatically injected).

    Returns:
        A list of dictionaries, where each dictionary represents a row
        with column names as keys.
    """
    pool = ctx.request_context.lifespan_context.get("db_pool")
    if not pool:
        logger.error("Database pool not available in context for execute_sql.")
        raise ConnectionError("Database connection is not available.")
    try:
        # The actual execution and read-only check happens in db.py
        results = await execute_readonly_query(pool, sql_query)
        await ctx.info(f"Executed query successfully, returned {len(results)} rows.")
        return results
    except ValueError as e:  # Catch specific validation errors from db.py
        logger.warning(f"SQL query validation failed: {e}")
        # Re-raise as ValueError, MCP will handle it as a tool error
        raise
    except Exception as e:
        logger.error(f"Failed to execute SQL query: {e}")
        # Re-raise as a standard exception for MCP error handling
        raise RuntimeError(f"Could not execute SQL query: {e}")
