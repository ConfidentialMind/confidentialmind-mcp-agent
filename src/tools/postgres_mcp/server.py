import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastmcp import Context, FastMCP

from .db import DatabaseClient, create_pool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Handles database pool setup and teardown.

    This context manager ensures proper creation and cleanup of the database
    connection pool, even when the server is interrupted with Ctrl+C.
    """
    pool = None
    state = {}

    try:
        # Create the pool
        pool = await create_pool()
        state["db_pool"] = pool
        logger.info("Database pool created and stored in application state.")
        yield state
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        # Handle cancellation specifically
        logger.info(f"Server startup interrupted: {type(e).__name__}")
        raise  # Re-raise to allow proper shutdown
    except Exception as e:
        logger.critical(f"Database connection failed on startup: {e}")
        yield state
    finally:
        # Always try to close the pool in the finally block
        if pool:
            logger.info("Shutting down server, closing database connection pool.")
            try:
                await pool.close()
                logger.info("Database connection pool closed successfully.")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}")


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
        raise RuntimeError("Database connection is not available.")

    client = DatabaseClient(pool)
    try:
        return await client.get_table_schemas()
    except Exception as e:
        logger.error(f"Failed to get schemas: {e}")
        raise RuntimeError(f"Could not retrieve database schemas: {e}")


@mcp_server.tool()
async def execute_sql(sql_query: str, ctx: Context) -> list[dict]:
    """
    Executes a read-only SQL query (must start with SELECT, WITH, or EXPLAIN
    and contain no modification keywords like INSERT, UPDATE, DELETE, CREATE, etc.)
    and returns the results. Use the 'postgres://schemas' resource to see
    available tables and columns first.

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
        raise RuntimeError("Database connection is not available.")

    client = DatabaseClient(pool)
    try:
        results = await client.execute_readonly_query(sql_query)
        await ctx.info(f"Executed query successfully, returned {len(results)} rows.")
        return results
    except ValueError as e:
        logger.warning(f"SQL query validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to execute SQL query: {e}")
        raise RuntimeError(f"Could not execute SQL query: {e}")
