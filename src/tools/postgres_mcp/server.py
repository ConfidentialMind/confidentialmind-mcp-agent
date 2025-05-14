import asyncio
import datetime
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from confidentialmind_core.config_manager import load_environment
from fastmcp import Context, FastMCP

from .connection_manager import ConnectionManager
from .db import DatabaseClient, DatabaseUnavailableError, create_pool
from .settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Handles database pool setup and teardown with ConfidentialMind support.
    """
    pool = None
    state = {}

    try:
        # Initialize environment
        load_environment()

        # Initialize ConnectionManager (this will handle all connector registration and URL polling)
        try:
            # Run with timeout protection
            await asyncio.wait_for(ConnectionManager.initialize(), timeout=5.0)
            logger.info("Database connection manager initialized")
        except asyncio.TimeoutError:
            logger.warning(
                "ConnectionManager initialization timed out, continuing without database connection"
            )
        except Exception as e:
            logger.error(f"Error initializing connection manager: {e}")

        # Create pool, handling failures appropriately
        try:
            # Non-blocking pool creation attempt
            pool = await asyncio.wait_for(create_pool(), timeout=3.0)
            if pool:
                state["db_pool"] = pool
                logger.info("Database pool created and stored in application state")
            else:
                logger.info("No database connection available yet. Server will start anyway.")
                state["db_pool"] = None
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Starting without database connection: {e}")
            state["db_pool"] = None

        # Quickly yield control back to FastMCP
        yield state

    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        logger.info(f"Server startup interrupted: {type(e).__name__}")
        raise
    finally:
        if pool:
            logger.info("Shutting down server, closing database pool")
            try:
                await ConnectionManager.close()
                logger.info("Database pool closed")
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
        raise RuntimeError("Database connection is not available yet. Please try again later.")

    client = DatabaseClient(pool)
    try:
        return await client.get_table_schemas()
    except DatabaseUnavailableError:
        raise RuntimeError("Database connection is not available yet. Please try again later.")
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
        raise RuntimeError("Database connection is not available yet. Please try again later.")

    client = DatabaseClient(pool)
    try:
        results = await client.execute_readonly_query(sql_query)
        await ctx.info(f"Executed query successfully, returned {len(results)} rows.")
        return results
    except DatabaseUnavailableError:
        raise RuntimeError("Database connection is not available yet. Please try again later.")
    except ValueError as e:
        logger.warning(f"SQL query validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to execute SQL query: {e}")
        raise RuntimeError(f"Could not execute SQL query: {e}")


@mcp_server.tool()
async def health_check(ctx: Context) -> dict:
    """
    Returns the health status of the server and database connection.
    This can be used to check if the server is functioning correctly.

    Returns:
        A dictionary with health status information.
    """
    pool = ctx.request_request.lifespan_context.get("db_pool")
    db_client = DatabaseClient(pool) if pool else None

    is_db_connected = False
    if db_client:
        is_db_connected = await db_client.check_health()

    return {
        "status": "healthy",
        "database_connected": is_db_connected,
        "server_mode": "stack_deployment" if settings.is_stack_deployment else "local",
        "server_time": datetime.datetime.now().isoformat(),
        "connector_id": settings.connector_id,
    }
