import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from confidentialmind_core.config_manager import ConfigManager, ConnectorSchema
from confidentialmind_core.config_manager import config as cm_config
from confidentialmind_core.config_manager import load_environment
from fastmcp import Context, FastMCP
from pydantic import BaseModel

from .db import DatabaseClient, create_pool

logger = logging.getLogger(__name__)

# Load environment variables to initialize LOCAL_DEV and LOCAL_CONFIGS flags
load_environment()


class PostgresConfig(BaseModel):
    """Minimal config for the Postgres MCP server."""

    name: str = "postgres-mcp-server"
    description: str = "Read-only PostgreSQL MCP Server"


# Initialize ConfigManager if we're in stack deployment mode
if not cm_config.LOCAL_CONFIGS:
    logger.info("Initializing ConfigManager for stack deployment mode")
    config_manager = ConfigManager()
    # Define the connector for the database
    connectors = [
        ConnectorSchema(type="database", label="PostgreSQL Database", config_id="DATABASE")
    ]
    # Initialize the ConfigManager
    config_manager.init_manager(
        config_model=PostgresConfig(),  # No config model needed for this server
        connectors=connectors,
    )
    logger.info("ConfigManager initialized with database connector")
else:
    logger.info("Running in local mode, skipping ConfigManager initialization")


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Handles database pool setup and teardown.

    This context manager ensures proper creation and cleanup of the database
    connection pool, even when the server is interrupted with Ctrl+C.
    """
    pool = None
    state = {}
    retry_task = None

    async def attempt_pool_creation():
        """Try to create a connection pool with retries."""
        nonlocal pool, state
        retry_count = 0
        max_retries = 10
        retry_delay = 5  # seconds

        while retry_count < max_retries and pool is None:
            if retry_count > 0:
                logger.info(f"Retry {retry_count}/{max_retries} to create database pool")
                await asyncio.sleep(retry_delay)

            pool = await create_pool()
            if pool:
                state["db_pool"] = pool
                logger.info("Database pool created and stored in application state")
                return

            retry_count += 1

        if pool is None:
            logger.warning(f"Failed to create database pool after {max_retries} attempts")

    try:
        # Initial attempt to create the pool
        pool = await create_pool()
        if pool:
            state["db_pool"] = pool
            logger.info("Database pool created and stored in application state")
        else:
            # If initial creation fails, start a background task for retries
            logger.warning("Initial database connection failed. Starting background retry task")
            if not cm_config.LOCAL_CONFIGS:
                # Only in stack mode we retry in the background
                retry_task = asyncio.create_task(attempt_pool_creation())

        # Yield state even if pool is None to allow the server to start
        yield state

        # Cancel retry task if it's still running
        if retry_task and not retry_task.done():
            retry_task.cancel()
            try:
                await retry_task
            except asyncio.CancelledError:
                logger.info("Database retry task cancelled")

    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        logger.info(f"Server startup interrupted: {type(e).__name__}")
        raise  # Re-raise to allow proper shutdown
    except Exception as e:
        logger.critical(f"Error during startup: {e}")
        yield state
    finally:
        # Always try to close the pool in the finally block
        if pool:
            logger.info("Shutting down server, closing database connection pool")
            try:
                await pool.close()
                logger.info("Database connection pool closed successfully")
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
        logger.warning("Database pool not available in context")
        await ctx.info("Database connection is not available yet. Please try again later.")
        return {}  # Return empty schemas if no pool is available

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
        logger.error("Database pool not available in context for execute_sql")
        await ctx.info("Database connection is not available yet. Please try again later.")
        raise RuntimeError("Database connection is not available.")

    client = DatabaseClient(pool)
    try:
        results = await client.execute_readonly_query(sql_query)
        await ctx.info(f"Executed query successfully, returned {len(results)} rows.")
        return results
    except ValueError as e:
        logger.warning(f"SQL query validation failed: {e}")
        await ctx.info(f"Query validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to execute SQL query: {e}")
        await ctx.info(f"Error executing query: {e}")
        raise RuntimeError(f"Could not execute SQL query: {e}")


@mcp_server.tool()
async def health_check(ctx: Context) -> dict:
    """
    Provides information about the health of the MCP server and database connection.

    Returns:
        A dictionary containing health information.
    """
    pool = ctx.request_context.lifespan_context.get("db_pool")

    health_info = {
        "status": "healthy",
        "database_connected": pool is not None,
        "deployment_mode": "stack" if not cm_config.LOCAL_CONFIGS else "local",
    }

    await ctx.info(f"Health status: {health_info}")
    return health_info
