import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from confidentialmind_core.config_manager import ConfigManager, ConnectorSchema, load_environment
from fastmcp import Context, FastMCP

from .connection_manager import ConnectionManager
from .db import DatabaseClient, create_pool
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
        # Initialize ConfidentialMind integration if enabled
        if settings.use_sdk_connector:
            load_environment()
            logger.info("Initializing ConfigManager...")

            try:
                config_manager = ConfigManager()
                connectors = [
                    ConnectorSchema(
                        type="database",
                        label="PostgreSQL Database",
                        config_id=settings.connector_id,
                    )
                ]
                config_manager.init_manager(connectors=connectors)
                logger.info("ConfigManager initialized successfully")
            except Exception as e:
                logger.warning(f"ConfigManager initialization warning: {e}")
                logger.info("Continuing with fallback settings")

        # Create the pool using our ConnectionManager
        pool = await create_pool()
        state["db_pool"] = pool
        logger.info("Database pool created and stored in application state")
        yield state

    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        logger.info(f"Server startup interrupted: {type(e).__name__}")
        raise
    except Exception as e:
        logger.critical(f"Database connection failed on startup: {e}")
        yield state
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
