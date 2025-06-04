import asyncio
import datetime
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from confidentialmind_core.config_manager import load_environment
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from langfuse.decorators import langfuse_context, observe
from starlette.requests import Request
from starlette.responses import JSONResponse

from .connection_manager import ConnectionManager
from .db import (
    DatabaseClient,
    DatabaseUnavailableError,
    QueryValidationError,
    create_pool,
)
from .settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Handles database pool setup and teardown with ConfidentialMind support.

    This function allows the server to start without a database connection,
    which is required for stack deployment scenarios where the database
    might be connected after the server is running.
    """
    pool = None
    state = {"db_pool": None}

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
            pool = await asyncio.wait_for(create_pool(), timeout=10.0)
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
@observe()
async def get_schemas() -> dict[str, list[dict]]:
    """
    Retrieves the schemas (table names, columns, types) for all tables
    accessible by the connected database user. Excludes system tables.
    """
    # Update observation metadata
    langfuse_context.update_current_observation(
        name="Postgres Get Schemas",
        metadata={"tool": "postgres_mcp", "operation": "get_schemas"},
    )
    # Use direct pool access instead of context
    pool = ConnectionManager.get_pool()

    if not pool:
        logger.error("Database pool not available.")
        raise RuntimeError("Database connection is not available yet. Please try again later.")

    client = DatabaseClient(pool)
    try:
        schemas = await client.get_table_schemas()

        # Update observation with results
        langfuse_context.update_current_observation(
            output=schemas,
            metadata={"tables_count": len(schemas.get("schemas", []))},
        )

        return schemas
    except DatabaseUnavailableError:
        raise RuntimeError("Database connection is not available yet. Please try again later.")
    except Exception as e:
        logger.error(f"Failed to get schemas: {e}")
        raise RuntimeError(f"Could not retrieve database schemas: {e}")


@mcp_server.tool()
@observe()
async def execute_sql(sql_query: str) -> list[dict]:
    """
    Executes a read-only SQL query (must start with SELECT, WITH, or EXPLAIN
    and contain no modification keywords like INSERT, UPDATE, DELETE, CREATE, etc.)
    and returns the results. Use the 'postgres://schemas' resource to see
    available tables and columns first.

    Args:
        sql_query: The read-only SQL query string to execute.

    Returns:
        A list of dictionaries, where each dictionary represents a row
        with column names as keys.
    """
    # Update observation metadata
    langfuse_context.update_current_observation(
        name="Postgres Execute SQL",
        input={"sql_query": sql_query},
        metadata={"tool": "postgres_mcp", "operation": "execute_sql"},
    )
    # Use direct pool access instead of context
    pool = ConnectionManager.get_pool()

    if not pool:
        logger.error("Database pool not available for execute_sql.")
        raise RuntimeError("Database connection is not available yet. Please try again later.")

    client = DatabaseClient(pool)
    try:
        results = await client.execute_readonly_query(sql_query)
        logger.info(f"Executed query successfully, returned {len(results)} rows.")

        # Update observation with results
        langfuse_context.update_current_observation(
            output=results,
            metadata={"rows_count": len(results)},
        )

        return results
    except DatabaseUnavailableError:
        raise RuntimeError("Database connection is not available yet. Please try again later.")
    except QueryValidationError as e:
        logger.warning(f"SQL query validation failed: {e}")
        langfuse_context.update_current_observation(
            level="WARNING",
            status_message=f"Query validation failed: {str(e)}",
        )
        raise ToolError(str(e))
    except Exception as e:
        logger.error(f"Failed to execute SQL query: {e}")
        raise RuntimeError(f"Could not execute SQL query: {e}")


@mcp_server.custom_route("/health", methods=["GET"])
async def http_health_endpoint(request: Request) -> JSONResponse:
    """
    HTTP health endpoint for Kubernetes liveness and readiness probes.

    This endpoint always returns a 200 status code when the server is running,
    regardless of database connection status, allowing the pod to stay alive
    while waiting for the database to be configured.

    For detailed health status, it provides the same information as the
    health_check MCP tool.

    Returns:
        JSONResponse with health status information.
    """
    pool = ConnectionManager.get_pool()
    db_client = DatabaseClient(pool) if pool else None

    is_db_connected = False
    db_error = None

    if db_client:
        try:
            is_db_connected = await db_client.check_health()
        except Exception as e:
            logger.error(f"Error checking database health: {e}")
            db_error = str(e)
    else:
        db_error = ConnectionManager.last_error() or "Database not configured"

    # Create health status response
    health_status = {
        "status": "healthy",
        "service": "postgres-mcp-server",
        "database_connected": is_db_connected,
        "database_error": db_error if not is_db_connected else None,
        "server_mode": "stack_deployment" if settings.is_stack_deployment else "local",
        "server_time": datetime.datetime.now().isoformat(),
        "connector_id": settings.connector_id,
    }

    # Always return a 200 status code to keep the pod alive
    return JSONResponse(content=health_status)
