"""
PostgreSQL MCP Server

A Model Context Protocol server for PostgreSQL database access.
Provides tools for querying and schema exploration.
"""

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, cast
from urllib.parse import ParseResult, urlparse, urlunparse

import asyncpg
import backoff
from confidentialmind_core.config_manager import get_api_parameters, load_environment
from pydantic import Field
from pydantic_settings import BaseSettings

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [PostgreSQL MCP Server] %(levelname)s: %(message)s"
)
logger = logging.getLogger("postgres-mcp-server")

# --- Constants ---
DATABASE_CONNECTOR_ID = "DATABASE"  # Identifier for the database connector
DEFAULT_READ_TIMEOUT = 60.0  # Default query timeout in seconds

# --- Environment Settings ---
LOCAL_DEV = os.environ.get("LOCAL_DEV", "").lower() in ("true", "1", "yes")
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")

# Set log level based on debug mode
if DEBUG_MODE:
    logging.getLogger("postgres-mcp-server").setLevel(logging.DEBUG)


# --- Backoff Configuration ---
def get_backoff_config() -> Dict[str, Any]:
    """Get backoff configuration based on environment"""
    return {
        "max_tries": 5,
        "max_time": 30,
        "on_backoff": lambda details: logger.info(
            f"DB Reconnection attempt {details['tries']} failed. "
            f"Retrying in {details['wait']} seconds"
        ),
    }


# --- Pydantic Settings for Database Connection ---
class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings with fallback defaults."""

    database_host: str = "localhost"
    database_port: int = 5432
    database_user: str = "app"
    database_password: str = "testpass"
    database_name: str = "vector-db"
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 30.0
    command_timeout: float = DEFAULT_READ_TIMEOUT
    sdk_db_url: Optional[str] = None  # Store the URL fetched during startup

    def get_connection_string(self) -> str:
        """Generate PostgreSQL connection string using stored SDK URL or defaults."""
        host_part = ""
        if self.sdk_db_url:
            # Use the stored SDK-provided URL (likely host or service name)
            host_part = self.sdk_db_url
            logger.debug(f"Using stored SDK DB host/endpoint: {host_part}")
        else:
            # Fallback to default host:port if no SDK URL was stored
            host_part = f"{self.database_host}:{self.database_port}"
            logger.debug(f"Using default DB host/port settings: {host_part}")

        dsn = f"postgresql://{self.database_user}:{self.database_password}@{host_part}/{self.database_name}"
        logger.debug(f"Constructed DSN: {self._get_safe_connection_string(dsn)}")
        return dsn

    def _get_safe_connection_string(self, connection_string: Optional[str]) -> str:
        """Return a connection string with password masked for logging."""
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


# --- Database Connection Management ---
@dataclass
class Database:
    """Database connection pool wrapper with query execution methods."""

    pool: asyncpg.pool.Pool

    async def execute_query(
        self, query: str, *args, fetch_type: Literal["all", "row", "val", "none"] = "all"
    ) -> Any:
        """Execute a database query using the pool.

        Args:
            query: SQL query to execute
            *args: Query parameters
            fetch_type: Type of fetch operation (all, row, val, none)

        Returns:
            Query results based on fetch_type
        """
        async with self.pool.acquire() as conn:
            try:
                # Allow setting statement_timeout for individual queries
                # await conn.execute("SET statement_timeout = '30s'")

                if fetch_type == "all":
                    return await conn.fetch(query, *args)
                elif fetch_type == "row":
                    return await conn.fetchrow(query, *args)
                elif fetch_type == "val":
                    return await conn.fetchval(query, *args)
                else:  # 'none' or execute
                    return await conn.execute(query, *args)
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                raise  # Re-raise to be handled by the caller


# --- Lifespan Context Type ---
@dataclass
class AppContext:
    """Context passed to resource/tool handlers via lifespan."""

    db: Database
    settings: PostgresSettings
    database_name: str
    resource_base_url_parts: Optional[ParseResult] = None


# --- Lifespan Management ---
@asynccontextmanager
async def app_lifespan(app: FastMCP):
    """MCP Server lifespan context manager.

    Manages database connection lifecycle and resource registration.
    """
    logger.info("Lifespan startup: Initializing PostgreSQL MCP Server...")
    db_pool = None
    settings = None
    db_url = None
    db_instance = None
    resource_base_url_parts = None
    database_name = ""

    try:
        # 1. Load environment and settings
        load_environment()
        settings = PostgresSettings()
        logger.info("PostgresSettings loaded.")

        # 2. Fetch Database URL from ConfigManager (only during startup)
        try:
            db_url, _ = get_api_parameters(DATABASE_CONNECTOR_ID)
            settings.sdk_db_url = db_url  # Store the fetched URL
            logger.info(f"Fetched initial DB URL from SDK: {db_url}")
        except Exception as e:
            logger.warning(
                f"Could not fetch initial DB URL from SDK: {e}. Will use defaults or retry."
            )
            settings.sdk_db_url = None  # Ensure it's None if fetch fails

        # 3. Establish Database Connection Pool with backoff
        @backoff.on_exception(
            backoff.expo,
            (asyncpg.PostgresConnectionError, OSError, ConnectionRefusedError),
            **get_backoff_config(),
        )
        async def connect_with_retry():
            nonlocal db_pool, db_instance, resource_base_url_parts, database_name
            connection_string = settings.get_connection_string()
            safe_connection_string = settings._get_safe_connection_string(connection_string)
            logger.info(f"Attempting to connect to database: {safe_connection_string}")

            db_pool = await asyncpg.create_pool(
                dsn=connection_string,
                min_size=settings.min_connections,
                max_size=settings.max_connections,
                timeout=settings.connection_timeout,
                command_timeout=settings.command_timeout,
                server_settings={
                    "statement_timeout": str(int(settings.command_timeout * 1000)),
                    "idle_in_transaction_session_timeout": "10000",
                    "lock_timeout": "2000",
                },
            )

            # Test connection and get database name
            async with db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
                database_name = await conn.fetchval("SELECT current_database()")

            db_instance = Database(pool=db_pool)

            # Parse connection string for building resource URIs
            resource_base_url_parts = urlparse(connection_string)
            logger.info(
                f"Successfully connected to database: {safe_connection_string} "
                f"(Database: {database_name})"
            )

        await connect_with_retry()  # Initial connection attempt with backoff

        # Yield context for the application
        if db_instance and settings:
            app_context = AppContext(
                db=db_instance,
                settings=settings,
                database_name=database_name,
                resource_base_url_parts=resource_base_url_parts,
            )
            # Register DB tables as resources during startup
            await register_db_resources(app, app_context)
            yield app_context
        else:
            # This should ideally not happen if backoff succeeds
            logger.error("Failed to establish database connection during startup after retries.")
            raise RuntimeError("Database connection failed during startup.")

    except Exception as e:
        logger.critical(f"Critical error during MCP server startup: {e}", exc_info=True)
        raise  # Prevent server from starting if essential setup fails
    finally:
        # --- Shutdown ---
        logger.info("Lifespan shutdown: Cleaning up PostgreSQL MCP Server resources...")
        if db_pool:
            await db_pool.close()
            logger.info("Database connection pool closed.")
        logger.info("Lifespan shutdown complete.")


# --- FastMCP Server Initialization ---
mcp = FastMCP("postgres-mcp", lifespan=app_lifespan)


# --- Register DB resources during startup ---
async def register_db_resources(app: FastMCP, context: AppContext) -> None:
    """Register database tables as resources during server startup."""
    logger.info("Registering database tables as resources...")
    try:
        db = context.db
        if not context.resource_base_url_parts:
            logger.warning("Cannot register resources: base URL parts missing.")
            return

        # Register the resource template for all tables
        @mcp.resource(
            "postgres://{schema_name}/{table_name}/schema",
            description="Database table schema information",
        )
        async def get_table_schema(schema_name: str, table_name: str) -> str:
            """Reads the schema information for a specific table."""
            ctx = mcp.get_context()

            logger.debug(f"Reading resource: postgres://{schema_name}/{table_name}/schema")
            db = ctx.request_context.lifespan_context.db
            database_name = ctx.request_context.lifespan_context.database_name

            # Basic validation
            if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
                raise McpError(
                    types.ErrorData(code=-32602, message=f"Invalid table name format: {table_name}")
                )
            if not re.match(r"^[a-zA-Z0-9_]+$", schema_name):
                raise McpError(
                    types.ErrorData(
                        code=-32602, message=f"Invalid schema name format: {schema_name}"
                    )
                )

            try:
                # First check if the table exists
                table_exists_query = """
                    SELECT EXISTS (
                        SELECT 1 
                        FROM pg_catalog.pg_class c
                        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relkind = 'r'
                        AND c.relname = $1
                        AND n.nspname = $2
                    ) AS exists
                """
                table_exists = await db.execute_query(
                    table_exists_query, table_name, schema_name, fetch_type="val"
                )

                if not table_exists:
                    raise McpError(
                        types.ErrorData(
                            code=-32002,
                            message=(
                                f"Resource not found: Table '{schema_name}.{table_name}' "
                                "does not exist."
                            ),
                        )
                    )

                # Get column information using a more compatible query
                columns_query = """
                    SELECT 
                        a.attname AS column_name,
                        pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
                        CASE WHEN a.attnotnull THEN 'NO' ELSE 'YES' END AS is_nullable,
                        pg_get_expr(d.adbin, d.adrelid) AS column_default
                    FROM 
                        pg_catalog.pg_attribute a
                    LEFT JOIN 
                        pg_catalog.pg_attrdef d ON (a.attrelid, a.attnum) = (d.adrelid, d.adnum)
                    JOIN 
                        pg_catalog.pg_class c ON a.attrelid = c.oid
                    JOIN 
                        pg_catalog.pg_namespace n ON c.relnamespace = n.oid
                    WHERE 
                        a.attnum > 0 
                        AND NOT a.attisdropped
                        AND c.relname = $1
                        AND n.nspname = $2
                    ORDER BY 
                        a.attnum
                """
                columns = await db.execute_query(columns_query, table_name, schema_name)

                if not columns:
                    # Table exists but no columns found (or permissions issue)
                    return json.dumps(
                        {
                            "message": (
                                f"Table '{schema_name}.{table_name}' exists but no column "
                                "info (check permissions?)."
                            )
                        }
                    )
                else:
                    columns_list = [dict(row) for row in columns]
                    return json.dumps(columns_list, indent=2, default=str)

            except asyncpg.PostgresError as pg_err:
                logger.error(
                    f"Postgres error reading schema for {schema_name}.{table_name}: {pg_err}"
                )
                raise McpError(types.ErrorData(code=-32000, message=f"Database error: {pg_err}"))
            except Exception as e:
                logger.error(
                    f"Unexpected error reading schema for {schema_name}.{table_name}: {e}",
                    exc_info=True,
                )
                raise McpError(types.ErrorData(code=-32000, message="Internal server error"))

        logger.info("Successfully registered database schema resource template.")

    except asyncpg.PostgresError as pg_err:
        logger.error(f"Postgres error registering resources: {pg_err}")
    except Exception as e:
        logger.error(f"Error registering resources: {e}", exc_info=True)


# --- MCP Tools ---
@mcp.tool(
    description="Run a read-only SQL query against the database.",
)
async def query(
    sql: str = Field(description="SQL query to execute (SELECT, EXPLAIN, SHOW, WITH only)"),
) -> str:
    """
    Executes a read-only SQL query against the PostgreSQL database.

    Only SELECT, EXPLAIN, SHOW, and WITH queries are allowed for security.
    Results are returned as JSON.
    """
    ctx = mcp.get_context()
    logger.debug(f"Executing tool 'query' with SQL: {sql[:100]}...")

    # Get database from context
    db = cast(Database, ctx.request_context.lifespan_context.db)
    settings = cast(PostgresSettings, ctx.request_context.lifespan_context.settings)

    # --- Security Checks ---
    sql_query = sql.strip()
    sql_upper = sql_query.upper()
    allowed_starts = ("SELECT", "EXPLAIN", "SHOW", "WITH")

    if not sql_upper.startswith(allowed_starts):
        raise McpError(
            types.ErrorData(
                code=-32602,
                message="Query error: Only SELECT, EXPLAIN, SHOW, WITH queries allowed.",
            )
        )

    # More robust check for disallowed keywords (case-insensitive, whole word)
    disallowed_keywords_pattern = r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|SET(?! statement_timeout)|COMMIT|ROLLBACK|SAVEPOINT|BEGIN|DECLARE|EXECUTE|COPY)\b"

    # Remove comments before checking
    sql_no_comments = re.sub(r"--.*$", "", sql_query, flags=re.MULTILINE)
    sql_no_comments = re.sub(r"/\*.*?\*/", "", sql_no_comments, flags=re.DOTALL)

    if re.search(disallowed_keywords_pattern, sql_no_comments, re.IGNORECASE):
        # Extract the found disallowed keyword(s) for a better error message
        found_disallowed = re.findall(disallowed_keywords_pattern, sql_no_comments, re.IGNORECASE)
        raise McpError(
            types.ErrorData(
                code=-32602,
                message=(
                    f"Query error: Query contains disallowed keywords: "
                    f"{', '.join(set(found_disallowed))}."
                ),
            )
        )
    # --- End Security Checks ---

    try:
        results = await db.execute_query(sql_query)

        # Handle different result types
        if isinstance(results, str) and results.startswith(
            tuple(kw + " " for kw in ["SELECT", "WITH", "EXPLAIN", "SHOW"])
        ):
            # Result might be a status message like "SELECT 0"
            result_json = json.dumps({"status": results})
        elif isinstance(results, list):
            # Convert list of records to JSON
            result_list = [dict(row) for row in results]
            result_json = json.dumps(result_list, indent=2, default=str)
            row_count = len(result_list)
            logger.info(
                f"Query successful, {row_count} row{'s' if row_count != 1 else ''} returned."
            )
        else:
            # Handle cases where execute might return row count or other status
            result_json = json.dumps(
                {"message": "Query executed successfully", "result": str(results)}
            )

        return result_json

    except asyncpg.PostgresError as pg_err:
        logger.error(f"SQL execution error: {pg_err}", exc_info=True)
        # Provide specific error message with hints
        pg_err_msg = str(pg_err).strip()
        hint = ""
        if "does not exist" in pg_err_msg:
            hint = " Hint: Check table/schema names and spelling, ensure schema qualification (schema.table)."
        elif "permission denied" in pg_err_msg:
            hint = " Hint: Check database user permissions."
        elif "timeout" in pg_err_msg:
            hint = f" Hint: Query exceeded the allowed time limit ({settings.command_timeout}s)."

        raise McpError(
            types.ErrorData(code=-32000, message=f"SQL Execution Error: {pg_err_msg}{hint}")
        )
    except Exception as e:
        logger.error(f"Unexpected error executing query tool: {e}", exc_info=True)
        raise McpError(types.ErrorData(code=-32000, message=f"Internal Server Error: {e}"))


# --- Main execution block ---
if __name__ == "__main__":
    # Configure logging level
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Choose transport based on environment
    transport = "stdio" if LOCAL_DEV else "sse"
    logger.info(f"Starting PostgreSQL MCP Server via {transport} (Debug Mode: {DEBUG_MODE})")

    try:
        # Run the server with selected transport
        mcp.run(transport=transport)
    except Exception as e:
        logger.critical(f"Server failed to start or exited unexpectedly: {e}", exc_info=True)
        import sys

        sys.exit(1)
