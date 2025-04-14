import asyncio
import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse, urlunparse

import backoff
import psycopg2
import psycopg2.extras
import psycopg2.pool
from confidentialmind_core.config_manager import (
    ConfigManager,
    ConnectorSchema,
    get_api_parameters,
)
from fastapi import FastAPI, HTTPException
from psycopg2 import sql
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Import MCP protocol definitions
from src.mcp.mcp_protocol import (
    CallToolRequestParams,
    CallToolResponse,
    JsonRpcResponse,
    ListResourcesResponse,
    ListToolsResponse,
    ReadResourceRequestParams,
    ReadResourceResponse,
    ResourceContent,
    ResourceIdentifier,
    TextContent,
    ToolDefinition,
    ToolInputSchema,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [PostgreSQL MCP Server] %(levelname)s: %(message)s"
)
logger = logging.getLogger("postgres-mcp")

# --- Constants ---
DATABASE_CONNECTOR_ID = "DATABASE"  # Assumed config_id for the database connector in ConfigManager
BACKGROUND_CONNECT_RETRY_SECONDS = 15  # How often to retry connection if down
BACKGROUND_CHECK_INTERVAL_SECONDS = 60  # How often to check if connection is still ok


# --- Backoff Configuration ---
def get_backoff_config() -> Dict[str, Any]:
    # Backoff for individual connection attempts within _connect_pool_with_backoff
    return {
        "wait_gen": backoff.expo,
        "exception": (psycopg2.Error, ConnectionError, TimeoutError),
        "max_tries": 5,  # Limit retries per attempt cycle
        "max_time": 60,  # Max time per attempt cycle
        "on_backoff": lambda details: logger.warning(
            f"DB connection attempt {details['tries']} failed within cycle. Retrying in {details['wait']:.1f}s..."
        ),
        "on_giveup": lambda details: logger.error(
            f"DB connection attempt failed after {details['tries']} tries within cycle."
        ),
    }


# --- Pydantic Settings ---
# (PostgresSettings class remains unchanged from the previous version)
class PostgresSettings(BaseSettings):
    # Connection Mode: 'local' or 'stack'
    POSTGRES_MCP_CONNECTION_MODE: str = "stack"
    # Local connection settings
    PG_LOCAL_HOST: str = "localhost"
    PG_LOCAL_PORT: int = 5432
    PG_LOCAL_USER: str = "postgres"
    PG_LOCAL_PASSWORD: str = "postgres"
    PG_LOCAL_DBNAME: str = "postgres"
    # Stack connection settings
    PG_CONNECTION_STRING: Optional[str] = None
    database_host: str = "localhost"
    database_port: int = 5432
    database_user: str = "app"
    database_password: str = "testpass"
    database_name: str = "vector-db"
    # Pool settings
    db_min_connections: int = 2
    db_max_connections: int = 10
    db_connection_timeout: float = 30.0
    db_command_timeout: float = 60.0

    def get_connection_string(self, db_url_from_sdk: Optional[str] = None) -> str:
        mode = self.POSTGRES_MCP_CONNECTION_MODE.lower()
        logger.debug(f"Determining connection string for mode: {mode}")
        if mode == "local":
            dsn = f"postgresql://{self.PG_LOCAL_USER}:{self.PG_LOCAL_PASSWORD}@{self.PG_LOCAL_HOST}:{self.PG_LOCAL_PORT}/{self.PG_LOCAL_DBNAME}"
            logger.info(
                f"Using LOCAL connection mode. DSN: {self._get_safe_connection_string(dsn)}"
            )
            return dsn
        elif mode == "stack":
            if db_url_from_sdk:
                host_part = db_url_from_sdk
                logger.info(f"Using STACK mode with SDK-provided DB host/endpoint: {host_part}")
                dsn = f"postgresql://{self.database_user}:{self.database_password}@{host_part}/{self.database_name}"
                return dsn
            elif self.PG_CONNECTION_STRING:
                logger.warning(
                    "Using STACK mode with legacy PG_CONNECTION_STRING (SDK URL not available/fetched)."
                )
                return self.PG_CONNECTION_STRING
            else:
                host_part = f"{self.database_host}:{self.database_port}"
                logger.warning(
                    f"Using STACK mode with default host settings: {host_part} (SDK URL not available/fetched and no legacy DSN)."
                )
                dsn = f"postgresql://{self.database_user}:{self.database_password}@{host_part}/{self.database_name}"
                return dsn
        else:
            raise ValueError(
                f"Invalid POSTGRES_MCP_CONNECTION_MODE: '{self.POSTGRES_MCP_CONNECTION_MODE}'. Must be 'local' or 'stack'."
            )

    # Helper to mask password for logging
    def _get_safe_connection_string(self, connection_string: Optional[str]) -> str:
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# --- PostgreSQL Handler Class ---
class PostgresHandler:
    """Handles interactions with the PostgreSQL database via MCP."""

    def __init__(self, settings: PostgresSettings):
        self.settings = settings
        self._pool: Optional[psycopg2.pool.SimpleConnectionPool] = None
        self._connection_error: Optional[str] = None
        self._last_sdk_url_used: Optional[str] = (
            None  # Store the SDK URL used for the current connection
        )
        self.database_url: Optional[str] = None  # The final DSN used for connection
        self.resource_base_url_parts: Optional[urlparse] = None  # Parsed DSN for URI building
        logger.info("PostgresHandler initialized.")

    async def try_connect_pool(self, sdk_db_url: Optional[str] = None) -> bool:
        """
        Attempts to establish or re-establish the database connection pool.
        Uses backoff for individual connection attempts within this call.

        Args:
            sdk_db_url: The database host/endpoint fetched from the SDK (for stack mode).

        Returns:
            True if connection pool is established successfully, False otherwise.
        """
        # Only proceed if not already connected or if the SDK URL has changed
        if (
            self._pool
            and self.settings.POSTGRES_MCP_CONNECTION_MODE.lower() == "stack"
            and sdk_db_url == self._last_sdk_url_used
        ):
            logger.debug("Already connected with the same SDK URL. Skipping connection attempt.")
            # Optionally add a check here to see if the existing pool is alive
            if self._is_pool_alive():
                return True
            else:
                logger.warning("Existing pool seems dead. Forcing reconnection attempt.")
                self._pool = None  # Force reconnect

        logger.info(f"Attempting to connect/reconnect DB pool. SDK URL: {sdk_db_url}")
        try:
            # Determine the final Database URL (DSN)
            self.database_url = self.settings.get_connection_string(db_url_from_sdk=sdk_db_url)
            safe_url = self.settings._get_safe_connection_string(
                self.database_url
            )  # Use helper from settings
            logger.info(f"Resolved PostgreSQL DSN for connection attempt: {safe_url}")

            # Parse the DSN for resource URI building
            self.resource_base_url_parts = urlparse(self.database_url)
            db_name = (self.resource_base_url_parts.path or "/").lstrip("/")
            logger.info(f"Target database: {db_name}")

            # Connect the pool using backoff for robustness
            await self._connect_pool_with_backoff()

            # Store the SDK URL that resulted in a successful connection
            if self.settings.POSTGRES_MCP_CONNECTION_MODE.lower() == "stack":
                self._last_sdk_url_used = sdk_db_url

            logger.info("DB Pool connection successful.")
            return True

        except Exception as e:
            self._connection_error = f"Failed to establish DB pool: {e}"
            logger.error(self._connection_error, exc_info=True)
            self._pool = None  # Ensure pool is None on failure
            self._last_sdk_url_used = None  # Clear last used URL on failure
            return False

    @backoff.on_exception(**get_backoff_config())
    async def _connect_pool_with_backoff(self):
        """Internal method to connect pool with backoff decorator."""
        if not self.database_url:
            raise ValueError("Database URL (DSN) not set, cannot connect pool.")

        logger.debug("Attempting to create/connect PostgreSQL connection pool (within backoff)...")
        # Close existing pool if reconnecting
        if self._pool:
            try:
                self._pool.closeall()
            except Exception as e:
                logger.warning(f"Error closing existing pool: {e}")
            finally:
                self._pool = None

        try:
            # Create the new pool
            new_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=self.settings.db_min_connections,
                maxconn=self.settings.db_max_connections,
                dsn=self.database_url,
            )
            # Test connection
            conn = new_pool.getconn()
            conn.cursor().execute("SELECT 1")
            new_pool.putconn(conn)
            # Assign successfully created pool
            self._pool = new_pool
            self._connection_error = None
            logger.debug("Connection pool established successfully (within backoff).")
        except (psycopg2.Error, Exception) as e:
            self._connection_error = f"Pool connection attempt failed: {e}"
            self._pool = None
            raise  # Re-raise for backoff

    def _is_pool_alive(self) -> bool:
        """Quick check if the pool exists and can provide a connection."""
        if not self._pool:
            return False
        conn = None
        try:
            conn = self._pool.getconn()
            # Quick ping
            conn.cursor().execute("SELECT 1")
            return True
        except psycopg2.Error:
            return False
        finally:
            if conn:
                self._pool.putconn(conn)

    @property
    def is_connected(self) -> bool:
        """Checks if the pool is initialized and no major error is preventing connection."""
        # Check if pool exists and no persistent error is logged
        # A more active check could be done here if needed, but relies on background loop
        return self._pool is not None and self._connection_error is None

    @property
    def last_error(self) -> Optional[str]:
        """Returns the last recorded connection error."""
        return self._connection_error

    @backoff.on_exception(**get_backoff_config())
    def _get_connection(self):
        """Gets a connection from the pool with retry logic."""
        if not self._pool:
            self._connection_error = "Connection pool is not initialized."
            logger.error(self._connection_error)
            # In the background loop model, we don't schedule reconnect here,
            # the main loop handles retry attempts. We just raise the error.
            raise ConnectionError(self._connection_error)
        try:
            conn = self._pool.getconn()
            # Optional: Add a quick check/ping here if needed
            if conn.status != psycopg2.extensions.STATUS_READY:
                logger.warning(
                    f"Got connection with bad status ({conn.status}). Closing and retrying."
                )
                self._pool.putconn(conn, close=True)  # Close bad connection
                raise ConnectionError(f"Connection status not ready ({conn.status})")
            # Test with a quick query and timeout
            try:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '1s'; SELECT 1; RESET statement_timeout;")
            except psycopg2.Error as ping_err:
                logger.warning(f"Connection ping failed: {ping_err}. Closing and retrying.")
                self._pool.putconn(conn, close=True)
                raise ConnectionError("Connection ping failed") from ping_err

            self._connection_error = None  # Clear error on successful get
            return conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError, ConnectionError) as e:
            self._connection_error = f"Failed to get connection: {e}"
            logger.error(self._connection_error)
            raise ConnectionError(
                "Failed to get a valid connection from pool"
            ) from e  # Reraise for backoff

    def _release_connection(self, conn):
        """Returns a connection to the pool safely."""
        if self._pool and conn:
            try:
                if not conn.closed:
                    self._pool.putconn(conn)
                else:
                    logger.warning("Attempted to release an already closed connection.")
            except Exception as e:
                logger.warning(f"Error putting connection back to pool: {e}")

    # --- MCP Methods (handle_list_resources, handle_read_resource, etc.) ---
    # These methods remain unchanged structurally but now rely on _get_connection,
    # which handles the retry logic. They should return appropriate errors
    # if _get_connection fails after retries.

    def handle_list_resources(self) -> ListResourcesResponse:
        """Lists schemas and tables as resources."""
        logger.debug("Handling mcp_listResources")
        conn = None
        try:
            conn = self._get_connection()  # Get connection with retry
            resources = []
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # (Query logic remains the same as previous version)
                cur.execute("""
                    SELECT schema_name FROM information_schema.schemata
                    WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
                      AND schema_name NOT LIKE 'pg_toast%' AND schema_name NOT LIKE 'pg_temp_%'
                """)
                schemas = [row["schema_name"] for row in cur.fetchall()]
                for schema in schemas:
                    cur.execute(
                        """
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = %s AND table_type = 'BASE TABLE'
                    """,
                        (schema,),
                    )
                    tables = cur.fetchall()
                    for row in tables:
                        table_name = row["table_name"]
                        try:
                            resource_uri = self._build_resource_uri(table_name, schema)
                            resources.append(
                                ResourceIdentifier(
                                    uri=resource_uri,
                                    name=f'"{schema}.{table_name}" table schema',
                                    mimeType="application/json",
                                )
                            )
                        except Exception as uri_err:
                            logger.warning(
                                f"Could not build URI for {schema}.{table_name}: {uri_err}"
                            )
            return ListResourcesResponse(resources=resources)
        except Exception as e:
            logger.error(f"Error listing resources: {e}", exc_info=True)
            raise  # Propagate error to be handled by the endpoint wrapper
        finally:
            if conn:
                self._release_connection(conn)

    def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
        """Reads the schema information for a specific table URI."""
        logger.debug(f"Handling mcp_readResource for URI: {params.uri}")
        conn = None
        try:
            # (URI parsing and validation logic remains the same)
            parsed_uri = urlparse(params.uri)
            path_parts = parsed_uri.path.strip("/").split("/")
            if len(path_parts) < 4 or path_parts[-1] != "schema":
                raise ValueError(f"Invalid resource URI path format: {parsed_uri.path}")
            table_name = path_parts[-2]
            schema_name = path_parts[-3]
            if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
                raise ValueError(f"Invalid table name: {table_name}")
            if not re.match(r"^[a-zA-Z0-9_]+$", schema_name):
                raise ValueError(f"Invalid schema name: {schema_name}")

            conn = self._get_connection()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # (Query logic remains the same)
                query = sql.SQL(
                    "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position"
                )
                cur.execute(query, (schema_name, table_name))
                columns = cur.fetchall()
                if not columns:
                    check_table_query = sql.SQL(
                        "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s"
                    )
                    cur.execute(check_table_query, (schema_name, table_name))
                    if not cur.fetchone():
                        raise ValueError(
                            f"Resource not found: Table or view '{schema_name}.{table_name}' does not exist."
                        )
                    else:
                        content_json = json.dumps([], indent=2)  # Table exists but no columns/perms
                else:
                    content_json = json.dumps(columns, indent=2, default=str)

                return ReadResourceResponse(
                    contents=[
                        ResourceContent(
                            uri=params.uri, text=content_json, mimeType="application/json"
                        )
                    ]
                )
        except ValueError as ve:
            logger.error(f"Validation error reading resource {params.uri}: {ve}")
            raise  # Propagate validation errors
        except Exception as e:
            logger.error(f"Error reading resource {params.uri}: {e}", exc_info=True)
            raise  # Propagate other errors
        finally:
            if conn:
                self._release_connection(conn)

    def handle_list_tools(self) -> ListToolsResponse:
        # (Remains the same)
        logger.debug("Handling mcp_listTools")
        query_tool = ToolDefinition(
            name="query",
            description="Run a read-only SQL query against the database.",
            inputSchema=ToolInputSchema(
                type="object",
                properties={"sql": {"type": "string", "description": "The read-only SQL query."}},
                required=["sql"],
            ),
        )
        return ListToolsResponse(tools=[query_tool])

    def handle_call_tool(self, params: CallToolRequestParams) -> CallToolResponse:
        """Executes the 'query' tool."""
        logger.debug(f"Handling mcp_callTool for tool: {params.name}")
        if params.name != "query":
            return CallToolResponse(
                content=[TextContent(text=f"Error: Unknown tool '{params.name}'.")], isError=True
            )
        if not params.arguments or "sql" not in params.arguments:
            return CallToolResponse(
                content=[TextContent(text="Error: Missing 'sql' argument.")], isError=True
            )
        sql_query = params.arguments["sql"].strip()

        # (Security checks remain the same)
        sql_upper = sql_query.upper()
        allowed_starts = ("SELECT", "EXPLAIN", "SHOW", "WITH")
        if not sql_upper.startswith(allowed_starts):
            return CallToolResponse(
                content=[
                    TextContent(text="Error: Only SELECT, EXPLAIN, SHOW, WITH queries allowed.")
                ],
                isError=True,
            )
        disallowed_keywords = {
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
            "SET",
        }
        found_disallowed = [
            kw for kw in disallowed_keywords if re.search(r"\b" + kw + r"\b", sql_upper)
        ]
        if found_disallowed:
            return CallToolResponse(
                content=[
                    TextContent(
                        text=f"Error: Query contains disallowed keywords ({', '.join(found_disallowed)})."
                    )
                ],
                isError=True,
            )

        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                try:
                    logger.info(
                        f"Executing read-only SQL: {sql_query[:100]}..."
                    )  # Log truncated query
                    cur.execute(sql_query)
                    if cur.description:
                        results = cur.fetchall()
                        result_json = json.dumps(results, indent=2, default=str)
                        logger.info(f"Query successful, {len(results)} rows.")
                    else:
                        result_json = json.dumps(
                            {"message": "Query executed, no rows returned."}, indent=2
                        )
                    conn.rollback()  # End read-only transaction
                    return CallToolResponse(content=[TextContent(text=result_json)], isError=False)
                except psycopg2.Error as query_error:
                    logger.error(f"SQL execution error: {query_error}", exc_info=True)
                    try:
                        conn.rollback()
                    except psycopg2.Error as rb_err:
                        logger.warning(f"Rollback failed: {rb_err}")
                    # Provide specific PG error if possible
                    pg_err_msg = str(query_error).strip()
                    # Basic check for common errors to provide hints
                    hint = ""
                    if "does not exist" in pg_err_msg:
                        hint = " Hint: Check table/schema names and spelling."
                    elif "permission denied" in pg_err_msg:
                        hint = " Hint: Check database user permissions."
                    return CallToolResponse(
                        content=[TextContent(text=f"SQL Execution Error: {pg_err_msg}{hint}")],
                        isError=True,
                    )
        except Exception as e:
            logger.error(f"Error calling tool '{params.name}': {e}", exc_info=True)
            raise  # Propagate error
        finally:
            if conn:
                self._release_connection(conn)

    # Added build_resource_uri helper matching previous version logic
    def _build_resource_uri(self, table_name: str, schema_name: str = "public") -> str:
        if not self.resource_base_url_parts or not self.database_url:
            raise RuntimeError("Cannot build resource URI: Handler not fully initialized.")
        scheme = "postgres"
        netloc = self.resource_base_url_parts.netloc
        db_path = (self.resource_base_url_parts.path or "/database").lstrip("/")
        resource_path = f"/{db_path}/{schema_name}/{table_name}/schema"
        uri_parts = urlparse(self.database_url)._replace(
            scheme=scheme, path=resource_path, query="", fragment=""
        )
        return urlunparse(uri_parts)

    def close(self):
        """Closes the connection pool."""
        logger.info("Closing PostgreSQL handler connection pool...")
        if self._pool:
            try:
                self._pool.closeall()
                logger.info("Connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
        self._pool = None


# --- Background Connection Task ---
async def background_db_connect_loop(handler: PostgresHandler, settings: PostgresSettings):
    """Background task to manage database connection."""
    logger.info("Starting background DB connection management loop.")
    current_sdk_url = None
    cm_initialized = False

    while True:
        try:
            if not handler.is_connected:
                logger.info("Background check: Handler not connected. Attempting connection...")
                # --- Ensure ConfigManager is ready (only needed for stack mode URL fetch) ---
                if settings.POSTGRES_MCP_CONNECTION_MODE.lower() == "stack" and not cm_initialized:
                    try:
                        # Check if CM has connectors; this implies init_manager ran and fetched.
                        # This is an indirect check.
                        if ConfigManager().connectors is not None:
                            cm_initialized = True
                            logger.info("ConfigManager appears initialized.")
                        else:
                            logger.warning(
                                "ConfigManager check failed (no connectors). Will retry later."
                            )
                            # Wait before next check if CM not ready
                            await asyncio.sleep(BACKGROUND_CONNECT_RETRY_SECONDS)
                            continue
                    except AttributeError:
                        logger.warning(
                            "ConfigManager check failed (AttributeError). Assuming not initialized. Will retry later."
                        )
                        await asyncio.sleep(BACKGROUND_CONNECT_RETRY_SECONDS)
                        continue
                    except Exception as cm_err:
                        logger.error(
                            f"Error checking ConfigManager status: {cm_err}. Will retry later."
                        )
                        await asyncio.sleep(BACKGROUND_CONNECT_RETRY_SECONDS)
                        continue

                # --- Fetch SDK URL if in stack mode and CM is ready ---
                sdk_url_to_use = None
                if settings.POSTGRES_MCP_CONNECTION_MODE.lower() == "stack":
                    if not cm_initialized:
                        logger.warning(
                            "Stack mode: ConfigManager not ready, cannot fetch SDK URL yet."
                        )
                    else:
                        try:
                            logger.debug(f"Fetching SDK URL for '{DATABASE_CONNECTOR_ID}'...")
                            # Use run_in_executor for the potentially blocking SDK call
                            loop = asyncio.get_running_loop()
                            fetched_url, _ = await loop.run_in_executor(
                                None, get_api_parameters, DATABASE_CONNECTOR_ID
                            )
                            if fetched_url:
                                if current_sdk_url != fetched_url:
                                    logger.info(f"New SDK URL fetched: {fetched_url}")
                                    current_sdk_url = fetched_url
                                else:
                                    logger.debug("SDK URL unchanged.")
                                sdk_url_to_use = current_sdk_url
                            else:
                                logger.warning(
                                    f"get_api_parameters('{DATABASE_CONNECTOR_ID}') returned None."
                                )
                                # Keep using old URL if fetch fails? Or reset? Resetting forces fallback.
                                # Let's keep using the last known good one if fetch fails now.
                                sdk_url_to_use = current_sdk_url
                        except Exception as e:
                            logger.error(
                                f"Error fetching SDK URL: {e}. Using last known URL: {current_sdk_url}",
                                exc_info=False,
                            )
                            sdk_url_to_use = (
                                current_sdk_url  # Keep using last known good URL on error
                            )

                # --- Attempt to connect the pool ---
                # Pass the fetched URL (or None if not stack mode or fetch failed)
                await handler.try_connect_pool(sdk_db_url=sdk_url_to_use)

                # --- Wait before next check ---
                if not handler.is_connected:
                    logger.warning(
                        f"Connection attempt failed. Retrying in {BACKGROUND_CONNECT_RETRY_SECONDS}s."
                    )
                    await asyncio.sleep(BACKGROUND_CONNECT_RETRY_SECONDS)
                else:
                    # If connected, wait longer before the next check cycle
                    await asyncio.sleep(BACKGROUND_CHECK_INTERVAL_SECONDS)

            else:
                # Handler is connected, sleep for the longer interval
                logger.debug(
                    f"Background check: Handler connected. Checking again in {BACKGROUND_CHECK_INTERVAL_SECONDS}s."
                )
                # Optional: Add a pool liveness check here if needed before sleeping
                # handler._is_pool_alive()
                await asyncio.sleep(BACKGROUND_CHECK_INTERVAL_SECONDS)

        except asyncio.CancelledError:
            logger.info("Background DB connection loop cancelled.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in background DB connection loop: {e}", exc_info=True)
            # Wait before retrying the loop on unexpected error
            await asyncio.sleep(BACKGROUND_CONNECT_RETRY_SECONDS)


# --- FastAPI Setup ---


# Request model for JSON-RPC
class MCPRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Any
    method: str
    params: Optional[Dict[str, Any]] = None


# Global instances and background task handle
postgres_handler: Optional[PostgresHandler] = None
postgres_settings: Optional[PostgresSettings] = None
db_connect_task: Optional[asyncio.Task] = None


# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup ===
    global postgres_handler, postgres_settings, db_connect_task
    logger.info("Lifespan startup: Initializing PostgreSQL MCP Server...")

    try:
        # 1. Load Settings
        postgres_settings = PostgresSettings()
        logger.info(f"Settings loaded (Mode: {postgres_settings.POSTGRES_MCP_CONNECTION_MODE}).")

        # 2. Initialize ConfigManager *unconditionally*
        #    It might be needed by other parts or for consistency.
        #    Ensure it happens before anything relies on it (like background task).
        logger.info("Initializing ConfigManager...")
        config_manager = ConfigManager()
        # Define potentially needed connectors for ConfigManager setup
        connectors_for_cm = [
            ConnectorSchema(type="database", label="Internal DB", config_id=DATABASE_CONNECTOR_ID)
        ]
        config_manager.init_manager(config_model=postgres_settings, connectors=connectors_for_cm)

        # 3. Create Handler Instance
        postgres_handler = PostgresHandler(postgres_settings)
        logger.info("PostgresHandler instance created.")

        # 4. Start Background Connection Loop
        #    This task runs independently, allowing FastAPI to start.
        loop = asyncio.get_event_loop()
        db_connect_task = loop.create_task(
            background_db_connect_loop(postgres_handler, postgres_settings)
        )
        logger.info("Background database connection task started.")

    except Exception as e:
        logger.error(f"FATAL: Exception during PostgreSQL MCP Server startup: {e}", exc_info=True)
        # Ensure globals are None if startup fails
        postgres_handler = None
        postgres_settings = None
        if db_connect_task:
            db_connect_task.cancel()  # Cancel task if started
        db_connect_task = None
        # Depending on severity, might want to raise to prevent app start? For now, log and continue.

    # --- Yield control to the application ---
    yield
    # --- End of application runtime ---

    # === Shutdown ===
    logger.info("Lifespan shutdown: Cleaning up PostgreSQL MCP Server resources...")
    # Cancel background task
    if db_connect_task and not db_connect_task.done():
        db_connect_task.cancel()
        try:
            await db_connect_task  # Wait for task to finish cancellation
        except asyncio.CancelledError:
            logger.info("Background DB connection task successfully cancelled.")
        except Exception as e:
            logger.error(f"Error waiting for background task cancellation: {e}")
    # Close handler resources (pool)
    if postgres_handler:
        postgres_handler.close()
    logger.info("Lifespan shutdown complete.")


# Create FastAPI app with the lifespan manager
app = FastAPI(title="PostgreSQL MCP Server", lifespan=lifespan)


# --- API Endpoints (/mcp, /health) ---


@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    """Handles incoming JSON-RPC requests for MCP methods."""
    # Check if handler is initialized
    if not postgres_handler:
        logger.error("MCP request rejected: PostgreSQL handler not initialized (startup error?).")
        # Use 503 Service Unavailable
        raise HTTPException(
            status_code=503, detail="Service Initializing: PostgreSQL handler not ready."
        )

    # Check if handler is connected *at this moment*
    if not postgres_handler.is_connected:
        last_err_msg = (
            f" Last Error: {postgres_handler.last_error}" if postgres_handler.last_error else ""
        )
        error_detail = f"Service Unavailable: Database connection not active.{last_err_msg}"
        logger.warning(f"MCP request failed: {error_detail}")
        # Return error in JSON-RPC format instead of raising HTTPException for tool errors
        response = JsonRpcResponse(
            id=request.id,
            error={"code": -32001, "message": error_detail},  # Custom error code for DB unavailable
        )
        return response.model_dump(exclude_none=True)

    logging.debug(f"Received MCP request: method={request.method}, id={request.id}")
    response_data = None
    error_data = None

    try:
        # Dispatch based on MCP method
        if request.method == "mcp_listResources":
            response_data = postgres_handler.handle_list_resources().model_dump()
        elif request.method == "mcp_readResource":
            params = ReadResourceRequestParams.model_validate(request.params or {})
            response_data = postgres_handler.handle_read_resource(params).model_dump()
        elif request.method == "mcp_listTools":
            response_data = postgres_handler.handle_list_tools().model_dump()
        elif request.method == "mcp_callTool":
            params = CallToolRequestParams.model_validate(request.params or {})
            tool_response = postgres_handler.handle_call_tool(params)  # Returns response model
            # Check if the tool call itself resulted in an error reported by the handler
            if tool_response.isError:
                # Extract error message from the tool response content
                error_message = "Tool execution failed."
                if tool_response.content and isinstance(tool_response.content[0], TextContent):
                    error_message = tool_response.content[0].text
                error_data = {
                    "code": -32000,
                    "message": error_message,
                }  # Generic server error for tool failure
                response_data = None  # No result if tool failed
            else:
                response_data = tool_response.model_dump()  # Success
        else:
            error_data = {"code": -32601, "message": f"Method not found: '{request.method}'"}

    except (ConnectionError, psycopg2.Error) as db_err:
        # Catch DB errors during request processing
        logger.error(f"Database error handling request {request.id}: {db_err}", exc_info=True)
        error_data = {"code": -32002, "message": f"Database Error: {str(db_err)}"}
    except ValueError as val_err:
        # Catch validation errors (bad URI, bad SQL, etc.)
        logger.warning(f"Invalid parameters handling request {request.id}: {val_err}")
        error_data = {"code": -32602, "message": f"Invalid Parameters: {str(val_err)}"}
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error handling request {request.id}: {e}", exc_info=True)
        error_data = {"code": -32000, "message": f"Internal Server Error: {str(e)}"}

    # Construct JSON-RPC response
    response = JsonRpcResponse(
        id=request.id, result=response_data if error_data is None else None, error=error_data
    )
    return response.model_dump(exclude_none=True)


@app.get("/health")
async def health_check():
    """Performs a health check on the service and its database connection."""
    health_status = {"status": "healthy", "db_connection": "disconnected", "mode": "unknown"}

    if postgres_settings:
        health_status["mode"] = postgres_settings.POSTGRES_MCP_CONNECTION_MODE

    if not postgres_handler:
        health_status["status"] = "unhealthy"
        health_status["db_connection"] = "handler_not_initialized"
        raise HTTPException(status_code=503, detail=health_status)

    if postgres_handler.is_connected:
        # Quick check if pool is alive - best effort
        if postgres_handler._is_pool_alive():
            health_status["db_connection"] = "ok"
        else:
            health_status["status"] = "unhealthy"
            health_status["db_connection"] = "stale_pool?"  # Pool exists but might be dead
            logger.warning("Health check: Handler reports connected, but pool seems stale.")
            # Don't raise 503 yet, background loop should fix it. Return current state.
    else:
        health_status["status"] = "unhealthy"  # unhealthy if DB not connected
        health_status["db_connection"] = "disconnected"
        if postgres_handler.last_error:
            health_status["last_db_error"] = postgres_handler.last_error
        # Return 200 but indicate unhealthy DB, or return 503?
        # Let's return 200 OK but signal db unhealthy state
        logger.warning(f"Health check: DB disconnected. Last error: {postgres_handler.last_error}")

    # If overall status is unhealthy, consider returning 503 status code?
    # For now, return 200 with detailed status.
    # if health_status["status"] == "unhealthy":
    #      raise HTTPException(status_code=503, detail=health_status)

    return health_status


# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn

    debug_mode = "--debug" in sys.argv
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    uvicorn_log_config["loggers"]["uvicorn.error"]["level"] = log_level
    uvicorn_log_config["loggers"]["uvicorn.access"]["level"] = log_level
    if debug_mode:
        logger.debug("Debug mode enabled.")
    uvicorn.run(
        "__main__:app", host="0.0.0.0", port=8001, reload=debug_mode, log_config=uvicorn_log_config
    )
