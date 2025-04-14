# gemini-code-dir/confidentialmind-mcp-agent/src/mcp/postgres_mcp_server.py
import asyncio
import json
import logging
import os
import re
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
class PostgresSettings(BaseSettings):
    # Removed POSTGRES_MCP_CONNECTION_MODE

    # Default connection settings (used if SDK doesn't provide a URL)
    database_host: str = "localhost"
    database_port: int = 5432
    database_user: str = "app"  # Default user for stack/local generic connection
    database_password: str = "testpass"  # Default password
    database_name: str = "vector-db"  # Default DB name

    # Pool settings
    db_min_connections: int = 2
    db_max_connections: int = 10
    db_connection_timeout: float = 30.0
    db_command_timeout: float = 60.0

    def get_connection_string(self, db_url_from_sdk: Optional[str] = None) -> str:
        """
        Generate PostgreSQL connection string.
        Uses the SDK-provided URL for the host part if available,
        otherwise falls back to default host/port settings.
        """
        logger.debug(f"Generating connection string. SDK URL provided: {db_url_from_sdk}")
        if db_url_from_sdk:
            # Use the SDK-provided URL (likely host or service name) as the host part
            host_part = db_url_from_sdk
            logger.info(f"Using SDK-provided DB host/endpoint: {host_part}")
        else:
            # Fallback to default host:port if no SDK URL is available
            host_part = f"{self.database_host}:{self.database_port}"
            logger.info(f"Using default DB host/port settings: {host_part}")

        # Construct DSN using potentially overridden user/pass/db from BaseSettings
        dsn = f"postgresql://{self.database_user}:{self.database_password}@{host_part}/{self.database_name}"
        logger.info(f"Constructed DSN: {self._get_safe_connection_string(dsn)}")
        return dsn

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
        env_file = ".env"  # BaseSettings automatically reads from .env
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
            sdk_db_url: The database host/endpoint fetched from the SDK.

        Returns:
            True if connection pool is established successfully, False otherwise.
        """
        # Check if connection is needed (pool doesn't exist or SDK URL changed)
        needs_connection_attempt = False
        if self._pool is None:
            needs_connection_attempt = True
            logger.info("No existing pool. Attempting connection.")
        elif sdk_db_url != self._last_sdk_url_used:
            needs_connection_attempt = True
            logger.info(
                f"SDK URL changed from '{self._last_sdk_url_used}' to '{sdk_db_url}'. Reconnecting."
            )
            # Close the old pool before creating a new one with the new URL
            if self._pool:
                try:
                    self._pool.closeall()
                    logger.info("Closed existing pool due to URL change.")
                except Exception as e:
                    logger.warning(f"Error closing existing pool during URL change: {e}")
                finally:
                    self._pool = None
        elif not self._is_pool_alive():
            needs_connection_attempt = True
            logger.warning("Existing pool seems dead. Forcing reconnection attempt.")
            self._pool = None  # Force reconnect by clearing the pool reference

        if not needs_connection_attempt:
            logger.debug(
                "Already connected with the same SDK URL and pool is alive. Skipping connection attempt."
            )
            return True

        logger.info(f"Attempting to connect/reconnect DB pool. SDK URL: {sdk_db_url}")
        try:
            # Determine the final Database URL (DSN) using the settings method
            self.database_url = self.settings.get_connection_string(db_url_from_sdk=sdk_db_url)
            # safe_url already logged within get_connection_string

            # Parse the DSN for resource URI building (only needs to happen once or when DSN changes)
            self.resource_base_url_parts = urlparse(self.database_url)
            db_name = (self.resource_base_url_parts.path or "/").lstrip("/")
            logger.info(f"Target database based on DSN path: {db_name}")

            # Connect the pool using backoff for robustness
            await self._connect_pool_with_backoff()

            # Store the SDK URL that resulted in a successful connection
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
        # Close existing pool if reconnecting (redundant if called from try_connect_pool after URL change, but safe)
        if self._pool:
            try:
                self._pool.closeall()
            except Exception as e:
                logger.warning(f"Error closing existing pool before reconnect: {e}")
            finally:
                self._pool = None

        try:
            # Create the new pool
            new_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=self.settings.db_min_connections,
                maxconn=self.settings.db_max_connections,
                dsn=self.database_url,
                # Add connect_timeout parameter if needed, e.g.,
                # connect_timeout=self.settings.db_connection_timeout
            )
            # Test connection
            conn = new_pool.getconn()
            # Set command timeout for the test query
            conn.cursor().execute(
                f"SET statement_timeout = {int(self.settings.db_command_timeout * 1000)}; SELECT 1; RESET statement_timeout;"
            )
            new_pool.putconn(conn)
            # Assign successfully created pool
            self._pool = new_pool
            self._connection_error = None
            logger.debug("Connection pool established successfully (within backoff).")
        except (psycopg2.Error, Exception) as e:
            self._connection_error = f"Pool connection attempt failed: {e}"
            logger.error(
                f"Pool connection attempt failed: {e}", exc_info=True
            )  # Log full trace on error
            self._pool = None
            raise  # Re-raise for backoff

    def _is_pool_alive(self) -> bool:
        """Quick check if the pool exists and can provide a connection."""
        if not self._pool:
            return False
        conn = None
        try:
            conn = self._pool.getconn(
                key=None
            )  # Use key=None for compatibility if pool keys are used
            # Quick ping with timeout
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '1s'; SELECT 1; RESET statement_timeout;")
            return True
        except psycopg2.Error as e:
            logger.warning(f"Pool liveness check failed: {e}")
            return False
        finally:
            if conn:
                try:
                    self._pool.putconn(
                        conn, key=None, close=False
                    )  # Return connection without closing
                except Exception as e:
                    logger.warning(f"Error returning connection during liveness check: {e}")
                    # Consider closing the potentially problematic connection
                    if conn:
                        try:
                            self._pool.putconn(conn, key=None, close=True)
                        except Exception:
                            pass  # Ignore errors during forced close

    @property
    def is_connected(self) -> bool:
        """Checks if the pool is initialized and no major error is preventing connection."""
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
            raise ConnectionError(self._connection_error)
        try:
            conn = self._pool.getconn()
            # Check status before ping
            if conn.status != psycopg2.extensions.STATUS_READY:
                logger.warning(
                    f"Got connection with bad status ({conn.status}). Closing and retrying."
                )
                try:
                    self._pool.putconn(conn, close=True)  # Close bad connection
                except Exception:
                    pass  # Ignore errors during forced close
                raise ConnectionError(f"Connection status not ready ({conn.status})")

            # Test with a quick query and timeout
            try:
                with conn.cursor() as cur:
                    # Use the configured command timeout
                    cur.execute(
                        f"SET statement_timeout = '{int(self.settings.db_command_timeout * 1000)}'; SELECT 1; RESET statement_timeout;"
                    )
            except psycopg2.Error as ping_err:
                logger.warning(f"Connection ping failed: {ping_err}. Closing and retrying.")
                try:
                    self._pool.putconn(conn, close=True)
                except Exception:
                    pass
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

    # --- MCP Methods ---

    def handle_list_resources(self) -> ListResourcesResponse:
        """Lists schemas and tables as resources."""
        logger.debug("Handling mcp_listResources")
        conn = None
        try:
            conn = self._get_connection()  # Get connection with retry
            resources = []
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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
            parsed_uri = urlparse(params.uri)
            path_parts = parsed_uri.path.strip("/").split("/")
            # Expecting format like /dbname/schemaname/tablename/schema
            if len(path_parts) < 4 or path_parts[-1] != "schema":
                raise ValueError(
                    f"Invalid resource URI path format: {parsed_uri.path}. Expected '/db/schema/table/schema'"
                )
            table_name = path_parts[-2]
            schema_name = path_parts[-3]
            # Basic validation
            if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
                raise ValueError(f"Invalid table name format in URI: {table_name}")
            if not re.match(r"^[a-zA-Z0-9_]+$", schema_name):
                raise ValueError(f"Invalid schema name format in URI: {schema_name}")

            conn = self._get_connection()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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
                        # Table exists but no columns found (or permissions issue)
                        content_json = json.dumps(
                            {
                                "message": f"Table '{schema_name}.{table_name}' exists but no column information retrieved (check permissions?)."
                            },
                            indent=2,
                        )
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

        # Security checks
        sql_upper = sql_query.upper()
        allowed_starts = ("SELECT", "EXPLAIN", "SHOW", "WITH")
        if not sql_upper.startswith(allowed_starts):
            # Allow CTEs that might have INSERT/UPDATE/DELETE inside but start with WITH
            if not sql_upper.startswith("WITH"):
                return CallToolResponse(
                    content=[
                        TextContent(text="Error: Only SELECT, EXPLAIN, SHOW, WITH queries allowed.")
                    ],
                    isError=True,
                )

        # More robust check for disallowed keywords, avoiding matches within identifiers
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
            "SET ",  # Added space to avoid matching SET statement_timeout
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT",
            "BEGIN",
            "DECLARE",
            "EXECUTE",  # Other potentially harmful commands
            "COPY",  # Potentially dangerous
        }
        # Use word boundaries to prevent matching substrings
        # Be careful with comments -- this simple check doesn't parse SQL fully
        # Remove simple SQL comments first
        sql_no_comments = re.sub(r"--.*$", "", sql_query, flags=re.MULTILINE)
        sql_no_comments = re.sub(r"/\*.*?\*/", "", sql_no_comments, flags=re.DOTALL)

        found_disallowed = [
            kw
            for kw in disallowed_keywords
            if re.search(r"\b" + kw + r"\b", sql_no_comments, re.IGNORECASE)
        ]

        # Allow SET statement_timeout specifically
        if "SET statement_timeout" in sql_upper:
            found_disallowed = [kw for kw in found_disallowed if kw != "SET "]

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
                # Set transaction to read-only for safety
                cur.execute("SET TRANSACTION READ ONLY;")
                # Apply command timeout
                cur.execute(
                    f"SET statement_timeout = '{int(self.settings.db_command_timeout * 1000)}';"
                )

                try:
                    logger.info(
                        f"Executing read-only SQL: {sql_query[:100]}..."
                    )  # Log truncated query
                    cur.execute(sql_query)
                    if cur.description:  # Check if the query returned columns
                        results = cur.fetchall()
                        result_json = json.dumps(results, indent=2, default=str)
                        row_count = len(results)
                        logger.info(
                            f"Query successful, {row_count} row{'s' if row_count != 1 else ''} returned."
                        )
                    else:
                        # Query executed but didn't return rows (e.g., EXPLAIN, SHOW)
                        status_message = (
                            cur.statusmessage or "Query executed successfully, no rows returned."
                        )
                        result_json = json.dumps({"message": status_message}, indent=2)
                        logger.info(f"Query executed: {status_message}")

                    # Reset timeout and rollback (ends read-only transaction)
                    try:
                        cur.execute("RESET statement_timeout;")
                        conn.rollback()
                    except psycopg2.Error as reset_err:
                        logger.warning(f"Error during query cleanup (reset/rollback): {reset_err}")

                    return CallToolResponse(content=[TextContent(text=result_json)], isError=False)

                except psycopg2.Error as query_error:
                    logger.error(f"SQL execution error: {query_error}", exc_info=True)
                    try:
                        conn.rollback()  # Rollback on error
                    except psycopg2.Error as rb_err:
                        logger.warning(f"Rollback failed after query error: {rb_err}")

                    # Provide specific PG error if possible
                    pg_err_msg = str(query_error).strip()
                    # Basic check for common errors to provide hints
                    hint = ""
                    if "does not exist" in pg_err_msg:
                        hint = " Hint: Check table/schema names and spelling, ensure schema qualification (schema.table)."
                    elif "permission denied" in pg_err_msg:
                        hint = " Hint: Check database user permissions."
                    elif "timeout" in pg_err_msg:
                        hint = f" Hint: Query exceeded the allowed time limit ({self.settings.db_command_timeout}s)."

                    return CallToolResponse(
                        content=[TextContent(text=f"SQL Execution Error: {pg_err_msg}{hint}")],
                        isError=True,
                    )
        except ConnectionError as conn_err:
            # Handle connection errors from _get_connection specifically
            logger.error(f"Failed to get database connection for tool '{params.name}': {conn_err}")
            return CallToolResponse(
                content=[TextContent(text=f"Database Connection Error: {conn_err}")], isError=True
            )
        except Exception as e:
            logger.error(f"Error calling tool '{params.name}': {e}", exc_info=True)
            # Return generic error, but log details
            return CallToolResponse(
                content=[TextContent(text=f"Internal Server Error executing tool: {e}")],
                isError=True,
            )
        finally:
            if conn:
                self._release_connection(conn)

    def _build_resource_uri(self, table_name: str, schema_name: str = "public") -> str:
        """Builds a resource URI for a table schema."""
        if not self.resource_base_url_parts or not self.database_url:
            raise RuntimeError("Cannot build resource URI: Handler not fully initialized.")

        # Use info parsed from the *actual* connection DSN
        scheme = self.resource_base_url_parts.scheme or "postgres"  # Should be postgresql
        netloc = self.resource_base_url_parts.netloc  # Includes user:pass@host:port
        db_path = (self.resource_base_url_parts.path or "/database").lstrip(
            "/"
        )  # Get DB name from path

        # Construct the resource path: /database_name/schema_name/table_name/schema
        resource_path = f"/{db_path}/{schema_name}/{table_name}/schema"

        # Create new URL parts based on the original DSN but with the resource path
        uri_parts = self.resource_base_url_parts._replace(path=resource_path, query="", fragment="")
        # Ensure scheme is 'postgres' for the MCP URI standard if DSN was 'postgresql'
        if uri_parts.scheme == "postgresql":
            uri_parts = uri_parts._replace(scheme="postgres")

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
    current_sdk_url: Optional[str] = None  # Explicitly Optional[str]
    cm_initialized = False

    while True:
        next_check_delay = BACKGROUND_CHECK_INTERVAL_SECONDS  # Default long interval

        try:
            # --- Ensure ConfigManager is ready ---
            # Moved outside the !is_connected block to allow URL updates even if connected
            if not cm_initialized:
                try:
                    if ConfigManager().connectors is not None:  # Check if CM has connectors list
                        cm_initialized = True
                        logger.info("ConfigManager appears initialized.")
                    else:
                        logger.warning(
                            "ConfigManager check failed (no connectors). Will retry initialization check later."
                        )
                        await asyncio.sleep(BACKGROUND_CONNECT_RETRY_SECONDS)
                        continue  # Retry CM check first
                except Exception as cm_err:
                    logger.error(
                        f"Error checking ConfigManager status: {cm_err}. Will retry later."
                    )
                    await asyncio.sleep(BACKGROUND_CONNECT_RETRY_SECONDS)
                    continue  # Retry CM check first

            # --- Fetch SDK URL if CM is ready ---
            # This determines if we are in 'stack' mode or using local overrides
            sdk_url_to_use: Optional[str] = None  # Explicitly Optional[str]
            if cm_initialized:
                try:
                    logger.debug(f"Fetching SDK URL for '{DATABASE_CONNECTOR_ID}'...")
                    # Use run_in_executor for the potentially blocking SDK call
                    loop = asyncio.get_running_loop()
                    fetched_url, _ = await loop.run_in_executor(
                        None, get_api_parameters, DATABASE_CONNECTOR_ID
                    )

                    if fetched_url:
                        if current_sdk_url != fetched_url:
                            logger.info(
                                f"New SDK URL fetched via get_api_parameters: {fetched_url}"
                            )
                            current_sdk_url = fetched_url
                        else:
                            logger.debug("SDK URL unchanged.")
                        sdk_url_to_use = current_sdk_url
                    else:
                        # If get_api_parameters returns None, it means the connector is not configured
                        # or we are in local mode without an override.
                        logger.info(
                            f"get_api_parameters('{DATABASE_CONNECTOR_ID}') returned None. Assuming local default connection."
                        )
                        # If URL becomes None, trigger potential reconnect if handler was using an old SDK URL
                        if current_sdk_url is not None:
                            logger.info("SDK URL removed. Will attempt connect with defaults.")
                        current_sdk_url = None  # Reset stored SDK URL
                        sdk_url_to_use = None  # Explicitly pass None

                except Exception as e:
                    logger.error(
                        f"Error fetching SDK URL: {e}. Using last known URL: {current_sdk_url}",
                        exc_info=False,
                    )
                    sdk_url_to_use = current_sdk_url  # Keep using last known good URL on error

            else:  # CM not initialized
                logger.warning("ConfigManager not ready, cannot fetch SDK URL yet.")
                # Cannot determine SDK URL, will rely on handler defaults if connecting now

            # --- Attempt connection if disconnected or SDK URL has changed ---
            # Note: try_connect_pool now internally checks if sdk_url has changed
            if not handler.is_connected or sdk_url_to_use != handler._last_sdk_url_used:
                if not handler.is_connected:
                    logger.info("Background check: Handler not connected. Attempting connection...")
                else:  # SDK URL must have changed
                    logger.info(
                        "Background check: SDK URL change detected. Attempting reconnection..."
                    )

                await handler.try_connect_pool(sdk_db_url=sdk_url_to_use)

                if not handler.is_connected:
                    logger.warning(
                        f"Connection attempt failed. Retrying connection in {BACKGROUND_CONNECT_RETRY_SECONDS}s."
                    )
                    next_check_delay = BACKGROUND_CONNECT_RETRY_SECONDS  # Short retry interval
                # else: Connection succeeded, use default long interval set at start of loop

            else:
                # Handler is connected, and SDK URL hasn't changed. Do a quick liveness check.
                logger.debug(f"Background check: Handler connected. Performing liveness check.")
                if not handler._is_pool_alive():
                    logger.warning(
                        "Background check: Pool liveness check failed! Attempting reconnect."
                    )
                    # Force reconnect by clearing the pool reference
                    handler._pool = None
                    handler._connection_error = "Pool failed liveness check"  # Set error state
                    await handler.try_connect_pool(
                        sdk_db_url=sdk_url_to_use
                    )  # Try immediate reconnect
                    if not handler.is_connected:
                        next_check_delay = (
                            BACKGROUND_CONNECT_RETRY_SECONDS  # Short retry if reconnect fails
                        )
                # else: Pool is alive, use default long interval

            # --- Wait before next check cycle ---
            logger.debug(f"Background check loop finished. Sleeping for {next_check_delay}s.")
            await asyncio.sleep(next_check_delay)

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
    global postgres_handler, postgres_settings, db_connect_task
    logger.info("Lifespan startup: Initializing PostgreSQL MCP Server...")

    try:
        # 1. Load Settings from .env and defaults
        postgres_settings = PostgresSettings()
        logger.info(f"PostgresSettings loaded (defaults + .env overrides).")

        # 2. Initialize ConfigManager
        #    Must happen before background loop starts if SDK URLs are needed.
        logger.info("Initializing ConfigManager...")
        config_manager = ConfigManager()
        # Define the connector this MCP server *needs* from the ConfigManager
        connectors_for_cm = [
            ConnectorSchema(
                type="database", label="Target Database", config_id=DATABASE_CONNECTOR_ID
            )
        ]
        config_manager.init_manager(
            config_model=postgres_settings,  # Pass settings to be managed (optional here)
            connectors=connectors_for_cm,
        )
        logger.info("ConfigManager initialized.")

        # 3. Create Handler Instance with loaded settings
        postgres_handler = PostgresHandler(postgres_settings)
        logger.info("PostgresHandler instance created.")

        # 4. Start Background Connection Loop
        loop = asyncio.get_event_loop()
        db_connect_task = loop.create_task(
            background_db_connect_loop(postgres_handler, postgres_settings)
        )
        logger.info("Background database connection task started.")

    except Exception as e:
        logger.error(f"FATAL: Exception during PostgreSQL MCP Server startup: {e}", exc_info=True)
        postgres_handler = None
        postgres_settings = None
        if db_connect_task:
            db_connect_task.cancel()
        db_connect_task = None

    # --- Yield control to the application ---
    yield
    # --- End of application runtime ---

    # === Shutdown ===
    logger.info("Lifespan shutdown: Cleaning up PostgreSQL MCP Server resources...")
    if db_connect_task and not db_connect_task.done():
        db_connect_task.cancel()
        try:
            await db_connect_task  # Wait for task to finish cancellation
        except asyncio.CancelledError:
            logger.info("Background DB connection task successfully cancelled.")
        except Exception as e:
            logger.error(f"Error waiting for background task cancellation: {e}")
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
        raise HTTPException(
            status_code=503, detail="Service Initializing: PostgreSQL handler not ready."
        )

    # Check if handler is connected *at this moment*
    # Use the property which checks pool and error status
    if not postgres_handler.is_connected:
        last_err_msg = (
            f" Last Error: {postgres_handler.last_error}" if postgres_handler.last_error else ""
        )
        error_detail = f"Service Unavailable: Database connection not active.{last_err_msg}"
        logger.warning(f"MCP request failed: {error_detail}")
        # Return error in JSON-RPC format
        response = JsonRpcResponse(
            id=request.id,
            error={"code": -32001, "message": error_detail},  # Custom error code
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
                error_message = "Tool execution failed."
                if tool_response.content and isinstance(tool_response.content[0], TextContent):
                    error_message = tool_response.content[0].text
                error_data = {
                    "code": -32000,  # Generic server error for tool failure
                    "message": error_message,
                }
                response_data = None  # No result if tool failed
            else:
                response_data = tool_response.model_dump()  # Success
        else:
            error_data = {"code": -32601, "message": f"Method not found: '{request.method}'"}

    except (ConnectionError, psycopg2.Error) as db_err:
        # Catch DB errors during request processing (e.g., connection lost mid-request)
        logger.error(f"Database error handling request {request.id}: {db_err}", exc_info=True)
        error_data = {"code": -32002, "message": f"Database Error during request: {str(db_err)}"}
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
    health_status = {"status": "healthy", "db_connection": "unknown"}

    if not postgres_handler or not postgres_settings:
        health_status["status"] = "unhealthy"
        health_status["db_connection"] = "handler_or_settings_not_initialized"
        # Return 503 if essential components are missing
        raise HTTPException(status_code=503, detail=health_status)

    # Check connection status via the handler's property
    if postgres_handler.is_connected:
        # Optional: Add a more active check like _is_pool_alive if needed
        if postgres_handler._is_pool_alive():
            health_status["db_connection"] = "ok"
        else:
            # Pool exists but might be stale - report potentially unhealthy
            health_status["status"] = "unhealthy"
            health_status["db_connection"] = "stale_pool?"
            logger.warning("Health check: Handler reports connected, but pool seems stale.")
    else:
        health_status["status"] = "unhealthy"  # unhealthy if DB not connected
        health_status["db_connection"] = "disconnected"
        if postgres_handler.last_error:
            health_status["last_db_error"] = postgres_handler.last_error
        logger.warning(f"Health check: DB disconnected. Last error: {postgres_handler.last_error}")

    # Decide whether to return 503 based on overall status
    if health_status["status"] == "unhealthy":
        # Return 503 Service Unavailable if DB connection is down
        raise HTTPException(status_code=503, detail=health_status)

    return health_status  # Return 200 OK if status is healthy


# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn

    # More robust debug flag check
    debug_mode = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")

    log_level = logging.DEBUG if debug_mode else logging.INFO
    # Configure root logger and specific loggers
    logging.basicConfig(level=log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logging.getLogger("postgres-mcp").setLevel(log_level)  # Set level for our logger
    logging.getLogger("backoff").setLevel(
        logging.WARNING
    )  # Quieten backoff library unless warning/error

    # Configure Uvicorn logging based on debug mode
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    if debug_mode:
        uvicorn_log_config["loggers"]["uvicorn.error"]["level"] = "DEBUG"
        uvicorn_log_config["loggers"]["uvicorn.access"]["level"] = "DEBUG"
    else:
        uvicorn_log_config["loggers"]["uvicorn.error"]["level"] = "INFO"
        uvicorn_log_config["loggers"]["uvicorn.access"]["level"] = "INFO"

    logger.info(f"Starting PostgreSQL MCP Server (Debug Mode: {debug_mode})")
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=8080,  # Standard port for MCP servers unless overridden
        reload=debug_mode,  # Enable reload only in debug mode
        log_config=uvicorn_log_config,
    )
