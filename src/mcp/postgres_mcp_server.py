# gemini-code-dir/confidentialmind-mcp-agent/src/mcp/postgres_mcp_server.py
import asyncio
import json
import logging
import re
import sys
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import backoff
import psycopg2
import psycopg2.extras
import psycopg2.pool
from confidentialmind_core.config_manager import get_api_parameters
from fastapi import FastAPI, HTTPException
from psycopg2 import sql
from pydantic import BaseModel
from pydantic_settings import BaseSettings

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

# Create FastAPI app
app = FastAPI(title="PostgreSQL MCP Server")


# --- Backoff Configuration ---
def get_backoff_config() -> Dict[str, Any]:
    """Get backoff configuration."""
    # Simple fixed config for now, can be made env-dependent like baserag if needed
    return {
        "wait_gen": backoff.expo,
        "exception": (psycopg2.Error, ConnectionError, TimeoutError),  # Exceptions to retry on
        "max_tries": 10,  # Limit retries slightly
        "max_time": 300,  # 5 minutes total retry time
        "on_backoff": lambda details: logger.warning(
            f"DB connection/operation failed (attempt {details['tries']}). Retrying in {details['wait']:.1f}s..."
        ),
        "on_giveup": lambda details: logger.error(
            f"DB connection/operation failed after {details['tries']} tries. Giving up."
        ),
    }


# PostgreSQL settings using Pydantic (Unchanged from previous version)
class PostgresSettings(BaseSettings):
    """
    Pydantic settings for PostgreSQL connection configuration.
    Supports both local and stack connection modes.
    """

    POSTGRES_MCP_CONNECTION_MODE: str = "stack"
    PG_LOCAL_HOST: str = "localhost"
    PG_LOCAL_PORT: int = 5432
    PG_LOCAL_USER: str = "postgres"
    PG_LOCAL_PASSWORD: str = "postgres"
    PG_LOCAL_DBNAME: str = "postgres"
    PG_CONNECTION_STRING: Optional[str] = None  # Legacy fallback
    database_host: str = "localhost"  # Default stack settings
    database_port: int = 5432
    database_user: str = "app"
    database_password: str = "testpass"
    database_name: str = "vector-db"
    # Pool settings
    db_min_connections: int = 2
    db_max_connections: int = 10
    db_connection_timeout: float = 30.0  # Seconds
    db_command_timeout: float = 60.0  # Seconds

    def get_connection_string(self, db_url_from_sdk: Optional[str] = None) -> str:
        if self.POSTGRES_MCP_CONNECTION_MODE.lower() == "local":
            conn_string = f"postgresql://{self.PG_LOCAL_USER}:{self.PG_LOCAL_PASSWORD}@{self.PG_LOCAL_HOST}:{self.PG_LOCAL_PORT}/{self.PG_LOCAL_DBNAME}"
            logger.info(f"Using LOCAL connection mode: {self.PG_LOCAL_HOST}:{self.PG_LOCAL_PORT}")
            return conn_string
        elif self.POSTGRES_MCP_CONNECTION_MODE.lower() == "stack":
            if db_url_from_sdk:
                host_part = db_url_from_sdk
                logger.info(f"Using stack mode with SDK-provided URL: {host_part}")
                conn_string = f"postgresql://{self.database_user}:{self.database_password}@{host_part}/{self.database_name}"
                return conn_string
            elif self.PG_CONNECTION_STRING:
                logger.warning(
                    "Using legacy PG_CONNECTION_STRING for stack mode as SDK lookup failed or returned None."
                )
                return self.PG_CONNECTION_STRING
            else:
                host_part = f"{self.database_host}:{self.database_port}"
                logger.warning(
                    f"Using stack mode with default host settings: {host_part}. SDK lookup failed/returned None and no legacy PG_CONNECTION_STRING found."
                )
                conn_string = f"postgresql://{self.database_user}:{self.database_password}@{host_part}/{self.database_name}"
                return conn_string
        else:
            raise ValueError(f"Invalid connection mode: {self.POSTGRES_MCP_CONNECTION_MODE}")


# PostgreSQL handler class - Now with state and resilience
class PostgresHandler:
    def __init__(self, settings: PostgresSettings):
        """Initialize PostgreSQL handler settings."""
        self.settings = settings
        self._pool: Optional[psycopg2.pool.SimpleConnectionPool] = None
        self._connection_error: Optional[str] = None
        self._is_connecting: bool = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self.database_url: Optional[str] = None
        self.resource_base_url_parts: Optional[urlparse] = None

    async def initialize(self):
        """Asynchronously initialize the handler: fetch URL and connect pool with retry."""
        if self._pool or self._is_connecting:
            logger.info("Initialization already complete or in progress.")
            return

        self._is_connecting = True
        self._connection_error = None  # Reset error state

        try:
            # 1. Fetch SDK URL with retry loop (only in stack mode)
            sdk_db_url = None
            if self.settings.POSTGRES_MCP_CONNECTION_MODE.lower() == "stack":
                sdk_db_url = await self._fetch_sdk_url_with_retry()

            # 2. Determine final Database URL
            self.database_url = self.settings.get_connection_string(db_url_from_sdk=sdk_db_url)
            safe_url = self._get_safe_connection_string(self.database_url)
            logger.info(f"Resolved PostgreSQL connection DSN: {safe_url}")

            # 3. Parse URL for resource identification
            self.resource_base_url_parts = urlparse(self.database_url)
            logger.info(f"Handler will manage database: {self.resource_base_url_parts.path}")

            # 4. Connect the pool with backoff
            await self._connect_pool_with_backoff()

            logger.info("PostgresHandler initialized successfully.")

        except Exception as e:
            # Errors during _connect_pool_with_backoff are logged there
            # This catches errors in URL fetching or parsing
            self._connection_error = f"Failed during handler initialization: {e}"
            logger.error(self._connection_error, exc_info=True)
            self._pool = None  # Ensure pool is None on failure
            # Schedule a reconnect attempt even if initial setup fails badly
            self._schedule_reconnect()
            # Re-raise to signal startup failure if needed elsewhere
            # raise RuntimeError(f"PostgresHandler initialization failed: {e}") from e
        finally:
            self._is_connecting = False

    async def _fetch_sdk_url_with_retry(self) -> Optional[str]:
        """Fetch SDK URL with retry logic, similar to baserag."""
        if not get_api_parameters:
            logger.error("Cannot fetch DATABASE URL from SDK: get_api_parameters not available.")
            return None

        attempt = 0
        max_attempts = 12  # Try for ~1 minute
        while attempt < max_attempts:
            attempt += 1
            logger.info(f"Attempt {attempt}/{max_attempts} to fetch DATABASE URL from SDK...")
            try:
                # get_api_parameters is synchronous, run it in executor if needed,
                # but assuming it's fast enough for now. If it blocks heavily,
                # loop = asyncio.get_running_loop()
                # sdk_db_url, _ = await loop.run_in_executor(None, get_api_parameters, "DATABASE")
                sdk_db_url, _ = get_api_parameters(
                    "DATABASE"
                )  # Assuming 'DATABASE' is the config_id

                if sdk_db_url:
                    logger.info(f"Successfully fetched DATABASE URL from SDK: {sdk_db_url}")
                    return sdk_db_url
                else:
                    logger.warning("get_api_parameters('DATABASE') returned None.")
            except Exception as e:
                logger.warning(f"Error fetching DATABASE URL from SDK: {e}")

            if attempt < max_attempts:
                await asyncio.sleep(5)  # Wait 5 seconds before retrying

        logger.error(f"Failed to fetch DATABASE URL from SDK after {max_attempts} attempts.")
        return None

    @backoff.on_exception(**get_backoff_config())
    async def _connect_pool_with_backoff(self):
        """Attempt to connect the pool, decorated with backoff."""
        if not self.database_url:
            raise ValueError("Database URL not set, cannot connect pool.")

        logger.info("Attempting to create/connect PostgreSQL connection pool...")
        # Close existing pool if retrying
        if self._pool:
            try:
                self._pool.closeall()
            except Exception as e:
                logger.warning(f"Error closing existing pool before reconnect: {e}")
            self._pool = None

        # Create and test the pool
        try:
            new_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=self.settings.db_min_connections,
                maxconn=self.settings.db_max_connections,
                dsn=self.database_url,
                # Note: psycopg2 pool doesn't directly take timeouts like asyncpg
                # Timeouts are typically handled at the connection/cursor level
            )
            # Test connection
            conn = new_pool.getconn()
            conn.cursor().execute("SELECT 1")  # Quick test query
            new_pool.putconn(conn)

            self._pool = new_pool  # Assign only on success
            self._connection_error = None  # Clear error on success
            logger.info("PostgreSQL connection pool established successfully.")

        except (psycopg2.Error, Exception) as e:
            self._connection_error = f"Failed to connect pool: {e}"
            logger.error(f"Attempt to connect pool failed: {e}")
            self._pool = None  # Ensure pool is None on failure
            raise  # Re-raise the exception for backoff to catch

    def _schedule_reconnect(self):
        """Schedule a reconnection attempt if not already running."""
        if self._reconnect_task and not self._reconnect_task.done():
            logger.debug("Reconnection task already scheduled.")
            return

        logger.info("Scheduling background task for DB reconnection attempts.")
        loop = asyncio.get_event_loop()
        self._reconnect_task = loop.create_task(self._connect_pool_with_backoff())

    @property
    def is_connected(self) -> bool:
        """Check if the pool is initialized and likely connected."""
        # Basic check: is pool object created and no persistent error recorded?
        # A more robust check might try acquiring a connection briefly.
        return self._pool is not None and self._connection_error is None

    @property
    def last_error(self) -> Optional[str]:
        """Get the last recorded connection error."""
        return self._connection_error

    def _get_safe_connection_string(self, connection_string: Optional[str]) -> str:
        # (Implementation unchanged - masks password)
        if not connection_string:
            return "DSN not configured"
        try:
            parts = urlparse(connection_string)
            if "@" in parts.netloc:
                credentials, host_port = parts.netloc.split("@", 1)
                if ":" in credentials:
                    username, _ = credentials.split(":", 1)
                    masked_netloc = f"{username}:****@{host_port}"
                    parts = parts._replace(netloc=masked_netloc)
                    return urlunparse(parts)
            return connection_string
        except Exception:
            return "postgresql://username:****@host:port/database"

    # Wrap connection acquisition with backoff for resilience during operations
    @backoff.on_exception(**get_backoff_config())
    def _get_connection(self):
        """Get a connection from the pool, with retry."""
        if not self._pool:
            logger.error("Connection pool is not available.")
            # Try to trigger reconnect explicitly if pool is None
            self._connection_error = "Connection pool is None."
            self._schedule_reconnect()
            raise ConnectionError("Connection pool is not initialized or connection failed.")
        try:
            conn = self._pool.getconn()
            # Minimal check: Ping the server if connection seems idle or potentially stale
            # This adds overhead but increases reliability.
            # Adjust based on performance needs. A simple status check might be enough.
            if conn.status == psycopg2.extensions.STATUS_READY:
                try:
                    # Use a very short timeout for the ping
                    conn.cursor().execute(
                        "SET statement_timeout = '1s'; SELECT 1; RESET statement_timeout;"
                    )
                    self._connection_error = None  # Clear error if ping succeeds
                    return conn
                except psycopg2.Error as ping_err:
                    logger.warning(
                        f"Connection ping failed: {ping_err}. Attempting to close and retry."
                    )
                    # Close the potentially broken connection
                    self._pool.putconn(conn, close=True)
                    raise ConnectionError("Connection ping failed, retrying") from ping_err
            else:
                logger.warning(
                    f"Connection status not ready ({conn.status}). Attempting to close and retry."
                )
                self._pool.putconn(conn, close=True)  # Close potentially bad connection
                raise ConnectionError(f"Connection status not ready ({conn.status}), retrying")

        except (psycopg2.Error, ConnectionError) as e:
            self._connection_error = f"Failed to get connection: {e}"
            logger.error(self._connection_error)
            # If getting a connection fails, schedule a reconnect of the whole pool
            self._schedule_reconnect()
            raise ConnectionError("Failed to get a valid connection from pool") from e

    def _release_connection(self, conn):
        """Release connection back to the pool."""
        if self._pool and conn:
            try:
                self._pool.putconn(conn)
            except psycopg2.Error as e:
                logger.warning(f"Error putting connection back to pool: {e}. Closing it.")
                try:
                    self._pool.putconn(conn, close=True)  # Close potentially broken conn
                except Exception as close_err:
                    logger.error(f"Failed to close connection after putconn error: {close_err}")

    def _build_resource_uri(self, table_name: str, schema_name: str = "public") -> str:
        # (Implementation largely unchanged, added check for initialization)
        if not self.resource_base_url_parts:
            raise RuntimeError("Cannot build resource URI: Handler not fully initialized.")
        netloc_user = self.resource_base_url_parts.username or "user"
        netloc_host = self.resource_base_url_parts.hostname or "host"
        netloc_port = self.resource_base_url_parts.port or 5432
        db_path = self.resource_base_url_parts.path
        if not db_path or not db_path.startswith("/"):
            db_path = f"/{db_path if db_path else 'database'}"
        parts = self.resource_base_url_parts._replace(
            netloc=f"{netloc_user}@{netloc_host}:{netloc_port}",
            path=f"{db_path.rstrip('/')}/{schema_name}/{table_name}/schema",
        )
        return urlunparse(parts._replace(scheme="postgres"))

    # --- MCP Methods (largely unchanged, but use resilient _get_connection) ---

    def handle_list_resources(self) -> ListResourcesResponse:
        # (Implementation unchanged, relies on _get_connection)
        logging.info("Handling mcp_listResources")
        conn = self._get_connection()
        # ... rest of the logic ...
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT LIKE 'pg_%' AND schema_name != 'information_schema'"
                )
                schemas = [row["schema_name"] for row in cur.fetchall()]
                resources = []
                for schema in schemas:
                    cur.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_type = 'BASE TABLE'",
                        (schema,),
                    )
                    tables = cur.fetchall()
                    for row in tables:
                        try:
                            resources.append(
                                ResourceIdentifier(
                                    uri=self._build_resource_uri(row["table_name"], schema),
                                    name=f'"{schema}.{row["table_name"]}" table schema',
                                    mimeType="application/json",
                                )
                            )
                        except Exception as uri_err:
                            logger.warning(
                                f"Could not build URI for {schema}.{row['table_name']}: {uri_err}"
                            )
                return ListResourcesResponse(resources=resources)
        finally:
            self._release_connection(conn)

    def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
        # (Implementation unchanged, relies on _get_connection)
        logging.info(f"Handling mcp_readResource for URI: {params.uri}")
        conn = self._get_connection()
        # ... rest of the logic ...
        try:
            parsed_uri = urlparse(params.uri)
            path_parts = parsed_uri.path.strip("/").split("/")
            if len(path_parts) < 3 or path_parts[-1] != "schema":
                raise ValueError(
                    f"Invalid resource URI path format: {parsed_uri.path}. Expected format like '.../schema_name/table_name/schema'"
                )
            table_name = path_parts[-2]
            schema_name = path_parts[-3]
            if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
                raise ValueError(f"Invalid table name characters extracted: {table_name}")
            if not re.match(r"^[a-zA-Z0-9_]+$", schema_name):
                raise ValueError(f"Invalid schema name characters extracted: {schema_name}")
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                query = sql.SQL(
                    "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position"
                )
                cur.execute(query, (schema_name, table_name))
                columns = cur.fetchall()
                if not columns:
                    check_table_query = sql.SQL(
                        "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s"
                    )
                    cur.execute(check_table_query, (schema_name, table_name))
                    if not cur.fetchone():
                        raise ValueError(f"Table or view '{schema_name}.{table_name}' not found.")
                    else:
                        raise ValueError(
                            f"Table '{schema_name}.{table_name}' found but has no columns or access denied."
                        )
                content_json = json.dumps(columns, indent=2)
                return ReadResourceResponse(
                    contents=[
                        ResourceContent(
                            uri=params.uri, text=content_json, mimeType="application/json"
                        )
                    ]
                )
        finally:
            self._release_connection(conn)

    def handle_list_tools(self) -> ListToolsResponse:
        # (Implementation unchanged)
        logging.info("Handling mcp_listTools")
        # ... rest of the logic ...
        return ListToolsResponse(
            tools=[
                ToolDefinition(
                    name="query",
                    description="Run a read-only SQL query against any accessible schema in the database.",
                    inputSchema=ToolInputSchema(
                        type="object",
                        properties={
                            "sql": {
                                "type": "string",
                                "description": "The read-only SQL query to execute.",
                            },
                        },
                        required=["sql"],
                    ),
                )
            ]
        )

    def handle_call_tool(self, params: CallToolRequestParams) -> CallToolResponse:
        # (Implementation unchanged, relies on _get_connection)
        logging.info(f"Handling mcp_callTool for tool: {params.name}")
        # ... rest of the logic ...
        if params.name != "query":
            return CallToolResponse(
                content=[
                    TextContent(
                        text=f"Error: Unknown tool '{params.name}'. Available tools: 'query'"
                    )
                ],
                isError=True,
            )
        if not params.arguments or "sql" not in params.arguments:
            return CallToolResponse(
                content=[TextContent(text="Error: Missing 'sql' argument for 'query' tool")],
                isError=True,
            )
        sql_query = params.arguments["sql"]
        if not sql_query.strip().upper().startswith(("SELECT", "EXPLAIN", "SHOW", "WITH")):
            return CallToolResponse(
                content=[
                    TextContent(
                        text="Error: Only SELECT, EXPLAIN, SHOW, or WITH (for CTEs) queries are allowed for safety."
                    )
                ],
                isError=True,
            )
        if any(
            keyword in sql_query.upper()
            for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        ):
            return CallToolResponse(
                content=[
                    TextContent(
                        text="Error: SQL query contains modification keywords (INSERT, UPDATE, DELETE, etc.). Only read-only queries are allowed."
                    )
                ],
                isError=True,
            )
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                logging.info("Starting READ ONLY transaction")
                cur.execute("SET TRANSACTION READ ONLY")
                try:
                    logging.info(f"Executing SQL: {sql_query}...")
                    cur.execute(sql.SQL(sql_query))
                    results = cur.fetchall()
                    result_json = json.dumps(results, indent=2, default=str)
                    logging.info(f"Query successful, returning {len(results)} rows.")
                    return CallToolResponse(content=[TextContent(text=result_json)], isError=False)
                except psycopg2.Error as query_error:
                    logging.error(f"SQL execution error: {query_error}", exc_info=True)
                    conn.rollback()
                    return CallToolResponse(
                        content=[TextContent(text=f"SQL Error: {str(query_error).strip()}")],
                        isError=True,
                    )
                except Exception as e:
                    logging.error(
                        f"Unexpected error during query execution or serialization: {e}",
                        exc_info=True,
                    )
                    conn.rollback()
                    return CallToolResponse(
                        content=[TextContent(text=f"Unexpected Execution Error: {str(e)}")],
                        isError=True,
                    )
                finally:
                    try:
                        if not conn.closed:
                            cur.execute("ROLLBACK")
                            logging.debug("Read-only transaction rolled back.")
                    except psycopg2.Error as rb_error:
                        logging.warning(f"Error during explicit rollback: {rb_error}")
        finally:
            self._release_connection(conn)

    def close(self):
        logging.info("Closing PostgreSQL handler resources")
        # Cancel reconnection task if running
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            logger.info("Cancelled pending reconnection task.")
        # Close the connection pool
        if self._pool:
            try:
                self._pool.closeall()
                logger.info("Connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
        self._pool = None


# Request model for JSON-RPC requests (Unchanged)
class MCPRequest(BaseModel):
    jsonrpc: str
    id: Any
    method: str
    params: Optional[Dict[str, Any]] = None


# Global PostgreSQL handler instance
postgres_handler: Optional[PostgresHandler] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the handler during application startup."""
    global postgres_handler
    if postgres_handler:  # Avoid re-initialization
        return

    try:
        settings = PostgresSettings()
        postgres_handler = PostgresHandler(settings)
        # Start the asynchronous initialization process
        await postgres_handler.initialize()
    except Exception as e:
        logger.error(
            f"FATAL: Unhandled exception during startup initialization: {e}", exc_info=True
        )
        # Ensure handler is None if startup fails
        postgres_handler = None


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up handler resources during application shutdown."""
    global postgres_handler
    if postgres_handler:
        postgres_handler.close()
        logging.info("PostgreSQL handler shut down.")


@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    """Handle incoming MCP requests."""
    global postgres_handler

    # Check if handler is initialized and connected
    if not postgres_handler or not postgres_handler.is_connected:
        # Log the specific reason if possible
        error_detail = "PostgreSQL handler not initialized."
        last_err_msg = ""
        if postgres_handler and postgres_handler.last_error:
            last_err_msg = f" Last Error: {postgres_handler.last_error}"
        if postgres_handler and not postgres_handler.is_connected:
            error_detail = f"PostgreSQL handler is not connected.{last_err_msg}"
            # Optionally trigger a reconnect attempt here if desired
            # postgres_handler._schedule_reconnect()
        logger.error(f"MCP request failed: {error_detail}")
        raise HTTPException(status_code=503, detail=error_detail)  # Service Unavailable

    logging.debug(
        f"Received MCP request: method={request.method}, id={request.id}"
    )  # Debug level for requests

    response_data = None
    error_data = None

    try:
        # Use a connection for the duration of the request handling if needed
        # conn = postgres_handler._get_connection() # Get connection here if needed across methods
        # try:
        if request.method == "mcp_listResources":
            response_data = postgres_handler.handle_list_resources().model_dump()
        elif request.method == "mcp_readResource":
            params = ReadResourceRequestParams.model_validate(request.params or {})
            response_data = postgres_handler.handle_read_resource(params).model_dump()
        elif request.method == "mcp_listTools":
            response_data = postgres_handler.handle_list_tools().model_dump()
        elif request.method == "mcp_callTool":
            params = CallToolRequestParams.model_validate(request.params or {})
            tool_response = postgres_handler.handle_call_tool(params)
            response_data = tool_response.model_dump()
            # JSON-RPC error field is typically not set if the tool itself indicates an error in its result
        else:
            error_data = {"code": -32601, "message": f"Unsupported MCP method: {request.method}"}
        # finally:
        # postgres_handler._release_connection(conn) # Release connection if acquired at start

    except ConnectionError as conn_err:
        logger.error(
            f"Database connection error handling request {request.id}: {conn_err}", exc_info=True
        )
        error_data = {"code": -32002, "message": f"Database Connection Error: {str(conn_err)}"}
    except ValueError as val_err:
        logger.warning(f"Validation error handling request {request.id}: {val_err}")
        error_data = {"code": -32602, "message": f"Invalid Parameters: {str(val_err)}"}
    except Exception as e:
        logger.error(f"Unexpected error handling request {request.id}: {e}", exc_info=True)
        error_data = {"code": -32000, "message": f"Internal Server Error: {str(e)}"}

    response = JsonRpcResponse(
        id=request.id, result=response_data if error_data is None else None, error=error_data
    )
    return response.model_dump(exclude_none=True)


@app.get("/health")
async def health_check():
    """Check the health of the service, including DB connection."""
    global postgres_handler

    if not postgres_handler:
        raise HTTPException(status_code=503, detail="PostgreSQL handler not initialized.")

    if not postgres_handler.is_connected:
        last_err = postgres_handler.last_error or "Pool not available or connection lost."
        logger.warning(f"Health check failed: Not connected. Last Error: {last_err}")
        raise HTTPException(status_code=503, detail=f"Database connection unhealthy: {last_err}")

    # Perform a quick connection check using the resilient getter
    conn = None
    try:
        # Get connection will retry based on backoff settings if needed
        conn = postgres_handler._get_connection()
        # Minimal check if needed, _get_connection already does a ping
        # with conn.cursor() as cur:
        #     cur.execute("SELECT 1")
        return {"status": "healthy", "db_connection": "ok"}
    except ConnectionError as e:  # Catch if _get_connection ultimately fails after retries
        logger.error(f"Health check failed: Unable to get connection from pool: {e}")
        raise HTTPException(status_code=503, detail=f"Database connection unhealthy: {str(e)}")
    except Exception as e:  # Catch unexpected errors during health check
        logger.error(f"Health check unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check internal error: {str(e)}")
    finally:
        if conn:
            postgres_handler._release_connection(conn)


if __name__ == "__main__":
    # (Main execution block largely unchanged)
    import uvicorn

    debug_mode = "--debug" in sys.argv
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    uvicorn_log_config["loggers"]["uvicorn.access"]["level"] = log_level
    if debug_mode:
        logger.debug("Debug mode enabled")
    uvicorn.run(
        "__main__:app", host="0.0.0.0", port=8001, reload=debug_mode, log_config=uvicorn_log_config
    )
