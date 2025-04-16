import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse, urlunparse

import asyncpg
import backoff
from confidentialmind_core.config_manager import (
    ConfigManager,
    ConnectorSchema,
    get_api_parameters,
    load_environment,
)
from fastapi import FastAPI
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


# --- Backoff Configuration Function - Moved outside the class to avoid the error ---
def get_backoff_config() -> Dict[str, Any]:
    """Get backoff configuration based on environment"""
    return {
        "max_tries": 5,
        "max_time": 30,
        "on_backoff": lambda details: logger.info(
            f"Reconnection attempt {details['tries']} failed. Retrying in {details['wait']} seconds"
        ),
    }


# --- Pydantic Settings ---
class PostgresSettings(BaseSettings):
    # Default connection settings
    database_host: str = "localhost"
    database_port: int = 5432
    database_user: str = "app"
    database_password: str = "testpass"
    database_name: str = "vector-db"

    # Pool settings
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 30.0
    command_timeout: float = 60.0

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


# --- Database Connection Management (baserag-style) ---
class Database:
    _pool: Optional[asyncpg.pool.Pool] = None
    _connection_error: Optional[str] = None
    _is_connecting: bool = False
    _reconnect_task: Optional[asyncio.Task] = None
    _settings: PostgresSettings = PostgresSettings()
    _current_db_url: Optional[str] = None

    @classmethod
    async def connect(cls, db_url: Optional[str] = None):
        """Establish connection to database with proper locking"""
        if cls._is_connecting:
            logger.info("Connection attempt already in progress")
            return

        if cls._pool is not None:
            return

        cls._is_connecting = True
        try:
            if db_url:
                cls._current_db_url = db_url

            # Build connection string and connect
            connection_string = cls._settings.get_connection_string(cls._current_db_url)

            logger.debug(
                f"Connecting to database with connection string: {cls._settings._get_safe_connection_string(connection_string)}"
            )

            cls._pool = await asyncpg.create_pool(
                dsn=connection_string,
                min_size=cls._settings.min_connections,
                max_size=cls._settings.max_connections,
                timeout=cls._settings.connection_timeout,
                command_timeout=cls._settings.command_timeout,
                server_settings={
                    "statement_timeout": str(int(cls._settings.command_timeout * 1000)),
                    "idle_in_transaction_session_timeout": "10000",
                    "lock_timeout": "2000",
                },
            )

            # Test the connection
            async with cls._pool.acquire() as conn:
                await conn.execute("SELECT 1")

            logger.info("Successfully connected to database")
            cls._connection_error = None

        except Exception as e:
            cls._connection_error = f"Failed to connect to database: {str(e)}"
            logger.error(cls._connection_error)
            cls._pool = None
            cls._schedule_reconnect()
            raise
        finally:
            cls._is_connecting = False

    @classmethod
    async def disconnect(cls):
        """Disconnect from database"""
        if cls._pool:
            logger.info("Disconnecting from database")
            if cls._reconnect_task and not cls._reconnect_task.done():
                cls._reconnect_task.cancel()
            await cls._pool.close()
            cls._pool = None
            logger.info("Successfully disconnected database connections")

    @classmethod
    def pool(cls) -> asyncpg.pool.Pool:
        """Get connection pool or raise error if not initialized"""
        if cls._pool is None:
            raise ValueError("Connection pool is not initialized.")
        return cls._pool

    @classmethod
    def _schedule_reconnect(cls):
        """Schedule a reconnection attempt with exponential backoff"""
        if not cls._reconnect_task or cls._reconnect_task.done():
            cls._reconnect_task = asyncio.create_task(cls._reconnect_with_backoff())

    @classmethod
    @backoff.on_exception(backoff.expo, Exception, **get_backoff_config())
    async def _reconnect_with_backoff(cls):
        """Attempt to reconnect with exponential backoff"""
        await cls.connect()

    @classmethod
    def is_connected(cls) -> bool:
        """Check if database is currently connected"""
        return cls._pool is not None and not cls._connection_error

    @classmethod
    def last_error(cls) -> Optional[str]:
        """Get the last connection error if any"""
        return cls._connection_error

    @classmethod
    async def ensure_connected(cls):
        """Ensure database is connected, waiting for reconnection if necessary"""
        if not cls.is_connected():
            await cls.connect()

    @classmethod
    async def execute_query(cls, query: str, *args, fetch_type: str = "all"):
        """Execute a database query with automatic connection management"""
        await cls.ensure_connected()
        async with cls._pool.acquire() as conn:
            try:
                # Set read-only mode for safety
                await conn.execute("SET TRANSACTION READ ONLY")

                if fetch_type == "all":
                    return await conn.fetch(query, *args)
                elif fetch_type == "row":
                    return await conn.fetchrow(query, *args)
                elif fetch_type == "val":
                    return await conn.fetchval(query, *args)
                else:
                    return await conn.execute(query, *args)
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                raise


# --- Async URL Discovery with Retry (baserag-style) ---
async def fetch_url():
    """
    Try to fetch the database URL from the connector until available.
    Similar to the approach used in baserag.
    """
    # First try from SDK ConfigManager
    try:
        url, _ = get_api_parameters(DATABASE_CONNECTOR_ID)
        if url:
            logger.info(f"Successfully retrieved database URL from ConfigManager: {url}")
            return url
    except Exception as e:
        logger.warning(f"Error fetching database URL from ConfigManager: {e}")

    # Then try environment variables
    env_var_url = os.environ.get("DATABASE_URL")
    if env_var_url:
        logger.info(f"Found DATABASE_URL in environment variables")
        return env_var_url

    # Try individual components
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "vector-db")
    db_user = os.environ.get("DB_USER", "app")
    db_pass = os.environ.get("DB_PASSWORD", "testpass")

    if db_host:
        constructed_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        logger.info(f"Constructed database URL from individual environment variables")
        return constructed_url

    # If all fails, return None and let the caller handle it
    logger.warning("No database connection details found. Will retry later.")
    return None


# --- PostgreSQL Handler Class with baserag-style Database Access ---
class PostgresHandler:
    """Handles interactions with the PostgreSQL database via MCP."""

    def __init__(self, settings: PostgresSettings):
        Database._settings = settings
        self.settings = settings
        self.resource_base_url_parts = None

    @property
    def is_connected(self) -> bool:
        """Returns database connection status"""
        return Database.is_connected()

    @property
    def last_error(self) -> Optional[str]:
        """Returns the last recorded connection error."""
        return Database.last_error()

    async def connect(self, sdk_db_url: Optional[str] = None) -> bool:
        """Connect to database using baserag-style Database class"""
        try:
            logger.info(f"Connecting to database with SDK URL: {sdk_db_url}")

            # Store the database URL parts for resource URI building
            connection_string = self.settings.get_connection_string(sdk_db_url)
            self.resource_base_url_parts = urlparse(connection_string)

            # Attempt connection
            await Database.connect(sdk_db_url)
            return Database.is_connected()

        except Exception as e:
            logger.error(f"Error in connect(): {e}", exc_info=True)
            return False

    async def disconnect(self):
        """Disconnect database connection"""
        await Database.disconnect()

    async def ensure_connected(self) -> bool:
        """Ensure database is connected, attempting reconnection if necessary"""
        try:
            await Database.ensure_connected()
            return Database.is_connected()
        except Exception as e:
            logger.error(f"Error ensuring connection: {e}")
            return False

    # --- MCP Methods ---

    async def handle_list_resources(self) -> ListResourcesResponse:
        """Lists schemas and tables as resources"""
        logger.debug("Handling mcp_listResources")

        try:
            await self.ensure_connected()
            resources = []

            # Get all schemas except system schemas
            schemas_query = """
                SELECT schema_name FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
                  AND schema_name NOT LIKE 'pg_toast%' AND schema_name NOT LIKE 'pg_temp_%'
            """
            schemas = await Database.execute_query(schemas_query)

            # For each schema, get all tables
            for schema_row in schemas:
                schema = schema_row["schema_name"]
                tables_query = """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = $1 AND table_type = 'BASE TABLE'
                """
                tables = await Database.execute_query(tables_query, schema)

                for table_row in tables:
                    table_name = table_row["table_name"]
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
                        logger.warning(f"Could not build URI for {schema}.{table_name}: {uri_err}")

            return ListResourcesResponse(resources=resources)

        except Exception as e:
            logger.error(f"Error listing resources: {e}", exc_info=True)
            raise

    async def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
        """Reads the schema information for a specific table URI."""
        logger.debug(f"Handling mcp_readResource for URI: {params.uri}")

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

            await self.ensure_connected()

            # Get column information
            columns_query = """
                SELECT column_name, data_type, is_nullable, column_default 
                FROM information_schema.columns 
                WHERE table_schema = $1 AND table_name = $2 
                ORDER BY ordinal_position
            """
            columns = await Database.execute_query(columns_query, schema_name, table_name)

            if not columns:
                # Check if table exists
                check_query = """
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = $1 AND table_name = $2
                """
                table_exists = await Database.execute_query(
                    check_query, schema_name, table_name, fetch_type="val"
                )

                if not table_exists:
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
                # Convert to list of dicts for JSON serialization
                columns_list = [dict(row) for row in columns]
                content_json = json.dumps(columns_list, indent=2, default=str)

            return ReadResourceResponse(
                contents=[
                    ResourceContent(uri=params.uri, text=content_json, mimeType="application/json")
                ]
            )

        except ValueError as ve:
            logger.error(f"Validation error reading resource {params.uri}: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error reading resource {params.uri}: {e}", exc_info=True)
            raise

    def handle_list_tools(self) -> ListToolsResponse:
        """List available database tools"""
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

    async def handle_call_tool(self, params: CallToolRequestParams) -> CallToolResponse:
        """Executes the 'query' tool."""
        logger.debug(f"Handling mcp_callTool for tool: {params.name}")

        # Only support 'query' tool for now
        if params.name != "query":
            return CallToolResponse(
                content=[TextContent(text=f"Error: Unknown tool '{params.name}'.")], isError=True
            )

        # Check for required SQL parameter
        if not params.arguments or "sql" not in params.arguments:
            return CallToolResponse(
                content=[TextContent(text="Error: Missing 'sql' argument.")], isError=True
            )

        sql_query = params.arguments["sql"].strip()

        # Security checks for read-only SQL
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

        # More robust check for disallowed keywords
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
            "SET ",
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT",
            "BEGIN",
            "DECLARE",
            "EXECUTE",
            "COPY",
        }

        # Remove SQL comments and check for disallowed keywords
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

        try:
            # Ensure connection is active
            await self.ensure_connected()

            # Execute query
            try:
                # Execute the query and get results
                logger.info(f"Executing read-only SQL: {sql_query[:100]}...")

                # For SELECT queries that return rows
                if sql_upper.startswith(("SELECT", "WITH")):
                    results = await Database.execute_query(sql_query)

                    # Convert results to list of dicts for JSON serialization
                    result_list = [dict(row) for row in results]
                    result_json = json.dumps(result_list, indent=2, default=str)

                    row_count = len(result_list)
                    logger.info(
                        f"Query successful, {row_count} row{'s' if row_count != 1 else ''} returned."
                    )

                # For EXPLAIN, SHOW queries that might return different structure
                else:
                    results = await Database.execute_query(sql_query)
                    if results:
                        # Convert to appropriate format
                        result_list = [dict(row) for row in results]
                        result_json = json.dumps(result_list, indent=2, default=str)
                    else:
                        # Handle case where query succeeded but returned no rows
                        result_json = json.dumps(
                            {"message": "Query executed successfully, no rows returned."}, indent=2
                        )

                return CallToolResponse(content=[TextContent(text=result_json)], isError=False)

            except Exception as query_error:
                logger.error(f"SQL execution error: {query_error}", exc_info=True)

                # Provide specific error message with hints
                pg_err_msg = str(query_error).strip()
                hint = ""

                if "does not exist" in pg_err_msg:
                    hint = " Hint: Check table/schema names and spelling, ensure schema qualification (schema.table)."
                elif "permission denied" in pg_err_msg:
                    hint = " Hint: Check database user permissions."
                elif "timeout" in pg_err_msg:
                    hint = f" Hint: Query exceeded the allowed time limit ({self.settings.command_timeout}s)."

                return CallToolResponse(
                    content=[TextContent(text=f"SQL Execution Error: {pg_err_msg}{hint}")],
                    isError=True,
                )

        except ConnectionError as conn_err:
            # Connection errors
            logger.error(f"Database connection error: {conn_err}")
            return CallToolResponse(
                content=[TextContent(text=f"Database Connection Error: {conn_err}")], isError=True
            )
        except Exception as e:
            # Other errors
            logger.error(f"Error executing tool '{params.name}': {e}", exc_info=True)
            return CallToolResponse(
                content=[TextContent(text=f"Internal Server Error: {e}")],
                isError=True,
            )

    def _build_resource_uri(self, table_name: str, schema_name: str = "public") -> str:
        """Builds a resource URI for a table schema."""
        if not self.resource_base_url_parts:
            raise RuntimeError("Cannot build resource URI: Handler not fully initialized.")

        # Use info parsed from the connection DSN
        scheme = self.resource_base_url_parts.scheme or "postgres"
        netloc = self.resource_base_url_parts.netloc
        db_path = (self.resource_base_url_parts.path or "/database").lstrip("/")

        # Construct the resource path: /database_name/schema_name/table_name/schema
        resource_path = f"/{db_path}/{schema_name}/{table_name}/schema"

        # Create new URL parts based on the original DSN but with the resource path
        uri_parts = self.resource_base_url_parts._replace(path=resource_path, query="", fragment="")

        # Ensure scheme is 'postgres' for the MCP URI standard if DSN was 'postgresql'
        if uri_parts.scheme == "postgresql":
            uri_parts = uri_parts._replace(scheme="postgres")

        return urlunparse(uri_parts)


# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global postgres_handler, postgres_settings
    logger.info("Lifespan startup: Initializing PostgreSQL MCP Server...")

    # Load environment variables
    load_environment()

    try:
        # 1. Load Settings
        postgres_settings = PostgresSettings()
        logger.info("PostgresSettings loaded.")

        # 2. Initialize ConfigManager for SDK integration
        try:
            logger.info("Initializing ConfigManager...")
            config_manager = ConfigManager()  # Get the singleton instance
            connectors_for_cm = [
                ConnectorSchema(
                    type="database", label="Target Database", config_id=DATABASE_CONNECTOR_ID
                )
            ]
            config_manager.init_manager(
                config_model=postgres_settings,
                connectors=connectors_for_cm,
            )
            logger.info("ConfigManager initialized.")
        except Exception as config_err:
            logger.warning(f"ConfigManager initialization warning: {config_err}")
            logger.info("Continuing startup with fallback settings.")

        # 3. Create Handler
        postgres_handler = PostgresHandler(postgres_settings)
        logger.info("PostgresHandler instance created.")

        # 4. Attempt to fetch URL with retry
        db_url = await fetch_url()
        logger.info(f"Initial DB URL fetched: {db_url}")

        # 5. Attempt initial connection
        # Unlike baserag, we don't block startup if connection fails
        connection_result = await postgres_handler.connect(db_url)

        if connection_result:
            logger.info("Successfully established initial database connection.")
        else:
            # We'll just warn and continue - the system will handle reconnection later
            logger.warning(
                "Initial database connection attempt failed. "
                "Server will start and retry connections when needed."
            )
            if postgres_handler.last_error:
                logger.warning(f"Initial connection failure reason: {postgres_handler.last_error}")

    except Exception as e:
        logger.error(f"Error during PostgreSQL MCP Server startup: {e}", exc_info=True)
        # Continue anyway, connections will be retried when needed
        if "postgres_handler" not in locals():
            postgres_handler = None
        if "postgres_settings" not in locals():
            postgres_settings = PostgresSettings()

    # --- Yield control to the application ---
    yield

    # --- Shutdown ---
    logger.info("Lifespan shutdown: Cleaning up PostgreSQL MCP Server resources...")
    if postgres_handler:
        await postgres_handler.disconnect()
    logger.info("Lifespan shutdown complete.")


# Create FastAPI app with the lifespan manager
app = FastAPI(title="PostgreSQL MCP Server", lifespan=lifespan)


# Request model for JSON-RPC
class MCPRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Any
    method: str
    params: Optional[Dict[str, Any]] = None


# --- API Endpoints ---
@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    """Handles incoming JSON-RPC requests for MCP methods."""
    if not postgres_handler:
        logger.error("MCP request rejected: PostgreSQL handler not initialized.")
        error_data = {"code": -32000, "message": "Service Unavailable: Handler not initialized"}
        return JsonRpcResponse(id=request.id, error=error_data).model_dump(exclude_none=True)

    logging.info(f"Received MCP request: method={request.method}, id={request.id}")
    response_data = None
    error_data = None

    try:
        # Dispatch request to appropriate handler
        if request.method == "mcp_listResources":
            response_data = (await postgres_handler.handle_list_resources()).model_dump()
        elif request.method == "mcp_readResource":
            params = ReadResourceRequestParams.model_validate(request.params or {})
            response_data = (await postgres_handler.handle_read_resource(params)).model_dump()
        elif request.method == "mcp_listTools":
            response_data = postgres_handler.handle_list_tools().model_dump()
        elif request.method == "mcp_callTool":
            params = CallToolRequestParams.model_validate(request.params or {})
            tool_response = await postgres_handler.handle_call_tool(params)

            # Check if the tool call itself resulted in an error
            if tool_response.isError:
                error_message = "Tool execution failed."
                if tool_response.content and isinstance(tool_response.content[0], TextContent):
                    error_message = tool_response.content[0].text
                error_data = {
                    "code": -32000,
                    "message": error_message,
                }
            else:
                response_data = tool_response.model_dump()
        else:
            error_data = {"code": -32601, "message": f"Method not found: '{request.method}'"}

    except Exception as e:
        logger.error(f"Error handling request {request.id}: {e}", exc_info=True)
        error_data = {"code": -32000, "message": f"Internal server error: {str(e)}"}

    # Create JSON-RPC response
    response = JsonRpcResponse(
        id=request.id, result=response_data if error_data is None else None, error=error_data
    )
    return response.model_dump(exclude_none=True)


@app.get("/health")
async def health_check():
    """Performs a health check on the service and its database connection."""
    health_status = {"status": "healthy", "db_connection": "unknown", "version": "1.0.0"}

    if not postgres_handler:
        health_status["status"] = "unhealthy"
        health_status["db_connection"] = "handler_not_initialized"
        return health_status

    # Check connection status
    if postgres_handler.is_connected:
        health_status["db_connection"] = "connected"
    else:
        # Unlike before, we don't report unhealthy just because DB is disconnected
        # This is a more resilient approach like in baserag
        health_status["db_connection"] = "disconnected"
        health_status["connection_details"] = {
            "retrying": True,
            "last_error": postgres_handler.last_error,
        }

    return health_status


# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn

    debug_mode = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")
    log_level = logging.DEBUG if debug_mode else logging.INFO

    logging.basicConfig(level=log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logging.getLogger("postgres-mcp").setLevel(log_level)

    logger.info(f"Starting PostgreSQL MCP Server (Debug Mode: {debug_mode})")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=debug_mode,
        log_level="debug" if debug_mode else "info",
    )
