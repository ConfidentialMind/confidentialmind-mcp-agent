import json
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import psycopg2
import psycopg2.extras
import psycopg2.pool
from fastapi import FastAPI, HTTPException
from psycopg2 import sql
from pydantic import BaseModel

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


# PostgreSQL handler class
class PostgresHandler:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool = psycopg2.pool.SimpleConnectionPool(1, 5, dsn=database_url)
        self.resource_base_url_parts = urlparse(database_url)
        logging.info(f"Initialized PostgresHandler for {self.resource_base_url_parts.path}")

    def _get_connection(self):
        return self._pool.getconn()

    def _release_connection(self, conn):
        self._pool.putconn(conn)

    def _build_resource_uri(self, table_name: str, schema_name: str = "public") -> str:
        # Build URI like postgres://user@host:port/db_name/schema_name/table_name/schema
        parts = self.resource_base_url_parts._replace(
            netloc=f"{self.resource_base_url_parts.username}@{self.resource_base_url_parts.hostname}:{self.resource_base_url_parts.port or 5432}",
            path=f"{self.resource_base_url_parts.path or ''}/{schema_name}/{table_name}/schema",
        )
        return urlunparse(parts._replace(scheme="postgres"))

    def handle_list_resources(self) -> ListResourcesResponse:
        logging.info("Handling mcp_listResources")
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # List schemas (excluding system schemas)
                cur.execute(
                    "SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name NOT LIKE 'pg_%' AND schema_name != 'information_schema'"
                )
                schemas = [row["schema_name"] for row in cur.fetchall()]

                resources = []
                # For each schema, list its tables
                for schema in schemas:
                    cur.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
                        (schema,),
                    )
                    tables = cur.fetchall()
                    for row in tables:
                        resources.append(
                            ResourceIdentifier(
                                uri=self._build_resource_uri(row["table_name"], schema),
                                name=f'"{schema}.{row["table_name"]}" table schema',
                                mimeType="application/json",
                            )
                        )
                return ListResourcesResponse(resources=resources)
        finally:
            self._release_connection(conn)

    def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
        logging.info(f"Handling mcp_readResource for URI: {params.uri}")
        conn = self._get_connection()
        try:
            parsed_uri = urlparse(params.uri)
            path_parts = parsed_uri.path.strip("/").split("/")

            # Expected path: /db_name/schema_name/table_name/schema
            if len(path_parts) < 3 or path_parts[-1] != "schema":
                raise ValueError(f"Invalid resource URI path format: {parsed_uri.path}")

            table_name = path_parts[-2]
            schema_name = path_parts[-3]

            # Simple validation
            if not table_name.isalnum() and "_" not in table_name:
                raise ValueError(f"Invalid table name extracted: {table_name}")
            if not schema_name.isalnum() and "_" not in schema_name:
                raise ValueError(f"Invalid schema name extracted: {schema_name}")

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s "
                    "ORDER BY ordinal_position",
                    (schema_name, table_name),
                )
                columns = cur.fetchall()
                if not columns:
                    raise ValueError(
                        f"Table '{schema_name}.{table_name}' not found or has no columns."
                    )

                content_json = json.dumps(columns, indent=2)
                return ReadResourceResponse(
                    contents=[
                        ResourceContent(
                            uri=params.uri,
                            text=content_json,
                            mimeType="application/json",
                        )
                    ]
                )
        finally:
            self._release_connection(conn)

    def handle_list_tools(self) -> ListToolsResponse:
        logging.info("Handling mcp_listTools")
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
        logging.info(f"Handling mcp_callTool for tool: {params.name}")
        if params.name != "query":
            raise ValueError(f"Unknown tool: {params.name}")

        if not params.arguments or "sql" not in params.arguments:
            raise ValueError("Missing 'sql' argument for 'query' tool")

        sql_query = params.arguments["sql"]
        # Basic validation: Disallow obviously non-SELECT queries (can be bypassed, needs robust parser)
        if not sql_query.strip().upper().startswith(("SELECT", "EXPLAIN", "SHOW")):
            raise ValueError("Only SELECT, EXPLAIN, or SHOW queries are allowed for safety.")

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                logging.info("Starting READ ONLY transaction")
                cur.execute("BEGIN TRANSACTION READ ONLY")
                try:
                    logging.info(f"Executing SQL: {sql_query}...")
                    cur.execute(sql.SQL(sql_query))  # Use sql.SQL for safety if params were allowed
                    results = cur.fetchall()
                    result_json = json.dumps(
                        results, indent=2, default=str
                    )  # Handle potential non-serializable types
                    logging.info(f"Query successful, returning {len(results)} rows.")
                    return CallToolResponse(content=[TextContent(text=result_json)], isError=False)
                except Exception as query_error:
                    logging.error(f"SQL execution error: {query_error}", exc_info=True)
                    conn.rollback()  # Ensure rollback on error within transaction
                    # Return error details in the standard MCP format
                    return CallToolResponse(
                        content=[TextContent(text=f"SQL Error: {str(query_error)}")],
                        isError=True,
                    )
                finally:
                    # Always rollback even if successful, as it was read-only
                    logging.info("Rolling back transaction")
                    cur.execute("ROLLBACK")
        finally:
            self._release_connection(conn)

    def close(self):
        logging.info("Closing connection pool")
        self._pool.closeall()


# Request model for JSON-RPC requests
class MCPRequest(BaseModel):
    jsonrpc: str
    id: Any
    method: str
    params: Optional[Dict[str, Any]] = None


# Global PostgreSQL handler instance
postgres_handler = None


@app.on_event("startup")
async def startup_event():
    global postgres_handler

    # Get database URL from environment variable
    database_url = os.environ.get("PG_CONNECTION_STRING")
    if not database_url:
        raise RuntimeError("PG_CONNECTION_STRING environment variable is not set")

    # Initialize PostgreSQL handler
    try:
        postgres_handler = PostgresHandler(database_url)
        logging.info("PostgreSQL handler initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize PostgreSQL handler: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    global postgres_handler
    if postgres_handler:
        postgres_handler.close()
        logging.info("PostgreSQL handler closed")


@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    global postgres_handler

    if not postgres_handler:
        raise HTTPException(status_code=500, detail="PostgreSQL handler not initialized")

    logging.info(f"Received MCP request: method={request.method}, id={request.id}")

    response_data = None
    error_data = None

    try:
        if request.method == "mcp_listResources":
            response_data = postgres_handler.handle_list_resources().model_dump()
        elif request.method == "mcp_readResource":
            params = ReadResourceRequestParams.model_validate(request.params or {})
            response_data = postgres_handler.handle_read_resource(params).model_dump()
        elif request.method == "mcp_listTools":
            response_data = postgres_handler.handle_list_tools().model_dump()
        elif request.method == "mcp_callTool":
            params = CallToolRequestParams.model_validate(request.params or {})
            response_data = postgres_handler.handle_call_tool(params).model_dump()
        else:
            raise ValueError(f"Unsupported MCP method: {request.method}")
    except Exception as e:
        logging.error(f"Error handling request {request.id}: {e}", exc_info=True)
        error_data = {"code": -32000, "message": str(e)}

    # Create JSON-RPC response
    response = JsonRpcResponse(id=request.id, result=response_data, error=error_data)
    return response.model_dump(exclude_none=True)


@app.get("/health")
async def health_check():
    global postgres_handler

    if not postgres_handler:
        raise HTTPException(status_code=500, detail="PostgreSQL handler not initialized")

    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
