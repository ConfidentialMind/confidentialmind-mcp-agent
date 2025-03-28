#!/usr/bin/env python
# postgres_mcp_server.py
import json
import logging
import sys
from urllib.parse import urlparse, urlunparse

import psycopg2
import psycopg2.extras
from psycopg2 import sql

from src.mcp.mcp_protocol import (
    CallToolRequestParams,
    CallToolResponse,
    JsonRpcRequest,
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [MCP Server] %(levelname)s: %(message)s"
)

# --- Database Logic (Similar to PostgreSQLMCP) ---


class PostgresHandler:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._conn = None
        self._pool = psycopg2.pool.SimpleConnectionPool(1, 5, dsn=database_url)
        self.resource_base_url_parts = urlparse(database_url)
        logging.info(f"Initialized PostgresHandler for {self.resource_base_url_parts.path}")

    def _get_connection(self):
        return self._pool.getconn()

    def _release_connection(self, conn):
        self._pool.putconn(conn)

    def _build_resource_uri(self, table_name: str) -> str:
        # Build URI like postgres://user@host:port/db_name/table_name/schema
        # Clear password for safety in URI
        parts = self.resource_base_url_parts._replace(
            netloc=f"{self.resource_base_url_parts.username}@{self.resource_base_url_parts.hostname}:{self.resource_base_url_parts.port or 5432}",
            path=f"{self.resource_base_url_parts.path or ''}/{table_name}/schema",  # Append /table/schema
        )
        return urlunparse(parts._replace(scheme="postgres"))  # Use postgres: scheme

    def handle_list_resources(self) -> ListResourcesResponse:
        logging.info("Handling mcp_listResources")
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Assuming public schema for simplicity, like the TS example
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
                rows = cur.fetchall()
                resources = [
                    ResourceIdentifier(
                        uri=self._build_resource_uri(row["table_name"]),
                        name=f'"{row["table_name"]}" table schema (public schema)',
                        mimeType="application/json",
                    )
                    for row in rows
                ]
                return ListResourcesResponse(resources=resources)
        finally:
            self._release_connection(conn)

    def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
        logging.info(f"Handling mcp_readResource for URI: {params.uri}")
        conn = self._get_connection()
        try:
            parsed_uri = urlparse(params.uri)
            path_parts = parsed_uri.path.strip("/").split("/")

            # Expected path: /db_name/table_name/schema (or just /table_name/schema if db is implicit)
            if len(path_parts) < 2 or path_parts[-1] != "schema":
                raise ValueError(f"Invalid resource URI path format: {parsed_uri.path}")

            table_name = path_parts[-2]
            # Simple validation, production code needs more robustness
            if not table_name.isalnum() and "_" not in table_name:
                raise ValueError(f"Invalid table name extracted: {table_name}")

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s "
                    "ORDER BY ordinal_position",
                    (table_name,),
                )
                columns = cur.fetchall()
                if not columns:
                    raise ValueError(f"Table 'public.{table_name}' not found or has no columns.")

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
                    description="Run a read-only SQL query against the 'public' schema.",
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
                    logging.info(f"Executing SQL: {sql_query[:100]}...")
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


# --- Main Server Loop (Simplified Stdio JSON-RPC) ---


def run_server(db_url: str):
    handler = PostgresHandler(db_url)
    logging.info("MCP Server Ready. Waiting for JSON-RPC requests on stdin...")

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                logging.info("Stdin closed, shutting down.")
                break

            logging.debug(f"Received line: {line.strip()}")
            try:
                request_data = json.loads(line)
                request = JsonRpcRequest.model_validate(request_data)
                logging.info(f"Processing request ID {request.id}, Method: {request.method}")

                response_data = None
                error_data = None

                try:
                    if request.method == "mcp_listResources":
                        response_data = handler.handle_list_resources().model_dump()
                    elif request.method == "mcp_readResource":
                        params = ReadResourceRequestParams.model_validate(request.params or {})
                        response_data = handler.handle_read_resource(params).model_dump()
                    elif request.method == "mcp_listTools":
                        response_data = handler.handle_list_tools().model_dump()
                    elif request.method == "mcp_callTool":
                        params = CallToolRequestParams.model_validate(request.params or {})
                        response_data = handler.handle_call_tool(params).model_dump()
                    else:
                        raise ValueError(f"Unsupported MCP method: {request.method}")

                except Exception as e:
                    logging.error(f"Error handling request {request.id}: {e}", exc_info=True)
                    error_data = {"code": -32000, "message": str(e)}

                response = JsonRpcResponse(id=request.id, result=response_data, error=error_data)

            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON: {line.strip()}")
                response = JsonRpcResponse(
                    id=None, error={"code": -32700, "message": "Parse error"}
                )
            except Exception as e:  # Catch validation errors etc.
                logging.error(f"General error processing input line: {e}", exc_info=True)
                # Try to get request ID if possible, otherwise use null
                req_id = request.id if "request" in locals() and hasattr(request, "id") else None
                response = JsonRpcResponse(
                    id=req_id,
                    error={"code": -32600, "message": f"Invalid Request: {e}"},
                )

            response_json = response.model_dump_json(exclude_none=True)
            logging.debug(f"Sending response: {response_json}")
            print(response_json, flush=True)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down.")
    finally:
        handler.close()
        logging.info("Server shutdown complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python postgres_mcp_server.py <database_url>", file=sys.stderr)
        sys.exit(1)
    database_url_arg = sys.argv[1]
    run_server(database_url_arg)
