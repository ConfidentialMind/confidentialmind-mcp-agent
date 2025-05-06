import asyncio
import json

import httpx
from fastmcp import Client
from fastmcp.exceptions import ClientError
from mcp.types import TextContent, TextResourceContents


async def run():
    """Test the Postgres MCP server using FastMCP Client."""
    # Assuming the server is running on localhost:8000
    server_url = "http://localhost:8080/sse"

    try:
        print(f"Connecting to Postgres MCP server at {server_url} using FastMCP...")

        # Create the FastMCP Client context manager. It handles connection and initialization.
        async with Client(server_url) as client:
            print("Connection established and initialized with FastMCP Client!")

            # List available prompts (Note: FastMCP Client's list_prompts returns mcp.types.ListPromptsResult)
            # For simplicity, let's assume we mainly care about the list itself.
            prompts_result = await client.list_prompts_mcp()
            print(f"Available prompts: {[p.name for p in prompts_result.prompts]}")

            # List available tools
            tools_response = await client.list_tools()
            print(f"Available tools: {[tool.name for tool in tools_response]}")

            # Check if the expected tool is available
            has_execute_sql = any(tool.name == "execute_sql" for tool in tools_response)
            if not has_execute_sql:
                print("\nERROR: 'execute_sql' tool is not available on the server")
                return

            # List available resources
            resources_response = await client.list_resources()
            print(f"Available resources: {[r.uri for r in resources_response]}")

            # Check if our expected resource is available
            expected_resource = "postgres://schemas"
            # FastMCP client's list_resources returns list[Resource], check URIs
            available_resources_uris = [str(r.uri) for r in resources_response]

            if expected_resource not in available_resources_uris:
                print(f"\nERROR: '{expected_resource}' resource is not available on the server")
                return

            # Test accessing the schemas resource
            print(f"\nTesting access to '{expected_resource}' resource...")
            try:
                # read_resource returns list[TextResourceContents | BlobResourceContents]
                schema_response_contents = await client.read_resource(expected_resource)

                if schema_response_contents:
                    content_item = schema_response_contents[0]
                    # Check if it's TextResourceContents and has text
                    if isinstance(content_item, TextResourceContents) and hasattr(
                        content_item, "text"
                    ):
                        try:
                            schemas_data = json.loads(content_item.text)
                            print(f"Successfully retrieved schemas for {len(schemas_data)} tables")
                            # Print first few tables
                            for i, (table_key, columns) in enumerate(schemas_data.items()):
                                print(f"  - {table_key} ({len(columns)} columns)")
                                if i >= 2 and len(schemas_data) > 3:
                                    print(f"  ... and {len(schemas_data) - 3} more tables")
                                    break
                        except json.JSONDecodeError:
                            print(f"Error parsing schemas JSON: {content_item.text[:100]}...")
                    else:
                        print(
                            f"Received non-text or unexpected content for schemas: {type(content_item)}"
                        )
                else:
                    print("Schemas resource returned no content.")

            except Exception as e:
                print(f"Error accessing schemas resource: {e}")

            # Test executing a SQL query
            print("\nTesting 'execute_sql' tool with version query...")
            try:
                # call_tool returns list[TextContent | ImageContent | ...]
                query_result_content = await client.call_tool(
                    "execute_sql", {"sql_query": "SELECT version() AS version"}
                )

                if query_result_content:
                    content = query_result_content[0]
                    if isinstance(content, TextContent) and hasattr(content, "text"):
                        try:
                            version_data = json.loads(content.text)
                            if isinstance(version_data, list) and len(version_data) > 0:
                                print(
                                    f"Query executed successfully: {version_data[0].get('version', 'Unknown')}"
                                )
                            else:
                                print(f"Query executed successfully: {version_data}")
                        except json.JSONDecodeError:
                            print(f"Error parsing query result JSON: {content.text[:100]}...")
                    else:
                        print(f"Query executed but returned non-text content: {type(content)}")
                else:
                    print("Query executed but no content returned")

            # Catch specific FastMCP ClientError for tool execution issues
            except ClientError as e:
                print(f"Tool call 'execute_sql' (version) failed: {e}")
            except Exception as e:
                print(f"Error executing SQL version query: {e}")

            # Try a query to list tables in the database
            print("\nTesting 'execute_sql' tool with tables query...")
            try:
                tables_query = """
                SELECT
                    table_schema,
                    table_name
                FROM
                    information_schema.tables
                WHERE
                    table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY
                    table_schema,
                    table_name
                LIMIT 10
                """

                tables_result_content = await client.call_tool(
                    "execute_sql", {"sql_query": tables_query}
                )

                if tables_result_content:
                    content = tables_result_content[0]
                    if isinstance(content, TextContent) and hasattr(content, "text"):
                        try:
                            tables_data = json.loads(content.text)
                            if isinstance(tables_data, list):
                                print(f"Found {len(tables_data)} tables:")
                                for i, table in enumerate(tables_data):
                                    schema = table.get("table_schema", "unknown")
                                    name = table.get("table_name", "unknown")
                                    print(f"  - {schema}.{name}")
                                    if i >= 9:  # Limit output display
                                        break
                            else:
                                print(f"Tables query returned unexpected format: {tables_data}")
                        except json.JSONDecodeError:
                            print(f"Error parsing tables result JSON: {content.text[:100]}...")
                    else:
                        print(
                            f"Tables query executed but returned non-text content: {type(content)}"
                        )
                else:
                    print("Tables query executed but no content returned")

            except ClientError as e:
                print(f"Tool call 'execute_sql' (tables) failed: {e}")
            except Exception as e:
                print(f"Error executing tables query: {e}")

            # Test read-only validation by attempting to execute a write query
            print("\nTesting read-only validation with a write query (should fail)...")
            try:
                write_query = "INSERT INTO test_table (id) VALUES (1)"
                await client.call_tool("execute_sql", {"sql_query": write_query})
                print("WARNING: Write query did not fail as expected!")

            except ClientError as e:
                print(f"Write query properly rejected by tool: {e}")
                print("Read-only validation likely working correctly on server-side.")
            except Exception as e:
                # Catch other potential errors during the call
                print(f"An unexpected error occurred during the write query attempt: {e}")

    # Keep original HTTP and connection error handling
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error connecting to server: {e}")
        print(f"Status code: {e.response.status_code}")
        print(f"Response body: {e.response.text}")
    except httpx.ConnectError:
        print(f"Connection Error: Could not connect to server at {server_url}")
        print("Make sure the server is running and the URL is correct.")
    except Exception as e:
        # Catch-all for other potential errors during client setup or connection
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(run())
