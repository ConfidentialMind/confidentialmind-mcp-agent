import asyncio
import json

import asyncpg
from fastmcp import Client
from fastmcp.exceptions import ClientError
from mcp.types import TextContent, TextResourceContents

from .logger import logger


async def create_test_tables(dsn: str) -> None:
    """Create test tables for MCP server testing."""
    logger.info("Setting up test database...")

    # Connect directly to the database
    conn = await asyncpg.connect(dsn)

    try:
        # Check if tables already exist
        table_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name IN ('test_users', 'test_products');
        """)

        if table_count > 0:
            logger.info("Test tables already exist, dropping them first...")
            await conn.execute("DROP TABLE IF EXISTS test_users, test_products;")

        # Create test_users table
        await conn.execute("""
            CREATE TABLE test_users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                email VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Insert sample data
        await conn.execute("""
            INSERT INTO test_users (username, email) VALUES 
            ('user1', 'user1@example.com'),
            ('user2', 'user2@example.com'),
            ('user3', 'user3@example.com');
        """)

        # Create test_products table
        await conn.execute("""
            CREATE TABLE test_products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                description TEXT
            );
        """)

        # Insert sample data
        await conn.execute("""
            INSERT INTO test_products (name, price, description) VALUES 
            ('Product 1', 19.99, 'This is product 1'),
            ('Product 2', 29.99, 'This is product 2'),
            ('Product 3', 39.99, 'This is product 3');
        """)

        logger.info("Test tables created successfully")
    finally:
        await conn.close()


async def cleanup_test_tables(dsn: str) -> None:
    """Remove test tables after testing is complete."""
    logger.info("Cleaning up test database...")

    # Connect directly to the database
    conn = await asyncpg.connect(dsn)

    try:
        # Drop test tables
        await conn.execute("DROP TABLE IF EXISTS test_users, test_products;")
        logger.info("Test tables removed successfully")
    except Exception as e:
        logger.error(f"Error cleaning up test tables: {e}")
    finally:
        await conn.close()


async def run():
    """Test the Postgres MCP server using FastMCP Client."""
    server_url = "http://localhost:8080/sse"
    database_dsn = "postgresql://postgres:postgres@localhost:5432/test_db"

    try:
        # Create test tables before running tests
        await create_test_tables(database_dsn)

        logger.info("Connecting to Postgres MCP server", url=server_url, client="FastMCP")

        # Create the FastMCP Client context manager. It handles connection and initialization.
        async with Client(server_url) as client:
            logger.info("Connection established and initialized with FastMCP Client")

            # List available prompts
            prompts_result = await client.list_prompts_mcp()
            logger.info("Available prompts", prompts=[p.name for p in prompts_result.prompts])

            # List available tools
            tools_response = await client.list_tools()
            logger.info("Available tools", tools=[tool.name for tool in tools_response])

            # Check if the expected tool is available
            has_execute_sql = any(tool.name == "execute_sql" for tool in tools_response)
            if not has_execute_sql:
                logger.error("'execute_sql' tool is not available on the server")
                return

            # List available resources
            resources_response = await client.list_resources()
            logger.info("Available resources", resources=[r.uri for r in resources_response])

            # Check if our expected resource is available
            expected_resource = "postgres://schemas"
            available_resources_uris = [str(r.uri) for r in resources_response]

            if expected_resource not in available_resources_uris:
                logger.error("Expected resource not available", resource=expected_resource)
                return

            # Test accessing the schemas resource
            logger.info("Testing access to resource", resource=expected_resource)
            try:
                schema_response_contents = await client.read_resource(expected_resource)

                if schema_response_contents:
                    content_item = schema_response_contents[0]
                    # Check if it's TextResourceContents and has text
                    if isinstance(content_item, TextResourceContents) and hasattr(
                        content_item, "text"
                    ):
                        try:
                            schemas_data = json.loads(content_item.text)
                            logger.info(
                                "Successfully retrieved schemas", table_count=len(schemas_data)
                            )
                            # Log first few tables
                            for i, (table_key, columns) in enumerate(schemas_data.items()):
                                logger.info(
                                    "Table schema", table=table_key, column_count=len(columns)
                                )
                                if i >= 2 and len(schemas_data) > 3:
                                    logger.info(
                                        "Additional tables not shown",
                                        remaining_count=len(schemas_data) - 3,
                                    )
                                    break
                        except json.JSONDecodeError:
                            logger.error(
                                "Error parsing schemas JSON",
                                content_preview=content_item.text[:100],
                            )
                    else:
                        logger.error(
                            "Received non-text or unexpected content for schemas",
                            content_type=type(content_item),
                        )
                else:
                    logger.warning("Schemas resource returned no content")

            except Exception as e:
                logger.error("Error accessing schemas resource", error=str(e))

            # Test executing a SQL query for PostgreSQL version
            logger.info("Testing 'execute_sql' tool with version query")
            try:
                query_result_content = await client.call_tool(
                    "execute_sql", {"sql_query": "SELECT version() AS version"}
                )

                if query_result_content:
                    content = query_result_content[0]
                    if isinstance(content, TextContent) and hasattr(content, "text"):
                        try:
                            version_data = json.loads(content.text)
                            if isinstance(version_data, list) and len(version_data) > 0:
                                logger.info(
                                    "Query executed successfully",
                                    version=version_data[0].get("version", "Unknown"),
                                )
                            else:
                                logger.info("Query executed successfully", result=version_data)
                        except json.JSONDecodeError:
                            logger.error(
                                "Error parsing query result JSON",
                                content_preview=content.text[:100],
                            )
                    else:
                        logger.warning(
                            "Query executed but returned non-text content",
                            content_type=type(content),
                        )
                else:
                    logger.warning("Query executed but no content returned")

            except ClientError as e:
                logger.error("Tool call 'execute_sql' (version) failed", error=str(e))
            except Exception as e:
                logger.error("Error executing SQL version query", error=str(e))

            # Try a query to list tables in the database
            logger.info("Testing 'execute_sql' tool with tables query")
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
                                logger.info("Found tables", count=len(tables_data))
                                for i, table in enumerate(tables_data):
                                    schema = table.get("table_schema", "unknown")
                                    name = table.get("table_name", "unknown")
                                    logger.info("Table", schema=schema, name=name)
                                    if i >= 9:  # Limit output display
                                        break
                            else:
                                logger.warning(
                                    "Tables query returned unexpected format", result=tables_data
                                )
                        except json.JSONDecodeError:
                            logger.error(
                                "Error parsing tables result JSON",
                                content_preview=content.text[:100],
                            )
                    else:
                        logger.warning(
                            "Tables query executed but returned non-text content",
                            content_type=type(content),
                        )
                else:
                    logger.warning("Tables query executed but no content returned")

            except ClientError as e:
                logger.error("Tool call 'execute_sql' (tables) failed", error=str(e))
            except Exception as e:
                logger.error("Error executing tables query", error=str(e))

            # Test a query against our test tables
            logger.info("Testing query against test_users table")
            try:
                users_query = "SELECT * FROM test_users LIMIT 5"
                users_result_content = await client.call_tool(
                    "execute_sql", {"sql_query": users_query}
                )

                if users_result_content:
                    content = users_result_content[0]
                    if isinstance(content, TextContent) and hasattr(content, "text"):
                        try:
                            users_data = json.loads(content.text)
                            if isinstance(users_data, list):
                                logger.info("Found test users", count=len(users_data))
                                for user in users_data:
                                    logger.info("User", username=user.get("username", "unknown"))
                            else:
                                logger.warning(
                                    "Test users query returned unexpected format", result=users_data
                                )
                        except json.JSONDecodeError:
                            logger.error(
                                "Error parsing test users JSON",
                                content_preview=content.text[:100],
                            )
                    else:
                        logger.warning(
                            "Test users query executed but returned non-text content",
                            content_type=type(content),
                        )
                else:
                    logger.warning("Test users query executed but no content returned")

            except ClientError as e:
                logger.error("Tool call for test_users query failed", error=str(e))
            except Exception as e:
                logger.error("Error executing test_users query", error=str(e))

            # Try a query with EXPLAIN
            logger.info("Testing SQL query validation with EXPLAIN")
            try:
                explain_query = """
                EXPLAIN (FORMAT JSON)
                SELECT 
                    table_name, 
                    column_name
                FROM 
                    information_schema.columns
                WHERE 
                    table_schema = 'public'
                LIMIT 5
                """

                explain_result_content = await client.call_tool(
                    "execute_sql", {"sql_query": explain_query}
                )

                if explain_result_content and isinstance(explain_result_content[0], TextContent):
                    logger.info("EXPLAIN query executed successfully")
                else:
                    logger.warning("EXPLAIN query returned unexpected or no content")

            except ClientError as e:
                logger.error("Tool call 'execute_sql' (EXPLAIN) failed", error=str(e))
            except Exception as e:
                logger.error("Error executing EXPLAIN query", error=str(e))

            # Test read-only validation by attempting to execute a write query
            logger.info("Testing read-only validation with a write query (should fail)")
            try:
                write_query = (
                    "INSERT INTO test_users (username, email) VALUES ('test', 'test@example.com')"
                )
                await client.call_tool("execute_sql", {"sql_query": write_query})
                logger.warning("Write query did not fail as expected!")

            except ClientError as e:
                logger.info("Write query properly rejected by tool", error=str(e))
                logger.info("Read-only validation working correctly on server-side")
            except Exception as e:
                logger.error(
                    "An unexpected error occurred during the write query attempt", error=str(e)
                )

            # Test a query with disallowed keywords in a comment (should still be rejected)
            logger.info(
                "Testing SQL validation with disallowed keywords in a comment (should fail)"
            )
            try:
                sneaky_query = """
                SELECT 1 
                -- INSERT INTO test_table VALUES (1)
                """
                await client.call_tool("execute_sql", {"sql_query": sneaky_query})
                logger.warning("Comment with disallowed keywords did not fail as expected!")

            except ClientError as e:
                logger.info(
                    "Query with disallowed keywords in comment properly rejected", error=str(e)
                )
            except Exception as e:
                logger.error("An unexpected error occurred with the comment test", error=str(e))

    except Exception as e:
        logger.critical("An unexpected error occurred", error_type=type(e).__name__, error=str(e))
    finally:
        # Always clean up test tables, even if tests fail
        try:
            await cleanup_test_tables(database_dsn)
        except Exception as e:
            logger.error(f"Failed to clean up test tables: {e}")


if __name__ == "__main__":
    asyncio.run(run())
