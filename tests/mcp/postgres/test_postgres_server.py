"""Integration tests for PostgreSQL MCP server.

These tests check the high-level behavior of the MCP server resources and tools
without mocking the database. They require a running PostgreSQL instance.
"""

import pytest

from src.mcp.postgres.database import DatabaseManager
from src.mcp.postgres.server import (
    describe_table,
    execute_query,
    get_database_info,
    get_table_schema_resource,
    list_tables,
)
from src.mcp.postgres.settings import PostgresSettings


# Mock Context that emulates the MCP Context for testing
class TestContext:
    """Test implementation of MCP Context."""

    async def info(self, message: str) -> None:
        """Log an info message."""
        pass

    async def error(self, message: str) -> None:
        """Log an error message."""
        pass

    async def report_progress(self, progress: int, total: int) -> None:
        """Report progress."""
        pass


@pytest.fixture
def settings() -> PostgresSettings:
    """Create test settings from environment variables."""
    return PostgresSettings()


@pytest.fixture
async def db_manager(settings: PostgresSettings) -> DatabaseManager:
    """Create and return a database manager."""
    manager = DatabaseManager(settings)
    yield manager
    # Cleanup
    await manager.close()


@pytest.mark.asyncio
class TestServerResources:
    """Test MCP server resources."""

    @pytest.fixture(scope="function")
    async def test_tables(self, db_manager: DatabaseManager):
        """Create test tables and data."""
        pool = await db_manager.get_connection_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_products (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    price DECIMAL(10, 2) NOT NULL,
                    category VARCHAR(50)
                )
            """)

            await conn.execute("""
                INSERT INTO test_products (name, price, category) VALUES
                ('Test Product 1', 19.99, 'Electronics'),
                ('Test Product 2', 29.99, 'Books')
            """)

        yield

        # Clean up
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS test_products")

    async def test_list_tables(self, db_manager: DatabaseManager, test_tables) -> None:
        """Test that list_tables returns a list of database tables."""
        ctx = TestContext()
        table_list = await list_tables(ctx)
        assert isinstance(table_list, str)

        # Since list_tables returns a formatted string, check for expected sections
        assert "Available Database Tables" in table_list
        # Given that we now have test tables, we should verify those appear
        assert "public.test_products" in table_list or "public.test_users" in table_list

    async def test_get_table_schema_resource(
        self, db_manager: DatabaseManager, test_tables
    ) -> None:
        """Test that get_table_schema_resource returns the schema for a table."""
        # Use the known test table
        schema_name = "public"  # Default schema in PostgreSQL
        table_name = "test_products"

        # Get the schema
        result = await get_table_schema_resource(schema_name, table_name)
        contents = list(result)

        assert len(contents) == 1
        assert contents[0].mime_type == "application/json"

        # Verify the content is valid JSON and has the expected structure
        content_str = contents[0].content
        assert "table_name" in content_str
        assert "schema_name" in content_str
        assert "columns" in content_str

        # Verify that our test table columns are in the schema
        assert "id" in content_str
        assert "name" in content_str
        assert "price" in content_str
        assert "category" in content_str


@pytest.mark.asyncio
class TestServerTools:
    """Test MCP server tools with controlled test data."""

    @pytest.fixture(scope="function")
    async def test_tables(self, db_manager: DatabaseManager):
        """Create test tables and data."""
        pool = await db_manager.get_connection_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_products (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    price DECIMAL(10, 2) NOT NULL,
                    category VARCHAR(50)
                )
            """)

            await conn.execute("""
                INSERT INTO test_products (name, price, category) VALUES
                ('Test Product 1', 19.99, 'Electronics'),
                ('Test Product 2', 29.99, 'Books')
            """)

        yield

        # Clean up
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS test_products")

    async def test_execute_query_select(self, test_tables) -> None:
        """Test with known test data."""
        ctx = TestContext()
        query = "SELECT * FROM test_products WHERE category = 'Electronics' LIMIT 10"

        result = await execute_query(query, ctx)

        # Verify the correct data is returned
        assert "Test Product 1" in result
        assert "19.99" in result
        assert "Electronics" in result
        # Verify the excluded data is not present
        assert "Test Product 2" not in result
        assert "29.99" not in result
        assert "Books" not in result

    async def test_execute_query_invalid(self) -> None:
        """Test that execute_query returns an error for an invalid query."""
        ctx = TestContext()
        query = "DELETE FROM non_existent_table"

        result = await execute_query(query, ctx)

        assert "Error" in result

    async def test_get_database_info(self) -> None:
        """Test that get_database_info returns database information."""
        ctx = TestContext()

        result = await get_database_info(ctx)

        assert "Database:" in result
        assert "Total tables:" in result
        assert "Schemas:" in result

    async def test_describe_table(self, test_tables, db_manager: DatabaseManager) -> None:
        """Test that describe_table returns table details for a test table."""
        ctx = TestContext()

        # Use the known test table
        schema_name = "public"  # Default schema in PostgreSQL
        table_name = "test_products"

        result = await describe_table(schema_name, table_name, ctx)

        assert f"Table: {schema_name}.{table_name}" in result
        assert "Columns:" in result
        # Verify specific columns are present in the output
        assert "id" in result
        assert "name" in result
        assert "price" in result
        assert "category" in result
        # Table should be formatted with markdown
        assert "|" in result

