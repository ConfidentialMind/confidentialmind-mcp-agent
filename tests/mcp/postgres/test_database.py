"""Integration tests for PostgreSQL database operations.

These tests verify the database operations without mocking the database.
They require a running PostgreSQL instance.
"""

import pytest

from src.mcp.postgres.database import DatabaseManager
from src.mcp.postgres.settings import PostgresSettings


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


@pytest.fixture(scope="function")
async def test_db_setup(db_manager: DatabaseManager):
    """Create test tables and data, then clean up after the test."""
    # Create test tables
    pool = await db_manager.get_connection_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert test data
        await conn.execute("""
            INSERT INTO test_users (name, email) VALUES
            ('Test User 1', 'user1@example.com'),
            ('Test User 2', 'user2@example.com')
        """)

    yield

    # Clean up
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS test_users")


@pytest.mark.asyncio
class TestDatabaseManager:
    """Test database operations."""

    async def test_connection_pool(self, db_manager: DatabaseManager) -> None:
        """Test that the connection pool can be created."""
        pool = await db_manager.get_connection_pool()
        assert pool is not None

        # Test that the same pool is returned on subsequent calls
        pool2 = await db_manager.get_connection_pool()
        assert pool is pool2

    async def test_execute_query_valid(self, db_manager: DatabaseManager) -> None:
        """Test executing a valid query."""
        success, result = await db_manager.execute_query("SELECT 1 as test LIMIT 1")

        assert success is True
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["test"] == 1

    async def test_execute_query_invalid(self, db_manager: DatabaseManager) -> None:
        """Test executing an invalid query."""
        success, result = await db_manager.execute_query("SELECT * FROM nonexistent_table LIMIT 10")

        assert success is False
        assert isinstance(result, str)
        assert "nonexistent_table" in result.lower()

    async def test_get_all_tables(self, db_manager: DatabaseManager) -> None:
        """Test retrieving all tables."""
        tables = await db_manager.get_all_tables()

        assert isinstance(tables, list)
        if tables:  # If there are tables, check the structure
            assert "table_name" in tables[0]
            assert "table_schema" in tables[0]

    async def test_get_database_info(self, db_manager: DatabaseManager) -> None:
        """Test retrieving database information."""
        db_info = await db_manager.get_database_info()

        assert "database_name" in db_info
        assert "total_tables" in db_info
        assert "schemas" in db_info
        assert isinstance(db_info["schemas"], list)

    async def test_get_table_schema(self, db_manager: DatabaseManager, test_db_setup) -> None:
        """Test retrieving table schema."""
        # Use the test table created by the fixture
        schema_name = "public"  # Default schema in PostgreSQL
        table_name = "test_users"

        schema = await db_manager.get_table_schema(schema_name, table_name)

        # Verify the schema structure
        assert schema["table_name"] == "test_users"
        assert schema["schema_name"] == "public"
        assert "columns" in schema
        assert "primary_keys" in schema
        assert isinstance(schema["columns"], list)
        assert isinstance(schema["primary_keys"], list)

        # Verify specific columns
        column_names = [col["column_name"] for col in schema["columns"]]
        assert "id" in column_names
        assert "name" in column_names
        assert "email" in column_names
        assert "created_at" in column_names

        # Verify primary key
        assert "id" in schema["primary_keys"]

        # Verify column details
        name_column = next(col for col in schema["columns"] if col["column_name"] == "name")
        assert name_column["data_type"] == "character varying"
        assert name_column["is_nullable"] == "NO"  # Not nullable

