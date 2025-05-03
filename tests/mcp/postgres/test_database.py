"""Integration tests for PostgreSQL database operations.

These tests verify the database operations without mocking the database.
They require a running PostgreSQL instance.
"""

import asyncio
from typing import Dict, List

import anyio
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


@pytest.mark.anyio
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
        success, result = await db_manager.execute_query("SELECT 1 as test")
        
        assert success is True
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["test"] == 1
    
    async def test_execute_query_invalid(self, db_manager: DatabaseManager) -> None:
        """Test executing an invalid query."""
        success, result = await db_manager.execute_query("SELECT * FROM nonexistent_table")
        
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
    
    async def test_get_table_schema(self, db_manager: DatabaseManager) -> None:
        """Test retrieving table schema."""
        # Skip if no tables are available
        tables = await db_manager.get_all_tables()
        if not tables:
            pytest.skip("No tables available for testing")
        
        # Get the first table's schema
        schema_name = tables[0]["table_schema"]
        table_name = tables[0]["table_name"]
        
        schema = await db_manager.get_table_schema(schema_name, table_name)
        
        assert "table_name" in schema
        assert "schema_name" in schema
        assert "columns" in schema
        assert "primary_keys" in schema
        assert isinstance(schema["columns"], list)
        assert isinstance(schema["primary_keys"], list)
        
        if schema["columns"]:
            column = schema["columns"][0]
            assert "column_name" in column
            assert "data_type" in column
            assert "is_nullable" in column