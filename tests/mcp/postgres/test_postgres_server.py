"""Integration tests for PostgreSQL MCP server.

These tests check the high-level behavior of the MCP server resources and tools
without mocking the database. They require a running PostgreSQL instance.
"""

import asyncio
import os
from typing import Any, Dict, List

import anyio
import pytest

from src.mcp.postgres.database import DatabaseManager
from src.mcp.postgres.server import (
    describe_table,
    execute_query,
    get_database_info,
    get_table_schema_resource,
    list_resources,
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


@pytest.mark.anyio
class TestServerResources:
    """Test MCP server resources."""
    
    async def test_list_resources(self, db_manager: DatabaseManager) -> None:
        """Test that list_resources returns a list of database tables."""
        resources = await list_resources()
        assert isinstance(resources, list)
        
        if resources:
            # Verify the structure of the resources
            resource = resources[0]
            assert hasattr(resource, "uri")
            assert hasattr(resource, "name")
            assert hasattr(resource, "description")
            assert hasattr(resource, "mimeType")
            assert resource.mimeType == "application/json"
    
    async def test_get_table_schema_resource(self, db_manager: DatabaseManager) -> None:
        """Test that get_table_schema_resource returns the schema for a table."""
        # This test will be skipped if there are no tables
        resources = await list_resources()
        if not resources:
            pytest.skip("No database tables available for testing")
            
        # Get the first table
        parts = resources[0].name.split(".")
        schema_name, table_name = parts[0], parts[1]
        
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


@pytest.mark.anyio
class TestServerTools:
    """Test MCP server tools."""
    
    async def test_execute_query_select(self) -> None:
        """Test that execute_query returns results for a valid SELECT query."""
        ctx = TestContext()
        query = "SELECT 1 as test_col"
        
        result = await execute_query(query, ctx)
        
        assert "test_col" in result
        assert "1" in result
    
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
    
    async def test_describe_table(self, db_manager: DatabaseManager) -> None:
        """Test that describe_table returns table details."""
        ctx = TestContext()
        
        # This test will be skipped if there are no tables
        tables = await db_manager.get_all_tables()
        if not tables:
            pytest.skip("No database tables available for testing")
        
        # Get the first table
        schema_name = tables[0]["table_schema"]
        table_name = tables[0]["table_name"]
        
        result = await describe_table(schema_name, table_name, ctx)
        
        assert f"Table: {schema_name}.{table_name}" in result
        assert "Columns:" in result
        # Table should be formatted with markdown
        assert "|" in result