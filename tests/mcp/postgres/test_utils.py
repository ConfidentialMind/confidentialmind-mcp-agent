"""Tests for PostgreSQL utility functions."""

from typing import Dict, List

from src.mcp.postgres.utils import (
    format_table_results,
    format_table_schema,
    sanitize_error,
)


class TestSanitizeError:
    """Test the sanitize_error function."""

    def test_password_sanitization(self) -> None:
        """Test that passwords are sanitized in error messages."""
        error = "Error connecting to PostgreSQL: password=secret123"
        sanitized = sanitize_error(error)
        assert "password=secret123" not in sanitized
        assert "password=***" in sanitized

    def test_user_sanitization(self) -> None:
        """Test that usernames are sanitized in error messages."""
        error = "Error connecting to PostgreSQL: user=admin"
        sanitized = sanitize_error(error)
        assert "user=admin" not in sanitized
        assert "user=***" in sanitized

    def test_host_sanitization(self) -> None:
        """Test that host information is sanitized in error messages."""
        error = "Error connecting to PostgreSQL: host=internal-db.example.com"
        sanitized = sanitize_error(error)
        assert "host=internal-db.example.com" not in sanitized
        assert "host=***" in sanitized

    def test_multiple_sanitization(self) -> None:
        """Test that multiple sensitive values are sanitized."""
        error = "Error: host=db.example.com user=admin password=secret123"
        sanitized = sanitize_error(error)
        assert "host=***" in sanitized
        assert "user=***" in sanitized
        assert "password=***" in sanitized


class TestFormatTableResults:
    """Test the format_table_results function."""

    def test_empty_results(self) -> None:
        """Test formatting empty results."""
        rows: List[Dict[str, str]] = []
        result = format_table_results(rows)
        assert result == "No results found."

    def test_single_row(self) -> None:
        """Test formatting a single row."""
        rows = [{"id": 1, "name": "Test"}]
        result = format_table_results(rows)
        assert "id | name" in result
        assert "1 | Test" in result

    def test_multiple_rows(self) -> None:
        """Test formatting multiple rows."""
        rows = [
            {"id": 1, "name": "Test1"},
            {"id": 2, "name": "Test2"},
        ]
        result = format_table_results(rows)
        assert "id | name" in result
        assert "1 | Test1" in result
        assert "2 | Test2" in result

    def test_null_values(self) -> None:
        """Test formatting rows with NULL values."""
        rows = [{"id": 1, "name": None}]
        result = format_table_results(rows)
        assert "NULL" in result

    def test_row_limit(self) -> None:
        """Test that row count is limited."""
        rows = [{"id": i} for i in range(200)]
        result = format_table_results(rows, max_rows=10)
        assert "(Showing 10 of 200 rows)" in result


class TestFormatTableSchema:
    """Test the format_table_schema function."""

    def test_basic_schema(self) -> None:
        """Test formatting a basic table schema."""
        schema = {
            "table_name": "users",
            "schema_name": "public",
            "columns": [
                {
                    "column_name": "id",
                    "data_type": "integer",
                    "is_nullable": "NO",
                    "column_default": "nextval('users_id_seq'::regclass)",
                },
                {
                    "column_name": "name",
                    "data_type": "character varying",
                    "is_nullable": "YES",
                    "column_default": None,
                },
            ],
            "primary_keys": ["id"],
        }

        result = format_table_schema(schema)

        assert "# Table: public.users" in result
        assert "Primary Key(s): id" in result
        assert "## Columns:" in result
        assert "| **id** (PK) | integer |" in result
        assert "| name | character varying |" in result

    def test_no_primary_keys(self) -> None:
        """Test formatting a schema with no primary keys."""
        schema = {
            "table_name": "logs",
            "schema_name": "public",
            "columns": [
                {
                    "column_name": "message",
                    "data_type": "text",
                    "is_nullable": "YES",
                    "column_default": None,
                },
            ],
            "primary_keys": [],
        }

        result = format_table_schema(schema)

        assert "# Table: public.logs" in result
        assert "Primary Key(s)" not in result
        assert "## Columns:" in result
        assert "| message | text |" in result

