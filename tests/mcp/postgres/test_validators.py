"""Tests for PostgreSQL query validators."""

import pytest

from src.mcp.postgres.validators import has_reasonable_limits, is_read_only_query, validate_query


class TestReadOnlyValidator:
    """Test the is_read_only_query validator."""
    
    def test_valid_select_query(self) -> None:
        """Test that a valid SELECT query passes validation."""
        query = "SELECT * FROM users"
        is_valid, _ = is_read_only_query(query)
        assert is_valid is True
    
    def test_invalid_insert_query(self) -> None:
        """Test that an INSERT query fails validation."""
        query = "INSERT INTO users (name) VALUES ('test')"
        is_valid, reason = is_read_only_query(query)
        assert is_valid is False
        assert "must start with SELECT" in reason
    
    def test_select_with_update(self) -> None:
        """Test that a SELECT query with embedded UPDATE fails validation."""
        query = "SELECT * FROM users WHERE id IN (UPDATE users SET active = true RETURNING id)"
        is_valid, reason = is_read_only_query(query)
        assert is_valid is False
        assert "UPDATE" in reason
    
    def test_comment_rejected(self) -> None:
        """Test that queries with comments are rejected."""
        query = "SELECT * FROM users -- with a comment"
        is_valid, reason = is_read_only_query(query)
        assert is_valid is False
        assert "comments" in reason


class TestLimitValidator:
    """Test the has_reasonable_limits validator."""
    
    def test_query_with_limit(self) -> None:
        """Test that a query with LIMIT passes validation."""
        query = "SELECT * FROM users LIMIT 10"
        is_valid, _ = has_reasonable_limits(query)
        assert is_valid is True
    
    def test_query_without_limit(self) -> None:
        """Test that a query without LIMIT fails validation."""
        query = "SELECT * FROM users"
        is_valid, reason = has_reasonable_limits(query)
        assert is_valid is False
        assert "LIMIT" in reason
    
    def test_aggregate_query_without_limit(self) -> None:
        """Test that an aggregate query without LIMIT passes validation."""
        query = "SELECT COUNT(*) FROM users"
        is_valid, _ = has_reasonable_limits(query)
        assert is_valid is True
    
    def test_group_by_query_without_limit(self) -> None:
        """Test that a GROUP BY query without LIMIT passes validation."""
        query = "SELECT department, COUNT(*) FROM users GROUP BY department"
        is_valid, _ = has_reasonable_limits(query)
        assert is_valid is True


class TestQueryValidator:
    """Test the full validate_query function."""
    
    def test_valid_query(self) -> None:
        """Test a completely valid query."""
        query = "SELECT * FROM users LIMIT 10"
        is_valid, _ = validate_query(query)
        assert is_valid is True
    
    def test_invalid_query_type(self) -> None:
        """Test an invalid query type."""
        query = "DELETE FROM users"
        is_valid, reason = validate_query(query)
        assert is_valid is False
        assert "must start with SELECT" in reason
    
    def test_missing_limit(self) -> None:
        """Test a query missing a limit."""
        query = "SELECT * FROM large_table"
        is_valid, reason = validate_query(query)
        assert is_valid is False
        assert "LIMIT" in reason