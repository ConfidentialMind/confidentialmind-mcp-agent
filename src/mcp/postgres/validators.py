"""Query validation for PostgreSQL MCP server."""

import re
from typing import Tuple


def is_read_only_query(query: str) -> Tuple[bool, str]:
    """Check if a query is read-only (SELECT only).

    Args:
        query: The SQL query to check

    Returns:
        Tuple[bool, str]: A tuple where:
            - first element is a boolean indicating if the query is read-only (True) or not (False)
            - second element is a string containing an error message if not read-only, or empty string if valid
    """
    query = query.strip().lower()

    # Check for empty query
    if not query:
        return False, "Query cannot be empty"

    # Must start with SELECT
    if not query.startswith("select"):
        return False, "Query must start with SELECT"

    # Check for dangerous operations
    dangerous_ops = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "grant",
        "revoke",
    ]

    for op in dangerous_ops:
        pattern = r"\b" + op + r"\b"
        if re.search(pattern, query):
            return False, f"Query contains disallowed operation: {op.upper()}"

    # Check for specific dangerous constructs
    if re.search(r"\binto\s+outfile\b", query):
        return False, "Query contains disallowed operation: INTO OUTFILE"

    if re.search(r"\binto\s+dumpfile\b", query):
        return False, "Query contains disallowed operation: INTO DUMPFILE"

    # Check for comments which could be hiding code
    if "--" in query or "#" in query or "/*" in query:
        return False, "Query contains comments, which are not allowed"

    return True, ""


def has_reasonable_limits(query: str) -> Tuple[bool, str]:
    """Check if a query has reasonable limits to prevent resource exhaustion.

    Args:
        query: The SQL query to check

    Returns:
        Tuple[bool, str]: A tuple where:
            - first element is a boolean indicating if the query has reasonable limits (True) or not (False)
            - second element is a string containing an error message if limits are needed, or empty string if valid
    """
    query = query.strip().lower()

    # Check if the query already has a LIMIT clause
    if not re.search(r"\blimit\s+\d+", query):
        # If doing aggregate functions (count, sum, etc.), limits aren't necessary
        if re.search(r"\b(count|sum|avg|min|max)\s*\(", query):
            return True, ""

        # Check for GROUP BY which often returns fewer rows
        if re.search(r"\bgroup\s+by\b", query):
            return True, ""

        # For other queries, suggest adding a LIMIT
        return (
            False,
            "Query does not have a LIMIT clause. Add 'LIMIT n' to prevent large result sets.",
        )

    return True, ""


def validate_query(query: str) -> Tuple[bool, str]:
    """Perform all validation checks on a query.

    Args:
        query: The SQL query to check

    Returns:
        Tuple[bool, str]: A tuple where:
            - first element is a boolean indicating if the query is valid (True) or invalid (False)
            - second element is a string containing an error message if invalid, or empty string if valid
    """
    # Check if query is read-only
    is_ro, reason = is_read_only_query(query)
    if not is_ro:
        return False, reason

    # Check if query has reasonable limits
    has_limits, limit_reason = has_reasonable_limits(query)
    if not has_limits:
        return False, limit_reason

    return True, ""

