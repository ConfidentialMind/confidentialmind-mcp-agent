"""Utility functions for PostgreSQL MCP server."""

import re
from typing import Any, Dict, List


def sanitize_error(error_message: str) -> str:
    """Sanitize error messages to remove sensitive information.
    
    Args:
        error_message: The original error message
        
    Returns:
        str: Sanitized error message
    """
    # Remove connection strings, passwords, etc.
    sanitized = re.sub(r'password=\S+', 'password=***', str(error_message))
    sanitized = re.sub(r'user=\S+', 'user=***', sanitized)
    sanitized = re.sub(r'host=\S+', 'host=***', sanitized)
    return sanitized


def format_table_results(rows: List[Dict[str, Any]], max_rows: int = 100) -> str:
    """Format database query results as a readable table.
    
    Args:
        rows: The query result rows
        max_rows: Maximum number of rows to include in the output
        
    Returns:
        str: Formatted table output
    """
    if not rows:
        return "No results found."
    
    # Extract column names
    columns = list(rows[0].keys())
    
    # Create header
    header = " | ".join(columns)
    separator = "-" * len(header)
    
    # Format rows
    formatted_rows = []
    for row in rows[:max_rows]:
        formatted_row = " | ".join(str(row[col] if row[col] is not None else "NULL") for col in columns)
        formatted_rows.append(formatted_row)
    
    # Build result string
    result_str = f"{header}\n{separator}\n" + "\n".join(formatted_rows)
    
    # Add truncation note if necessary
    if len(rows) > max_rows:
        result_str += f"\n\n(Showing {max_rows} of {len(rows)} rows)"
    
    return result_str


def format_table_schema(schema: Dict[str, Any]) -> str:
    """Format table schema as markdown.
    
    Args:
        schema: The table schema information
        
    Returns:
        str: Markdown formatted schema
    """
    schema_name = schema["schema_name"]
    table_name = schema["table_name"]
    
    # Format output in a readable way
    output = [f"# Table: {schema_name}.{table_name}"]
    output.append("")
    
    # Show primary keys
    if schema["primary_keys"]:
        output.append(f"Primary Key(s): {', '.join(schema['primary_keys'])}")
        output.append("")
    
    # Format columns
    output.append("## Columns:")
    output.append("")
    output.append("| Column | Type | Nullable | Default |")
    output.append("| ------ | ---- | -------- | ------- |")
    
    for col in schema["columns"]:
        col_name = col["column_name"]
        data_type = col["data_type"]
        nullable = "YES" if col["is_nullable"] == "YES" else "NO"
        default = col["column_default"] if col["column_default"] else ""
        
        # Mark primary key columns
        if col_name in schema["primary_keys"]:
            col_name = f"**{col_name}** (PK)"
            
        output.append(f"| {col_name} | {data_type} | {nullable} | {default} |")
    
    return "\n".join(output)