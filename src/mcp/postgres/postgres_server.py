"""PostgreSQL MCP server implementation."""

import asyncio
import json
import logging
import os
import re
import time
from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, List, Optional

import asyncpg
from pydantic import AnyUrl
from pydantic_settings import BaseSettings

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import Resource as MCPResource, Tool


# Configuration using environment variables
class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings from environment variables."""
    
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "postgres"
    connection_timeout: int = 10
    max_connections: int = 10
    debug: bool = False
    
    class Config:
        """Pydantic config with environment variable prefix."""
        
        env_prefix = "PG_"


# Initialize settings
settings = PostgresSettings()

# Initialize FastMCP server
mcp = FastMCP(
    name="postgres-server",
    instructions="This server provides access to a PostgreSQL database. Use the tools to execute read-only queries and explore database structure."
)

# Shared connection pool
pool: Optional[asyncpg.Pool] = None


# Helper functions
async def get_connection_pool() -> asyncpg.Pool:
    """Get or create the database connection pool.
    
    Returns:
        asyncpg.Pool: The database connection pool
    """
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
            timeout=settings.connection_timeout,
            max_size=settings.max_connections,
        )
    return pool


async def is_read_only_query(query: str) -> bool:
    """Check if a query is read-only (SELECT only).
    
    Args:
        query: The SQL query to check
        
    Returns:
        bool: True if the query is read-only, False otherwise
    """
    query = query.strip().lower()
    # Simple check - a more robust implementation would use a SQL parser
    return query.startswith('select') and not re.search(
        r'\b(insert|update|delete|drop|alter|create|truncate)\b', 
        query
    )


async def get_all_tables() -> List[Dict[str, str]]:
    """Get a list of all tables in the database.
    
    Returns:
        List[Dict[str, str]]: List of tables with schema and table name
    """
    pool = await get_connection_pool()
    async with pool.acquire() as conn:
        # Set a timeout for the query
        await conn.execute("SET statement_timeout TO 5000")  # 5 seconds
        tables = await conn.fetch("""
            SELECT table_name, table_schema 
            FROM information_schema.tables 
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """)
        return [dict(table) for table in tables]


async def get_table_schema(schema_name: str, table_name: str) -> Dict[str, Any]:
    """Get the schema for a specific table.
    
    Args:
        schema_name: The name of the schema
        table_name: The name of the table
        
    Returns:
        Dict[str, Any]: Table schema information with columns and primary keys
    """
    pool = await get_connection_pool()
    async with pool.acquire() as conn:
        # Set a timeout for the query
        await conn.execute("SET statement_timeout TO 5000")  # 5 seconds
        
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable, 
                   column_default, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """, schema_name, table_name)
        
        primary_keys = await conn.fetch("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = ($1 || '.' || $2)::regclass AND i.indisprimary
        """, schema_name, table_name)
        
        pk_list = [pk['attname'] for pk in primary_keys]
        
        return {
            "table_name": table_name,
            "schema_name": schema_name,
            "columns": [dict(col) for col in columns],
            "primary_keys": pk_list
        }


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


# Rate limiter for query execution
class RateLimiter:
    """Rate limiter for database queries.
    
    Ensures that queries aren't executed too frequently.
    """
    
    def __init__(self, rate_limit_seconds: float = 1.0):
        """Initialize the rate limiter.
        
        Args:
            rate_limit_seconds: Minimum seconds between queries
        """
        self.rate_limit = rate_limit_seconds
        self.last_query_time = 0.0
    
    async def limit(self) -> None:
        """Apply rate limiting by sleeping if necessary.
        
        Ensures that at least rate_limit_seconds have passed since the last query.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_query_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self.last_query_time = time.time()


# Create rate limiter instance
rate_limiter = RateLimiter(rate_limit_seconds=0.5)


# Resource implementation
@mcp.resource
async def list_resources() -> List[MCPResource]:
    """List all tables in the database as resources.
    
    Returns:
        List[MCPResource]: List of database tables as MCP resources
    """
    tables = await get_all_tables()
    resources = []
    
    for table in tables:
        schema_name = table['table_schema']
        table_name = table['table_name']
        uri = f"postgres:///{schema_name}/{table_name}"
        
        resources.append(MCPResource(
            uri=AnyUrl(uri),
            name=f"{schema_name}.{table_name}",
            description=f"Schema for {schema_name}.{table_name} table",
            mimeType="application/json"
        ))
    
    return resources


@mcp.resource("postgres:///{schema_name}/{table_name}")
async def get_table_schema_resource(schema_name: str, table_name: str) -> Iterable[ReadResourceContents]:
    """Get the schema for a table.
    
    Args:
        schema_name: Schema name
        table_name: Table name
        
    Returns:
        Iterable[ReadResourceContents]: Table schema as JSON
    """
    try:
        schema = await get_table_schema(schema_name, table_name)
        json_content = json.dumps(schema, indent=2)
        return [ReadResourceContents(
            content=json_content,
            mime_type="application/json"
        )]
    except Exception as e:
        return [ReadResourceContents(
            content=f"Error retrieving schema: {sanitize_error(str(e))}",
            mime_type="text/plain"
        )]


# Tool implementations
@mcp.tool(
    description="Execute a read-only SQL query against the PostgreSQL database."
)
async def execute_query(query: str, ctx: Context) -> str:
    """Execute a read-only SQL query against the PostgreSQL database.
    
    Args:
        query: The SQL query to execute (must be SELECT only)
        ctx: MCP context for logging and progress reporting
        
    Returns:
        str: Formatted query results
        
    Examples:
        SELECT * FROM users LIMIT 10
        SELECT count(*) FROM orders WHERE status = 'completed'
    """
    # Log the query request
    await ctx.info(f"Executing query: {query[:50]}...")
    
    # Apply rate limiting
    await rate_limiter.limit()
    
    # Validate that the query is read-only
    if not await is_read_only_query(query):
        await ctx.error("Non-SELECT query attempted")
        return "Error: Only SELECT queries are allowed for security reasons."
    
    try:
        # Track progress
        await ctx.report_progress(10, 100)
        
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            # Set statement timeout to 10 seconds
            await conn.execute("SET statement_timeout TO 10000")
            
            # Execute the query
            await ctx.report_progress(30, 100)
            results = await conn.fetch(query)
            await ctx.report_progress(80, 100)
            
            # Format results as a readable string
            if not results:
                return "Query executed successfully. No results returned."
            
            # Convert to list of dicts for easier formatting
            rows = [dict(row) for row in results]
            
            # Build a formatted table-like output
            columns = list(rows[0].keys())
            header = " | ".join(columns)
            separator = "-" * len(header)
            
            formatted_rows = []
            for row in rows[:100]:  # Limit to 100 rows
                formatted_row = " | ".join(str(row[col]) for col in columns)
                formatted_rows.append(formatted_row)
            
            result_str = f"{header}\n{separator}\n" + "\n".join(formatted_rows)
            
            # If there are too many rows, truncate and indicate
            total_rows = len(rows)
            if total_rows > 100:
                result_str += f"\n\n(Showing 100 of {total_rows} rows)"
            
            await ctx.report_progress(100, 100)
            await ctx.info(f"Query complete, returned {total_rows} rows")
                
            return result_str
    except asyncio.TimeoutError:
        await ctx.error("Query timeout")
        return "Error: Query timed out. Please try a simpler query."
    except Exception as e:
        error_msg = sanitize_error(str(e))
        await ctx.error(f"Query error: {error_msg}")
        return f"Error executing query: {error_msg}"


@mcp.tool(
    description="Get information about the database structure, including schemas and tables."
)
async def get_database_info(ctx: Context) -> str:
    """Get general information about the database, including available schemas and tables.
    
    Args:
        ctx: MCP context for logging and progress reporting
    
    Returns:
        str: Formatted overview of the database structure with schemas and table counts
    """
    try:
        await ctx.report_progress(10, 100)
        
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            # Set statement timeout
            await conn.execute("SET statement_timeout TO 5000")
            
            # Get database name
            db_name = await conn.fetchval("SELECT current_database()")
            await ctx.report_progress(30, 100)
            
            # Get schema information
            schemas = await conn.fetch("""
                SELECT schema_name, COUNT(table_name) as table_count
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                GROUP BY schema_name
                ORDER BY schema_name
            """)
            await ctx.report_progress(60, 100)
            
            # Get total tables
            total_tables = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            """)
            await ctx.report_progress(80, 100)
            
            # Format output
            info = [f"# Database: {db_name}", ""]
            info.append(f"Total tables: {total_tables}")
            info.append("")
            info.append("## Schemas:")
            
            for schema in schemas:
                info.append(f"- **{schema['schema_name']}**: {schema['table_count']} tables")
            
            await ctx.report_progress(100, 100)
            return "\n".join(info)
    except Exception as e:
        error_msg = sanitize_error(str(e))
        await ctx.error(f"Error getting database info: {error_msg}")
        return f"Error getting database info: {error_msg}"


@mcp.tool(
    description="Get detailed information about a specific database table."
)
async def describe_table(schema_name: str, table_name: str, ctx: Context) -> str:
    """Get detailed information about a specific database table.
    
    Args:
        schema_name: The schema name where the table is located
        table_name: The name of the table to describe
        ctx: MCP context for logging and progress reporting
        
    Returns:
        str: Detailed table structure in Markdown format
    """
    try:
        await ctx.report_progress(10, 100)
        schema = await get_table_schema(schema_name, table_name)
        await ctx.report_progress(80, 100)
        
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
        
        await ctx.report_progress(100, 100)
        return "\n".join(output)
    except Exception as e:
        error_msg = sanitize_error(str(e))
        await ctx.error(f"Error describing table: {error_msg}")
        return f"Error describing table {schema_name}.{table_name}: {error_msg}"


# Graceful shutdown handler
@asynccontextmanager
async def lifespan(app: FastMCP) -> None:
    """Handle server lifecycle events.
    
    Args:
        app: The FastMCP server instance
    """
    try:
        yield {}
    finally:
        # Close pool on shutdown
        global pool
        if pool:
            await pool.close()


# Main function to run the server
def main() -> None:
    """Run the PostgreSQL MCP server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not settings.debug else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize and run the server
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport, lifespan=lifespan)


if __name__ == "__main__":
    main()