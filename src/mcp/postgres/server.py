"""PostgreSQL MCP server implementation."""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Iterable

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.lowlevel.helper_types import ReadResourceContents

from .database import DatabaseManager
from .settings import PostgresSettings
from .utils import format_table_results, format_table_schema

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize settings
settings = PostgresSettings()

# Initialize FastMCP server
mcp = FastMCP(
    name="postgres-server",
    instructions="This server provides access to a PostgreSQL database. Use the tools to execute read-only queries and explore database structure.",
)

# Initialize database manager
db_manager = DatabaseManager(settings)


# Rate limiter for query execution
class RateLimiter:
    """Rate limiter for database queries."""

    def __init__(self, rate_limit_seconds: float = 1.0):
        """Initialize the rate limiter.

        Args:
            rate_limit_seconds: Minimum seconds between queries
        """
        self.rate_limit = rate_limit_seconds
        self.last_query_time = 0.0

    async def limit(self) -> None:
        """Apply rate limiting by sleeping if necessary."""
        current_time = time.time()
        time_since_last = current_time - self.last_query_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self.last_query_time = time.time()


# Create rate limiter instance
rate_limiter = RateLimiter(rate_limit_seconds=settings.rate_limit)


# Resource implementation - use a template for handling table resources
@mcp.resource("postgres:///{schema_name}/{table_name}")
async def get_table_schema_resource(
    schema_name: str, table_name: str
) -> Iterable[ReadResourceContents]:
    """Get the schema for a table.

    Args:
        schema_name: Schema name
        table_name: Table name

    Returns:
        Iterable[ReadResourceContents]: Table schema as JSON
    """
    try:
        schema = await db_manager.get_table_schema(schema_name, table_name)
        json_content = json.dumps(schema, indent=2)
        return [ReadResourceContents(content=json_content, mime_type="application/json")]
    except Exception as e:
        logger.error(f"Error retrieving schema for {schema_name}.{table_name}: {str(e)}")
        return [
            ReadResourceContents(
                content=f"Error retrieving schema: {str(e)}", mime_type="text/plain"
            )
        ]


# Tool implementations
@mcp.tool(
    name="query",  # Renamed to match project requirements
    description="Execute a read-only SQL query against the PostgreSQL database.",
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
    await ctx.report_progress(10, 100)

    # Execute the query
    success, result = await db_manager.execute_query(query)
    await ctx.report_progress(80, 100)

    if not success:
        # If result is an error message
        await ctx.error(f"Query error: {result}")
        return f"Error: {result}"

    # Format the results
    if not result:
        return "Query executed successfully. No results returned."

    formatted_result = format_table_results(result, max_rows=settings.max_rows)

    # Log completion
    total_rows = len(result)
    await ctx.report_progress(100, 100)
    await ctx.info(f"Query complete, returned {total_rows} rows")

    return formatted_result


@mcp.tool(description="Get information about the database structure, including schemas and tables.")
async def get_database_info(ctx: Context) -> str:
    """Get general information about the database, including available schemas and tables.

    Args:
        ctx: MCP context for logging and progress reporting

    Returns:
        str: Formatted overview of the database structure with schemas and table counts
    """
    try:
        await ctx.report_progress(10, 100)
        db_info = await db_manager.get_database_info()
        await ctx.report_progress(80, 100)

        # Format output
        info = [f"# Database: {db_info['database_name']}", ""]
        info.append(f"Total tables: {db_info['total_tables']}")
        info.append("")
        info.append("## Schemas:")

        for schema in db_info["schemas"]:
            info.append(f"- **{schema['schema_name']}**: {schema['table_count']} tables")

        await ctx.report_progress(100, 100)
        return "\n".join(info)
    except Exception as e:
        await ctx.error(f"Error getting database info: {str(e)}")
        return f"Error getting database info: {str(e)}"


@mcp.tool(description="Get detailed information about a specific database table.")
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
        schema = await db_manager.get_table_schema(schema_name, table_name)
        await ctx.report_progress(80, 100)

        # Format output using the utility function
        result = format_table_schema(schema)

        await ctx.report_progress(100, 100)
        return result
    except Exception as e:
        await ctx.error(f"Error describing table: {str(e)}")
        return f"Error describing table {schema_name}.{table_name}: {str(e)}"


@mcp.tool(description="List all tables in the database as resources that can be accessed.")
async def list_tables(ctx: Context) -> str:
    """List all available tables in the database.

    This tool returns a list of all tables with their PostgreSQL resource URIs
    that can be used to access schema information.

    Args:
        ctx: MCP context for logging and progress reporting

    Returns:
        str: Formatted list of tables with their resource URIs
    """
    try:
        await ctx.report_progress(10, 100)
        tables = await db_manager.get_all_tables()
        await ctx.report_progress(80, 100)

        # Format output
        result = ["# Available Database Tables", ""]
        result.append("The following tables are available as resources:")
        result.append("")

        for table in tables:
            schema_name = table["table_schema"]
            table_name = table["table_name"]
            uri = f"postgres:///{schema_name}/{table_name}"

            result.append(f"- **{schema_name}.{table_name}**")
            result.append(f"  - Resource URI: `{uri}`")
            result.append("")

        await ctx.report_progress(100, 100)
        return "\n".join(result)
    except Exception as e:
        await ctx.error(f"Error listing tables: {str(e)}")
        return f"Error listing tables: {str(e)}"


# Graceful shutdown handler
@asynccontextmanager
async def lifespan(app: FastMCP) -> None:
    """Handle server lifecycle events.

    Args:
        app: The FastMCP server instance
    """
    logger.info("🚀 PostgreSQL MCP Server starting up...")
    try:
        yield {}
        logger.info("PostgreSQL MCP Server shutting down normally...")
    except Exception as e:
        logger.error(f"Error during server operation: {e}")
    finally:
        # Close pool on shutdown
        logger.info("Closing database connections...")
        await db_manager.close()
        logger.info("Database connections closed")


# Main function to run the server
def main() -> None:
    """Run the PostgreSQL MCP server."""
    # Configure logging
    log_level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Print startup banner to stderr so it doesn't interfere with stdio communication
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    print(
        f"""
╔═══════════════════════════════════════════════════╗
║             PostgreSQL MCP Server                 ║
╠═══════════════════════════════════════════════════╣
║ Transport: {transport:<37} ║
║ Database:  {settings.database}@{settings.host}:{settings.port:<17} ║
║ Debug:     {str(settings.debug):<37} ║
╚═══════════════════════════════════════════════════╝
    """,
        file=sys.stderr,
    )

    if transport == "stdio":
        print("Server is running in stdio mode. Waiting for input...", file=sys.stderr)
        print("Note: When running in stdio mode, the server will appear to hang.", file=sys.stderr)
        print("This is normal - it's waiting for JSON-RPC messages on stdin.", file=sys.stderr)
        print("Press Ctrl+D to exit or send a proper JSON-RPC message.", file=sys.stderr)

    # Set the lifespan handler
    mcp.settings.lifespan = lifespan

    # Setup signal handlers (only on Unix platforms)
    if hasattr(signal, "SIGINT"):

        def signal_handler(sig, frame):
            print("\nReceived interrupt signal. Shutting down...", file=sys.stderr)
            # Force exit after 3 seconds if clean shutdown fails
            time.sleep(3)
            sys.exit(0)

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)

    try:
        # Run the server
        mcp.run(transport=transport)
    except KeyboardInterrupt:
        print("\nServer stopped by keyboard interrupt", file=sys.stderr)
    except Exception as e:
        logger.error(f"Error running server: {e}")
        print(f"Error running server: {e}", file=sys.stderr)
        sys.exit(1)

    print("Server shutdown complete", file=sys.stderr)


if __name__ == "__main__":
    main()

