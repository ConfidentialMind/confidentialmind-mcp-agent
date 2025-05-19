# Postgres MCP Server

A FastMCP server designed to provide **read-only** access to a PostgreSQL database. It allows MCP clients (like LLMs or other applications) to inspect database schemas and execute safe `SELECT` queries.

## Features

- **Schema Inspection:** Exposes a resource (`postgres://schemas`) that lists accessible tables and their column details (names, types, nullability).
- **Read-Only SQL Execution:** Provides a tool (`execute_sql`) to run SQL queries against the database.
  - **Security:** Includes basic validation to ensure queries start with `SELECT`, `WITH`, or `EXPLAIN` and do not contain explicit data modification keywords (e.g., `UPDATE`, `DELETE`, `INSERT`, `DROP`). **See Security Considerations below.**
- **Asynchronous:** Built using `asyncpg` for non-blocking database operations.
- **ConfidentialMind Integration:**
  - Supports both local development and stack deployment modes
  - Automatically discovers database connection details from the stack
  - Continues operating even when database is not initially available
  - Background polling for database URL changes in stack deployment
- **Graceful Operation:** Can start without an initial database connection and connect later when available.

## Requirements

- Python 3.10+
- Access to a PostgreSQL database
- Dependencies:
  - `fastmcp>=2.0.0`
  - `asyncpg>=0.29.0`
  - `pydantic-settings>=2.0.0`
  - `confidentialmind_core` (for stack integration)

## Configuration

The server supports two operational modes:

### 1. Local Development Mode

Configure database connection details using **environment variables** or a `.env` file in the directory where you run the server.

**Required Environment Variables:**

- `PG_HOST`: Hostname or IP address of the Postgres server (default: `localhost`)
- `PG_PORT`: Port number (default: `5432`)
- `PG_USER`: Database username (default: `postgres`)
- `PG_PASSWORD`: Database password (default: `postgres`)
- `PG_DATABASE`: Name of the database to connect to (default: `test_db`)

**Optional:**

- `PG_DSN`: Alternatively, provide a full [PostgreSQL Connection URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING) instead of individual parameters.

**Example `.env` file:**

```dotenv
PG_HOST=your_db_host.example.com
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=your_secure_password
PG_DATABASE=mydatabase
```

### 2. Stack Deployment Mode

In stack deployment mode, the server automatically:

1. Registers a database connector with the ConfigManager
2. Discovers database connection URL from the stack
3. Continuously polls for URL changes in the background
4. Attempts to connect when database becomes available

This enables seamless operation in container orchestration environments where the database might be ready after the server starts.

**Required Environment Variables:**

- `SERVICE_NAME`: Service identifier
- `CONNECTOR_ID`: Optional, defaults to "DATABASE"
- `PG_USER` and `PG_PASSWORD`: Still required for database authentication
- `CONFIDENTIAL_MIND_LOCAL_CONFIG`: Set to "False" to enable stack mode

## Running the Server

The server runs on port 8080 by default and uses Streamable HTTP transport (via `/mcp` endpoint).

To start the server:

```bash
# From the project root
python -m src.tools.postgres_mcp

# Or if inside the src/tools/postgres_mcp directory
python __main__.py
```

Once running, the MCP server will be accessible at `http://localhost:8080/mcp`.
A health check endpoint is available at `http://localhost:8080/health`.

### Important Endpoints

- `/mcp` - The main MCP endpoint for Streamable HTTP communication
- `/health` - Health check endpoint that returns server status

## Testing

You can test the server functionality using the provided test script in `tests/test_postgres_mcp.py`. This script connects to the MCP server and verifies all key functionality:

1. Connection to the MCP server
2. Listing available resources and tools
3. Retrieving database schema information
4. Executing read-only SQL queries
5. Verifying security validation against write operations

### Running the Tests

```bash
# Make sure the server is running first
python -m src.tools.postgres_mcp

# Then in another terminal, run the test script
python -m tests.test_postgres_mcp
```

## Security Considerations

This server is designed only for **read-only** access to databases. The security measures include:

1. **Query Validation:** Only queries starting with `SELECT`, `WITH`, or `EXPLAIN` are allowed.
2. **Blacklisted Keywords:** Queries containing modification keywords like `INSERT`, `UPDATE`, `DELETE`, etc. are rejected.

**Warning:** These measures provide basic protection but are not foolproof. For production use:

- Connect using a database user with **read-only privileges**.
- Consider using database-level statement timeouts.
- Implement proper authentication and authorization at the application level.

## Integration with FastMCP Agent

The Postgres MCP server works seamlessly with the FastMCP agent:

1. Start the Postgres MCP server
2. Configure the agent to connect to the server:
   - For CLI mode: Specify the path to the server in the config file
   - For API mode: Specify the server URL in the config file
3. The agent will discover available tools and resources

Example `config.json` for the agent in API mode:

```json
{
  "mcp_servers": {
    "postgres": "http://localhost:8080/mcp"
  }
}
```

## Customization

To modify server settings:

- Change the port by editing the `run()` call in `__main__.py`: `mcp_server.run(transport="streamable-http", port=your_port)`
- Add additional tools by extending the `server.py` file with new `@mcp_server.tool()` decorated functions
- Enhance schema resources by adding more resource endpoints with `@mcp_server.resource()`
- Customize database connection settings through environment variables or `.env` file
