# Postgres MCP Server (Read-Only)

A FastMCP server designed to provide **read-only** access to a PostgreSQL database. It allows MCP clients (like LLMs or other applications) to inspect database schemas and execute safe `SELECT` queries.

## Features

- **Schema Inspection:** Exposes a resource (`postgres://schemas`) that lists accessible tables and their column details (names, types, nullability).
- **Read-Only SQL Execution:** Provides a tool (`execute_sql`) to run SQL queries against the database.
  - **Security:** Includes basic validation to ensure queries start with `SELECT`, `WITH`, or `EXPLAIN` and do not contain explicit data modification keywords (e.g., `UPDATE`, `DELETE`, `INSERT`, `DROP`). **See Security Considerations below.**
- **Asynchronous:** Built using `asyncpg` for non-blocking database operations.
- **Configurable:** Database connection details are managed via environment variables or a `.env` file.

## Requirements

- Python 3.10+
- Access to a PostgreSQL database
- Dependencies:
  - `fastmcp>=2.0.0`
  - `asyncpg>=0.29.0`
  - `pydantic-settings>=2.0.0`

## Configuration

Database connection details are needed to run the server. Configure these using **environment variables** or a `.env` file in the directory where you run the server.

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

## Running the Server

The server runs on port 8080 by default and uses SSE for communication.

To start the server:

```bash
# From the project root
python -m src.tools.postgres_mcp

# Or if inside the src/tools/postgres_mcp directory
python __main__.py
```

Once running, the server will be accessible at `http://localhost:8080/sse`.

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
python tests/test_postgres_mcp.py
```

## Security Considerations

This server is designed only for **read-only** access to databases. The security measures include:

1. **Query Validation:** Only queries starting with `SELECT`, `WITH`, or `EXPLAIN` are allowed.
2. **Blacklisted Keywords:** Queries containing modification keywords like `INSERT`, `UPDATE`, `DELETE`, etc. are rejected.

**Warning:** These measures provide basic protection but are not foolproof. For production use:

- Connect using a database user with **read-only privileges**.
- Consider using database-level statement timeouts.
- Implement proper authentication and authorization at the application level.

### Customization

To modify server settings:

- Change the port by editing the `run()` call in `__main__.py`: `mcp_server.run(transport="sse", port=your_port)`
- Add additional tools by extending the `server.py` file with new `@mcp_server.tool()` decorated functions
- Enhance schema resources by adding more resource endpoints with `@mcp_server.resource()`
