# Postgres MCP Server (Read-Only)

[![Powered by FastMCP](https://img.shields.io/badge/Powered%20by-FastMCP-blueviolet)](https://github.com/jlowin/fastmcp)

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

## Testing

The server includes integration tests located in `tests/mcp/postgres/test_server.py`. These tests verify the functionality of the exposed MCP resources and tools using `pytest`.

### Test Requirements

1. **Running PostgreSQL Instance:** The tests quire a running PostgreSQL database server accessible based on the connection settings (defaults to `localhost:5432`).
2. **Test Database:** A database named `test_db` (or matching `PG_DATABASE`) must exist.
3. **Permissions:** The database user (`postgres` or `PG_USER`) needs permissions to connect to the `test_db` database and `CREATE`/`DROP` tables within it. The tests create and tear down a table named `mcp_test_table`.

### Running the Tests

1. Ensure you have the development dependencies installed, including `pytest` and `pytest-asyncio`.
2. Make sure your PostgreSQL server is running and configured as described above.
3. Navigate to the root directory of the `confidentialmind-mcp-agent` project and run:

```bash
PYTHONPATH=. uv run --frozen pytest tests/mcp/postgres/
```

The tests use FastMCP's in-memory client (`fastmcp.Client`) to directly interact with the server instance, avoiding the need for network connections or subprocess management during the test execution itself, while still performing real database operations against the configured test database.
