# Postgres MCP Server

A FastMCP server designed to provide **read-only** access to a PostgreSQL database with comprehensive observability features. It allows MCP clients (like LLMs or other applications) to inspect database schemas and execute safe `SELECT` queries while providing structured JSON logging and distributed tracing.

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
- **Comprehensive Observability:**
  - **Structured JSON Logging**: All operations logged in OpenTelemetry-compatible format
  - **Distributed Tracing**: Request correlation across service boundaries
  - **Performance Monitoring**: Query execution timing and metrics
  - **Error Context**: Rich error logging with query context and stack traces
  - **Health Monitoring**: Detailed health check logging and metrics
- **Graceful Operation:** Can start without an initial database connection and connect later when available.

## Requirements

- Python 3.10+
- Access to a PostgreSQL database
- Dependencies:
  - `fastmcp>=2.0.0`
  - `asyncpg>=0.29.0`
  - `pydantic-settings>=2.0.0`
  - `confidentialmind_core` (for stack integration)
  - `structlog>=25.3.0` (for observability)

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
- `DEBUG`: Set to "true" for human-readable logs (default: JSON format)

**Optional:**

- `PG_DSN`: Alternatively, provide a full [PostgreSQL Connection URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING) instead of individual parameters.

**Example `.env` file:**

```dotenv
PG_HOST=your_db_host.example.com
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=your_secure_password
PG_DATABASE=mydatabase
DEBUG=true
```

### 2. Stack Deployment Mode

In stack deployment mode, the server automatically:

1. Registers a database connector with the ConfigManager
2. Discovers database connection URL from the stack
3. Continuously polls for URL changes in the background
4. Attempts to connect when database becomes available
5. Outputs structured JSON logs for OpenTelemetry Collector

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
- `/health` - Health check endpoint that returns server status with observability metrics

## Observability Features

The server provides comprehensive observability through structured logging and distributed tracing:

### Structured JSON Logging

All operations are logged in OpenTelemetry-compatible JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "logger": "postgres.mcp",
  "event": "SQL query completed successfully",
  "event_type": "postgres.query.complete",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "span_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "session_id": "user-session-123",
  "duration_ms": 150.5,
  "success": true,
  "data": {
    "row_count": 42,
    "query_preview": "SELECT * FROM users WHERE...",
    "tool_name": "execute_sql"
  }
}
```

### Event Types

The server emits events with consistent taxonomy:

- `postgres.query.start` - SQL query execution initiated
- `postgres.query.complete` - SQL query completed (success/failure)
- `postgres.query.validation_error` - Query failed validation
- `postgres.query.error` - Query execution error
- `resource.schemas.requested` - Schema resource accessed
- `health.check.requested` - Health check performed
- `service.init.complete` - Server initialization completed

### Distributed Tracing

Request traces flow from the agent through the MCP server:

- Automatic trace context extraction from MCP requests
- Parent-child span relationships maintained
- Trace correlation across database operations
- Performance timing for all database interactions

### Debug vs Production Logging

**Development Mode** (`DEBUG=true`):

```bash
2024-01-15 10:30:45 [info] SQL query completed successfully
├── trace_id: 550e8400-e29b-41d4-a716-446655440000
├── duration_ms: 150.5
└── data: {"row_count": 42, "query_preview": "SELECT * FROM users..."}
```

**Production Mode** (`DEBUG=false`):

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "event": "SQL query completed successfully",
  "event_type": "postgres.query.complete",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "duration_ms": 150.5,
  "success": true,
  "data": { "row_count": 42 }
}
```

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

The Postgres MCP server works seamlessly with the FastMCP agent and provides full observability integration:

1. Start the Postgres MCP server
2. Configure the agent to connect to the server:
   - For CLI mode: Specify the path to the server in the config file
   - For API mode: Specify the server URL in the config file
3. The agent will discover available tools and resources
4. All operations are traced end-to-end with correlation IDs

Example `config.json` for the agent in API mode:

```json
{
  "mcp_servers": {
    "postgres": "http://localhost:8080/mcp"
  }
}
```

## Monitoring and Troubleshooting

### Health Check Endpoint

The `/health` endpoint provides detailed status information:

```json
{
  "status": "healthy",
  "service": "postgres-mcp-server",
  "database_connected": true,
  "database_error": null,
  "server_mode": "stack_deployment",
  "server_time": "2024-01-15T10:30:45.123Z",
  "connector_id": "DATABASE"
}
```

### Log Analysis

Query performance monitoring:

```bash
# Filter query performance logs
cat logs/postgres-mcp.log | jq 'select(.event_type == "postgres.query.complete") | {duration_ms, row_count, success}'

# Monitor query errors
cat logs/postgres-mcp.log | jq 'select(.event_type == "postgres.query.error") | {error, error_type, data}'
```

### OpenTelemetry Integration

The structured logs are designed for OpenTelemetry Collector ingestion:

```yaml
# otel-collector.yml
receivers:
  filelog:
    include: ["/var/log/postgres-mcp/*.log"]
    operators:
      - type: json_parser
        timestamp:
          parse_from: attributes.timestamp
          layout: "2006-01-02T15:04:05.000Z"

processors:
  batch:

exporters:
  jaeger:
    endpoint: http://jaeger:14250

service:
  pipelines:
    logs:
      receivers: [filelog]
      processors: [batch]
      exporters: [jaeger]
```

## Customization

To modify server settings:

- Change the port by editing the `run()` call in `__main__.py`: `mcp_server.run(transport="streamable-http", port=your_port)`
- Add additional tools by extending the `server.py` file with new `@mcp_server.tool()` decorated functions
- Enhance schema resources by adding more resource endpoints with `@mcp_server.resource()`
- Customize database connection settings through environment variables or `.env` file
- Add custom observability events by using the structured logger:

```python
from src.shared.logging import get_logger

structlog_logger = get_logger("postgres.mcp")

# Custom event logging
structlog_logger.info(
    "Custom operation completed",
    event_type="postgres.custom.complete",
    data={"custom_field": "value"}
)
```

For complete observability documentation and best practices, see [guides/observability.md](../../guides/observability.md).
