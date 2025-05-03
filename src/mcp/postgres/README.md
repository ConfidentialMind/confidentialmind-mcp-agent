# PostgreSQL MCP Server

A Model Context Protocol (MCP) server implementation for PostgreSQL databases that enables Large Language Models to safely query database information.

## Features

- **Read-only Query Execution**: Safely execute SELECT queries against PostgreSQL databases
- **Database Exploration**: Discover database structure, schemas, and tables
- **Table Schema Inspection**: Get detailed information about table structures
- **Secure by Design**: Input validation, rate limiting, and read-only operations

## Usage

### Environment Variables

Configure the PostgreSQL connection using these environment variables:

```bash
PG_HOST=localhost             # Database host
PG_PORT=5432                  # Database port
PG_USER=postgres              # Database user
PG_PASSWORD=password          # Database password
PG_DATABASE=postgres          # Database name
PG_CONNECTION_TIMEOUT=10      # Connection timeout in seconds
PG_MAX_CONNECTIONS=10         # Maximum connections in the pool
PG_DEBUG=false                # Enable debug logging
PG_STATEMENT_TIMEOUT=10000    # Statement timeout in milliseconds
PG_RATE_LIMIT=0.5             # Minimum seconds between queries
PG_MAX_ROWS=100               # Maximum rows to return in query results
```

### Running the Server

```bash
# Run the server directly
python -m src.mcp.postgres

# Run with custom transport (stdio is default)
MCP_TRANSPORT=http python -m src.mcp.postgres
```

## Security Considerations

This server implements multiple security measures:

1. **Read-only Validation**: Only allows SELECT queries
2. **Input Sanitization**: Prevents SQL injection and cleans error messages
3. **Resource Limits**: Enforces query timeouts and result size limits
4. **Rate Limiting**: Prevents overwhelming the database with too many queries

## MCP Tools and Resources

### Tools

- `execute_query`: Executes read-only SQL queries
- `get_database_info`: Provides an overview of database structure
- `describe_table`: Returns detailed table schema information

### Resources

- Table schemas exposed as resources with URI pattern: `postgres:///{schema_name}/{table_name}`

## Database Access Requirements

For optimal security, create a dedicated read-only database user:

```sql
-- Create a dedicated read-only user for the MCP server
CREATE USER mcp_readonly WITH PASSWORD 'secure_password';

-- Grant connection permission to the database
GRANT CONNECT ON DATABASE your_database TO mcp_readonly;

-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO mcp_readonly;

-- Grant SELECT permissions on all existing tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_readonly;

-- Grant SELECT permissions on all future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mcp_readonly;

-- Revoke any unnecessary privileges
REVOKE CREATE ON SCHEMA public FROM mcp_readonly;
```