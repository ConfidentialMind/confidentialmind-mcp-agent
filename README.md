# ConfidentialMind PostgreSQL Agent

A powerful AI-powered agent that interfaces with PostgreSQL databases, enabling natural language interactions with your database. This agent uses the Model-Client-Protocol (MCP) architecture to provide a secure and flexible connection between LLMs and PostgreSQL databases.

## Overview

ConfidentialMind PostgreSQL Agent allows you to:

- Query PostgreSQL databases using natural language
- Explore database schema information
- Execute SQL queries based on natural language instructions
- Get results in a structured, easy-to-understand format

The agent uses a LangGraph-based workflow to process queries, plan actions, execute database operations, and generate human-friendly responses.

## Architecture

The system consists of the following components:

1. **Main Application** (`postgres_integration.py`): Entry point that handles CLI arguments, sets up components, and provides an interactive query session.

2. **MCP Protocol** (`src/mcp/mcp_protocol.py`): Defines the JSON-RPC based protocol for communication.

3. **MCP Client** (`src/mcp/mcp_client.py`): Client that communicates with the PostgreSQL MCP server.

4. **PostgreSQL MCP Server** (`src/mcp/postgres_mcp_server.py`): Server that connects to a PostgreSQL database and provides MCP protocol access.

5. **Agent Core** (`src/core/agent.py`): LangGraph-based workflow that processes queries, plans actions, and generates responses.

6. **LLM Connector** (`src/connectors/llm.py`): Handles communication with LLM APIs.

7. **ConfidentialMind Integration** (`src/connectors/confidentialmind.py`): Optional integration with ConfidentialMind for enhanced security and management.

## Installation

```bash
# Install dependencies
uv pip install -e .

# Install optional dependencies (recommended for development)
uv pip install python-dotenv
```

## Configuration

The agent requires the following environment variables:

```bash
# Required: PostgreSQL connection string
PG_CONNECTION_STRING="postgresql://username:password@hostname:5432/database"

# LLM API configuration
LLM_URL="http://localhost:8080/v1"  # Default
LLM_API_KEY="your_api_key"  # Optional based on your LLM service

CONFIDENTIALMIND_INTEGRATION=true
CONFIDENTIAL_MIND_LOCAL_CONFIG=True

# Optional: Set to enable debug logging
DEBUG=true
```

You can create a `.env` file to store these variables.

## Usage

### Basic Usage

```bash
# Run interactive mode using environment variables for configuration
python postgres_integration.py

# Run a single query
python postgres_integration.py --query "What tables do I have in my database?"

# Run with ConfidentialMind integration
python postgres_integration.py --mode cm
```

### Example Queries

```
# List tables
What tables are in my database?

# Explore schema
What columns are in the users table?

# Run a query
Show me the top 5 users with the most orders

# Data analysis
What is the average order value in the last week?
```

## Development

```bash
# Format code
black . && isort .

# Lint
ruff check .

# Type check
mypy .

# Run tests
pytest

# Run specific test
pytest tests/path/to/test_file.py::test_function -v
```

## Security Considerations

- The PostgreSQL MCP server restricts queries to read-only operations (SELECT, EXPLAIN, SHOW)
- All SQL executions run in a READ ONLY transaction mode
- Connection credentials are never exposed in resource URIs
- Always use a database user with appropriate permissions based on your security requirements

## Module Details

### MCP Protocol

The agent uses a Model-Client-Protocol architecture:

1. **MCP Protocol**: JSON-RPC 2.0 based protocol that defines four key methods:

   - `mcp_listResources`: Lists available resources (database tables)
   - `mcp_readResource`: Reads a specific resource (table schema)
   - `mcp_listTools`: Lists available tools (SQL query execution)
   - `mcp_callTool`: Executes a tool (runs a SQL query)

2. **MCP Client**: Manages communication with the MCP server, handling request/response cycles.

3. **MCP Server**: Connects to PostgreSQL and implements the MCP protocol, providing access to database functionality.

### Agent Workflow

The agent uses a LangGraph-based workflow:

1. **Parse Query**: Analyzes the natural language query to determine required actions
2. **Execute MCP Actions**: Executes MCP operations (list resources, read schemas, execute queries)
3. **Generate Response**: Synthesizes results into a human-friendly response

## Requirements

- Python 3.10 or higher
- PostgreSQL database
- Access to an LLM API service (local or remote)
- ConfidentialMind account (optional)

