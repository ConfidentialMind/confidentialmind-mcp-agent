# ConfidentialMind MCP Agent

A flexible FastMCP client application for interacting with Model Context Protocol (MCP) servers to execute complex workflows. The agent supports both CLI and API modes with configurable transport options and includes an OpenAI API-compatible endpoint.

## Overview

The ConfidentialMind MCP Agent serves as a bridge between users and MCP servers, providing a unified interface for accessing various capabilities. It can:

- Process natural language queries and convert them into structured actions
- Connect to multiple MCP servers simultaneously
- Execute complex workflows with automatic error handling and replanning
- Maintain conversation context across sessions
- Operate as either a CLI tool or an API server with OpenAI compatibility

## Features

- **Dual-Mode Operation**:

  - **CLI Mode**: Run as a command-line tool using stdio transport for local development
  - **API Mode**: Run as a standalone server with an OpenAI-compatible API endpoint using streamable-HTTP transport

- **ConfidentialMind Integration**:

  - Automatic connector registration and URL discovery in stack deployment
  - Background polling for service URLs in production environments
  - Graceful operation even when services are not immediately available

- **Flexible Transport Configuration**:

  - Support for multiple MCP servers via configurable transports
  - Dynamic transport selection based on runtime mode
  - Automatic conversion of legacy SSE URLs to streamable-HTTP

- **Robust Database Integration**:

  - Automatic schema validation and creation
  - Connection pooling with reconnection logic
  - PostgreSQL-based session management

- **Intelligent Agent Workflow**:

  - Multi-stage processing pipeline:
    1. Initialize context - discover available tools and resources
    2. Parse query - use LLM to plan actions
    3. Execute MCP actions - call tools on servers
    4. Evaluate results - check for errors and replan if needed
    5. Generate response - create final response based on results
  - Handles action failures with automatic replanning
  - Maintains conversation history for context

- **OpenAI Compatibility**:
  - Drop-in replacement for OpenAI's chat completions API
  - Session management via headers or cookies

## Installation

```bash
# Clone the repository
git clone https://github.com/ConfidentialMind/confidentialmind-mcp-agent.git

# Navigate to project root
cd confidentialmind-mcp-agent

# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
## MacOS:
brew install uv

## or
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.10

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Operational Modes

The agent supports two main operational modes:

### 1. Local Development Mode

In local development mode:

- Configuration loaded from `.env` file and environment variables
- Connection details retrieved from environment with pattern `{CONFIG_ID}_URL` and `{CONFIG_ID}_APIKEY`
- Services can be configured independently

### 2. Stack Deployment Mode

In stack deployment mode:

- Automatically registers connectors with ConfigManager
- Discovers service URLs dynamically from the stack
- Continuously polls for URL changes in the background
- Operates gracefully even when services become available after startup

## Configuration

### Environment Variables

- `AGENT_TOOLS_URL`: Default URL of the MCP server (default: <http://localhost:8080/mcp>)
- `MCP_SERVER_*`: Additional MCP servers in the format MCP_SERVER_NAME=url
- `DB_CONFIG_ID`: Database connector config ID (default: DATABASE)
- `LLM_CONFIG_ID`: LLM service config ID (default: LLM)
- `CONFIDENTIAL_MIND_LOCAL_CONFIG`: Set to "False" for stack deployment mode

### Config File

You can provide a JSON configuration file with server definitions:

```json
{
  "mcp_servers": {
    "postgres": "src/tools/postgres_mcp/__main__.py", // For CLI mode (path to script)
    "other_server": "path/to/other/server/script.py"
  }
}
```

Or for API mode:

```json
{
  "mcp_servers": {
    "postgres": "http://0.0.0.0:8080/mcp", // Using streamable-HTTP endpoint
    "other_server": "http://other-server:8080/mcp"
  }
}
```

## Usage

The agent provides three main commands through its Typer CLI interface:

### 1. Query Mode

Run a single query in CLI mode:

```bash
# Basic query
python -m src.agent.main query "What tools are available?"

# With session ID
python -m src.agent.main query "Tell me more about table users" --session abc123

# With debug logging
python -m src.agent.main query "Why did the previous action fail?" --debug

# With custom config file
python -m src.agent.main query "Query the database" --config config.json

# With custom database and LLM connectors
python -m src.agent.main query "Complex query" --db CUSTOM_DB --llm CUSTOM_LLM
```

### 2. API Server Mode

Run as a standalone server with an OpenAI-compatible API endpoint:

```bash
# Start the API server with default settings
python -m src.agent.main serve

# Configure host and port
python -m src.agent.main serve --host 127.0.0.1 --port 3000

# With debug logging
python -m src.agent.main serve --debug

# With custom config file
python -m src.agent.main serve --config api_config.json

# With custom database and LLM connectors
python -m src.agent.main serve --db PRODUCTION_DB --llm GPT4
```

### 3. Clear Session History

Utility to clear the conversation history for a specific session:

```bash
# Clear history for a specific session ID
python -m src.agent.main clear-history abc123

# With custom database connector
python -m src.agent.main clear-history abc123 --db CUSTOM_DB
```

## API Endpoints

### Chat Completions Endpoint

```
POST /v1/chat/completions
```

Request format:

```json
{
  "model": "cm-llm",
  "messages": [{ "role": "user", "content": "What tools are available?" }]
}
```

Response format:

```json
{
  "id": "chatcmpl-123456",
  "object": "chat.completion",
  "created": 1692372149,
  "model": "cm-llm",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The available tools are..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### Health Check

```
GET /health
```

Response:

```json
{
  "status": "healthy",
  "deployment_mode": "stack",
  "database": {
    "connected": true,
    "error": null
  },
  "llm": {
    "connected": true
  },
  "mcp_servers": {
    "count": 2,
    "servers": ["postgres", "otherServer"]
  },
  "timestamp": "2023-08-18T12:34:56.789Z"
}
```

## Session Management

Session IDs can be provided in one of these ways:

1. In the CLI using the `--session` option
2. In the API via the `X-Session-ID` header
3. Via a `session_id` cookie in the API
4. Automatically generated if not provided

## Agent Workflow

The agent follows a multi-stage workflow for processing queries:

1. **Initialize Context**:

   - Discover available tools and resources from MCP servers
   - Set up execution context with available capabilities

2. **Parse Query**:

   - Use LLM to understand the user's intent
   - Create a plan with specific actions to execute
   - Determine which MCP servers and tools are needed

3. **Execute MCP Actions**:

   - Call tools on appropriate MCP servers
   - Process results and handle any errors
   - Track execution state for complex workflows

4. **Evaluate Results**:

   - Check if actions were successful
   - Identify errors or unexpected results
   - Determine if replanning is necessary

5. **Replan Actions** (if needed):

   - Use LLM to generate a revised plan based on errors
   - Adjust subsequent actions to recover from failures
   - Create new approach to solve the original intent

6. **Generate Response**:
   - Create a coherent response from all gathered information
   - Format data appropriately for the user
   - Include error explanations if things went wrong

## Available MCP Servers

The repository includes the following MCP servers:

### PostgreSQL MCP Server

A FastMCP server providing read-only access to PostgreSQL databases. It allows:

- Schema inspection via the `postgres://schemas` resource
- SQL query execution via the `execute_sql` tool (read-only)

For more details on the PostgreSQL MCP server, see `src/tools/postgres_mcp/README.md`.

Additional MCP servers will be added in the future to expand the agent's capabilities.

## Connecting MCP Servers

To connect the agent to an MCP server:

```bash
# Start the PostgreSQL MCP server
python -m src.tools.postgres_mcp

# Then start the agent in API mode, configuring the PostgreSQL server
python -m src.agent.main serve --config postgres_config.json
```

Example `postgres_config.json`:

```json
{
  "mcp_servers": {
    "postgres": "http://localhost:8080/mcp"
  }
}
```

## Docker Support

Docker configurations are provided for both the agent and PostgreSQL MCP server:

```bash
# Build the agent container
docker build -f docker/agent.Dockerfile -t confidentialmind-mcp-agent .

# Build the PostgreSQL MCP server container
docker build -f docker/postgres-mcp.Dockerfile -t confidentialmind-postgres-mcp .

# Run the PostgreSQL MCP server
docker run -p 8080:8080 -e PG_HOST=host.docker.internal confidentialmind-postgres-mcp

# Run the agent in API mode
docker run -p 8000:8080 -e AGENT_TOOLS_URL=http://host.docker.internal:8080/mcp confidentialmind-mcp-agent
```

## Development

### Prerequisites

- Python 3.10+
- PostgreSQL database
- FastMCP-compatible MCP servers
- Access to confidentialmind_core SDK

### Architecture

The agent uses a modular architecture with these key components:

- **ConnectorConfigManager**: Manages connection configuration for stack deployment
- **TransportManager**: Handles transport configuration and client management
- **Agent**: Core workflow logic for executing MCP-based queries
- **Database**: Session management and history storage
- **LLMConnector**: Interface to language model services

### Running Tests

To test the API mode:

- Initialize the Postgres MCP server: `python -m src.tools.postgres_mcp`
- Initialize the agent in API mode: `python -m src.agent.main serve`
- Run the test: `python -m tests.test_agent_api`

To test the CLI mode:

- Run the test: `python -m tests.test_agent_cli`

### Debugging

Enable debug logging with the `--debug` flag in any command mode:

```bash
python -m src.agent.main query "Debug this" --debug
python -m src.agent.main serve --debug
```

## Additional Documentation

- For more details on the agent implementation, see `src/agent/README.md`
- For PostgreSQL MCP server documentation, see `src/tools/postgres_mcp/README.md`
- For guidance on building MCP servers, see `guides/mcp_server_guide.md`
- For details on ConfidentialMind connection management, see `guides/confidentialmind.md`

## Resilience Features

The agent includes several resilience features:

- **Connection Retries**: Automatic reconnection with exponential backoff
- **Background Polling**: Continuous polling for URL changes in stack mode
- **Graceful Degradation**: Operation with partial service availability
- **Error Recovery**: Replanning logic to handle action failures
- **Stateless Fallback**: Can operate without database connection if needed
