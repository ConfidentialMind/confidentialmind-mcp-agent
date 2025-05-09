# FastMCP Agent

A flexible FastMCP client application that interacts with MCP servers to execute complex workflows using either CLI or API mode. The agent supports configurable transport options (stdio/SSE) and includes an OpenAI API-compatible endpoint.

## Features

- **Dual-Mode Operation**:
  - **CLI Mode**: Run as a command-line tool using stdio transport for local development
  - **API Mode**: Run as a standalone server with an OpenAI-compatible API endpoint using SSE transport
- **Flexible Transport Configuration**:
  - Support for multiple MCP servers via configurable transports
  - Dynamic transport selection based on runtime mode
- **Robust Database Integration**:
  - Automatic schema validation and creation
  - PostgreSQL-based session management
- **Intelligent Agent Workflow**:
  - Multi-stage processing: initialize context, parse query, execute MCP actions, evaluate results, replan actions, generate response
  - Handles action failures with automatic replanning
- **OpenAI Compatibility**:
  - Drop-in replacement for OpenAI's chat completions API
  - Session management via headers

## Installation

```bash
# Clone the repository
git clone git@github.com:ConfidentialMind/confidentialmind-mcp-agent.git

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

## Configuration

### Environment Variables

- `AGENT_TOOLS_URL`: Default URL of the MCP server (default: <http://localhost:8080/sse>)
- `MCP_SERVER_*`: Additional MCP servers in the format MCP_SERVER_NAME=url
- `DB_CONFIG_ID`: Database connector config ID (default: DATABASE)
- `LLM_CONFIG_ID`: LLM service config ID (default: LLM)

### Config File

You can provide a JSON configuration file with server definitions:

```json
{
  "mcp_servers": {
    "agentTools": "src/tools/postgres/__main__.py" // For CLI mode (path to script)
  }
}
```

Or for API mode:

```json
{
  "mcp_servers": {
    "agentTools": "http://0.0.0.0:8080/sse"
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
  "database": true
}
```

## Session Management

Session IDs can be provided in one of these ways:

1. In the CLI using the `--session` option
2. In the API via the `X-Session-ID` header
3. Via a `session_id` cookie in the API
4. Automatically generated if not provided

## Connecting MCP Servers

### PostgreSQL MCP Server

The agent can connect to the included PostgreSQL MCP server for database access:

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
    "postgres": "http://localhost:8080/sse"
  }
}
```

## Development

### Prerequisites

- Python 3.10+
- PostgreSQL database
- FastMCP-compatible MCP servers
- Access to confidentialmind SDK

### Architecture

The agent uses a modular architecture with these key components:

- **TransportManager**: Handles transport configuration and client management
- **Agent**: Core workflow logic for executing MCP-based queries
- **Database**: Session management and history storage
- **LLMConnector**: Interface to language model services

### Running Tests

```bash
# TODO
```

### Debugging

Enable debug logging with the `--debug` flag in any command mode:

```bash
python -m src.agent.main query "Debug this" --debug
python -m src.agent.main serve --debug
```

## License

Confidential Mind proprietary software.
