# FastMCP Agent

A standalone FastMCP client application that interacts with MCP servers to execute complex workflows. This agent includes both a CLI tool and an OpenAI API-compatible endpoint.

## Features

- **Flexible Deployment**: Run as a CLI tool or as an API server
- **OpenAI Compatibility**: API endpoint compatible with OpenAI's chat completion format
- **Workflow States**: Initialize context, parse query, execute MCP actions, evaluate results, replan actions, generate response
- **FastMCP Client**: Connects to MCP servers via SSE transport
- **PostgreSQL Database**: For session management and conversation history
- **Confidentialmind SDK Integration**: For LLM connections
- **Clean Architecture**: Separation of concerns with modular design

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

- `AGENT_TOOLS_URL`: Default URL of the MCP server (default: <http://localhost:8000/sse>)
- `MCP_SERVER_*`: Additional MCP servers in the format MCP_SERVER_NAME=url
- `DB_CONFIG_ID`: Database connector config ID (default: DATABASE)
- `LLM_CONFIG_ID`: LLM service config ID (default: LLM)

### Config File

You can also provide a JSON configuration file with:

```json
{
  "mcp_servers": {
    "agentTools": "http://tools-server:8000/sse",
    "dataService": "http://data-server:8001/sse"
  }
}
```

## Usage

### CLI Mode

```bash
# Basic usage
python -m agent.main "What tools are available?"

# With session ID
python -m agent.main "Tell me more about table users" --session abc123

# With debug logging
python -m agent.main "Why did the previous action fail?" --debug

# With custom config file
python -m agent.main "Query the database" --config config.json
```

### API Server Mode

```bash
# Start the API server
python -m agent.main --api

# Configure host and port
python -m agent.main --api --host 127.0.0.1 --port 3000

# With debug logging
python -m agent.main --api --debug
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
  "model": "gpt-3.5-turbo",
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

## Session Management

Session IDs can be provided in one of these ways:

1. In the CLI using the `--session` argument
2. In the API via the `X-Session-ID` header
3. Via a `session_id` cookie in the API
4. Automatically generated if not provided

## Development

### Prerequisites

- Python 3.10+
- PostgreSQL database
- FastMCP-compatible MCP servers
- Access to confidentialmind SDK

### Running Tests

```bash
pytest tests/
```

### Debugging

Enable debug logging with the `--debug` flag in either CLI or API mode.
