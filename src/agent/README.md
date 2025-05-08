# FastMCP Agent

A standalone FastMCP client application that interacts with MCP servers to execute complex workflows. This agent maintains the same workflow states as the original LangGraph implementation but is built on the cleaner, more Pythonic FastMCP client architecture.

## Features

- Workflow states: initialize context, parse query, execute MCP actions, evaluate results, replan actions, generate response
- FastMCP client that connects to MCP servers via SSE transport
- PostgreSQL database for session management
- Confidentialmind SDK integration for LLM connections
- Clean separation of concerns with modular design
- Robust error handling and replanning capabilities

## Architecture

```
fastmcp_agent/
├── __init__.py           # Package exports
├── state.py              # Pydantic models for agent state
├── database.py           # PostgreSQL connection management
├── llm.py                # LLM connector using confidentialmind SDK
├── agent.py              # Core agent implementation
└── main.py               # Command-line entry point
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fastmcp-agent.git
cd fastmcp-agent

# Install the package in development mode
pip install -e .
```

## Configuration

The agent can be configured through environment variables, command-line arguments, or a configuration file.

### Environment Variables

- `AGENT_TOOLS_URL`: URL of the MCP server (default: http://localhost:8000/sse)
- `DATABASE_URL`: PostgreSQL connection string (optional, falls back to config manager)
- `LLM_URL`: LLM service URL (optional, falls back to config manager)

### Command-line Arguments

```bash
python -m fastmcp_agent.main "What tools are available?" --session my-session-id --db DATABASE --llm LLM --config config.json --debug
```

- `query`: The user query to process
- `--session`: Session ID to use (optional, generates a UUID if not provided)
- `--db`: Database config ID for confidentialmind SDK (default: DATABASE)
- `--llm`: LLM config ID for confidentialmind SDK (default: LLM)
- `--config`: Path to a JSON configuration file (optional)
- `--debug`: Enable debug logging

### Configuration File (JSON)

```json
{
  "mcp_servers": {
    "agentTools": "http://tools-server:8000/sse",
    "dataService": "http://data-server:8001/sse"
  },
  "database": {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
    "database": "agent_sessions"
  }
}
```

## Usage

### As a Command-line Tool

```bash
# Simple query
python -m fastmcp_agent.main "What's in the database?"

# Continue conversation with a session ID
python -m fastmcp_agent.main "Tell me more about table users" --session abc123

# Debugging
python -m fastmcp_agent.main "Why did the previous action fail?" --debug
```

### As a Python Library

```python
import asyncio
from fastmcp_agent import Agent, Database, DatabaseSettings, LLMConnector

async def run_agent():
    # Initialize components
    db_settings = DatabaseSettings()
    database = Database(db_settings)
    await database.connect()

    llm = LLMConnector(config_id="LLM")
    await llm.initialize()

    # Create agent with MCP servers
    mcp_servers = {
        "agentTools": "http://localhost:8000/sse",
        "dataService": "http://localhost:8001/sse"
    }

    agent = Agent(database, llm, mcp_servers)
    await agent.initialize()

    # Use context manager to ensure proper cleanup
    async with agent:
        # Run query
        state = await agent.run("What tools are available?")

        # Access results
        print(f"Response: {state.response}")

    # Cleanup
    await database.disconnect()
    await llm.close()

# Run the async function
asyncio.run(run_agent())
```

## Special Commands

The agent recognizes some special commands:

- `clear history`: Clears the conversation history for the current session
- `show history`: Displays the conversation history for the current session

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

Enable debug logging with the `--debug` flag to see detailed logs of each workflow step, including agent thoughts and LLM interactions.

## License

[MIT License](LICENSE)
