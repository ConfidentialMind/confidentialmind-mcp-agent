# BaseRAG MCP Server

A FastMCP server that provides access to BaseRAG's retrieval-augmented generation capabilities through a standardized MCP interface. This server wraps the BaseRAG API and exposes key endpoints as MCP tools and resources.

## Features

- **RAG Context Retrieval:** Get relevant context for queries from the knowledge base.
- **Content Retrieval:** Access stored content by ID.
- **Content Chunks:** Examine how documents are chunked for retrieval.
- **RAG-enhanced Chat Completions:** Generate responses using knowledge from the document store.
- **ConfidentialMind Integration:**
  - Supports both local development and stack deployment modes
  - Automatically discovers BaseRAG API connection details from the stack
  - Continues operating even when BaseRAG API is not initially available
  - Background polling for BaseRAG API URL changes in stack deployment
- **Graceful Operation:** Can start without an initial API connection and connect later when available.

## Requirements

- Python 3.10+
- Access to BaseRAG API
- Dependencies:
  - `fastmcp>=2.0.0`
  - `aiohttp>=3.8.0`
  - `pydantic-settings>=2.0.0`
  - `confidentialmind_core` (for stack integration)

## Configuration

The server supports two operational modes:

### 1. Local Development Mode

Configure BaseRAG API connection details using **environment variables** or a `.env` file in the directory where you run the server.

**Required Environment Variables:**

- `BASERAG_API_URL`: The URL of the BaseRAG API (default: `http://localhost:8080`)
- `BASERAG_API_KEY`: API key for authentication (optional)

**Example `.env` file:**

```dotenv
BASERAG_API_URL=http://localhost:8080
BASERAG_API_KEY=your_api_key_here
```

### 2. Stack Deployment Mode

In stack deployment mode, the server automatically:

1. Registers a BaseRAG API connector with the ConfigManager
2. Discovers BaseRAG API connection URL from the stack
3. Continuously polls for URL changes in the background
4. 4. 4. 4. Attempts to connect when BaseRAG API becomes available

This enables seamless operation in container orchestration environments where the BaseRAG API might be ready after the server starts.

**Required Environment Variables:**

- `SERVICE_NAME`: Service identifier
- `CONNECTOR_ID`: Optional, defaults to "BASERAG_API"
- `CONFIDENTIAL_MIND_LOCAL_CONFIG`: Set to "False" to enable stack mode

## Running the Server

The server runs on port 8080 by default and uses Streamable HTTP transport (via /mcp endpoint).

To start the server:

```bash
# From the project root
python -m src.tools.baserag_mcp

# Or if inside the src/tools/baserag_mcp directory
python __main__.py
```

Once running, the MCP server will be accessible at `http://localhost:8080/mcp`.
A health check endpoint is available at `http://localhost:8080/health`.

## Exposed MCP Tools

The server exposes these MCP tools:

- `get_context`: Retrieve relevant context for a query from the knowledge base.
- `get_content`: Retrieve a specific content by ID.
- `get_content_chunks`: Get the chunks that a document has been split into.
- `chat_completion`: Generate a chat completion with retrieval-augmented generation.

## Exposed MCP Resources

The server exposes this MCP resource:

1. `baserag://resources`: Information about available BaseRAG endpoints and their status.

## Integration with FastMCP Agent

The BaseRAG MCP server works seamlessly with the FastMCP agent:

1. Start the BaseRAG MCP server
2. Configure the agent to connect to the server:
   - For CLI mode: Specify the path to the server in the config file
   - For API mode: Specify the server URL in the config file
3. The agent will discover available tools and resources

Example `config.json` for the agent in API mode:

```json
{
  "mcp_servers": {
    "baserag": "<http://localhost:8080/mcp>"
  }
}
```

## Security Considerations

- API keys are passed securely in headers
- In stack deployment, internal service communication is used
- Consider network policies to restrict access to the BaseRAG API

## Testing

To test the server functionality:

- Make sure the BaseRAG API application is running:
- Start the BaseRAG MCP server: `python -m src.tools.baserag_mcp`
- Run the test: `python -m tests.test_baserag_mcp`

## Handling Key Requirements

### Authentication

Authentication is handled through:

1. In local mode - Environment variables (`BASERAG_API_KEY`)
2. In stack mode - Internal Kubernetes service discovery

### Deployment Modes

The implementation supports three deployment modes:

1. **Stack Deployment**: Fetches BaseRAG API URL from ConfidentialMind stack
2. **Local API Mode**: Uses environment variables to configure HTTP transport
3. **Local CLI Mode**: Uses environment variables with stdio transport

### Error Handling & Resilience

The implementation includes several resilience features:

1. **Connection Retries**: Automatic reconnection with exponential backoff
2. **Background Polling**: Continuous polling for URL changes in stack mode
3. **Graceful Degradation**: Operation with partial service availability
4. **Error Recovery**: Detailed error handling and reporting

### Clean Code Structure

The code follows clean separation of concerns:

1. **Settings**: Configuration management (`settings.py`)
2. **Connectors**: Stack integration (`connectors.py`)
3. **Connection Manager**: API session management (`connection_manager.py`)
4. **API Client**: BaseRAG API interaction (`api_client.py`)
5. **Server**: FastMCP tools and resources (`server.py`)
6. **Entry Point**: Server startup and shutdown (`__main__.py`)

## 4. Testing Strategy

I recommend the following testing approach:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test connectivity with BaseRAG API
3. **End-to-End Tests**: Test complete workflows with FastMCP agent

## 5. Conclusion

This implementation provides a robust FastMCP server that wraps the BaseRAG API, supporting all required endpoints while following the established patterns from the existing codebase. The server supports both local development and stack deployment, with proper error handling and resilience features.

The code structure is clean, with clear separation of concerns, and follows the same patterns used in the PostgreSQL MCP server implementation.
