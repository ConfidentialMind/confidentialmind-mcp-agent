# BaseRAG MCP Server

A FastMCP server that provides access to BaseRAG's retrieval-augmented generation capabilities through a standardized MCP interface with comprehensive observability features. This server wraps the BaseRAG API and exposes key endpoints as MCP tools and resources while providing structured JSON logging and distributed tracing.

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
- **Comprehensive Observability:**
  - **Structured JSON Logging**: All operations logged in OpenTelemetry-compatible format
  - **Distributed Tracing**: Request correlation across service boundaries with BaseRAG API
  - **Performance Monitoring**: RAG operation timing and context retrieval metrics
  - **Error Context**: Rich error logging with query context and API response details
  - **Health Monitoring**: BaseRAG API health check integration with detailed logging
- **Graceful Operation:** Can start without an initial API connection and connect later when available.

## Requirements

- Python 3.10+
- Access to BaseRAG API
- Dependencies:
  - `fastmcp>=2.0.0`
  - `aiohttp>=3.8.0`
  - `pydantic-settings>=2.0.0`
  - `confidentialmind_core` (for stack integration)
  - `structlog>=25.3.0` (for observability)

## Configuration

The server supports two operational modes:

### 1. Local Development Mode

Configure BaseRAG API connection details using **environment variables** or a `.env` file in the directory where you run the server.

**Required Environment Variables:**

- `BASERAG_API_URL`: The URL of the BaseRAG API (default: `http://localhost:8080`)
- `BASERAG_API_KEY`: API key for authentication (optional)
- `DEBUG`: Set to "true" for human-readable logs (default: JSON format)

**Example `.env` file:**

```dotenv
BASERAG_API_URL=http://localhost:8080
BASERAG_API_KEY=your_api_key_here
DEBUG=true
```

### 2. Stack Deployment Mode

In stack deployment mode, the server automatically:

1. Registers a BaseRAG API connector with the ConfigManager
2. Discovers BaseRAG API connection URL from the stack
3. Continuously polls for URL changes in the background
4. Attempts to connect when BaseRAG API becomes available
5. Outputs structured JSON logs for OpenTelemetry Collector

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

The server exposes these MCP tools with full observability:

- `get_context`: Retrieve relevant context for a query from the knowledge base.
- `get_content`: Retrieve a specific content by ID.
- `get_content_chunks`: Get the chunks that a document has been split into.
- `chat_completion`: Generate a chat completion with retrieval-augmented generation.

## Exposed MCP Resources

The server exposes this MCP resource:

1. `baserag://resources`: Information about available BaseRAG endpoints and their status.

## Observability Features

The server provides comprehensive observability through structured logging and distributed tracing:

### Structured JSON Logging

All operations are logged in OpenTelemetry-compatible JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "logger": "baserag.mcp",
  "event": "Context retrieval completed",
  "event_type": "baserag.context.complete",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "span_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "session_id": "user-session-123",
  "duration_ms": 450.2,
  "success": true,
  "data": {
    "result_count": 5,
    "query_preview": "What are the benefits of...",
    "tool_name": "get_context"
  }
}
```

### Event Types

The server emits events with consistent taxonomy:

- `baserag.context.start` - Context retrieval initiated
- `baserag.context.complete` - Context retrieval completed (success/failure)
- `baserag.context.error` - Context retrieval error
- `baserag.chat.start` - Chat completion initiated
- `baserag.chat.complete` - Chat completion completed
- `baserag.chat.error` - Chat completion error
- `resource.baserag.requested` - BaseRAG resources accessed
- `health.check.requested` - Health check performed
- `service.init.complete` - Server initialization completed

### Distributed Tracing

Request traces flow from the agent through the MCP server to the BaseRAG API:

- Automatic trace context extraction from MCP requests
- Parent-child span relationships maintained
- Trace correlation across RAG operations
- Performance timing for all BaseRAG API interactions

### Debug vs Production Logging

**Development Mode** (`DEBUG=true`):

```bash
2024-01-15 10:30:45 [info] Context retrieval completed
├── trace_id: 550e8400-e29b-41d4-a716-446655440000
├── duration_ms: 450.2
└── data: {"result_count": 5, "query_preview": "What are the benefits..."}
```

**Production Mode** (`DEBUG=false`):

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "event": "Context retrieval completed",
  "event_type": "baserag.context.complete",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "duration_ms": 450.2,
  "success": true,
  "data": { "result_count": 5 }
}
```

## Integration with FastMCP Agent

The BaseRAG MCP server works seamlessly with the FastMCP agent and provides full observability integration:

1. Start the BaseRAG MCP server
2. Configure the agent to connect to the server:
   - For CLI mode: Specify the path to the server in the config file
   - For API mode: Specify the server URL in the config file
3. The agent will discover available tools and resources
4. All RAG operations are traced end-to-end with correlation IDs

Example `config.json` for the agent in API mode:

```json
{
  "mcp_servers": {
    "baserag": "http://localhost:8080/mcp"
  }
}
```

## Security Considerations

- API keys are passed securely in headers
- In stack deployment, internal service communication is used
- Consider network policies to restrict access to the BaseRAG API

## Testing

To test the server functionality:

- Make sure the BaseRAG API application is running
- Start the BaseRAG MCP server: `python -m src.tools.baserag_mcp`
- Run the test: `python -m tests.test_baserag_mcp`

## Monitoring and Troubleshooting

### Health Check Endpoint

The `/health` endpoint provides detailed status information:

```json
{
  "status": "healthy",
  "service": "baserag-mcp-server",
  "api_connected": true,
  "api_error": null,
  "server_mode": "stack_deployment",
  "server_time": "2024-01-15T10:30:45.123Z",
  "connector_id": "BASERAG_API"
}
```

### Log Analysis

RAG operation performance monitoring:

```bash
# Filter context retrieval performance logs
cat logs/baserag-mcp.log | jq 'select(.event_type == "baserag.context.complete") | {duration_ms, result_count, success}'

# Monitor chat completion metrics
cat logs/baserag-mcp.log | jq 'select(.event_type == "baserag.chat.complete") | {duration_ms, message_count, success}'

# Monitor API errors
cat logs/baserag-mcp.log | jq 'select(.event_type | endswith(".error")) | {error, error_type, data}'
```

### OpenTelemetry Integration

The structured logs are designed for OpenTelemetry Collector ingestion:

```yaml
# otel-collector.yml
receivers:
  filelog:
    include: ["/var/log/baserag-mcp/*.log"]
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
5. **Health Monitoring**: Continuous BaseRAG API health checking

### Clean Code Structure

The code follows clean separation of concerns:

1. **Settings**: Configuration management (`settings.py`)
2. **Connectors**: Stack integration (`connectors.py`)
3. **Connection Manager**: API session management (`connection_manager.py`)
4. **API Client**: BaseRAG API interaction (`api_client.py`)
5. **Server**: FastMCP tools and resources with observability (`server.py`)
6. **Entry Point**: Server startup and shutdown (`__main__.py`)

## Customization

To modify server settings:

- Change the port by editing the `run()` call in `__main__.py`
- Add additional tools by extending the `server.py` file with new `@mcp_server.tool()` decorated functions
- Enhance resource endpoints by adding more resource endpoints with `@mcp_server.resource()`
- Customize BaseRAG API connection settings through environment variables or `.env` file
- Add custom observability events by using the structured logger:

```python
from src.shared.logging import get_logger

structlog_logger = get_logger("baserag.mcp")

# Custom event logging
structlog_logger.info(
    "Custom RAG operation completed",
    event_type="baserag.custom.complete",
    data={"custom_field": "value", "context_count": 3}
)
```

## Testing Strategy

I recommend the following testing approach:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test connectivity with BaseRAG API
3. **End-to-End Tests**: Test complete workflows with FastMCP agent
4. **Observability Tests**: Verify log output and trace correlation

## Conclusion

This implementation provides a robust FastMCP server that wraps the BaseRAG API, supporting all required endpoints while following the established patterns from the existing codebase. The server supports both local development and stack deployment, with comprehensive observability features including:

- Structured JSON logging compatible with OpenTelemetry Collector
- Distributed tracing across all RAG operations
- Performance metrics and timing data
- Rich error context and debugging information
- Health monitoring and status reporting

The code structure is clean, with clear separation of concerns, and follows the same patterns used in the PostgreSQL MCP server implementation.

For complete observability documentation and best practices, see [guides/observability.md](../../guides/observability.md).
