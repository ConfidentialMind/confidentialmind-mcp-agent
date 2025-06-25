# Observability Guide

This guide covers the comprehensive observability features implemented across the ConfidentialMind MCP Agent monorepo, including structured JSON logging, distributed tracing, and OpenTelemetry integration.

## Overview

The observability stack provides:

- **Structured JSON Logging**: All logs are output in JSON format compatible with OpenTelemetry Collector
- **Distributed Tracing**: Request tracing across service boundaries with span hierarchies
- **Automatic Instrumentation**: Decorators for seamless observability integration
- **Context Propagation**: Trace context flows through async operations and HTTP requests
- **Performance Metrics**: Duration tracking and operation timing
- **Error Tracking**: Comprehensive error logging with stack traces and context

## Architecture

### Core Components

1. **Shared Logging Module** (`src/shared/logging/`)

   - Centralized logging configuration
   - Trace context management
   - Automatic instrumentation decorators

2. **Trace Context System**

   - Uses Python's `contextvars` for async-safe context propagation
   - Maintains trace/span relationships across async boundaries
   - Supports OpenTelemetry-compatible headers

3. **Structured Loggers**
   - Component-specific loggers (e.g., `agent.core`, `postgres.mcp`, `baserag.mcp`)
   - Consistent log format across all services
   - Automatic enrichment with trace context

## Configuration

### Environment Variables

```bash
# Enable debug output (human-readable in dev, JSON in prod)
DEBUG=true

# Service identification
SERVICE_NAME=mcp-agent
CONFIDENTIAL_MIND_LOCAL_CONFIG=false  # Enable stack deployment mode
```

### Automatic Setup

The logging system is automatically configured when any shared logging component is imported:

```python
from src.shared.logging import get_logger, traced_async

# Logger is automatically configured with structlog
logger = get_logger("my.component")
```

## Usage Patterns

### Basic Structured Logging

```python
from src.shared.logging import get_logger

logger = get_logger("agent.core")

# Standard logging with automatic JSON formatting
logger.info("Starting operation")
logger.error("Operation failed", error="Connection timeout")

# Structured logging with data
logger.info(
    "Database query completed",
    event_type="db.query.complete",
    data={
        "query_type": "SELECT",
        "duration_ms": 150,
        "row_count": 42
    }
)
```

### Distributed Tracing

#### Initializing Trace Context

```python
from src.shared.logging import TraceContext

# Initialize trace context (typically at request entry)
trace_info = TraceContext.initialize(
    session_id="user-session-123",
    api_key="api-key-hash",
    origin_service="agent-api"
)

# From HTTP headers (for service-to-service calls)
trace_info = TraceContext.from_headers(request.headers, api_key="key")
```

#### Function Tracing with Decorators

```python
from src.shared.logging import traced_async, traced

# Async function tracing
@traced_async("agent.workflow", "agent.core")
async def process_query(query: str) -> str:
    # Function implementation
    return result

# Sync function tracing
@traced("db.operation", "database")
def validate_schema(schema: dict) -> bool:
    # Function implementation
    return True

# Custom event types and attribute extraction
@traced_async("mcp.call", "mcp.transport", extract_args=True)
async def call_mcp_tool(server_id: str, tool_name: str, args: dict):
    # Automatically extracts server_id, tool_name from function args
    return await mcp_client.call_tool(tool_name, args)
```

#### Manual Operation Logging

```python
from src.shared.logging import log_operation

# Context manager for operation timing
with log_operation(
    "database.query",
    "database",
    data={"table": "users", "operation": "SELECT"}
):
    result = await execute_query("SELECT * FROM users")
```

### HTTP Request Tracing

The system automatically propagates trace context through HTTP requests:

```python
# In FastAPI endpoints
@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    # Initialize trace context from request
    session_id = get_session_id(request)
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    trace_info = TraceContext.initialize(
        session_id=session_id,
        api_key=api_key,
        origin_service="agent-api"
    )

    # Trace context automatically flows to all decorated functions
    result = await process_chat_request(request)

    # Clear context when done
    TraceContext.clear()
```

### MCP Server Integration

MCP servers automatically extract and use trace context:

```python
from src.shared.logging import TraceContext, get_logger

structlog_logger = get_logger("postgres.mcp")

def extract_trace_headers(ctx: Context) -> Dict[str, str]:
    """Extract trace headers from FastMCP context."""
    trace_headers = {}
    if hasattr(ctx, "request_context") and hasattr(ctx.request_context, "headers"):
        headers = ctx.request_context.headers
        trace_headers = {
            "trace_id": headers.get("X-Trace-ID"),
            "span_id": headers.get("X-Span-ID"),
            "parent_span_id": headers.get("X-Parent-Span-ID"),
            "session_id": headers.get("X-Session-ID"),
            "origin_service": headers.get("X-Origin-Service"),
        }
    return trace_headers

@mcp_server.tool()
async def execute_sql(sql_query: str, ctx: Context) -> list[dict]:
    # Extract and initialize trace context
    trace_headers = extract_trace_headers(ctx)
    if trace_headers.get("trace_id"):
        TraceContext.initialize(
            session_id=trace_headers.get("session_id", "unknown"),
            trace_id=trace_headers.get("trace_id"),
            origin_service=trace_headers.get("origin_service", "unknown"),
        )

    # Structured logging with trace context
    structlog_logger.info(
        "Starting SQL query execution",
        event_type="postgres.query.start",
        data={
            "query_preview": sql_query[:100] + "..." if len(sql_query) > 100 else sql_query,
            "query_length": len(sql_query),
        },
    )

    # Execute operation...
```

## Log Schema

### Standard Fields

Every log entry includes:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "logger": "agent.core",
  "event": "Operation completed successfully",
  "event_type": "agent.workflow.complete",

  // Trace context (when available)
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "span_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "parent_span_id": "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
  "session_id": "user-session-123",
  "api_key_hash": "abc123...",
  "origin_service": "agent-api",

  // Operation-specific data
  "data": {
    "duration_ms": 150.5,
    "success": true,
    "custom_field": "value"
  }
}
```

### Event Types

Standard event type patterns:

- `{component}.{operation}.{stage}` - e.g., `agent.workflow.start`, `db.query.complete`
- `{service}.{function}.{result}` - e.g., `postgres.query.error`, `mcp.transport.success`

Common event types:

- `*.start` - Operation initiation
- `*.complete` - Operation completion (with success=true/false)
- `*.error` - Error conditions
- `*.metrics` - Performance metrics
- `*.discovery.*` - Service/tool discovery events

### Error Schema

Error logs include additional fields:

```json
{
  "level": "error",
  "event": "Database query failed",
  "event_type": "postgres.query.error",
  "success": false,
  "error": "Connection timeout after 30 seconds",
  "error_type": "ConnectionTimeoutError",
  "data": {
    "query_preview": "SELECT * FROM users WHERE...",
    "retry_count": 3,
    "duration_ms": 30000
  }
}
```

## Component-Specific Logging

### Agent Core

```python
# Agent workflow events
logger.info(
    "Starting agent workflow",
    event_type="agent.workflow.metadata",
    data={
        "query_preview": query[:100],
        "session_type": "conversation",
        "connections": {
            "database": True,
            "llm": True,
            "tools": 2
        }
    }
)
```

### Database Operations

```python
# Database connection events
self.logger.info(
    "Message saved successfully",
    event_type="db.message.saved",
    data={
        "session_id": session_id,
        "message_order": 1,
        "role": "user",
        "content_length": 150,
        "has_trace": True
    }
)
```

### MCP Transport

```python
# MCP call events
self.logger.info(
    "MCP transport configured successfully",
    event_type="transport.configured",
    data={
        "server_id": "postgres",
        "transport_type": "streamable-http",
        "url": "http://localhost:8080/mcp"
    }
)
```

### MCP Server Tools

```python
# Tool execution events
structlog_logger.info(
    "SQL query completed successfully",
    event_type="postgres.query.complete",
    duration_ms=duration_ms,
    success=True,
    data={
        "row_count": len(results),
        "query_preview": sql_query[:100]
    }
)
```

## Performance Monitoring

### Duration Tracking

All decorated functions automatically track execution time:

```python
@traced_async("expensive.operation")
async def complex_computation():
    # Automatically logs duration_ms when complete
    pass
```

### Custom Metrics

```python
# Manual timing with context manager
with log_operation("custom.metric", data={"batch_size": 100}) as op:
    process_batch()
# Automatically logs duration and success/failure
```

### Streaming Metrics

```python
# Track streaming operations
structlog_logger.info(
    "Streaming completed",
    event_type="agent.streaming.metrics",
    data={
        "chunk_count": 42,
        "streaming_duration_ms": 1500,
        "total_duration_ms": 2000
    }
)
```

## Debugging and Development

### Debug Mode

Set `DEBUG=true` to enable human-readable console output during development:

```bash
2024-01-15 10:30:45 [info] Starting agent workflow
├── trace_id: 550e8400-e29b-41d4-a716-446655440000
├── span_id: 6ba7b810-9dad-11d1-80b4-00c04fd430c8
└── data: {"query_preview": "What tools are available?"}
```

### Production Mode

In production (`DEBUG=false`), logs are output as structured JSON:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "logger": "agent.core",
  "event": "Starting agent workflow",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": { "query_preview": "What tools are available?" }
}
```

## OpenTelemetry Integration

### Collector Configuration

The logs are compatible with OpenTelemetry Collector. Example collector config:

```yaml
receivers:
  filelog:
    include: ["/var/log/app/*.log"]
    operators:
      - type: json_parser
        timestamp:
          parse_from: attributes.timestamp
          layout: "2006-01-02T15:04:05.000Z"

processors:
  batch:

exporters:
  logging:
    loglevel: debug
  jaeger:
    endpoint: http://jaeger:14250
    tls:
      insecure: true

service:
  pipelines:
    logs:
      receivers: [filelog]
      processors: [batch]
      exporters: [logging]
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger]
```

### Trace Correlation

Logs automatically correlate with distributed traces through:

- Consistent trace_id and span_id fields
- Parent-child span relationships
- Cross-service trace propagation via HTTP headers

## Best Practices

### 1. Use Appropriate Event Types

```python
# Good: Descriptive, hierarchical event types
event_type="postgres.query.validation_error"
event_type="agent.discovery.complete"

# Avoid: Generic or unclear event types
event_type="error"
event_type="done"
```

### 2. Include Relevant Context

```python
# Good: Include operation context
logger.info(
    "Query completed",
    event_type="db.query.complete",
    data={
        "table": "users",
        "row_count": 150,
        "duration_ms": 45
    }
)

# Avoid: Missing context
logger.info("Query completed")
```

### 3. Use Structured Data

```python
# Good: Structured data object
data={
    "server_count": 3,
    "success_rate": 0.95,
    "error_types": ["timeout", "connection"]
}

# Avoid: String formatting
f"Found {count} servers with {rate}% success"
```

### 4. Handle Sensitive Data

```python
# Good: Hash or truncate sensitive data
data={
    "user_id": user_id,
    "api_key_hash": hash_api_key(api_key)[:16],
    "query_preview": query[:100] + "..."
}

# Avoid: Logging full sensitive data
data={"api_key": full_api_key, "query": full_query}
```

### 5. Use Decorators for Function Tracing

```python
# Good: Automatic tracing with context
@traced_async("service.operation", "service.component")
async def important_operation(param1: str, param2: int):
    return result

# Supplement with manual logging for specific events
logger.info("Special condition detected", event_type="service.condition.detected")
```

## Troubleshooting

### Common Issues

1. **Missing Trace Context**

   - Ensure `TraceContext.initialize()` is called at request entry points
   - Check that async operations properly propagate context

2. **Log Format Issues**

   - Verify `DEBUG` environment variable setting
   - Check structlog configuration in shared logging module

3. **Performance Impact**
   - Tracing decorators have minimal overhead (~1-5ms per operation)
   - Use `extract_args=False` for high-frequency functions if needed

### Debugging Tips

1. **Enable Debug Logging**

   ```bash
   DEBUG=true python -m src.agent.main serve
   ```

2. **Check Trace Propagation**

   ```python
   from src.shared.logging import get_current_trace

   trace = get_current_trace()
   if trace:
       print(f"Current trace: {trace.trace_id}")
   else:
       print("No trace context available")
   ```

3. **Validate Log Output**

   ```bash
   # Check JSON formatting
   python -m src.agent.main serve 2>&1 | jq .

   # Filter specific event types
   python -m src.agent.main serve 2>&1 | jq 'select(.event_type | startswith("agent."))'
   ```

## Migration Guide

### Adding Observability to Existing Code

1. **Import shared logging**

   ```python
   from src.shared.logging import get_logger, traced_async
   ```

2. **Replace existing loggers**

   ```python
   # Old
   import logging
   logger = logging.getLogger(__name__)

   # New
   from src.shared.logging import get_logger
   logger = get_logger("component.name")
   ```

3. **Add tracing decorators**

   ```python
   # Add to important functions
   @traced_async("component.operation", "component.name")
   async def important_function():
       pass
   ```

4. **Update log statements**

   ```python
   # Old
   logger.info(f"Operation completed in {duration}ms")

   # New
   logger.info(
       "Operation completed",
       event_type="component.operation.complete",
       data={"duration_ms": duration}
   )
   ```

This observability system provides comprehensive insight into system behavior while maintaining performance and ease of use. The structured JSON format ensures compatibility with modern observability platforms and enables powerful log analysis and correlation.
