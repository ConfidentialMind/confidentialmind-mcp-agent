# Building FastMCP Servers with ConfidentialMind Integration

This guide walks you through creating a FastMCP server that integrates with the ConfidentialMind stack. FastMCP provides a standardized way to expose tools and resources that can be used by AI agents, while ConfidentialMind provides the infrastructure to deploy, connect, and manage these services in both development and production environments.

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Project Structure](#project-structure)
- [Setting Up Your FastMCP Server](#setting-up-your-fastmcp-server)
  - [Step 1: Basic Server Setup](#step-1-basic-server-setup)
  - [Step 2: Connection Management](#step-2-connection-management)
  - [Step 3: Server Configuration](#step-3-server-configuration)
  - [Step 4: Tools and Resources](#step-4-tools-and-resources)
  - [Step 5: Running the Server](#step-5-running-the-server)
- [Local vs. Stack Deployment](#local-vs-stack-deployment)
- [Testing Your Server](#testing-your-server)
- [Advanced Patterns](#advanced-patterns)
- [Full Example: Database MCP Server](#full-example-database-mcp-server)

## Overview

FastMCP provides a framework for creating MCP (Model Context Protocol) servers that expose tools and resources to be used by AI agents. The ConfidentialMind stack adds configuration management, deployment capabilities, and service discovery to create a production-ready ecosystem of interconnected services.

This guide will show you how to build MCP servers that:

- Work in both local development and stack deployment modes
- Connect to various backend services (databases, LLMs, etc.)
- Register with the ConfidentialMind configuration manager
- Gracefully handle service availability
- Follow best practices for resilience

## Key Concepts

Before diving into the code, it's important to understand these key concepts:

**FastMCP Concepts:**

- **Tools**: Functions that can be called by agents to perform actions
- **Resources**: Data that agents can access (schemas, configs, etc.)
- **Transport**: How clients communicate with the server (STDIO, Streamable HTTP)

**ConfidentialMind Concepts:**

- **ConfigManager**: Singleton that manages service configuration and connections
- **Connectors**: Service connection configurations (databases, LLMs, etc.)
- **Stack Deployment**: Production deployment mode where services discover each other
- **Local Development**: Development mode where services are configured locally

## Project Structure

A typical FastMCP server project with ConfidentialMind integration might have this structure:

```
my_fastmcp_server/
├── __init__.py
├── __main__.py           # Entry point for running the server
├── connection_manager.py # Manages connections to backend services
├── connectors.py         # Registers and configures connectors
├── server.py             # FastMCP server definition with tools/resources
├── settings.py           # Configuration settings
└── README.md             # Documentation
```

## Setting Up Your FastMCP Server

### Step 1: Basic Server Setup

First, let's set up a basic FastMCP server with ConfidentialMind integration:

**settings.py**:

```python
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class MyServiceSettings(BaseSettings):
    """Configuration settings with ConfidentialMind support."""

    # Define config for loading from environment variables
    model_config = SettingsConfigDict(env_prefix="MYSERVICE_", env_file=".env", extra="ignore")

    # Service settings
    name: str = Field(default="my-fastmcp-service")
    description: str = Field(default="My FastMCP Service")

    # Backend service connection details
    host: str = Field(default="localhost")
    port: int = Field(default=8080)
    api_key: Optional[str] = Field(default=None)

    # ConfidentialMind integration
    connector_id: str = Field(default="MYSERVICE")
    use_sdk_connector: bool = Field(default=False)

    @property
    def is_stack_deployment(self) -> bool:
        """Determine if running in stack deployment mode."""
        return os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"

# Single settings instance
settings = MyServiceSettings()
```

### Step 2: Connection Management

Next, create a connection manager to handle backend service connections:

**connection_manager.py**:

```python
import asyncio
import logging
import os
from typing import Optional

from confidentialmind_core.config_manager import load_environment

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages service connections with ConfidentialMind integration."""

    _current_url: Optional[str] = None
    _background_poll_task: Optional[asyncio.Task] = None
    _initialized: bool = False
    _is_connected: bool = False

    @classmethod
    async def initialize(cls) -> bool:
        """Initialize connection manager and discover URLs if needed."""
        if cls._initialized:
            logger.debug("ConnectionManager: Already initialized")
            return True

        # Determine deployment mode
        load_environment()
        cls._is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"ConnectionManager: Initializing in {'stack' if cls._is_stack_deployment else 'local'} mode"
        )

        # Initialize the connector manager
        from .connectors import MyServiceConnectorManager
        from .settings import settings

        cls._connector_manager = MyServiceConnectorManager(settings.connector_id)
        try:
            # Always register connectors in stack mode, but make it optional in local mode
            register_connectors = cls._is_stack_deployment
            await cls._connector_manager.initialize(register_connectors=register_connectors)

            # Try to get a URL initially
            cls._current_url = await cls._connector_manager.fetch_service_url()
            if cls._current_url:
                logger.info(f"ConnectionManager: Initial URL available: {cls._current_url}")
                cls._is_connected = True
            else:
                logger.info("ConnectionManager: No initial URL available")
                cls._is_connected = False

            # Start background polling for URL changes in stack mode
            if cls._is_stack_deployment:
                await cls._start_background_polling()

            cls._initialized = True
            return True
        except Exception as e:
            logger.error(f"ConnectionManager: Error during initialization: {e}")
            return False

    @classmethod
    async def _start_background_polling(cls):
        """Start background polling for URL changes."""
        if cls._background_poll_task is not None and not cls._background_poll_task.done():
            logger.info("ConnectionManager: URL polling task already running")
            return

        logger.info("ConnectionManager: Starting background polling for URL changes")
        cls._background_poll_task = asyncio.create_task(cls._poll_for_url_changes())

    @classmethod
    async def _poll_for_url_changes(cls):
        """Poll for URL changes and update connections as needed."""
        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries

        while True:
            try:
                url = await cls._connector_manager.fetch_service_url()

                # Log at appropriate intervals
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    if url:
                        logger.debug(f"ConnectionManager: Poll {retry_count}: Found URL {url}")
                    else:
                        logger.debug(f"ConnectionManager: Poll {retry_count}: No URL available")

                # Handle URL changes
                if url and url != cls._current_url:
                    logger.info(
                        f"ConnectionManager: URL changed from {cls._current_url or 'None'} to {url}"
                    )
                    cls._current_url = url
                    cls._is_connected = True
                    # Update connection if URL changed - implement your reconnection logic here
            except Exception as e:
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.error(f"ConnectionManager: Error polling for URL changes: {e}")

            # Increment and wait
            retry_count += 1
            # Exponential backoff with a maximum wait time
            wait_time = min(30, 5 * (1.5 ** min(retry_count, 5)))
            await asyncio.sleep(wait_time)

    @classmethod
    def is_connected(cls) -> bool:
        """Check if service is currently connected."""
        return cls._is_connected

    @classmethod
    def get_current_url(cls) -> Optional[str]:
        """Get the current service URL."""
        return cls._current_url

    @classmethod
    async def close(cls):
        """Close connections and stop background polling."""
        # Cancel background polling task
        if cls._background_poll_task and not cls._background_poll_task.done():
            logger.info("ConnectionManager: Cancelling background polling task")
            cls._background_poll_task.cancel()
            try:
                await cls._background_poll_task
            except asyncio.CancelledError:
                pass
            cls._background_poll_task = None

        # Implement any additional cleanup needed for your specific service
        cls._is_connected = False
        logger.info("ConnectionManager: Connections closed")
```

### Step 3: Server Configuration

Create a connectors module to register with the ConfidentialMind ConfigManager:

**connectors.py**:

```python
import asyncio
import logging
import os
from typing import Optional

from confidentialmind_core.config_manager import (
    ConfigManager,
    ConnectorSchema,
    get_api_parameters,
    load_environment,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MyServiceConfig(BaseModel):
    """Minimal config for the MCP server."""
    name: str = "my-fastmcp-server"
    description: str = "My FastMCP Server"

class MyServiceConnectorManager:
    """Manages connector configurations for my service."""

    def __init__(self, connector_id: str = "MYSERVICE"):
        """
        Initialize the connector manager.

        Args:
            connector_id: The connector ID to use for service access
        """
        self.initialized = False
        self.connector_id = connector_id

        load_environment()
        self.is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"ServiceConnectorManager: Initializing in {'stack deployment' if self.is_stack_deployment else 'local development'} mode"
        )
        self._background_fetch_task = None

    async def initialize(self, register_connectors=True):
        """
        Initialize connector configuration for the MCP server.

        Args:
            register_connectors: Whether to register connectors with ConfigManager
        """
        if self.initialized:
            logger.info("ServiceConnectorManager: Already initialized")
            return

        if register_connectors and self.is_stack_deployment:
            # Register connector with the ConfigManager in stack deployment mode
            try:
                logger.info(
                    "ServiceConnectorManager: Registering connector with ConfigManager"
                )
                config_manager = ConfigManager()
                connectors = [
                    # Define the appropriate connector type for your service
                    ConnectorSchema(
                        type="api",  # or "llm", "database", etc.
                        label="My Service",
                        config_id=self.connector_id,
                    ),
                ]
                config_manager.init_manager(
                    config_model=MyServiceConfig(),
                    connectors=connectors,
                )
                logger.info(
                    f"ServiceConnectorManager: Registered connector with config_id={self.connector_id}"
                )
            except Exception as e:
                logger.error(f"ServiceConnectorManager: Error registering connectors: {e}")
                # Continue without raising - allow server to initialize without connectors

        self.initialized = True

    async def fetch_service_url(self) -> Optional[str]:
        """Fetch service URL using the appropriate method based on deployment mode."""
        try:
            url, _ = get_api_parameters(self.connector_id)
            if url:
                logger.debug(
                    f"ServiceConnectorManager: Successfully retrieved URL: {url}"
                )
                return url
            logger.debug(f"ServiceConnectorManager: No URL found for {self.connector_id}")
            return None
        except Exception as e:
            logger.error(f"ServiceConnectorManager: Error fetching URL: {e}")
            return None
```

### Step 4: Tools and Resources

Now create the FastMCP server with tools and resources:

**server.py**:

```python
import asyncio
import datetime
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from confidentialmind_core.config_manager import load_environment
from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .connection_manager import ConnectionManager
from .settings import settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Handles service setup and teardown with ConfidentialMind support.

    This function allows the server to start without a connection,
    which is required for stack deployment scenarios where services
    might be connected after the server is running.
    """
    state = {"connected": False}

    try:
        # Initialize environment
        load_environment()

        # Initialize ConnectionManager (this will handle all connector registration and URL polling)
        try:
            # Run with timeout protection
            await asyncio.wait_for(ConnectionManager.initialize(), timeout=5.0)
            logger.info("Connection manager initialized")
        except asyncio.TimeoutError:
            logger.warning(
                "ConnectionManager initialization timed out, continuing without connection"
            )
        except Exception as e:
            logger.error(f"Error initializing connection manager: {e}")

        # Check connection state
        state["connected"] = ConnectionManager.is_connected()
        if state["connected"]:
            logger.info("Service connection established")
        else:
            logger.info("No service connection available yet. Server will start anyway.")

        # Quickly yield control back to FastMCP
        yield state

    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        logger.info(f"Server startup interrupted: {type(e).__name__}")
        raise
    finally:
        logger.info("Shutting down server, closing connections")
        try:
            await ConnectionManager.close()
            logger.info("Connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# Instantiate the FastMCP server with the lifespan manager
mcp_server = FastMCP(
    name="My FastMCP Server",
    instructions="Provides access to my service.",
    lifespan=lifespan,
)

@mcp_server.resource("my-service://info")
async def get_service_info(ctx: Context) -> Dict[str, Any]:
    """
    Retrieves information about the service.
    """
    connected = ctx.request_context.lifespan_context.get("connected", False)
    if not connected:
        logger.error("Service not connected.")
        return {"status": "unavailable", "message": "Service connection is not available yet."}

    try:
        # Return service information
        return {
            "status": "available",
            "service_name": settings.name,
            "service_description": settings.description,
            "url": ConnectionManager.get_current_url()
        }
    except Exception as e:
        logger.error(f"Failed to get service info: {e}")
        return {"status": "error", "message": f"Could not retrieve service info: {e}"}

@mcp_server.tool()
async def my_tool(input_param: str, ctx: Context) -> str:
    """
    Example tool that performs an action with the service.

    Args:
        input_param: An input parameter for the tool.
        ctx: The MCP context (automatically injected).

    Returns:
        A result string.
    """
    connected = ctx.request_context.lifespan_context.get("connected", False)
    if not connected:
        logger.error("Service not connected for my_tool.")
        raise RuntimeError("Service connection is not available yet. Please try again later.")

    try:
        # Implement your tool functionality here
        await ctx.info(f"Processing input: {input_param}")
        result = f"Processed: {input_param}"
        return result
    except Exception as e:
        logger.error(f"Failed to execute tool: {e}")
        raise RuntimeError(f"Could not execute tool: {e}")

@mcp_server.custom_route("/health", methods=["GET"])
async def http_health_endpoint(request: Request) -> JSONResponse:
    """
    HTTP health endpoint for Kubernetes liveness and readiness probes.

    This endpoint always returns a 200 status code when the server is running,
    regardless of connection status, allowing the pod to stay alive
    while waiting for the service to be configured.
    """
    # Create health status response
    health_status = {
        "status": "healthy",
        "service": settings.name,
        "service_connected": ConnectionManager.is_connected(),
        "server_mode": "stack_deployment" if settings.is_stack_deployment else "local",
        "server_time": datetime.datetime.now().isoformat(),
        "connector_id": settings.connector_id,
    }

    # Always return a 200 status code to keep the pod alive
    return JSONResponse(content=health_status)
```

### Step 5: Running the Server

Finally, create the entry point for running the server:

****main**.py**:

```python
import asyncio
import logging
import os
import signal
import sys
from threading import Event

from confidentialmind_core.config_manager import load_environment
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from .connection_manager import ConnectionManager
from .server import http_health_endpoint, mcp_server
from .settings import settings

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Event for signaling shutdown
shutdown_event = Event()
main_task = None

def signal_handler(sig, frame):
    """Handle interrupt signals by cancelling the main task."""
    global shutdown_event, main_task

    if shutdown_event.is_set():
        logger.warning("Forced shutdown requested, exiting immediately!")
        sys.exit(1)

    logger.info("Shutdown requested, gracefully shutting down...")
    shutdown_event.set()

    # If we have access to the main task, cancel it
    if main_task and not main_task.done():
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(main_task.cancel)
        except Exception as e:
            logger.error(f"Error cancelling main task: {e}")
            # Force exit if we can't cancel gracefully
            sys.exit(1)

async def root_health_check(request):
    """Root level health check that's always accessible."""
    return await http_health_endpoint(request)

async def run_server(transport_type, **kwargs):
    """
    Run the MCP server with shutdown handling and proactive initialization.

    This function initializes the ConnectionManager before starting the server
    to ensure that connections can be established in the background
    even if not available at startup.
    """
    try:
        # Proactively initialize ConnectionManager before server starts
        logger.info("Proactively initializing ConnectionManager...")
        await ConnectionManager.initialize()
        logger.info("ConnectionManager initialized successfully")

        if transport_type == "streamable-http":
            # For HTTP-based transports, create a combined app with root-level health endpoint
            # Get the ASGI app for the MCP server
            mcp_app = mcp_server.http_app(path="")

            # Create a Starlette app with both the MCP app and a root-level health endpoint
            combined_app = Starlette(
                routes=[
                    Route("/health", endpoint=root_health_check, methods=["GET"]),
                    Mount("/", app=mcp_app),
                ],
                lifespan=mcp_app.router.lifespan_context,  # Important: use MCP server's lifespan
            )

            # Use additional kwargs (port, host, etc.) passed to this function
            port = kwargs.get("port", 8080)
            host = kwargs.get("host", "0.0.0.0")
            log_level = kwargs.get("log_level", "info")

            import uvicorn

            config = uvicorn.Config(combined_app, host=host, port=port, log_level=log_level)
            server = uvicorn.Server(config)
            await server.serve()
        elif transport_type == "stdio":
            # For STDIO transport, use the normal run method
            return await mcp_server.run_async(transport=transport_type, **kwargs)
        else:
            logger.warning(
                f"Unsupported transport type: {transport_type}, defaulting to streamable-http"
            )
            # Fall back to streamable-http for any unknown transport type
            return await run_server("streamable-http", **kwargs)
    except asyncio.CancelledError:
        logger.info("Server task cancelled, shutting down gracefully")
    except Exception as e:
        logger.critical(f"Server failed to run: {e}", exc_info=True)
    finally:
        # Ensure ConnectionManager is closed on shutdown
        try:
            await ConnectionManager.close()
            logger.info("ConnectionManager closed during shutdown")
        except Exception as e:
            logger.error(f"Error closing ConnectionManager: {e}")
        logger.info("Server shutdown complete.")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize environment variables and check deployment mode
    load_environment()

    # Determine deployment mode
    is_stack_deployment = (
        os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
    )

    if is_stack_deployment:
        settings.use_sdk_connector = True
        logger.info("Running in stack deployment mode with SDK connector integration")
    else:
        logger.info(f"Running in local mode")

    try:
        # Modified logic for transport selection
        # Only use stdio if explicitly requested or when launched as a subprocess
        use_stdio = (
            "FastMCP_TRANSPORT" in os.environ
            and os.environ["FastMCP_TRANSPORT"] == "stdio"
            or "--stdio" in sys.argv
        )

        # Override stdio mode if streamable-http is explicitly requested
        use_streamable = "--streamable-http" in sys.argv
        if use_streamable:
            use_stdio = False

        loop = asyncio.get_event_loop()
        if use_stdio:
            logger.info("Using stdio transport for agent communication")
            main_task = asyncio.ensure_future(run_server("stdio"))
        else:
            # Default to Streamable HTTP transport for standalone server mode
            logger.info("Using Streamable HTTP transport on port 8080")
            main_task = asyncio.ensure_future(
                run_server("streamable-http", port=8080, log_level="debug")
            )

        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        if "loop" in locals() and loop.is_running():
            try:
                tasks = asyncio.all_tasks(loop=loop)
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                loop.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
```

## Local vs. Stack Deployment

Your server now supports two deployment modes:

### Local Development Mode

In local development mode:

- Configuration loaded from `.env` file and environment variables
- Connection details retrieved from environment with pattern `{CONFIG_ID}_URL` and `{CONFIG_ID}_APIKEY`
- Services can be configured independently
- Set `CONFIDENTIAL_MIND_LOCAL_CONFIG=True` in your environment

Example `.env` file:

```
MYSERVICE_HOST=localhost
MYSERVICE_PORT=8000
MYSERVICE_API_KEY=your_api_key
MYSERVICE_CONNECTOR_ID=my-service

# Connection details for local development
MY-SERVICE_URL=http://localhost:8000
MY-SERVICE_APIKEY=your_api_key
```

### Stack Deployment Mode

In stack deployment mode:

- Automatically registers connectors with ConfigManager
- Discovers service URLs dynamically from the stack
- Continuously polls for URL changes in the background
- Operates gracefully even when services become available after startup
- Set `CONFIDENTIAL_MIND_LOCAL_CONFIG=False` in your environment

## Testing Your Server

Create a simple test script to verify your server works:

```python
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def test_mcp_server():
    # Create a client with streamable HTTP transport
    transport = StreamableHttpTransport(url="http://localhost:8080/mcp")
    client = Client(transport)

    # Connect to the server
    async with client:
        # List available resources
        resources = await client.list_resources()
        print(f"Available resources: {resources}")

        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        # Read a resource
        service_info = await client.read_resource("my-service://info")
        print(f"Service info: {service_info}")

        # Call a tool
        result = await client.call_tool("my_tool", {"input_param": "test"})
        print(f"Tool result: {result}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
```

## Advanced Patterns

### Connection Pooling

For database connections or other resources that benefit from connection pooling:

```python
import asyncpg

class DatabaseConnectionManager:
    _pool: Optional[asyncpg.pool.Pool] = None

    @classmethod
    async def create_pool(cls) -> Optional[asyncpg.Pool]:
        if cls._pool is not None:
            return cls._pool

        # Get connection string using the URL
        url = await cls.fetch_url()
        connection_string = settings.get_connection_string(url)

        # Create the pool
        cls._pool = await asyncpg.create_pool(
            dsn=connection_string,
            min_size=1,
            max_size=5,
            command_timeout=60.0,
        )

        return cls._pool

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
```

### Graceful Error Handling

When working with services that might not be available immediately:

```python
async def execute_action(params):
    if not ConnectionManager.is_connected():
        logger.warning("Service not connected, retrying with backoff")
        # Use exponential backoff
        for attempt in range(5):
            await asyncio.sleep(2 ** attempt)
            if ConnectionManager.is_connected():
                break

        if not ConnectionManager.is_connected():
            raise RuntimeError("Service connection not available")

    # Proceed with the action
    return await perform_actual_action(params)
```

### Multiple Service Connectors

For services that need to connect to multiple backends:

```python
# In your settings.py
class MultiServiceSettings(BaseSettings):
    database_connector_id: str = "DATABASE"
    llm_connector_id: str = "LLM"
    api_connector_id: str = "API"

# In your connectors.py
connectors = [
    ConnectorSchema(type="database", label="Database", config_id="DATABASE"),
    ConnectorSchema(type="llm", label="Language Model", config_id="LLM"),
    ConnectorSchema(type="api", label="API Service", config_id="API"),
]

config_manager.init_manager(
    config_model=ServiceConfig(),
    connectors=connectors,
)
```

## Full Example: Database MCP Server

Here's a simplified example of a full database MCP server implementation:

```python
# server.py
import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from confidentialmind_core.config_manager import load_environment
from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

class DBConnectionManager:
    # Database connection management code here...
    @classmethod
    async def initialize(cls):
        # Initialize database connection
        pass

    @classmethod
    async def execute_query(cls, query):
        # Execute a database query
        pass

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    state = {"db_connected": False}
    try:
        # Initialize database connection
        await DBConnectionManager.initialize()
        state["db_connected"] = True
        yield state
    finally:
        # Clean up
        await DBConnectionManager.close()

# Create the server
db_server = FastMCP(
    name="Database MCP Server",
    instructions="Provides read-only access to a database.",
    lifespan=lifespan,
)

@db_server.resource("db://schemas")
async def get_schemas(ctx: Context) -> Dict[str, List[Dict]]:
    """Get database schemas."""
    if not ctx.request_context.lifespan_context.get("db_connected"):
        raise RuntimeError("Database not connected")

    # Return database schemas
    return await DBConnectionManager.get_schemas()

@db_server.tool()
async def execute_sql(query: str, ctx: Context) -> List[Dict]:
    """Execute a read-only SQL query."""
    if not ctx.request_context.lifespan_context.get("db_connected"):
        raise RuntimeError("Database not connected")

    # Execute the query and return results
    return await DBConnectionManager.execute_query(query)

@db_server.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "database_connected": DBConnectionManager.is_connected()
    })

# Run the server
if __name__ == "__main__":
    db_server.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

This creates a complete FastMCP server that exposes database schemas as resources and an SQL execution tool, with proper health checks and ConfidentialMind integration.

---

By following this guide, you've created a FastMCP server that works both locally and in the ConfidentialMind stack deployment, with proper connection management, error handling, and graceful operation.
