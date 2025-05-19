# Understanding ConfidentialMind Connection Management

This technical document explains how connection management works in the ConfidentialMind ecosystem, focusing on the integration patterns between services, agents, and databases.

## Core Components

The connection management system consists of several key components:

1. **ConfigManager (Singleton)**: Central configuration management system
2. **ConnectorConfigManager**: Service-specific wrapper for ConfigManager
3. **ConnectionManager**: Database-specific connection management
4. **TransportManager**: MCP server communication management
5. **Database**: Connection pool management for PostgreSQL
6. **LLMConnector**: API client management for language models

## Connection Flow Diagram

The following diagram illustrates the typical connection flow in both local development and stack deployment modes:

```
┌─────────────────┐       ┌─────────────────┐      ┌─────────────────┐
│                 │       │                 │      │                 │
│  Application    │──────▶│  Config Manager │─────▶│   Connectors    │
│                 │       │                 │      │                 │
└─────────────────┘       └─────────────────┘      └─────────────────┘
        │                         │                        │
        │                         │                        │
        ▼                         ▼                        ▼
┌─────────────────┐       ┌─────────────────┐      ┌─────────────────┐
│                 │       │                 │      │                 │
│  Local Dev Mode │       │ Stack Mode      │      │ Service URLs    │
│  (.env file)    │       │ (Stack Manager) │      │ and Credentials │
│                 │       │                 │      │                 │
└─────────────────┘       └─────────────────┘      └─────────────────┘
```

## 1. ConfigManager (Singleton)

The `ConfigManager` is the core of ConfidentialMind's connection management system. It handles:

- Service discovery and registration
- Configuration management
- URL construction for various service types
- Real-time configuration updates via websockets

### Key Components:

```python
class ConfigManager(SingletonClass):
    def init_manager(
        self,
        config_model: Optional[BaseModel],
        connectors: Optional[list[ConnectorSchema]] = None,
        array_connectors: Optional[list[ArrayConnectorSchema]] = None,
    ):
        # Initialize settings and register connectors
```

### Connector Schema:

```python
class ConnectorSchema(BaseModel):
    """A schema representing a connector between services in the stack."""
    type: ConnectorType  # "llm", "bucket", "database", "api", "endpoint", "data_source", "agent_tool"
    label: str
    config_id: str
    stack_id: Optional[str] = None
```

### URL Construction:

```python
def getUrlForConnector(self, config_id: str) -> Union[str, List[str], None]:
    """Construct service URL based on connector config_id."""
    stack_ids = self.getStackIdForConnector(config_id)
    namespace = self.getNamespaceForConnector(config_id)

    def construct_url(stack_id: str) -> str:
        if namespace == "databases":
            return f"{stack_id}-rw.{namespace}.svc.cluster.local"
        return f"http://{stack_id}.{namespace}.svc.cluster.local:8080"
```

## 2. Connection Modes

The system supports two primary operational modes:

### Local Development Mode

In local development mode:

- Environment variables are loaded from `.env` file
- `LOCAL_DEV` flag is set to `True`
- Service URLs are retrieved from environment variables in format: `{CONFIG_ID}_URL`
- API keys are retrieved from environment variables in format: `{CONFIG_ID}_APIKEY`

### Stack Deployment Mode

In stack deployment mode:

- `LOCAL_DEV` is `False` and `LOCAL_CONFIGS` is `False`
- Service URLs are constructed based on stack IDs and namespaces
- Connections are discovered dynamically from the stack manager
- Background polling ensures updates to connection details are applied

## 3. API Parameter Retrieval

The standard way to get connection details is through `get_api_parameters()`:

```python
def get_api_parameters(config_id: str) -> tuple[str, Optional[str]]:
    """Get URL and headers for a service connection."""
    load_environment()
    if config.LOCAL_DEV:
        # Get from environment variables
        url_base = os.environ.get(f"{config_id}_URL", None)
        apikey = os.environ.get(f"{config_id}_APIKEY", None)
        headers = {"Authorization": f"Bearer {apikey}"} if apikey else {}
    else:
        # Get from ConfigManager in stack mode
        configManager = ConfigManager()
        url_base = configManager.getUrlForConnector(config_id)
        headers = {}  # Not used in stack mode

    return (url_base, headers)
```

This function provides a consistent interface for both local development and stack deployment.

## 4. ConnectionManager Patterns

### Database ConnectionManager

The PostgreSQL connection management follows a standard pattern:

```python
class ConnectionManager:
    """Manages PostgreSQL connections with ConfidentialMind integration."""
    _pool: Optional[asyncpg.pool.Pool] = None
    _connection_error: Optional[str] = None
    _is_connecting: bool = False
    _reconnect_task: Optional[asyncio.Task] = None
    _current_url: Optional[str] = None

    @classmethod
    async def initialize(cls) -> bool:
        # Initialize connection manager and register connectors if needed

    @classmethod
    async def create_pool(cls) -> Optional[asyncpg.Pool]:
        # Create connection pool with retry support

    @classmethod
    async def _poll_for_url_changes(cls):
        # Background polling for URL changes
```

Key features include:

- Connection pooling for efficiency
- Reconnection logic with exponential backoff
- Background polling for URL changes
- Graceful operation when database is not initially available

### LLM Connector

```python
class LLMConnector:
    """Connector for LLM services with ConfidentialMind integration."""
    def __init__(self, config_id: str = "LLM"):
        self.config_id = config_id
        self._last_base_url = None
        self._last_headers = None
        self._session = None

    async def initialize(self) -> bool:
        # Get connection details using ConnectorConfigManager
        current_base_url, headers = await connector_manager.fetch_llm_url(self.config_id)
        self._session = aiohttp.ClientSession(headers=headers or {})
```

## 5. TransportManager for MCP Servers

The `TransportManager` handles connections to MCP servers:

```python
class TransportManager:
    """Manages transport configurations for FastMCP clients."""
    def __init__(self, mode: Literal["cli", "api"] = "cli"):
        self.mode = mode
        self.transports = {}
        self.clients = {}

    def configure_transport(
        self,
        server_id: str,
        server_path: Optional[str] = None,
        server_url: Optional[str] = None,
        use_module: bool = True,
    ):
        # Configure appropriate transport based on mode
        if self.mode == "cli":
            # Use stdio transport for Python modules
            self.transports[server_id] = ModuleStdioTransport(module_path=module_path)
        elif self.mode == "api":
            # Use streamable HTTP transport for web endpoints
            self.transports[server_id] = StreamableHttpTransport(url=server_url)

    async def configure_from_stack(self):
        # Configure transports using MCP servers from the stack
        connector_manager = ConnectorConfigManager()
        servers = await connector_manager.fetch_mcp_servers()
        for server_id, server_url in servers.items():
            self.configure_transport(server_id, server_url=server_url)
```

## 6. Integration Examples

### Registering Connectors

```python
# Initialize ConnectorConfigManager
connector_manager = ConnectorConfigManager()

# Register connectors with the ConfigManager
connectors = [
    ConnectorSchema(
        type="database",
        label="Session Management Database",
        config_id="DATABASE",
    ),
    ConnectorSchema(
        type="llm",
        label="Language Model",
        config_id="LLM",
    ),
]

# Register array connectors for multiple MCP servers
array_connectors = [
    ArrayConnectorSchema(
        type="agent_tool",
        label="MCP Tool Servers",
        config_id="agentTools",
    )
]

# Initialize with the ConfigManager
config_manager = ConfigManager()
config_manager.init_manager(
    config_model=AgentConfig(),
    connectors=connectors,
    array_connectors=array_connectors,
)
```

### Fetching Connection Details

```python
# Get database URL
async def fetch_db_url(config_id: str = "DATABASE") -> Optional[str]:
    connector_manager = ConnectorConfigManager()
    await connector_manager.initialize(register_connectors=False)
    return await connector_manager.fetch_database_url(config_id)

# Get LLM connection details
async def initialize_llm(config_id: str = "LLM"):
    connector_manager = ConnectorConfigManager()
    await connector_manager.initialize(register_connectors=False)
    current_base_url, headers = await connector_manager.fetch_llm_url(config_id)
    return current_base_url, headers

# Get MCP server URLs
async def fetch_mcp_servers(config_id: str = "agentTools"):
    connector_manager = ConnectorConfigManager()
    await connector_manager.initialize(register_connectors=False)
    return await connector_manager.fetch_mcp_servers(config_id)
```

## 7. Best Practices

1. **Always use ConnectorConfigManager** for consistent connection handling
2. **Support both modes** (local development and stack deployment)
3. **Implement background polling** for URL changes in stack mode
4. **Handle graceful degradation** when services are unavailable
5. **Use connection pooling** for database connections
6. **Implement reconnection logic** with exponential backoff
7. **Log connection status** for easier debugging

## 8. Common Connection Patterns

### PostgreSQL Connection

```python
# Initialize database with settings
db_settings = DatabaseSettings()
database = Database(db_settings)

# Connect with URL from connector
db_url = await fetch_db_url()
success = await database.connect(db_url)

# Ensure schema if connected
if success:
    await database.ensure_schema()
```

### LLM Connection

```python
# Initialize LLM connector with config ID
llm_connector = LLMConnector("LLM")
success = await llm_connector.initialize()

# Generate text if connected
if success:
    response = await llm_connector.generate(prompt)
```

### MCP Server Connection

```python
# Create transport manager for API mode
transport_manager = TransportManager(mode="api")

# Configure from stack in production
if is_stack_deployment:
    await transport_manager.configure_from_stack()
else:
    # Configure from environment in development
    for server_id, server_url in mcp_servers.items():
        transport_manager.configure_transport(server_id, server_url=server_url)

# Create clients for all transports
clients = transport_manager.create_all_clients()
```

## Conclusion

The ConfidentialMind connection management system provides a robust framework for handling service connections in both local development and stack deployment modes. By following the patterns outlined in this document, developers can create applications that seamlessly integrate with the ConfidentialMind ecosystem and provide resilient operation even in dynamic environments.
