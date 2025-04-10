# ConfidentialMind MCP Agent

A powerful AI-powered agent that interfaces with PostgreSQL databases, RAG systems, and Obsidian vaults, enabling natural language interactions with your data. This agent uses the Model Context Protocol (MCP) architecture to provide a secure and flexible connection between LLMs, databases, and knowledge sources.

## Overview

ConfidentialMind MCP Agent allows you to:

- Query PostgreSQL databases using natural language
- Explore database schema information
- Execute SQL queries based on natural language instructions
- Get results in a structured, easy-to-understand format
- Retrieve relevant information from RAG systems
- Access and search your Obsidian vault notes
- Combine database queries with knowledge retrieval

The agent uses a LangGraph-based workflow to process queries, plan actions, execute database operations, retrieve information from RAG systems, and generate human-friendly responses.

## Microservice Architecture

The system now uses a modern microservice architecture with containerized components:

1. **Agent API Service** (`api/main.py`): Central FastAPI service that provides:
   - OpenAI-compatible chat completions API
   - Conversation history tracking in PostgreSQL
   - Orchestration of MCP tool servers
   - Configuration management via ConfidentialMind SDK

2. **PostgreSQL MCP Server** (`src/mcp/postgres_mcp_server.py`): Containerized service that provides:
   - Schema exploration and read-only SQL query execution
   - MCP protocol access to PostgreSQL databases

3. **RAG MCP Server** (`src/mcp/rag_mcp_server.py`): Containerized service that provides:
   - Semantic search across document repositories
   - MCP protocol access to RAG systems

4. **Obsidian MCP Server** (`src/mcp/obsidian_mcp_server.py`): Containerized service that provides:
   - Note retrieval and search capabilities
   - MCP protocol access to Obsidian vaults

5. **Database Service**: PostgreSQL instance for conversation history storage

Each service runs in its own Docker container, communicates via HTTP using the MCP protocol, and includes health checks for service orchestration.

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/confidentialmind-mcp-agent.git
cd confidentialmind-mcp-agent

# Create .env file with required configuration
cp .env.example .env
# Edit .env with your configuration values

# Start all services
docker-compose up -d
```

### Local Development

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv --python 3.12.9

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Configuration

Create a `.env` file with the following variables:

```bash
# PostgreSQL connection for the agent's conversation history
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/agents

# PostgreSQL connection for the tool server to connect to your data
PG_CONNECTION_STRING="postgresql://username:password@hostname:5432/database"

# LLM API configuration
LLM_URL="https://api.your-model-service.com/v1/api/your-model-id"
LLM_API_KEY="your_api_key"

# RAG API configuration
RAG_API_URL="https://api.your-rag-service.com/v1/api/your-project-id"
RAG_API_KEY="your-api-key-here"

# Obsidian configuration
OBSIDIAN_VAULT_PATH="/absolute/path/to/your/obsidian/vault"

# MCP server URLs (when running locally)
PG_MCP_URL=http://localhost:8001
RAG_MCP_URL=http://localhost:8002
OBSIDIAN_MCP_URL=http://localhost:8003

# ConfidentialMind configuration
CONFIDENTIAL_MIND_LOCAL_CONFIG=true

# Optional: Debug logging
DEBUG=true
```

## Usage

### Using the API

The agent now exposes an OpenAI-compatible chat completions API:

```bash
# Example API request using curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What tables are in my database?"}
    ],
    "model": "cm-agent-1"
  }'
```

### Running Individual MCP Servers (for development)

```bash
# Run PostgreSQL MCP server
python -m src.mcp.postgres_mcp_server

# Run RAG MCP server
python -m src.mcp.rag_mcp_server

# Run Obsidian MCP server
python -m src.mcp.obsidian_mcp_server
```

### CLI Mode (still supported)

```bash
# Run interactive mode 
python mcp_integration.py

# Run a single query
python mcp_integration.py --query "What tables do I have in my database?"
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

# RAG queries
What information do we have about customer satisfaction?
Find documents related to our return policy.

# Obsidian queries
What notes do I have about project planning?
Find information about meeting notes in my vault
Summarize my research notes on machine learning
```

## Docker Services

The `docker-compose.yaml` defines the following services:

- **db**: PostgreSQL database for conversation history
- **postgres-mcp**: MCP server for PostgreSQL database access (port 8001)
- **rag-mcp**: MCP server for RAG document retrieval (port 8002)
- **obsidian-mcp**: MCP server for Obsidian vault access (port 8003)
- **agent-api**: API service orchestrating all MCP servers (port 8000)

Each service includes health checks and appropriate dependencies.

## Security Considerations

- All MCP servers are containerized for isolation
- The PostgreSQL MCP server restricts queries to read-only operations (SELECT, EXPLAIN, SHOW)
- All SQL executions run in a READ ONLY transaction mode
- Connection credentials are never exposed in resource URIs
- API service manages all authentication and authorization
- CORS middleware configured for secure browser access
- Health check endpoints for service monitoring

## MCP Protocol

The agent uses MCP architecture:

1. **MCP Protocol**: JSON-RPC 2.0 based protocol that defines four key methods:
   - `mcp_listResources`: Lists available resources (database tables, RAG collections, Obsidian notes)
   - `mcp_readResource`: Reads a specific resource (table schema, RAG collection details, note content)
   - `mcp_listTools`: Lists available tools (SQL query execution, RAG document retrieval, Obsidian search)
   - `mcp_callTool`: Executes a tool (runs a SQL query, retrieves relevant documents, searches notes)

2. Each MCP server implements this protocol independently, enabling a consistent interface across different tool types.

## Agent Workflow

The agent uses a LangGraph-based workflow:

1. **Initialize Context**: Discovers available tools and resources from all connected MCP services
2. **Parse Query**: Analyzes the natural language query to determine required actions and creates a multi-step plan
3. **Execute MCP Actions**: Executes operations through appropriate MCP clients (reading schemas, running queries, retrieving documents)
4. **Evaluate Results**: Assesses each action's outcome to determine if more actions are needed or if replanning is required
5. **Generate Response**: Synthesizes all gathered information into a comprehensive human-friendly response

This workflow supports multi-hop reasoning and can adapt to execution failures with intelligent replanning.

## Requirements

- Docker and Docker Compose (for containerized deployment)
- Python 3.10 or higher (for local development)
- PostgreSQL database
- Access to an LLM API service (local or remote)
- Access to a RAG API service (optional)
- Obsidian vault (optional, for Obsidian integration)

## Creating Custom MCP Servers

Custom MCP servers can be developed following the examples in `src/mcp/`. Each MCP server must:

1. Implement the four core MCP methods
2. Provide FastAPI endpoints for health checks
3. Use proper environment variable configuration
4. Define appropriate tool schemas and resource types

Add your custom MCP server to the `docker-compose.yaml` file to integrate it with the system.
