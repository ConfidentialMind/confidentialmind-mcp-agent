# ConfidentialMind MCP Agent

A powerful AI-powered agent that interfaces with PostgreSQL databases and RAG systems, enabling natural language interactions with your data. This agent uses the Model context protocol (MCP) architecture to provide a secure and flexible connection between LLMs, PostgreSQL databases, and RAG services.

## Overview

ConfidentialMind MCP Agent allows you to:

- Query PostgreSQL databases using natural language
- Explore database schema information
- Execute SQL queries based on natural language instructions
- Get results in a structured, easy-to-understand format
- Retrieve relevant information from RAG systems
- Combine database queries with RAG knowledge retrieval

The agent uses a LangGraph-based workflow to process queries, plan actions, execute database operations, retrieve information from RAG systems, and generate human-friendly responses.

## Architecture

The system consists of the following components:

1. **Main Application** (`mcp_integration.py`): Entry point that handles CLI arguments, sets up components, and provides an interactive query session.

2. **MCP Protocol** (`src/mcp/mcp_protocol.py`): Defines the JSON-RPC based protocol for communication.

3. **MCP Client** (`src/mcp/mcp_client.py`): Client that communicates with the MCP servers.

4. **PostgreSQL MCP Server** (`src/mcp/postgres_mcp_server.py`): Server that connects to a PostgreSQL database and provides MCP protocol access.

5. **RAG MCP Server** (`src/mcp/rag_mcp_server.py`): Server that connects to a RAG service and provides MCP protocol access for knowledge retrieval.

6. **Agent Core** (`src/core/agent.py`): LangGraph-based workflow that processes queries, plans actions, and generates responses.

7. **LLM Connector** (`src/connectors/llm.py`): Handles communication with LLM APIs.

8. **ConfidentialMind Integration** (`src/connectors/confidentialmind.py`): Optional integration with ConfidentialMind for enhanced security and management.

## Installation

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
## MacOS & Linux
curl -LsSf https://astral.sh/uv/install.sh | less

# Create virtual environment with uv
uv venv --python 3.12.9

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Configuration

The agent requires the following environment variables:

```bash
# Required: PostgreSQL connection string
PG_CONNECTION_STRING="postgresql://username:password@hostname:5432/database"

# LLM API configuration
LLM_URL="https://api.your-model-service.com/v1/api/your-model-id"
LLM_API_KEY="your_api_key"  # Optional based on your LLM service

# RAG API configuration
RAG_API_URL="https://api.your-rag-service.com/v1/api/your-project-id"
RAG_API_KEY="your-api-key-here"

CONFIDENTIAL_MIND_LOCAL_CONFIG=true

# Optional: Set to enable debug logging
DEBUG=true
```

You can create a `.env` file to store these variables.
Optionally, you can copy the example .env file.

```bash
cp .env.example .env
```

## Usage

### Basic Usage

```bash
# Run interactive mode using environment variables for configuration
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
```

## Notes

- You may find the agent to sometimes fail depending on the model. It may even need a bit of pushing and prompt engineering to complete the query.
- Harder queries will naturally fare better with stronger models.
- Llama 70B seems to work pretty well.
- Some hackathon ideas include building your own MCP tool server or modifying the agent workflow.

## Security Considerations

- The PostgreSQL MCP server restricts queries to read-only operations (SELECT, EXPLAIN, SHOW)
- All SQL executions run in a READ ONLY transaction mode
- Connection credentials are never exposed in resource URIs
- Always use a database user with appropriate permissions based on your security requirements

## Module Details

### MCP Protocol

The agent uses MCP architecture:

1. **MCP Protocol**: JSON-RPC 2.0 based protocol that defines four key methods:

   - `mcp_listResources`: Lists available resources (database tables, RAG collections)
   - `mcp_readResource`: Reads a specific resource (table schema, RAG collection details)
   - `mcp_listTools`: Lists available tools (SQL query execution, RAG document retrieval)
   - `mcp_callTool`: Executes a tool (runs a SQL query, retrieves relevant documents)

2. **MCP Client**: Manages communication with the MCP servers, handling request/response cycles.

3. **PostgreSQL MCP Server**: Connects to PostgreSQL and implements the MCP protocol, providing access to database functionality.

4. **RAG MCP Server**: Connects to a RAG service and implements the MCP protocol, providing access to document retrieval functionality.

### Agent Workflow

The agent uses a LangGraph-based workflow:

1. **Initialize Context**: Discovers available tools and resources from all connected MCP services
2. **Parse Query**: Analyzes the natural language query to determine required actions and creates a multi-step plan
3. **Execute MCP Actions**: Executes operations through appropriate MCP clients (reading schemas, running queries, retrieving documents)
4. **Evaluate Results**: Assesses each action's outcome to determine if more actions are needed or if replanning is required
5. **Replan Actions** (when needed): Revises the plan based on execution results, handling errors like missing schema qualifications
6. **Plan Follow-up** (when needed): Plans additional actions if more information is needed to fully answer the query
7. **Generate Response**: Synthesizes all gathered information into a comprehensive human-friendly response

This workflow supports multi-hop reasoning and can adapt to execution failures with intelligent replanning.

## Requirements

- Python 3.10 or higher
- PostgreSQL database
- Access to an LLM API service (local or remote)
- Access to a RAG API service
- ConfidentialMind account
