import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

from confidentialmind_core.config_manager import (
    ArrayConnectorSchema,
    ConfigManager,
    ConnectorSchema,
)
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.connectors.cm_llm_connector import CMLLMConnector
from src.connectors.cm_mcp_connector import CMMCPManager
from src.core.agent import Agent
from src.core.agent_db_connection import (
    AgentDatabase,
    AgentPostgresSettings,
    fetch_agent_db_url,
)
from src.core.agent_db_migration import AgentMigration

# --- Configuration ---
# Define config_ids used by the application
LLM_CONFIG_ID = "LLM"
MCP_CONFIG_ID = "agentTools"  # Single config ID for all MCP servers
AGENT_SESSION_DB_CONFIG_ID = "DATABASE"

# Global handlers - initialized during lifespan
mcp_manager: CMMCPManager = None
llm_connector: CMLLMConnector = None

logger = logging.getLogger(__name__)


# Config model with minimal requirements
class DummyConfig(BaseModel):
    pass


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup section
    global mcp_manager, llm_connector
    logger.info("Starting up MCP Agent API...")

    # Initialize Config Manager and Connectors
    config_manager = ConfigManager()

    # Regular connectors - for LLM and database
    regular_connectors: list[ConnectorSchema] = [
        ConnectorSchema(type="llm", label="Agent LLM", config_id=LLM_CONFIG_ID),
        ConnectorSchema(
            type="database",
            label="Agent Session Database",
            config_id=AGENT_SESSION_DB_CONFIG_ID,
        ),
    ]

    # Array connector for MCP services - allows dynamic configuration of multiple servers
    array_connectors: list[ArrayConnectorSchema] = [
        ArrayConnectorSchema(
            type="agent_tool",
            label="MCP Servers",
            config_id=MCP_CONFIG_ID,
        ),
    ]

    try:
        # Initialize the ConfigManager with both regular and array connectors
        config_manager.init_manager(
            config_model=DummyConfig(),
            connectors=regular_connectors,
            array_connectors=array_connectors,
        )
        logger.info("ConfigManager initialized with regular and array connectors.")

        # Initialize SDK-based MCP manager (now supports array connectors)
        mcp_manager = CMMCPManager()
        logger.info("CMMCPManager initialized with support for array connectors.")

        # Initialize SDK-based LLM connector
        llm_connector = CMLLMConnector(config_id=LLM_CONFIG_ID)
        logger.info(f"CMLLMConnector initialized for config_id: {LLM_CONFIG_ID}")

    except Exception as e:
        logger.error(f"FATAL: Error during application startup: {e}", exc_info=True)
        # Depending on policy, you might want to exit or prevent the app from starting fully
        # For now, we log the error and continue, but endpoints might fail
        mcp_manager = None  # Ensure manager is None if init fails
        llm_connector = None

    try:
        logger.info("Initializing database connection and verifying schema")

        # Create database connection
        db_settings = AgentPostgresSettings()
        agent_db = AgentDatabase(settings=db_settings)

        # Fetch database URL
        db_url = await fetch_agent_db_url(AGENT_SESSION_DB_CONFIG_ID)
        if db_url:
            # Connect to database
            if await agent_db.connect(db_url):
                logger.info("Successfully connected to agent database")

                # Initialize and run schema verification/migration
                migrator = AgentMigration(db=agent_db)

                # Verify schema structure and run migrations if needed
                schema_valid, missing_elements = await migrator.check_schema_status()

                if schema_valid:
                    logger.info("Database schema structure verification passed")
                else:
                    logger.warning(
                        f"Schema verification failed. Missing elements: {missing_elements}"
                    )
                    logger.info("Running schema migration to fix issues")

                    # Apply migrations
                    migration_result = await migrator.ensure_schema()
                    if migration_result:
                        logger.info("Database schema migration completed successfully")
                    else:
                        logger.warning("Database schema migration encountered issues")
            else:
                logger.error("Failed to connect to agent database during startup")
        else:
            logger.warning("No database URL available from SDK during startup")
    except Exception as e:
        logger.error(f"Error initializing database during startup: {e}", exc_info=True)
        logger.warning("Continuing startup despite database initialization error")

    yield  # Application runs here

    # Shutdown section
    logger.info("Shutting down MCP Agent API...")
    # Add any cleanup code here if needed (e.g., closing connections)
    # ConfigManager doesn't require explicit shutdown in the SDK provided


# --- FastAPI Application Setup ---
# Use the lifespan context manager
app = FastAPI(title="ConfidentialMind MCP Agent API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency Injection Functions ---
# These functions provide initialized handlers/connectors to endpoints


def get_mcp_manager_dependency() -> CMMCPManager:
    """Dependency function to get the initialized MCP Manager."""
    if mcp_manager is None:
        logger.error("MCP Manager is not available. Startup might have failed.")
        raise HTTPException(
            status_code=503, detail="MCP Manager service unavailable due to initialization error."
        )
    return mcp_manager


def get_llm_connector_dependency() -> CMLLMConnector:
    """Dependency function to get the initialized LLM Connector."""
    if llm_connector is None:
        logger.error("LLM Connector is not available. Startup might have failed.")
        raise HTTPException(
            status_code=503, detail="LLM Connector service unavailable due to initialization error."
        )
    return llm_connector


# --- API Endpoint Models ---
# Models for OpenAI-compatible API
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "default"  # Model selection might be handled by LLM connector config
    user: Optional[str] = None
    session_id: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., example="chatcmpl-123")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., example=1677858242)
    model: str = Field(..., example="cm-agent-1")  # Or dynamically set based on used model
    choices: List[Dict] = Field(...)
    usage: Dict = Field(...)


# --- API Endpoints ---


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    # Use the new dependency functions
    current_mcp_manager: CMMCPManager = Depends(get_mcp_manager_dependency),
    current_llm_connector: CMLLMConnector = Depends(get_llm_connector_dependency),
):
    try:
        # Get or generate session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Get latest user message for the agent to process
        latest_user_msg_obj = None
        if request.messages and request.messages[-1].role == "user":
            latest_user_msg_content = request.messages[-1].content
        else:
            # Handle cases where the last message isn't from the user or request is empty
            raise HTTPException(
                status_code=400, detail="Request must contain at least one user message."
            )

        # Create agent instance with SDK-based connectors passed from dependencies
        agent = Agent(
            llm_connector=current_llm_connector,
            mcp_manager=current_mcp_manager,  # Pass the initialized manager
            db_config_id=AGENT_SESSION_DB_CONFIG_ID,
            debug=(os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")),
        )

        # Process the query - this now handles session management and DB persistence
        result = await agent.run(latest_user_msg_content, session_id=session_id)

        # Format the response in OpenAI-compatible format
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            # Use model from request or default/resolved model
            "model": request.model or "cm-agent-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.response or "I couldn't generate a response.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting if needed
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        return response

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
async def health_check():
    # Add checks for DB connection, config manager status etc. if needed
    if mcp_manager is None or llm_connector is None:
        return {"status": "unhealthy", "reason": "Components failed to initialize"}
    return {"status": "healthy", "version": "1.0.2"}  # Updated version


# Add route to check which MCP servers are available
@app.get("/mcps")
async def list_mcps(current_mcp_manager: CMMCPManager = Depends(get_mcp_manager_dependency)):
    try:
        # Get clients from the manager (it now supports array connectors)
        clients = current_mcp_manager.get_all_clients()
        available_servers = {}

        for server_id, client in clients.items():
            try:
                # Test the connection by listing tools
                tools = client.list_tools()
                available_servers[server_id] = {
                    "status": "available",
                    "tools_count": len(tools.get("tools", [])),
                    "config_id": server_id,  # Add config_id for clarity
                }
            except Exception as e:
                logger.warning(f"MCP server '{server_id}' check failed: {e}")
                available_servers[server_id] = {
                    "status": "error",
                    "error": str(e),
                    "config_id": server_id,
                }

        # Get connector info from ConfigManager
        registered_connectors = []
        array_connectors = []

        if config_manager := ConfigManager():  # Check if config_manager is initialized
            # Regular connectors
            if config_manager.connectors:
                registered_connectors = [
                    {"config_id": c.config_id, "label": c.label, "type": c.type}
                    for c in config_manager.connectors
                    if c.type != "llm"  # Filter out LLM connector
                ]

            # Array connectors
            if config_manager.array_connectors:
                array_connectors = [
                    {
                        "config_id": c.config_id,
                        "label": c.label,
                        "type": c.type,
                        "stack_ids": c.stack_ids or [],
                    }
                    for c in config_manager.array_connectors
                ]

        return {
            "registered_connectors": registered_connectors,
            "registered_array_connectors": array_connectors,
            "available_mcp_servers": available_servers,
        }
    except Exception as e:
        logger.error(f"Error listing MCP servers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing MCP servers: {str(e)}")


# Add route to get conversation history for a session
@app.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    current_mcp_manager: CMMCPManager = Depends(get_mcp_manager_dependency),
    current_llm_connector: CMLLMConnector = Depends(get_llm_connector_dependency),
):
    try:
        # Create agent instance
        agent = Agent(
            llm_connector=current_llm_connector,
            mcp_manager=current_mcp_manager,
            db_config_id=AGENT_SESSION_DB_CONFIG_ID,
        )

        # Get history from the agent's database
        history = await agent.get_history(session_id)

        # Format history for response
        formatted_history = [{"role": msg.role, "content": msg.content} for msg in history]

        return {
            "session_id": session_id,
            "messages": formatted_history,
            "message_count": len(formatted_history),
        }
    except Exception as e:
        logger.error(f"Error retrieving session history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {str(e)}")


# Add route to delete a conversation session
@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_mcp_manager: CMMCPManager = Depends(get_mcp_manager_dependency),
    current_llm_connector: CMLLMConnector = Depends(get_llm_connector_dependency),
):
    try:
        # Create agent instance
        agent = Agent(
            llm_connector=current_llm_connector,
            mcp_manager=current_mcp_manager,
            db_config_id=AGENT_SESSION_DB_CONFIG_ID,
        )

        # Clear history for this session
        success = await agent.clear_history(session_id)

        if success:
            return {"status": "success", "message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete session {session_id}")
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Setup basic logging for local execution
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting server locally with uvicorn...")
    # Note: Lifespan won't run fully when executing directly like this
    # It's better to run with `uvicorn api.main:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=8000)
