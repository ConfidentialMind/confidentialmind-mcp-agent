import logging  # Added logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

from confidentialmind_core.config_manager import ConfigManager, ConnectorSchema
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.database import get_db, get_messages_for_session, init_db, save_message
from src.connectors.cm_llm_connector import CMLLMConnector
from src.connectors.cm_mcp_connector import CMMCPManager
from src.core.agent import Agent, Message

# --- Configuration ---
# Define config_ids used by the application
LLM_CONFIG_ID = "LLM"
PG_MCP_CONFIG_ID = "postgres"
RAG_MCP_CONFIG_ID = "rag"
OBSIDIAN_MCP_CONFIG_ID = "obsidian"

# Global handlers - initialized during lifespan
mcp_manager: CMMCPManager = None
llm_connector: CMLLMConnector = None

logger = logging.getLogger(__name__)  # Added logger


# Delete later
class DummyConfig(BaseModel):
    pass


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup section
    global mcp_manager, llm_connector
    logger.info("Starting up MCP Agent API...")

    # Initialize Database
    await init_db()
    logger.info("Database initialized.")

    # Initialize Config Manager and Connectors
    config_manager = ConfigManager()
    connectors: list[ConnectorSchema] = [
        ConnectorSchema(type="llm", label="Agent LLM", config_id=LLM_CONFIG_ID),
        ConnectorSchema(
            type="api",
            label="PostgreSQL MCP Server",
            config_id=PG_MCP_CONFIG_ID,
        ),
        ConnectorSchema(
            type="api",
            label="RAG MCP Server",
            config_id=RAG_MCP_CONFIG_ID,
        ),
        ConnectorSchema(
            type="api",
            label="Obsidian MCP Server",
            config_id=OBSIDIAN_MCP_CONFIG_ID,
        ),
        # Add other connectors here if needed
    ]

    try:
        # Initialize the ConfigManager with the defined connectors
        # The SDK will handle loading actual connection details (from env vars or portal)
        config_manager.init_manager(
            config_model=DummyConfig(),
            connectors=connectors,
        )
        logger.info("ConfigManager initialized with connectors.")

        # Initialize SDK-based MCP manager (now relies on initialized ConfigManager)
        mcp_manager = CMMCPManager()
        logger.info("CMMCPManager initialized.")

        # Initialize SDK-based LLM connector
        llm_connector = CMLLMConnector(config_id=LLM_CONFIG_ID)
        logger.info(f"CMLLMConnector initialized for config_id: {LLM_CONFIG_ID}")

    except Exception as e:
        logger.error(f"FATAL: Error during application startup: {e}", exc_info=True)
        # Depending on policy, you might want to exit or prevent the app from starting fully
        # For now, we log the error and continue, but endpoints might fail
        mcp_manager = None  # Ensure manager is None if init fails
        llm_connector = None

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
    db=Depends(get_db),
    # Use the new dependency functions
    current_mcp_manager: CMMCPManager = Depends(get_mcp_manager_dependency),
    current_llm_connector: CMLLMConnector = Depends(get_llm_connector_dependency),
):
    try:
        # Get or generate session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Get conversation history from database
        conversation_history = await get_messages_for_session(db, session_id)

        # Get latest user message for the agent to process
        latest_user_msg_obj = None
        if request.messages and request.messages[-1].role == "user":
            latest_user_msg_content = request.messages[-1].content
            latest_user_msg_obj = Message(role="user", content=latest_user_msg_content)

            # Check if this message is genuinely new compared to the stored history
            is_new_user_message = True
            if conversation_history:
                last_stored_message = conversation_history[-1]
                if (
                    last_stored_message.role == "user"
                    and last_stored_message.content == latest_user_msg_content
                ):
                    is_new_user_message = False

            # If new, save it and add to the history list we pass to the agent
            if is_new_user_message:
                await save_message(db, session_id, latest_user_msg_obj)
                conversation_history.append(latest_user_msg_obj)
        else:
            # Handle cases where the last message isn't from the user or request is empty
            raise HTTPException(
                status_code=400, detail="Request must contain at least one user message."
            )

        # Create agent instance with SDK-based connectors passed from dependencies
        agent = Agent(
            llm_connector=current_llm_connector,
            mcp_manager=current_mcp_manager,  # Pass the initialized manager
            debug=(os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")),
        )

        # Set the conversation history in the agent
        agent.conversation_history = conversation_history

        # Process the query
        result = agent.run(latest_user_msg_obj.content)

        # Save the assistant's response to database
        if result.response:
            assistant_message = Message(role="assistant", content=result.response)
            await save_message(db, session_id, assistant_message)

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
    # TODO: Add check for DB connection status if possible
    return {"status": "healthy", "version": "1.0.1"}  # Updated version


# Add route to check which MCP servers are available
@app.get("/mcps")
async def list_mcps(current_mcp_manager: CMMCPManager = Depends(get_mcp_manager_dependency)):
    try:
        # Get clients from the manager (it checks availability internally now)
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
        if config_manager := ConfigManager():  # Check if config_manager is initialized
            if config_manager.connectors:
                registered_connectors = [
                    {"config_id": c.config_id, "label": c.label, "type": c.type}
                    for c in config_manager.connectors
                    if c.type != "llm"  # Filter out LLM connector
                ]

        return {
            "registered_connectors": registered_connectors,
            "available_mcp_servers": available_servers,
        }
    except Exception as e:
        logger.error(f"Error listing MCP servers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing MCP servers: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Setup basic logging for local execution
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting server locally with uvicorn...")
    # Note: Lifespan won't run fully when executing directly like this
    # It's better to run with `uvicorn api.main:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=8000)
