# api/main.py
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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.connectors.cm_llm_connector import CMLLMConnector
from src.connectors.cm_mcp_connector import CMMCPManager
from src.core.agent import Agent, Message
from src.core.agent_db_connection import (
    AgentDatabase,
    AgentPostgresSettings,
    fetch_agent_db_url,
)
from src.core.agent_db_migration import AgentMigration

# --- Configuration ---
LLM_CONFIG_ID = "LLM"
MCP_CONFIG_ID = "agentTools"
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
    global mcp_manager, llm_connector  # agent_db removed from globals
    logger.info("Starting up MCP Agent API...")

    # Initialize Config Manager and Connectors
    config_manager = ConfigManager()  # Use singleton instance

    # Regular connectors
    regular_connectors: list[ConnectorSchema] = [
        ConnectorSchema(type="llm", label="Agent LLM", config_id=LLM_CONFIG_ID),
        ConnectorSchema(
            type="database",
            label="Agent Session Database",
            config_id=AGENT_SESSION_DB_CONFIG_ID,
        ),
    ]

    # Array connector for MCP services
    array_connectors: list[ArrayConnectorSchema] = [
        ArrayConnectorSchema(
            type="agent_tool",
            label="MCP Servers",
            config_id=MCP_CONFIG_ID,
        ),
    ]

    try:
        # Initialize the ConfigManager
        config_manager.init_manager(
            config_model=DummyConfig(),
            connectors=regular_connectors,
            array_connectors=array_connectors,
        )
        logger.info("ConfigManager initialized with regular and array connectors.")

        # Initialize SDK-based MCP manager
        mcp_manager = CMMCPManager()
        logger.info("CMMCPManager initialized.")

        # Initialize SDK-based LLM connector
        llm_connector = CMLLMConnector(config_id=LLM_CONFIG_ID)
        logger.info(f"CMLLMConnector initialized for config_id: {LLM_CONFIG_ID}")

    except Exception as e:
        logger.error(f"FATAL: Error during connector initialization: {e}", exc_info=True)
        mcp_manager = None
        llm_connector = None
        app.state.agent_db = None  # Ensure state is None on error
        # Potentially raise to stop startup

    # --- Database Initialization ---
    agent_db_instance = None  # Local instance for initialization
    try:
        logger.info("Initializing database connection and verifying schema")
        db_settings = AgentPostgresSettings()
        agent_db_instance = AgentDatabase(settings=db_settings)  # Create instance

        db_url = await fetch_agent_db_url(AGENT_SESSION_DB_CONFIG_ID)
        if db_url:
            if await agent_db_instance.connect(db_url):  # Connect this instance
                logger.info("Successfully connected to agent database")
                migrator = AgentMigration(db=agent_db_instance)  # Use connected instance
                schema_valid, missing_elements = await migrator.check_schema_status()
                if schema_valid:
                    logger.info("Database schema structure verification passed")
                else:
                    logger.warning(
                        f"Schema verification failed. Missing elements: {missing_elements}"
                    )
                    logger.info("Running schema migration to fix issues")
                    migration_result = await migrator.ensure_schema()
                    if migration_result:
                        logger.info("Database schema migration completed successfully")
                    else:
                        logger.warning("Database schema migration encountered issues")
                        # Decide if this is fatal? For now, we store the potentially problematic instance
            else:
                logger.error("Failed to connect to agent database during startup")
                agent_db_instance = None  # Ensure instance is None if connect fails
        else:
            logger.warning("No database URL available from SDK during startup")
            agent_db_instance = None  # Ensure instance is None if URL fetch fails

        # Store the (potentially None) instance in app.state
        app.state.agent_db = agent_db_instance
        if agent_db_instance and agent_db_instance.is_connected():
            logger.info("AgentDatabase instance stored in app.state and is connected.")
        elif agent_db_instance:
            logger.warning("AgentDatabase instance stored in app.state but is NOT connected.")
        else:
            logger.warning("No AgentDatabase instance stored in app.state.")

    except Exception as e:
        logger.error(f"Error initializing database during startup: {e}", exc_info=True)
        app.state.agent_db = None  # Ensure state is None on error
        logger.warning("Continuing startup despite database initialization error")

    yield  # Application runs here

    # Shutdown section
    logger.info("Shutting down MCP Agent API...")
    # Close the shared database connection if it exists
    if hasattr(app.state, "agent_db") and app.state.agent_db:
        logger.info("Closing shared database connection pool.")
        await app.state.agent_db.disconnect()
    else:
        logger.info("No shared database connection pool to close.")


# --- FastAPI Application Setup ---
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


def get_mcp_manager() -> CMMCPManager:
    """Dependency function to get the initialized MCP Manager."""
    if mcp_manager is None:
        logger.error("MCP Manager is not available. Startup might have failed.")
        raise HTTPException(status_code=503, detail="MCP Manager service unavailable.")
    return mcp_manager


def get_llm_connector() -> CMLLMConnector:
    """Dependency function to get the initialized LLM Connector."""
    if llm_connector is None:
        logger.error("LLM Connector is not available. Startup might have failed.")
        raise HTTPException(status_code=503, detail="LLM Connector service unavailable.")
    return llm_connector


# --- NEW: Dependency function for AgentDatabase ---
async def get_agent_db(request: Request) -> AgentDatabase:
    """Dependency function to get the shared AgentDatabase instance."""
    if not hasattr(request.app.state, "agent_db") or request.app.state.agent_db is None:
        logger.error("AgentDatabase is not available in app state. Startup might have failed.")
        raise HTTPException(status_code=503, detail="Agent Session Database service unavailable.")

    # Optional: Add a check here to ensure the connection is still alive or attempt reconnect
    # For simplicity, we assume the ensure_connected() call within agent methods handles this.
    # Example check (might add overhead):
    # db_instance = request.app.state.agent_db
    # if not db_instance.is_connected():
    #     logger.warning("DB connection lost, attempting reconnect via dependency.")
    #     await db_instance.ensure_connected() # This might block
    #     if not db_instance.is_connected():
    #          raise HTTPException(status_code=503, detail="Agent Session Database connection lost.")

    return request.app.state.agent_db


# --- API Endpoint Models ---
# (Keep existing models: ChatMessage, ChatCompletionRequest, ChatCompletionResponse)
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "default"
    user: Optional[str] = None
    session_id: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., example="chatcmpl-123")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., example=1677858242)
    model: str = Field(..., example="cm-agent-1")
    choices: List[Dict] = Field(...)
    usage: Dict = Field(...)


# --- API Endpoints ---


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request_body: ChatCompletionRequest,  # Renamed to avoid conflict with FastAPI Request
    # Use dependency injection for all components
    current_mcp_manager: CMMCPManager = Depends(get_mcp_manager),
    current_llm_connector: CMLLMConnector = Depends(get_llm_connector),
    shared_agent_db: AgentDatabase = Depends(get_agent_db),  # Inject the DB
):
    try:
        # Get or generate session ID
        session_id = request_body.session_id or str(uuid.uuid4())

        # Get latest user message
        if request_body.messages and request_body.messages[-1].role == "user":
            latest_user_msg_content = request_body.messages[-1].content
        else:
            raise HTTPException(
                status_code=400, detail="Request must contain at least one user message."
            )

        # --- Create agent instance using injected components ---
        agent = Agent(
            llm_connector=current_llm_connector,
            mcp_manager=current_mcp_manager,
            agent_db=shared_agent_db,  # Pass the injected DB instance
            # db_config_id="DATABASE", # No longer needed here
            debug=(os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")),
        )

        # Process the query using the agent instance (which now uses the shared DB)
        result_state = await agent.run(latest_user_msg_content, session_id=session_id)

        # Format the response
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_body.model or "cm-agent-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        # Handle potential errors during agent run
                        "content": result_state.response
                        if not result_state.error
                        else f"Error: {result_state.error}\n{result_state.response or ''}".strip(),
                    },
                    "finish_reason": "stop" if not result_state.error else "error",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        return response

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}", exc_info=True)
        # Include session ID in error logging if available
        sid = request_body.session_id if request_body else "unknown"
        logger.error(f"Error occurred for session_id: {sid}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
async def health_check(request: Request):  # Add request to access app state
    """Performs a health check on the service and its components."""
    health_status = {
        "status": "healthy",
        "version": "1.0.2",
        "components": {
            "llm_connector": "ok" if llm_connector else "unavailable",
            "mcp_manager": "ok" if mcp_manager else "unavailable",
            "agent_database": "unknown",
        },
    }
    all_ok = True

    if not llm_connector or not mcp_manager:
        health_status["status"] = "unhealthy"
        all_ok = False

    # Check database status from app state
    agent_db = getattr(request.app.state, "agent_db", None)
    if agent_db:
        if agent_db.is_connected():
            health_status["components"]["agent_database"] = "connected"
        else:
            health_status["components"]["agent_database"] = "disconnected"
            health_status["db_last_error"] = agent_db.last_error()
            # Potentially mark as unhealthy if DB connection is critical
            # health_status["status"] = "unhealthy"
            # all_ok = False
    else:
        health_status["components"]["agent_database"] = "unavailable"
        health_status["status"] = "unhealthy"  # DB unavailable is likely critical
        all_ok = False

    if not all_ok:
        logger.warning(f"Health check failed: {health_status}")

    return health_status


# Update other endpoints to use dependency injection for DB


@app.get("/mcps")
async def list_mcps(current_mcp_manager: CMMCPManager = Depends(get_mcp_manager)):
    # This endpoint doesn't need the DB, so no change here needed for DB injection
    # Code remains the same as before...
    try:
        clients = current_mcp_manager.get_all_clients()
        available_servers = {}
        # ... rest of the existing code for this endpoint ...
        config_manager = ConfigManager()  # Get singleton
        registered_connectors = []
        array_connectors = []

        if config_manager:
            if config_manager.connectors:
                registered_connectors = [
                    {"config_id": c.config_id, "label": c.label, "type": c.type}
                    for c in config_manager.connectors
                    if c.type != "llm" and c.type != "database"  # Filter out LLM and DB
                ]
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
        # ... rest of the existing code for this endpoint ...
        return {
            "registered_connectors": registered_connectors,
            "registered_array_connectors": array_connectors,
            "available_mcp_servers": available_servers,
        }

    except Exception as e:
        logger.error(f"Error listing MCP servers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing MCP servers: {str(e)}")


@app.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    # Inject dependencies
    shared_agent_db: AgentDatabase = Depends(get_agent_db),
):
    try:
        # Directly use the injected DB instance
        if not shared_agent_db.is_connected():
            raise HTTPException(status_code=503, detail="Database not connected")

        # Use a simplified history loading logic directly here or call a static method
        results = await shared_agent_db.execute_query(
            """
             SELECT role, content
             FROM conversation_messages
             WHERE session_id = $1
             ORDER BY message_order
             """,
            session_id,
        )
        history = [Message(role=row["role"], content=row["content"]) for row in results]

        formatted_history = [{"role": msg.role, "content": msg.content} for msg in history]

        return {
            "session_id": session_id,
            "messages": formatted_history,
            "message_count": len(formatted_history),
        }
    except Exception as e:
        logger.error(f"Error retrieving session history for {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {str(e)}")


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    # Inject DB dependency
    shared_agent_db: AgentDatabase = Depends(get_agent_db),
):
    try:
        # Directly use the injected DB instance
        if not shared_agent_db.is_connected():
            raise HTTPException(status_code=503, detail="Database not connected")

        # Use DB instance to delete
        await shared_agent_db.execute_query(
            "DELETE FROM conversation_messages WHERE session_id = $1",
            session_id,
            fetch_type="none",
        )
        success = True  # Assume success if execute_query doesn't raise

        if success:
            return {"status": "success", "message": f"Session {session_id} deleted successfully"}
        else:
            # This path might not be reachable if execute_query raises on failure
            raise HTTPException(status_code=500, detail=f"Failed to delete session {session_id}")
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting server locally with uvicorn...")
    # Run with uvicorn command for proper lifespan execution
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Please run using: uvicorn api.main:app --host 0.0.0.0 --port 8000 [--reload]")
