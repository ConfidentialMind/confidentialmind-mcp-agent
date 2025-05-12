"""OpenAI API-compatible endpoint for the FastMCP agent."""

import logging
import os
import uuid
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Dict, List, Literal, Optional

import uvicorn
from confidentialmind_core.config_manager import load_environment
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.agent import Agent
from src.agent.connectors import ConnectorConfigManager
from src.agent.database import Database, DatabaseSettings
from src.agent.llm import LLMConnector
from src.agent.transport import TransportManager

# Configure logging
logger = logging.getLogger("fastmcp_agent.api")

load_environment()


# Define Pydantic models for OpenAI API compatibility
class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user: Optional[str] = None


class ChatCompletionChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: Literal["stop", "length", "function_call"] = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Tokens in the prompt")
    completion_tokens: int = Field(..., description="Tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique ID for this completion")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class AgentComponents:
    """Shared agent components for API endpoints"""

    def __init__(self):
        self.database: Optional[Database] = None
        self.llm_connector: Optional[LLMConnector] = None
        self.transport_manager: Optional[TransportManager] = None
        self.mcp_servers: Dict[str, str] = {}
        self.exit_stack = AsyncExitStack()
        self.initialized = False
        self.db_config_id = os.environ.get("DB_CONFIG_ID", "DATABASE")
        self.llm_config_id = os.environ.get("LLM_CONFIG_ID", "LLM")

        # Get MCP server URLs from environment with a default
        default_mcp_server = os.environ.get("AGENT_TOOLS_URL", "http://localhost:8080/sse")
        self.mcp_servers = {"agentTools": default_mcp_server}

        # Check if there are any additional MCP servers specified in the format MCP_SERVER_NAME=url
        for key, value in os.environ.items():
            if key.startswith("MCP_SERVER_") and value:
                server_name = key[11:].lower()  # Remove "MCP_SERVER_" prefix and lowercase
                if server_name and value:
                    self.mcp_servers[server_name] = value

    async def initialize(self):
        """Initialize agent components"""
        if self.initialized:
            return True

        try:
            # Initialize connector configuration
            connector_manager = ConnectorConfigManager()
            await connector_manager.initialize()

            # Initialize database
            db_url = await connector_manager.fetch_database_url(self.db_config_id)
            db_settings = DatabaseSettings()
            self.database = Database(db_settings)
            success = await self.database.connect(db_url)
            if not success:
                logger.error("Failed to connect to database")
                return False

            # Initialize schema
            success = await self.database.ensure_schema()
            if not success:
                logger.warning("Could not ensure database schema. Some operations might fail.")

            # Initialize LLM connector
            self.llm_connector = LLMConnector(self.llm_config_id)
            success = await self.llm_connector.initialize()
            if not success:
                logger.error("Failed to initialize LLM connector")
                return False

            # Initialize transport manager for API mode
            self.transport_manager = TransportManager(mode="api")

            if connector_manager.is_stack_deployment:
                # Get MCP servers from stack
                await self.transport_manager.configure_from_stack()

            else:
                # Configure transports
                for server_id, server_url in self.mcp_servers.items():
                    try:
                        self.transport_manager.configure_transport(server_id, server_url=server_url)
                    except Exception as e:
                        logger.error(f"Error configuring transport for {server_id}: {e}")
                        return False

            # Create all clients
            self.transport_manager.create_all_clients()

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing agent components: {e}", exc_info=True)
            # Clean up any potentially partially initialized resources
            await self.cleanup()
            return False

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.database:
                await self.database.disconnect()
                self.database = None

            if self.llm_connector:
                await self.llm_connector.close()
                self.llm_connector = None

            await self.exit_stack.aclose()
            self.initialized = False
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)


# Create FastAPI app and shared components
app = FastAPI(
    title="FastMCP Agent API",
    description="OpenAI API-compatible endpoint for FastMCP agent",
    version="0.1.0",
)
components = AgentComponents()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get agent components
async def get_components():
    if not components.initialized:
        success = await components.initialize()
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to initialize agent components",
            )
    return components


# Extract session ID from request
def get_session_id(request: Request) -> str:
    """Extract or generate session ID from request."""
    # Try to get session ID from header
    session_id = request.headers.get("X-Session-ID")

    # If no session ID in header, try to get from cookie
    if not session_id:
        session_id = request.cookies.get("session_id")

    # If still no session ID, generate a new one
    if not session_id:
        session_id = str(uuid.uuid4())

    return session_id


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: Request,
    req: ChatCompletionRequest,
    components: AgentComponents = Depends(get_components),
):
    """OpenAI-compatible chat completion endpoint."""
    if req.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming is not supported yet",
        )

    # Get or create session ID
    session_id = get_session_id(request)

    # Convert OpenAI messages to a single query
    query = ""
    for msg in req.messages:
        if msg.role == "user":
            query = msg.content
            break  # Just use the last user message as the query

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user message provided",
        )

    try:
        # Create agent
        agent = Agent(
            components.database,
            components.llm_connector,
            components.transport_manager,
            debug=False,
        )

        await agent.initialize()

        # Execute agent within context manager for proper resource cleanup
        async with agent:
            # Run agent
            state = await agent.run(query, session_id)

            if state.error:
                logger.warning(f"Agent error: {state.error}")
                # Return the error as part of the response, don't raise exception
                # This allows the client to see the error message

            # Create response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(datetime.now().timestamp()),
                model=req.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionChoiceMessage(
                            content=state.response or f"Error: {state.error}"
                        )
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=len(query) // 4,  # Rough estimate
                    completion_tokens=len(state.response or "") // 4,  # Rough estimate
                    total_tokens=(len(query) + len(state.response or "")) // 4,  # Rough estimate
                ),
            )

            return response
    except Exception as e:
        logger.error(f"Error processing chat completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )


@app.get("/health")
async def health_check(components: AgentComponents = Depends(get_components)):
    """Health check endpoint."""
    return {"status": "healthy", "database": components.database.is_connected()}


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting FastMCP Agent API server")
    await components.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down FastMCP Agent API server")
    await components.cleanup()


def start_api_server(host: str = "0.0.0.0", port: int = 8000, log_level: str = "info"):
    """Start the API server."""
    uvicorn.run(
        "agent.api:app",
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start API server
    start_api_server()
