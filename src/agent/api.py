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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.agent import Agent
from src.agent.connectors import ConnectorConfigManager
from src.agent.conversation_manager import ConversationManager
from src.agent.database import Database, DatabaseSettings
from src.agent.llm import LLMConnector
from src.agent.state import Message
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
    model: str = "cm-llm"
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user: Optional[str] = None
    # Optional conversation management
    conversation_id: Optional[str] = Field(
        None, description="Optional conversation ID for explicit session management"
    )


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


# Streaming response models
class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: Dict[str, str] = Field(default_factory=dict)
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionStreamChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class AgentComponents:
    """Shared agent components for API endpoints"""

    def __init__(self):
        """Initialize shared components"""
        self.database: Optional[Database] = None
        self.llm_connector: Optional[LLMConnector] = None
        self.transport_manager: Optional[TransportManager] = None
        self.mcp_servers: Dict[str, str] = {}
        self.exit_stack = AsyncExitStack()
        self.initialized = False
        self.db_config_id = os.environ.get("DB_CONFIG_ID", "DATABASE")
        self.llm_config_id = os.environ.get("LLM_CONFIG_ID", "LLM")
        self.mcp_config_id = os.environ.get("MCP_CONFIG_ID", "agentTools")

        # Determine if running in stack deployment mode
        load_environment()
        self.is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"AgentComponents: Initializing in {'stack deployment' if self.is_stack_deployment else 'local config'} mode"
        )

        # For local development, get MCP server URLs from environment
        if not self.is_stack_deployment:
            # Get MCP server URLs from environment
            mcp_env_servers = {}
            for key, value in os.environ.items():
                if key.startswith("MCP_SERVER_") and value:
                    server_name = key[11:].lower()  # Remove "MCP_SERVER_" prefix and lowercase
                    if server_name and value:
                        # Convert legacy SSE URLs to streamable HTTP format
                        if value.endswith("/sse"):
                            value = value.rsplit("/sse", 1)[0] + "/mcp"
                            logger.info(
                                f"Converting legacy SSE URL to streamable HTTP for {server_name}: {value}"
                            )
                        mcp_env_servers[server_name] = value

            # Only use default if no environment-specified servers exist
            if mcp_env_servers:
                self.mcp_servers = mcp_env_servers
                logger.info(
                    f"AgentComponents: Found {len(self.mcp_servers)} MCP servers in environment"
                )
            else:
                # Get default MCP server URL
                default_mcp_server = os.environ.get("AGENT_TOOLS_URL", "http://localhost:8080/mcp")
                # Convert legacy SSE URLs to streamable HTTP format
                if default_mcp_server.endswith("/sse"):
                    default_mcp_server = default_mcp_server.rsplit("/sse", 1)[0] + "/mcp"
                    logger.info(
                        f"Converting legacy SSE URL to streamable HTTP: {default_mcp_server}"
                    )
                self.mcp_servers = {"agentTools": default_mcp_server}
                logger.info("AgentComponents: Using default MCP server configuration")

            logger.info(
                f"AgentComponents: Found {len(self.mcp_servers)} MCP servers in environment"
            )

    async def initialize(self):
        """
        Initialize agent components

        This method supports both stack deployment and local development modes.
        In stack deployment mode, it will not fail if connectors are not available,
        allowing the API to start and waiting for connectors to be configured.
        """
        if self.initialized:
            logger.debug("AgentComponents: Already initialized")
            return True

        try:
            # Initialize connector configuration
            connector_manager = ConnectorConfigManager()
            await connector_manager.initialize()

            # Initialize database
            db_url = await connector_manager.fetch_database_url(self.db_config_id)
            db_settings = DatabaseSettings()
            self.database = Database(db_settings)
            if db_url:
                success = await self.database.connect(db_url)
                if not success:
                    logger.warning(
                        "AgentComponents: Failed to connect to database, continuing without it"
                    )
            else:
                logger.warning(
                    "AgentComponents: No database URL available, continuing without database connection"
                )

            # Initialize schema if database is connected
            if self.database.is_connected():
                success = await self.database.ensure_schema()
                if not success:
                    logger.warning(
                        "AgentComponents: Could not ensure database schema. Some operations might fail."
                    )
            else:
                logger.info(
                    "AgentComponents: Database not connected, skipping schema initialization"
                )

            # Initialize LLM connector
            self.llm_connector = LLMConnector(self.llm_config_id)
            success = await self.llm_connector.initialize()
            if not success:
                logger.warning(
                    "AgentComponents: Failed to initialize LLM connector, continuing without it"
                )

            # Initialize transport manager for API mode
            self.transport_manager = TransportManager(mode="api")

            if self.is_stack_deployment:
                # Get MCP servers from stack (this also starts background polling)
                await self.transport_manager.configure_from_stack()
                logger.info("AgentComponents: Configured MCP servers from stack")
            else:
                # Configure transports from environment
                for server_id, server_url in self.mcp_servers.items():
                    try:
                        self.transport_manager.configure_transport(server_id, server_url=server_url)
                        logger.info(
                            f"AgentComponents: Configured transport for {server_id}: {server_url}"
                        )
                    except Exception as e:
                        logger.error(
                            f"AgentComponents: Error configuring transport for {server_id}: {e}"
                        )

            # Create all clients
            self.transport_manager.create_all_clients()

            # Log available clients
            clients = self.transport_manager.get_all_clients()
            client_info = ", ".join([f"{k}" for k in clients.keys()]) if clients else "none"
            logger.info(f"AgentComponents: Initialized with MCP clients: {client_info}")

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"AgentComponents: Error initializing components: {e}", exc_info=True)
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
            logger.error(f"AgentComponents: Error during cleanup: {e}", exc_info=True)


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
            logger.warning(
                "AgentComponents: Failed to initialize components, but will continue anyway"
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


async def _stream_chat_completion(
    query: str, session_id: str, req: ChatCompletionRequest, components: AgentComponents
):
    """Generate streaming chat completion response"""

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(datetime.now().timestamp())

    try:
        # Create agent
        agent = Agent(
            components.database,
            components.llm_connector,
            components.transport_manager,
            debug=False,
        )

        await agent.initialize()

        async with agent:
            # Send initial chunk to establish the stream
            initial_chunk = ChatCompletionStreamChunk(
                id=completion_id,
                created=created_timestamp,
                model=req.model,
                choices=[
                    ChatCompletionStreamChoice(delta={"role": "assistant"}, finish_reason=None)
                ],
            )
            yield f"data: {initial_chunk.model_dump_json()}\n\n"

            # Stream agent execution
            async for chunk_data in agent.run_streaming(query, session_id):
                if chunk_data["type"] == "workflow_status":
                    # Optionally send workflow status as metadata (commented out to reduce noise)
                    # This could be enabled via a request parameter if desired
                    pass

                elif chunk_data["type"] == "response_chunk":
                    # Send content chunk
                    content_chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created_timestamp,
                        model=req.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                delta={"content": chunk_data["content"]}, finish_reason=None
                            )
                        ],
                    )
                    yield f"data: {content_chunk.model_dump_json()}\n\n"

                elif chunk_data["type"] == "response_complete":
                    # Send final chunk
                    final_chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created_timestamp,
                        model=req.model,
                        choices=[ChatCompletionStreamChoice(delta={}, finish_reason="stop")],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    break

                elif chunk_data["type"] == "error":
                    # Send error as content and then finish
                    error_chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created_timestamp,
                        model=req.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                delta={"content": f"\n\nError: {chunk_data['error']}"},
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    break

    except Exception as e:
        logger.error(f"Error in streaming completion: {e}", exc_info=True)
        # Send error as final chunk
        error_chunk = ChatCompletionStreamChunk(
            id=completion_id,
            created=created_timestamp,
            model=req.model,
            choices=[
                ChatCompletionStreamChoice(
                    delta={"content": f"\n\nError: {str(e)}"}, finish_reason="stop"
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    finally:
        # Send termination signal
        yield "data: [DONE]\n\n"


async def _create_stateless_completion(
    query: str,
    conversation_id: str,
    req: ChatCompletionRequest,
    components: AgentComponents,
    conv_manager: ConversationManager,
    existing_hashes: List[str],
) -> ChatCompletionResponse:
    """Create non-streaming completion for stateless mode."""

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
            # Run agent using conversation_id as session_id
            state = await agent.run(query, conversation_id)

            if state.error:
                logger.warning(f"Agent error: {state.error}")

            # Store the new messages in the conversation
            new_messages = [
                Message(role="user", content=query),
                Message(role="assistant", content=state.response or f"Error: {state.error}"),
            ]

            # Generate hashes for new messages
            parent_hash = existing_hashes[-1] if existing_hashes else ""
            user_hash = conv_manager.generate_message_hash(
                {"role": "user", "content": query}, parent_hash
            )
            assistant_hash = conv_manager.generate_message_hash(
                {"role": "assistant", "content": state.response or f"Error: {state.error}"},
                user_hash,
            )
            new_hashes = [user_hash, assistant_hash]

            # Append to conversation
            await components.database.append_messages_to_conversation(
                conversation_id, new_messages, new_hashes
            )

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
        logger.error(f"Error processing stateless chat completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )


async def _stream_stateless_completion(
    query: str,
    conversation_id: str,
    req: ChatCompletionRequest,
    components: AgentComponents,
    conv_manager: ConversationManager,
    existing_hashes: List[str],
):
    """Generate streaming chat completion response for stateless mode."""

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(datetime.now().timestamp())
    collected_response = ""

    try:
        # Create agent
        agent = Agent(
            components.database,
            components.llm_connector,
            components.transport_manager,
            debug=False,
        )

        await agent.initialize()

        async with agent:
            # Send initial chunk to establish the stream
            initial_chunk = ChatCompletionStreamChunk(
                id=completion_id,
                created=created_timestamp,
                model=req.model,
                choices=[
                    ChatCompletionStreamChoice(delta={"role": "assistant"}, finish_reason=None)
                ],
            )
            yield f"data: {initial_chunk.model_dump_json()}\n\n"

            # Stream agent execution using conversation_id as session_id
            async for chunk_data in agent.run_streaming(query, conversation_id):
                if chunk_data["type"] == "workflow_status":
                    # Optionally send workflow status as metadata
                    pass

                elif chunk_data["type"] == "response_chunk":
                    # Collect response for saving
                    collected_response += chunk_data["content"]

                    # Send content chunk
                    content_chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created_timestamp,
                        model=req.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                delta={"content": chunk_data["content"]}, finish_reason=None
                            )
                        ],
                    )
                    yield f"data: {content_chunk.model_dump_json()}\n\n"

                elif chunk_data["type"] == "response_complete":
                    # Store the complete conversation
                    new_messages = [
                        Message(role="user", content=query),
                        Message(role="assistant", content=collected_response),
                    ]

                    # Generate hashes for new messages
                    parent_hash = existing_hashes[-1] if existing_hashes else ""
                    user_hash = conv_manager.generate_message_hash(
                        {"role": "user", "content": query}, parent_hash
                    )
                    assistant_hash = conv_manager.generate_message_hash(
                        {"role": "assistant", "content": collected_response}, user_hash
                    )
                    new_hashes = [user_hash, assistant_hash]

                    # Append to conversation
                    await components.database.append_messages_to_conversation(
                        conversation_id, new_messages, new_hashes
                    )

                    # Send final chunk
                    final_chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created_timestamp,
                        model=req.model,
                        choices=[ChatCompletionStreamChoice(delta={}, finish_reason="stop")],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    break

                elif chunk_data["type"] == "error":
                    # Send error as content and then finish
                    error_chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created_timestamp,
                        model=req.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                delta={"content": f"\n\nError: {chunk_data['error']}"},
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    break

    except Exception as e:
        logger.error(f"Error in stateless streaming completion: {e}", exc_info=True)
        # Send error as final chunk
        error_chunk = ChatCompletionStreamChunk(
            id=completion_id,
            created=created_timestamp,
            model=req.model,
            choices=[
                ChatCompletionStreamChoice(
                    delta={"content": f"\n\nError: {str(e)}"}, finish_reason="stop"
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    finally:
        # Send termination signal
        yield "data: [DONE]\n\n"


async def _create_non_streaming_completion(
    query: str, session_id: str, req: ChatCompletionRequest, components: AgentComponents
) -> ChatCompletionResponse:
    """Create non-streaming completion response (original logic)"""

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


async def _process_stateful_request(
    request: Request,
    session_id: str,
    req: ChatCompletionRequest,
    components: AgentComponents,
):
    """Process request in stateful mode with explicit session ID."""

    # Convert OpenAI messages to a single query (existing logic)
    query = ""
    for msg in req.messages:
        if msg.role == "user":
            query = msg.content
            break

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user message provided",
        )

    # Handle streaming vs non-streaming
    if req.stream:
        return StreamingResponse(
            _stream_chat_completion(query, session_id, req, components),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Session-ID": session_id,
            },
        )
    else:
        response = await _create_non_streaming_completion(query, session_id, req, components)
        # Add session ID to response headers
        return response


async def _process_stateless_request(
    req: ChatCompletionRequest,
    components: AgentComponents,
):
    """Process request in stateless mode using conversation fingerprinting."""

    # Extract and validate message array
    messages = req.messages
    if not messages or not any(msg.role == "user" for msg in messages):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user messages found",
        )

    # Convert messages to dict format for hashing
    message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

    # Generate conversation fingerprint
    conv_manager = ConversationManager()

    # Generate hash chain for all messages except the current query
    # The last user message is the current query
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    if last_user_idx == -1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user message found in conversation",
        )

    # Create hash chain for messages up to (but not including) the current query
    context_messages = message_dicts[:last_user_idx]
    message_hashes = conv_manager.create_hash_chain(context_messages) if context_messages else []

    # Find or create conversation
    conversation_id = None
    if message_hashes:
        conversation_id = await components.database.find_conversation_by_hash_chain(message_hashes)

    if not conversation_id:
        # New conversation - generate ID from first user message
        first_user_msg = next((msg.content for msg in messages if msg.role == "user"), "")
        conversation_id = conv_manager.generate_conversation_id(first_user_msg)
        await components.database.create_conversation_chain(conversation_id)

        # Store all context messages in the new conversation
        if context_messages:
            context_message_objs = [
                Message(role=msg["role"], content=msg["content"]) for msg in context_messages
            ]
            await components.database.append_messages_to_conversation(
                conversation_id, context_message_objs, message_hashes
            )
    else:
        # Check for conversation branching
        stored_messages = await components.database.load_conversation_by_id(conversation_id)
        stored_dicts = [{"role": msg.role, "content": msg.content} for msg in stored_messages]
        stored_hashes = conv_manager.create_hash_chain(stored_dicts)

        branch_point = conv_manager.detect_conversation_branch(stored_hashes, message_hashes)
        if branch_point is not None:
            # Create a new branch
            old_conversation_id = conversation_id
            conversation_id = conv_manager.create_conversation_branch(
                old_conversation_id, branch_point
            )
            await components.database.create_conversation_chain(
                conversation_id,
                metadata={"branched_from": old_conversation_id, "branch_point": branch_point},
            )

            # Copy messages up to branch point
            if branch_point > 0:
                branch_messages = stored_messages[:branch_point]
                branch_hashes = stored_hashes[:branch_point]
                await components.database.append_messages_to_conversation(
                    conversation_id, branch_messages, branch_hashes
                )

            # Add new messages after branch point
            new_messages_from_branch = context_messages[branch_point:]
            new_hashes_from_branch = message_hashes[branch_point:]
            if new_messages_from_branch:
                new_message_objs = [
                    Message(role=msg["role"], content=msg["content"])
                    for msg in new_messages_from_branch
                ]
                await components.database.append_messages_to_conversation(
                    conversation_id, new_message_objs, new_hashes_from_branch
                )

    # Get the current query
    current_query = messages[last_user_idx].content

    # Handle streaming vs non-streaming
    if req.stream:
        return StreamingResponse(
            _stream_stateless_completion(
                current_query, conversation_id, req, components, conv_manager, message_hashes
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Conversation-ID": conversation_id,
            },
        )
    else:
        return await _create_stateless_completion(
            current_query, conversation_id, req, components, conv_manager, message_hashes
        )


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    req: ChatCompletionRequest,
    components: AgentComponents = Depends(get_components),
):
    """OpenAI-compatible chat completion endpoint with streaming support."""

    # Get session ID if provided
    session_id = get_session_id(request)

    # Detect mode based on session ID presence
    # If session_id is a UUID that wasn't in the request, it's auto-generated
    has_explicit_session = False
    if request.headers.get("X-Session-ID") or request.cookies.get("session_id"):
        has_explicit_session = True

    if has_explicit_session:
        # Stateful mode - use existing logic
        return await _process_stateful_request(request, session_id, req, components)
    else:
        # Stateless mode - use conversation fingerprinting
        return await _process_stateless_request(req, components)


@app.get("/health")
async def health_check(components: AgentComponents = Depends(get_components)):
    """
    Health check endpoint.

    Returns information about the agent's components and their connection status.
    """
    # Get conversation statistics if database is connected
    conversation_stats = {}
    if components.database and components.database.is_connected():
        try:
            # Get conversation count
            conv_count = await components.database.execute_query(
                "SELECT COUNT(DISTINCT conversation_id) FROM conversation_chains", fetch_type="val"
            )
            message_count = await components.database.execute_query(
                "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id IS NOT NULL",
                fetch_type="val",
            )
            conversation_stats = {
                "conversation_count": conv_count or 0,
                "message_count": message_count or 0,
                "hybrid_mode_enabled": True,
            }
        except Exception as e:
            conversation_stats = {"error": str(e)}

    status = {
        "status": "healthy",
        "deployment_mode": "stack" if components.is_stack_deployment else "local",
        "database": {
            "connected": components.database.is_connected() if components.database else False,
            "error": components.database.last_error() if components.database else "Not initialized",
            "conversation_storage": conversation_stats,
        },
        "llm": {
            "connected": components.llm_connector.is_connected()
            if components.llm_connector
            else False,
        },
        "mcp_servers": {
            "count": len(components.transport_manager.get_all_clients())
            if components.transport_manager
            else 0,
            "servers": list(components.transport_manager.get_all_clients().keys())
            if components.transport_manager
            else [],
        },
        "hybrid_mode": {
            "enabled": True,
            "description": "Supports both stateful (session-based) and stateless (conversation fingerprinting) modes",
        },
        "timestamp": datetime.now().isoformat(),
    }

    return status


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
        "src.agent.api:app",
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
