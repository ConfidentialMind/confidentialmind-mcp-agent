import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from confidentialmind_core.config_manager import ConfigManager, ConnectorSchema
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.database import get_db, get_messages_for_session, init_db, save_message
from src.connectors.cm_llm_connector import CMLLMConnector
from src.connectors.cm_mcp_connector import CMMCPManager
from src.core.agent import Agent, Message

app = FastAPI(title="ConfidentialMind MCP Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models for OpenAI-compatible API
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
    object: str = Field("chat.completion", const=True)
    created: int = Field(..., example=1677858242)
    model: str = Field(..., example="cm-agent-1")
    choices: List[Dict] = Field(...)
    usage: Dict = Field(...)


# Initialize SDK-based MCP manager
def get_mcp_manager():
    manager = CMMCPManager()
    manager.register_from_environment()
    return manager


# Initialize SDK-based LLM connector
def get_llm_connector():
    return CMLLMConnector(config_id="LLM")


@app.on_event("startup")
async def startup_event():
    await init_db()

    # Initialize SDK Config Manager during startup
    from confidentialmind_core.config_manager import ConfigManager

    config_manager = ConfigManager()

    # Register standard MCP servers in the SDK configuration
    if os.environ.get("PG_MCP_URL"):
        server_url = os.environ.get("PG_MCP_URL")
        config_manager.set_connector("postgres", server_url, {"Content-Type": "application/json"})
        print(f"Registered PostgreSQL MCP server at {server_url}")

    if os.environ.get("RAG_MCP_URL"):
        server_url = os.environ.get("RAG_MCP_URL")
        config_manager.set_connector("rag", server_url, {"Content-Type": "application/json"})
        print(f"Registered RAG MCP server at {server_url}")

    if os.environ.get("OBSIDIAN_MCP_URL"):
        server_url = os.environ.get("OBSIDIAN_MCP_URL")
        config_manager.set_connector("obsidian", server_url, {"Content-Type": "application/json"})
        print(f"Registered Obsidian MCP server at {server_url}")

    # Register LLM configuration (assuming LLM_URL and LLM_API_KEY environment variables)
    if os.environ.get("LLM_URL"):
        llm_url = os.environ.get("LLM_URL")
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        config_manager.set_connector(
            "LLM",
            llm_url,
            {"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"},
        )
        print(f"Registered LLM provider at {llm_url}")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    db=Depends(get_db),
    mcp_manager=Depends(get_mcp_manager),
    llm_connector=Depends(get_llm_connector),
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

        # Create agent instance with SDK-based connectors
        agent = Agent(
            llm_connector=llm_connector,
            mcp_manager=mcp_manager,
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
                "prompt_tokens": 0,  # We don't track these yet
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


# Add route to check which MCP servers are available
@app.get("/mcps")
async def list_mcps(mcp_manager=Depends(get_mcp_manager)):
    try:
        clients = mcp_manager.get_all_clients()
        available_servers = {}

        for server_id, client in clients.items():
            try:
                # Test the connection by listing tools
                tools = client.list_tools()
                available_servers[server_id] = {
                    "status": "available",
                    "tools_count": len(tools.get("tools", [])),
                }
            except Exception as e:
                available_servers[server_id] = {"status": "error", "error": str(e)}

        return {
            "registered_servers": list(mcp_manager.registered_servers),
            "available_servers": available_servers,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing MCP servers: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
