import os
import time
import uuid
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.database import get_db, get_messages_for_session, init_db, save_message
from src.connectors.llm import LLMConnector
from src.core.agent import Agent, Message
from src.mcp.mcp_client import MCPClient

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


# Initialize MCP clients
def get_mcp_clients():
    clients = {}

    # PostgreSQL MCP client
    if os.environ.get("PG_MCP_URL"):
        clients["postgres"] = MCPClient(base_url=os.environ.get("PG_MCP_URL"))

    # RAG MCP client
    if os.environ.get("RAG_MCP_URL"):
        clients["rag"] = MCPClient(base_url=os.environ.get("RAG_MCP_URL"))

    # Obsidian MCP client
    if os.environ.get("OBSIDIAN_MCP_URL"):
        clients["obsidian"] = MCPClient(base_url=os.environ.get("OBSIDIAN_MCP_URL"))

    if not clients:
        raise ValueError("No MCP clients could be created. Check your configuration.")

    return clients


# Initialize LLM connector
def get_llm_connector():
    llm_url = os.environ.get("LLM_URL", "http://localhost:8080/v1")
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    return LLMConnector(base_url=llm_url, headers={"Authorization": f"Bearer {llm_api_key}"})


@app.on_event("startup")
async def startup_event():
    await init_db()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    db=Depends(get_db),
    mcp_clients=Depends(get_mcp_clients),
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

        # Create agent instance
        agent = Agent(llm_connector, mcp_clients)

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
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
