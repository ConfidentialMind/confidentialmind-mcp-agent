"""FastMCP client-based agent for interacting with MCP servers."""

from src.agent.agent import Agent
from src.agent.database import Database, DatabaseSettings, fetch_db_url
from src.agent.llm import LLMConnector
from src.agent.state import AgentState, Message

# Make API components accessible when installed
try:
    from .api import app as api_app
    from .api import start_api_server
except ImportError:
    # FastAPI might not be installed - this is okay for CLI-only usage
    pass

__all__ = [
    "Agent",
    "Database",
    "DatabaseSettings",
    "fetch_db_url",
    "LLMConnector",
    "AgentState",
    "Message",
]

# Export API components if available
try:
    __all__.extend(["api_app", "start_api_server"])
except NameError:
    pass
