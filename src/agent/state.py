from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in the conversation history"""

    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role.upper()}: {self.content}"


class AgentState(BaseModel):
    """Agent workflow state."""

    query: str = ""
    session_id: Optional[str] = None
    mcp_results: Dict[str, Any] = Field(default_factory=dict)
    thoughts: List[str] = Field(default_factory=list)
    response: Optional[str] = None
    error: Optional[str] = None
    planned_actions: List[Dict[str, Any]] = Field(default_factory=list)
    current_action_index: int = 0
    needs_more_info: bool = False
    follow_up_question: Optional[str] = None
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    requires_replanning: bool = False
    replan_count: int = 0
    max_replan_attempts: int = 5  # TODO: parameterize this to CM SDK later
