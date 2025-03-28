# mcp_protocol.py
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# --- Generic JSON-RPC ---
class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None


class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


# --- MCP Specific Resources ---
class ResourceIdentifier(BaseModel):
    uri: str
    name: Optional[str] = None
    mimeType: Optional[str] = None


class ResourceContent(BaseModel):
    uri: str
    text: str
    mimeType: str


# --- MCP ListResources ---
class ListResourcesRequestParams(BaseModel):
    pass  # No specific params for ListResources in this example


class ListResourcesResponse(BaseModel):
    resources: List[ResourceIdentifier]


# --- MCP ReadResource ---
class ReadResourceRequestParams(BaseModel):
    uri: str


class ReadResourceResponse(BaseModel):
    contents: List[ResourceContent]


# --- MCP Tools ---
class ToolInputSchema(BaseModel):
    type: Literal["object"]
    properties: Dict[str, Dict[str, str]]
    required: Optional[List[str]] = None


class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: Optional[ToolInputSchema] = None


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


# --- MCP ListTools ---
class ListToolsRequestParams(BaseModel):
    pass  # No specific params


class ListToolsResponse(BaseModel):
    tools: List[ToolDefinition]


# --- MCP CallTool ---
class CallToolRequestParams(BaseModel):
    name: str
    arguments: Optional[Dict[str, Any]] = None


class CallToolResponse(BaseModel):
    content: List[TextContent]  # Simplified, could include other types
    isError: bool = False
