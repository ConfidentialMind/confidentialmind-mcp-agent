import json
import logging
import os
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.mcp.mcp_protocol import (
    CallToolRequestParams,
    CallToolResponse,
    JsonRpcResponse,
    ListResourcesResponse,
    ListToolsResponse,
    ReadResourceRequestParams,
    ReadResourceResponse,
    ResourceContent,
    TextContent,
    ToolDefinition,
    ToolInputSchema,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [RAG MCP Server] %(levelname)s: %(message)s"
)
logger = logging.getLogger("rag-mcp")

# Create FastAPI app
app = FastAPI(title="RAG MCP Server")


# RAG handler class
class RAGHandler:
    """Handler for RAG-specific MCP requests"""

    def __init__(self, rag_base_url: str, api_key: Optional[str] = None):
        self.rag_base_url = rag_base_url.rstrip("/")  # Remove trailing slash if present
        self.api_key = api_key
        self.headers = {}

        # Set up authentication headers if API key is provided
        if api_key:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            logging.info(f"Initialized RAGHandler with base URL: {rag_base_url} and API key")
        else:
            logging.info(f"Initialized RAGHandler with base URL: {rag_base_url} (no API key)")

    def handle_list_tools(self) -> ListToolsResponse:
        """Provide tool definitions for RAG operations"""
        logging.info("Handling mcp_listTools")
        return ListToolsResponse(
            tools=[
                ToolDefinition(
                    name="rag_get_context",
                    description="Retrieves relevant context chunks from the RAG system based on a query.",
                    inputSchema=ToolInputSchema(
                        type="object",
                        properties={
                            "query": {
                                "type": "string",
                                "description": "The query to retrieve context for.",
                            },
                            "max_chunks": {
                                "type": "integer",
                                "description": "Maximum number of context chunks to retrieve.",
                            },
                        },
                        required=["query"],
                    ),
                ),
            ]
        )

    def handle_call_tool(self, params: CallToolRequestParams) -> CallToolResponse:
        """Handle RAG tool calls by routing to appropriate handler method"""
        logging.info(f"Handling mcp_callTool for tool: {params.name}")

        if params.name == "rag_get_context":
            return self._handle_get_context(params.arguments or {})
        elif params.name == "rag_generate_completion":
            return self._handle_generate_completion(params.arguments or {})
        else:
            raise ValueError(f"Unknown tool: {params.name}")

    def _handle_get_context(self, arguments: Dict[str, Any]) -> CallToolResponse:
        """Handle context retrieval from RAG system"""
        if "query" not in arguments:
            raise ValueError("Missing 'query' argument for 'rag_get_context' tool")

        query = arguments["query"]
        max_chunks = arguments.get("max_chunks", 3)  # Default to 3 chunks

        try:
            logging.info(f"Requesting context for query: {query}")

            # Log the request for debugging
            logging.debug(
                f"RAG context request to {self.rag_base_url}/context: {json.dumps({'query': query, 'max_chunks': max_chunks})}"
            )

            response = requests.post(
                f"{self.rag_base_url}/context",
                json={"query": query, "max_chunks": max_chunks},
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Format the result as a string for the CallToolResponse
            result_json = json.dumps(result, indent=2)
            logging.info(
                f"Context retrieval successful, returned {len(result.get('chunks', []))} chunks"
            )
            return CallToolResponse(content=[TextContent(text=result_json)], isError=False)

        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors specifically (like 401, 403, etc.)
            status_code = e.response.status_code if hasattr(e, "response") else "unknown"
            error_detail = ""

            if status_code == 401 or status_code == 403:
                error_detail = "Authentication failed. Please check the API key."
            elif status_code == 422:
                # For 422 errors, try to extract more details from the response
                error_detail = "The request was rejected due to invalid parameters. "
                try:
                    if hasattr(e, "response") and e.response.text:
                        error_json = e.response.json()
                        if "detail" in error_json:
                            error_detail += f"Details: {error_json['detail']}"
                        elif "error" in error_json:
                            error_detail += f"Details: {error_json['error']}"
                except Exception:
                    error_detail += "For chat completions, ensure messages are in the format [{role: 'user', content: '...'}]."

                # Log the actual request payload for debugging
                logging.error(f"Request payload that caused 422 error: {json.dumps(arguments)}")

            elif status_code == 404:
                error_detail = "The requested endpoint was not found."
            elif status_code == 429:
                error_detail = "Rate limit exceeded. Please try again later."
            elif status_code >= 500:
                error_detail = "Server error. The RAG service might be experiencing issues."

            logging.error(f"Context retrieval HTTP error: {status_code} - {e}", exc_info=True)
            return CallToolResponse(
                content=[
                    TextContent(text=f"RAG Context Error ({status_code}): {str(e)}. {error_detail}")
                ],
                isError=True,
            )
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Context retrieval connection error: {e}", exc_info=True)
            return CallToolResponse(
                content=[
                    TextContent(
                        text=f"RAG Connection Error: Could not connect to the RAG service at {self.rag_base_url}. Please check if the service is running and the URL is correct."
                    )
                ],
                isError=True,
            )
        except Exception as e:
            logging.error(f"Context retrieval error: {e}", exc_info=True)
            return CallToolResponse(
                content=[TextContent(text=f"RAG Context Error: {str(e)}")],
                isError=True,
            )

    def _handle_generate_completion(self, arguments: Dict[str, Any]) -> CallToolResponse:
        # This method would be implemented if needed, but we're keeping the same functionality
        # as the original code which doesn't fully implement this method
        raise ValueError("rag_generate_completion is not implemented in this server")

    def handle_list_resources(self) -> ListResourcesResponse:
        """
        For the RAG server, there are no traditional resources to list.
        Return an empty list.
        """
        logging.info("Handling mcp_listResources (empty for RAG)")
        return ListResourcesResponse(resources=[])

    def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
        """
        For the RAG server, there are no traditional resources to read.
        Return an empty response or error.
        """
        logging.info(f"Handling mcp_readResource (not supported in RAG)")
        return ReadResourceResponse(
            contents=[
                ResourceContent(
                    uri=params.uri,
                    text="Resource reading is not supported by the RAG MCP server.",
                    mimeType="text/plain",
                )
            ]
        )


# Request model for JSON-RPC requests
class MCPRequest(BaseModel):
    jsonrpc: str
    id: Any
    method: str
    params: Optional[Dict[str, Any]] = None


# Global RAG handler instance
rag_handler = None


@app.on_event("startup")
async def startup_event():
    global rag_handler

    # Get RAG API URL and key from environment variables
    rag_api_url = os.environ.get("RAG_API_URL")
    if not rag_api_url:
        raise RuntimeError("RAG_API_URL environment variable is not set")

    rag_api_key = os.environ.get("RAG_API_KEY")

    # Initialize RAG handler
    try:
        rag_handler = RAGHandler(rag_api_url, rag_api_key)
        logging.info("RAG handler initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize RAG handler: {e}", exc_info=True)
        raise


@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    global rag_handler

    if not rag_handler:
        raise HTTPException(status_code=500, detail="RAG handler not initialized")

    logging.info(f"Received MCP request: method={request.method}, id={request.id}")

    response_data = None
    error_data = None

    try:
        if request.method == "mcp_listResources":
            response_data = rag_handler.handle_list_resources().model_dump()
        elif request.method == "mcp_readResource":
            params = ReadResourceRequestParams.model_validate(request.params or {})
            response_data = rag_handler.handle_read_resource(params).model_dump()
        elif request.method == "mcp_listTools":
            response_data = rag_handler.handle_list_tools().model_dump()
        elif request.method == "mcp_callTool":
            params = CallToolRequestParams.model_validate(request.params or {})
            response_data = rag_handler.handle_call_tool(params).model_dump()
        else:
            raise ValueError(f"Unsupported MCP method: {request.method}")
    except Exception as e:
        logging.error(f"Error handling request {request.id}: {e}", exc_info=True)
        error_data = {"code": -32000, "message": str(e)}

    # Create JSON-RPC response
    response = JsonRpcResponse(id=request.id, result=response_data, error=error_data)
    return response.model_dump(exclude_none=True)


@app.get("/health")
async def health_check():
    global rag_handler

    if not rag_handler:
        raise HTTPException(status_code=500, detail="RAG handler not initialized")

    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
