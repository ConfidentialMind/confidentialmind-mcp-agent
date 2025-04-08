import json
import logging
import sys
from typing import Any, Dict, Optional

import requests
from mcp_protocol import (
    CallToolRequestParams,
    CallToolResponse,
    JsonRpcRequest,
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [RAG MCP Server] %(levelname)s: %(message)s"
)


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

    def _accumulate_streaming_response(self, response) -> Dict[str, Any]:
        """Accumulate chunks from a streaming response into a single response object."""
        accumulated_response = None

        try:
            for line in response.iter_lines():
                if line:
                    # Remove the "data: " prefix
                    line_text = line.decode("utf-8")
                    if line_text.startswith("data:"):
                        data_str = line_text[5:].strip()
                        if data_str == "[DONE]":
                            break

                        # Parse the JSON chunk
                        try:
                            chunk = json.loads(data_str)

                            # Initialize the accumulated response from the first chunk
                            if accumulated_response is None:
                                accumulated_response = chunk
                                # Initialize an empty content in the message
                                if (
                                    "choices" in accumulated_response
                                    and accumulated_response["choices"]
                                ):
                                    first_choice = accumulated_response["choices"][0]
                                    if "delta" in first_choice:
                                        # Convert delta to message structure
                                        role = first_choice["delta"].get("role", "assistant")
                                        accumulated_response["choices"][0]["message"] = {
                                            "role": role,
                                            "content": "",
                                        }
                                        # Remove delta from accumulated response
                                        if "delta" in accumulated_response["choices"][0]:
                                            del accumulated_response["choices"][0]["delta"]

                            # Accumulate content from delta
                            if (
                                "choices" in chunk
                                and chunk["choices"]
                                and "delta" in chunk["choices"][0]
                            ):
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta and delta["content"]:
                                    # Append to existing content
                                    accumulated_response["choices"][0]["message"]["content"] += (
                                        delta["content"]
                                    )

                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse streaming JSON chunk: {e}")

            # If we didn't receive any valid chunks
            if accumulated_response is None:
                accumulated_response = {
                    "choices": [{"message": {"role": "assistant", "content": ""}}]
                }

            return accumulated_response

        except Exception as e:
            logging.error(f"Error accumulating streaming response: {e}", exc_info=True)
            return {
                "error": f"Streaming error: {str(e)}",
                "choices": [{"message": {"role": "assistant", "content": ""}}],
            }

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


# --- Main Server Loop (Simplified Stdio JSON-RPC) ---


def run_server(rag_base_url: str, api_key: Optional[str] = None):
    handler = RAGHandler(rag_base_url, api_key)
    logging.info("RAG MCP Server Ready. Waiting for JSON-RPC requests on stdin...")

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                logging.info("Stdin closed, shutting down.")
                break

            logging.debug(f"Received line: {line.strip()}")
            try:
                request_data = json.loads(line)
                request = JsonRpcRequest.model_validate(request_data)
                logging.info(f"Processing request ID {request.id}, Method: {request.method}")

                response_data = None
                error_data = None

                try:
                    if request.method == "mcp_listResources":
                        response_data = handler.handle_list_resources().model_dump()
                    elif request.method == "mcp_readResource":
                        params = ReadResourceRequestParams.model_validate(request.params or {})
                        response_data = handler.handle_read_resource(params).model_dump()
                    elif request.method == "mcp_listTools":
                        response_data = handler.handle_list_tools().model_dump()
                    elif request.method == "mcp_callTool":
                        params = CallToolRequestParams.model_validate(request.params or {})
                        response_data = handler.handle_call_tool(params).model_dump()
                    else:
                        raise ValueError(f"Unsupported MCP method: {request.method}")

                except Exception as e:
                    logging.error(f"Error handling request {request.id}: {e}", exc_info=True)
                    error_data = {"code": -32000, "message": str(e)}

                response = JsonRpcResponse(id=request.id, result=response_data, error=error_data)

            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON: {line.strip()}")
                response = JsonRpcResponse(
                    id=None, error={"code": -32700, "message": "Parse error"}
                )
            except Exception as e:  # Catch validation errors etc.
                logging.error(f"General error processing input line: {e}", exc_info=True)
                # Try to get request ID if possible, otherwise use null
                req_id = request.id if "request" in locals() and hasattr(request, "id") else None
                response = JsonRpcResponse(
                    id=req_id,
                    error={"code": -32600, "message": f"Invalid Request: {e}"},
                )

            response_json = response.model_dump_json(exclude_none=True)
            logging.debug(f"Sending response: {response_json}")
            print(response_json, flush=True)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_mcp_server.py <rag_base_url> [api_key]", file=sys.stderr)
        sys.exit(1)

    rag_base_url_arg = sys.argv[1]
    api_key_arg = sys.argv[2] if len(sys.argv) > 2 else None

    run_server(rag_base_url_arg, api_key_arg)
