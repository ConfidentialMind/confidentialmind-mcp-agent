import json
import logging
from typing import Any, Dict, Optional

import httpx

from src.mcp.mcp_protocol import JsonRpcRequest, JsonRpcResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [MCP Client] %(levelname)s: %(message)s"
)


class MCPClient:
    """HTTP-based connector for MCP-compliant tool servers"""

    def __init__(self, base_url: str):
        """Initialize MCP client with the base URL of the MCP server

        Args:
            base_url: URL of the MCP server (e.g., http://postgres-mcp:8001)
        """
        self.base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self._request_id_counter = 0
        self.timeout = 30  # Default timeout in seconds

        logging.info(f"Initialized MCP client for server at {self.base_url}")

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a request to the MCP server using HTTP"""
        # Increment request ID counter
        self._request_id_counter += 1
        request_id = self._request_id_counter

        # Create JSON-RPC request
        request = JsonRpcRequest(id=request_id, method=method, params=params)
        request_json = request.model_dump(exclude_none=True)

        logging.debug(f"Sending request to {self.base_url}/mcp: {json.dumps(request_json)}")

        try:
            # Send HTTP POST request to the MCP server
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/mcp",
                    json=request_json,
                    headers={"Content-Type": "application/json"},
                )

                # Raise exception for HTTP errors
                response.raise_for_status()

                # Parse the JSON response
                response_data = response.json()
                logging.debug(f"Received response: {json.dumps(response_data)}")

                # Validate the response
                response_obj = JsonRpcResponse.model_validate(response_data)

                # Check for error
                if response_obj.error:
                    error_msg = response_obj.error.get("message", "Unknown error")
                    error_code = response_obj.error.get("code", -1)
                    logging.error(f"MCP server returned error: [{error_code}] {error_msg}")
                    raise RuntimeError(f"MCP Error: {error_msg}")

                # Return the result
                return response_obj.result

        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"HTTP error communicating with MCP server: {e}")
        except httpx.RequestError as e:
            logging.error(f"Request error: {e}")
            raise RuntimeError(f"Error communicating with MCP server: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise RuntimeError(f"Unexpected error in MCP client: {e}")

    # --- Convenience methods for MCP calls ---

    def list_resources(self) -> Dict[str, Any]:
        """List available resources from the MCP server"""
        return self._send_request("mcp_listResources")

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a specific resource from the MCP server"""
        return self._send_request("mcp_readResource", {"uri": uri})

    def list_tools(self) -> Dict[str, Any]:
        """List available tools from the MCP server"""
        return self._send_request("mcp_listTools")

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        return self._send_request("mcp_callTool", {"name": name, "arguments": arguments})
