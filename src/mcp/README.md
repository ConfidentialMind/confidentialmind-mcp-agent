# Building a Custom MCP Tool Server (Hackathon Guide)

This guide provides a quick, step-by-step process for creating your own custom MCP (Model-Client Protocol) tool server based on the existing framework in this repository. This is ideal for quickly integrating a new tool or data source during a hackathon.

## Goal

To create a new Python script (e.g., `src/mcp/my_custom_mcp_server.py`) that acts as an MCP server for a specific service (like a weather API, a different database, a custom API, etc.) and integrate it into the main `mcp_integration.py` application.

## Core Concepts

- **MCP Protocol (`mcp_protocol.py`):** Defines the standard JSON-RPC messages (like `listTools`, `callTool`) used for communication. You'll use the Pydantic models defined here.
- **MCP Client (`mcp_client.py`):** The main application uses this client to start and communicate with your MCP server process via standard input/output. You generally don't need to modify this.
- **MCP Server (Your Script):** A standalone Python script that:
  - Listens for JSON-RPC requests on `stdin`.
  - Processes requests based on the MCP protocol (e.g., listing available tools, executing a tool).
  - Sends JSON-RPC responses back via `stdout`.
- **Handler Class (In Your Script):** Contains the core logic for interacting with your target service (e.g., calling an API, querying a database) and mapping it to MCP methods.
- **Integration (`mcp_integration.py`):** Needs minor additions to know how to start and use your new MCP server based on environment variables.

## Steps

### 1. Define Your Service & Tools

- **What service will your server connect to?** (e.g., OpenWeatherMap API, a specific internal API, another database type).
- **What specific actions (tools) should the LLM be able to trigger?** (e.g., `get_current_weather`, `lookup_user_data`, `query_vector_db`).
- **What information (resources) might the LLM need to know about?** (e.g., available API endpoints, database table schemas - though often tools are sufficient).

### 2. Create Your Server File

- **Copy an existing server:** Duplicate `src/mcp/postgres_mcp_server.py` or `src/mcp/rag_mcp_server.py` and rename it (e.g., `src/mcp/weather_mcp_server.py`). This gives you the basic structure.
- **Clear out specifics:** Remove the PostgreSQL or RAG-specific logic from the copied file's Handler class (e.g., `PostgresHandler`, `RAGHandler`).

### 3. Implement Your Handler Class

- **Rename the Handler:** Rename the class (e.g., `WeatherHandler`).
- **`__init__`:** Modify the constructor to accept necessary configuration passed from the command line (like API keys, base URLs). Store these in instance variables.

```python
import json
import logging
import sys
from typing import Any, Dict, Optional

import requests  # Or other relevant libraries

# Import necessary models from mcp_protocol
from src.mcp.mcp_protocol import (
    CallToolRequestParams,
    CallToolResponse,
    JsonRpcRequest,
    JsonRpcResponse,
    ListResourcesResponse,
    ListToolsResponse,
    ReadResourceRequestParams,
    ReadResourceResponse,
    ResourceContent,
    ResourceIdentifier,
    TextContent,
    ToolDefinition,
    ToolInputSchema,
)

class WeatherHandler:
    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Accept": "application/json"}  # Example
        logging.info(f"Initialized WeatherHandler with base URL: {self.base_url}")
        if not api_key:
             logging.warning("API Key not provided for WeatherHandler")
             # Decide how to handle missing keys - maybe raise error or operate without auth if possible
```

- **`handle_list_tools`:** Define the tools your server provides. Use `ToolDefinition` and `ToolInputSchema` from `mcp_protocol.py`.

```python
def handle_list_tools(self) -> ListToolsResponse:
    logging.info("Handling mcp_listTools for Weather Service")
    return ListToolsResponse(
        tools=[
            ToolDefinition(
                name="get_current_weather",
                description="Gets the current weather for a specified city.",
                inputSchema=ToolInputSchema(
                    type="object",
                    properties={
                        "city": {
                            "type": "string",
                            "description": "The name of the city (e.g., London, Tokyo).",
                        },
                         "units": {
                            "type": "string",
                            "description": "Units for temperature (metric, imperial, standard). Default: metric.",
                        },
                    },
                    required=["city"],
                ),
            )
            # Add more ToolDefinitions here if needed
        ]
    )
```

- **`handle_call_tool`:** Implement the logic to execute your defined tools.
  - Check `params.name` to see which tool was called.
  - Extract arguments from `params.arguments`.
  - Call your target service (e.g., make an API request).
  - Format the result according to `CallToolResponse` (usually using `TextContent`). Handle errors gracefully.

```python
def handle_call_tool(self, params: CallToolRequestParams) -> CallToolResponse:
    logging.info(f"Handling mcp_callTool for tool: {params.name}")

    if params.name == "get_current_weather":
        if not params.arguments or "city" not in params.arguments:
            raise ValueError("Missing 'city' argument for 'get_current_weather'")

        city = params.arguments["city"]
        units = params.arguments.get("units", "metric")  # Default to metric

        try:
            api_endpoint = f"{self.base_url}/weather"
            query_params = {
                "q": city,
                "appid": self.api_key,
                "units": units,
            }
            response = requests.get(api_endpoint, params=query_params, headers=self.headers, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            weather_data = response.json()
            # Format the relevant data nicely for the LLM
            result_text = json.dumps({
                "city": weather_data.get("name"),
                "description": weather_data["weather"][0]["description"],
                "temperature": weather_data["main"]["temp"],
                "feels_like": weather_data["main"]["feels_like"],
                "humidity": weather_data["main"]["humidity"],
                "wind_speed": weather_data["wind"]["speed"],
            }, indent=2)

            logging.info(f"Successfully retrieved weather for {city}")
            return CallToolResponse(content=[TextContent(text=result_text)], isError=False)

        except requests.exceptions.RequestException as e:
            logging.error(f"API call failed for get_current_weather: {e}", exc_info=True)
            error_msg = f"API Error: Failed to get weather for {city}. Details: {str(e)}"
            # Try to get more specific error from response if available
            if hasattr(e, 'response') and e.response is not None:
                 try:
                     error_detail = e.response.json().get('message', e.response.text)
                     error_msg = f"API Error ({e.response.status_code}): {error_detail}"
                 except json.JSONDecodeError:
                     error_msg = f"API Error ({e.response.status_code}): {e.response.text}"

            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)
        except Exception as e:
             logging.error(f"Unexpected error in get_current_weather: {e}", exc_info=True)
             return CallToolResponse(content=[TextContent(text=f"Unexpected Server Error: {str(e)}")], isError=True)

    else:
        # Handle unknown tools
        raise ValueError(f"Unknown tool requested: {params.name}")
```

- **`handle_list_resources` / `handle_read_resource`:** Implement these if your service has discoverable resources. For many API-based tools, you might just return empty or minimal responses.

```python
def handle_list_resources(self) -> ListResourcesResponse:
    """
    For a Weather API, there might not be traditional resources to list.
    You could return empty or provide available endpoints as resources.
    """
    logging.info("Handling mcp_listResources for Weather Service")

    # Example: List available resource endpoints
    return ListResourcesResponse(
        resources=[
            ResourceIdentifier(
                uri="weather://api.openweathermap.org/current",
                name="Current Weather Data",
                mimeType="application/json"
            ),
            ResourceIdentifier(
                uri="weather://api.openweathermap.org/forecast",
                name="5-day Weather Forecast",
                mimeType="application/json"
            )
        ]
    )

def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
    """
    Read specific resource information based on URI.
    For APIs, this could provide documentation about endpoints.
    """
    logging.info(f"Handling mcp_readResource for URI: {params.uri}")

    uri = params.uri
    if "current" in uri:
        # Return documentation about the current weather endpoint
        content = json.dumps({
            "endpoint": "/weather",
            "parameters": {
                "q": "City name (required)",
                "units": "Units (metric, imperial, standard)",
                "lang": "Language code"
            },
            "description": "Gets current weather for a specific location"
        }, indent=2)

        return ReadResourceResponse(
            contents=[
                ResourceContent(
                    uri=uri,
                    text=content,
                    mimeType="application/json"
                )
            ]
        )
    elif "forecast" in uri:
        # Return documentation about the forecast endpoint
        content = json.dumps({
            "endpoint": "/forecast",
            "parameters": {
                "q": "City name (required)",
                "units": "Units (metric, imperial, standard)",
                "cnt": "Number of timestamps to return"
            },
            "description": "Gets 5-day forecast for a specific location"
        }, indent=2)

        return ReadResourceResponse(
            contents=[
                ResourceContent(
                    uri=uri,
                    text=content,
                    mimeType="application/json"
                )
            ]
        )
    else:
        return ReadResourceResponse(
            contents=[
                ResourceContent(
                    uri=uri,
                    text="Resource not found or not supported.",
                    mimeType="text/plain"
                )
            ]
        )
```

- **Resource Cleanup:** If your service requires connection cleanup (like database connections), implement a `close()` method that will be called when the server shuts down.

```python
def close(self):
    """Clean up any persistent connections or resources."""
    logging.info("Cleaning up Weather handler resources")
    # Example: close persistent connections if applicable
    # self.session.close()
```

### 4. Adapt the Server Loop (`if __name__ == "__main__":`)

- Modify the argument parsing (`sys.argv`) at the bottom of your server file to accept the configuration your Handler needs (e.g., API key, URL).
- Instantiate your Handler with these arguments.
- Keep the `run_server` function largely the same, ensuring it calls your Handler's methods.

```python
# In your src/mcp/weather_mcp_server.py (at the bottom)

def run_server(api_key: str, base_url: Optional[str] = None):
    # Pass arguments to your handler
    handler_args = {"api_key": api_key}
    if base_url:
        handler_args["base_url"] = base_url
    handler = WeatherHandler(**handler_args)

    logging.info("Weather MCP Server Ready. Waiting for JSON-RPC requests on stdin...")
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
    finally:
        # Call close() if implemented to clean up resources
        if hasattr(handler, 'close'):
            handler.close()
        logging.info("Server shutdown complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:  # Expecting at least the API Key
        print("Usage: python weather_mcp_server.py <api_key> [base_url]", file=sys.stderr)
        sys.exit(1)

    api_key_arg = sys.argv[1]
    base_url_arg = sys.argv[2] if len(sys.argv) > 2 else None  # Optional base URL

    run_server(api_key_arg, base_url_arg)
```

### 5. Integrate with mcp_integration.py

- **Add Environment Variables:** Decide on environment variable names for your server's configuration (e.g., WEATHER_API_KEY, WEATHER_API_URL). Add these to your `.env` file.

```
# Example .env file entries for Weather MCP server
WEATHER_API_KEY=your_openweathermap_api_key
WEATHER_API_URL=https://api.openweathermap.org/data/2.5  # Optional, default in code

# Other existing variables
PG_CONNECTION_STRING=postgresql://username:password@hostname:5432/database
LLM_URL=http://localhost:8080/v1
LLM_API_KEY=your_llm_api_key
RAG_API_URL=https://api.your-rag-service.com/v1/api/your-project-id
RAG_API_KEY=your-api-key-here
```

- **Create Client Function:** Add a new function `create_weather_mcp_client` (or similar) in `mcp_integration.py`. This function should:
  - Read the necessary environment variables.
  - Construct the server_command string to run your new server script with the required arguments (e.g., API key).
  - Instantiate and return `MCPClient(server_command=server_command)`.
  - Add the client instance to the global `mcp_client_instances` dictionary for cleanup.

```python
# In mcp_integration.py

def create_weather_mcp_client() -> MCPClient:
    """Creates and starts the MCPClient for the Weather server."""
    global mcp_client_instances
    if "weather" in mcp_client_instances:
        logger.warning("Weather MCPClient instance already exists.")
        return mcp_client_instances["weather"]

    weather_api_key = os.environ.get("WEATHER_API_KEY")
    if not weather_api_key:
        logger.error("FATAL: WEATHER_API_KEY environment variable is not set.")
        raise ValueError("WEATHER_API_KEY must be set in the environment.")

    weather_base_url = os.environ.get("WEATHER_API_URL")  # Optional

    # Define the command to run the MCP server
    server_script_path = os.path.join(os.path.dirname(__file__), "src", "mcp", "weather_mcp_server.py")
    # Add fallback path checking similar to other clients
    if not os.path.exists(server_script_path):
        # Check alternative paths
        for alt_path in ["src/mcp/weather_mcp_server.py", "weather_mcp_server.py"]:
            if os.path.exists(alt_path):
                server_script_path = alt_path
                break
        else:
            logger.warning("Weather MCP server script not found at any expected path")
            # Proceed with the original path as a last resort
            server_script_path = "weather_mcp_server.py"

    server_command = f"{sys.executable} {server_script_path} {weather_api_key}"
    if weather_base_url:
         server_command += f" {weather_base_url}"  # Append optional URL if provided

    try:
        logger.info("Initializing Weather MCPClient...")
        client = MCPClient(server_command=server_command)
        mcp_client_instances["weather"] = client  # Use a unique key like "weather"
        logger.info("Weather MCPClient initialized and server process start initiated.")
        return client
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize Weather MCPClient: {e}", exc_info=True)
        cleanup_mcp_clients()  # Ensure cleanup attempt
        raise
```

- **Modify `create_mcp_clients`:** Add a check for your environment variables in `create_mcp_clients` and call your new creation function.

```python
# In mcp_integration.py, inside create_mcp_clients function

def create_mcp_clients() -> Dict[str, MCPClient]:
    """Creates and starts MCPClients for all supported services."""
    clients = {}

    # ... (existing code for postgres and rag clients) ...

    # Create Weather client if configured
    if os.environ.get("WEATHER_API_KEY"):
        try:
            weather_client = create_weather_mcp_client()
            clients["weather"] = weather_client  # Use the same unique key
        except Exception as e:
            logger.error(f"Failed to create Weather client: {e}")
    else:
        logger.warning("WEATHER_API_KEY not set, skipping Weather MCP client creation")

    if not clients:
        raise ValueError("No MCP clients could be created. Check your configuration.")

    return clients
```

### 6. Test

- Ensure your `.env` file has the new environment variables set (e.g., `WEATHER_API_KEY=yourkey`).

- Run the main integration script: `python mcp_integration.py`

- In the interactive prompt, try queries that should use your new tool:

  - "What tools are available?" (You should see your new tool listed).
  - "What is the current weather in Paris?" (Or another query targeting your tool).

- Check the logs for output from both the main script and your `weather_mcp_server.py` script for debugging.

## Hackathon Tips

- **Keep it Simple:** Implement only the essential tools needed for your hackathon project.

- **Reuse Code:** Leverage the structure and error handling from `postgres_mcp_server.py` or `rag_mcp_server.py`.

- **Focus on call_tool:** This is where the core logic of your integration lies.

- **Stateless:** Aim for stateless tool calls if possible, as it simplifies the server.

- **Error Handling:** Return informative error messages in the `CallToolResponse(isError=True)` format so the LLM knows what went wrong.

- **Logging:** Add `logging.info()` and `logging.debug()` statements liberally to help with debugging.

- **Test Incrementally:** Test each component as you build it. For example, test your handler methods directly before integrating them into the server.

- **Graceful Degradation:** Design your server to handle missing configurations and partial functionality. For example, if a specific API endpoint is down, the server should still handle other tool calls.
