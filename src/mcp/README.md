# Building a Simple Weather MCP Server (Hackathon Guide)

This guide shows how to create a simple MCP tool server that provides weather information. Users will be able to ask for weather by city name, and your server will handle all the complexity behind the scenes.

## What We'll Build

A simple MCP server that:

1. Accepts a city name as input
2. Performs geocoding (converting city to coordinates) automatically
3. Fetches current weather and forecasts
4. Returns a nicely formatted response

## Prerequisites

- Python 3.8+
- An OpenWeatherMap API key ([sign up here](https://home.openweathermap.org/users/sign_up))
- Basic understanding of HTTP requests and JSON

## Step 1: Create Your Server File

Create a new file `src/mcp/weather_mcp_server.py` with the basic structure:

```python
import json
import logging
import sys
from typing import Optional

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
    level=logging.INFO, format="%(asctime)s [Weather MCP Server] %(levelname)s: %(message)s"
)

class WeatherHandler:
    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org"):
        self.api_key = api_key
        self.base_url = base_url
        logging.info(f"Initialized WeatherHandler with base URL: {self.base_url}")
        if not api_key:
            logging.warning("API Key not provided for WeatherHandler")
            raise ValueError("API key is required for the Weather MCP server")
```

## Step 2: Implement Tool Definition

Add the method to handle listing available tools:

```python
def handle_list_tools(self) -> ListToolsResponse:
    """Provide tool definitions for weather operations"""
    logging.info("Handling mcp_listTools")
    return ListToolsResponse(
        tools=[
            ToolDefinition(
                name="get_weather",
                description="Gets current weather and forecast for a city.",
                inputSchema=ToolInputSchema(
                    type="object",
                    properties={
                        "city": {
                            "type": "string",
                            "description": "The name of the city (e.g., London, Tokyo, New York).",
                        },
                        "units": {
                            "type": "string",
                            "description": "Units for temperature (metric, imperial, standard). Default: metric.",
                        },
                    },
                    required=["city"],
                ),
            ),
        ]
    )
```

## Step 3: Implement Tool Execution

Add the method to handle tool execution:

```python
def handle_call_tool(self, params: CallToolRequestParams) -> CallToolResponse:
    """Handle weather-related tool calls with city name input"""
    logging.info(f"Handling mcp_callTool for tool: {params.name}")

    if params.name == "get_weather":
        if not params.arguments or "city" not in params.arguments:
            raise ValueError("Missing 'city' argument for 'get_weather' tool")

        city = params.arguments["city"]
        units = params.arguments.get("units", "metric")  # Default to metric

        try:
            # Step 1: Convert city to coordinates using the Geocoding API
            logging.info(f"Converting city name to coordinates: {city}")
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {
                "q": city,
                "limit": 1,
                "appid": self.api_key
            }

            geo_response = requests.get(geo_url, params=geo_params, timeout=10)
            geo_response.raise_for_status()

            locations = geo_response.json()
            if not locations:
                return CallToolResponse(
                    content=[TextContent(text=f"City not found: {city}")],
                    isError=True
                )

            # Get coordinates from first result
            lat = locations[0]["lat"]
            lon = locations[0]["lon"]
            actual_city_name = locations[0].get("name", city)
            country = locations[0].get("country", "")

            # Step 2: Fetch weather data with the coordinates
            weather_url = f"{self.base_url}/data/3.0/onecall"
            weather_params = {
                "lat": lat,
                "lon": lon,
                "units": units,
                "exclude": "minutely",  # Exclude minutely data
                "appid": self.api_key
            }

            weather_response = requests.get(weather_url, params=weather_params, timeout=15)
            weather_response.raise_for_status()

            weather_data = weather_response.json()

            # Step 3: Format the data in a simple structure
            temp_unit = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
            speed_unit = "m/s" if units == "metric" else "mph" if units == "imperial" else "m/s"

            current = weather_data.get("current", {})
            weather_desc = current.get("weather", [{}])[0].get("description", "unknown") if current.get("weather") else "unknown"

            # Create a simple weather summary
            formatted_data = {
                "location": f"{actual_city_name}, {country}",
                "current_weather": {
                    "temperature": f"{current.get('temp')} {temp_unit}",
                    "feels_like": f"{current.get('feels_like')} {temp_unit}",
                    "description": weather_desc,
                    "humidity": f"{current.get('humidity')}%",
                    "wind_speed": f"{current.get('wind_speed')} {speed_unit}",
                    "cloud_coverage": f"{current.get('clouds')}%"
                },
                "forecast": []
            }

            # Add daily forecast for next 3 days
            for day in weather_data.get("daily", [])[:3]:
                from datetime import datetime
                day_date = datetime.fromtimestamp(day.get("dt", 0)).strftime("%A, %b %d")
                day_weather = day.get("weather", [{}])[0].get("description", "unknown") if day.get("weather") else "unknown"

                forecast_day = {
                    "date": day_date,
                    "min_temp": f"{day.get('temp', {}).get('min')} {temp_unit}",
                    "max_temp": f"{day.get('temp', {}).get('max')} {temp_unit}",
                    "description": day_weather,
                    "chance_of_rain": f"{int(day.get('pop', 0) * 100)}%"
                }
                formatted_data["forecast"].append(forecast_day)

            result_json = json.dumps(formatted_data, indent=2)
            return CallToolResponse(content=[TextContent(text=result_json)], isError=False)

        except requests.exceptions.RequestException as e:
            error_msg = f"Weather API Error: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

        except Exception as e:
            error_msg = f"Error fetching weather: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)
    else:
        # Handle unknown tools
        raise ValueError(f"Unknown tool requested: {params.name}")
```

## Step 4: Implement Resource Handlers

Add simple implementations for resource listing and reading:

```python
def handle_list_resources(self) -> ListResourcesResponse:
    """List available resources (minimal for weather API)"""
    logging.info("Handling mcp_listResources")
    return ListResourcesResponse(resources=[])  # No browsable resources needed

def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
    """Read resources (minimal for weather API)"""
    logging.info(f"Handling mcp_readResource for URI: {params.uri}")
    return ReadResourceResponse(
        contents=[
            ResourceContent(
                uri=params.uri,
                text="Resource reading is not supported by the Weather MCP server.",
                mimeType="text/plain",
            )
        ]
    )
```

## Step 5: Set Up the Server Loop

Add the main server loop and command-line interface:

```python
def run_server(api_key: str, base_url: Optional[str] = None):
    """Run the Weather MCP server"""
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python weather_mcp_server.py <api_key> [base_url]", file=sys.stderr)
        sys.exit(1)

    api_key_arg = sys.argv[1]
    base_url_arg = sys.argv[2] if len(sys.argv) > 2 else None

    run_server(api_key_arg, base_url_arg)
```

## Step 6: Integrate with mcp_integration.py

Now let's add the code to `mcp_integration.py` to create and manage your Weather MCP client:

1. Add a new function to create the Weather MCP client:

```python
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

    # Define the command to run the MCP server
    server_script_path = os.path.join(os.path.dirname(__file__), "src", "mcp", "weather_mcp_server.py")
    # Add fallback path checking
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

2. Modify the `create_mcp_clients` function to include your weather client:

```python
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

## Step 7: Test Your Integration

1. Add your OpenWeatherMap API key to your `.env` file:

```
WEATHER_API_KEY=your_openweathermap_api_key
```

2. Run the integration script:

```bash
python mcp_integration.py
```

3. Test with sample queries in the interactive prompt:

- "What's the weather like in London?"
- "What's the forecast for Tokyo for the next few days?"
- "Is it going to rain in New York today?"

## Conclusion

You've now created a simple Weather MCP server that:

- Handles geocoding automatically (city name to coordinates)
- Fetches current weather and forecasts
- Formats the data in a clean, easy-to-read structure
- Provides helpful error messages

The LLM can now use this tool to answer weather-related questions without needing to know about coordinates or the underlying API complexity.

## Next steps

Now that you have created your own MCP tool server, consider:

- Making your own custom MCP server for a database
- Extending the internal logic of the agent workflow
- Testing different models and how they perform
