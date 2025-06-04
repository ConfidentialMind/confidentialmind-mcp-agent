import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import AsyncExitStack
from typing import Any, AsyncGenerator, Dict, List, Optional

from confidentialmind_core.config_manager import load_environment
from fastmcp.exceptions import ClientError

from src.agent.connectors import ConnectorConfigManager
from src.agent.database import Database
from src.agent.llm import LLMConnector
from src.agent.observability import observe, update_observation_metadata
from src.agent.state import AgentState, Message
from src.agent.transport import TransportManager

logger = logging.getLogger(__name__)


class Agent:
    """FastMCP client-based agent implementation with flexible transport support"""

    def __init__(
        self,
        database: Database,
        llm_connector: LLMConnector,
        transport_manager: TransportManager,
        debug: bool = False,
    ):
        """
        Initialize the agent with necessary components.

        Args:
            database: Database instance for session storage
            llm_connector: LLM connector for generating responses
            transport_manager: Transport manager for MCP client configuration
            debug: Enable debug logging
        """
        self.db = database
        self.llm = llm_connector
        self.transport_manager = transport_manager
        self.debug = debug
        self.current_history: List[Message] = []
        self._has_tools = False
        self._db_connected = False
        self._llm_connected = False

        if debug:
            logger.setLevel(logging.DEBUG)

        self._exit_stack = AsyncExitStack()

        # Check if running in stack deployment mode
        load_environment()
        self._is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

    async def initialize(self) -> bool:
        """Initialize agent components."""
        logger.info("Initializing agent...")

        # Check connection status
        self._db_connected = self.db.is_connected()
        self._llm_connected = self.llm.is_connected()

        # Check if we're in stack deployment mode
        connector_manager = ConnectorConfigManager()
        is_stack_deployment = connector_manager.is_stack_deployment

        if is_stack_deployment:
            # Initialize connectors with stack
            await connector_manager.initialize()

            # Configure MCP transports from stack
            await self.transport_manager.configure_from_stack()

        # Create all clients (will use whichever transports were configured)
        self.transport_manager.create_all_clients()
        self._has_tools = len(self.transport_manager.get_all_clients()) > 0

        # Log available clients
        clients = self.transport_manager.get_all_clients()
        client_info = ", ".join([f"{k}" for k in clients.keys()]) if clients else "none"
        logger.info(f"Initialized agent with MCP clients: {client_info}")
        logger.info(f"Database connected: {self._db_connected}")
        logger.info(f"LLM connected: {self._llm_connected}")

        return True

    async def __aenter__(self):
        """Context manager entry that connects all clients."""
        await self._exit_stack.__aenter__()

        # Connect to all MCP clients
        for server_id, client in self.transport_manager.get_all_clients().items():
            try:
                logger.debug(f"Connecting to MCP server: {server_id}")
                await client.__aenter__()
                # Register exit callback to ensure cleanup
                self._exit_stack.push_async_callback(client.__aexit__, None, None, None)
                logger.debug(f"Successfully connected to MCP server: {server_id}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")
                # Continue with other servers

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    # --- Streaming Support ---

    @observe()
    async def run_streaming(
        self, query: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute agent workflow with streaming response generation.

        Yields:
            Dict containing workflow updates and final streaming response
        """
        # Update observation metadata
        update_observation_metadata(
            name="Agent Streaming Run",
            input={"query": query, "session_id": session_id},
            metadata={"streaming": True},
        )
        logger.info(f"Starting streaming agent run for query: '{query[:50]}...'")

        try:
            # Execute all stages up to response generation normally
            state = await self._execute_workflow_until_response(query, session_id)

            # If there was an error in the workflow, yield it and return
            if state.error:
                yield {
                    "type": "error",
                    "error": state.error,
                    "response": state.response or f"Error: {state.error}",
                }
                return

            # Yield initial workflow status
            yield {
                "type": "workflow_status",
                "stage": "response_generation",
                "thoughts": state.thoughts,
                "actions_executed": len(
                    [a for a in state.planned_actions[: state.current_action_index]]
                ),
                "total_actions": len(state.planned_actions),
            }

            # Stream the final response generation
            async for chunk in self._generate_response_streaming(state):
                yield chunk

        except Exception as e:
            logger.error(f"Error during streaming agent execution: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "response": "I encountered an error processing your request. Please try again or contact support if the issue persists.",
            }

    @observe()
    async def _execute_workflow_until_response(
        self, query: str, session_id: Optional[str] = None
    ) -> AgentState:
        """Execute workflow stages 1-5 (everything except response generation)"""
        # Update observation metadata
        update_observation_metadata(
            name="Execute Workflow",
            metadata={"stages": "1-5", "query_length": len(query)},
        )

        # Update connection status
        self._db_connected = self.db.is_connected()
        self._llm_connected = self.llm.is_connected()
        self._has_tools = len(self.transport_manager.get_all_clients()) > 0

        # Generate or use session ID
        definite_session_id = session_id or str(uuid.uuid4())
        logger.info(f"Session ID: {definite_session_id}")

        # Check if this is a conversation ID (starts with "conv_")
        conversation_id = None
        if definite_session_id.startswith("conv_"):
            conversation_id = definite_session_id
            # Use conversation_id as session_id for compatibility
            logger.info(f"Using conversation mode with ID: {conversation_id}")

        # Load history if database is connected
        await self._load_history(definite_session_id, conversation_id)

        # Handle management commands
        if query.strip().lower() == "clear history":
            if self._db_connected:
                await self.db.clear_history(definite_session_id)
                return AgentState(
                    query=query, session_id=definite_session_id, response="History cleared."
                )
            else:
                return AgentState(
                    query=query,
                    session_id=definite_session_id,
                    response="I'm running in stateless mode without access to database. No history to clear.",
                )

        if query.strip().lower() == "show history":
            history_text = "History:\n" + (
                "\n".join(map(str, self.current_history)) if self.current_history else "(empty)"
            )
            if not self._db_connected:
                history_text += "\n\n*Note: I'm running in stateless mode without database access. This is the in-memory history for this session only.*"

            return AgentState(query=query, session_id=definite_session_id, response=history_text)

        # Save user message (to memory even if DB not connected)
        user_message = Message(role="user", content=query)
        await self._save_message(definite_session_id, user_message)

        # Initialize state
        state = AgentState(query=query, session_id=definite_session_id)

        # If LLM is not connected, return error state
        if not self._llm_connected:
            state.error = "LLM service unavailable"
            state.response = "I'm currently unable to process your request as my language model service is unavailable. Please try again later or contact support."
            return state

        # Execute workflow stages 1-5
        state = await self._initialize_context(state)

        # Skip parsing if there are no available tools
        if not self._has_tools:
            # If no tools available, set up for simple response
            logger.info("No MCP clients available, will generate simple response")
            state.execution_context["simple_mode"] = True
            return state

        if not state.error:
            state = await self._parse_query(state)

            # Execute actions if any were planned
            if state.planned_actions and not state.error:
                while state.current_action_index < len(state.planned_actions):
                    state = await self._execute_mcp_actions(state)
                    state = await self._evaluate_results(state)

                    if (
                        state.requires_replanning
                        and state.replan_count >= state.max_replan_attempts
                    ):
                        logger.error(
                            f"Maximum replanning attempts ({state.max_replan_attempts}) reached."
                        )
                        break

                    if state.requires_replanning:
                        state = await self._replan_actions(state)
                    elif state.error:
                        break

        return state

    @observe()
    async def _generate_response_streaming(
        self, state: AgentState
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response using LLM streaming capabilities"""
        # Update observation metadata
        update_observation_metadata(
            name="Generate Streaming Response",
            metadata={
                "simple_mode": state.execution_context.get("simple_mode", False),
                "has_mcp_results": bool(state.mcp_results),
            },
        )

        logger.info("Starting streaming response generation")

        try:
            # Handle simple mode (no tools available)
            if state.execution_context.get("simple_mode", False):
                prompt = f"""
                Answer the user's question as best you can without any external tools.
                
                User Query: {state.query}
                
                Conversation History:
                {self._format_conversation_history(self.current_history)}
                
                Note: I currently don't have access to any external tools, so I can only answer based on my general knowledge.
                """
            else:
                # Build the same prompt as _generate_response()
                formatted_results = self._format_results(state.mcp_results)
                conversation_context = self._format_conversation_history(self.current_history)
                status_info = self._get_connection_status_info()

                prompt = f"""
                Generate a comprehensive response to the user's query based on the conversation history and the results of MCP actions.

                User Query: {state.query}

                Conversation History:
                {conversation_context}

                MCP Interaction Results:
                {formatted_results}

                Agent Thought Process:
                {self._format_thoughts(state.thoughts)}

                System Status:
                {status_info}

                Instructions:
                1. Synthesize information from successful MCP actions.
                2. Explain any errors encountered and whether they were resolved.
                3. If database queries were run, present the data clearly.
                4. Address all parts of the original user query.
                5. Use a helpful and informative tone. Do not just dump raw JSON or technical details.
                6. Provide a clear conclusion or answer to the user's query.
                7. If the system status indicates missing connections, briefly mention this at the end of your response.
                """

            # Stream from LLM
            full_response = ""
            async for chunk in self.llm.generate_streaming(prompt):
                full_response += chunk
                yield {
                    "type": "response_chunk",
                    "content": chunk,
                    "partial_response": full_response,
                }

            # Append status information if necessary
            if not self._db_connected:
                status_note = "\n\n*Note: I'm currently running in stateless mode without access to conversation history storage.*"
                full_response += status_note
                yield {
                    "type": "response_chunk",
                    "content": status_note,
                    "partial_response": full_response,
                }

            # Save final response to database
            assistant_message = Message(role="assistant", content=full_response)
            await self._save_message(state.session_id, assistant_message)

            # Signal completion
            yield {"type": "response_complete", "full_response": full_response}

        except Exception as e:
            logger.error(f"Error during streaming response generation: {e}", exc_info=True)
            error_response = (
                "I encountered an error while generating the response. Please try again."
            )

            # Try to save error to history
            try:
                error_message = Message(role="assistant", content=f"Error: {str(e)}")
                await self._save_message(state.session_id, error_message)
            except:
                pass  # Don't let history saving errors prevent error reporting

            yield {"type": "error", "error": str(e), "response": error_response}

    # --- Core Workflow Methods (unchanged from original) ---

    @observe()
    async def _initialize_context(self, state: AgentState) -> AgentState:
        """Initialize context by discovering available tools and resources."""
        # Update observation metadata
        update_observation_metadata(
            name="Initialize Context",
            metadata={"session_id": state.session_id},
        )
        logger.info("Initializing agent context...")

        # Update status flags
        self._db_connected = self.db.is_connected()
        self._llm_connected = self.llm.is_connected()

        # If no clients are available, don't attempt to discover tools
        clients = self.transport_manager.get_all_clients()
        if not clients:
            logger.warning("No MCP clients available, skipping tool discovery")
            state.execution_context["available_tools"] = []
            state.execution_context["available_resources"] = []
            state.execution_context["server_ids"] = []
            self._has_tools = False
            return state

        try:
            # Initialize containers in execution_context
            state.execution_context["available_tools"] = []
            state.execution_context["available_resources"] = []
            state.execution_context["server_ids"] = list(
                self.transport_manager.get_all_clients().keys()
            )
            self._has_tools = True

            # Discover tools and resources from each client concurrently
            tasks = {}
            for server_id, client in self.transport_manager.get_all_clients().items():
                tasks[server_id] = asyncio.gather(
                    client.list_tools(),
                    client.list_resources(),
                    return_exceptions=True,  # Don't let one failure stop others
                )

            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            for i, server_id in enumerate(tasks.keys()):
                server_result = results[i]
                if isinstance(server_result, Exception):
                    logger.error(
                        f"Error initializing context from {server_id}: {server_result}",
                        exc_info=self.debug,
                    )
                    continue  # Skip this server

                tools_result, resources_result = server_result

                # Process tools
                if isinstance(tools_result, Exception):
                    logger.error(
                        f"Error listing tools from {server_id}: {tools_result}",
                        exc_info=self.debug,
                    )
                elif isinstance(tools_result, list):
                    tools_list = []
                    for tool in tools_result:
                        # Convert Tool object to dict for planning prompt
                        tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else vars(tool)
                        tool_dict["server_id"] = server_id
                        tools_list.append(tool_dict)
                    state.execution_context["available_tools"].extend(tools_list)
                    logger.debug(f"Found {len(tools_list)} tools from {server_id}")

                # Process resources
                if isinstance(resources_result, Exception):
                    logger.error(
                        f"Error listing resources from {server_id}: {resources_result}",
                        exc_info=self.debug,
                    )
                elif isinstance(resources_result, list):
                    resource_list = []
                    for resource in resources_result:
                        # Convert Resource object to dict
                        res_dict = (
                            resource.model_dump()
                            if hasattr(resource, "model_dump")
                            else vars(resource)
                        )
                        res_dict["server_id"] = server_id
                        resource_list.append(res_dict)
                    state.execution_context["available_resources"].extend(resource_list)
                    logger.debug(f"Found {len(resource_list)} resources from {server_id}")

            return state

        except Exception as e:
            state.error = f"Failed to initialize context: {str(e)}"
            logger.error(f"Error in initialize_context: {e}", exc_info=True)
            return state

    @observe()
    async def _parse_query(self, state: AgentState) -> AgentState:
        """Use LLM to determine which MCPs and actions are needed."""
        # Update observation metadata
        update_observation_metadata(
            name="Parse Query",
            input=state.query,
            metadata={
                "available_tools": len(state.execution_context.get("available_tools", [])),
                "available_servers": len(state.execution_context.get("server_ids", [])),
            },
        )
        logger.info("Parsing user query: %s", state.query)

        try:
            # Format available tools/resources for LLM
            mcp_info = "Available Tool Servers:\n"
            server_details = []
            for server_id in state.execution_context.get("server_ids", []):
                server_info = f"Server ID: '{server_id}'\n"
                tools = [
                    tool
                    for tool in state.execution_context.get("available_tools", [])
                    if tool.get("server_id") == server_id
                ]
                resources = [
                    res
                    for res in state.execution_context.get("available_resources", [])
                    if res.get("server_id") == server_id
                ]

                if tools:
                    server_info += "  Tools:\n"
                    for tool in tools:
                        schema_str = json.dumps(tool.get("inputSchema", {}))
                        server_info += f"    - Name: {tool.get('name', 'N/A')}\n      Description: {tool.get('description', 'N/A')}\n      Input Schema: {schema_str}\n"
                        # Include annotations if available
                        annotations = tool.get("annotations")
                        if annotations:
                            server_info += f"      Annotations: {json.dumps(annotations)}\n"
                else:
                    server_info += "  No tools available.\n"

                if resources:
                    server_info += "  Resources (Schemas/Data):\n"
                    for res in resources:
                        res_str = json.dumps(
                            {k: v for k, v in res.items() if k != "server_id"}, default=str
                        )
                        server_info += f"    - {res_str}\n"
                else:
                    server_info += "  No resources available.\n"

                server_details.append(server_info)

            mcp_info += "\n".join(server_details)

            # Format conversation history
            conversation_context = self._format_conversation_history(self.current_history)

            # Build the prompt for the LLM
            prompt = f"""
            Analyze the user's query and create a plan using the available MCP tool servers.

            MCP Methods Overview:
            - `listResources`: Lists available data resources (e.g., schemas). Returns a list of resources (URI, name, mimeType).
            - `readResource`: Reads content of a resource URI. Params: {{uri: string}}. Returns resource content (text or blob).
            - `listTools`: Lists available tools. Returns a list of tools (name, description, inputSchema).
            - `callTool`: Executes a tool. Params: {{name: string, arguments: dict}}. Returns tool output (text, image, etc.) or an error flag.

            Available Servers, Tools, and Resources:
            {mcp_info}

            Conversation History:
            {conversation_context}

            User Query: {state.query}

            Think step-by-step to create a plan. Your plan should be a sequence of actions.
            1. Identify the goal of the user query.
            2. Determine which server(s) and tool(s)/resource(s) are needed.
            3. If database access is needed:
               - Use the appropriate tool via `callTool` for executing database operations.
               - Ensure SQL uses schema qualification (e.g., `schema_name.table_name`).
            4. For other tools, use `callTool` with the correct server_id, tool name, and arguments based on the inputSchema.
            5. Plan all necessary steps sequentially if one depends on another.

            Output Format (JSON):
            {{
                "thought": "Your reasoning process.",
                "plan": "High-level plan description.",
                "actions": [
                    {{
                        "server_id": "...", // REQUIRED: server ID from the list
                        "mcp_method": "...", // e.g., "readResource", "callTool"
                        "params": {{...}},    // Parameters for the method
                        "reason": "Why this action is needed.",
                        "expected_outcome": "What this action should provide."
                    }}
                    // ... more actions if needed
                ]
            }}
            If no actions are needed, use "actions": [].
            If more info needed, add "needs_more_info": true, "follow_up_question": "...".
            Do **NOT** put comments in the JSON.
            """

            # Call the LLM
            logger.debug("Sending query to LLM for planning...")
            response = await self.llm.generate(prompt)
            parsed_response = self._parse_json_response(response)
            logger.debug("Received and parsed LLM planning response")

            # Extract planning information
            thought = parsed_response.get("thought", "No thought process provided.")
            plan = parsed_response.get("plan", "No plan provided.")
            state.thoughts.append(thought)
            if plan != "No plan provided.":
                state.thoughts.append(f"Plan: {plan}")
            logger.info(f"Agent Thought: {thought}")
            logger.info(f"Plan: {plan}")

            # Set planned actions
            state.planned_actions = parsed_response.get("actions", [])
            state.current_action_index = 0
            logger.info(f"Planned {len(state.planned_actions)} actions.")

            # Log the planned actions for debugging
            for i, action in enumerate(state.planned_actions):
                logger.debug(
                    f"Action {i + 1}: {action.get('server_id')}.{action.get('mcp_method')}({action.get('params', {})}) - Reason: {action.get('reason')}"
                )

            # Check if more info is needed
            if parsed_response.get("needs_more_info", False):
                state.needs_more_info = True
                state.follow_up_question = parsed_response.get("follow_up_question")
                state.thoughts.append(
                    f"Needs more info: {state.follow_up_question or 'General clarification'}"
                )

            return state

        except Exception as e:
            state.error = f"Failed to parse query: {str(e)}"
            logger.error(f"Error in parse_query: {e}", exc_info=True)
            return state

    @observe()
    async def _execute_mcp_actions(self, state: AgentState) -> AgentState:
        """Execute the next planned action using the appropriate MCP client."""
        # Update observation metadata
        current_action = (
            state.planned_actions[state.current_action_index]
            if state.current_action_index < len(state.planned_actions)
            else None
        )
        update_observation_metadata(
            name="Execute MCP Action",
            metadata={
                "action_index": state.current_action_index,
                "total_actions": len(state.planned_actions),
                "server_id": current_action.get("server_id") if current_action else None,
                "mcp_method": current_action.get("mcp_method") if current_action else None,
            },
        )
        if state.current_action_index >= len(state.planned_actions):
            logger.debug("No more actions to execute in the current plan.")
            return state

        action_spec = state.planned_actions[state.current_action_index]
        server_id = action_spec.get("server_id")
        mcp_method = action_spec.get("mcp_method")
        params = action_spec.get("params", {})
        action_key = f"action_{state.current_action_index}_{server_id}_{mcp_method}"

        # Validate action parameters
        if not server_id or not mcp_method:
            error_msg = "Invalid action: Missing server_id or mcp_method"
            state.mcp_results[action_key] = {"error": error_msg}
            state.current_action_index += 1
            state.error = error_msg
            return state

        # Get the MCP client
        if not self.transport_manager.get_client(server_id):
            error_msg = f"MCP client for '{server_id}' not available"
            state.mcp_results[action_key] = {"error": error_msg}
            state.current_action_index += 1
            state.error = error_msg
            return state

        client = self.transport_manager.get_client(server_id)
        action_label = f"{server_id}.{mcp_method}({json.dumps(params, default=str)})"
        logger.info(
            f"Executing action {state.current_action_index + 1}/{len(state.planned_actions)}: {action_label}"
        )
        state.thoughts.append(f"Executing: {action_label}")

        try:
            result = None

            if mcp_method == "listResources":
                result = await client.list_resources()

            elif mcp_method == "readResource":
                if "uri" not in params:
                    raise ValueError("Missing 'uri' parameter for readResource")
                result = await client.read_resource(params["uri"])

            elif mcp_method == "listTools":
                result = await client.list_tools()

            elif mcp_method == "callTool":
                if "name" not in params:
                    raise ValueError("Missing 'name' parameter for callTool")
                result = await client.call_tool(params["name"], params.get("arguments", {}))

            else:
                raise ValueError(f"Unsupported MCP method: {mcp_method}")

            logger.debug(f"Action successful: {action_label}")
            state.thoughts.append(f"Action Result: Success for {action_label}")
            state.mcp_results[action_key] = result

        except Exception as e:
            error_msg = str(e)
            state.mcp_results[action_key] = {"error": error_msg}
            state.thoughts.append(f"Action Result: Error: {error_msg}")

            # Special handling for common errors
            if isinstance(e, ClientError):
                if "does not exist" in error_msg.lower():
                    state.execution_context["schema_qualification_needed"] = True
                    state.thoughts.append(
                        "Error suggests table/resource wasn't found. Ensure proper names/paths."
                    )

            logger.error(f"Error executing {action_label}: {error_msg}", exc_info=self.debug)
            state.requires_replanning = True
            state.execution_context["last_error"] = error_msg

        state.current_action_index += 1
        return state

    @observe()
    async def _evaluate_results(self, state: AgentState) -> AgentState:
        """Evaluate action results to determine if replanning is needed."""
        # Update observation metadata
        update_observation_metadata(
            name="Evaluate Results",
            metadata={
                "has_error": bool(state.error),
                "action_index": state.current_action_index - 1,
            },
        )
        if not state.mcp_results or state.error:  # Skip if no results or critical error occurred
            return state

        last_action_index = state.current_action_index - 1
        if last_action_index < 0:
            return state  # No action executed yet

        # Find the key for the last action's result
        last_result_key = None
        action_spec = state.planned_actions[last_action_index]
        server_id = action_spec.get("server_id", "")
        mcp_method = action_spec.get("mcp_method", "")
        possible_key = f"action_{last_action_index}_{server_id}_{mcp_method}"

        if possible_key in state.mcp_results:
            last_result_key = possible_key

        if not last_result_key:
            logger.warning(f"Could not find result key for last action index {last_action_index}")
            return state  # Cannot evaluate if result key is missing

        last_result = state.mcp_results.get(last_result_key)

        # Check if the result indicates an error
        if isinstance(last_result, dict) and "error" in last_result:
            logger.info(
                f"Action {last_action_index + 1} ({action_spec.get('mcp_method')}) failed, initiating replanning."
            )
            state.requires_replanning = True
            state.execution_context["last_error"] = last_result["error"]

            # Check for specific SQL errors indicating schema issues
            if (
                server_id == "postgres"
                and "relation" in str(last_result["error"]).lower()
                and "does not exist" in str(last_result["error"]).lower()
            ):
                state.execution_context["schema_qualification_needed"] = True
                state.thoughts.append(
                    "Error suggests table wasn't found. Ensure schema qualification."
                )

        # No error found in the last step
        else:
            state.requires_replanning = False
            state.execution_context.pop("last_error", None)  # Clear last error
            state.execution_context.pop("schema_qualification_needed", None)

        return state

    @observe()
    async def _replan_actions(self, state: AgentState) -> AgentState:
        """Replan actions based on execution errors."""
        # Update observation metadata
        update_observation_metadata(
            name="Replan Actions",
            metadata={
                "replan_count": state.replan_count,
                "max_attempts": state.max_replan_attempts,
                "last_error": state.execution_context.get("last_error"),
            },
            level="WARNING" if state.replan_count > 1 else "DEFAULT",
        )
        logger.info("Replanning actions due to execution issues...")

        if state.replan_count >= state.max_replan_attempts:
            logger.error(
                f"Reached maximum replanning attempts ({state.max_replan_attempts}). Stopping execution."
            )
            state.error = f"Failed after {state.max_replan_attempts} replanning attempts. Last error: {state.execution_context.get('last_error', 'Unknown error')}"
            state.requires_replanning = False
            state.planned_actions = state.planned_actions[: state.current_action_index]
            return state

        last_error = state.execution_context.get("last_error", "Unknown error")
        schema_qualification_needed = state.execution_context.get(
            "schema_qualification_needed", False
        )

        # Format available tools/resources again for the replanning prompt
        mcp_info = "Available Servers, Tools, and Resources:\n"
        server_details = []
        for server_id in state.execution_context.get("server_ids", []):
            server_info = f"Server ID: '{server_id}'\n"
            tools = [
                t
                for t in state.execution_context.get("available_tools", [])
                if t.get("server_id") == server_id
            ]
            resources = [
                r
                for r in state.execution_context.get("available_resources", [])
                if r.get("server_id") == server_id
            ]

            if tools:
                server_info += "  Tools:\n"
                for tool in tools:
                    schema_str = json.dumps(tool.get("inputSchema", {}))
                    server_info += f"    - Name: {tool.get('name', 'N/A')}\n      Description: {tool.get('description', 'N/A')}\n      Input Schema: {schema_str}\n"

            if resources:
                server_info += "  Resources (Schemas/Data):\n"
                for res in resources:
                    res_str = json.dumps(
                        {k: v for k, v in res.items() if k != "server_id"}, default=str
                    )
                    server_info += f"    - {res_str}\n"

            server_details.append(server_info)

        mcp_info += "\n".join(server_details)

        # Build the replanning prompt
        prompt = f"""
        The previous plan failed. Analyze the error and the context to create a revised plan.

        User Query: {state.query}
        Original Plan Actions Executed:
        {self._format_actions_executed(state)}

        Last Action Error: {last_error}
        {"Hint: The error suggests a table/relation wasn't found. Ensure SQL uses schema qualification (schema_name.table_name)." if schema_qualification_needed else ""}

        Available Servers, Tools, and Resources:
        {mcp_info}

        Conversation History:
        {self._format_conversation_history(self.current_history)}

        Successful Results from Previous Actions:
        {self._format_successful_results(state)}

        Think step-by-step to revise the plan:
        1. Understand why the last action failed based on the error.
        2. If it was a database query error: Did it lack schema qualification? Was the table/column name wrong? Refer to available resources.
        3. If it was another tool error: Can different arguments fix it? Is there an alternative tool?
        4. Create a NEW sequence of actions to achieve the original goal, correcting the mistake. Start the new plan from the point of failure.

        Output Format (JSON):
        {{
            "thought": "Your reasoning about the failure and the revised approach.",
            "revised_plan": "Description of the new plan.",
            "actions": [
                {{
                    "server_id": "...", // REQUIRED
                    "mcp_method": "...",
                    "params": {{...}},
                    "reason": "Why this new action will fix the issue or progress.",
                    "expected_outcome": "What this action should provide now."
                }}
                // ... potentially more actions
            ]
        }}
        If you cannot recover or determine a fix, respond with "actions": [].
        """

        # Call the LLM for replanning
        logger.debug("Sending replanning prompt to LLM...")
        response = await self.llm.generate(prompt)
        parsed_response = self._parse_json_response(response)

        # Extract replanning information
        thought = parsed_response.get("thought", "Revising plan due to execution error.")
        revised_plan = parsed_response.get("revised_plan", "Trying alternative approach.")
        state.thoughts.append(f"Replanning Thought: {thought}")
        state.thoughts.append(f"Revised Plan: {revised_plan}")

        # Set new actions
        new_actions = parsed_response.get("actions", [])
        if new_actions:
            # Replace the rest of the plan with the new actions
            state.planned_actions = (
                state.planned_actions[: state.current_action_index - 1] + new_actions
            )
            state.current_action_index = max(
                0, state.current_action_index - 1
            )  # Reset index to retry/continue

            # Increment replan counter
            state.replan_count += 1

            logger.info(
                f"Replanned starting from action {state.current_action_index + 1} with {len(new_actions)} new action(s)."
            )

            # Log the new actions for debugging
            for i, action in enumerate(new_actions):
                logger.debug(
                    f"New Action {i + 1}: {action.get('server_id')}.{action.get('mcp_method')}({action.get('params', {})})"
                )

        else:
            logger.warning(
                "Replanning did not produce new actions. Will proceed to generate response based on current results."
            )
            state.planned_actions = state.planned_actions[
                : state.current_action_index
            ]  # Stop executing plan

        # Reset replanning flags
        state.requires_replanning = False
        state.execution_context.pop("last_error", None)
        state.execution_context.pop("schema_qualification_needed", None)

        return state

    @observe()
    async def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response based on all gathered information."""
        # Update observation metadata
        update_observation_metadata(
            name="Generate Response",
            metadata={
                "has_mcp_results": bool(state.mcp_results),
                "thought_count": len(state.thoughts),
                "db_connected": self._db_connected,
            },
        )
        logger.info("Generating final response...")

        # Format results and history for the LLM
        formatted_results = self._format_results(state.mcp_results)
        conversation_context = self._format_conversation_history(self.current_history)

        # Get status information
        status_info = self._get_connection_status_info()

        # Build the response generation prompt
        prompt = f"""
        Generate a comprehensive response to the user's query based on the conversation history and the results of MCP actions.

        User Query: {state.query}

        Conversation History:
        {conversation_context}

        MCP Interaction Results:
        {formatted_results}

        Agent Thought Process:
        {self._format_thoughts(state.thoughts)}

        System Status:
        {status_info}

        Instructions:
        1. Synthesize information from successful MCP actions.
        2. Explain any errors encountered and whether they were resolved.
        3. If database queries were run, present the data clearly.
        4. Address all parts of the original user query.
        5. Use a helpful and informative tone. Do not just dump raw JSON or technical details.
        6. Provide a clear conclusion or answer to the user's query.
        7. If the system status indicates missing connections, briefly mention this at the end of your response.
        """

        # Call the LLM for response generation
        logger.debug("Sending response generation prompt to LLM.")
        response = await self.llm.generate(prompt)

        # Append status information if necessary
        if not self._db_connected:
            response += "\n\n*Note: I'm currently running in stateless mode without access to conversation history storage.*"

        state.response = response
        logger.info(f"Final response generated (length: {len(response)} chars).")

        # Save the assistant message to the execution context for later database saving
        state.execution_context["assistant_message_to_save"] = Message(
            role="assistant", content=response
        )

        return state

    # --- Helper Methods ---
    def _get_connection_status_info(self) -> str:
        """Get connection status info for the prompt."""
        status_lines = []

        if not self._db_connected:
            status_lines.append(
                "Database: Not connected. Running in stateless mode without conversation history storage."
            )
        else:
            status_lines.append("Database: Connected. Conversation history is being saved.")

        if not self._llm_connected:
            status_lines.append("LLM: Not connected. Using fallback response generation.")
        else:
            status_lines.append("LLM: Connected. Using full language model capabilities.")

        if not self._has_tools:
            status_lines.append(
                "Tools: No tools available. Running in basic Q&A mode without tool access."
            )
        else:
            tools_count = len(self.transport_manager.get_all_clients())
            status_lines.append(f"Tools: Connected to {tools_count} tool server(s).")

        return "\n".join(status_lines)

    def _format_conversation_history(self, history: List[Message]) -> str:
        """Format conversation history for prompts."""
        if not history:
            return "No history."
        return "\n".join(f"{msg.role.upper()}: {msg.content}" for msg in history)

    def _format_thoughts(self, thoughts: List[str]) -> str:
        """Format thought process for the prompt."""
        if not thoughts:
            return "No thoughts recorded."
        return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(thoughts))

    def _format_actions_executed(self, state: AgentState) -> str:
        """Format the actions executed so far for replanning prompts."""
        if state.current_action_index == 0:
            return "No actions executed yet."

        formatted = ""
        for i in range(state.current_action_index):
            if i < len(state.planned_actions):
                action = state.planned_actions[i]
                server_id = action.get("server_id", "N/A")
                mcp_method = action.get("mcp_method", "N/A")
                params_str = json.dumps(action.get("params", {}), default=str)
                formatted += (
                    f"{i + 1}. Server: {server_id}, Method: {mcp_method}, Params: {params_str}\n"
                )

                # Add result/error summary
                action_key = f"action_{i}_{server_id}_{mcp_method}"
                result_data = state.mcp_results.get(action_key)

                if isinstance(result_data, dict) and "error" in result_data:
                    formatted += f"   Result: Error - {result_data['error']}\n"
                elif result_data:
                    formatted += (
                        "   Result: Success (details omitted)\n"  # Avoid dumping large raw results
                    )
                else:
                    formatted += "   Result: Not found\n"

        return formatted.strip()

    def _format_successful_results(self, state: AgentState) -> str:
        """Format results from successful actions for replanning prompts."""
        successful_data = {}

        for i in range(state.current_action_index):
            action = state.planned_actions[i]
            server_id = action.get("server_id", "N/A")
            mcp_method = action.get("mcp_method", "N/A")
            action_key = f"action_{i}_{server_id}_{mcp_method}"
            result = state.mcp_results.get(action_key)

            if result and not (isinstance(result, dict) and "error" in result):
                # Try to format the result safely
                try:
                    successful_data[action_key] = self._safe_serialize(result)
                except Exception:
                    successful_data[action_key] = (
                        f"<{type(result).__name__} object - Non-JSON Serializable>"
                    )

        return (
            json.dumps(successful_data, indent=2, default=str)
            if successful_data
            else "No successful results yet."
        )

    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize any object to JSON-compatible format."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "__dict__"):
            return vars(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._safe_serialize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._safe_serialize(v) for k, v in obj.items()}
        else:
            return str(obj)

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format MCP results for clarity in prompts."""
        formatted = ""

        for key, result in results.items():
            formatted += f"--- Result for {key} ---\n"

            if isinstance(result, dict) and "error" in result:
                formatted += f"Type: Error\nMessage: {result['error']}\n"
            elif hasattr(result, "__str__"):
                # For special MCP types, summarize them
                result_str = str(result)
                formatted += f"{result_str}\n"
            else:
                # For other types, do a simple representation
                formatted += f"Type: {type(result).__name__}\n"
                formatted += f"Content: {self._safe_serialize(result)}...\n"

            formatted += "---\n"

        return formatted if formatted else "No results available."

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with improved reliability."""
        try:
            # Try to find JSON in the response
            match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                match = re.search(r"(\{[\s\S]*\})", response)
                json_str = match.group(1) if match else response

            # Extract JSON block from the first { to the last }
            first_brace = json_str.find("{")
            last_brace = json_str.rfind("}")
            if first_brace != -1 and last_brace != -1:
                json_str = json_str[first_brace : last_brace + 1]
            else:
                raise json.JSONDecodeError("No valid JSON object found", json_str, 0)

            return json.loads(json_str)

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response}", exc_info=self.debug)
            return {"thought": f"Error parsing LLM response: {e}", "actions": []}

        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}", exc_info=self.debug)
            return {"thought": f"Unexpected error: {e}", "actions": []}

    # --- Database helpers ---

    async def _load_history(self, session_id: str, conversation_id: str = None) -> List[Message]:
        """
        Load conversation history from the database.

        Args:
            session_id: Session identifier
            conversation_id: Optional conversation identifier (takes precedence if provided)

        Returns:
            List of messages in the conversation
        """
        if not self._db_connected:
            logger.warning("Database not connected. Running in stateless mode.")
            return []

        try:
            history = await self.db.load_history(session_id, conversation_id)
            self.current_history = history
            return history
        except Exception as e:
            identifier = conversation_id if conversation_id else session_id
            logger.error(f"Failed to load history for {identifier}: {e}")
            return []  # Return empty history on error

    async def _save_message(self, session_id: str, message: Message) -> bool:
        """Save a message to the database and update in-memory history."""
        try:
            if self._db_connected:
                success = await self.db.save_message(session_id, message)
            else:
                logger.warning("Database not connected. Message not saved to database.")
                success = False

            # Always update in-memory history regardless of DB connection
            if message not in self.current_history:
                self.current_history.append(message)
            return success
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            # Still update in-memory history
            if message not in self.current_history:
                self.current_history.append(message)
            return False

    async def clear_history(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        try:
            success = await self.db.clear_history(session_id)
            self.current_history = []
            return success
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False

    # --- Main method ---

    @observe()
    async def run(self, query: str, session_id: Optional[str] = None) -> AgentState:
        """Execute the complete agent workflow."""
        # Update observation metadata
        update_observation_metadata(
            name="Agent Run",
            input={"query": query, "session_id": session_id},
            metadata={"streaming": False},
        )
        logger.info(f"Starting agent run for query: '{query[:50]}...'")

        # Update connection status
        self._db_connected = self.db.is_connected()
        self._llm_connected = self.llm.is_connected()
        self._has_tools = len(self.transport_manager.get_all_clients()) > 0

        # Generate or use session ID
        definite_session_id = session_id or str(uuid.uuid4())
        logger.info(f"Session ID: {definite_session_id}")

        # Check if this is a conversation ID (starts with "conv_")
        conversation_id = None
        if definite_session_id.startswith("conv_"):
            conversation_id = definite_session_id
            # Use conversation_id as session_id for compatibility
            logger.info(f"Using conversation mode with ID: {conversation_id}")

        # Load history if database is connected
        await self._load_history(definite_session_id, conversation_id)

        # Handle management commands
        if query.strip().lower() == "clear history":
            if self._db_connected:
                await self.db.clear_history(definite_session_id)
                final_state = AgentState(
                    query=query, session_id=definite_session_id, response="History cleared."
                )
            else:
                final_state = AgentState(
                    query=query,
                    session_id=definite_session_id,
                    response="I'm running in stateless mode without access to database. No history to clear.",
                )
            return final_state

        if query.strip().lower() == "show history":
            history_text = "History:\n" + (
                "\n".join(map(str, self.current_history)) if self.current_history else "(empty)"
            )
            if not self._db_connected:
                history_text += "\n\n*Note: I'm running in stateless mode without database access. This is the in-memory history for this session only.*"

            final_state = AgentState(
                query=query, session_id=definite_session_id, response=history_text
            )
            return final_state

        # Save user message (to memory even if DB not connected)
        user_message = Message(role="user", content=query)
        await self._save_message(definite_session_id, user_message)

        # Initialize state
        state = AgentState(query=query, session_id=definite_session_id)

        try:
            # If LLM is not connected, return a simple message
            if not self._llm_connected:
                state.response = "I'm currently unable to process your request as my language model service is unavailable. Please try again later or contact support."
                return state

            # Execute workflow
            state = await self._initialize_context(state)

            # Skip parsing if there are no available tools
            if not self._has_tools:
                # If no tools available, just generate a simple response
                logger.info("No MCP clients available, skipping parsing and execution")
                prompt = f"""
                Answer the user's question as best you can without any external tools.
                
                User Query: {query}
                
                Conversation History:
                {self._format_conversation_history(self.current_history)}
                
                Note: I currently don't have access to any external tools, so I can only answer based on my general knowledge.
                """

                state.response = await self.llm.generate(prompt)

                if not self._db_connected:
                    state.response += "\n\n*Note: I'm currently running in stateless mode without access to conversation history storage.*"

                # Save assistant message
                assistant_message = Message(role="assistant", content=state.response)
                await self._save_message(definite_session_id, assistant_message)

                return state

            if not state.error:
                state = await self._parse_query(state)

                # Execute actions if any were planned
                if state.planned_actions and not state.error:
                    while state.current_action_index < len(state.planned_actions):
                        state = await self._execute_mcp_actions(state)
                        state = await self._evaluate_results(state)

                        if (
                            state.requires_replanning
                            and state.replan_count >= state.max_replan_attempts
                        ):
                            logger.error(
                                f"Maximum replanning attempts ({state.max_replan_attempts}) reached."
                            )
                            break

                        if state.requires_replanning:
                            state = await self._replan_actions(state)
                        elif state.error:
                            break

            # Generate final response
            state = await self._generate_response(state)

            # Save assistant message
            if state.execution_context.get("assistant_message_to_save"):
                await self._save_message(
                    definite_session_id, state.execution_context["assistant_message_to_save"]
                )

        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            state.error = f"Agent execution failed: {e}"
            state.response = "I encountered an error processing your request. Please try again or contact support if the issue persists."

            if not self._db_connected:
                state.response += "\n\n*Note: I'm currently running in stateless mode without access to conversation history storage.*"

            # Try to save error message to history
            error_message = Message(
                role="assistant", content=f"Error: {state.error}\n{state.response or ''}"
            )
            await self._save_message(definite_session_id, error_message)

        return state
