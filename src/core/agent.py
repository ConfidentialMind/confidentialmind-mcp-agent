# src/core/agent.py
import asyncio
import json
import logging
import re
import sys
import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

import mcp.types as mcp_types

# Import SDK ClientSession and types
from mcp import ClientSession, McpError

# Use CM connectors for LLM and MCP session management
from src.connectors.cm_llm_connector import CMLLMConnector
from src.connectors.cm_mcp_connector import CMMCPManager

# Database imports remain the same
from src.core.agent_db_connection import AgentDatabase  # Assuming this is used for history

# from src.core.agent_db_migration import AgentMigration # Migration handled elsewhere

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent_orchestrator")


class Message(BaseModel):
    """A message in the conversation history"""

    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role.upper()}: {self.content}"


class AgentState(BaseModel):
    """Agent workflow state - updated for SDK types"""

    query: str = ""
    session_id: Optional[str] = None
    # Store raw results keyed by a unique identifier for the action step
    mcp_results: Dict[str, Any] = Field(default_factory=dict)  # Store raw SDK results or McpError
    thoughts: List[str] = Field(default_factory=list)
    response: Optional[str] = None
    error: Optional[str] = None
    # Removed mcp_options, use execution_context instead
    planned_actions: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Plan still uses dict structure
    current_action_index: int = 0
    needs_more_info: bool = False
    follow_up_question: Optional[str] = None
    # available_schemas now derived from context
    execution_context: Dict[str, Any] = Field(
        default_factory=dict
    )  # Store tools, resources, errors etc.
    requires_replanning: bool = False


class Agent:
    """LangGraph-based agent using MCP SDK ClientSessions"""

    def __init__(
        self,
        agent_db: AgentDatabase,
        llm_connector: CMLLMConnector,
        mcp_manager: CMMCPManager,
        debug: bool = False,
    ):
        """
        Initialize the agent with SDK-based connections.

        Args:
            agent_db: An initialized and connected AgentDatabase instance for history.
            llm_connector: Initialized SDK-based LLM connector.
            mcp_manager: Initialized SDK-based MCP manager to get sessions.
            debug: Enable debug logging.
        """
        self.llm = llm_connector
        self.mcp_manager = mcp_manager
        self.mcp_sessions: Dict[str, ClientSession] = {}  # Will be populated in run/initialize
        self.debug = debug
        self.db = agent_db
        self.current_history: List[
            Message
        ] = []  # Temporary storage for current conversation history

        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        self.graph = self._build_graph().compile()
        logger.info("Agent workflow graph compiled")

    # --- History Management (using AgentDatabase, assumed similar methods) ---
    async def _load_history(self, session_id: str) -> List[Message]:
        """Load conversation history from the database for a session."""
        try:
            await self.db.ensure_connected()
            results = await self.db.execute_query(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE session_id = $1
                ORDER BY message_order
                """,
                session_id,
            )
            messages = [Message(role=row["role"], content=row["content"]) for row in results]
            logger.debug(f"Loaded {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Error loading history for session {session_id}: {e}")
            return []

    async def _save_message(self, session_id: str, message: Message) -> bool:
        """Save a message to the database for a session."""
        try:
            await self.db.ensure_connected()
            max_order = await self.db.execute_query(
                "SELECT MAX(message_order) FROM conversation_messages WHERE session_id = $1",
                session_id,
                fetch_type="val",
            )
            message_order = 0 if max_order is None else max_order + 1
            await self.db.execute_query(
                """
                INSERT INTO conversation_messages
                (session_id, message_order, role, content, timestamp)
                VALUES ($1, $2, $3, $4, NOW())
                """,
                session_id,
                message_order,
                message.role,
                message.content,
                fetch_type="none",
            )
            self.current_history.append(message)  # Also update in-memory history
            logger.debug(f"Saved message for session {session_id} order {message_order}")
            return True
        except Exception as e:
            if not self.db.is_connected():
                logger.warning(
                    f"DB not connected saving msg for {session_id}: {self.db.last_error()}"
                )
            else:
                logger.error(f"Error saving message for session {session_id}: {e}")
            # Still add to in-memory history if DB fails
            if message not in self.current_history:
                self.current_history.append(message)
            return False

    async def clear_history(self, session_id: str) -> bool:
        """Clear the conversation history for a session in the database."""
        try:
            await self.db.ensure_connected()
            await self.db.execute_query(
                "DELETE FROM conversation_messages WHERE session_id = $1",
                session_id,
                fetch_type="none",
            )
            logger.info(f"Cleared conversation history for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing history for session {session_id}: {e}")
            return False

    # --- End History Management ---

    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph (structure remains similar)."""
        graph = StateGraph(AgentState)
        graph.add_node("initialize_context", self._initialize_context)
        graph.add_node("parse_query", self._parse_query)
        graph.add_node("execute_mcp_actions", self._execute_mcp_actions)
        graph.add_node("evaluate_results", self._evaluate_results)
        graph.add_node("replan_actions", self._replan_actions)
        graph.add_node("generate_response", self._generate_response)
        # Removed plan_follow_up node as it wasn't fully implemented/used consistently
        # If needed, it can be re-added with updated logic.

        graph.add_edge("initialize_context", "parse_query")
        graph.add_conditional_edges(
            "parse_query",
            self._determine_next_step_after_parsing,
            {"execute": "execute_mcp_actions", "respond": "generate_response"},
        )
        graph.add_edge("execute_mcp_actions", "evaluate_results")
        graph.add_conditional_edges(
            "evaluate_results",
            self._determine_next_step_after_evaluation,
            {
                "more_actions": "execute_mcp_actions",
                "replan": "replan_actions",
                # "follow_up": "plan_follow_up", # Removed
                "generate_response": "generate_response",
            },
        )
        graph.add_edge("replan_actions", "execute_mcp_actions")
        # graph.add_edge("plan_follow_up", "execute_mcp_actions") # Removed

        graph.set_entry_point("initialize_context")
        graph.add_edge("generate_response", END)
        return graph

    async def _initialize_context(self, state: AgentState) -> AgentState:
        """Initialize context by getting MCP sessions and discovering tools/resources."""
        logger.info("Initializing agent context: getting MCP sessions...")
        try:
            # Get all configured sessions from the manager
            self.mcp_sessions = await self.mcp_manager.get_all_sessions()
            logger.info(f"Retrieved {len(self.mcp_sessions)} MCP sessions.")
        except Exception as e:
            logger.error(f"Failed to get MCP sessions: {e}", exc_info=self.debug)
            state.error = f"Failed to initialize MCP connections: {e}"
            # Allow continuing without MCP connections if desired, or raise here
            # For now, we continue but log the error.
            self.mcp_sessions = {}

        # Initialize containers in execution_context
        state.execution_context["available_tools"] = []
        state.execution_context["available_resources"] = []
        state.execution_context["server_ids"] = list(self.mcp_sessions.keys())

        # Discover tools and resources from each session concurrently
        tasks = {}
        for server_id, session in self.mcp_sessions.items():
            tasks[server_id] = asyncio.gather(
                session.list_tools(),
                session.list_resources(),
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
                    f"Error listing tools from {server_id}: {tools_result}", exc_info=self.debug
                )
            elif isinstance(tools_result, mcp_types.ListToolsResult) and tools_result.tools:
                tools_list = []
                for tool in tools_result.tools:
                    # Convert SDK Tool object to dict for planning prompt
                    tool_dict = tool.model_dump(exclude_none=True)
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
            elif (
                isinstance(resources_result, mcp_types.ListResourcesResult)
                and resources_result.resources
            ):
                resource_list = []
                for resource in resources_result.resources:
                    # Convert SDK Resource object to dict
                    res_dict = resource.model_dump(exclude_none=True)
                    res_dict["server_id"] = server_id
                    resource_list.append(res_dict)
                state.execution_context["available_resources"].extend(resource_list)
                logger.debug(f"Found {len(resource_list)} resources from {server_id}")

        return state

    def _determine_next_step_after_parsing(self, state: AgentState) -> str:
        """Determine whether to execute actions or generate response directly."""
        if state.error:  # If init or parsing failed
            return "generate_response"
        if state.planned_actions:
            return "execute"
        return "respond"

    def _determine_next_step_after_evaluation(self, state: AgentState) -> str:
        """Determine next step after evaluating execution results."""
        if state.error:  # If execution failed critically
            return "generate_response"
        if state.requires_replanning:
            state.requires_replanning = False  # Reset flag before going to replan
            return "replan"
        # Check if there are more actions in the *current* plan
        if state.current_action_index < len(state.planned_actions):
            return "more_actions"
        # Removed follow_up logic for simplicity, can be re-added
        # elif state.needs_more_info:
        #     return "follow_up"
        return "generate_response"

    def _parse_query(self, state: AgentState) -> AgentState:
        """Use LLM to determine which MCPs and actions are needed."""
        logger.info("Parsing user query: %s", state.query)
        state.execution_context["user_message_to_save"] = Message(role="user", content=state.query)

        # --- Format available tools/resources for LLM ---
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
                    # Use model_dump_json for schema to ensure it's serializable JSON string
                    schema_str = (
                        json.dumps(tool.get("inputSchema")) if tool.get("inputSchema") else "{}"
                    )
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
                    # Use model_dump_json for resource details
                    res_str = json.dumps(
                        {k: v for k, v in res.items() if k != "server_id"}, default=str
                    )
                    server_info += f"    - {res_str}\n"  # URI is primary identifier
            else:
                server_info += "  No resources available.\n"
            server_details.append(server_info)

        mcp_info += "\n".join(server_details)
        # --- End Formatting ---

        conversation_context = self._format_conversation_history(self.current_history)

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
        3. If database access is needed (server_id 'postgres'):
           - Use `listResources` or the available resources list above to find table schema URIs.
           - Use `readResource` with the correct URI (e.g., postgres://.../schema_name/table_name/schema) to get column details *before* writing a query.
           - Use the `query` tool via `callTool` for executing SQL. Ensure SQL uses schema qualification (e.g., `schema_name.table_name`).
        4. For other tools, use `callTool` with the correct server_id, tool name, and arguments based on the inputSchema.
        5. Plan all necessary steps sequentially if one depends on another.

        Output Format (JSON):
        {{
            "thought": "Your reasoning process.",
            "plan": "High-level plan description.",
            "actions": [
                {{
                    "server_id": "...", // REQUIRED: e.g., "postgres" or other server ID from the list
                    "mcp_method": "...", // e.g., "readResource", "callTool"
                    "params": {{...}},    // Parameters for the method (e.g., {{"uri": "..."}} or {{"name": "query", "arguments": {{"sql": "..."}}}})
                    "reason": "Why this action is needed.",
                    "expected_outcome": "What this action should provide."
                }}
                // ... more actions if needed
            ]
        }}
        If no actions are needed, use "actions": [].
        If more info needed, add "needs_more_info": true, "follow_up_question": "...".
        """

        logger.debug("Sending query to LLM for planning...")
        response = self.llm.generate(prompt)
        parsed_response = self._parse_json_response(response)
        logger.debug("Received and parsed LLM planning response")

        thought = parsed_response.get("thought", "No thought process provided.")
        plan = parsed_response.get("plan", "No plan provided.")
        state.thoughts.append(thought)
        if plan != "No plan provided.":
            state.thoughts.append(f"Plan: {plan}")
        logger.info(f"Agent Thought: {thought}")
        logger.info(f"Plan: {plan}")

        state.planned_actions = parsed_response.get("actions", [])
        state.current_action_index = 0
        logger.info(f"Planned {len(state.planned_actions)} actions.")
        for i, action in enumerate(state.planned_actions):
            logger.debug(
                f"Action {i + 1}: {action.get('server_id')}.{action.get('mcp_method')}({action.get('params', {})}) - Reason: {action.get('reason')}"
            )

        # Check if more info is needed (though follow-up node is removed, flag might be useful)
        if parsed_response.get("needs_more_info", False):
            state.needs_more_info = True
            state.follow_up_question = parsed_response.get("follow_up_question")
            state.thoughts.append(
                f"Needs more info: {state.follow_up_question or 'General clarification'}"
            )

        return state

    async def _execute_mcp_actions(self, state: AgentState) -> AgentState:
        """Execute the next planned action using the appropriate MCP ClientSession."""
        if state.current_action_index >= len(state.planned_actions):
            logger.debug("No more actions to execute in the current plan.")
            return state

        action_spec = state.planned_actions[state.current_action_index]
        server_id = action_spec.get("server_id")
        mcp_method = action_spec.get("mcp_method")
        params = action_spec.get("params", {})
        action_key = (
            f"action_{state.current_action_index}_{server_id}_{mcp_method}"  # Unique key for result
        )

        if not server_id or not mcp_method:
            error_msg = f"Invalid action specification at index {state.current_action_index}: Missing server_id or mcp_method."
            logger.error(error_msg)
            state.thoughts.append(f"Error: {error_msg}")
            state.mcp_results[action_key] = McpError(
                mcp_types.ErrorData(code=-32602, message=error_msg)
            )
            state.current_action_index += 1
            state.error = error_msg  # Signal critical error
            return state

        if server_id not in self.mcp_sessions:
            error_msg = f"MCP session for server_id '{server_id}' not available."
            logger.error(error_msg)
            state.thoughts.append(f"Error: {error_msg}")
            state.mcp_results[action_key] = McpError(
                mcp_types.ErrorData(code=-32000, message=error_msg)
            )
            state.current_action_index += 1
            state.error = error_msg  # Signal critical error
            return state

        session = self.mcp_sessions[server_id]
        action_label = f"{server_id}.{mcp_method}({json.dumps(params, default=str)})"
        logger.info(
            f"Executing action {state.current_action_index + 1}/{len(state.planned_actions)}: {action_label}"
        )
        state.thoughts.append(f"Executing: {action_label}")

        try:
            result: Any = None
            if mcp_method == "listResources":
                result = await session.list_resources()
            elif mcp_method == "readResource":
                if "uri" not in params:
                    raise ValueError("Missing 'uri' parameter")
                result = await session.read_resource(uri=params["uri"])
            elif mcp_method == "listTools":
                result = await session.list_tools()
            elif mcp_method == "callTool":
                if "name" not in params:
                    raise ValueError("Missing 'name' parameter")
                # TODO: Add schema qualification logic back here if needed, similar to previous version
                # sql = params.get("arguments", {}).get("sql")
                # if server_id == "postgres" and params["name"] == "query" and sql:
                #     enhanced_sql = self._ensure_schema_qualification(sql, state) # Ensure this method exists or adapt
                #     if enhanced_sql != sql:
                #          logger.info(f"Enhanced SQL: {enhanced_sql}")
                #          params["arguments"]["sql"] = enhanced_sql
                result = await session.call_tool(
                    name=params["name"], arguments=params.get("arguments")
                )
            else:
                raise ValueError(f"Unsupported MCP method: {mcp_method}")

            logger.debug(f"Action successful: {action_label}")
            state.thoughts.append(f"Action Result: Success for {action_label}")
            state.mcp_results[action_key] = result  # Store the raw SDK result object

            # Specific handling for CallToolResult to check isError
            if isinstance(result, mcp_types.CallToolResult) and result.isError:
                error_msg = "Tool execution failed."
                if result.content and isinstance(result.content[0], mcp_types.TextContent):
                    error_msg = result.content[0].text
                logger.warning(f"Tool call {action_label} reported error: {error_msg}")
                state.thoughts.append(f"Action Result: Tool error for {action_label}: {error_msg}")
                # Store error info for evaluation/replanning
                state.mcp_results[action_key] = McpError(
                    mcp_types.ErrorData(code=-32001, message=error_msg)
                )

        except McpError as e:
            error_msg = e.error.message
            logger.error(f"MCPError executing {action_label}: {error_msg}", exc_info=self.debug)
            state.thoughts.append(f"Action Result: MCP Error for {action_label}: {error_msg}")
            state.mcp_results[action_key] = e  # Store the McpError object
        except Exception as e:
            error_msg = f"Unexpected error executing {action_label}: {str(e)}"
            logger.error(error_msg, exc_info=self.debug)
            state.thoughts.append(f"Action Result: Unexpected Error for {action_label}: {str(e)}")
            state.mcp_results[action_key] = McpError(
                mcp_types.ErrorData(code=-32000, message=error_msg)
            )
            state.error = error_msg  # Signal critical error if unexpected exception occurs

        state.current_action_index += 1
        return state

    def _evaluate_results(self, state: AgentState) -> AgentState:
        """Evaluate action results to determine if replanning is needed."""
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

        # Check if the result is an McpError object (indicating failure)
        if isinstance(last_result, McpError):
            logger.info(
                f"Action {last_action_index + 1} ({action_spec.get('mcp_method')}) failed, initiating replanning."
            )
            state.requires_replanning = True
            state.execution_context["last_error"] = last_result.error.message
            # Check for specific SQL errors indicating schema issues
            if (
                server_id == "postgres"
                and "relation" in str(last_result.error.message).lower()
                and "does not exist" in str(last_result.error.message).lower()
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

    def _replan_actions(self, state: AgentState) -> AgentState:
        """Replan actions based on execution errors."""
        logger.info("Replanning actions due to execution issues...")

        last_error = state.execution_context.get("last_error", "Unknown error")
        schema_qualification_needed = state.execution_context.get(
            "schema_qualification_needed", False
        )

        # Format available tools/resources again for the replanning prompt
        mcp_info = (
            "Available Servers, Tools, and Resources:\n"  # Reuse formatting from _parse_query
        )
        server_details = []
        for server_id in state.execution_context.get("server_ids", []):
            # ... (reuse formatting logic from _parse_query) ...
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
            # ... (add tool/resource details as in _parse_query) ...
            server_details.append(server_info)
        mcp_info += "\n".join(server_details)

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

        logger.debug("Sending replanning prompt to LLM...")
        response = self.llm.generate(prompt)
        parsed_response = self._parse_json_response(response)

        thought = parsed_response.get("thought", "Revising plan due to execution error.")
        revised_plan = parsed_response.get("revised_plan", "Trying alternative approach.")
        state.thoughts.append(f"Replanning Thought: {thought}")
        state.thoughts.append(f"Revised Plan: {revised_plan}")

        new_actions = parsed_response.get("actions", [])
        if new_actions:
            # Replace the rest of the plan with the new actions
            state.planned_actions = (
                state.planned_actions[: state.current_action_index - 1] + new_actions
            )
            state.current_action_index = max(
                0, state.current_action_index - 1
            )  # Reset index to retry/continue
            logger.info(
                f"Replanned starting from action {state.current_action_index + 1} with {len(new_actions)} new action(s)."
            )
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

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response based on all gathered information."""
        logger.info("Generating final response...")
        formatted_results = self._format_results(
            state.mcp_results
        )  # Use helper to format SDK results
        conversation_context = self._format_conversation_history(self.current_history)  # Use helper

        prompt = f"""
        Generate a comprehensive response to the user's query based on the conversation history and the results of MCP actions.

        User Query: {state.query}

        Conversation History:
        {conversation_context}

        MCP Interaction Results (Raw):
        {formatted_results}

        Agent Thought Process:
        {self._format_thoughts(state.thoughts)}

        Instructions:
        1. Synthesize information from successful MCP actions.
        2. Explain any errors encountered and whether they were resolved.
        3. If database queries were run, present the data clearly. Extract data from the 'text' field of TextContent within CallToolResult.
        4. Address all parts of the original user query.
        5. Use a helpful and informative tone. Do not just dump raw JSON.
        """

        logger.debug("Sending response generation prompt to LLM.")
        response = self.llm.generate(prompt)
        state.response = response
        logger.info("Final response generated (length: %d chars).", len(response))

        # Prepare assistant message for saving
        state.execution_context["assistant_message_to_save"] = Message(
            role="assistant", content=response
        )

        return state

    # --- Helper Methods ---
    def _format_conversation_history(self, history: List[Message]) -> str:
        """Format conversation history for prompts."""
        if not history:
            return "No history."
        return "\n".join(map(str, history))

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
                if isinstance(result_data, McpError):
                    formatted += f"   Result: Error - {result_data.error.message}\n"
                elif result_data:
                    formatted += (
                        f"   Result: Success (details omitted)\n"  # Avoid dumping large raw results
                    )
                else:
                    formatted += f"   Result: Not found\n"

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
            if result and not isinstance(result, McpError):
                # Serialize the SDK result object safely
                try:
                    successful_data[action_key] = json.loads(
                        result.model_dump_json(exclude_none=True)
                    )
                except Exception:
                    successful_data[action_key] = (
                        f"<{type(result).__name__} object - Non-JSON Serializable>"
                    )

        return (
            json.dumps(successful_data, indent=2, default=str)
            if successful_data
            else "No successful results yet."
        )

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format MCP results for the final response generation prompt."""
        formatted = ""
        for key, result in results.items():
            formatted += f"--- Result for {key} ---\n"
            if isinstance(result, McpError):
                formatted += f"Type: Error\nMessage: {result.error.message}\n"
            elif isinstance(result, mcp_types.ListResourcesResult):
                formatted += f"Type: ListResourcesResult\nResources:\n"
                if result.resources:
                    for res in result.resources:
                        formatted += f"  - URI: {res.uri}, Name: {res.name}, Type: {res.mimeType}\n"
                else:
                    formatted += "  (No resources listed)\n"
            elif isinstance(result, mcp_types.ReadResourceResult):
                formatted += f"Type: ReadResourceResult\nContent:\n"
                if result.contents:
                    content = result.contents[0]  # Assume single content for now
                    if isinstance(content, mcp_types.TextResourceContents):
                        formatted += f"  URI: {content.uri}, Type: {content.mimeType}\n  Text: {content.text[:200]}...\n"  # Truncate long text
                    elif isinstance(content, mcp_types.BlobResourceContents):
                        formatted += f"  URI: {content.uri}, Type: {content.mimeType}\n  Blob: (base64 data omitted)\n"
                else:
                    formatted += "  (No content returned)\n"
            elif isinstance(result, mcp_types.ListToolsResult):
                formatted += f"Type: ListToolsResult\nTools:\n"
                if result.tools:
                    for tool in result.tools:
                        schema_str = json.dumps(tool.inputSchema) if tool.inputSchema else "{}"
                        formatted += f"  - Name: {tool.name}, Desc: {tool.description}, Schema: {schema_str}\n"
                else:
                    formatted += "  (No tools listed)\n"
            elif isinstance(result, mcp_types.CallToolResult):
                formatted += f"Type: CallToolResult\nError: {result.isError}\nContent:\n"
                if result.content:
                    for content_item in result.content:
                        if isinstance(content_item, mcp_types.TextContent):
                            formatted += f"  - Text: {content_item.text[:200]}...\n"  # Truncate
                        elif isinstance(content_item, mcp_types.ImageContent):
                            formatted += (
                                f"  - Image: Type={content_item.mimeType}, (data omitted)\n"
                            )
                        elif isinstance(content_item, mcp_types.EmbeddedResource):
                            formatted += f"  - Embedded Resource: URI={content_item.resource.uri}, Type={content_item.resource.mimeType}\n"  # Omit content
                else:
                    formatted += "  (No content returned)\n"
            else:
                formatted += f"Type: Unknown ({type(result).__name__})\nData: {str(result)[:200]}...\n"  # Truncate unknown
            formatted += "---\n"
        return formatted if formatted else "No results available."

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with improved reliability (implementation unchanged)."""
        try:
            match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                match = re.search(r"(\{[\s\S]*\})", response)
                json_str = match.group(1) if match else response

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

    # --- End Helper Methods ---

    async def run(self, query: str, session_id: Optional[str] = None) -> AgentState:
        """Execute the agent workflow."""
        logger.info(f"Starting agent run for query: '{query[:50]}...'")
        start_time = asyncio.get_event_loop().time()

        # Generate or use session ID
        definite_session_id = session_id or str(uuid.uuid4())
        logger.info(f"Session ID: {definite_session_id}")

        # Load history
        try:
            await self.db.ensure_connected()  # Ensure DB is connected before loading
            self.current_history = await self._load_history(definite_session_id)
        except Exception as e:
            logger.error(f"Failed to load history for session {definite_session_id}: {e}")
            self.current_history = []  # Proceed with empty history on DB error

        # Handle management commands
        if query.strip().lower() == "clear history":
            await self.clear_history(definite_session_id)
            self.current_history = []
            final_state = AgentState(
                query=query, session_id=definite_session_id, response="History cleared."
            )
            return final_state
        if query.strip().lower() == "show history":
            history_text = "History:\n" + (
                "\n".join(map(str, self.current_history)) if self.current_history else "(empty)"
            )
            final_state = AgentState(
                query=query, session_id=definite_session_id, response=history_text
            )
            return final_state

        # Save user message (attempt even if DB disconnected, updates in-memory)
        user_message = Message(role="user", content=query)
        await self._save_message(definite_session_id, user_message)

        # Run graph
        initial_state = AgentState(query=query, session_id=definite_session_id)
        final_state = AgentState()  # Default empty state
        try:
            # The graph invoke is synchronous in LangGraph, but our nodes are async.
            # LangGraph handles running async nodes within its sync invoke method.
            # However, for clarity and potential future compatibility, running within asyncio.run might be better if LangGraph changes.
            # For now, assume LangGraph's astream handles the async nodes correctly.
            async for event in self.graph.astream(initial_state):
                # For this refactor, we'll just get the final state
                # Note: Need to check the exact structure returned by astream
                # It typically yields dicts with node names as keys
                # The final result is often the value associated with the '__end__' key or the last node executed.
                # Simpler: Collect the latest state from events.
                last_state = None
                for node_name, node_value in event.items():
                    # Check if the value is the state dictionary itself or contains it
                    current_state_dict = {}
                    if isinstance(node_value, AgentState):
                        current_state_dict = node_value.model_dump()
                    elif isinstance(node_value, dict):  # Sometimes state is nested
                        current_state_dict = node_value

                    if "query" in current_state_dict:  # Heuristic: check if it looks like our state
                        # Try parsing into AgentState to be sure
                        try:
                            last_state = AgentState.model_validate(current_state_dict)
                        except Exception:  # If validation fails, ignore
                            pass

            final_state = (
                last_state if last_state else AgentState()
            )  # Use last valid state observed
            logger.debug("Agent workflow graph execution completed.")

            if not final_state.response and not final_state.error:
                if final_state == AgentState():  # Check if it's still the default empty state
                    logger.error("Graph finished without producing a final state.")
                    final_state = initial_state  # Fallback to initial state? Or create error state
                    final_state.error = "Workflow ended unexpectedly without a final result."
                    final_state.response = "An internal error occurred."

        except Exception as e:
            logger.error(f"Error during agent graph execution: {e}", exc_info=True)
            final_state = AgentState(query=query, session_id=definite_session_id)
            final_state.error = f"Agent execution failed: {e}"
            final_state.response = "I encountered an error processing your request."

        # Save assistant message (attempt even if DB disconnected)
        if final_state.response and not final_state.error:
            assistant_message = Message(role="assistant", content=final_state.response)
            await self._save_message(definite_session_id, assistant_message)
        elif final_state.error and final_state.response:
            # Save the error response as assistant message
            assistant_message = Message(role="assistant", content=final_state.response)
            await self._save_message(definite_session_id, assistant_message)

        end_time = asyncio.get_event_loop().time()
        logger.info(f"Agent run completed in {end_time - start_time:.2f} seconds.")
        return final_state
