import json
import logging
import re
import sys
import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.connectors.cm_llm_connector import CMLLMConnector
from src.connectors.cm_mcp_connector import CMMCPManager
from src.core.agent_db_connection import AgentDatabase, AgentPostgresSettings, fetch_agent_db_url
from src.mcp.mcp_client import MCPClient

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
    """Agent workflow state"""

    query: str = ""
    session_id: Optional[str] = None
    mcp_results: Dict[str, Any] = Field(default_factory=dict)
    thoughts: List[str] = Field(default_factory=list)
    response: Optional[str] = None
    error: Optional[str] = None
    mcp_options: Optional[Dict[str, Any]] = None
    planned_actions: List[Dict[str, Any]] = Field(default_factory=list)
    current_action_index: int = 0
    needs_more_info: bool = False
    follow_up_question: Optional[str] = None
    available_schemas: List[Dict[str, str]] = Field(default_factory=list)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    requires_replanning: bool = False


class Agent:
    """LangGraph-based agent implementation with SDK-based connections and database-backed conversation history"""

    def __init__(
        self,
        llm_connector: Optional[CMLLMConnector] = None,
        mcp_manager: Optional[CMMCPManager] = None,
        mcp_clients: Optional[Dict[str, MCPClient]] = None,
        db_config_id: str = "AGENT_SESSION_DB",
        debug: bool = False,
    ):
        """
        Initialize the agent with SDK-based connections.

        Args:
            llm_connector: SDK-based LLM connector (will create if None)
            mcp_manager: SDK-based MCP manager (will create if None)
            mcp_clients: Optional dictionary of MCP clients (overrides mcp_manager)
            db_config_id: Config ID for the database connection in ConfigManager
            debug: Enable debug logging
        """
        # Initialize LLM connector
        self.llm = llm_connector or CMLLMConnector(config_id="LLM")

        # Initialize MCP clients
        if mcp_clients is not None:
            # Use provided clients directly
            self.mcp_clients = mcp_clients
            logger.info(f"Using {len(mcp_clients)} provided MCP clients")
        else:
            # Use MCP manager passed in or created
            current_mcp_manager = mcp_manager or CMMCPManager()

            # Get clients from the manager. This relies on ConfigManager being initialized.
            try:
                self.mcp_clients = current_mcp_manager.get_all_clients()
                logger.info(
                    f"Using {len(self.mcp_clients)} MCP clients retrieved via CMMCPManager."
                )
            except Exception as e:
                logger.error(
                    f"Failed to retrieve MCP clients via CMMCPManager: {e}. Agent might lack tool access.",
                    exc_info=True,
                )
                self.mcp_clients = {}  # Initialize empty if retrieval fails

        self.debug = debug

        # Database configuration
        self.db_config_id = db_config_id
        self.db = AgentDatabase(settings=AgentPostgresSettings())
        self.current_history = []  # Temporary storage for current conversation

        # Set logging level based on debug flag
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        self.graph = self._build_graph().compile()
        logger.info("Agent workflow graph compiled")

    async def get_history(self, session_id: str) -> List[Message]:
        """Get the conversation history for a session from the database.

        Args:
            session_id: The session ID to retrieve history for

        Returns:
            List of messages in the conversation history
        """
        return await self._load_history(session_id)

    async def clear_history(self, session_id: str) -> bool:
        """Clear the conversation history for a session in the database.

        Args:
            session_id: The session ID to clear history for

        Returns:
            True if successfully cleared, False otherwise
        """
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

    async def _load_history(self, session_id: str) -> List[Message]:
        """Load conversation history from the database for a session.

        Args:
            session_id: The session ID to load history for

        Returns:
            List of Message objects representing the conversation history
        """
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
            logger.info(f"Loaded {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Error loading history for session {session_id}: {e}")
            return []  # Return empty list on error

    async def _save_message(self, session_id: str, message: Message) -> bool:
        """Save a message to the database for a session.

        Args:
            session_id: The session ID to save the message for
            message: The message to save

        Returns:
            True if successfully saved, False otherwise
        """
        try:
            await self.db.ensure_connected()

            # Get the next message order
            max_order = await self.db.execute_query(
                "SELECT MAX(message_order) FROM conversation_messages WHERE session_id = $1",
                session_id,
                fetch_type="val",
            )

            # If this is the first message, start at 0
            message_order = 0 if max_order is None else max_order + 1

            # Insert the message
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

            # Also add to current history for this run
            self.current_history.append(message)

            logger.info(f"Saved message for session {session_id} with order {message_order}")
            return True
        except Exception as e:
            logger.error(f"Error saving message for session {session_id}: {e}")
            return False  # Return False on error

    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph with improved planning capabilities"""
        graph = StateGraph(AgentState)

        # Define core nodes
        graph.add_node("initialize_context", self._initialize_context)
        graph.add_node("parse_query", self._parse_query)
        graph.add_node("execute_mcp_actions", self._execute_mcp_actions)
        graph.add_node("evaluate_results", self._evaluate_results)
        graph.add_node("replan_actions", self._replan_actions)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("plan_follow_up", self._plan_follow_up)

        # Define graph structure
        graph.add_edge("initialize_context", "parse_query")

        # Define conditional edges
        graph.add_conditional_edges(
            "parse_query",
            self._determine_next_step_after_parsing,
            {
                "execute": "execute_mcp_actions",
                "respond": "generate_response",
            },
        )

        graph.add_edge("execute_mcp_actions", "evaluate_results")

        graph.add_conditional_edges(
            "evaluate_results",
            self._determine_next_step_after_evaluation,
            {
                "more_actions": "execute_mcp_actions",
                "replan": "replan_actions",
                "follow_up": "plan_follow_up",
                "generate_response": "generate_response",
            },
        )

        graph.add_edge("replan_actions", "execute_mcp_actions")

        graph.add_conditional_edges(
            "plan_follow_up",
            lambda state: "execute" if state.planned_actions else "respond",
            {
                "execute": "execute_mcp_actions",
                "respond": "generate_response",
            },
        )

        # Set entry point and terminal node
        graph.set_entry_point("initialize_context")
        graph.add_edge("generate_response", END)

        return graph

    def _initialize_context(self, state: AgentState) -> AgentState:
        """Initialize the agent context with available tools and resources from all MCP clients"""
        logger.info("Initializing agent context from all MCP clients")

        # Initialize containers for tools and resources
        state.execution_context["available_tools"] = []
        state.available_schemas = []
        state.execution_context["available_resources"] = []
        state.execution_context["server_ids"] = list(self.mcp_clients.keys())

        # Discover available tools and resources from all clients
        for server_id, client in self.mcp_clients.items():
            try:
                # Get available tools from this client
                tools_response = client.list_tools()
                tools = tools_response.get("tools", [])

                # Add server_id to each tool for identification
                for tool in tools:
                    tool["server_id"] = server_id

                state.execution_context["available_tools"].extend(tools)

                # Get available resources (schemas) from this client
                resources_response = client.list_resources()
                resources = resources_response.get("resources", [])

                # Add server_id to each resource for identification
                for resource in resources:
                    resource["server_id"] = server_id

                state.available_schemas.extend(resources)
                state.execution_context["available_resources"].extend(resources)

                logger.debug(
                    f"Initialized context from {server_id} with {len(tools)} tools and {len(resources)} resources"
                )

            except Exception as e:
                logger.error(
                    f"Error initializing context from {server_id}: {e}", exc_info=self.debug
                )
                state.error = f"Failed to initialize agent context from {server_id}: {e}"

        return state

    def _determine_next_step_after_parsing(self, state: AgentState) -> str:
        """Determine whether to execute actions or generate response directly"""
        if state.planned_actions:
            return "execute"
        return "respond"

    def _determine_next_step_after_evaluation(self, state: AgentState) -> str:
        """Determine next step after evaluating execution results"""
        if state.requires_replanning:
            return "replan"
        if state.current_action_index < len(state.planned_actions):
            return "more_actions"
        elif state.needs_more_info:
            return "follow_up"
        return "generate_response"

    def _parse_query(self, state: AgentState) -> AgentState:
        """Use LLM to determine which MCPs and actions are needed with improved planning"""
        logger.info(
            "Parsing user query: %s",
            state.query,
        )

        # Add the user query to conversation history
        new_message = Message(role="user", content=state.query)

        # Store the message in the database (need to handle this in run since this isn't async)
        state.execution_context["user_message_to_save"] = new_message

        # Prepare MCP tool information for the LLM prompt
        mcp_info = "Available Tool Servers:\n"
        for server_id in state.execution_context.get("server_ids", []):
            mcp_info += f"Server: {server_id}\n"

            # Get tools for this server
            tools = [
                tool
                for tool in state.execution_context.get("available_tools", [])
                if tool.get("server_id") == server_id
            ]

            if tools:
                mcp_info += "  Available Tools:\n"
                for tool in tools:
                    schema_desc = (
                        f" Requires input: {json.dumps(tool.get('inputSchema', {}))}"
                        if tool.get("inputSchema")
                        else ""
                    )
                    mcp_info += f"  - Name: {tool.get('name', 'N/A')}, Description: {tool.get('description', 'N/A')}{schema_desc}\n"
            else:
                mcp_info += "  No tools currently available.\n"

            # Format schema information for this server
            resources = [
                res for res in state.available_schemas if res.get("server_id") == server_id
            ]

            if resources:
                mcp_info += "\n  Available Data Resources:\n"
                for res in resources:
                    mcp_info += f"  - {res.get('name', res.get('uri'))} (URI: {res.get('uri')})\n"

        # Format conversation history for the prompt
        conversation_context = self._format_conversation_history(self.current_history)

        # Generate the planning prompt with improved multi-hop guidance
        prompt = f"""
        You need to analyze the user's query and create a comprehensive plan to address it using the available tool servers.
        
        You can interact with tool servers using these MCP methods:
        - `mcp_listResources`: Lists available data resources (like table schemas). Returns a list of {{uri, name, mimeType}}.
        - `mcp_readResource`: Reads the content of a specific resource URI. Requires 'uri' parameter. Returns {{contents: [{{uri, text, mimeType}}]}}. Use this to get schema details.
        - `mcp_listTools`: Lists available tools and their input requirements.
        - `mcp_callTool`: Executes a specific tool. Requires 'name' and 'arguments' parameters.

        Current Available Resources and Tools:
        {mcp_info}

        Conversation history:
        {conversation_context}

        User query: {state.query}

        Think step-by-step to create a complete plan that addresses the user's question. When creating your plan, consider the following guidelines:
        
        1. For database-related questions:
           - If you need schema knowledge, plan to first fetch the schema using `mcp_readResource`
           - SQL queries should ALWAYS include schema qualification (e.g., use "schemaname.tablename" not just "tablename")
           - Always inspect resource URIs carefully to extract the correct schema name
           - Remember that a resource URI like "postgres://user@host:port/db/schema_name/table_name/schema" indicates the table is in schema "schema_name"
        
        2. For RAG-related queries:
           - Use the 'rag_get_context' tool to retrieve relevant context chunks
           - Use the 'rag_generate_completion' tool to generate completions with context
        
        3. Multi-step reasoning:
           - Break down complex queries into distinct stages
           - If information from one step is needed for another, plan the entire sequence
           - Don't leave follow-up actions implicit or for future steps
        
        4. Error handling:
           - Consider alternative approaches if a primary action might fail
           - For database queries, plan fallback actions in case of schema or table name issues
        
        IMPORTANT: 
        - You MUST specify the server_id for each action (which server to use: "postgres" or "rag")
        - Use previous conversations for context when relevant. If the user refers to previous queries or results, use that context in your planning.

        Format your response as JSON:
        {{
            "thought": "your reasoning",
            "plan": "high-level plan description",
            "actions": [
                {{
                    "server_id": "postgres", // REQUIRED: Specify which server to use: postgres or rag
                    "mcp_method": "mcp_readResource", // or mcp_callTool, etc.
                    "params": {{ // Parameters for the MCP method
                        "uri": "postgres://..." // for readResource
                        // OR
                        "name": "query",       // for callTool
                        "arguments": {{ "sql": "SELECT ..." }} // for callTool
                    }},
                    "reason": "why this action is needed",
                    "expected_outcome": "what we expect to learn from this action"
                }}
                // ... more actions if needed
            ]
        }}
        
        If no actions are needed, use "actions": [].
        If more info needed, include "needs_more_info": true, "follow_up_question": "..."
        """

        logger.debug("Sending query to LLM for planning with MCP methods")
        response = self.llm.generate(prompt)
        parsed_response = self._parse_json_response(response)
        logger.debug("Received and parsed LLM planning response")

        # Process the response
        thought = parsed_response.get("thought", "No thought process provided")
        plan = parsed_response.get("plan", "No plan provided")

        # Store the thought and plan
        state.thoughts.append(thought)
        if plan and plan != "No plan provided":
            state.thoughts.append(f"Plan: {plan}")

        logger.info("Agent thought: %s", thought)
        logger.info("Plan: %s", plan)

        # Process planned actions
        if "actions" in parsed_response:
            state.planned_actions = parsed_response.get("actions", [])
            state.current_action_index = 0
            action_count = len(state.planned_actions)
            logger.info("Planned %d actions", action_count)
            for i, action in enumerate(state.planned_actions):
                logger.debug(
                    "Action %d: %s - %s - Server: %s",
                    i + 1,
                    action.get("mcp_method", "unknown"),
                    json.dumps(action.get("params", {}), default=str),
                    action.get("server_id", "unknown"),
                )
                # Store reason if provided
                if "reason" in action:
                    logger.debug("Reason: %s", action.get("reason"))

        # Check if more info is needed
        if parsed_response.get("needs_more_info", False):
            state.needs_more_info = True
            state.follow_up_question = parsed_response.get("follow_up_question")
            follow_up = state.follow_up_question or "No specific question provided"
            state.thoughts.append(f"Need more information: {follow_up}")
            logger.info("Agent needs more information: %s", follow_up)

        return state

    def _execute_mcp_actions(self, state: AgentState) -> AgentState:
        """Execute actions through the appropriate MCPs"""
        if state.current_action_index >= len(state.planned_actions):
            logger.debug("No more actions to execute")
            return state

        action_spec = state.planned_actions[state.current_action_index]
        server_id = action_spec.get("server_id")
        mcp_method = action_spec.get("mcp_method")
        params = action_spec.get("params", {})

        # Validate we have a server_id
        if not server_id:
            error_msg = "Missing server_id in action specification"
            logger.error(error_msg)
            state.thoughts.append(f"Error: {error_msg}")
            state.mcp_results[f"action_{state.current_action_index}_{mcp_method}"] = {
                "error": error_msg
            }
            state.current_action_index += 1
            return state

        # Check if the server exists
        if server_id not in self.mcp_clients:
            error_msg = f"Unknown server_id: {server_id}"
            logger.error(error_msg)
            state.thoughts.append(f"Error: {error_msg}")
            state.mcp_results[f"action_{state.current_action_index}_{mcp_method}"] = {
                "error": error_msg
            }
            state.current_action_index += 1
            return state

        # Get the appropriate client
        client = self.mcp_clients[server_id]

        # Create a shortened version for logging
        params_str = json.dumps(params, default=str)
        action_label = f"{server_id}.{mcp_method}({params_str})"

        logger.info(
            "Executing action %d/%d: %s",
            state.current_action_index + 1,
            len(state.planned_actions),
            action_label,
        )
        state.thoughts.append(f"Executing MCP action: {action_label}")

        result = None
        error_msg = None
        try:
            # Use the specific MCPClient to execute the action
            if mcp_method == "mcp_listResources":
                result = client.list_resources()
            elif mcp_method == "mcp_readResource":
                if "uri" not in params:
                    raise ValueError("Missing 'uri' parameter for readResource")
                result = client.read_resource(uri=params["uri"])
            elif mcp_method == "mcp_listTools":
                result = client.list_tools()
            elif mcp_method == "mcp_callTool":
                if "name" not in params:
                    raise ValueError("Missing 'name' parameter for callTool")

                # Special handling for SQL queries (postgres server only)
                if (
                    server_id == "postgres"
                    and params["name"] == "query"
                    and "arguments" in params
                    and "sql" in params["arguments"]
                ):
                    # Check if SQL has schema qualification in FROM clauses
                    sql = params["arguments"]["sql"]
                    enhanced_sql = self._ensure_schema_qualification(sql, state)
                    if enhanced_sql != sql:
                        logger.info(f"Enhanced SQL with schema qualification: {enhanced_sql}")
                        state.thoughts.append(
                            f"Enhanced SQL with schema qualification: {enhanced_sql}"
                        )
                        params["arguments"]["sql"] = enhanced_sql

                result = client.call_tool(name=params["name"], arguments=params.get("arguments"))
            else:
                raise ValueError(f"Unsupported MCP method in plan: {mcp_method}")

            logger.debug(f"MCP action execution successful via {server_id} client")
            state.thoughts.append(f"Action result: Successfully executed {server_id}.{mcp_method}")

            # Store result with a unique key per action
            action_key = f"action_{state.current_action_index}_{server_id}_{mcp_method}"
            state.mcp_results[action_key] = result

            # For SQL queries, extract and store results in a more accessible format
            if (
                server_id == "postgres"
                and mcp_method == "mcp_callTool"
                and params["name"] == "query"
            ):
                try:
                    # Extract actual data from the tool response
                    if "content" in result and result["content"]:
                        content_text = result["content"][0].get("text", "")
                        if content_text:
                            # Try to parse as JSON
                            try:
                                parsed_data = json.loads(content_text)
                                state.execution_context[
                                    f"query_result_{state.current_action_index}"
                                ] = parsed_data
                                logger.debug(
                                    f"Stored parsed query result for action {state.current_action_index}"
                                )
                            except json.JSONDecodeError:
                                state.execution_context[
                                    f"query_result_{state.current_action_index}"
                                ] = content_text
                except Exception as e:
                    logger.warning(f"Failed to extract and parse query result: {e}")

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Error executing MCP action {server_id}.{mcp_method}: {error_msg}",
                exc_info=self.debug,
            )
            state.thoughts.append(f"Error executing {server_id}.{mcp_method}: {error_msg}")
            action_key = f"action_{state.current_action_index}_{server_id}_{mcp_method}"
            state.mcp_results[action_key] = {"error": error_msg}

        # Advance to next action
        state.current_action_index += 1
        return state

    def _evaluate_results(self, state: AgentState) -> AgentState:
        """Evaluate action results to determine if replanning is needed"""
        if not state.mcp_results:
            return state

        # Get the most recent action result
        last_action_index = state.current_action_index - 1
        action_spec = (
            state.planned_actions[last_action_index]
            if last_action_index >= 0 and last_action_index < len(state.planned_actions)
            else None
        )

        if not action_spec:
            return state

        server_id = action_spec.get("server_id", "")
        mcp_method = action_spec.get("mcp_method", "")
        last_result_key = f"action_{last_action_index}_{server_id}_{mcp_method}"

        # If the key doesn't exist, try the old format without server_id
        if last_result_key not in state.mcp_results:
            last_result_key = f"action_{last_action_index}_{mcp_method}"

        # Check if the last action had an error
        last_result = state.mcp_results.get(last_result_key, {})

        # For callTool actions, check isError flag
        if mcp_method == "mcp_callTool" and isinstance(last_result, dict):
            if last_result.get("isError", False) or "error" in last_result:
                logger.info("Last action failed, may need replanning")
                state.requires_replanning = True

                # Extract error message for better context
                error_msg = "Unknown error"
                if "error" in last_result:
                    error_msg = last_result["error"]
                elif "content" in last_result and last_result["content"]:
                    content = (
                        last_result["content"][0]
                        if isinstance(last_result["content"], list)
                        else last_result["content"]
                    )
                    if isinstance(content, dict) and "text" in content:
                        error_msg = content["text"]

                state.execution_context["last_error"] = error_msg
                state.thoughts.append(f"Action failed with error: {error_msg}")
                logger.debug(f"Identified error in result: {error_msg}")

                # Check for specific SQL errors that suggest schema qualification issues (postgres only)
                if (
                    server_id == "postgres"
                    and "relation" in str(error_msg)
                    and "does not exist" in str(error_msg)
                ):
                    state.execution_context["schema_qualification_needed"] = True
                    state.thoughts.append(
                        "The error suggests a relation/table was not found. May need schema qualification."
                    )

        return state

    def _replan_actions(self, state: AgentState) -> AgentState:
        """Replan actions based on execution results and errors"""
        logger.info("Replanning actions due to execution issues")

        # Extract context for replanning
        last_error = state.execution_context.get("last_error", "Unknown error")
        schema_qualification_needed = state.execution_context.get(
            "schema_qualification_needed", False
        )

        # Build a prompt that includes the error information and guidance
        prompt = f"""
        The previous plan encountered an issue and needs to be revised.
        
        User query: {state.query}
        
        Current plan actions executed so far:
        {self._format_actions_executed(state)}
        
        Error encountered: {last_error}
        
        {"The error suggests a table or relation wasn't found - ensure you use proper schema qualification (schema_name.table_name) in SQL queries." if schema_qualification_needed else ""}
        
        Available schemas in the database:
        {self._format_available_schemas(state)}
        
        Results from previously successful actions:
        {self._format_successful_results(state)}
        
        Think step-by-step to revise the plan and fix the issue. Consider:
        1. Is there a schema qualification issue? 
        2. Are the table names correct?
        3. Does the query syntax need adjustments?
        4. Are you using the correct server (postgres or rag) for this action?
        
        IMPORTANT: You MUST specify the server_id for each action.
        
        Format your response as JSON:
        {{
            "thought": "your reasoning on what went wrong",
            "revised_plan": "description of the revised approach",
            "actions": [
                {{
                    "server_id": "postgres", // REQUIRED: Specify which server to use
                    "mcp_method": "mcp_method_name",
                    "params": {{ parameters }},
                    "reason": "why this action will resolve the issue"
                }}
            ]
        }}
        """

        logger.debug("Sending replanning prompt to LLM")
        response = self.llm.generate(prompt)
        parsed_response = self._parse_json_response(response)

        # Extract the revised plan
        thought = parsed_response.get("thought", "Revising plan due to execution error")
        revised_plan = parsed_response.get("revised_plan", "Trying alternative approach")

        state.thoughts.append(thought)
        state.thoughts.append(f"Revised plan: {revised_plan}")

        # Update planned actions with the revised ones
        if "actions" in parsed_response and parsed_response["actions"]:
            remaining_actions = (
                state.planned_actions[state.current_action_index :]
                if state.current_action_index < len(state.planned_actions)
                else []
            )
            state.planned_actions = parsed_response["actions"] + remaining_actions
            state.current_action_index = 0
            logger.info(f"Replanned with {len(state.planned_actions)} actions")
        else:
            logger.warning("Replanning didn't produce any new actions")
            # If no new actions, proceed with current plan

        # Reset the replanning flag
        state.requires_replanning = False

        return state

    def _ensure_schema_qualification(self, sql: str, state: AgentState) -> str:
        """Ensure that SQL FROM clauses have schema qualification"""
        # Simple regex-based approach to add schema qualification
        # This is a basic implementation and might need more sophistication for complex queries

        # Extract schema names from URIs for all tables
        schema_table_map = {}
        for resource in state.available_schemas:
            # Only process postgres resources
            if resource.get("server_id") != "postgres":
                continue

            uri = resource.get("uri", "")
            # Extract schema and table from URI
            match = re.search(r"/([^/]+)/([^/]+)/schema", uri)
            if match:
                schema_name = match.group(1)
                table_name = match.group(2)
                schema_table_map[table_name.lower()] = schema_name

        if not schema_table_map:
            return sql  # No schema info available

        # Simple regex to find table references in FROM and JOIN clauses
        # This is basic and won't handle all SQL variations
        def replace_table(match):
            table_ref = match.group(1).strip()
            # Skip if already has schema qualification or is a subquery
            if "." in table_ref or "(" in table_ref:
                return match.group(0)

            # Try to find matching schema
            schema = schema_table_map.get(table_ref.lower())
            if schema:
                return match.group(0).replace(table_ref, f"{schema}.{table_ref}")
            return match.group(0)

        # Process FROM clauses
        enhanced_sql = re.sub(
            r"FROM\s+([^\s,()]+)", lambda m: replace_table(m), sql, flags=re.IGNORECASE
        )

        # Process JOIN clauses
        enhanced_sql = re.sub(
            r"JOIN\s+([^\s,()]+)", lambda m: replace_table(m), enhanced_sql, flags=re.IGNORECASE
        )

        return enhanced_sql

    def _plan_follow_up(self, state: AgentState) -> AgentState:
        """Plan follow-up actions based on execution results"""
        prompt = f"""
        Review the results of the actions taken so far and determine if additional actions are needed.
        
        User query: {state.query}
        
        Actions executed and their results:
        {self._format_results(state.mcp_results)}
        
        Based on these results, determine what to do next:
        1. Are there additional actions needed to fully answer the query?
        2. Is the information complete or do we need more data?
        3. Can we now provide a final answer to the user?
        
        IMPORTANT: You MUST specify the server_id for each action.
        
        Format your response as JSON:
        {{
            "thought": "your reasoning here",
            "actions": [
                {{
                    "server_id": "postgres", // REQUIRED: Specify which server to use
                    "mcp_method": "method_name",
                    "params": {{ parameter details }},
                    "reason": "why this action is needed"
                }}
            ]
        }}
        
        If no further actions are needed, respond with:
        {{
            "thought": "your reasoning here",
            "actions": []
        }}
        """

        response = self.llm.generate(prompt)
        parsed_response = self._parse_json_response(response)

        thought = parsed_response.get("thought", "Determining if additional actions are needed")
        state.thoughts.append(thought)

        # Update actions list
        new_actions = parsed_response.get("actions", [])
        if new_actions:
            state.planned_actions = new_actions
            state.current_action_index = 0
            state.thoughts.append(f"Planning additional actions: {len(new_actions)} actions")
        else:
            state.planned_actions = []
            state.thoughts.append("No additional actions needed")

        # Reset needs_more_info flag
        state.needs_more_info = False

        return state

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response based on all gathered information"""
        logger.info("Generating final response based on MCP action results")
        formatted_results = self._format_results(state.mcp_results)

        # Format conversation history for the prompt
        conversation_context = self._format_conversation_history(
            self.current_history, exclude_current=True
        )

        # Enhanced prompt with better guidance for different MCP response types
        prompt = f"""
        Based on the user's query and the results from interacting with multiple MCP tool servers, generate a helpful response.

        User query: {state.query}
        
        Previous conversation:
        {conversation_context}

        MCP Interaction Results:
        {formatted_results}

        Agent thought process:
        {self._format_thoughts(state.thoughts)}

        Instructions for response generation:
        1. Synthesize information from all MCP interactions to provide a complete answer
        2. If database queries were performed, present the results clearly 
        3. If RAG tools were used, incorporate the context and generated completions
        4. If there were errors during execution, explain what happened and the resolution
        5. Refer back to the user's original question to ensure all aspects are addressed
        6. Use a friendly, informative tone
        
        Look for data in MCP results with these patterns:
        - `mcp_listResources` results contain a list of resources under the 'resources' key
        - `mcp_readResource` results contain file content under the 'contents[0].text' key (usually JSON schema)
        - `mcp_callTool` results with "query" tool contain data under 'content[0].text' (usually JSON result)
        - `mcp_callTool` results for "rag_get_context" contain the retrieved context chunks
        - `mcp_callTool` results for "rag_generate_completion" contain the generated completion
        
        If there were SQL results, extract the actual data values from the JSON in the response.
        """

        logger.debug("Sending response generation prompt to LLM")
        response = self.llm.generate(prompt)
        state.response = response
        logger.info("Response generated (length: %d characters)", len(response))

        # Add the assistant's response to conversation (will be saved to DB in the run method)
        state.execution_context["assistant_message_to_save"] = Message(
            role="assistant", content=response
        )

        return state

    def _format_conversation_history(self, history: List[Message], exclude_current=False) -> str:
        """Format conversation history for prompts"""
        if not history:
            return ""

        history_to_include = history
        if exclude_current and len(history_to_include) > 0:
            history_to_include = history_to_include[:-1]

        if not history_to_include:
            return ""

        context = "\n"
        for message in history_to_include:
            context += f"{message}\n"
        return context

    def _format_thoughts(self, thoughts: List[str]) -> str:
        """Format thought process for the prompt"""
        if not thoughts:
            return "No recorded thoughts"

        formatted = ""
        for i, thought in enumerate(thoughts):
            formatted += f"{i + 1}. {thought}\n"
        return formatted

    def _format_actions_executed(self, state: AgentState) -> str:
        """Format the actions that have been executed so far"""
        if state.current_action_index == 0:
            return "No actions executed yet"

        formatted = ""
        for i in range(state.current_action_index):
            if i < len(state.planned_actions):
                action = state.planned_actions[i]
                server_id = action.get("server_id", "unknown")
                mcp_method = action.get("mcp_method", "unknown")
                params = json.dumps(action.get("params", {}), default=str)
                formatted += (
                    f"{i + 1}. Server: {server_id}, Method: {mcp_method} with params: {params}\n"
                )

                # Add result if available
                result_key = f"action_{i}_{server_id}_{mcp_method}"
                # Try old format if not found
                if result_key not in state.mcp_results:
                    result_key = f"action_{i}_{mcp_method}"

                if result_key in state.mcp_results:
                    result = state.mcp_results[result_key]
                    result_str = str(result)
                    formatted += f"   Result: {result_str}\n"

        return formatted

    def _format_available_schemas(self, state: AgentState) -> str:
        """Format available schemas for prompts"""
        if not state.available_schemas:
            return "No schema information available"

        formatted = ""
        # Group by server
        for server_id in state.execution_context.get("server_ids", []):
            server_resources = [
                res for res in state.available_schemas if res.get("server_id") == server_id
            ]

            if server_resources:
                formatted += f"Server: {server_id}\n"
                for res in server_resources:
                    name = res.get("name", "")
                    uri = res.get("uri", "")
                    formatted += f"- {name} (URI: {uri})\n"

        return formatted

    def _format_successful_results(self, state: AgentState) -> str:
        """Format results from successful actions"""
        successful_results = {}

        for key, result in state.mcp_results.items():
            # Skip results with errors
            if isinstance(result, dict) and (result.get("isError", False) or "error" in result):
                continue

            # Include this result
            successful_results[key] = result

        return json.dumps(successful_results, indent=2, default=str)

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format MCP results for LLM prompt"""
        try:
            return json.dumps(results, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error formatting results for prompt: {e}")
            return f"Error formatting results: {e}\nRaw results: {str(results)}"

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with improved reliability"""
        import re

        try:
            # Extract JSON block if present within code blocks
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if match:
                json_str = match.group(1)
            else:
                # Look for content within curly braces
                match = re.search(r"(\{[\s\S]*\})", response)
                if match:
                    json_str = match.group(1)
                else:
                    # Maybe the whole response is JSON
                    json_str = response

            # Clean up the string
            # Remove non-JSON parts before the first {
            first_brace = json_str.find("{")
            if first_brace > 0:
                json_str = json_str[first_brace:]

            # Remove non-JSON parts after the last }
            last_brace = json_str.rfind("}")
            if last_brace != -1 and last_brace < len(json_str) - 1:
                json_str = json_str[: last_brace + 1]

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response from LLM: {e}\nResponse was:\n{response}",
                exc_info=self.debug,
            )
            # Try to salvage what we can
            return {
                "thought": "Error parsing LLM response as JSON. Continuing with best effort.",
                "actions": [],
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}", exc_info=self.debug)
            return {
                "thought": f"Error: Unexpected error parsing LLM response. {e}",
                "actions": [],
            }

    async def run(self, query: str, session_id: Optional[str] = None) -> AgentState:
        """Execute the complete agent workflow with database-backed conversation history

        Args:
            query: The user query to process
            session_id: Optional session ID. If not provided, a new one will be generated.

        Returns:
            AgentState: The final agent state after workflow execution
        """
        logger.info("Starting agent workflow execution")
        logger.info("Query: %s", query)

        # Generate or use provided session ID
        definite_session_id = session_id or str(uuid.uuid4())
        logger.info(f"Using session ID: {definite_session_id}")

        # Connect to database if not already connected
        try:
            # Fetch database URL with retry logic
            db_url = await fetch_agent_db_url(self.db_config_id)
            await self.db.connect(db_url)
        except Exception as e:
            logger.error(f"Database connection error: {e}", exc_info=True)
            # Continue without database connection - we'll handle history in memory only

        # Load conversation history from database
        try:
            if self.db.is_connected():
                self.current_history = await self._load_history(definite_session_id)
                logger.info(
                    f"Loaded {len(self.current_history)} messages from database for session {definite_session_id}"
                )
            else:
                logger.warning("Database not connected, using empty conversation history")
                self.current_history = []
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}", exc_info=True)
            self.current_history = []  # Use empty history on error

        # Check for conversation management commands
        if query.strip().lower() == "clear history":
            logger.info("Clearing conversation history")
            if self.db.is_connected():
                await self.clear_history(definite_session_id)
            self.current_history = []
            response_state = AgentState(query=query, session_id=definite_session_id)
            response_state.response = "Conversation history has been cleared."
            return response_state

        if query.strip().lower() == "show history":
            logger.info("Showing conversation history")
            history_text = "Conversation history:\n\n"
            for i, message in enumerate(self.current_history):
                history_text += f"{i + 1}. {message}\n"
            response_state = AgentState(query=query, session_id=definite_session_id)
            response_state.response = history_text
            return response_state

        # Create initial state with session ID
        initial_state = AgentState(query=query, session_id=definite_session_id)

        # Save user message to database
        user_message = Message(role="user", content=query)
        if self.db.is_connected():
            await self._save_message(definite_session_id, user_message)
        self.current_history.append(user_message)

        logger.debug("Invoking agent workflow graph")
        result = self.graph.invoke(initial_state)
        logger.debug("Agent workflow graph execution completed")

        # Save assistant message to database if response was generated
        result_state = result
        # Handle different return types from newer LangGraph versions
        if isinstance(result, dict):
            logger.debug("Result is dictionary, extracting state")
            # Check if state is directly in the dictionary
            if "state" in result:
                logger.debug("Found 'state' in result dictionary")
                result_state = result["state"]
            # If there's a response, create a new state with it
            elif "response" in result:
                logger.debug("Found 'response' in result dictionary")
                result_state = AgentState(query=query, session_id=definite_session_id)
                result_state.response = result["response"]
                if "thoughts" in result:
                    result_state.thoughts = result["thoughts"]
                if "mcp_results" in result:
                    result_state.mcp_results = result["mcp_results"]
                if "error" in result:
                    result_state.error = result["error"]

        # If result_state has an assistant message to save, save it to the database
        if result_state.response and self.db.is_connected():
            assistant_message = Message(role="assistant", content=result_state.response)
            await self._save_message(definite_session_id, assistant_message)

        return result_state
