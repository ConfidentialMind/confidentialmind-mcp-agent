import json
import logging
import sys
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.mcp.mcp_client import MCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent_orchestrator")


class AgentState(BaseModel):
    """Agent workflow state"""

    query: str = ""
    mcp_results: Dict[str, Any] = Field(default_factory=dict)
    thoughts: List[str] = Field(default_factory=list)
    response: Optional[str] = None
    error: Optional[str] = None
    mcp_options: Optional[Dict[str, Any]] = None
    planned_actions: List[Dict[str, Any]] = Field(default_factory=list)
    current_action_index: int = 0
    needs_more_info: bool = False
    follow_up_question: Optional[str] = None


class Agent:
    """LangGraph-based agent implementation for Aiven services integration"""

    def __init__(self, llm_connector, mcp_client: MCPClient, debug=False):
        self.llm = llm_connector
        self.mcp_client = mcp_client
        self.debug = debug

        # Set logging level based on debug flag
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # logger.info("Initializing Agent with %d available MCPs", len(mcp_registry.list_mcps()))
        # for mcp_name in mcp_registry.list_mcps():
        #     mcp = mcp_registry.get_mcp(mcp_name)
        #     if mcp:
        #         logger.debug("MCP available: %s - %s", mcp_name, mcp.description)

        self.graph = self._build_graph().compile()
        logger.info("Agent workflow graph compiled")

    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph"""
        graph = StateGraph(AgentState)

        # Define core nodes
        graph.add_node("parse_query", self._parse_query)
        graph.add_node("execute_mcp_actions", self._execute_mcp_actions)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("plan_follow_up", self._plan_follow_up)

        # Define conditional edges
        graph.add_conditional_edges(
            "parse_query",
            self._determine_next_step_after_parsing,
            {
                "execute": "execute_mcp_actions",
                "respond": "generate_response",
            },
        )

        graph.add_conditional_edges(
            "execute_mcp_actions",
            self._determine_next_step_after_execution,
            {
                "more_actions": "execute_mcp_actions",
                "generate_response": "generate_response",
                "follow_up": "plan_follow_up",
            },
        )

        graph.add_conditional_edges(
            "plan_follow_up",
            lambda state: "execute" if state.planned_actions else "respond",
            {
                "execute": "execute_mcp_actions",
                "respond": "generate_response",
            },
        )

        # Set entry point and terminal node
        graph.set_entry_point("parse_query")
        graph.add_edge("generate_response", END)

        return graph

    def _determine_next_step_after_parsing(self, state: AgentState) -> str:
        """Determine whether to execute actions or generate response directly"""
        if state.planned_actions:
            return "execute"
        return "respond"

    def _determine_next_step_after_execution(self, state: AgentState) -> str:
        """Determine next step after executing an action"""
        if state.current_action_index < len(state.planned_actions):
            return "more_actions"
        elif state.needs_more_info:
            return "follow_up"
        return "generate_response"

    def _parse_query(self, state: AgentState) -> AgentState:
        """Use LLM to determine which MCPs and actions are needed"""
        logger.info(
            "Parsing user query: %s",
            state.query[:100] + "..." if len(state.query) > 100 else state.query,
        )

        # available_mcps = self.mcp_registry.list_mcps()
        # mcp_details = []
        #
        # for mcp_name in available_mcps:
        #     mcp = self.mcp_registry.get_mcp(mcp_name)
        #     if mcp:
        #         mcp_details.append(
        #             f"{mcp.name}: {mcp.description}. Supported actions: {', '.join(mcp.supported_actions)}"
        #         )
        #
        # mcp_info = "\n".join(mcp_details)
        # logger.debug("Available MCPs for planning: %s", ", ".join(available_mcps))

        # Discover tools via the MCP Client
        mcp_info = "Tool Server: 'postgres' (Provides read-only PostgreSQL access)\n"
        try:
            # Dynamically list tools from the server
            tools_response = self.mcp_client.list_tools()
            tools = tools_response.get("tools", [])
            if tools:
                mcp_info += "Available Tools:\n"
                for tool in tools:
                    schema_desc = (
                        f" Requires input: {json.dumps(tool.get('inputSchema', {}))}"
                        if tool.get("inputSchema")
                        else ""
                    )
                    mcp_info += f"- Name: {tool.get('name', 'N/A')}, Description: {tool.get('description', 'N/A')}{schema_desc}\n"

            else:
                mcp_info += "No tools currently listed by the server.\n"

            logger.debug("Fetched tools description via MCPClient")

        except Exception as e:
            logger.error(
                f"Failed to list tools via MCPClient: {e}", exc_info=self.debug
            )
            mcp_info += "Error: Could not retrieve tools from the server.\n"

            # Optionally surface this error in the agent state
            state.error = f"Failed to contact tool server: {e}"
            state.planned_actions = []  # Cannot plan without tool info
            return state

        # Discover resources (schemas) if needed - or let LLM request them
        # Option 1: Proactive Resource Listing (might overload prompt)
        try:
            resources_response = self.mcp_client.list_resources()
            resources = resources_response.get("resources", [])
            if resources:
                mcp_info += (
                    "\nAvailable Data Resources (Schemas - use readResource to view):\n"
                )
                for res in resources[:10]:  # Limit for prompt length
                    mcp_info += (
                        f"- {res.get('name', res.get('uri'))} (URI: {res.get('uri')})\n"
                    )
                if len(resources) > 10:
                    mcp_info += "- ... and more\n"
        except Exception as e:
            logger.warning(f"Could not list resources: {e}")
            mcp_info += "\nWarning: Could not list available data resources."

        prompt = f"""
        You need to analyze the user's query and determine which actions to perform using the available tool server.
        You can interact with the 'postgres' tool server using these MCP methods:
        - `mcp_listResources`: Lists available data resources (like table schemas). Returns a list of {{uri, name, mimeType}}.
        - `mcp_readResource`: Reads the content of a specific resource URI. Requires 'uri' parameter. Returns {{contents: [{{uri, text, mimeType}}]}}. Use this to get schema details.
        - `mcp_listTools`: Lists available tools and their input requirements.
        - `mcp_callTool`: Executes a specific tool. Requires 'name' and 'arguments' parameters.

        Tool Server Information:
        {mcp_info}

        User query: {state.query}

        Think step-by-step. If the query requires database schema knowledge you don't have, first plan an `mcp_readResource` action to fetch the schema for the relevant table(s). Then, plan the `mcp_callTool` action for the 'query' tool using the schema information.

        Format your response as JSON:
        {{
            "thought": "your reasoning",
            "actions": [
                {{
                    "mcp_method": "mcp_readResource", // or mcp_callTool, etc.
                    "params": {{ // Parameters for the MCP method
                        "uri": "postgres://..." // for readResource
                        // OR
                        "name": "query",       // for callTool
                        "arguments": {{ "sql": "SELECT ..." }} // for callTool
                    }}
                }}
                // ... more actions if needed
            ]
        }}
        If no actions are needed, use "actions": [].
        If more info needed, use "needs_more_info": true, "follow_up_question": "..."
        """

        logger.debug("Sending query to LLM for planning with MCP methods")
        response = self.llm.generate(prompt)
        parsed_response = self._parse_json_response(response)
        logger.debug("Received and parsed LLM planning response")

        thought = parsed_response.get("thought", "No thought process provided")
        state.thoughts.append(thought)
        logger.info(
            "Agent thought: %s",
            thought[:100] + "..." if len(thought) > 100 else thought,
        )

        # Process actions
        if "actions" in parsed_response:
            state.planned_actions = parsed_response.get("actions", [])
            state.current_action_index = 0
            action_count = len(state.planned_actions)
            logger.info("Planned %d actions", action_count)
            for i, action in enumerate(state.planned_actions):
                logger.debug(
                    "Action %d: %s - %s",
                    i + 1,
                    action.get("mcp", "unknown"),
                    action.get("action", {}).get("operation", "unknown"),
                )

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
        mcp_method = action_spec.get("mcp_method")
        params = action_spec.get("params", {})
        action_label = f"{mcp_method}({json.dumps(params, default=str)[:50]}...)"

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
            # Use the MCPClient to execute the action
            if mcp_method == "mcp_listResources":
                result = self.mcp_client.list_resources()
            elif mcp_method == "mcp_readResource":
                if "uri" not in params:
                    raise ValueError("Missing 'uri' parameter for readResource")
                result = self.mcp_client.read_resource(uri=params["uri"])
            elif mcp_method == "mcp_listTools":
                result = self.mcp_client.list_tools()
            elif mcp_method == "mcp_callTool":
                if "name" not in params:
                    raise ValueError("Missing 'name' parameter for callTool")
                result = self.mcp_client.call_tool(
                    name=params["name"], arguments=params.get("arguments")
                )
            else:
                raise ValueError(f"Unsupported MCP method in plan: {mcp_method}")

            logger.debug("MCP action execution successful via MCPClient")
            state.thoughts.append(f"Action result: Successfully executed {mcp_method}")
            # Store result with a unique key per action
            state.mcp_results[f"action_{state.current_action_index}_{mcp_method}"] = (
                result
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Error executing MCP action %s via MCPClient: %s",
                mcp_method,
                error_msg,
                exc_info=self.debug,
            )
            state.thoughts.append(f"Error executing {mcp_method}: {error_msg}")
            state.mcp_results[f"action_{state.current_action_index}_{mcp_method}"] = {
                "error": error_msg
            }
            # Decide if agent should stop on error or try to continue/report

        state.current_action_index += 1
        return state

    def _plan_follow_up(self, state: AgentState) -> AgentState:
        """Plan follow-up actions based on execution results"""
        prompt = f"""
        Review the results of the actions taken so far and determine if additional actions are needed.
        
        User query: {state.query}
        
        Actions executed and their results:
        {self._format_results(state.mcp_results)}
        
        Based on these results, determine what to do next.
        Format your response as JSON with the structure:
        {{
            "thought": "your reasoning here",
            "actions": [
                {{
                    "mcp": "mcp_name",
                    "action": {{
                        "operation": "operation_name",
                        "parameters": {{ param details }}
                    }}
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

        thought = parsed_response.get(
            "thought", "Determining if additional actions are needed"
        )
        state.thoughts.append(thought)

        # Update actions list
        new_actions = parsed_response.get("actions", [])
        if new_actions:
            state.planned_actions = new_actions
            state.current_action_index = 0
            state.thoughts.append(
                f"Planning additional actions: {len(new_actions)} actions"
            )
        else:
            state.planned_actions = []
            state.thoughts.append("No additional actions needed")

        # Reset needs_more_info flag
        state.needs_more_info = False

        return state

    def _generate_response(self, state: AgentState) -> AgentState:
        logger.info("Generating final response based on MCP action results")
        formatted_results = self._format_results(
            state.mcp_results
        )  # Assumes this formats dicts well

        # Prompt needs to understand the structure of MCP responses
        prompt = f"""
        Based on the user's query and the results from interacting with the MCP tool server, generate a helpful response.

        User query: {state.query}

        MCP Interaction Results:
        {formatted_results}

        Remember:
        - `mcp_listResources` results contain a list of resources under the 'resources' key.
        - `mcp_readResource` results contain file content under the 'contents' key (usually a list with one item). The 'text' field holds the actual data (e.g., JSON schema).
        - `mcp_listTools` results contain tool definitions under the 'tools' key.
        - `mcp_callTool` results are under the 'content' key (usually a list with one text item). The 'text' field holds the tool output (e.g., JSON query results). Check the 'isError' flag.

        Synthesize these results into a clear answer for the user. Explain any errors encountered.
        """

        logger.debug("Sending response generation prompt to LLM")
        response = self.llm.generate(prompt)
        state.response = response
        logger.info("Response generated (length: %d characters)", len(response))
        return state

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format MCP results for LLM prompt"""
        try:
            return json.dumps(results, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error formatting results for prompt: {e}")
            return f"Error formatting results: {e}\nRaw results: {str(results)}"

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        import re

        # try:
        #     # Extract JSON if it's within a code block
        #     if "```json" in response:
        #         start = response.find("```json") + 7
        #         end = response.find("```", start)
        #         json_str = response[start:end].strip()
        #     elif "```" in response:
        #         # Try to find any code block
        #         start = response.find("```") + 3
        #         # Check if there's a language specifier
        #         if response[start : start + 10].split("\n")[0].strip():
        #             start = response.find("\n", start) + 1
        #         end = response.find("```", start)
        #         json_str = response[start:end].strip()
        #     else:
        #         # Try to find JSON-like content
        #         match = re.search(r"(\{.*\})", response, re.DOTALL)
        #         json_str = match.group(0) if match else response
        #
        #     return json.loads(json_str)
        # except Exception as e:
        #     return {"thought": f"Failed to parse LLM response: {str(e)}", "actions": []}
        try:
            # Extract JSON block if present
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
            if match:
                json_str = match.group(1)
            else:
                # Fallback: Look for the outermost curly braces
                first_brace = response.find("{")
                last_brace = response.rfind("}")
                if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                    json_str = response[first_brace : last_brace + 1]
                else:
                    # Maybe the whole response is JSON
                    json_str = response

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response from LLM: {e}\nResponse was:\n{response}",
                exc_info=self.debug,
            )
            return {
                "thought": f"Error: LLM response was not valid JSON. {e}",
                "actions": [],
            }
        except Exception as e:
            logger.error(
                f"Unexpected error parsing LLM response: {e}", exc_info=self.debug
            )
            return {
                "thought": f"Error: Unexpected error parsing LLM response. {e}",
                "actions": [],
            }

    # def execute_mcp_action(self, mcp_name: str, action: Dict[str, Any]) -> Any:
    #     """Execute a single MCP action directly, bypassing the graph
    #
    #     This is useful for fetching information needed before running the main agent workflow.
    #     """
    #     logger.info(
    #         "Direct execution of MCP %s action: %s", mcp_name, action.get("operation", "unknown")
    #     )
    #
    #     mcp = self.mcp_registry.get_mcp(mcp_name)
    #     if not mcp:
    #         logger.error("MCP '%s' not found for direct execution", mcp_name)
    #         raise ValueError(f"MCP '{mcp_name}' not found")
    #
    #     try:
    #         logger.debug("Executing MCP %s with action: %s", mcp_name, action)
    #         result = mcp.execute(action)
    #         logger.debug("Direct MCP execution successful")
    #
    #         # Log result summary
    #         if isinstance(result, dict):
    #             if "error" in result:
    #                 logger.info("MCP execution returned error: %s", result["error"])
    #             else:
    #                 keys = list(result.keys())
    #                 logger.info("MCP execution returned result with keys: %s", keys)
    #         elif isinstance(result, list):
    #             logger.info("MCP execution returned list with %d items", len(result))
    #         else:
    #             logger.info("MCP execution returned result of type: %s", type(result).__name__)
    #
    #         return result
    #     except Exception as e:
    #         logger.error(
    #             "Error in direct execution of %s action: %s", mcp_name, str(e), exc_info=self.debug
    #         )
    #         raise RuntimeError(f"Error executing {mcp_name} action: {str(e)}")

    def run(
        self, query: str, mcp_options: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Execute the complete agent workflow"""
        logger.info("Starting agent workflow execution")
        logger.info("Query: %s", query[:100] + "..." if len(query) > 100 else query)

        initial_state = AgentState(query=query)

        logger.debug("Invoking agent workflow graph")
        result = self.graph.invoke(initial_state)
        logger.debug("Agent workflow graph execution completed")

        # If result is already an AgentState, return it directly
        if isinstance(result, AgentState):
            logger.debug("Result is AgentState, returning directly")
            return result

        # Handle different return types from newer LangGraph versions
        if isinstance(result, dict):
            logger.debug("Result is dictionary, extracting state")
            # Check if state is directly in the dictionary
            if "state" in result:
                logger.debug("Found 'state' in result dictionary")
                return result["state"]

            # If there's a response, create a new state with it
            if "response" in result:
                logger.debug("Found 'response' in result dictionary")
                new_state = AgentState(query=query)
                new_state.response = result["response"]
                return new_state

        # If we couldn't convert it, create a fallback state
        logger.warning("Could not extract state from result, creating fallback")
        fallback_state = AgentState(query=query)
        fallback_state.response = "Could not extract a response from the agent result"
        return fallback_state
