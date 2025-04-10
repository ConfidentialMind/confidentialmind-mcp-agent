import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

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
    ResourceIdentifier,
    TextContent,
    ToolDefinition,
    ToolInputSchema,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [Obsidian MCP Server] %(levelname)s: %(message)s"
)
logger = logging.getLogger("obsidian-mcp")

# Create FastAPI app
app = FastAPI(title="Obsidian MCP Server")


# Obsidian handler class
class ObsidianHandler:
    """Handler for Obsidian vault MCP requests"""

    def __init__(self, vault_path: str):
        """Initialize with path to Obsidian vault"""
        self.vault_path = os.path.abspath(vault_path)
        self.vault_name = os.path.basename(self.vault_path)

        logging.info(
            f"Initialized ObsidianHandler for vault: {self.vault_name} at {self.vault_path}"
        )

        # Validate that the vault path exists and is a directory
        if not os.path.isdir(self.vault_path):
            error_msg = f"Vault path {self.vault_path} is not a valid directory"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Store found notes metadata for quick access
        self.notes_cache = {}
        self._cache_notes_metadata()

    def _cache_notes_metadata(self):
        """Cache notes metadata for quick access"""
        logging.info("Building notes metadata cache...")
        count = 0

        try:
            # Recursively walk the vault directory
            for root, _, files in os.walk(self.vault_path):
                for file in files:
                    if file.endswith(".md"):
                        # Get the full path to the file
                        file_path = os.path.join(root, file)

                        # Create the relative path from the vault root
                        rel_path = os.path.relpath(file_path, self.vault_path)

                        # Create the URI in the format obsidian://<vault_name>/<relative_path>
                        uri = f"obsidian://{self.vault_name}/{rel_path.replace(os.sep, '/')}"

                        # Get the note name (filename without extension)
                        note_name = os.path.splitext(os.path.basename(file))[0]

                        # Get file stats
                        stats = os.stat(file_path)

                        # Store metadata in cache
                        self.notes_cache[uri] = {
                            "name": note_name,
                            "path": file_path,
                            "rel_path": rel_path,
                            "size": stats.st_size,
                            "mtime": stats.st_mtime,
                        }
                        count += 1

            logging.info(f"Built cache with metadata for {count} notes")
        except Exception as e:
            logging.error(f"Error building notes cache: {e}", exc_info=True)
            # Continue with empty cache

    def handle_list_resources(self) -> ListResourcesResponse:
        """List all markdown files in the vault as resources"""
        logging.info("Handling mcp_listResources")

        resources = []

        try:
            # Use the cache if available
            if self.notes_cache:
                for uri, metadata in self.notes_cache.items():
                    resources.append(
                        ResourceIdentifier(uri=uri, name=metadata["name"], mimeType="text/markdown")
                    )
            else:
                # Fallback to scanning the vault directory if cache is empty
                for root, _, files in os.walk(self.vault_path):
                    for file in files:
                        if file.endswith(".md"):
                            # Get the full path to the file
                            file_path = os.path.join(root, file)

                            # Create the relative path from the vault root
                            rel_path = os.path.relpath(file_path, self.vault_path)

                            # Create the URI in the format obsidian://<vault_name>/<relative_path>
                            uri = f"obsidian://{self.vault_name}/{rel_path.replace(os.sep, '/')}"

                            # Get the note name (filename without extension)
                            note_name = os.path.splitext(os.path.basename(file))[0]

                            resources.append(
                                ResourceIdentifier(
                                    uri=uri, name=note_name, mimeType="text/markdown"
                                )
                            )

            logging.info(f"Found {len(resources)} markdown files in vault")
            return ListResourcesResponse(resources=resources)

        except Exception as e:
            logging.error(f"Error listing resources: {e}", exc_info=True)
            raise

    def handle_read_resource(self, params: ReadResourceRequestParams) -> ReadResourceResponse:
        """Read the content of a specific markdown file"""
        logging.info(f"Handling mcp_readResource for URI: {params.uri}")

        try:
            # Validate URI format and extract path
            file_path = self._uri_to_file_path(params.uri)

            # Read the file content
            content = self._read_file(file_path)

            return ReadResourceResponse(
                contents=[ResourceContent(uri=params.uri, text=content, mimeType="text/markdown")]
            )

        except Exception as e:
            logging.error(f"Error reading resource {params.uri}: {e}", exc_info=True)
            raise

    def handle_list_tools(self) -> ListToolsResponse:
        """Define and list available tools for working with the Obsidian vault"""
        logging.info("Handling mcp_listTools")

        return ListToolsResponse(
            tools=[
                ToolDefinition(
                    name="get_note_content",
                    description="Retrieves the content of a note by its URI.",
                    inputSchema=ToolInputSchema(
                        type="object",
                        properties={
                            "uri": {
                                "type": "string",
                                "description": "The URI of the note to retrieve (obsidian://<vault_name>/<path>).",
                            }
                        },
                        required=["uri"],
                    ),
                ),
                ToolDefinition(
                    name="get_multiple_notes",
                    description="Retrieves the content of multiple notes by their URIs.",
                    inputSchema=ToolInputSchema(
                        type="object",
                        properties={
                            "uris": {
                                "type": "array",
                                "description": "List of URIs to retrieve (each in format obsidian://<vault_name>/<path>).",
                            }
                        },
                        required=["uris"],
                    ),
                ),
                ToolDefinition(
                    name="search_notes",
                    description="Searches for notes containing the specified query text.",
                    inputSchema=ToolInputSchema(
                        type="object",
                        properties={
                            "query": {
                                "type": "string",
                                "description": "The search query text to find in notes.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of search results to return (default: 5).",
                            },
                        },
                        required=["query"],
                    ),
                ),
                ToolDefinition(
                    name="get_notes_by_topic",
                    description="Searches for notes on a specific topic and returns both URIs and content.",
                    inputSchema=ToolInputSchema(
                        type="object",
                        properties={
                            "topic": {
                                "type": "string",
                                "description": "The topic to search for in notes.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5).",
                            },
                        },
                        required=["topic"],
                    ),
                ),
                ToolDefinition(
                    name="summarize_notes",
                    description="Extracts key information from provided note contents.",
                    inputSchema=ToolInputSchema(
                        type="object",
                        properties={
                            "contents": {
                                "type": "array",
                                "description": "Array of note contents to summarize.",
                            },
                            "focus": {
                                "type": "string",
                                "description": "Optional focus area for the summary (e.g., 'meeting prep', 'project status').",
                            },
                        },
                        required=["contents"],
                    ),
                ),
                ToolDefinition(
                    name="get_vault_stats",
                    description="Returns statistics about the Obsidian vault.",
                    inputSchema=ToolInputSchema(type="object", properties={}, required=[]),
                ),
            ]
        )

    def handle_call_tool(self, params: CallToolRequestParams) -> CallToolResponse:
        """Handle tool calls by dispatching to the appropriate handler method"""
        logging.info(f"Handling mcp_callTool for tool: {params.name}")

        try:
            if params.name == "get_note_content":
                return self._handle_get_note_content(params.arguments or {})
            elif params.name == "get_multiple_notes":
                return self._handle_get_multiple_notes(params.arguments or {})
            elif params.name == "search_notes":
                return self._handle_search_notes(params.arguments or {})
            elif params.name == "get_notes_by_topic":
                return self._handle_get_notes_by_topic(params.arguments or {})
            elif params.name == "summarize_notes":
                return self._handle_summarize_notes(params.arguments or {})
            elif params.name == "get_vault_stats":
                return self._handle_get_vault_stats(params.arguments or {})
            else:
                error_msg = f"Unknown tool: {params.name}"
                logging.error(error_msg)

                # Provide information about available tools in the error message
                available_tools = [
                    "get_note_content",
                    "get_multiple_notes",
                    "search_notes",
                    "get_notes_by_topic",
                    "summarize_notes",
                    "get_vault_stats",
                ]
                tool_info = f"Available tools: {', '.join(available_tools)}"

                return CallToolResponse(
                    content=[TextContent(text=f"{error_msg}\n{tool_info}")], isError=True
                )

        except Exception as e:
            error_msg = f"Error executing tool {params.name}: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

    def _handle_get_note_content(self, arguments: Dict[str, Any]) -> CallToolResponse:
        """Get note content by URI"""
        if "uri" not in arguments:
            error_msg = "Missing 'uri' parameter for get_note_content tool"
            logging.error(error_msg)

            # Provide an example URI in the error message
            example_uri = (
                next(iter(self.notes_cache.keys()))
                if self.notes_cache
                else f"obsidian://{self.vault_name}/example.md"
            )
            help_msg = f"Example valid URI: {example_uri}"

            return CallToolResponse(
                content=[TextContent(text=f"{error_msg}\n{help_msg}")], isError=True
            )

        uri = arguments["uri"]

        try:
            # Convert URI to file path
            file_path = self._uri_to_file_path(uri)

            # Read the file content
            content = self._read_file(file_path)

            return CallToolResponse(content=[TextContent(text=content)], isError=False)

        except Exception as e:
            error_msg = f"Error retrieving note content for {uri}: {str(e)}"
            logging.error(error_msg, exc_info=True)

            # Provide more helpful error with example of a valid URI
            example_uri = (
                next(iter(self.notes_cache.keys()))
                if self.notes_cache
                else f"obsidian://{self.vault_name}/example.md"
            )
            help_msg = f"Example valid URI: {example_uri}"

            return CallToolResponse(
                content=[TextContent(text=f"{error_msg}\n{help_msg}")], isError=True
            )

    def _handle_get_multiple_notes(self, arguments: Dict[str, Any]) -> CallToolResponse:
        """Get content from multiple notes by URIs"""
        if "uris" not in arguments:
            error_msg = "Missing 'uris' parameter for get_multiple_notes tool"
            logging.error(error_msg)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

        uris = arguments["uris"]

        # Validate that uris is a list
        if not isinstance(uris, list):
            error_msg = f"'uris' parameter must be a list, got {type(uris).__name__}"
            logging.error(error_msg)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

        results = []
        errors = []

        for uri in uris:
            try:
                file_path = self._uri_to_file_path(uri)
                content = self._read_file(file_path)

                # Get note name from URI
                note_name = os.path.splitext(os.path.basename(file_path))[0]

                results.append({"uri": uri, "name": note_name, "content": content})
            except Exception as e:
                error_str = f"Error retrieving {uri}: {str(e)}"
                logging.warning(error_str)
                errors.append(error_str)

        # Prepare the response
        response = {"notes": results, "total": len(results)}

        if errors:
            response["errors"] = errors

        return CallToolResponse(
            content=[TextContent(text=json.dumps(response, indent=2))], isError=False
        )

    def _handle_search_notes(self, arguments: Dict[str, Any]) -> CallToolResponse:
        """Search for notes containing the query text"""
        if "query" not in arguments:
            error_msg = "Missing 'query' parameter for search_notes tool"
            logging.error(error_msg)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

        query = arguments["query"]
        max_results = int(arguments.get("max_results", 5))

        try:
            # Perform search
            search_results = self._search_vault(query, max_results)

            # Format results as JSON
            result_json = json.dumps(search_results, indent=2)

            return CallToolResponse(content=[TextContent(text=result_json)], isError=False)

        except Exception as e:
            error_msg = f"Error searching notes for '{query}': {str(e)}"
            logging.error(error_msg, exc_info=True)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

    def _handle_get_notes_by_topic(self, arguments: Dict[str, Any]) -> CallToolResponse:
        """Search for notes on a topic and return both URIs and contents"""
        if "topic" not in arguments:
            error_msg = "Missing 'topic' parameter for get_notes_by_topic tool"
            logging.error(error_msg)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

        topic = arguments["topic"]
        max_results = int(arguments.get("max_results", 5))

        try:
            # First, search for the notes
            search_results = self._search_vault(topic, max_results)

            if not search_results:
                return CallToolResponse(
                    content=[
                        TextContent(
                            text=json.dumps(
                                {
                                    "topic": topic,
                                    "notes": [],
                                    "message": f"No notes found for topic: {topic}",
                                },
                                indent=2,
                            )
                        )
                    ],
                    isError=False,
                )

            # Now get the full content for each note
            notes_with_content = []

            for result in search_results:
                uri = result["uri"]
                try:
                    file_path = self._uri_to_file_path(uri)
                    content = self._read_file(file_path)

                    notes_with_content.append(
                        {
                            "uri": uri,
                            "name": result["name"],
                            "content": content,
                            "context": result.get("context", ""),
                        }
                    )
                except Exception as e:
                    logging.warning(f"Error getting content for {uri}: {e}")
                    # Include the note with an error message instead of content
                    notes_with_content.append(
                        {
                            "uri": uri,
                            "name": result["name"],
                            "error": str(e),
                            "context": result.get("context", ""),
                        }
                    )

            # Prepare the response
            response = {
                "topic": topic,
                "notes": notes_with_content,
                "total": len(notes_with_content),
            }

            return CallToolResponse(
                content=[TextContent(text=json.dumps(response, indent=2))], isError=False
            )

        except Exception as e:
            error_msg = f"Error retrieving notes for topic '{topic}': {str(e)}"
            logging.error(error_msg, exc_info=True)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

    def _handle_summarize_notes(self, arguments: Dict[str, Any]) -> CallToolResponse:
        """Extract key information from provided note contents"""
        contents = arguments.get("contents")
        focus = arguments.get("focus", "")

        if not contents:
            # Check for older parameter name variations
            if "note_contents" in arguments:
                contents = arguments["note_contents"]
            else:
                error_msg = "Missing 'contents' parameter for summarize_notes tool"
                logging.error(error_msg)
                return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

        # Handle different input formats
        processed_contents = []

        if isinstance(contents, list):
            # If it's a list, process each item
            for item in contents:
                if isinstance(item, dict) and "content" in item:
                    # If it's a dictionary with content key
                    processed_contents.append(item["content"])
                else:
                    # Otherwise treat the item itself as content
                    processed_contents.append(str(item))
        elif isinstance(contents, dict):
            # If it's a dictionary, extract content values
            for key, value in contents.items():
                if isinstance(value, dict) and "content" in value:
                    processed_contents.append(value["content"])
                else:
                    processed_contents.append(str(value))
        else:
            # Otherwise, treat the entire contents as a single string
            processed_contents.append(str(contents))

        try:
            # This is where you would normally perform actual summarization
            # Since we don't have NLP capabilities in this script, we'll provide structured content
            # that the LLM agent can then summarize itself

            # Calculate some basic metrics
            total_chars = sum(len(content) for content in processed_contents)
            total_notes = len(processed_contents)

            # Extract potential key concepts (simple implementation)
            key_concepts = self._extract_key_concepts(processed_contents, focus)

            # Prepare response with the processed contents in a structured format
            response = {
                "notes_count": total_notes,
                "total_characters": total_chars,
                "focus_area": focus if focus else "general",
                "key_concepts": key_concepts,
                "contents": processed_contents,
            }

            return CallToolResponse(
                content=[TextContent(text=json.dumps(response, indent=2))], isError=False
            )

        except Exception as e:
            error_msg = f"Error summarizing notes: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

    def _handle_get_vault_stats(self, arguments: Dict[str, Any]) -> CallToolResponse:
        """Get statistics about the Obsidian vault"""
        try:
            # Get basic stats
            total_notes = len(self.notes_cache)

            # Count total size
            total_size = sum(meta["size"] for meta in self.notes_cache.values())

            # Find oldest and newest notes
            if total_notes > 0:
                oldest_note = min(self.notes_cache.items(), key=lambda x: x[1]["mtime"])
                newest_note = max(self.notes_cache.items(), key=lambda x: x[1]["mtime"])

                oldest_uri = oldest_note[0]
                oldest_name = oldest_note[1]["name"]
                newest_uri = newest_note[0]
                newest_name = newest_note[1]["name"]
            else:
                oldest_uri = ""
                oldest_name = ""
                newest_uri = ""
                newest_name = ""

            # Get folder structure
            folders = set()
            for meta in self.notes_cache.values():
                folder = os.path.dirname(meta["rel_path"])
                if folder:
                    folders.add(folder)

            # Prepare stats
            stats = {
                "vault_name": self.vault_name,
                "total_notes": total_notes,
                "total_size_bytes": total_size,
                "folders_count": len(folders),
                "oldest_note": {"uri": oldest_uri, "name": oldest_name},
                "newest_note": {"uri": newest_uri, "name": newest_name},
            }

            return CallToolResponse(
                content=[TextContent(text=json.dumps(stats, indent=2))], isError=False
            )

        except Exception as e:
            error_msg = f"Error getting vault stats: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return CallToolResponse(content=[TextContent(text=error_msg)], isError=True)

    def _extract_key_concepts(self, contents: List[str], focus: str = "") -> List[str]:
        """Extract key concepts from the contents based on word frequency"""
        # Combine all contents
        all_text = " ".join(contents).lower()

        # Split into words and remove common punctuation
        words = re.findall(r"\b\w+\b", all_text)

        # Count word frequency
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Skip very short words
                word_counts[word] = word_counts.get(word, 0) + 1

        # Find words that appear multiple times
        frequent_words = [(word, count) for word, count in word_counts.items() if count > 1]

        # Sort by frequency
        frequent_words.sort(key=lambda x: x[1], reverse=True)

        # If focus is provided, prioritize concepts related to the focus
        if focus:
            focus_related = []
            other_concepts = []

            focus_terms = set(re.findall(r"\b\w+\b", focus.lower()))

            for word, count in frequent_words[:20]:  # Look at top 20 words
                if word in focus_terms or any(term in word for term in focus_terms):
                    focus_related.append(word)
                else:
                    other_concepts.append(word)

            # Combine with focus-related terms first
            key_concepts = focus_related + other_concepts
        else:
            # Just take the most frequent words
            key_concepts = [word for word, _ in frequent_words]

        # Return top concepts (up to 10)
        return key_concepts[:10]

    def _uri_to_file_path(self, uri: str) -> str:
        """Convert an Obsidian URI to a file path, with security validation"""
        # Check cache first for quick lookup
        if uri in self.notes_cache:
            return self.notes_cache[uri]["path"]

        # Validate URI format
        if not uri.startswith(f"obsidian://{self.vault_name}/"):
            example_uri = (
                next(iter(self.notes_cache.keys()))
                if self.notes_cache
                else f"obsidian://{self.vault_name}/example.md"
            )
            raise ValueError(
                f"Invalid URI format: {uri}. Expected format: obsidian://{self.vault_name}/path/to/note.md\nExample URI: {example_uri}"
            )

        # Extract the relative path from the URI
        parsed_uri = urlparse(uri)

        # Get the path component (removing leading slash if present)
        path_part = parsed_uri.path.lstrip("/")

        # Remove the vault name from the beginning if present
        if path_part.startswith(f"{self.vault_name}/"):
            path_part = path_part[len(self.vault_name) + 1 :]

        # Construct the absolute file path
        file_path = os.path.normpath(os.path.join(self.vault_path, unquote(path_part)))

        # Security check: ensure the file path is within the vault directory
        if not file_path.startswith(self.vault_path):
            raise ValueError(f"Security violation: Path {file_path} is outside the vault directory")

        # Ensure the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Note file not found: {file_path}")

        return file_path

    def _read_file(self, file_path: str) -> str:
        """Read content from a file with error handling"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Error reading file {file_path}: {str(e)}")

    def _search_vault(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search the vault for notes containing the query"""
        search_results = []

        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()

        # If we have a cache, use it for faster searching
        if self.notes_cache:
            for uri, metadata in self.notes_cache.items():
                if len(search_results) >= max_results:
                    break

                try:
                    # Read the file content
                    with open(metadata["path"], "r", encoding="utf-8") as f:
                        content = f.read()

                    # Check if query exists in content (case-insensitive)
                    if query_lower in content.lower():
                        # Extract context around the match
                        context = self._extract_context(content, query_lower)

                        # Add to results
                        search_results.append(
                            {"uri": uri, "name": metadata["name"], "context": context}
                        )
                except Exception as e:
                    logging.warning(f"Error reading file during search {metadata['path']}: {e}")
                    continue
        else:
            # Fallback to directory traversal if cache is not available
            for root, _, files in os.walk(self.vault_path):
                for file in files:
                    if file.endswith(".md"):
                        # Get the full path to the file
                        file_path = os.path.join(root, file)

                        try:
                            # Read the file content
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()

                            # Check if query exists in content (case-insensitive)
                            if query_lower in content.lower():
                                # Get the relative path from the vault root
                                rel_path = os.path.relpath(file_path, self.vault_path)

                                # Create the URI
                                uri = (
                                    f"obsidian://{self.vault_name}/{rel_path.replace(os.sep, '/')}"
                                )

                                # Get the note name
                                note_name = os.path.splitext(os.path.basename(file))[0]

                                # Extract context around the match
                                context = self._extract_context(content, query_lower)

                                # Add to results
                                search_results.append(
                                    {"uri": uri, "name": note_name, "context": context}
                                )

                                # Check if we've reached the maximum results
                                if len(search_results) >= max_results:
                                    return search_results
                        except Exception as e:
                            logging.warning(f"Error reading file during search {file_path}: {e}")
                            continue

        return search_results

    def _extract_context(self, content: str, query_lower: str, context_size: int = 100) -> str:
        """Extract a snippet of text around the query match"""
        # Find the position of the match
        pos = content.lower().find(query_lower)

        if pos == -1:
            return ""

        # Calculate start and end positions for the context
        start = max(0, pos - context_size // 2)
        end = min(len(content), pos + len(query_lower) + context_size // 2)

        # Extract the context
        context = content[start:end]

        # Add ellipsis if we're not at the beginning or end
        if start > 0:
            context = "..." + context
        if end < len(content):
            context = context + "..."

        return context


# Request model for JSON-RPC requests
class MCPRequest(BaseModel):
    jsonrpc: str
    id: Any
    method: str
    params: Optional[Dict[str, Any]] = None


# Global Obsidian handler instance
obsidian_handler = None


@app.on_event("startup")
async def startup_event():
    global obsidian_handler

    # Get vault path from environment variable
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        raise RuntimeError("OBSIDIAN_VAULT_PATH environment variable is not set")

    # Initialize Obsidian handler
    try:
        obsidian_handler = ObsidianHandler(vault_path)
        logging.info("Obsidian handler initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Obsidian handler: {e}", exc_info=True)
        raise


@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    global obsidian_handler

    if not obsidian_handler:
        raise HTTPException(status_code=500, detail="Obsidian handler not initialized")

    logging.info(f"Received MCP request: method={request.method}, id={request.id}")

    response_data = None
    error_data = None

    try:
        if request.method == "mcp_listResources":
            response_data = obsidian_handler.handle_list_resources().model_dump()
        elif request.method == "mcp_readResource":
            params = ReadResourceRequestParams.model_validate(request.params or {})
            response_data = obsidian_handler.handle_read_resource(params).model_dump()
        elif request.method == "mcp_listTools":
            response_data = obsidian_handler.handle_list_tools().model_dump()
        elif request.method == "mcp_callTool":
            params = CallToolRequestParams.model_validate(request.params or {})
            response_data = obsidian_handler.handle_call_tool(params).model_dump()
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
    global obsidian_handler

    if not obsidian_handler:
        raise HTTPException(status_code=500, detail="Obsidian handler not initialized")

    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
