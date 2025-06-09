import asyncio
import datetime
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from confidentialmind_core.config_manager import load_environment
from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.shared.logging.config import get_logger
from src.shared.logging.trace_context import TraceContext

from .api_client import APIConnectionError, APIError, BaseRAGClient
from .connection_manager import ConnectionManager
from .settings import settings

logger = logging.getLogger(__name__)
structlog_logger = get_logger("baserag.mcp")


def extract_trace_headers(ctx: Context) -> Dict[str, str]:
    """Extract trace headers from FastMCP context."""
    trace_headers = {}
    if hasattr(ctx, "request_context") and hasattr(ctx.request_context, "headers"):
        headers = ctx.request_context.headers
        trace_headers = {
            "trace_id": headers.get("X-Trace-ID"),
            "span_id": headers.get("X-Span-ID"),
            "parent_span_id": headers.get("X-Parent-Span-ID"),
            "session_id": headers.get("X-Session-ID"),
            "origin_service": headers.get("X-Origin-Service"),
        }
    return trace_headers


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Handles API connection setup and teardown with ConfidentialMind support.

    This function allows the server to start without an API connection,
    which is required for stack deployment scenarios where the API
    might be connected after the server is running.
    """
    state = {"api_connected": False}

    try:
        # Initialize environment
        load_environment()

        # Initialize ConnectionManager (this will handle all connector registration and URL polling)
        try:
            # Run with timeout protection
            await asyncio.wait_for(ConnectionManager.initialize(), timeout=5.0)
            logger.info("BaseRAG API connection manager initialized")
        except asyncio.TimeoutError:
            logger.warning(
                "ConnectionManager initialization timed out, continuing without API connection"
            )
        except Exception as e:
            logger.error(f"Error initializing connection manager: {e}")

        # Create API session, handling failures appropriately
        try:
            # Non-blocking session creation attempt
            session = await asyncio.wait_for(ConnectionManager.create_session(), timeout=10.0)
            if session:
                state["api_connected"] = True
                logger.info("BaseRAG API session created and stored in application state")
            else:
                logger.info("No BaseRAG API connection available yet. Server will start anyway.")
                state["api_connected"] = False
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Starting without BaseRAG API connection: {e}")
            state["api_connected"] = False

        # Quickly yield control back to FastMCP
        yield state

    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        logger.info(f"Server startup interrupted: {type(e).__name__}")
        raise
    finally:
        if ConnectionManager.is_connected():
            logger.info("Shutting down server, closing BaseRAG API session")
            try:
                await ConnectionManager.close()
                logger.info("BaseRAG API session closed")
            except Exception as e:
                logger.error(f"Error closing BaseRAG API session: {e}")


# Instantiate the FastMCP server with the lifespan manager
mcp_server = FastMCP(
    name="BaseRAG MCP Server",
    instructions="Provides access to BaseRAG API for knowledge-base context and chat completions with RAG.",
    lifespan=lifespan,
)


@mcp_server.resource("baserag://resources")
async def get_baserag_resources(ctx: Context) -> Dict[str, Any]:
    """
    Retrieves information about the available BaseRAG resources.
    This resource provides metadata about the BaseRAG APIs exposed through this MCP server.
    """
    if not ctx.request_context.lifespan_context.get("api_connected", False):
        logger.warning("BaseRAG API not connected when retrieving resources.")
        return {
            "status": "unavailable",
            "message": "BaseRAG API connection is not available yet. Please try again later.",
            "endpoints": [
                {
                    "name": "Chat Completions",
                    "path": "/v1/chat/completions",
                    "method": "POST",
                    "status": "unavailable",
                },
                {
                    "name": "Context Retrieval",
                    "path": "/context",
                    "method": "POST",
                    "status": "unavailable",
                },
                {
                    "name": "Content Retrieval",
                    "path": "/content/{content_id}",
                    "method": "GET",
                    "status": "unavailable",
                },
                {
                    "name": "Content Chunks",
                    "path": "/content/{content_id}/chunks",
                    "method": "GET",
                    "status": "unavailable",
                },
            ],
        }

    try:
        # Check health to verify connection
        is_healthy = await BaseRAGClient.check_health()

        status = "available" if is_healthy else "degraded"

        return {
            "status": status,
            "message": "BaseRAG API is available"
            if is_healthy
            else "BaseRAG API is experiencing issues",
            "endpoints": [
                {
                    "name": "Chat Completions",
                    "path": "/v1/chat/completions",
                    "method": "POST",
                    "status": status,
                },
                {
                    "name": "Context Retrieval",
                    "path": "/context",
                    "method": "POST",
                    "status": status,
                },
                {
                    "name": "Content Retrieval",
                    "path": "/content/{content_id}",
                    "method": "GET",
                    "status": status,
                },
                {
                    "name": "Content Chunks",
                    "path": "/content/{content_id}/chunks",
                    "method": "GET",
                    "status": status,
                },
            ],
            "base_url": ConnectionManager.get_base_url(),
        }
    except Exception as e:
        logger.error(f"Failed to get BaseRAG resources: {e}")
        return {
            "status": "error",
            "message": f"Error accessing BaseRAG API: {str(e)}",
            "endpoints": [],
        }


@mcp_server.tool()
async def get_context(query: str, ctx: Context) -> Dict[str, Any]:
    """
    Retrieve relevant context for a query from the knowledge base.

    This tool fetches context for a given query using the BaseRAG's retriever.

    Args:
        query: The query to retrieve context for.
        ctx: The MCP context (automatically injected).

    Returns:
        A dictionary containing the contexts retrieved, including document sources,
        content, and relevance scores.

    Example response:
    {
        "contexts": [
            {
                "content": "Context text...",
                "document": {
                    "id": "doc123",
                    "metadata": {"title": "Document Title", "author": "Author Name"}
                },
                "score": 0.92
            },
            ...
        ]
    }
    """
    # Extract trace headers
    trace_headers = extract_trace_headers(ctx)

    # Initialize trace context from headers if available
    if trace_headers.get("trace_id"):
        TraceContext.initialize(
            session_id=trace_headers.get("session_id", "unknown"),
            trace_id=trace_headers.get("trace_id"),
            origin_service=trace_headers.get("origin_service", "unknown"),
            parent_span_id=trace_headers.get("span_id"),
        )

    structlog_logger.info(
        "Starting context retrieval",
        event_type="baserag.context.start",
        data={
            "query_length": len(query),
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
        },
    )

    start_time = time.time()

    if not ctx.request_context.lifespan_context.get("api_connected", False):
        structlog_logger.info(
            "BaseRAG API not connected",
            event_type="baserag.context.no_connection",
            data={"error": "BaseRAG API not connected"},
        )
        logger.error("BaseRAG API not connected when retrieving context.")
        raise RuntimeError("BaseRAG API connection is not available yet. Please try again later.")

    try:
        result = await BaseRAGClient.get_context(query)

        duration_ms = (time.time() - start_time) * 1000
        structlog_logger.info(
            "Context retrieval completed",
            event_type="baserag.context.complete",
            duration_ms=duration_ms,
            success=True,
            data={"result_count": len(result.get("contexts", []))},
        )

        await ctx.info(f"Retrieved context for query: {query}")
        return result

    except APIConnectionError as e:
        duration_ms = (time.time() - start_time) * 1000
        structlog_logger.error(
            "API connection error during context retrieval",
            event_type="baserag.context.error",
            duration_ms=duration_ms,
            success=False,
            error=str(e),
            error_type="APIConnectionError",
        )
        logger.error(f"API connection error while retrieving context: {e}")
        raise RuntimeError(f"BaseRAG API connection error: {str(e)}")

    except APIError as e:
        duration_ms = (time.time() - start_time) * 1000
        structlog_logger.error(
            "API error during context retrieval",
            event_type="baserag.context.error",
            duration_ms=duration_ms,
            success=False,
            error=str(e),
            error_type="APIError",
        )
        logger.error(f"API error while retrieving context: {e}")
        raise RuntimeError(f"BaseRAG API error: {str(e)}")

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        structlog_logger.error(
            "Unexpected error during context retrieval",
            event_type="baserag.context.error",
            duration_ms=duration_ms,
            success=False,
            error=str(e),
            error_type=type(e).__name__,
        )
        logger.error(f"Unexpected error while retrieving context: {e}")
        raise RuntimeError(f"Error retrieving context: {str(e)}")


@mcp_server.tool()
async def get_content(content_id: str, ctx: Context) -> Dict[str, Any]:
    """
    Retrieve content by its ID from the knowledge base.

    This tool fetches the full content of a document stored in the BaseRAG system.

    Args:
        content_id: The unique identifier of the content to retrieve.
        ctx: The MCP context (automatically injected).

    Returns:
        A dictionary containing the content details, including text, metadata, and creation info.

    Example response:
    {
        "id": "content123",
        "text": "Full document text...",
        "metadata": {
            "title": "Document Title",
            "author": "Author Name",
            "created_at": "2023-01-01T00:00:00Z"
        }
    }
    """
    if not ctx.request_context.lifespan_context.get("api_connected", False):
        logger.error("BaseRAG API not connected when retrieving content.")
        raise RuntimeError("BaseRAG API connection is not available yet. Please try again later.")

    try:
        result = await BaseRAGClient.get_content(content_id)
        await ctx.info(f"Retrieved content with ID: {content_id}")
        return result
    except APIConnectionError as e:
        logger.error(f"API connection error while retrieving content: {e}")
        raise RuntimeError(f"BaseRAG API connection error: {str(e)}")
    except APIError as e:
        logger.error(f"API error while retrieving content: {e}")
        raise RuntimeError(f"BaseRAG API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while retrieving content: {e}")
        raise RuntimeError(f"Error retrieving content: {str(e)}")


@mcp_server.tool()
async def get_content_chunks(content_id: str, ctx: Context) -> Dict[str, Any]:
    """
    Retrieve the individual chunks of a content by its ID.

    This tool fetches the chunks that a document has been split into for retrieval purposes.

    Args:
        content_id: The unique identifier of the content to retrieve chunks for.
        ctx: The MCP context (automatically injected).

    Returns:
        A dictionary containing the content's chunks with their individual embedding info.

    Example response:
    {
        "content_id": "content123",
        "chunks": [
            {
                "id": "chunk1",
                "text": "Chunk text...",
                "metadata": {"position": 1, "embedding_model": "model-name"}
            },
            ...
        ]
    }
    """
    if not ctx.request_context.lifespan_context.get("api_connected", False):
        logger.error("BaseRAG API not connected when retrieving content chunks.")
        raise RuntimeError("BaseRAG API connection is not available yet. Please try again later.")

    try:
        result = await BaseRAGClient.get_content_chunks(content_id)
        await ctx.info(f"Retrieved chunks for content with ID: {content_id}")
        return result
    except APIConnectionError as e:
        logger.error(f"API connection error while retrieving content chunks: {e}")
        raise RuntimeError(f"BaseRAG API connection error: {str(e)}")
    except APIError as e:
        logger.error(f"API error while retrieving content chunks: {e}")
        raise RuntimeError(f"BaseRAG API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while retrieving content chunks: {e}")
        raise RuntimeError(f"Error retrieving content chunks: {str(e)}")


@mcp_server.tool()
async def chat_completion(
    ctx: Context,
    messages: List[Dict[str, Any]],
    model: Optional[str] = "cm-llm",
    temperature: Optional[float] = 0.7,
    stream: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Generate a chat completion using BaseRAG's retrieval-augmented generation.

    This tool calls the BaseRAG API to generate a response to the conversation,
    using the knowledge base for context.

    Args:
        ctx: The MCP context (automatically injected).
        messages: List of message objects with 'role' and 'content' fields.
        model: The model to use for completion (default: "cm-llm").
        temperature: Controls randomness (0-1, default: 0.7).
        stream: Whether to stream the response (not currently supported, default: False).

    Returns:
        A dictionary containing the completion response with model-generated content.

    Example message input:
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the benefits of RAG?"}
    ]

    Example response:
    {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "cm-llm",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Response text..."
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 56,
            "completion_tokens": 31,
            "total_tokens": 87
        }
    }
    """
    # Extract trace headers
    trace_headers = extract_trace_headers(ctx)

    # Initialize trace context from headers if available
    if trace_headers.get("trace_id"):
        TraceContext.initialize(
            session_id=trace_headers.get("session_id", "unknown"),
            trace_id=trace_headers.get("trace_id"),
            origin_service=trace_headers.get("origin_service", "unknown"),
            parent_span_id=trace_headers.get("span_id"),
        )

    structlog_logger.info(
        "Starting chat completion",
        event_type="baserag.chat.start",
        data={
            "model": model,
            "temperature": temperature,
            "message_count": len(messages),
            "stream": stream,
        },
    )

    start_time = time.time()

    if not ctx.request_context.lifespan_context.get("api_connected", False):
        logger.error("BaseRAG API not connected when generating chat completion.")
        raise RuntimeError("BaseRAG API connection is not available yet. Please try again later.")

    # Validate messages
    if not messages or not isinstance(messages, list):
        raise ValueError("Messages must be a non-empty list")

    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError("Each message must have 'role' and 'content' fields")

    # Stream is not supported in MCP
    if stream:
        await ctx.warning("Streaming is not supported in MCP. Setting stream=False.")
        stream = False

    try:
        result = await BaseRAGClient.chat_completion(
            messages=messages, model=model, temperature=temperature
        )

        # Log the first part of the response for debugging
        user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
        response_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        truncated_response = (
            response_content[:100] + "..." if len(response_content) > 100 else response_content
        )

        duration_ms = (time.time() - start_time) * 1000
        structlog_logger.info(
            "Chat completion completed",
            event_type="baserag.chat.complete",
            duration_ms=duration_ms,
            success=True,
            data={
                "user_query_preview": user_message[:50] + "..."
                if len(user_message) > 50
                else user_message,
                "response_preview": truncated_response,
            },
        )

        await ctx.info(
            f"Generated completion for: '{user_message[:50]}...' → '{truncated_response}'"
        )

        return result
    except APIConnectionError as e:
        duration_ms = (time.time() - start_time) * 1000
        structlog_logger.error(
            "API connection error during chat completion",
            event_type="baserag.chat.error",
            duration_ms=duration_ms,
            success=False,
            error=str(e),
            error_type="APIConnectionError",
        )
        logger.error(f"API connection error while generating chat completion: {e}")
        raise RuntimeError(f"BaseRAG API connection error: {str(e)}")
    except APIError as e:
        duration_ms = (time.time() - start_time) * 1000
        structlog_logger.error(
            "API error during chat completion",
            event_type="baserag.chat.error",
            duration_ms=duration_ms,
            success=False,
            error=str(e),
            error_type="APIError",
        )
        logger.error(f"API error while generating chat completion: {e}")
        raise RuntimeError(f"BaseRAG API error: {str(e)}")
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        structlog_logger.error(
            "Unexpected error during chat completion",
            event_type="baserag.chat.error",
            duration_ms=duration_ms,
            success=False,
            error=str(e),
            error_type=type(e).__name__,
        )
        logger.error(f"Unexpected error while generating chat completion: {e}")
        raise RuntimeError(f"Error generating chat completion: {str(e)}")


@mcp_server.custom_route("/health", methods=["GET"])
async def http_health_endpoint(request: Request) -> JSONResponse:
    """
    HTTP health endpoint for Kubernetes liveness and readiness probes.

    This endpoint always returns a 200 status code when the server is running,
    regardless of API connection status, allowing the pod to stay alive
    while waiting for the API to be configured.

    Returns:
        JSONResponse with health status information.
    """
    is_api_connected = ConnectionManager.is_connected()
    api_error = None

    if is_api_connected:
        try:
            is_healthy = await BaseRAGClient.check_health()
            if not is_healthy:
                api_error = "BaseRAG API health check failed"
        except Exception as e:
            logger.error(f"Error checking BaseRAG API health: {e}")
            api_error = str(e)
            is_api_connected = False
    else:
        api_error = ConnectionManager.last_error() or "BaseRAG API not configured"

    # Create health status response
    health_status = {
        "status": "healthy",
        "service": "baserag-mcp-server",
        "api_connected": is_api_connected,
        "api_error": api_error if not is_api_connected else None,
        "server_mode": "stack_deployment" if settings.is_stack_deployment else "local",
        "server_time": datetime.datetime.now().isoformat(),
        "connector_id": settings.connector_id,
    }

    # Always return a 200 status code to keep the pod alive
    return JSONResponse(content=health_status)
