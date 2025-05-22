"""Test the BaseRAG MCP server functionality.

This test executes against a running BaseRAG MCP server to test its
functionality including tools and resources availability.

Before running this test, start the BaseRAG MCP server:
    python -m src.tools.baserag_mcp

Usage:
    python -m tests.test_baserag_mcp
"""

import asyncio
import json
import uuid
from typing import Optional, Tuple

import aiohttp
from fastmcp import Client
from fastmcp.exceptions import ClientError, ToolError
from mcp.types import TextContent, TextResourceContents

from .logger import logger


async def setup_test_environment() -> None:
    """Set up test environment for BaseRAG MCP server testing."""
    logger.info("Setting up test environment for BaseRAG MCP testing...")
    # Unlike PostgreSQL testing, we don't need to set up tables or data
    # BaseRAG should have existing data, or our tests will use sample queries
    # that work even without data


async def cleanup_test_environment() -> None:
    """Clean up test environment after testing is complete."""
    logger.info("Cleaning up test environment after BaseRAG MCP testing...")
    # No specific cleanup needed for BaseRAG MCP tests


async def test_server_connection(client: Client) -> bool:
    """Test connection to the BaseRAG MCP server."""
    try:
        # List available tools
        tools_response = await client.list_tools()
        logger.info("Available tools", tools=[tool.name for tool in tools_response])

        # Check if the expected tools are available
        expected_tools = ["get_context", "get_content", "get_content_chunks", "chat_completion"]
        available_tools = [tool.name for tool in tools_response]

        missing_tools = [tool for tool in expected_tools if tool not in available_tools]
        if missing_tools:
            logger.error("Missing expected tools", missing=missing_tools)
            return False

        logger.info("All expected tools are available")

        # List available resources
        resources_response = await client.list_resources()
        logger.info("Available resources", resources=[r.uri for r in resources_response])

        # Check if our expected resource is available
        expected_resource = "baserag://resources"
        available_resources_uris = [str(r.uri) for r in resources_response]

        if expected_resource not in available_resources_uris:
            logger.error("Expected resource not available", resource=expected_resource)
            return False

        logger.info("All expected resources are available")
        return True
    except Exception as e:
        logger.error("Error connecting to BaseRAG MCP server", error=str(e))
        return False


async def test_baserag_resources(client: Client) -> bool:
    """Test accessing the baserag://resources resource."""
    logger.info("Testing access to resource", resource="baserag://resources")
    try:
        schema_response_contents = await client.read_resource("baserag://resources")

        if schema_response_contents:
            content_item = schema_response_contents[0]
            # Check if it's TextResourceContents and has text
            if isinstance(content_item, TextResourceContents) and hasattr(content_item, "text"):
                try:
                    resources_data = json.loads(content_item.text)
                    logger.info(
                        "Successfully retrieved BaseRAG resources",
                        status=resources_data.get("status"),
                        endpoints_count=len(resources_data.get("endpoints", [])),
                    )

                    # Check if the response has the expected structure
                    if not all(key in resources_data for key in ["status", "endpoints"]):
                        logger.error("BaseRAG resources missing required fields")
                        return False

                    # Log the endpoints for visibility
                    for i, endpoint in enumerate(resources_data.get("endpoints", [])[:3]):
                        logger.info(
                            "Endpoint info",
                            name=endpoint.get("name"),
                            path=endpoint.get("path"),
                            status=endpoint.get("status"),
                        )

                    return True
                except json.JSONDecodeError:
                    logger.error(
                        "Error parsing BaseRAG resources JSON",
                        content_preview=content_item.text[:100],
                    )
                    return False
            else:
                logger.error(
                    "Received non-text or unexpected content for BaseRAG resources",
                    content_type=type(content_item),
                )
                return False
        else:
            logger.warning("BaseRAG resources returned no content")
            return False

    except Exception as e:
        logger.error("Error accessing BaseRAG resources", error=str(e))
        return False


async def test_get_context_tool(client: Client) -> Tuple[bool, Optional[str]]:
    """
    Test the get_context tool to retrieve relevant context for a query.

    Returns:
        Tuple of (success, content_id) where content_id can be used for subsequent tests
    """
    logger.info("Testing 'get_context' tool")
    try:
        # Use a query that should work even without specific data
        query = "What are the benefits of retrieval augmented generation?"

        query_result_content = await client.call_tool("get_context", {"query": query})

        if not query_result_content:
            logger.warning("get_context tool returned no content")
            return False, None

        content = query_result_content[0]
        if not isinstance(content, TextContent) or not hasattr(content, "text"):
            logger.error(
                "get_context tool returned unexpected content type", content_type=type(content)
            )
            return False, None

        try:
            context_data = json.loads(content.text)
            contexts = context_data.get("contexts", [])
            logger.info("Query executed successfully", context_count=len(contexts))

            # Extract a content ID for subsequent tests if available
            if context_data:
                content_id = context_data["files"][0]["id"]
                logger.info("Found content ID from context results", content_id=content_id)

            # Log some sample contexts for visibility
            for i, ctx in enumerate(contexts[:2]):
                preview = ctx.get("content", "")[:50] + (
                    "..." if len(ctx.get("content", "")) > 50 else ""
                )
                logger.info(
                    f"Context {i + 1}",
                    preview=preview,
                    score=ctx.get("score"),
                    document_id=ctx.get("document", {}).get("id") if "document" in ctx else None,
                )

            return True, content_id
        except json.JSONDecodeError:
            logger.error(
                "Error parsing get_context result JSON",
                content_preview=content.text[:100],
            )
            return False, None

    except ClientError as e:
        logger.error("Tool call 'get_context' failed", error=str(e))
        return False, None
    except Exception as e:
        logger.error("Error executing get_context tool", error=str(e))
        return False, None


async def test_get_content_tool(client: Client, content_id: Optional[str] = None) -> bool:
    """
    Test the get_content tool to retrieve a specific document by ID.

    Args:
        client: FastMCP client
        content_id: Content ID to retrieve, if None we'll try with a generic ID
    """
    logger.info("Testing 'get_content' tool")

    # If no content_id provided, use a test ID or try to get one
    if not content_id:
        logger.warning("No content ID provided, using a test ID that may not exist")
        # Generate a predictable test ID
        content_id = f"test-content-{uuid.uuid4()}"

    try:
        get_content_result = await client.call_tool("get_content", {"content_id": content_id})

        if not get_content_result:
            logger.warning("get_content tool returned no content")
            if not content_id.startswith("test-content-"):
                # Only return false if we were using a real content ID
                return False
            else:
                # For test IDs, we expect this might fail, so we'll consider it a "pass"
                logger.info("Test content ID not found, as expected")
                return True

        content = get_content_result[0]
        if not isinstance(content, TextContent) or not hasattr(content, "text"):
            logger.error(
                "get_content tool returned unexpected content type", content_type=type(content)
            )
            return False

        try:
            content_data = json.loads(content.text)

            # Check for expected fields
            if "id" not in content_data:
                logger.error("Content response missing required fields")
                return False

            # Log content details
            text_preview = content_data.get("text", "")[:50] + (
                "..." if len(content_data.get("text", "")) > 50 else ""
            )
            logger.info(
                "Content retrieved successfully",
                id=content_data.get("id"),
                text_preview=text_preview,
            )

            # Log metadata if available
            if "metadata" in content_data:
                logger.info("Content metadata", metadata=content_data.get("metadata"))

            return True
        except json.JSONDecodeError:
            logger.error(
                "Error parsing get_content result JSON",
                content_preview=content.text[:100],
            )
            return False

    except ClientError as e:
        logger.error("Tool call 'get_content' failed", error=str(e))
        if not content_id.startswith("test-content-"):
            # Only return false if we were using a real content ID
            return False
        else:
            # For test IDs, we expect this might fail, so we'll consider it a "pass"
            logger.info("Test content ID not found, as expected")
            return True
    except Exception as e:
        logger.error("Error executing get_content tool", error=str(e))
        return False


async def test_get_content_chunks_tool(client: Client, content_id: Optional[str] = None) -> bool:
    """
    Test the get_content_chunks tool API connectivity.

    This test only verifies that the API call can be made successfully,
    not whether actual content is retrieved.

    Args:
        client: FastMCP client
        content_id: Content ID to retrieve chunks for, if None we'll use a test ID

    Returns:
        bool: True if the API call was processed (even if no content found), False if API error
    """
    logger.info("Testing 'get_content_chunks' tool API connectivity")

    # If no content_id provided, use a test ID
    if not content_id:
        # Use a consistent test ID format
        content_id = f"test-content-{uuid.uuid4()}"
        logger.info(f"No content ID provided, using generated ID: {content_id}")

    try:
        # Make the API call
        await client.call_tool("get_content_chunks", {"content_id": content_id})

        # If we get here without an exception, the API call itself worked
        logger.info("get_content_chunks API call completed successfully")
        return True

    except ClientError as e:
        # Check if this is a "content not found" type of error, which is acceptable
        error_str = str(e).lower()
        if (
            "not found" in error_str
            or "does not exist" in error_str
            or "no such content" in error_str
        ):
            logger.info("Content not found error, but API connectivity confirmed", error=str(e))
            return True
        else:
            # This is a different type of API error
            logger.error("Tool call 'get_content_chunks' failed with API error", error=str(e))
            return False

    except Exception as e:
        # Unexpected exception indicates a test failure
        logger.error("Error executing get_content_chunks tool", error=str(e), exc_info=True)
        return False


async def test_chat_completion_tool(client: Client) -> bool:
    """Test the chat_completion tool with RAG capabilities."""
    logger.info("Testing 'chat_completion' tool")

    try:
        # Prepare a simple conversation for the chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the benefits of retrieval augmented generation?"},
        ]

        chat_completion_result = await client.call_tool(
            "chat_completion", {"messages": messages, "model": "cm-llm", "temperature": 0.7}
        )

        if not chat_completion_result:
            logger.warning("chat_completion tool returned no content")
            return False

        content = chat_completion_result[0]
        if not isinstance(content, TextContent) or not hasattr(content, "text"):
            logger.error(
                "chat_completion tool returned unexpected content type", content_type=type(content)
            )
            return False

        try:
            completion_data = json.loads(content.text)

            # Check for expected fields in a chat completion response
            if "choices" not in completion_data or not completion_data.get("choices"):
                logger.error("Chat completion response missing required fields")
                return False

            # Extract and log the response content
            message = completion_data["choices"][0].get("message", {})
            content_text = message.get("content", "")
            content_preview = content_text[:100] + ("..." if len(content_text) > 100 else "")

            logger.info(
                "Chat completion successful",
                content_preview=content_preview,
                model=completion_data.get("model"),
                finish_reason=completion_data["choices"][0].get("finish_reason"),
            )

            # Log usage statistics if available
            if "usage" in completion_data:
                logger.info(
                    "Usage statistics",
                    prompt_tokens=completion_data["usage"].get("prompt_tokens"),
                    completion_tokens=completion_data["usage"].get("completion_tokens"),
                    total_tokens=completion_data["usage"].get("total_tokens"),
                )

            return True
        except json.JSONDecodeError:
            logger.error(
                "Error parsing chat_completion result JSON",
                content_preview=content.text[:100],
            )
            return False

    except ClientError as e:
        logger.error("Tool call 'chat_completion' failed", error=str(e))
        return False
    except Exception as e:
        logger.error("Error executing chat_completion tool", error=str(e))
        return False


async def test_chat_completion_invalid_input(client: Client) -> bool:
    """Test the chat_completion tool with invalid input."""
    logger.info("Testing 'chat_completion' tool with invalid input")

    try:
        # Prepare an invalid conversation for the chat completion (missing role)
        messages = [{"content": "This message is missing the role field"}]

        await client.call_tool("chat_completion", {"messages": messages, "model": "cm-llm"})

        # If we get here, the tool didn't reject the invalid input
        logger.error("chat_completion tool accepted invalid input")
        return False

    except (ClientError, ToolError) as e:
        # We expect an error here, so this is a successful test
        logger.info("chat_completion tool properly rejected invalid input", error=str(e))
        return True
    except Exception as e:
        logger.error("Unexpected error testing chat_completion with invalid input", error=str(e))
        return False


async def test_get_context_invalid_input(client: Client) -> bool:
    """Test the get_context tool with invalid input."""
    logger.info("Testing 'get_context' tool with invalid input")

    try:
        # Call get_context without required query parameter
        await client.call_tool("get_context", {})

        # If we get here, the tool didn't reject the invalid input
        logger.error("get_context tool accepted missing required parameter")
        return False

    except (ClientError, ToolError) as e:
        # We expect an error here, so this is a successful test
        logger.info("get_context tool properly rejected missing parameter", error=str(e))
        return True
    except Exception as e:
        logger.error("Unexpected error testing get_context with invalid input", error=str(e))
        return False


async def test_health_endpoint() -> bool:
    """Test the health endpoint of the BaseRAG MCP server."""
    logger.info("Testing health endpoint")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8080/health") as response:
                if response.status != 200:
                    logger.error("Health endpoint returned non-200 status", status=response.status)
                    return False

                data = await response.json()
                logger.info(
                    "Health endpoint response",
                    status=data.get("status"),
                    service=data.get("service"),
                    api_connected=data.get("api_connected"),
                )

                # Check if the response has the expected structure
                if "status" not in data or "service" not in data:
                    logger.error("Health endpoint response missing required fields")
                    return False

                return True
    except Exception as e:
        logger.error("Error accessing health endpoint", error=str(e))
        return False


async def run():
    """Test the BaseRAG MCP server using FastMCP Client."""
    server_url = "http://localhost:8080/mcp"

    try:
        # Set up test environment
        await setup_test_environment()

        logger.info("Connecting to BaseRAG MCP server", url=server_url)

        # Test results dictionary
        test_results = {}

        # Test health endpoint first (doesn't require MCP client)
        test_results["health_endpoint"] = await test_health_endpoint()

        # Create the FastMCP Client context manager
        async with Client(server_url) as client:
            logger.info("Connection established with FastMCP Client")

            # Test connection and tool/resource discovery
            test_results["server_connection"] = await test_server_connection(client)

            if not test_results["server_connection"]:
                logger.error("Server connection test failed, aborting remaining tests")
                return

            # Test BaseRAG resources
            test_results["baserag_resources"] = await test_baserag_resources(client)

            # Test get_context tool and try to get a content ID for subsequent tests
            success, content_id = await test_get_context_tool(client)
            test_results["get_context_tool"] = success

            # Test get_content tool with the content ID if available
            test_results["get_content_tool"] = await test_get_content_tool(client, content_id)

            # Test get_content_chunks tool with the content ID if available
            test_results["get_content_chunks_tool"] = await test_get_content_chunks_tool(
                client, content_id
            )

            # Test chat_completion tool
            test_results["chat_completion_tool"] = await test_chat_completion_tool(client)

            # Test tools with invalid inputs
            test_results[
                "chat_completion_invalid_input"
            ] = await test_chat_completion_invalid_input(client)
            test_results["get_context_invalid_input"] = await test_get_context_invalid_input(client)

        # Print test results summary
        logger.info("Test Results Summary")
        all_passed = True
        for test_name, result in test_results.items():
            logger.info(f"Test: {test_name}", result="PASSED" if result else "FAILED")
            if not result:
                all_passed = False
                logger.warning(f"Failed tests: {result}")

        if all_passed:
            logger.info("All BaseRAG MCP tests PASSED")
        else:
            logger.warning("Some BaseRAG MCP tests FAILED")

    except Exception as e:
        logger.critical("An unexpected error occurred", error_type=type(e).__name__, error=str(e))
    finally:
        # Clean up test environment
        await cleanup_test_environment()


if __name__ == "__main__":
    asyncio.run(run())
