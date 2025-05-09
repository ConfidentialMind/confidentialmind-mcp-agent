"""Test the FastMCP agent API functionality.

This test assumes the agent API is already running and accessible at the
configured host and port. It tests various API endpoints including the health
endpoint and chat completions endpoint.

Usage:
    python -m tests.test_agent_api
"""

import asyncio
import uuid
from typing import Any, Dict, Optional

import aiohttp

from tests.logger import logger

# Configuration
TEST_API_HOST = "127.0.0.1"
TEST_API_PORT = 8000
TEST_SESSION_ID = str(uuid.uuid4())


async def test_health_endpoint():
    """Test the health endpoint."""
    logger.info("Testing health endpoint...")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"http://{TEST_API_HOST}:{TEST_API_PORT}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(
                        "Health endpoint response",
                        status=data.get("status"),
                        database=data.get("database"),
                    )
                    return data.get("status") == "healthy" and data.get("database") == True
                else:
                    logger.error("Health endpoint failed", status=response.status)
                    return False
        except aiohttp.ClientError as e:
            logger.error("Connection error", error=str(e))
            return False


async def send_chat_query(query: str, session_id: str) -> Dict[str, Any]:
    """Send a chat query to the API and return the response."""
    logger.info("Sending chat query", query=query)

    headers = {"Content-Type": "application/json", "X-Session-ID": session_id}

    data = {"model": "cm-llm", "messages": [{"role": "user", "content": query}]}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"http://{TEST_API_HOST}:{TEST_API_PORT}/v1/chat/completions",
                headers=headers,
                json=data,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.error("Chat query failed", status=response.status, error=error_text)
                    return {"error": f"Failed with status {response.status}: {error_text}"}
        except aiohttp.ClientError as e:
            logger.error("Connection error", error=str(e))
            return {"error": f"Connection error: {str(e)}"}


def get_response_text(result: Dict[str, Any]) -> Optional[str]:
    """Extract the response text from a chat result."""
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.error("Failed to extract response text", error=str(e))
        return None


async def test_simple_query():
    """Test a simple query without tools."""
    logger.info("Testing simple query...")

    result = await send_chat_query("Hello, how are you today?", TEST_SESSION_ID)

    if "error" in result:
        logger.error("Simple query error", error=result["error"])
        return False

    response_text = get_response_text(result)
    if not response_text:
        return False

    logger.info(
        "Simple query response",
        response=response_text[:100] + "..." if len(response_text) > 100 else response_text,
    )
    return True


async def test_postgres_query():
    """Test a query using the postgres MCP tool."""
    logger.info("Testing postgres query...")

    result = await send_chat_query(
        "Query the test_users table and show me all users", TEST_SESSION_ID
    )

    if "error" in result:
        logger.error("Postgres query error", error=result["error"])
        return False

    response_text = get_response_text(result)
    if not response_text:
        return False

    logger.info(
        "Postgres query response",
        response=response_text[:100] + "..." if len(response_text) > 100 else response_text,
    )

    # Look for indicators of successful PostgreSQL query response
    success_indicators = ["user1", "user2", "email"]
    # Check if at least one indicator is in the response
    if any(indicator in response_text.lower() for indicator in success_indicators):
        return True
    else:
        logger.warning("Postgres query might have failed - response doesn't contain expected info")
        return False


async def test_conversation_management():
    """Test multi-turn conversation management."""
    logger.info("Testing conversation management...")

    # First query
    query1_result = await send_chat_query("What is your name?", TEST_SESSION_ID)

    if "error" in query1_result:
        logger.error("Conversation first query error", error=query1_result["error"])
        return False

    response1_text = get_response_text(query1_result)
    if not response1_text:
        return False

    logger.info(
        "Conversation first response",
        response=response1_text[:100] + "..." if len(response1_text) > 100 else response1_text,
    )

    # Second query that references the first
    query2_result = await send_chat_query(
        "Can you remember what I just asked you?", TEST_SESSION_ID
    )

    if "error" in query2_result:
        logger.error("Conversation second query error", error=query2_result["error"])
        return False

    response2_text = get_response_text(query2_result)
    if not response2_text:
        return False

    logger.info(
        "Conversation second response",
        response=response2_text[:100] + "..." if len(response2_text) > 100 else response2_text,
    )

    # Check if the response references the first query
    if "name" in response2_text.lower():
        logger.info("Conversation management successful - agent remembered context")
        return True
    else:
        logger.warning(
            "Conversation management might have failed - agent didn't clearly reference previous context"
        )
        return False


async def test_show_history():
    """Test the 'show history' command."""
    logger.info("Testing 'show history' command...")

    result = await send_chat_query("show history", TEST_SESSION_ID)

    if "error" in result:
        logger.error("Show history error", error=result["error"])
        return False

    response_text = get_response_text(result)
    if not response_text:
        return False

    logger.info(
        "Show history response",
        response=response_text[:100] + "..." if len(response_text) > 100 else response_text,
    )

    # Verify that the history contains our previous queries
    if "What is your name" in response_text:
        logger.info("Show history successful - found previous query")
        return True
    else:
        logger.warning("Show history might have failed - couldn't find previous query")
        return False


async def test_clear_history():
    """Test the 'clear history' command."""
    logger.info("Testing 'clear history' command...")

    # First clear the history
    clear_result = await send_chat_query("clear history", TEST_SESSION_ID)

    if "error" in clear_result:
        logger.error("Clear history error", error=clear_result["error"])
        return False

    clear_response_text = get_response_text(clear_result)
    if not clear_response_text:
        return False

    logger.info("Clear history response", response=clear_response_text)

    # Then check if history is cleared
    show_result = await send_chat_query("show history", TEST_SESSION_ID)

    if "error" in show_result:
        logger.error("Show history after clear error", error=show_result["error"])
        return False

    show_response_text = get_response_text(show_result)
    if not show_response_text:
        return False

    logger.info("Show history after clear response", response=show_response_text)

    if "(empty)" in show_response_text or "empty" in show_response_text.lower():
        logger.info("Clear history successful - history is now empty")
        return True
    else:
        logger.warning("Clear history might have failed - history doesn't appear empty")
        return False


async def run():
    """Run the tests."""
    logger.info("Starting FastMCP agent API tests...")

    # List of tests to run
    tests = [
        ("Health endpoint", test_health_endpoint),
        ("Simple query", test_simple_query),
        ("PostgreSQL query", test_postgres_query),
        ("Conversation management", test_conversation_management),
        ("Show history", test_show_history),
        ("Clear history", test_clear_history),
    ]

    # Run tests sequentially
    results = {}
    for test_name, test_func in tests:
        logger.info(f"Running test: {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
            logger.info(f"Test {test_name} {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test {test_name} FAILED with exception", error=str(e))
            results[test_name] = False

    # Log summary
    logger.info("Test Summary")
    for test_name, result in results.items():
        logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'}")

    # Overall result
    if all(results.values()):
        logger.info("All tests PASSED")
    else:
        logger.warning("Some tests FAILED")


if __name__ == "__main__":
    asyncio.run(run())
