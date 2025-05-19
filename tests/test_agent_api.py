"""Test the FastMCP agent API functionality.

This test assumes the agent API is already running and accessible at the
configured host and port. It tests various API endpoints including the health
endpoint and chat completions endpoint.

Before running this test:
1. Start the PostgreSQL database
2. Start the Postgres MCP server: python -m src.tools.postgres_mcp
3. Start the agent API server: python -m src.agent.main serve

Usage:
    python -m tests.test_agent_api
"""

import asyncio
import uuid
from typing import Any, Dict, Optional

import aiohttp
import asyncpg

from tests.logger import logger

# Configuration
TEST_API_HOST = "127.0.0.1"
TEST_API_PORT = 8000
TEST_SESSION_ID = str(uuid.uuid4())
DATABASE_DSN = "postgresql://postgres:postgres@localhost:5432/test_db"


async def create_test_tables() -> None:
    """Create test tables for API server testing."""
    logger.info("Setting up test database for API testing...")

    # Connect directly to the database
    conn = await asyncpg.connect(DATABASE_DSN)

    try:
        # Check if tables already exist
        table_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name IN ('api_test_users', 'api_test_products');
        """)

        if table_count > 0:
            logger.info("Test tables already exist, dropping them first...")
            await conn.execute("DROP TABLE IF EXISTS api_test_users, api_test_products;")

        # Create api_test_users table
        await conn.execute("""
            CREATE TABLE api_test_users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                email VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Insert sample data
        await conn.execute("""
            INSERT INTO api_test_users (username, email) VALUES 
            ('apiuser1', 'apiuser1@example.com'),
            ('apiuser2', 'apiuser2@example.com'),
            ('apiuser3', 'apiuser3@example.com');
        """)

        # Create api_test_products table
        await conn.execute("""
            CREATE TABLE api_test_products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                description TEXT
            );
        """)

        # Insert sample data
        await conn.execute("""
            INSERT INTO api_test_products (name, price, description) VALUES 
            ('API Product 1', 19.99, 'This is API product 1'),
            ('API Product 2', 29.99, 'This is API product 2'),
            ('API Product 3', 39.99, 'This is API product 3');
        """)

        logger.info("Test tables created successfully for API testing")
    finally:
        await conn.close()


async def cleanup_test_tables() -> None:
    """Remove test tables after testing is complete."""
    logger.info("Cleaning up test database after API testing...")

    # Connect directly to the database
    conn = await asyncpg.connect(DATABASE_DSN)

    try:
        # Drop test tables
        await conn.execute("DROP TABLE IF EXISTS api_test_users, api_test_products;")
        logger.info("Test tables removed successfully")
    except Exception as e:
        logger.error(f"Error cleaning up test tables: {e}")
    finally:
        await conn.close()


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

                    # Check if status is healthy
                    status_healthy = data.get("status") == "healthy"

                    # Check if database is connected - database is now an object with 'connected' field
                    database_connected = False
                    database_info = data.get("database")
                    if isinstance(database_info, dict):
                        database_connected = database_info.get("connected", False)

                    # Optional: Check if LLM is connected - you might want to verify this as well
                    llm_connected = False
                    llm_info = data.get("llm")
                    if isinstance(llm_info, dict):
                        llm_connected = llm_info.get("connected", False)

                    # Return overall health status
                    return status_healthy and database_connected
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
    logger.info("Testing simple query without tools...")

    result = await send_chat_query("Hello, how are you today?", TEST_SESSION_ID)

    if "error" in result:
        logger.error("Simple query error", error=result["error"])
        return False

    response_text = get_response_text(result)
    if not response_text:
        return False

    logger.info(
        "Simple query response",
        response=response_text,
    )
    return True


async def test_postgres_query():
    """Test a query using the postgres MCP tool to access the test tables."""
    logger.info("Testing postgres query with test tables...")

    query = """
    Query the api_test_users table and show me all users. 
    Display the usernames and emails in a nice format.
    """

    result = await send_chat_query(query, TEST_SESSION_ID)

    if "error" in result:
        logger.error("Postgres query error", error=result["error"])
        return False

    response_text = get_response_text(result)
    if not response_text:
        return False

    logger.info(
        "Postgres query response",
        response=response_text,
    )

    # Look for indicators of successful PostgreSQL query response
    success_indicators = ["apiuser1", "apiuser2", "email"]
    # Check if at least one indicator is in the response
    if any(indicator in response_text.lower() for indicator in success_indicators):
        logger.info("Postgres query successful - found expected data in response")
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
        response=response1_text,
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
        response=response2_text,
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
        response=response_text,
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


async def test_postgres_products_query():
    """Test a query using the postgres MCP tool to access the products test table."""
    logger.info("Testing postgres query on products table...")

    query = """
    Show me all products from the api_test_products table that cost more than $25.
    Include the name, price, and description.
    """

    result = await send_chat_query(query, TEST_SESSION_ID)

    if "error" in result:
        logger.error("Products query error", error=result["error"])
        return False

    response_text = get_response_text(result)
    if not response_text:
        return False

    logger.info(
        "Products query response",
        response=response_text,
    )

    # Look for indicators of successful PostgreSQL query response
    # Only API Products 2 and 3 cost more than $25
    success_indicators = ["API Product 2", "API Product 3", "29.99", "39.99"]

    # Check if at least one indicator is in the response
    if any(indicator in response_text for indicator in success_indicators):
        logger.info("Products query successful - found expected data in response")
        return True
    else:
        logger.warning("Products query might have failed - response doesn't contain expected info")
        return False


async def run():
    """Run the tests."""
    logger.info("Starting FastMCP agent API tests...")

    try:
        # Create test tables
        await create_test_tables()

        # List of tests to run
        tests = [
            ("Health endpoint", test_health_endpoint),
            ("Simple query", test_simple_query),
            ("PostgreSQL users query", test_postgres_query),
            ("PostgreSQL products query", test_postgres_products_query),
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

    finally:
        # Clean up test tables
        logger.info("Cleaning up test environment...")
        await cleanup_test_tables()


if __name__ == "__main__":
    asyncio.run(run())
