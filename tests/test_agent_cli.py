"""Test the FastMCP agent CLI functionality.

This test executes the agent's CLI interface directly to test its
functionality in CLI mode using stdio transport.

Usage:
    python -m tests.test_agent_cli
"""

import json
import os
import subprocess
import tempfile
import uuid
from typing import Optional

from tests.logger import logger

# Configuration
TEST_SCRIPT_PATH = "src/agent/main.py"
TEST_SESSION_ID = str(uuid.uuid4())


def create_config_file():
    """Create a temporary config file for CLI tests."""
    temp_dir = tempfile.gettempdir()
    config_path = os.path.join(temp_dir, f"agent_cli_test_config_{uuid.uuid4()}.json")

    # Write config to file
    config = {
        "mcp_servers": {
            "postgres": "src/tools/postgres_mcp/__main__.py"  # Use stdio for CLI mode
        }
    }

    with open(config_path, "w") as f:
        json.dump(config, f)

    logger.info("Created temporary CLI config file", path=config_path)
    return config_path


def run_cli_command(
    query: str,
    session_id: Optional[str] = None,
    debug: bool = False,
    config_path: Optional[str] = None,
):
    """Run the agent CLI with the given query and return the output."""
    cmd = ["python", TEST_SCRIPT_PATH, query]

    if session_id:
        cmd.extend(["--session", session_id])

    if debug:
        cmd.append("--debug")

    if config_path:
        cmd.extend(["--config", config_path])

    logger.info("Running CLI command", command=" ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("CLI command failed", returncode=result.returncode, stderr=result.stderr)
            return {
                "error": f"Command failed with return code {result.returncode}: {result.stderr}"
            }

        return {"output": result.stdout}
    except subprocess.SubprocessError as e:
        logger.error("Subprocess error", error=str(e))
        return {"error": f"Subprocess error: {str(e)}"}


def test_simple_query(config_path: str):
    """Test a simple query without tools."""
    logger.info("Testing simple CLI query...")

    result = run_cli_command(
        query="Hello, how are you today?", session_id=TEST_SESSION_ID, config_path=config_path
    )

    if "error" in result:
        logger.error("Simple CLI query error", error=result["error"])
        return False

    output = result.get("output", "")
    logger.info(
        "Simple CLI query response", output=output[:100] + "..." if len(output) > 100 else output
    )

    # Check for a successful response
    if "Response:" in output:
        return True
    else:
        logger.warning("Simple CLI query might have failed - unexpected output format")
        return False


def test_postgres_query(config_path: str):
    """Test a query using the postgres MCP tool."""
    logger.info("Testing postgres CLI query...")

    result = run_cli_command(
        query="Query the test_users table and show me all users",
        session_id=TEST_SESSION_ID,
        config_path=config_path,
    )

    if "error" in result:
        logger.error("Postgres CLI query error", error=result["error"])
        return False

    output = result.get("output", "")
    logger.info(
        "Postgres CLI query response", output=output[:100] + "..." if len(output) > 100 else output
    )

    # Look for indicators of successful PostgreSQL query response
    success_indicators = ["user1", "user2", "email"]
    # Check if at least one indicator is in the output
    if any(indicator in output.lower() for indicator in success_indicators):
        return True
    else:
        logger.warning(
            "Postgres CLI query might have failed - response doesn't contain expected info"
        )
        return False


def test_conversation_management(config_path: str):
    """Test multi-turn conversation management via CLI."""
    logger.info("Testing CLI conversation management...")

    # First query
    query1_result = run_cli_command(
        query="What is your name?", session_id=TEST_SESSION_ID, config_path=config_path
    )

    if "error" in query1_result:
        logger.error("CLI conversation first query error", error=query1_result["error"])
        return False

    output1 = query1_result.get("output", "")
    logger.info(
        "CLI conversation first response",
        output=output1[:100] + "..." if len(output1) > 100 else output1,
    )

    # Second query that references the first
    query2_result = run_cli_command(
        query="Can you remember what I just asked you?",
        session_id=TEST_SESSION_ID,
        config_path=config_path,
    )

    if "error" in query2_result:
        logger.error("CLI conversation second query error", error=query2_result["error"])
        return False

    output2 = query2_result.get("output", "")
    logger.info(
        "CLI conversation second response",
        output=output2[:100] + "..." if len(output2) > 100 else output2,
    )

    # Check if the response references the first query
    if "name" in output2.lower():
        logger.info("CLI conversation management successful - agent remembered context")
        return True
    else:
        logger.warning(
            "CLI conversation management might have failed - agent didn't clearly reference previous context"
        )
        return False


def test_show_history(config_path: str):
    """Test the 'show history' command via CLI."""
    logger.info("Testing CLI 'show history' command...")

    result = run_cli_command(
        query="show history", session_id=TEST_SESSION_ID, config_path=config_path
    )

    if "error" in result:
        logger.error("CLI show history error", error=result["error"])
        return False

    output = result.get("output", "")
    logger.info(
        "CLI show history response", output=output[:100] + "..." if len(output) > 100 else output
    )

    # Verify that the history contains our previous queries
    if "What is your name" in output:
        logger.info("CLI show history successful - found previous query")
        return True
    else:
        logger.warning("CLI show history might have failed - couldn't find previous query")
        return False


def test_clear_history(config_path: str):
    """Test the 'clear history' command via CLI."""
    logger.info("Testing CLI 'clear history' command...")

    # First clear the history
    clear_result = run_cli_command(
        query="clear history", session_id=TEST_SESSION_ID, config_path=config_path
    )

    if "error" in clear_result:
        logger.error("CLI clear history error", error=clear_result["error"])
        return False

    clear_output = clear_result.get("output", "")
    logger.info("CLI clear history response", output=clear_output)

    # Then check if history is cleared
    show_result = run_cli_command(
        query="show history", session_id=TEST_SESSION_ID, config_path=config_path
    )

    if "error" in show_result:
        logger.error("CLI show history after clear error", error=show_result["error"])
        return False

    show_output = show_result.get("output", "")
    logger.info("CLI show history after clear response", output=show_output)

    if "(empty)" in show_output or "empty" in show_output.lower():
        logger.info("CLI clear history successful - history is now empty")
        return True
    else:
        logger.warning("CLI clear history might have failed - history doesn't appear empty")
        return False


def main():
    """Run the CLI tests."""
    logger.info("Starting FastMCP agent CLI tests...")

    # Create config file
    config_path = create_config_file()

    try:
        # List of tests to run
        tests = [
            ("Simple CLI query", lambda: test_simple_query(config_path)),
            ("PostgreSQL CLI query", lambda: test_postgres_query(config_path)),
            ("CLI conversation management", lambda: test_conversation_management(config_path)),
            ("CLI show history", lambda: test_show_history(config_path)),
            ("CLI clear history", lambda: test_clear_history(config_path)),
        ]

        # Run tests sequentially
        results = {}
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = test_func()
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
            logger.info("All CLI tests PASSED")
        else:
            logger.warning("Some CLI tests FAILED")

    finally:
        # Clean up
        try:
            if os.path.exists(config_path):
                os.remove(config_path)
                logger.info("Removed temporary CLI config file")
        except Exception as e:
            logger.error("Error removing temporary CLI config file", error=str(e))


if __name__ == "__main__":
    main()
