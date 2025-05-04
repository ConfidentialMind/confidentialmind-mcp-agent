import asyncio
from asyncio.events import set_event_loop

import pytest

from src.mcp.postgres.database import DatabaseManager
from src.mcp.postgres.settings import PostgresSettings

pytestmark = pytest.mark.asyncio(scope="session")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def settings() -> PostgresSettings:
    """Create test settings from environment variables."""
    return PostgresSettings()


@pytest.fixture
async def db_manager(settings: PostgresSettings):
    """Create and return a database manager."""
    manager = DatabaseManager(settings)
    await manager.close()  # Close any existing connections first
    yield manager
    # Cleanup
    await manager.close()


@pytest.fixture(autouse=True)
async def reset_db_pool():
    """Reset the database pool between tests."""
    # This runs before each test
    yield
    # This runs after each test

