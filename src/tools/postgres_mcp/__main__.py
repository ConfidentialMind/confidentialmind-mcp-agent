import logging

from .server import mcp_server
from .settings import settings

# Configure basic logging if needed when running directly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(
        f"Starting Postgres MCP server for database '{settings.database}' on {settings.host}:{settings.port}"
    )
    try:
        mcp_server.run()
    except Exception as e:
        logger.critical(f"Server failed to run: {e}", exc_info=True)
