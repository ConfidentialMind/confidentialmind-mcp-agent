import logging
import signal
import sys

from .server import mcp_server
from .settings import settings

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to track shutdown status
shutdown_requested = False


def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    if shutdown_requested:
        logger.warning("Forced shutdown requested, exiting immediately!")
        sys.exit(1)

    logger.info("Shutdown requested, gracefully shutting down... (press Ctrl+C again to force)")
    shutdown_requested = True


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(
        f"Starting Postgres MCP server for database '{settings.database}' "
        f"on {settings.host}:{settings.port}"
    )

    try:
        # Start the server (this is a blocking call)
        mcp_server.run(transport="sse", port=8080)
    except KeyboardInterrupt:
        # This might still be triggered despite our signal handler
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.critical(f"Server failed to run: {e}", exc_info=True)
    finally:
        logger.info("Server shutdown complete.")
