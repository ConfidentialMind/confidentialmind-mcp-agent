import asyncio
import logging
import os
import signal
import sys
from threading import Event

from confidentialmind_core.config_manager import config as cm_config
from confidentialmind_core.config_manager import load_environment

from .server import mcp_server
from .settings import settings

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables to initialize LOCAL_DEV and LOCAL_CONFIGS flags
load_environment()

# Event for signaling shutdown
shutdown_event = Event()
main_task = None


def signal_handler(sig, frame):
    """Handle interrupt signals by cancelling the main task."""
    global shutdown_event, main_task

    if shutdown_event.is_set():
        logger.warning("Forced shutdown requested, exiting immediately!")
        sys.exit(1)

    logger.info("Shutdown requested, gracefully shutting down...")
    shutdown_event.set()

    # If we have access to the main task, cancel it
    if main_task and not main_task.done():
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(main_task.cancel)
        except Exception as e:
            logger.error(f"Error cancelling main task: {e}")
            # Force exit if we can't cancel gracefully
            sys.exit(1)


async def run_server(transport_type, **kwargs):
    """Run the MCP server with shutdown handling."""
    try:
        logger.info(f"Starting Postgres MCP server with {transport_type} transport")
        return await mcp_server.run_async(transport=transport_type, **kwargs)
    except asyncio.CancelledError:
        logger.info("Server task cancelled, shutting down gracefully")
    except Exception as e:
        logger.critical(f"Server failed to run: {e}", exc_info=True)
    finally:
        logger.info("Server shutdown complete.")


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    deployment_mode = "stack" if not cm_config.LOCAL_CONFIGS else "local"
    logger.info(
        f"Starting Postgres MCP server in {deployment_mode} mode"
        f" for database '{settings.database}' on {settings.host}:{settings.port}"
    )

    try:
        use_stdio = (
            "FastMCP_TRANSPORT" in os.environ
            and os.environ["FastMCP_TRANSPORT"] == "stdio"
            or "--stdio" in sys.argv
        )

        loop = asyncio.get_event_loop()
        if use_stdio:
            logger.info("Using stdio transport for agent communication")
            main_task = asyncio.ensure_future(run_server("stdio"))
        else:
            logger.info("Using SSE transport on port 8080")
            main_task = asyncio.ensure_future(run_server("sse", port=8080))

        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        if "loop" in locals() and loop.is_running():
            try:
                tasks = asyncio.all_tasks(loop=loop)
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                loop.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
