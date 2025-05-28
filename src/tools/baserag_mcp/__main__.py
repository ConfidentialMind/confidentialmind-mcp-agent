import argparse
import asyncio
import logging
import os
import signal
import sys
from threading import Event

from confidentialmind_core.config_manager import load_environment
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from .connection_manager import ConnectionManager
from .server import http_health_endpoint, mcp_server
from .settings import settings

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


async def root_health_check(request):
    """Root level health check that's always accessible."""
    return await http_health_endpoint(request)


async def run_server(transport_type, **kwargs):
    """
    Run the MCP server with shutdown handling and proactive initialization.

    This function initializes the ConnectionManager before starting the server
    to ensure that API connections can be established in the background
    even if not available at startup.
    """
    try:
        # Proactively initialize ConnectionManager before server starts
        logger.info("Proactively initializing ConnectionManager...")
        await ConnectionManager.initialize()
        logger.info("ConnectionManager initialized successfully")

        if transport_type == "streamable-http":
            # For HTTP-based transports, create a combined app with root-level health endpoint
            # Get the ASGI app for the MCP server
            mcp_app = mcp_server.http_app(path="")

            # Create a Starlette app with both the MCP app and a root-level health endpoint
            combined_app = Starlette(
                routes=[
                    Route("/health", endpoint=root_health_check, methods=["GET"]),
                    Mount("/", app=mcp_app),
                ],
                lifespan=mcp_app.router.lifespan_context,  # Important: use MCP server's lifespan
            )

            # Use additional kwargs (port, host, etc.) passed to this function
            port = kwargs.get("port", 8080)
            host = kwargs.get("host", "0.0.0.0")
            log_level = kwargs.get("log_level", "info")

            import uvicorn

            config = uvicorn.Config(combined_app, host=host, port=port, log_level=log_level)
            server = uvicorn.Server(config)
            await server.serve()
        elif transport_type == "stdio":
            # For STDIO transport, use the normal run method
            return await mcp_server.run_async(transport=transport_type, **kwargs)
        else:
            logger.warning(
                f"Unsupported transport type: {transport_type}, defaulting to streamable-http"
            )
            # Fall back to streamable-http for any unknown transport type
            return await run_server("streamable-http", **kwargs)
    except asyncio.CancelledError:
        logger.info("Server task cancelled, shutting down gracefully")
    except Exception as e:
        logger.critical(f"Server failed to run: {e}", exc_info=True)
    finally:
        # Ensure ConnectionManager is closed on shutdown
        try:
            await ConnectionManager.close()
            logger.info("ConnectionManager closed during shutdown")
        except Exception as e:
            logger.error(f"Error closing ConnectionManager: {e}")
        logger.info("Server shutdown complete.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BaseRAG MCP Server")
    parser.add_argument(
        "--stdio", action="store_true", help="Use stdio transport for agent communication"
    )
    parser.add_argument(
        "--streamable-http", action="store_true", help="Use streamable HTTP transport (default)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP transport (default: 8080)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host for HTTP transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    # Handle legacy --sse flag
    if "--sse" in sys.argv:
        logger.warning("SSE transport is deprecated, using streamable-http instead")
        sys.argv.remove("--sse")
        if "--streamable-http" not in sys.argv:
            sys.argv.append("--streamable-http")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize environment variables and check deployment mode
    load_environment()

    # Determine deployment mode
    is_stack_deployment = (
        os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
    )

    if is_stack_deployment:
        settings.use_sdk_connector = True
        logger.info("Running in stack deployment mode with SDK connector integration")
    else:
        logger.info(f"Running in local mode with BaseRAG API URL: {settings.api_url}")

    try:
        # Determine transport type based on arguments and environment
        use_stdio = args.stdio or (
            "FastMCP_TRANSPORT" in os.environ and os.environ["FastMCP_TRANSPORT"] == "stdio"
        )

        # Override stdio mode if streamable-http is explicitly requested
        if args.streamable_http:
            use_stdio = False

        loop = asyncio.get_event_loop()
        if use_stdio:
            logger.info("Using stdio transport for agent communication")
            main_task = asyncio.ensure_future(run_server("stdio"))
        else:
            # Use streamable HTTP transport with specified port
            logger.info(f"Using Streamable HTTP transport on {args.host}:{args.port}")
            main_task = asyncio.ensure_future(
                run_server(
                    "streamable-http", port=args.port, host=args.host, log_level=args.log_level
                )
            )

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
