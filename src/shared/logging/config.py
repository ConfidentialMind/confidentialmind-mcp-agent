"""
Unified logging configuration for ConfidentialMind services.
Provides structured logging with OpenTelemetry-compatible JSON format.
"""

import logging
import os
import sys

import structlog
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def add_otel_context_processor(logger, method_name, event_dict):
    """Processor to add OpenTelemetry context to log entries."""
    from .trace_context import get_current_trace

    trace = get_current_trace()
    if trace:
        event_dict.update(
            {
                "trace_id": trace.trace_id,
                "span_id": trace.span_id,
                "parent_span_id": trace.parent_span_id,
                "session_id": trace.session_id,
                "api_key_hash": trace.api_key_hash,
                "origin_service": trace.origin_service,
            }
        )

    return event_dict


def configure_structlog():
    """Configure structlog with OpenTelemetry-compatible processors."""
    debug_output = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")

    common_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        # Add custom processor for OpenTelemetry fields
        add_otel_context_processor,
    ]

    if debug_output:
        # Development: Human-readable console output
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Production: JSON format for OpenTelemetry Collector
        renderer = structlog.processors.JSONRenderer()

    log_level = logging.DEBUG if debug_output else logging.INFO

    structlog.configure(
        processors=common_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structlog logger instance.

    Args:
        name: Logger name (e.g., "agent.core", "mcp.transport")

    Returns:
        Configured structlog bound logger
    """
    if not structlog.is_configured():
        configure_structlog()
    return structlog.get_logger(name)


def configure_python_logging():
    """Configure standard Python logging to work with structlog."""
    debug_output = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
    log_level = logging.DEBUG if debug_output else logging.INFO

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Integrate with structlog
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

    # Use structlog's processors for stdlib logging
    logging.getLogger().handlers[0].setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer()
            if debug_output
            else structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                add_otel_context_processor,
            ],
        )
    )


# Initialize logging configuration when module is imported
configure_structlog()
configure_python_logging()
