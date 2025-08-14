"""
Structured logging configuration for ConfidentialMind services.
Provides automatic trace context injection and OpenTelemetry-compatible formatting.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

import structlog


def add_trace_context_processor(logger, method_name, event_dict):
    """Add trace context to log entries automatically."""
    from .tracing import get_current_trace
    
    trace = get_current_trace()
    if trace:
        event_dict.update({
            "trace_id": trace.trace_id,
            "span_id": trace.span_id,
            "parent_span_id": trace.parent_span_id,
            "session_id": trace.session_id,
            "api_key_hash": trace.api_key_hash,
            "origin_service": trace.origin_service,
        })
    
    return event_dict


def configure_structlog():
    """Configure structlog with trace context and OpenTelemetry processors."""
    debug_output = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
    
    common_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        # Automatically add trace context to all logs
        add_trace_context_processor,
    ]
    
    if debug_output:
        # Human-readable output for development
        processors = common_processors + [
            structlog.dev.ConsoleRenderer()
        ]
    else:
        # JSON output for production (OpenTelemetry compatible)
        processors = common_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def configure_python_logging():
    """Configure Python's standard logging to work with structlog."""
    debug_output = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
    
    # Set up the root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG if debug_output else logging.INFO,
    )
    
    # Integrate with structlog
    if debug_output:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()
    
    # Rebuild common processors for formatter
    common_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        add_trace_context_processor,
    ]
    
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=common_processors[:-1],  # All except the last processor
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if debug_output else logging.INFO)


def get_logger(name: Optional[str] = None) -> Any:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (e.g., "agent.llm", "service.api")
        
    Returns:
        Configured structlog logger with trace context support
    """
    if not structlog.is_configured():
        configure_structlog()
        configure_python_logging()
    
    return structlog.get_logger(name)


# Auto-configure on import for convenience
configure_structlog()
configure_python_logging()