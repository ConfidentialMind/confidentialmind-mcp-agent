"""
Shared logging module for all ConfidentialMind components.
Provides structured logging with OpenTelemetry-compatible JSON format.
"""

from .config import configure_python_logging, configure_structlog, get_logger

# Re-export tracing and decorators from SDK for consistency
from confidentialmind_core import TraceContext, get_current_trace, traced, traced_async, log_operation

__all__ = [
    "configure_structlog",
    "configure_python_logging",
    "get_logger",
    "TraceContext",
    "get_current_trace",
    "traced",
    "traced_async",
    "log_operation",
]
