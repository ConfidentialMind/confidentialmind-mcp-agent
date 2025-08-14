"""
Shared logging module for all ConfidentialMind components.
Provides structured logging with OpenTelemetry-compatible JSON format.
"""

from .config import configure_python_logging, configure_structlog, get_logger
from .decorators import log_operation, traced, traced_async

# Re-export tracing from SDK
from confidentialmind_core import TraceContext, get_current_trace

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
