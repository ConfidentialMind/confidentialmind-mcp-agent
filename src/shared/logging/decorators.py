"""
Decorators for automatic function tracing with minimal invasion.
Provides async-safe tracing with structlog integration.
"""

import functools
import inspect
import time
import uuid
from typing import Any, Callable, Dict, Optional, TypeVar

from .config import get_logger
from .trace_context import TraceContext, get_current_trace

F = TypeVar("F", bound=Callable[..., Any])


def extract_function_attributes(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """
    Extract relevant attributes from function arguments for tracing.

    Args:
        func: Function being traced
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary of extracted attributes
    """
    attributes = {}

    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Extract common parameters
        for param_name in ["query", "session_id", "server_id", "tool_name", "method"]:
            if param_name in bound.arguments:
                value = bound.arguments[param_name]
                if value is not None:
                    attributes[param_name] = str(value)[:100]  # Limit string length

        # Extract state information if available
        if "self" in bound.arguments:
            self_obj = bound.arguments["self"]
            # Check for common state attributes
            for attr in ["_has_tools", "_db_connected", "_llm_connected"]:
                if hasattr(self_obj, attr):
                    attributes[attr.lstrip("_")] = getattr(self_obj, attr)

    except Exception:
        # Don't fail if attribute extraction fails
        pass

    return attributes


def traced_async(
    event_type: Optional[str] = None, logger_name: Optional[str] = None, extract_args: bool = True
) -> Callable[[F], F]:
    """
    Decorator for tracing async functions with structlog.

    Args:
        event_type: Event type (defaults to module.function)
        logger_name: Logger name (defaults to module name)
        extract_args: Whether to extract function arguments

    Usage:
        @traced_async("agent.workflow", "agent.core")
        async def run(self, query: str) -> AgentState:
            # Function implementation
            pass
    """

    def decorator(func: F) -> F:
        nonlocal event_type, logger_name

        if not event_type:
            event_type = f"{func.__module__.split('.')[-1]}.{func.__name__}"

        if not logger_name:
            # Extract sensible logger name from module path
            module_parts = func.__module__.split(".")
            if len(module_parts) >= 2 and module_parts[0] == "src":
                if module_parts[1] == "agent":
                    logger_name = (
                        "agent." + module_parts[2] if len(module_parts) > 2 else "agent.core"
                    )
                elif module_parts[1] == "tools":
                    logger_name = module_parts[2] if len(module_parts) > 2 else "tools"
                elif module_parts[1] == "shared":
                    logger_name = "shared." + module_parts[2] if len(module_parts) > 2 else "shared"
                else:
                    logger_name = module_parts[1]
            else:
                logger_name = module_parts[-1] if module_parts else "agent"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)

            # Create child span for this operation
            trace = get_current_trace()
            span_id = str(uuid.uuid4())
            parent_span_id = trace.span_id if trace else None

            # Extract function attributes
            attributes = {}
            if extract_args:
                attributes = extract_function_attributes(func, args, kwargs)

            # Set current span
            old_span = None
            if trace:
                old_span = trace.span_id
                TraceContext.set_span(span_id, parent_span_id)

            start_time = time.time()

            # Log start event
            logger.info(
                f"Starting {event_type}",
                event_type=f"{event_type}.start",
                span_id=span_id,
                parent_span_id=parent_span_id,
                data=attributes,
            )

            try:
                result = await func(*args, **kwargs)

                # Log success
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Completed {event_type}",
                    event_type=f"{event_type}.complete",
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    duration_ms=duration_ms,
                    success=True,
                    data=attributes,
                )

                return result

            except Exception as e:
                # Log error
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Failed {event_type}",
                    event_type=f"{event_type}.complete",
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                    data=attributes,
                )
                raise

            finally:
                # Restore parent span
                if trace and old_span:
                    TraceContext.set_span(old_span, trace.parent_span_id)

        return wrapper

    return decorator


def traced(
    event_type: Optional[str] = None, logger_name: Optional[str] = None, extract_args: bool = True
) -> Callable[[F], F]:
    """
    Decorator for tracing synchronous functions.

    Args:
        event_type: Event type (defaults to module.function)
        logger_name: Logger name (defaults to module name)
        extract_args: Whether to extract function arguments
    """

    def decorator(func: F) -> F:
        nonlocal event_type, logger_name

        if not event_type:
            event_type = f"{func.__module__.split('.')[-1]}.{func.__name__}"

        if not logger_name:
            # Extract sensible logger name from module path
            module_parts = func.__module__.split(".")
            if len(module_parts) >= 2 and module_parts[0] == "src":
                if module_parts[1] == "agent":
                    logger_name = (
                        "agent." + module_parts[2] if len(module_parts) > 2 else "agent.core"
                    )
                elif module_parts[1] == "tools":
                    logger_name = module_parts[2] if len(module_parts) > 2 else "tools"
                elif module_parts[1] == "shared":
                    logger_name = "shared." + module_parts[2] if len(module_parts) > 2 else "shared"
                else:
                    logger_name = module_parts[1]
            else:
                logger_name = module_parts[-1] if module_parts else "agent"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)

            # Create child span for this operation
            trace = get_current_trace()
            span_id = str(uuid.uuid4())
            parent_span_id = trace.span_id if trace else None

            # Extract function attributes
            attributes = {}
            if extract_args:
                attributes = extract_function_attributes(func, args, kwargs)

            # Set current span
            old_span = None
            if trace:
                old_span = trace.span_id
                TraceContext.set_span(span_id, parent_span_id)

            start_time = time.time()

            # Log start event
            logger.info(
                f"Starting {event_type}",
                event_type=f"{event_type}.start",
                span_id=span_id,
                parent_span_id=parent_span_id,
                data=attributes,
            )

            try:
                result = func(*args, **kwargs)

                # Log success
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Completed {event_type}",
                    event_type=f"{event_type}.complete",
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    duration_ms=duration_ms,
                    success=True,
                    data=attributes,
                )

                return result

            except Exception as e:
                # Log error
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Failed {event_type}",
                    event_type=f"{event_type}.complete",
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                    data=attributes,
                )
                raise

            finally:
                # Restore parent span
                if trace and old_span:
                    TraceContext.set_span(old_span, trace.parent_span_id)

        return wrapper

    return decorator


class log_operation:
    """
    Context manager for logging operations with timing.

    Usage:
        with log_operation("mcp.call", data={"server": "postgres"}):
            # Do operation
            pass
    """

    def __init__(
        self,
        event_type: str,
        logger_name: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        self.event_type = event_type
        self.logger = get_logger(logger_name or "agent")
        self.data = data or {}
        self.start_time = None
        self.span_id = str(uuid.uuid4())
        self.parent_span_id = None
        self.old_span = None

    def __enter__(self):
        trace = get_current_trace()
        self.parent_span_id = trace.span_id if trace else None

        if trace:
            self.old_span = trace.span_id
            TraceContext.set_span(self.span_id, self.parent_span_id)

        self.start_time = time.time()
        self.logger.info(
            f"Starting {self.event_type}",
            event_type=f"{self.event_type}.start",
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            data=self.data,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is None:
            self.logger.info(
                f"Completed {self.event_type}",
                event_type=f"{self.event_type}.complete",
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
                duration_ms=duration_ms,
                success=True,
                data=self.data,
            )
        else:
            self.logger.error(
                f"Failed {self.event_type}",
                event_type=f"{self.event_type}.complete",
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
                duration_ms=duration_ms,
                success=False,
                error=str(exc_val),
                error_type=exc_type.__name__,
                data=self.data,
            )

        # Restore parent span
        trace = get_current_trace()
        if trace and self.old_span:
            TraceContext.set_span(self.old_span, trace.parent_span_id)
