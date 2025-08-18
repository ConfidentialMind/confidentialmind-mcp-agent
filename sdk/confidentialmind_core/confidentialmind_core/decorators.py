"""
Logging decorators for automatic function tracing and operation timing.
Provides seamless observability integration with minimal code changes.
"""

import functools
import inspect
import time
import uuid
from typing import Any, Callable, Dict, Optional, TypeVar

from .logging import get_logger
from .tracing import TraceContext, get_current_trace

F = TypeVar("F", bound=Callable[..., Any])


def extract_function_attributes(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """
    Extract function attributes for logging based on signature.

    Args:
        func: Function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary of extracted attributes
    """
    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Extract meaningful attributes (skip 'self' and large objects)
        attributes = {}
        for name, value in bound_args.arguments.items():
            if name == "self":
                continue

            # Convert value to loggable format
            if isinstance(value, (str, int, float, bool, type(None))):
                attributes[name] = value
            elif isinstance(value, (list, dict)):
                # Include size for collections
                attributes[f"{name}_size"] = len(value)
                if isinstance(value, list) and value and isinstance(value[0], str):
                    # For string lists, include first item as preview
                    attributes[f"{name}_preview"] = value[0][:100] + "..." if len(value[0]) > 100 else value[0]
            else:
                # For other objects, include type info
                attributes[f"{name}_type"] = type(value).__name__

        return attributes
    except Exception:
        # Fallback to empty dict if extraction fails
        return {}


def traced_async(
    event_type: Optional[str] = None, logger_name: Optional[str] = None, extract_args: bool = True
) -> Callable[[F], F]:
    """
    Decorator for tracing async functions with automatic span management.

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
                if module_parts[1] == "baserag":
                    if len(module_parts) > 3:
                        # src.baserag.implementations.chunkers -> baserag.chunkers
                        if module_parts[2] == "implementations":
                            logger_name = f"baserag.{module_parts[3]}"
                        else:
                            logger_name = f"baserag.{module_parts[2]}"
                    else:
                        logger_name = "baserag.core"
                elif module_parts[1] == "agent":
                    logger_name = "agent." + module_parts[2] if len(module_parts) > 2 else "agent.core"
                elif module_parts[1] == "tools":
                    logger_name = module_parts[2] if len(module_parts) > 2 else "tools"
                elif module_parts[1] == "shared":
                    logger_name = "shared." + module_parts[2] if len(module_parts) > 2 else "shared"
                else:
                    logger_name = module_parts[1]
            else:
                logger_name = module_parts[-1] if module_parts else "default"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)

            # Create child span for this operation
            trace = get_current_trace()
            span_id = str(uuid.uuid4().hex[:16])
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
                if module_parts[1] == "baserag":
                    if len(module_parts) > 3:
                        # src.baserag.implementations.chunkers -> baserag.chunkers
                        if module_parts[2] == "implementations":
                            logger_name = f"baserag.{module_parts[3]}"
                        else:
                            logger_name = f"baserag.{module_parts[2]}"
                    else:
                        logger_name = "baserag.core"
                elif module_parts[1] == "agent":
                    logger_name = "agent." + module_parts[2] if len(module_parts) > 2 else "agent.core"
                elif module_parts[1] == "tools":
                    logger_name = module_parts[2] if len(module_parts) > 2 else "tools"
                elif module_parts[1] == "shared":
                    logger_name = "shared." + module_parts[2] if len(module_parts) > 2 else "shared"
                else:
                    logger_name = module_parts[1]
            else:
                logger_name = module_parts[-1] if module_parts else "default"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)

            # Create child span for this operation
            trace = get_current_trace()
            span_id = str(uuid.uuid4().hex[:16])
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
        with log_operation("database.query", "database", data={"table": "users"}):
            result = execute_query()
    """

    def __init__(
        self,
        event_type: str,
        logger_name: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        self.event_type = event_type
        self.logger_name = logger_name or "default"
        self.data = data or {}
        self.logger = get_logger(self.logger_name)
        self.start_time = None
        self.span_id = str(uuid.uuid4().hex[:16])
        self.parent_span_id = None
        self.old_span = None

    def __enter__(self):
        # Create child span
        trace = get_current_trace()
        self.parent_span_id = trace.span_id if trace else None

        # Set current span
        if trace:
            self.old_span = trace.span_id
            TraceContext.set_span(self.span_id, self.parent_span_id)

        self.start_time = time.time()

        # Log start event
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
            # Success
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
            # Error
            self.logger.error(
                f"Failed {self.event_type}",
                event_type=f"{self.event_type}.complete",
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
                duration_ms=duration_ms,
                success=False,
                error=str(exc_val),
                error_type=exc_type.__name__ if exc_type else "Unknown",
                data=self.data,
            )

        # Restore parent span
        trace = get_current_trace()
        if trace and self.old_span:
            TraceContext.set_span(self.old_span, trace.parent_span_id)