"""
Logging decorators for automatic function tracing and operation timing.
Provides seamless observability integration with minimal code changes.
"""

import functools
import inspect
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, TypeVar

from .logging import get_logger
from .tracing import TraceContext, get_current_trace

F = TypeVar("F", bound=Callable[..., Any])

# Store metrics for the current span
_span_metrics: ContextVar[Dict[str, Any]] = ContextVar("span_metrics", default={})


class SpanMetrics:
    """Simple API to add metrics to the current span."""
    
    @staticmethod
    def add(**kwargs):
        """Add metrics that will appear in the span's complete log."""
        current = _span_metrics.get().copy()
        current.update(kwargs)
        _span_metrics.set(current)
    
    @staticmethod
    def clear():
        """Clear current span metrics."""
        _span_metrics.set({})
    
    @staticmethod
    def get() -> Dict[str, Any]:
        """Get current span metrics."""
        return _span_metrics.get()


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


def _get_logger_name(func: Callable) -> str:
    """
    Extract a sensible logger name from the function's module path.
    
    Args:
        func: The function being decorated
        
    Returns:
        A logger name based on the module path
    """
    module_parts = func.__module__.split(".")
    
    # Simple heuristic: use the last 2 parts of the module path
    # e.g., "src.agent.core" -> "agent.core"
    #       "myapp.services.auth" -> "services.auth"
    #       "main" -> "main"
    
    if len(module_parts) >= 2:
        # Skip common prefixes like 'src'
        if module_parts[0] in ('src', 'app', 'lib'):
            return '.'.join(module_parts[1:3]) if len(module_parts) > 2 else module_parts[1]
        else:
            return '.'.join(module_parts[-2:])
    else:
        return module_parts[0] if module_parts else "default"


def traced_async(
    event_type: Optional[str] = None, 
    logger_name: Optional[str] = None, 
    extract_args: bool = True
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
            # Default to last module part + function name
            module_last = func.__module__.split('.')[-1]
            event_type = f"{module_last}.{func.__name__}"

        if not logger_name:
            logger_name = _get_logger_name(func)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Clear any previous span metrics
            SpanMetrics.clear()
            
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

                # Get accumulated metrics
                span_metrics = SpanMetrics.get()
                
                # Combine original attributes with accumulated metrics
                complete_data = attributes.copy() if extract_args else {}
                complete_data.update(span_metrics)

                # Log success with enriched data
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Completed {event_type}",
                    event_type=f"{event_type}.complete",
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    duration_ms=duration_ms,
                    success=True,
                    data=complete_data,
                )

                return result

            except Exception as e:
                # Get any metrics that were accumulated before the error
                span_metrics = SpanMetrics.get()
                error_data = attributes.copy() if extract_args else {}
                error_data.update(span_metrics)
                
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
                    data=error_data,
                )
                raise

            finally:
                # Restore parent span and clear metrics
                if trace and old_span:
                    TraceContext.set_span(old_span, trace.parent_span_id)
                SpanMetrics.clear()

        return wrapper

    return decorator


def traced(
    event_type: Optional[str] = None, 
    logger_name: Optional[str] = None, 
    extract_args: bool = True
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
            module_last = func.__module__.split('.')[-1]
            event_type = f"{module_last}.{func.__name__}"

        if not logger_name:
            logger_name = _get_logger_name(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Clear any previous span metrics
            SpanMetrics.clear()
            
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

                # Get accumulated metrics
                span_metrics = SpanMetrics.get()
                
                # Combine original attributes with accumulated metrics
                complete_data = attributes.copy() if extract_args else {}
                complete_data.update(span_metrics)

                # Log success with enriched data
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Completed {event_type}",
                    event_type=f"{event_type}.complete",
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    duration_ms=duration_ms,
                    success=True,
                    data=complete_data,
                )

                return result

            except Exception as e:
                # Get any metrics that were accumulated before the error
                span_metrics = SpanMetrics.get()
                error_data = attributes.copy() if extract_args else {}
                error_data.update(span_metrics)
                
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
                    data=error_data,
                )
                raise

            finally:
                # Restore parent span and clear metrics
                if trace and old_span:
                    TraceContext.set_span(old_span, trace.parent_span_id)
                SpanMetrics.clear()

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
        # Clear any previous span metrics
        SpanMetrics.clear()
        
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
    
    def add_metrics(self, **kwargs):
        """Add metrics to be included in the complete log."""
        SpanMetrics.add(**kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        # Get accumulated metrics
        span_metrics = SpanMetrics.get()
        complete_data = self.data.copy()
        complete_data.update(span_metrics)

        if exc_type is None:
            # Success
            self.logger.info(
                f"Completed {self.event_type}",
                event_type=f"{self.event_type}.complete",
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
                duration_ms=duration_ms,
                success=True,
                data=complete_data,
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
                data=complete_data,
            )

        # Restore parent span and clear metrics
        trace = get_current_trace()
        if trace and self.old_span:
            TraceContext.set_span(self.old_span, trace.parent_span_id)
        SpanMetrics.clear()