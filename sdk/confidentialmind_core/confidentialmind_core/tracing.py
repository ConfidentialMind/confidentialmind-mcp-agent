"""
Distributed tracing support for ConfidentialMind SDK.
Provides context propagation and automatic integration with ModelClient.
"""

import hashlib
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

# Thread-safe context variable for trace information
_trace_context: ContextVar[Optional["TraceInfo"]] = ContextVar("trace_context", default=None)


@dataclass
class TraceInfo:
    """Immutable trace information for distributed tracing."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    session_id: str
    api_key_hash: str  # Keep for backward compatibility with logging
    api_key: Optional[str]  # Original API key for usage tracking
    origin_service: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def create_child_span(self) -> "TraceInfo":
        """Create a child span with this span as parent."""
        return TraceInfo(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4().hex[:16]),
            parent_span_id=self.span_id,
            session_id=self.session_id,
            api_key_hash=self.api_key_hash,
            api_key=self.api_key,
            origin_service=self.origin_service,
        )

    def to_headers(self) -> Dict[str, str]:
        """Convert trace info to HTTP headers for propagation."""
        return {
            "X-Trace-ID": self.trace_id,
            "X-Span-ID": self.span_id,
            "X-Parent-Span-ID": self.parent_span_id or "",
            "X-Session-ID": self.session_id,
            "X-Origin-Service": self.origin_service,
        }

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "api_key_hash": self.api_key_hash,  # Match RAG naming
            "origin_service": self.origin_service,
        }


class TraceContext:
    """Manages trace context for distributed tracing."""

    @staticmethod
    def initialize(
        session_id: str,
        api_key: Optional[str] = None,
        origin_service: str = "unknown",
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> TraceInfo:
        """
        Initialize a new trace context.

        Args:
            session_id: Session or conversation ID
            api_key: API key (will be hashed for logging, stored as-is for usage)
            origin_service: Service that initiated the request
            trace_id: Optional existing trace ID
            parent_span_id: Optional parent span ID
            span_id: Optional span ID (auto-generated if not provided)

        Returns:
            TraceInfo object that was set in context
        """
        trace_info = TraceInfo(
            trace_id=trace_id or str(uuid.uuid4().hex),
            span_id=span_id or str(uuid.uuid4().hex[:16]),
            parent_span_id=parent_span_id,
            session_id=session_id,
            api_key_hash=hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "",
            api_key=api_key,  # Store original for usage tracking
            origin_service=origin_service,
        )

        _trace_context.set(trace_info)
        return trace_info

    @staticmethod
    def from_headers(headers: Dict[str, str], api_key: Optional[str] = None) -> Optional[TraceInfo]:
        """
        Initialize trace context from HTTP headers.

        Args:
            headers: HTTP headers containing trace information
            api_key: Optional API key to include

        Returns:
            TraceInfo if headers contain trace information, None otherwise
        """
        trace_id = headers.get("X-Trace-ID")
        if not trace_id:
            return None

        trace_info = TraceInfo(
            trace_id=trace_id,
            span_id=headers.get("X-Span-ID", str(uuid.uuid4().hex[:16])),
            parent_span_id=headers.get("X-Parent-Span-ID") or None,
            session_id=headers.get("X-Session-ID", "unknown"),
            api_key_hash=hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "",
            api_key=api_key,  # Store original for usage tracking
            origin_service=headers.get("X-Origin-Service", "unknown"),
        )

        _trace_context.set(trace_info)
        return trace_info

    @staticmethod
    def get() -> Optional[TraceInfo]:
        """Get current trace context."""
        return _trace_context.get()

    @staticmethod
    def set(trace_info: TraceInfo):
        """Set trace context."""
        _trace_context.set(trace_info)

    @staticmethod
    def set_span(span_id: str, parent_span_id: Optional[str] = None):
        """Update current span in context."""
        trace = _trace_context.get()
        if trace:
            updated_trace = TraceInfo(
                trace_id=trace.trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id or trace.span_id,
                session_id=trace.session_id,
                api_key_hash=trace.api_key_hash,
                api_key=trace.api_key,
                origin_service=trace.origin_service,
            )
            _trace_context.set(updated_trace)

    @staticmethod
    def clear():
        """Clear current trace context."""
        _trace_context.set(None)
    
    @staticmethod
    def override(api_key: Optional[str] = None, session_id: Optional[str] = None):
        """
        Context manager to temporarily override trace context values.
        
        Usage:
            with TraceContext.override(api_key="service-key"):
                # Use service API key for this block
                await internal_service_call()
            # Original context restored
        """
        return _TraceContextOverride(api_key=api_key, session_id=session_id)


class _TraceContextOverride:
    """Context manager for temporarily overriding trace context values."""
    
    def __init__(self, api_key: Optional[str] = None, session_id: Optional[str] = None):
        self.override_api_key = api_key
        self.override_session_id = session_id
        self.original_trace = None
    
    def __enter__(self):
        self.original_trace = _trace_context.get()
        if self.original_trace:
            # Create modified trace with overrides
            modified_trace = TraceInfo(
                trace_id=self.original_trace.trace_id,
                span_id=self.original_trace.span_id,
                parent_span_id=self.original_trace.parent_span_id,
                session_id=self.override_session_id or self.original_trace.session_id,
                api_key_hash=hashlib.sha256(self.override_api_key.encode()).hexdigest()[:16] 
                    if self.override_api_key else self.original_trace.api_key_hash,
                api_key=self.override_api_key or self.original_trace.api_key,
                origin_service=self.original_trace.origin_service,
            )
            _trace_context.set(modified_trace)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        _trace_context.set(self.original_trace)


def get_current_trace() -> Optional[TraceInfo]:
    """Convenience function to get current trace context."""
    return TraceContext.get()


def copy_context():
    """
    Copy current context for use in thread pools or background tasks.
    
    Usage:
        ctx = copy_context()
        
        def background_task():
            with ctx:
                # Context is available here
                do_work()
        
        executor.submit(background_task)
    """
    current_trace = _trace_context.get()
    
    class ContextCopy:
        def __enter__(self):
            _trace_context.set(current_trace)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            _trace_context.set(None)
    
    return ContextCopy()