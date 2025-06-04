"""
Observability module for Langfuse integration.

This module provides utilities for tracing agent execution with Langfuse,
including decorators, context management, and configuration.
"""

import logging
import os
from typing import Any, Dict, Optional

from langfuse.decorators import langfuse_context, observe

logger = logging.getLogger(__name__)


def initialize_langfuse() -> None:
    """
    Initialize Langfuse with environment variables.

    This should be called at application startup to configure Langfuse.
    Falls back gracefully if keys are not provided.
    """
    # Check if Langfuse is explicitly disabled
    if os.environ.get("LANGFUSE_ENABLED", "true").lower() == "false":
        logger.info("Langfuse observability is disabled")
        langfuse_context.configure(enabled=False)
        return

    # Check for required keys
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse API keys not found. Observability will be disabled. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable."
        )
        langfuse_context.configure(enabled=False)
        return

    # Configure Langfuse
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    debug = os.environ.get("LANGFUSE_DEBUG", "false").lower() == "true"

    langfuse_context.configure(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        debug=debug,
        enabled=True,
    )

    logger.info(f"Langfuse observability initialized with host: {host}")


def update_trace_metadata(
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Update the current trace with metadata.

    Args:
        name: Name for the trace
        user_id: User identifier
        session_id: Session/conversation identifier
        tags: List of tags
        metadata: Additional metadata dictionary
        **kwargs: Additional trace attributes
    """
    try:
        update_params = {}

        if name:
            update_params["name"] = name
        if user_id:
            update_params["user_id"] = user_id
        if session_id:
            update_params["session_id"] = session_id
        if tags:
            update_params["tags"] = tags
        if metadata:
            # Ensure metadata is serializable
            try:
                import json
                json.dumps(metadata)
                update_params["metadata"] = metadata
            except (TypeError, ValueError) as e:
                logger.warning(f"Metadata not serializable, skipping: {e}")
                # Try to extract serializable parts
                safe_metadata = {}
                for k, v in metadata.items():
                    try:
                        json.dumps({k: v})
                        safe_metadata[k] = v
                    except:
                        safe_metadata[k] = str(v)
                if safe_metadata:
                    update_params["metadata"] = safe_metadata

        # Add any additional kwargs
        update_params.update(kwargs)

        if update_params:
            langfuse_context.update_current_trace(**update_params)
    except Exception as e:
        logger.debug(f"Failed to update trace metadata: {e}")


def update_observation_metadata(
    name: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Update the current observation with metadata.

    Args:
        name: Name for the observation
        input: Input data to log
        output: Output data to log
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        status_message: Status message
        metadata: Additional metadata dictionary
        **kwargs: Additional observation attributes
    """
    try:
        import json
        update_params = {}

        if name:
            update_params["name"] = name
        if input is not None:
            # Ensure input is serializable
            try:
                json.dumps(input)
                update_params["input"] = input
            except (TypeError, ValueError):
                logger.warning("Input not serializable, converting to string")
                update_params["input"] = str(input)
        if output is not None:
            # Ensure output is serializable
            try:
                json.dumps(output)
                update_params["output"] = output
            except (TypeError, ValueError):
                logger.warning("Output not serializable, converting to string")
                update_params["output"] = str(output)
        if level:
            update_params["level"] = level
        if status_message:
            update_params["status_message"] = status_message
        if metadata:
            # Ensure metadata is serializable
            try:
                json.dumps(metadata)
                update_params["metadata"] = metadata
            except (TypeError, ValueError) as e:
                logger.warning(f"Metadata not serializable, skipping: {e}")
                # Try to extract serializable parts
                safe_metadata = {}
                for k, v in metadata.items():
                    try:
                        json.dumps({k: v})
                        safe_metadata[k] = v
                    except:
                        safe_metadata[k] = str(v)
                if safe_metadata:
                    update_params["metadata"] = safe_metadata

        # Add any additional kwargs
        update_params.update(kwargs)

        if update_params:
            langfuse_context.update_current_observation(**update_params)
    except Exception as e:
        logger.debug(f"Failed to update observation metadata: {e}")


def capture_llm_generation(
    model: str,
    input_messages: Any,
    output_content: Any,
    usage: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> None:
    """
    Capture LLM generation details for the current observation.

    Args:
        model: Model name/identifier
        input_messages: Input messages or prompt
        output_content: Generated content
        usage: Token usage details (input, output, total)
        **kwargs: Additional generation parameters
    """
    try:
        update_params = {
            "model": model,
            "input": input_messages,
            "output": output_content,
        }

        if usage:
            update_params["usage"] = usage

        # Add model parameters if provided
        model_params = kwargs.get("model_parameters", {})
        if model_params:
            update_params["model_parameters"] = model_params

        # Add any additional metadata
        if kwargs.get("metadata"):
            update_params["metadata"] = kwargs["metadata"]

        langfuse_context.update_current_observation(**update_params)
    except Exception as e:
        logger.debug(f"Failed to capture LLM generation: {e}")


def flush_traces() -> None:
    """
    Flush all pending traces to Langfuse.

    This should be called before application shutdown or in serverless
    environments to ensure all traces are sent.
    """
    try:
        langfuse_context.flush()
        logger.debug("Flushed Langfuse traces")
    except Exception as e:
        logger.debug(f"Failed to flush traces: {e}")


# Re-export the observe decorator for convenience
__all__ = [
    "observe",
    "langfuse_context",
    "initialize_langfuse",
    "update_trace_metadata",
    "update_observation_metadata",
    "capture_llm_generation",
    "flush_traces",
]

