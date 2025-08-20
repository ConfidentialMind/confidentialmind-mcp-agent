import uuid
from typing import Optional, AsyncGenerator, Any, Union
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from .config_manager import get_api_parameters, ConfigManager
from .usage_tracker import get_usage_tracker, UsageType, UsageRecord, extract_api_key_middle_part
from .tracing import get_current_trace
from .logging import get_logger
from datetime import datetime

# Set up logging
logger = get_logger("sdk.model_client")


class ConnectorNotConfiguredError(Exception):
    """Exception raised when a connector is not properly configured."""
    pass


class ModelClient:
    """
    Unified OpenAI-compatible client that automatically tracks token usage.
    Replaces the existing ClientManager pattern across applications.
    
    This is our custom model client wrapper - not to be confused with OpenAI's client.
    """
    
    def __init__(
        self, 
        config_id: str, 
        url_suffix: str = "/v1/",
        auto_track_usage: bool = True,
        auto_trace: bool = True,
        auto_log: bool = True,
        index: Optional[int] = None
    ):
        """
        Initialize the ModelClient.
        
        Args:
            config_id: Configuration ID for the connector
            url_suffix: URL suffix to append to the base URL (default: "/v1/")
            auto_track_usage: Whether to automatically track token usage
            auto_trace: Whether to automatically extract trace context
            auto_log: Whether to automatically log LLM requests and responses
            index: For array connectors, the index of the specific endpoint to use (0-based)
        """
        self.config_id = config_id
        self.url_suffix = url_suffix
        self.auto_track_usage = auto_track_usage
        self.auto_trace = auto_trace
        self.auto_log = auto_log
        self.index = index
        
        # Client management (similar to existing ClientManager)
        self._client = None
        self._last_base_url = None
        self._api_key = "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        
        # Usage tracking
        self.usage_tracker = get_usage_tracker() if auto_track_usage else None
    
    def _generate_trace_id(self) -> str:
        """Generate a new trace ID for the request"""
        return str(uuid.uuid4().hex)
    
    def _log_completion_request(self, messages: list, **kwargs):
        """Log completion request details."""
        if not self.auto_log:
            return
            
        # Extract prompt from messages
        prompt = ""
        system_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                prompt = msg.get("content", "")
        
        event_type = "llm.streaming_request" if kwargs.get("stream") else "llm.request"
        
        logger.info(
            f"LLM {'streaming ' if kwargs.get('stream') else ''}request",
            event_type=event_type,
            data={
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "system_prompt_length": len(system_prompt) if system_prompt else 0,
                "messages_count": len(messages),
                "model": kwargs.get("model", "unknown"),
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
                "stream": kwargs.get("stream", False)
            }
        )
    
    def _log_completion_response(self, response, accumulated_content: Optional[str] = None, chunk_count: Optional[int] = None):
        """Log completion response details."""
        if not self.auto_log:
            return
        
        if accumulated_content is not None:
            # Streaming response
            logger.info(
                "LLM streaming response completed",
                event_type="llm.streaming_response",
                data={
                    "total_response_length": len(accumulated_content),
                    "response_preview": (accumulated_content[:10000] + "...") if len(accumulated_content) > 10000 else accumulated_content,
                    "chunks_received": chunk_count or 0
                }
            )
        else:
            # Non-streaming response
            content = response.choices[0].message.content if response.choices else ""
            logger.info(
                "LLM response",
                event_type="llm.response",
                data={
                    "response_length": len(content) if content else 0,
                    "response_preview": (content[:10000] + "...") if content and len(content) > 10000 else content,
                    "usage_prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                    "usage_completion_tokens": getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0,
                    "usage_total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
                }
            )
    
    def _log_embedding_request(self, **kwargs):
        """Log embedding request details."""
        if not self.auto_log:
            return
        
        input_text = kwargs.get("input", "")
        if isinstance(input_text, list):
            input_text = " ".join(str(item) for item in input_text)
        
        logger.info(
            "Embedding request",
            event_type="embedding.request", 
            data={
                "input_length": len(str(input_text)),
                "input_preview": str(input_text)[:200] + "..." if len(str(input_text)) > 200 else str(input_text),
                "model": kwargs.get("model", "unknown"),
                "dimensions": kwargs.get("dimensions"),
            }
        )
    
    def _log_embedding_response(self, response):
        """Log embedding response details."""
        if not self.auto_log:
            return
        
        embeddings_count = len(response.data) if hasattr(response, 'data') else 0
        embedding_dimensions = len(response.data[0].embedding) if embeddings_count > 0 else 0
        
        logger.info(
            "Embedding response",
            event_type="embedding.response",
            data={
                "embeddings_count": embeddings_count,
                "embedding_dimensions": embedding_dimensions,
                "usage_prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                "usage_total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
            }
        )
    
    def get_client(self) -> AsyncOpenAI:
        """
        Get an OpenAI client, creating a new one only if connection details have changed.
        
        Raises:
            ConnectorNotConfiguredError: If the connector has not been properly configured.
        """
        # Get current connection details, using index if provided for array connectors
        current_base_url, headers = get_api_parameters(self.config_id, index=self.index)
        
        # Check if connector is configured
        if not current_base_url:
            if self.index is not None:
                logger.error(f"Connector for {self.config_id}[{self.index}] is not configured - missing URL")
                raise ConnectorNotConfiguredError(
                    f"The connector for '{self.config_id}' at index {self.index} has not been configured. "
                    f"Please set up the connector in the portal before using this feature."
                )
            else:
                logger.error(f"Connector for {self.config_id} is not configured - missing URL")
                raise ConnectorNotConfiguredError(
                    f"The connector for '{self.config_id}' has not been configured. "
                    f"Please set up the connector in the portal before using this feature."
                )
        
        current_base_url = current_base_url + self.url_suffix if current_base_url else None
        
        # Extract API key from headers if present
        # This is only used for local development, when we access the model through the API
        # from outside the cluster (.env must have the url and api key set)
        if headers and "Authorization" in headers:
            self._api_key = headers["Authorization"].split(" ")[1]
        
        # Create a new client only if the URL or API key has changed, or if no client exists
        if self._client is None or current_base_url != self._last_base_url:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=current_base_url,
            )
            self._last_base_url = current_base_url
            if self.index is not None:
                logger.info(f"Created new OpenAI client for {self.config_id}[{self.index}]")
            else:
                logger.info(f"Created new OpenAI client for {self.config_id}")
        
        return self._client
    
    
    async def _report_embedding_usage(self, usage: Any, trace_id: str, user_api_key: Optional[str] = None) -> bool:
        """
        Report embedding usage data to the usage tracker.
        
        Args:
            usage: Usage object from OpenAI embeddings response
            trace_id: Trace ID for this request
            user_api_key: User's API key from request header (for usage attribution)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.usage_tracker:
            return False
        
        try:
            config_manager = ConfigManager()
            origin_stack_id = config_manager.service_id or 'unknown'
            
            # Safely get target stack ID
            try:
                target_stack_id = config_manager.getStackIdForConnector(self.config_id)
                if isinstance(target_stack_id, list):
                    # For array connectors, use the index if provided
                    if self.index is not None and 0 <= self.index < len(target_stack_id):
                        target_stack_id = target_stack_id[self.index]
                    else:
                        # If no index provided or invalid index, use a composite identifier
                        target_stack_id = f'array-{self.config_id}'
                elif not target_stack_id:  # Handles None and empty string
                    target_stack_id = 'unknown'
            except (AttributeError, Exception):
                # In local dev mode, connectors might not be fully initialized
                target_stack_id = f'local-{self.config_id}'
            
            api_key_middle = extract_api_key_middle_part(user_api_key)
            timestamp = int(datetime.now().timestamp())
            
            usage_records = []
            
            # For embeddings, we typically have prompt_tokens and total_tokens
            # Map this to embedding usage
            if hasattr(usage, 'prompt_tokens') and usage.prompt_tokens:
                usage_records.append(UsageRecord(
                    originId=origin_stack_id,
                    targetId=target_stack_id or 'unknown',
                    timestamp=timestamp,
                    apiKey=api_key_middle,
                    traceId=trace_id,
                    type=UsageType.EMBEDDING_USAGE.value,
                    value=usage.prompt_tokens  # Use prompt_tokens as embedding usage count
                ))
            
            if usage_records:
                return await self.usage_tracker.report_usage(usage_records)
            
            return False  # No usage data to report
            
        except Exception as e:
            logger.error(f"Error reporting embedding usage: {e}")
            return False

    async def _report_completion_usage(self, usage: Any, trace_id: str, user_api_key: Optional[str] = None) -> bool:
        """
        Report completion usage data to the usage tracker.
        
        Args:
            usage: Usage object from OpenAI response
            trace_id: Trace ID for this request
            user_api_key: User's API key from request header (for usage attribution)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.usage_tracker:
            return False
        
        try:
            config_manager = ConfigManager()
            origin_stack_id = config_manager.service_id or 'unknown'
            
            # Safely get target stack ID
            try:
                target_stack_id = config_manager.getStackIdForConnector(self.config_id)
                if isinstance(target_stack_id, list):
                    # For array connectors, use the index if provided
                    if self.index is not None and 0 <= self.index < len(target_stack_id):
                        target_stack_id = target_stack_id[self.index]
                    else:
                        # If no index provided or invalid index, use a composite identifier
                        target_stack_id = f'array-{self.config_id}'
                elif not target_stack_id:  # Handles None and empty string
                    target_stack_id = 'unknown'
            except (AttributeError, Exception):
                # In local dev mode, connectors might not be fully initialized
                target_stack_id = f'local-{self.config_id}'
            
            api_key_middle = extract_api_key_middle_part(user_api_key)
            timestamp = int(datetime.now().timestamp())
            
            usage_records = []
            
            # Add input tokens usage
            if hasattr(usage, 'prompt_tokens') and usage.prompt_tokens:
                usage_records.append(UsageRecord(
                    originId=origin_stack_id,
                    targetId=target_stack_id or 'unknown',
                    timestamp=timestamp,
                    apiKey=api_key_middle,
                    traceId=trace_id,
                    type=UsageType.INPUT_TOKENS.value,
                    value=usage.prompt_tokens
                ))
            
            # Add output tokens usage
            if hasattr(usage, 'completion_tokens') and usage.completion_tokens:
                usage_records.append(UsageRecord(
                    originId=origin_stack_id,
                    targetId=target_stack_id or 'unknown',
                    timestamp=timestamp,
                    apiKey=api_key_middle,
                    traceId=trace_id,
                    type=UsageType.OUTPUT_TOKENS.value,
                    value=usage.completion_tokens
                ))
            
            if usage_records:
                return await self.usage_tracker.report_usage(usage_records)
            
            return False  # No usage data to report
            
        except Exception as e:
            logger.error(f"Error reporting completion usage: {e}")
            return False
    
    async def _process_streaming_response(
        self, 
        response_stream: AsyncGenerator, 
        trace_id: str, 
        forward_usage_chunk: bool,
        user_api_key: Optional[str] = None
    ) -> AsyncGenerator[Any, None]:
        """
        Process streaming response, extract usage data, and optionally filter usage chunk.
        
        Args:
            response_stream: The original streaming response
            trace_id: Trace ID for this request
            forward_usage_chunk: Whether to forward the usage chunk to the caller
            user_api_key: User's API key from request header (for usage attribution)
            
        Yields:
            Response chunks (with usage chunk filtered based on forward_usage_chunk)
        """
        usage_data = None
        accumulated_content = ""
        chunk_count = 0
        
        async for chunk in response_stream:
            # Check if this is the final usage chunk
            if (hasattr(chunk, 'usage') and chunk.usage is not None and 
                hasattr(chunk, 'choices') and len(chunk.choices) == 0):
                
                # This is the usage chunk - extract the data
                usage_data = chunk.usage
                
                # Only forward to caller if they originally requested usage
                if forward_usage_chunk:
                    yield chunk
                # Otherwise, we consume this chunk and don't forward it
                
            else:
                # Regular content chunk - track content for logging
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated_content += delta.content
                        chunk_count += 1
                
                # Always forward regular chunks
                yield chunk
        
        # Log streaming response completion if auto_log enabled
        if self.auto_log:
            self._log_completion_response(None, accumulated_content, chunk_count)
        
        # Report usage data after stream completes
        if self.auto_track_usage and usage_data:
            await self._report_completion_usage(usage_data, trace_id, user_api_key)
    
    async def completions_with_usage(
        self, 
        trace_id: Optional[str] = None,
        user_api_key: Optional[str] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[Any, None]]:
        """
        Create chat completions with automatic usage tracking.
        Clear naming to distinguish from openai.chat.completions.create()
        
        Args:
            trace_id: Optional trace ID for request correlation
            user_api_key: User's API key from request header (for usage attribution)
            **kwargs: Arguments to pass to the OpenAI API
            
        Returns:
            ChatCompletion object for non-streaming requests, or
            AsyncGenerator for streaming requests
        """
        # Auto-extract from trace context if enabled and not provided
        if self.auto_trace:
            trace = get_current_trace()
            if trace:
                trace_id = trace_id or trace.trace_id
                user_api_key = user_api_key or trace.api_key
        
        request_trace_id = trace_id or self._generate_trace_id()
        
        # Log the request if auto_log enabled
        if self.auto_log:
            self._log_completion_request(kwargs.get('messages', []), **kwargs)
        
        # Check if original request wanted usage data (only relevant for streaming)
        original_wants_usage = kwargs.get('stream_options', {}).get('include_usage', False)
        
        # Always enable usage tracking internally for streaming requests
        if kwargs.get('stream'):
            if 'stream_options' not in kwargs:
                kwargs['stream_options'] = {}
            kwargs['stream_options']['include_usage'] = True
        
        # Make the API call
        response = await self.get_client().chat.completions.create(**kwargs)
        
        if kwargs.get('stream'):
            # Return custom async iterator that handles usage chunk filtering
            return self._process_streaming_response(
                response, 
                request_trace_id, 
                original_wants_usage,
                user_api_key
            )
        else:
            # Non-streaming: log response and report usage
            if self.auto_log:
                self._log_completion_response(response)
            if self.auto_track_usage and hasattr(response, 'usage'):
                await self._report_completion_usage(response.usage, request_trace_id, user_api_key)
            return response
    
    async def embeddings_with_usage(
        self, 
        trace_id: Optional[str] = None,
        user_api_key: Optional[str] = None,
        **kwargs
    ) -> CreateEmbeddingResponse:
        """
        Create embeddings with automatic usage tracking.
        
        Args:
            trace_id: Optional trace ID for request correlation
            user_api_key: User's API key from request header (for usage attribution)
            **kwargs: Arguments to pass to the OpenAI embeddings API
            
        Returns:
            CreateEmbeddingResponse object
        """
        # Auto-extract from trace context if enabled and not provided
        if self.auto_trace:
            trace = get_current_trace()
            if trace:
                trace_id = trace_id or trace.trace_id
                user_api_key = user_api_key or trace.api_key
        
        request_trace_id = trace_id or self._generate_trace_id()
        
        # Log the request if auto_log enabled
        if self.auto_log:
            self._log_embedding_request(**kwargs)
        
        # Make the API call
        response = await self.get_client().embeddings.create(**kwargs)
        
        # Log the response if auto_log enabled
        if self.auto_log:
            self._log_embedding_response(response)
        
        # Report usage if available and tracking is enabled
        if self.auto_track_usage and hasattr(response, 'usage'):
            await self._report_embedding_usage(response.usage, request_trace_id, user_api_key)
        
        return response