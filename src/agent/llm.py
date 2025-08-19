from typing import AsyncGenerator, Optional

from confidentialmind_core.model_client import ConnectorNotConfiguredError, ModelClient
from confidentialmind_core import get_current_trace, get_logger, traced_async


class LLMConnector:
    """
    A wrapper around the SDK's ModelClient that provides a simplified interface
    for the agent while automatically tracking token usage.

    This connector handles dynamic configuration changes - the LLM can be:
    - Not configured initially
    - Configured/reconfigured at runtime via the web portal
    - Changed to point to different models
    - Cleared (unconfigured)

    The underlying ModelClient and ConfigManager handle configuration updates automatically.
    """

    def __init__(self, config_id: str = "LLM"):
        """
        Initialize the LLM connector with an SDK config ID.

        Args:
            config_id: The connector configuration ID in the confidentialmind system
        """
        self.config_id = config_id
        self._model_client: Optional[ModelClient] = None

        # Use structlog logger
        self.logger = get_logger("agent.llm")

        self.logger.info(f"LLMConnector: Initialized for config_id={config_id}")

    async def initialize(self) -> bool:
        """
        Initialize the connector by creating a ModelClient instance.

        Note: This will succeed even if no LLM is configured yet, allowing
        the service to start and wait for configuration via the portal.

        Returns:
            bool: True if initialization was successful (not necessarily connected)
        """
        try:
            # Create ModelClient instance with automatic usage tracking and tracing
            # The ModelClient will handle configuration updates internally
            self._model_client = ModelClient(
                config_id=self.config_id,
                url_suffix="/v1/",
                auto_track_usage=True,
                auto_trace=True,  # Enable automatic trace context extraction
            )

            # Test if an LLM is currently configured
            try:
                self._model_client.get_client()
                self.logger.info(
                    f"LLMConnector: ModelClient initialized and LLM is configured "
                    f"for {self.config_id}"
                )
            except ConnectorNotConfiguredError:
                self.logger.info(
                    f"LLMConnector: ModelClient initialized but no LLM configured yet "
                    f"for {self.config_id}"
                )

            # Initialization is successful even if no LLM is configured
            # This allows the service to start and wait for configuration
            return True

        except Exception as e:
            self.logger.error(f"LLMConnector: Error initializing ModelClient: {e}")
            return False

    def is_connected(self) -> bool:
        """
        Check if LLM is currently connected.

        This checks the current configuration state - it may change at runtime
        as users configure/reconfigure the connector via the portal.
        """
        if not self._model_client:
            return False

        try:
            # Try to get the client - this will check current configuration
            self._model_client.get_client()
            return True
        except ConnectorNotConfiguredError:
            return False
        except Exception:
            return False

    @traced_async("llm.generate")
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from the LLM based on the prompt.

        If the LLM is not connected, returns a message indicating the LLM is unavailable.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        if not self._model_client:
            return (
                "I'm currently unable to generate a response as my language model "
                "service is not initialized. Please contact support."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Log the LLM request
        self.logger.info(
            "LLM request",
            event_type="llm.request",
            data={
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "system_prompt_length": len(system_prompt) if system_prompt else 0,
                "messages_count": len(messages),
                "model": "cm-llm",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )

        try:
            # Use ModelClient's completions_with_usage method
            # This will automatically extract trace context and use the latest configuration
            response = await self._model_client.completions_with_usage(
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                model="cm-llm",
            )

            # Extract the response content
            generated_content = response.choices[0].message.content
            
            # Log the LLM response
            self.logger.info(
                "LLM response",
                event_type="llm.response",
                data={
                    "response_length": len(generated_content) if generated_content else 0,
                    "response_preview": (generated_content[:200] + "...") if generated_content and len(generated_content) > 200 else generated_content,
                    "usage_prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                    "usage_completion_tokens": getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0,
                    "usage_total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
                }
            )
            
            return generated_content

        except ConnectorNotConfiguredError as e:
            self.logger.debug(f"LLMConnector: No LLM configured: {e}")
            return (
                "I'm currently unable to generate a response as my language model "
                "connection is unavailable. Please configure an LLM in the portal or "
                "contact support."
            )
        except Exception as e:
            self.logger.error(f"LLMConnector: Error generating text: {e}")
            return (
                "I'm currently unable to generate a response due to a technical issue "
                "with my language model service. Please try again later."
            )

    @traced_async("llm.generate_streaming")
    async def generate_streaming(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming text from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Yields:
            String chunks of the generated response
        """
        if not self._model_client:
            yield (
                "I'm currently unable to generate a response as my language model "
                "service is not initialized. Please contact support."
            )
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Log the LLM streaming request
        self.logger.info(
            "LLM streaming request",
            event_type="llm.streaming_request",
            data={
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "system_prompt_length": len(system_prompt) if system_prompt else 0,
                "messages_count": len(messages),
                "model": "cm-llm",
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": True
            }
        )

        try:
            # Use ModelClient's completions_with_usage method with streaming
            # This will automatically extract trace context and use the latest configuration
            response_stream = await self._model_client.completions_with_usage(
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                model="cm-llm",
                stream=True,
            )

            # Process the streaming response
            accumulated_content = ""
            chunk_count = 0
            
            async for chunk in response_stream:
                # Extract content from chunk if available
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated_content += delta.content
                        chunk_count += 1
                        yield delta.content
            
            # Log the streaming response completion
            self.logger.info(
                "LLM streaming response completed",
                event_type="llm.streaming_response",
                data={
                    "total_response_length": len(accumulated_content),
                    "response_preview": (accumulated_content[:200] + "...") if len(accumulated_content) > 200 else accumulated_content,
                    "chunks_received": chunk_count
                }
            )

        except ConnectorNotConfiguredError as e:
            self.logger.debug(f"LLMConnector: No LLM configured: {e}")
            yield (
                "I'm currently unable to generate a response as my language model "
                "connection is unavailable. Please configure an LLM in the portal or "
                "contact support."
            )
        except Exception as e:
            self.logger.error(f"LLMConnector: Error during streaming generation: {e}")
            yield (
                "\n\nI'm currently unable to generate a response due to a technical "
                "issue with my language model service. Please try again later."
            )

    async def close(self):
        """Close the connector and release resources"""
        self.logger.info("LLMConnector: Closing ModelClient connection")
        # ModelClient handles its own cleanup internally
        self._model_client = None
        self.logger.info("LLMConnector: Connection closed")
