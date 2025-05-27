import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Optional

import aiohttp
from confidentialmind_core.config_manager import load_environment

from src.agent.connectors import ConnectorConfigManager

logger = logging.getLogger(__name__)


class LLMConnector:
    """
    A lightweight LLM connector that uses the confidentialmind SDK to manage connections.
    """

    def __init__(self, config_id: str = "LLM"):
        """
        Initialize the LLM connector with an SDK config ID.

        Args:
            config_id: The connector configuration ID in the confidentialmind system
        """
        self.config_id = config_id
        self._last_base_url = None
        self._last_headers = None
        self._session = None
        self._background_fetch_task = None
        self._is_connected = False

        # Determine if running in stack deployment mode
        load_environment()
        self._is_stack_deployment = (
            os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"
        )

        logger.info(
            f"LLMConnector: Initialized in {'stack deployment' if self._is_stack_deployment else 'local config'} mode for config_id={config_id}"
        )

    async def _start_background_polling(self):
        """Start background polling for LLM URL if in stack deployment mode."""
        if self._background_fetch_task is not None and not self._background_fetch_task.done():
            logger.info("LLMConnector: Background polling already started")
            return  # Already polling

        if self._is_stack_deployment:
            logger.info("LLMConnector: Starting background polling for LLM URL")
            self._background_fetch_task = asyncio.create_task(self._poll_for_url_in_background())

    async def _poll_for_url_in_background(self):
        """Continuously poll for LLM URL and update when available."""
        connector_manager = ConnectorConfigManager()
        await connector_manager.initialize(register_connectors=False)

        retry_count = 0
        max_retry_log = 10  # Log less frequently after this many retries

        while True:
            try:
                current_base_url, headers = await connector_manager.fetch_llm_url(self.config_id)

                # Only log at appropriate intervals to reduce noise
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    if current_base_url:
                        logger.debug(
                            f"LLMConnector: Poll {retry_count}: Found URL {current_base_url}"
                        )
                    else:
                        logger.debug(f"LLMConnector: Poll {retry_count}: No URL available")

                if current_base_url and current_base_url != self._last_base_url:
                    logger.info("LLMConnector: Found new LLM URL, updating connection")
                    self._last_base_url = current_base_url
                    self._last_headers = headers or {}

                    # Create new session with updated headers
                    if self._session:
                        await self._session.close()
                    self._session = aiohttp.ClientSession(headers=self._last_headers)
                    self._is_connected = True
                    logger.info("LLMConnector: Successfully connected to LLM in background")
            except Exception as e:
                # Log less frequently as retries increase
                if retry_count < max_retry_log or retry_count % 10 == 0:
                    logger.error(f"LLMConnector: Error in background polling: {e}")

            retry_count += 1
            # Exponential backoff with a maximum wait time
            wait_time = min(30, 5 * (1.5 ** min(retry_count, 5)))
            await asyncio.sleep(wait_time)

    async def initialize(self) -> bool:
        """
        Initialize the connector by fetching API parameters.

        This method will set up background polling for URL changes in stack deployment mode
        and attempt to establish an initial connection. It will not fail if the connection
        is not available, allowing the agent to start without an LLM connection.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Get connection details using ConnectorConfigManager for consistency
            connector_manager = ConnectorConfigManager()

            # In stack mode, make sure connectors are registered
            if self._is_stack_deployment:
                await connector_manager.initialize(register_connectors=True)
            else:
                await connector_manager.initialize(register_connectors=False)

            current_base_url, headers = await connector_manager.fetch_llm_url(self.config_id)

            # Start background polling for URL changes
            if self._is_stack_deployment:
                await self._start_background_polling()

            # Check if connector is configured
            if not current_base_url:
                logger.warning(
                    f"LLMConnector: Connector for {self.config_id} is not configured - missing URL"
                )
                self._is_connected = False
                return False

            # Use the URL and headers
            self._last_base_url = current_base_url
            self._last_headers = headers or {}

            # Create aiohttp session
            if self._session is None:
                self._session = aiohttp.ClientSession(headers=self._last_headers)

            logger.info(f"LLMConnector: Successfully initialized for {self.config_id}")
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"LLMConnector: Error initializing: {e}")
            self._is_connected = False
            return False

    def is_connected(self) -> bool:
        """Check if LLM is currently connected."""
        return self._is_connected and self._last_base_url is not None and self._session is not None

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
        if not self.is_connected():
            return "I'm currently unable to generate a response as my language model connection is unavailable. Please try again later or contact support."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "cm-llm",
        }

        url = f"{self._last_base_url}/v1/chat/completions"

        try:
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLMConnector: Request failed: {response.status} - {error_text}")
                    return "I encountered an error while processing your request. My language model service is experiencing issues."

                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLMConnector: Error generating text: {e}")
            self._is_connected = False  # Mark as disconnected on error
            return "I'm currently unable to generate a response due to a technical issue with my language model service. Please try again later."

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
        if not self.is_connected():
            yield "I'm currently unable to generate a response as my language model connection is unavailable. Please try again later or contact support."
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "cm-llm",
            "stream": True,  # Enable streaming
        }

        url = f"{self._last_base_url}/v1/chat/completions"

        try:
            timeout = aiohttp.ClientTimeout(total=120, connect=30)  # Longer timeout for streaming
            async with self._session.post(url, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"LLMConnector: Streaming request failed: {response.status} - {error_text}"
                    )
                    yield "I encountered an error while processing your request. My language model service is experiencing issues."
                    return

                # Parse Server-Sent Events (SSE) stream
                buffer = ""
                async for chunk in response.content.iter_chunked(1024):
                    try:
                        # Decode chunk and add to buffer
                        chunk_str = chunk.decode("utf-8")
                        buffer += chunk_str

                        # Process complete lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            # Skip empty lines and non-data lines
                            if not line or not line.startswith("data: "):
                                continue

                            # Extract data payload
                            data_str = line[6:]  # Remove 'data: ' prefix

                            # Check for stream termination
                            if data_str == "[DONE]":
                                return

                            # Parse JSON chunk
                            try:
                                chunk_data = json.loads(data_str)

                                # Extract content from the delta
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content

                            except json.JSONDecodeError:
                                # Skip malformed JSON chunks
                                logger.debug(
                                    f"LLMConnector: Skipping malformed JSON chunk: {data_str}"
                                )
                                continue

                    except UnicodeDecodeError:
                        # Skip chunks that can't be decoded
                        logger.debug("LLMConnector: Skipping chunk with decode error")
                        continue

        except asyncio.TimeoutError:
            logger.error("LLMConnector: Streaming request timed out")
            yield "\n\nI encountered a timeout while generating the response. Please try again with a shorter query."

        except Exception as e:
            logger.error(f"LLMConnector: Error during streaming generation: {e}")
            self._is_connected = False  # Mark as disconnected on error
            yield "I'm currently unable to generate a response due to a technical issue with my language model service. Please try again later."

    async def close(self):
        """Close the connector and release resources"""
        if self._session:
            logger.info("LLMConnector: Closing session")
            await self._session.close()
            self._session = None
            self._is_connected = False
            logger.info("LLMConnector: Session closed")

        # Cancel background polling task
        if self._background_fetch_task and not self._background_fetch_task.done():
            logger.info("LLMConnector: Cancelling background polling task")
            self._background_fetch_task.cancel()
            try:
                await self._background_fetch_task
            except asyncio.CancelledError:
                pass
            self._background_fetch_task = None
