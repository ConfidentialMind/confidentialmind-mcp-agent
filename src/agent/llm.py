import asyncio
import logging
from typing import Optional

import aiohttp
from confidentialmind_core.config_manager import ConfigManager
from confidentialmind_core.config_manager import config as cm_config
from confidentialmind_core.config_manager import get_api_parameters

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


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
        self._initialized = False
        self._unavailable = False
        self._last_error: Optional[str] = None
        self._retry_task: Optional[asyncio.Task] = None

    async def initialize(self) -> bool:
        """
        Initialize the connector by fetching API parameters.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Get current connection details from SDK
            current_base_url, headers = get_api_parameters(self.config_id)

            # Check if connector is configured
            if not current_base_url:
                # In stack mode, this might be because configs haven't been set yet
                if not cm_config.LOCAL_CONFIGS:
                    config_manager = ConfigManager()
                    # Check if ConfigManager is properly initialized
                    if (
                        not hasattr(config_manager, "_ConfigManager__initialized")
                        or not config_manager._ConfigManager__initialized
                    ):
                        logger.warning(
                            "ConfigManager not initialized yet. LLM connection will be retried later."
                        )
                        self._unavailable = True
                        self._last_error = "LLM connection not configured yet"
                        self._schedule_retry()
                        return False

                    # ConfigManager is initialized but no LLM is connected
                    logger.warning(f"No LLM connector configured for {self.config_id}")
                    self._unavailable = True
                    self._last_error = "No LLM connector configured"
                    self._schedule_retry()
                    return False
                else:
                    # In local mode, this is a configuration error
                    logger.error(f"Connector for {self.config_id} is not configured - missing URL")
                    self._unavailable = True
                    self._last_error = (
                        f"LLM connector {self.config_id} not configured in environment"
                    )
                    return False

            # URL has changed or first time initializing
            if current_base_url != self._last_base_url or self._session is None:
                # Close existing session if any
                if self._session is not None:
                    await self._session.close()

                # Update connection details
                self._last_base_url = current_base_url
                self._last_headers = headers or {}

                # Create new aiohttp session
                self._session = aiohttp.ClientSession(headers=self._last_headers)
                logger.info(
                    f"Initialized LLM connector for {self.config_id} with URL: {current_base_url}"
                )

            self._initialized = True
            self._unavailable = False
            self._last_error = None
            return True

        except Exception as e:
            self._last_error = f"Error initializing LLM connector: {e}"
            logger.error(self._last_error)
            self._unavailable = True
            self._schedule_retry()
            return False

    def _schedule_retry(self):
        """Schedule a retry for initializing the LLM connector."""
        if not cm_config.LOCAL_CONFIGS and (not self._retry_task or self._retry_task.done()):
            self._retry_task = asyncio.create_task(self._retry_init())

    async def _retry_init(self):
        """Retry initialization with exponential backoff."""
        retry_count = 0
        max_retries = 5
        base_delay = 5  # seconds

        while retry_count < max_retries and (not self._initialized or self._unavailable):
            retry_count += 1
            delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff

            logger.info(
                f"Retrying LLM connector initialization ({retry_count}/{max_retries}) in {delay} seconds"
            )
            await asyncio.sleep(delay)

            try:
                success = await self.initialize()
                if success:
                    logger.info(f"Successfully initialized LLM connector on retry {retry_count}")
                    return
            except Exception as e:
                logger.error(f"Retry {retry_count} failed: {e}")

        if retry_count >= max_retries:
            logger.warning(f"Failed to initialize LLM connector after {max_retries} retries")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from the LLM based on the prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        if self._unavailable:
            no_llm_message = (
                f"The LLM connector ({self.config_id}) is not available: {self._last_error}"
            )
            logger.warning(no_llm_message)
            return f"I'm unable to process your request at this time. {no_llm_message} Please try again later or contact support."

        if not self._initialized or not self._last_base_url:
            await self.initialize()
            if self._unavailable:
                no_llm_message = (
                    f"The LLM connector ({self.config_id}) is not available: {self._last_error}"
                )
                logger.warning(no_llm_message)
                return f"I'm unable to process your request at this time. {no_llm_message} Please try again later or contact support."

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
            # Ensure session is created
            if self._session is None:
                self._session = aiohttp.ClientSession(headers=self._last_headers)

            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM request failed: {response.status} - {error_text}")
                    return f"I encountered an error processing your request. Please try again later or contact support if the issue persists."

                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Mark as unavailable and schedule retry
            self._unavailable = True
            self._last_error = f"Error generating LLM response: {e}"
            self._schedule_retry()
            return f"I encountered an error processing your request. Error: {str(e)}. Please try again later."

    async def close(self):
        """Close the connector and release resources"""
        if self._session:
            await self._session.close()
            self._session = None

        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
