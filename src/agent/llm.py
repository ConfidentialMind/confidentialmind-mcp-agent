import logging
from typing import Optional

import aiohttp
from confidentialmind_core.config_manager import get_api_parameters

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
                logger.error(f"Connector for {self.config_id} is not configured - missing URL")
                return False

            # Use the URL and headers
            self._last_base_url = current_base_url
            self._last_headers = headers or {}

            # Create aiohttp session
            if self._session is None:
                self._session = aiohttp.ClientSession(headers=self._last_headers)

            return True
        except Exception as e:
            logger.error(f"Error initializing LLM connector: {e}")
            return False

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from the LLM based on the prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        if not self._last_base_url:
            raise ValueError(f"Connector {self.config_id} not configured")

        if self._session is None:
            await self.initialize()
            if self._session is None:
                raise ValueError("Failed to initialize LLM connector")

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

        async with self._session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"LLM request failed: {response.status} - {error_text}")

            result = await response.json()
            return result["choices"][0]["message"]["content"]

    async def close(self):
        """Close the connector and release resources"""
        if self._session:
            await self._session.close()
            self._session = None
