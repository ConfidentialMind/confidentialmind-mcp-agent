import logging
from typing import Optional

from confidentialmind_core.config_manager import get_api_parameters
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class CMLLMConnector:
    """
    A lightweight LLM connector that uses the confidentialmind SDK to manage connections.
    Based on patterns from baserag's ClientManager.
    """

    def __init__(self, config_id: str = "LLM"):
        """
        Initialize the LLM connector with an SDK config ID.

        Args:
            config_id: The connector configuration ID in the confidentialmind system
        """
        self.config_id = config_id
        self._client = None
        self._last_base_url = None
        self._api_key = "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    def get_client(self) -> AsyncOpenAI:
        """
        Get an OpenAI-compatible client, creating a new one only if connection details have changed.

        Returns:
            AsyncOpenAI client for the configured LLM

        Raises:
            ValueError: If the connector is not properly configured
        """
        # Get current connection details from SDK
        current_base_url, headers = get_api_parameters(self.config_id)

        # Check if connector is configured
        if not current_base_url:
            logger.error(f"Connector for {self.config_id} is not configured - missing URL")
            raise ValueError(
                f"The connector for '{self.config_id}' has not been configured. "
                f"Please set up the connector in the portal before using this feature."
            )

        # Extract API key from headers if present
        if headers and "Authorization" in headers:
            # Usually in format "Bearer sk-123..."
            auth_parts = headers["Authorization"].split(" ")
            self._api_key = auth_parts[1] if len(auth_parts) > 1 else auth_parts[0]

        # Create a new client only if the URL has changed, or if no client exists
        if self._client is None or current_base_url != self._last_base_url:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=current_base_url,
            )
            self._last_base_url = current_base_url
            logger.info(f"Created new OpenAI client for {self.config_id}")

        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Legacy synchronous interface for text generation.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        import requests

        # Get current connection details from SDK
        current_base_url, headers = get_api_parameters(self.config_id)

        if not current_base_url:
            raise ValueError(f"Connector {self.config_id} not configured")

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

        url = f"{current_base_url}/v1/chat/completions"
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def generate_completion(self, messages, **kwargs):
        """
        Generate completions using the SDK-managed LLM client.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the completion API

        Returns:
            Completion response from the LLM
        """
        client = self.get_client()
        return await client.chat.completions.create(messages=messages, **kwargs)
