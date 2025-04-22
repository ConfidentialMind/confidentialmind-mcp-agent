import logging
from typing import List, Optional

from confidentialmind_core.config_manager import get_api_parameters
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class CMLLMConnector:
    """
    A lightweight LLM connector that uses the confidentialmind SDK to manage connections.
    Supports both regular and array connectors for dynamic LLM configuration.
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
        self._clients = {}  # For array connectors with multiple URLs
        self._last_base_url = None
        self._last_base_urls = []
        self._api_key = "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        self._active_model_index = 0

    def get_client(self) -> AsyncOpenAI:
        """
        Get an OpenAI-compatible client, creating a new one only if connection details have changed.
        Supports both regular and array connector configurations.

        Returns:
            AsyncOpenAI client for the configured LLM

        Raises:
            ValueError: If the connector is not properly configured
        """
        # Get current connection details from SDK
        current_base_url_or_urls, headers = get_api_parameters(self.config_id)

        # Check if connector is configured
        if not current_base_url_or_urls:
            logger.error(f"Connector for {self.config_id} is not configured - missing URL")
            raise ValueError(
                f"The connector for '{self.config_id}' has not been configured. "
                f"Please set up the connector in the portal before using this feature."
            )

        # Handle array connectors (multiple URLs)
        if isinstance(current_base_url_or_urls, list):
            logger.info(
                f"Using array connector with {len(current_base_url_or_urls)} URLs for {self.config_id}"
            )

            # Create clients for each URL if they don't exist or URLs have changed
            if self._last_base_urls != current_base_url_or_urls:
                self._last_base_urls = current_base_url_or_urls
                self._clients = {}

                # Extract API key from headers if present
                if headers and "Authorization" in headers:
                    auth_parts = headers["Authorization"].split(" ")
                    self._api_key = auth_parts[1] if len(auth_parts) > 1 else auth_parts[0]

                # Create a client for each URL
                for i, url in enumerate(current_base_url_or_urls):
                    self._clients[i] = AsyncOpenAI(
                        api_key=self._api_key,
                        base_url=url,
                    )
                    logger.info(
                        f"Created new OpenAI client {i} for {self.config_id} with URL: {url}"
                    )

            # Use the active client (could be enhanced for load balancing/failover)
            if 0 <= self._active_model_index < len(self._clients):
                return self._clients[self._active_model_index]
            elif self._clients:
                # Reset to first client if index is out of range
                self._active_model_index = 0
                return self._clients[0]
            else:
                raise ValueError(f"No clients available for array connector {self.config_id}")

        else:
            # Single URL connector (traditional approach)
            # Extract API key from headers if present
            if headers and "Authorization" in headers:
                # Usually in format "Bearer sk-123..."
                auth_parts = headers["Authorization"].split(" ")
                self._api_key = auth_parts[1] if len(auth_parts) > 1 else auth_parts[0]

            # Create a new client only if the URL has changed, or if no client exists
            if self._client is None or current_base_url_or_urls != self._last_base_url:
                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=current_base_url_or_urls,
                )
                self._last_base_url = current_base_url_or_urls
                logger.info(f"Created new OpenAI client for {self.config_id}")

            return self._client

    def set_active_model(self, index: int) -> bool:
        """
        Set the active model by index for array connectors.

        Args:
            index: The index of the model to use

        Returns:
            bool: True if successful, False if index is out of range
        """
        if isinstance(self._last_base_urls, list) and 0 <= index < len(self._last_base_urls):
            self._active_model_index = index
            logger.info(f"Switched to LLM model at index {index} for {self.config_id}")
            return True
        return False

    def get_available_models(self) -> List[str]:
        """
        Get a list of available LLM URLs for array connectors.

        Returns:
            List of available model URLs
        """
        if isinstance(self._last_base_urls, list):
            return self._last_base_urls
        elif self._last_base_url:
            return [self._last_base_url]
        return []

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
        current_base_url_or_urls, headers = get_api_parameters(self.config_id)

        if not current_base_url_or_urls:
            raise ValueError(f"Connector {self.config_id} not configured")

        # Handle array connectors by using the current active model index
        if isinstance(current_base_url_or_urls, list):
            if not current_base_url_or_urls:
                raise ValueError(f"No URLs available for array connector {self.config_id}")

            if self._active_model_index >= len(current_base_url_or_urls):
                self._active_model_index = 0

            current_base_url = current_base_url_or_urls[self._active_model_index]
        else:
            current_base_url = current_base_url_or_urls

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
        Supports both regular and array connectors.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the completion API

        Returns:
            Completion response from the LLM
        """
        client = self.get_client()
        return await client.chat.completions.create(messages=messages, **kwargs)
