from typing import Dict, List, Optional

import requests


class LLMConnector:
    """Connector for LLM services"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        base_urls: Optional[List[str]] = None,
        headers_list: Optional[List[Dict[str, str]]] = None,
    ):
        # Single LLM configuration
        self.base_url = base_url
        self.headers = headers or {}

        # Multiple LLM configuration
        self.base_urls = base_urls or ([] if base_url is None else [base_url])
        self.headers_list = headers_list or ([] if headers is None else [headers])

        # Ensure headers_list has the same length as base_urls
        while len(self.headers_list) < len(self.base_urls):
            self.headers_list.append({})

        # Current active LLM index
        self.active_llm_index = 0

        # If single URL is provided but no base_urls, add to base_urls
        if base_url and not base_urls:
            self.base_urls = [base_url]
            self.headers_list = [headers or {}]

    @property
    def active_base_url(self) -> str:
        """Get the active base URL"""
        if self.base_urls:
            return self.base_urls[self.active_llm_index]
        return self.base_url

    @property
    def active_headers(self) -> Dict[str, str]:
        """Get the active headers"""
        if self.headers_list:
            return self.headers_list[self.active_llm_index]
        return self.headers

    def set_active_llm(self, index: int) -> bool:
        """Set the active LLM by index"""
        if 0 <= index < len(self.base_urls):
            self.active_llm_index = index
            return True
        return False

    def get_available_llms(self) -> List[str]:
        """Get a list of available LLM URLs"""
        return self.base_urls

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM"""
        url = f"{self.active_base_url}/v1/chat/completions"

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

        response = requests.post(url, json=payload, headers=self.active_headers)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def generate_with_stream(
        self, prompt: str, callback, system_prompt: Optional[str] = None
    ) -> str:
        """Generate a streaming response from the LLM"""
        url = f"{self.active_base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": True,
            "model": "cm-llm",
        }

        full_response = ""
        with requests.post(url, json=payload, headers=self.active_headers, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    # Process SSE format
                    line_text = line.decode("utf-8")
                    if line_text.startswith("data:"):
                        data_str = line_text[5:].strip()
                        if data_str == "[DONE]":
                            break

                        import json

                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_response += delta
                                if callback:
                                    callback(delta)
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            print(f"Error parsing SSE data: {e}")

        return full_response
