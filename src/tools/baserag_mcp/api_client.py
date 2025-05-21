import logging
from typing import Any, Dict, List, Optional

import aiohttp

from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API-related errors."""

    pass


class APIConnectionError(APIError):
    """Exception raised when API connection is not available."""

    pass


class BaseRAGClient:
    """Client for interacting with BaseRAG API."""

    @staticmethod
    async def _ensure_connected():
        """Ensure API connection is available."""
        if not ConnectionManager.is_connected():
            session = await ConnectionManager.create_session()
            if not session:
                raise APIConnectionError("BaseRAG API connection is not available yet")
        return ConnectionManager.get_session()

    @classmethod
    async def _make_request(
        cls,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a request to the BaseRAG API."""
        session = await cls._ensure_connected()
        base_url = ConnectionManager.get_base_url()

        if not base_url:
            raise APIConnectionError("BaseRAG API base URL is not available")

        base_url = base_url.rstrip("/")
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        url = f"{base_url}{endpoint}"

        try:
            logger.info(f"Making {method} request to {url}")
            async with session.request(method, url, json=json_data, params=params) as response:
                # Log the response status for debugging
                logger.info(f"API response status: {response.status}")

                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    raise APIError(f"BaseRAG API request failed: {response.status} - {error_text}")

                if response.status == 204:  # No content
                    return None

                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {e}")
            raise APIConnectionError(f"Failed to connect to BaseRAG API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during API request: {e}")
            raise APIError(f"Error during BaseRAG API request: {e}")

    @classmethod
    async def get_context(cls, query: str, **kwargs) -> Dict[str, Any]:
        """Get context for a query."""
        logger.info(f"Getting context for query: {query[:50]}...")
        data = {"query": query, **kwargs}
        return await cls._make_request("POST", "/context", json_data=data)

    @classmethod
    async def get_content(cls, content_id: str) -> Dict[str, Any]:
        """Get content by ID."""
        logger.info(f"Getting content with ID: {content_id}")
        return await cls._make_request("GET", f"/content/{content_id}")

    @classmethod
    async def get_content_chunks(cls, content_id: str) -> Dict[str, Any]:
        """Get content chunks by content ID."""
        logger.info(f"Getting content chunks for ID: {content_id}")
        return await cls._make_request("GET", f"/content/{content_id}/chunks")

    @classmethod
    async def chat_completion(cls, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Create a chat completion."""
        data = {"messages": messages, **kwargs}
        # Log a preview of the request for debugging (but not the full content for privacy)
        msg_preview = []
        for msg in messages:
            content = msg.get("content", "")
            preview = content[:50] + "..." if len(content) > 50 else content
            msg_preview.append({"role": msg.get("role"), "content": preview})

        logger.info(f"Creating chat completion with messages: {msg_preview}")
        return await cls._make_request("POST", "/v1/chat/completions", json_data=data)

    @classmethod
    async def check_health(cls) -> bool:
        """Check API health."""
        try:
            logger.info("Checking BaseRAG API health")
            result = await cls._make_request("GET", "/health")
            is_healthy = result.get("status") == "ok"
            logger.info(
                f"BaseRAG API health check result: {'healthy' if is_healthy else 'unhealthy'}"
            )
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
