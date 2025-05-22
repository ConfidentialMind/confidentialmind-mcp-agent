import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseRAGSettings(BaseSettings):
    """Configuration settings for BaseRAG API connection with ConfidentialMind support."""

    model_config = SettingsConfigDict(env_prefix="BASERAG_", env_file=".env", extra="ignore")

    # BaseRAG API connection settings
    api_url: str = Field(default="http://localhost:8080")
    api_key: Optional[str] = Field(default=None)
    api_key_header: str = Field(default="Authorization")

    # ConfidentialMind integration
    connector_id: str = Field(default="BASERAG_API")
    use_sdk_connector: bool = Field(default=False)

    @property
    def is_stack_deployment(self) -> bool:
        """Determine if running in stack deployment mode."""
        return os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"


# Single settings instance
settings = BaseRAGSettings()
