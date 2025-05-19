import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    """Configuration settings for PostgreSQL connection with ConfidentialMind support."""

    model_config = SettingsConfigDict(env_prefix="PG_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    user: str = Field(default="app")
    password: str = Field(default="testpass")
    database: str = Field(default="vector-db")

    # Optional direct DSN
    dsn: Optional[str] = Field(default=None)

    # ConfidentialMind integration
    connector_id: str = Field(default="DATABASE")
    use_sdk_connector: bool = Field(default=False)

    @property
    def is_stack_deployment(self) -> bool:
        """Determine if running in stack deployment mode."""
        return os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG", "False").lower() != "true"

    @property
    def effective_dsn(self) -> str:
        """Return the effective DSN, with support for SDK-provided connection info."""
        if self.dsn:
            return self.dsn
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_connection_string(self, sdk_url: Optional[str] = None) -> str:
        """
        Generate connection string with optional SDK URL incorporation.

        Args:
            sdk_url: Optional URL provided by the SDK

        Returns:
            PostgreSQL connection string
        """
        # If DSN is explicitly set, use it over everything else
        if self.dsn:
            return self.dsn

        is_local_config = not self.is_stack_deployment

        if is_local_config and sdk_url:
            # Local config mode with URL - use the SDK-provided URL as the host part
            # but still use credentials from settings
            return f"postgresql://{self.user}:{self.password}@{sdk_url}/{self.database}"
        elif is_local_config:
            # Local config mode with no URL - use default settings
            host_part = f"{self.host}:{self.port}"
            return f"postgresql://{self.user}:{self.password}@{host_part}/{self.database}"
        else:
            # Stack deployment mode - use the URL as the endpoint
            # with credentials from settings
            if not sdk_url:
                raise ValueError("No database URL provided in stack deployment mode")

            # In stack deployment, use the URL as the hostname/port
            return f"postgresql://{self.user}:{self.password}@{sdk_url}/{self.database}"


# Single settings instance
settings = PostgresSettings()
