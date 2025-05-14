import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    """Configuration settings for PostgreSQL connection with ConfidentialMind support."""

    model_config = SettingsConfigDict(env_prefix="PG_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")
    database: str = Field(default="test_db")

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
        """Generate connection string with optional SDK URL incorporation."""
        if self.dsn:
            return self.dsn

        if sdk_url:
            # Extract host:port from SDK URL and use with credentials
            return f"postgresql://{self.user}:{self.password}@{sdk_url}/{self.database}"

        return self.effective_dsn


# Single settings instance
settings = PostgresSettings()
