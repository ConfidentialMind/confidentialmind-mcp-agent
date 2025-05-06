from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    """Configuration settings for connecting to the PostgreSQL database."""

    # Load settings from environment variables (e.g., PG_HOST, PG_PORT)
    # or a .env file
    model_config = SettingsConfigDict(env_prefix="PG_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")
    database: str = Field(default="test_db")

    # Optional: Alternatively, provide a full DSN
    dsn: str | None = Field(default=None)

    @property
    def effective_dsn(self) -> str:
        """Return the DSN string, constructing it if not provided explicitly."""
        if self.dsn:
            return self.dsn
        # Construct DSN from individual components
        # Ensure proper escaping if needed, though asyncpg handles basic cases
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# Create a single settings instance for the application
settings = PostgresSettings()
