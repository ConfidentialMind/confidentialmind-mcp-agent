"""Settings for the PostgreSQL MCP server."""

from pydantic_settings import BaseSettings


class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings from environment variables."""
    
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "postgres"
    connection_timeout: int = 10
    max_connections: int = 10
    debug: bool = False
    statement_timeout: int = 10000  # 10 seconds in milliseconds
    rate_limit: float = 0.5  # Minimum seconds between queries
    max_rows: int = 100  # Maximum rows to return in a result set
    
    class Config:
        """Pydantic config with environment variable prefix."""
        
        env_prefix = "PG_"