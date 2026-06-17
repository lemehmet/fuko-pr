"""Configuration loaded from environment variables (prefix FUKO_) and .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the fuko-pr sidecar."""

    model_config = SettingsConfigDict(env_prefix="FUKO_", env_file=".env", extra="ignore")

    database_url: str = ""

    embed_base_url: str = "http://localhost:11434/v1"
    embed_model: str = "bge-m3"
    embed_api_key: str | None = None
    embed_dim: int = 1024
    embed_batch_size: int = 32

    host: str = "0.0.0.0"
    port: int = 8000
    auth_token: str | None = None
    top_k: int = 6
    candidate_k: int = 50


settings = Settings()
