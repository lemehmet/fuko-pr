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
    # Longest single input sent to the embedder, in characters. Oversized input
    # is not a slow request, it is a failed review: embo serves bge-m3 with a
    # 4096-token batch and 500s anything past it, and bge-m3 itself stops at
    # 8192 tokens, so no server-side setting makes a long enough text work.
    # 8000 characters keeps ~2 chars/token of headroom under that batch, which
    # even symbol-dense diffs stay inside, while leaving file digests (6000
    # chars) and markdown chunks (1500) untouched.
    embed_max_chars: int = 8000

    host: str = "0.0.0.0"
    port: int = 8000
    auth_token: str | None = None
    top_k: int = 6
    candidate_k: int = 50
    ingest_max_new: int = 10

    # File digests (#158) ship dark. Population is already opt-in -- nothing
    # writes a digest until someone runs `fuko digest` -- but the knowledge base
    # is shared across repositories and seats, so population alone is the wrong
    # off-switch: one ingest would change what every gating seat reads. Gating
    # retrieval instead means a populated store still reaches nobody until a
    # deployment turns it on, which is what #159's trial seat is for.
    digest_retrieval: bool = False


settings = Settings()
