"""Unified per-repo configuration, loaded from ``.fuko.toml``.

This is the single surface an engineer edits to choose a review backend, the
underlying model/provider, where the knowledge base lives, and the embedding
provider. Secrets never live here -- each provider preset declares the env var
that holds its key. Distinct from :mod:`sidecar.config`, which holds runtime
(sidecar/server) settings read from the ``FUKO_`` environment.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = ".fuko.toml"


class ModelConfig(BaseModel):
    """The model a review backend should talk to."""

    provider: str = "ollama"
    name: str = "qwen2.5-coder"
    base_url: str | None = None


class ReviewConfig(BaseModel):
    """Which backend to run, with which model, tools, and runtime image."""

    backend: str = "pr-agent"
    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: list[str] = Field(default_factory=lambda: ["review", "improve"])
    image: str | None = None
    docker_extra_args: list[str] = Field(default_factory=list)


class PostgresStoreConfig(BaseModel):
    """Settings for the Postgres/pgvector knowledge store."""

    url_env: str = "FUKO_DATABASE_URL"


class ObjectStoreConfig(BaseModel):
    """Where a sqlite-vec knowledge file lives in object storage."""

    backend: str = "s3"
    bucket: str | None = None
    key: str | None = None
    endpoint_url: str | None = None
    creds_env_prefix: str = "FUKO_S3"


class KnowledgeConfig(BaseModel):
    """Which store backs the knowledge base, and its settings."""

    store: str = "postgres"
    postgres: PostgresStoreConfig = Field(default_factory=PostgresStoreConfig)
    object_store: ObjectStoreConfig | None = None


class EmbeddingConfig(BaseModel):
    """Embedding provider for the knowledge base (OpenAI-compatible endpoint)."""

    provider: str = "ollama"
    model: str = "bge-m3"
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str | None = None


class FukoConfig(BaseModel):
    """The full ``.fuko.toml`` document."""

    review: ReviewConfig = Field(default_factory=ReviewConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> FukoConfig:
    """Load ``.fuko.toml`` from ``path``, returning defaults if it does not exist."""
    p = Path(path)
    if not p.exists():
        return FukoConfig()
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    return FukoConfig.model_validate(data)
