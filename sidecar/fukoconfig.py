"""Unified per-repo configuration, loaded from ``.fuko.toml``.

This is the single surface an engineer edits to choose a review backend, the
underlying model/provider, where the knowledge base lives, and the embedding
provider. Secrets never live here -- each provider preset declares the env var
that holds its key. Distinct from :mod:`sidecar.config`, which holds runtime
(sidecar/server) settings read from the ``FUKO_`` environment.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = ".fuko.toml"


class ModelConfig(BaseModel):
    """The model a review backend should talk to.

    ``max_context`` is the model's context window in tokens, used for context-fit
    routing (a provider whose window can't hold the job is demoted to last
    resort). ``max_model_tokens`` overrides PR-Agent's per-review token budget cap
    (``CONFIG__MAX_MODEL_TOKENS``, which PR-Agent otherwise defaults to 32000 and
    applies as a hard ``min()`` over the model's window) — leave it unset to take
    the provider preset's default.
    """

    provider: str = "ollama"
    name: str = "qwen2.5-coder"
    base_url: str | None = None
    max_context: int | None = None
    max_model_tokens: int | None = None

    @field_validator("max_context", "max_model_tokens")
    @classmethod
    def _positive_token_count(cls, value: int | None) -> int | None:
        """A token count, when set, must be positive."""
        if value is not None and value <= 0:
            raise ValueError("token counts must be > 0 when set")
        return value


class ReviewConfig(BaseModel):
    """Which backend to run, with which model(s), tools, and runtime image.

    For throttle resilience, ``providers`` may list an ordered pool of models;
    config order is priority and the first eligible provider is pinned for the
    whole job, failing over to the next only on a throttle (see ``strategy``).
    When ``providers`` is empty the single ``model`` is used as a one-entry pool,
    so the legacy config keeps working.
    """

    backend: str = "pr-agent"
    model: ModelConfig = Field(default_factory=ModelConfig)
    providers: list[ModelConfig] = Field(
        default_factory=list,
        description=("Ordered provider pool (priority = order). Empty means use `model`."),
    )
    strategy: str = "failover"
    cooldown_seconds: int = 300
    tools: list[str] = Field(default_factory=lambda: ["review", "improve"])

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        """Reject an unimplemented pool strategy at config-parse time."""
        allowed = {"failover"}
        if value not in allowed:
            raise ValueError(
                f"unknown review strategy {value!r}; supported: {', '.join(sorted(allowed))}"
            )
        return value

    @field_validator("cooldown_seconds")
    @classmethod
    def _positive_cooldown(cls, value: int) -> int:
        """Require a positive circuit-breaker cooldown window."""
        if value <= 0:
            raise ValueError("cooldown_seconds must be > 0")
        return value

    image: str | None = None
    docker_extra_args: list[str] = Field(default_factory=list)
    tool_timeout: int = 900
    optional_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tools whose failure (incl. timeout) is a warning, not a fuko-review "
            "failure -- e.g. ['improve'] so a stalled code-suggestions pass doesn't "
            "red an observe-only review check once 'review' has posted."
        ),
    )


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
