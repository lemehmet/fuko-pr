"""Unified per-repo configuration, loaded from ``.fuko.toml``.

This is the single surface an engineer edits to choose a review backend, the
underlying model/provider, where the knowledge base lives, and the embedding
provider. Secrets never live here -- each provider preset declares the env var
that holds its key. Distinct from :mod:`sidecar.config`, which holds runtime
(sidecar/server) settings read from the ``FUKO_`` environment.
"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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
    auth: Literal["auto", "api-key", "subscription"] = Field(
        default="auto",
        description=(
            "How the backend authenticates to the model provider. 'api-key' "
            "uses the preset's key env var (ANTHROPIC_KEY for the anthropic "
            "preset); 'subscription' uses the runner's own logged-in session "
            "(Claude Code OAuth) and passes NO key; 'auto' picks api-key when "
            "that same preset key env var is set, else subscription. Read by "
            "the agentic backend only -- pr-agent always authenticates by key. "
            "Set it explicitly to pin who pays: 'auto' resolves to api-key on "
            "any runner that happens to export ANTHROPIC_KEY, so a "
            "subscription runner that also holds a key bills per token unless "
            "it says auth = 'subscription'. Note the resolution input is "
            "ANTHROPIC_KEY (fuko's own config var), NOT the ANTHROPIC_API_KEY "
            "that Claude Code itself reads -- the backend strips every ambient "
            "Anthropic credential from the agent and injects only the one this "
            "mode selects, so the mode decides billing, not the environment."
        ),
    )
    extra_instructions: str | None = Field(
        default=None,
        description=(
            "Per-entry review steering prepended to the shared knowledge blob "
            "in the backend's extra-instructions channel — e.g. a reviewing "
            "focus that differentiates this entry from the rest of the fleet. "
            "Entry-keyed: it travels with the model, so a promoted backup "
            "applies its OWN instructions (if any), not the rescued branch's."
        ),
    )

    @field_validator("max_context", "max_model_tokens")
    @classmethod
    def _positive_token_count(cls, value: int | None) -> int | None:
        """A token count, when set, must be positive."""
        if value is not None and value <= 0:
            raise ValueError("token counts must be > 0 when set")
        return value


class CompareModel(ModelConfig):
    """One branch of an A/B comparison: a model plus an optional dedicated identity.

    ``token_env`` names the GitHub token this branch posts and edits comments
    under. When *every* compare entry sets a ``token_env`` whose env var resolves
    to a distinct token, the branches run concurrently, each under its own bot
    identity so comments are separable by author (marker injection additionally
    filters to the branch's own comments -- a repo-write token could technically
    edit a sibling's). If any branch lacks ``token_env``, its env var is unset, or two
    branches resolve to the same identity, the whole run falls back to the
    sequential single-token path under the shared ``GITHUB_TOKEN``.
    """

    token_env: str | None = None


class ReviewModel(CompareModel):
    """One entry of the unified ``[[review.models]]`` list.

    ``role`` decides when the model runs and whether consumers gate on it.

    - ``"active"`` reviews the PR on every run and is a **gating** reviewer:
      one active is a plain solo review, two or more actives each review as
      their own A/B branch. Downstream tooling (e.g. the review-loop skill)
      waits on and gates merge against active instances.
    - ``"trial"`` runs on every PR **exactly like an active branch** (its own
      A/B branch, its own header, its findings marked and surfaced) but is
      **non-gating**: consumers evaluate its output without blocking the loop
      on it. It is the on-ramp for vetting a new candidate model alongside the
      actives before promoting it. Like an active it wants a ``token_env`` for
      a distinct identity / concurrent run.
    - ``"backup"`` never starts a review of its own -- it is a shared failover
      target every active/trial branch may fall back to when its own provider
      throttles or is cooling. Backups need no ``token_env``: a promoted
      backup posts under the identity of the branch it rescued, and the
      visible model label keeps the output attributable.
    """

    role: Literal["active", "backup", "trial"] = "active"
    backend: str | None = Field(
        default=None,
        description=(
            "Review driver (harness) for THIS entry -- e.g. 'pr-agent' or 'agentic'. "
            "None inherits [review].backend, so a pr-agent-only config is unchanged. "
            "Lets one fleet mix harnesses (harness diversity is the correlation fix "
            "the 2026-07-31 audit motivated); an unknown name fails at config load, "
            "not mid-run."
        ),
    )
    promoted: bool = Field(
        default=False,
        description=(
            "RUNTIME-set, not user config: marks an entry the escalation path copied "
            "from 'backup' to 'active' for one round. Escalation used to forge "
            "role='active' with nothing recording that it had done so, leaving a "
            "receipt with a null slot and an 'active' role that no configured active "
            "matched. Carrying the fact forward keeps a promoted branch attributable "
            "as what it is."
        ),
    )


class ReviewConfig(BaseModel):
    """Which backend to run, with which model(s), tools, and runtime image.

    ``models`` is the canonical surface: one unified ``[[review.models]]`` list
    where each entry carries a ``role`` (see :class:`ReviewModel`). All active
    entries review every PR -- one active is a solo review, several are an A/B
    comparison with one branch per active -- and backup entries form a shared
    failover pool each branch falls back to on throttling. Active branches run
    concurrently when every active has a distinct ``token_env`` identity (see
    :class:`CompareModel`), else sequentially under the shared token. The
    ``describe`` tool is suppressed whenever more than one active runs, because
    a PR has a single description the branches would otherwise overwrite.

    ``model``, ``providers``, and ``compare`` are the deprecated pre-unification
    sections; when ``models`` is empty they are mapped onto it by
    :func:`sidecar.pool.resolve_models` (``compare`` = all-active, ``providers``
    = first active plus backups, ``model`` = one active), so existing configs
    keep working unchanged.
    """

    backend: str = "pr-agent"
    models: list[ReviewModel] = Field(
        default_factory=list,
        description=(
            "Unified model list: every active entry reviews each PR (2+ actives "
            "= A/B), backups are shared failover targets. Non-empty supersedes "
            "the deprecated model/providers/compare sections."
        ),
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    providers: list[ModelConfig] = Field(
        default_factory=list,
        description=("Deprecated: ordered provider pool. Superseded by `models` roles."),
    )
    compare: list[CompareModel] = Field(
        default_factory=list,
        description="Deprecated: models to A/B on one PR. Superseded by `models` roles.",
    )

    @field_validator("models")
    @classmethod
    def _needs_an_active(cls, value: list[ReviewModel]) -> list[ReviewModel]:
        """A non-empty unified list must contain a model that actually runs."""
        if value and not any(m.role == "active" for m in value):
            raise ValueError("[[review.models]] needs at least one entry with role = 'active'")
        return value

    @model_validator(mode="after")
    def _known_backends(self) -> "ReviewConfig":
        """Reject a backend name no driver is registered for, at config-parse time.

        Runs ``mode="after"`` because it must see the whole-config ``backend`` and
        every per-entry ``models[*].backend`` together. The registry import is LAZY
        (``sidecar.backends`` imports this module, so the reverse edge cannot be a
        top-level import) -- deferring it to validation time, after both modules are
        loaded, breaks the cycle. Failing here surfaces an unknown driver at
        ``load_config`` as a ``ValidationError`` rather than a mid-run crash.
        """
        from .backends import known_backends

        known = known_backends()
        # Filter on ``is not None``, not truthiness: an explicit ``backend = ""``
        # is a mistake, not an inherit request, so it must reach ``named`` and be
        # rejected here rather than silently falling back to ``self.backend`` in
        # ``_backend_for`` (which treats "" as "unset" via ``or``).
        named = {self.backend, *(m.backend for m in self.models if m.backend is not None)}
        unknown = sorted(n for n in named if n not in known)
        if unknown:
            raise ValueError(
                f"unknown review backend(s) {unknown}; registered: {', '.join(sorted(known))}"
            )
        return self

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
    job_budget_minutes: int | None = Field(
        default=None,
        description=(
            "Wall-clock the CI job allows this review, i.e. the workflow's "
            "`timeout-minutes`. Escalation promotes backups into extra branches, and "
            "the sequential worst case is branches * len(tools) * tool_timeout -- so "
            "adding one backup can push a round past a cap that was computed without "
            "it, and an overrun kills the run MID-REVIEW, which is worse than a slow "
            "round because a starved round is indistinguishable from a clean one. "
            "When set, the runner refuses a promotion the budget cannot hold and "
            "says so. Left unset, promotions are unrestricted and the computed cost "
            "is logged anyway, so the arithmetic is printed rather than remembered."
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
