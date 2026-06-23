"""Pydantic request/response models for the sidecar API."""

from pydantic import BaseModel, Field


class IngestItem(BaseModel):
    """A single learning to store."""

    text: str
    source: str
    source_url: str | None = None
    file_globs: list[str] = Field(default_factory=list)
    topic: str | None = None
    origin_user: str | None = None
    expires_at: str | None = None


class IngestRequest(BaseModel):
    """Body of ``POST /ingest``."""

    repo: str
    items: list[IngestItem]


class QueryRequest(BaseModel):
    """Body of ``POST /query``."""

    repo: str
    files: list[str] = Field(default_factory=list)
    pr_body: str | None = None
    query_text: str | None = None
    top_k: int | None = None


class LearningResult(BaseModel):
    """One retrieved learning with its similarity score."""

    id: str
    text: str
    source: str
    source_url: str | None
    file_globs: list[str]
    topic: str | None
    score: float


class QueryResponse(BaseModel):
    """Body returned by ``POST /query``."""

    results: list[LearningResult]


class StoredLearning(BaseModel):
    """One stored learning as listed by ``GET /learnings`` (no similarity score)."""

    id: str
    repo: str
    text: str
    source: str
    source_url: str | None
    file_globs: list[str]
    topic: str | None
    created_at: str | None = None


class ListLearningsResponse(BaseModel):
    """Body returned by ``GET /learnings``.

    ``count`` is the total number of learnings matching the filters (for paging),
    independent of ``limit``/``offset``; ``learnings`` is the requested page.
    """

    learnings: list[StoredLearning]
    count: int


class IngestResponse(BaseModel):
    """Body returned by ``POST /ingest``."""

    inserted: int
    skipped: int


class ForgetRequest(BaseModel):
    """Body of ``POST /forget``."""

    repo: str
    id: str | None = None
    source: str | None = None
    all: bool = False


class CommentRequest(BaseModel):
    """Body of ``POST /comment``: a raw PR comment to interpret."""

    repo: str
    body: str
    source_url: str | None = None
    origin_user: str | None = None


class IngestThreadsRequest(BaseModel):
    """Body of ``POST /ingest-threads``: resolved review threads to mine."""

    repo: str
    threads: list[dict]
    bot_login: str | None = None


class CooldownsResponse(BaseModel):
    """Body returned by ``GET /cb/cooldowns``: provider -> ISO cooldown end."""

    cooldowns: dict[str, str] = Field(default_factory=dict)


class TripRequest(BaseModel):
    """Body of ``POST /cb/trip``: open a provider's circuit breaker."""

    provider: str
    cooldown_seconds: int = 300
    reason: str | None = None


class TripResponse(BaseModel):
    """Body returned by ``POST /cb/trip``."""

    provider: str
    cooldown_until: str | None = None
