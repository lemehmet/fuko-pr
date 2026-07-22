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


class ReviewerObservation(BaseModel):
    """One reviewer's observed state, as reported by the runner after a review."""

    reviewer: str
    state: str
    detail: str | None = None


class ObserveHealthRequest(BaseModel):
    """Body of ``POST /rh/observe``: batch-record reviewer states for one PR round."""

    repo: str
    pr: int | None = None
    observations: list[ReviewerObservation] = Field(default_factory=list)


class ReviewerHealthRow(BaseModel):
    """One stored reviewer-health row, as returned by ``GET /rh/state``."""

    reviewer: str
    state: str
    observed_at: str
    pr: int | None = None
    detail: str | None = None


class ReviewerHealthResponse(BaseModel):
    """Body returned by ``GET /rh/state``: last observed state rows for a repo."""

    reviewers: list[ReviewerHealthRow] = Field(default_factory=list)


class RunMetricRequest(BaseModel):
    """Body of ``POST /metrics/run``: one review-run row from the runner."""

    repo: str
    pr: int
    provider: str
    model: str
    slot: str | None = None
    duration_s: float = Field(default=0.0, ge=0)
    attempts: int = Field(default=1, ge=1)
    outcome: str = "ok"
    findings: int | None = None
    detail: str | None = None


class RunSummaryRow(BaseModel):
    """One provider+model aggregate returned by ``GET /metrics/summary``."""

    provider: str
    model: str
    runs: int
    ok: int
    not_ok: int
    avg_duration_s: float | None = None
    findings: int


class RunSummaryResponse(BaseModel):
    """Body returned by ``GET /metrics/summary``."""

    summary: list[RunSummaryRow] = Field(default_factory=list)
