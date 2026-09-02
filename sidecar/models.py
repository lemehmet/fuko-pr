"""Pydantic request/response models and the shared knowledge-store vocabulary.

Beyond the API bodies, this module holds the three things the store contract and
the API layer both need and neither owns: the :data:`SOURCES` vocabulary, the
:data:`UNSET` sentinel that lets a partial update distinguish "clear this field"
from "leave it alone", and :class:`DuplicateLearningError`. It imports nothing
from the rest of the package, so every layer can depend on it.
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field, StrictBool, StrictInt

SOURCES: tuple[str, ...] = ("remember", "review_thread", "docs", "digest")
"""Where a learning came from.

The same four values are pinned by the ``learnings_source_check`` CHECK
constraint (``migrations/011_digest_source.sql``); changing one means changing
both.

Where the two backends are actually checked differs, and it is worth being exact
about: :func:`check_source` guards the *edit* paths (``update_learning`` and the
``fuko kb edit`` / console forms that reach it) on both backends, while on the
*ingest* path only Postgres validates, via that CHECK constraint -- sqlite-vec
has neither the constraint nor a call to :func:`check_source` there. So this
tuple is the vocabulary, not an enforcement point that every write passes
through.
"""


class Unset:
    """Type of the :data:`UNSET` sentinel."""

    def __repr__(self) -> str:
        """Render as ``UNSET`` so a signature default reads clearly."""
        return "UNSET"


UNSET: Any = Unset()
"""Marks a partial-update argument the caller did not supply.

``None`` cannot serve this role: clearing ``topic`` and leaving ``topic``
untouched are different writes, and both would otherwise arrive as ``None``.
"""


class DuplicateLearningError(ValueError):
    """Raised when a write would collide with the ``(repo, text, source)`` unique key."""


class InvalidLearningError(ValueError):
    """Raised when a write's field values are not storable."""


class UnknownSourceError(InvalidLearningError):
    """Raised when a write names a source outside :data:`SOURCES`."""


def check_source(source: str) -> str:
    """Return ``source`` if it is a known one, else raise :class:`UnknownSourceError`.

    Validating in Python keeps the two stores' behaviour identical: Postgres has
    a CHECK constraint and sqlite-vec has none, so without this the same bad
    write would fail on one backend and silently succeed on the other.
    """
    if source not in SOURCES:
        raise UnknownSourceError(f"unknown source '{source}'; known sources: {', '.join(SOURCES)}")
    return source


def check_text(text: object) -> str:
    """Return ``text`` if it is storable as a learning body, else raise.

    ``text`` is the column the embedding is derived from and it is ``NOT NULL``,
    so a null or blank update is not a "clear this field" — it is a write with
    nowhere to go. Rejecting it here stops a ``null`` reaching the embedder,
    which would fail deep inside an HTTP call rather than at the request edge.
    """
    if not isinstance(text, str) or not text.strip():
        raise InvalidLearningError("text must be a non-empty string")
    return text


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
    expires_at: str | None = None


class UpdateLearningRequest(BaseModel):
    """Body of ``PATCH /learnings/{id}``: the fields to change, and nothing else.

    Every field is optional and only the ones actually present in the request
    are written -- ``model_fields_set`` is what separates "clear this field" from
    "leave it alone", so sending ``{"topic": null}`` clears the topic while
    omitting ``topic`` preserves it.
    """

    repo: str
    text: str | None = None
    source: str | None = None
    source_url: str | None = None
    file_globs: list[str] | None = None
    topic: str | None = None
    expires_at: str | None = None

    def changes(self) -> dict:
        """Return only the supplied fields, ready to splat into ``update_learning``."""
        return {name: getattr(self, name) for name in self.model_fields_set if name != "repo"}


class RepoSummary(BaseModel):
    """One repository's knowledge-base footprint, as returned by ``GET /repos``."""

    repo: str
    count: int
    sources: dict[str, int] = Field(default_factory=dict)


class ReposResponse(BaseModel):
    """Body returned by ``GET /repos``."""

    repos: list[RepoSummary] = Field(default_factory=list)


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


class IngestThreadsResponse(BaseModel):
    """Body returned by ``POST /ingest-threads``.

    ``remaining`` is how many mined learnings were left unembedded because this
    call hit its per-request cap. A caller drains a backlog by re-sending the
    same batch until it reports zero; the already-stored ones dedup away, so each
    pass costs only the embed work it is bounded to.
    """

    considered: int
    inserted: int
    skipped: int
    remaining: int = 0


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


NonNegativeCount = Annotated[StrictInt, Field(ge=0)]
"""A count: a non-negative integer, and an integer as SPELLED.

Used for every count on :class:`TranscriptIndexRequest`, not only the mapping's
values, so one spelling cannot be strict while the field beside it is lax.

The constraint has to live on the annotation rather than on the ``Field`` when
the count is a dict *value*: ``ge`` given to the field constrains the mapping,
not what is in it.

``StrictInt`` rather than ``int`` because lax coercion turns JSON ``true`` into
``1``, which would store a shape error as a real measurement -- and would make
the two metrics transports disagree, since the direct path's own filter
(:func:`sidecar.run_metrics._tool_calls`) drops a ``bool``. Every producer in
this repo posts figures :class:`sidecar.reviewer.transcript.TranscriptIndex`
derived, which are already ``int``; strictness costs them nothing and refuses
only a caller that was never counting.
"""


class TranscriptIndexRequest(BaseModel):
    """One run's session-transcript index row, riding ``POST /metrics/run`` (#239).

    Derived by the runner AT CAPTURE, from the feed it was already streaming to
    the transcript sink (:class:`sidecar.reviewer.transcript.TranscriptIndex`),
    so nothing re-downloads a blob to count what was in it.

    Nested rather than flattened onto :class:`RunMetricRequest` because these
    figures describe the transcript and land in their own table -- ``review_runs``
    gains exactly one column, the reference. Absent (``None``) is the normal case:
    every pr-agent run, and every agentic run whose capture is off or failed.

    Every field is REQUIRED except the counts, which default to the empty
    measurement rather than to nothing-measured: this object only exists for a
    transcript that was captured, so a run that genuinely called no tools is a
    real zero -- unlike ``review_runs``' token columns, where a zero would claim
    an unmeasured run was free.
    """

    key: str = Field(
        description=(
            "The transcript's own key, minted at run start and naming its blob in "
            "the store. Not validated as a blob key here: a malformed one must cost "
            "the reference, never the metrics row it rides with."
        )
    )
    complete: StrictBool = Field(
        description="Whether the feed reached its terminal `result` event."
    )
    tool_calls: dict[str, NonNegativeCount] = Field(
        default_factory=dict,
        description="Call counts keyed by tool name.",
    )
    tool_result_bytes: NonNegativeCount = Field(
        default=0, description="Total UTF-8 bytes of tool-result content the run was fed."
    )
    repeated_read_files: NonNegativeCount = Field(
        default=0,
        description=(
            "Distinct files read more than once in this run -- one file read three "
            "times counts once."
        ),
    )


class RunMetricRequest(BaseModel):
    """Body of ``POST /metrics/run``: one review-run row from the runner.

    Every token/cost field defaults to ``None`` so a runner older than #152 --
    or one whose backend has no usage feed -- posts a valid body that records
    "not measured", which is the truth for those runs and never a zero.
    """

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
    backend: str = "pr-agent"
    endpoint: str = ""
    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Fresh (uncached) input tokens; excludes the two cache counts below.",
    )
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Input tokens served from the provider's prompt cache. Their ratio against "
            "input_tokens is what shows whether a gateway honours prompt caching at all."
        ),
    )
    cache_write_tokens: int | None = Field(
        default=None, ge=0, description="Input tokens written into the prompt cache."
    )
    cost_usd: float | None = Field(default=None, ge=0)
    turns: int | None = Field(default=None, ge=0)
    transcript: TranscriptIndexRequest | None = Field(
        default=None,
        description=(
            "This run's session transcript, if one was captured (#239). Defaults to "
            "None so a runner older than this change -- or any backend with no "
            "capture path -- posts a valid body that records no reference, which is "
            "the truth for those runs rather than a key naming a blob that does not "
            "exist."
        ),
    )


class RunSummaryRow(BaseModel):
    """One provider+model aggregate returned by ``GET /metrics/summary``.

    The token/cost totals are ``None`` -- not ``0`` -- for a model whose runs
    reported none, e.g. every pr-agent row. They are declared here because a
    ``response_model`` silently DROPS undeclared keys, so an aggregate missing
    from this class is an aggregate the endpoint cannot return.
    """

    provider: str
    model: str
    runs: int
    ok: int
    not_ok: int
    avg_duration_s: float | None = None
    findings: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    turns: int | None = None


class RunSummaryResponse(BaseModel):
    """Body returned by ``GET /metrics/summary``."""

    summary: list[RunSummaryRow] = Field(default_factory=list)


class TranscriptRunRow(BaseModel):
    """One captured transcript returned by ``GET /transcripts`` (#240).

    Mirrors :class:`sidecar.transcripts.TranscriptRun` field for field, because
    a ``response_model`` silently DROPS undeclared keys -- a figure missing from
    this class is a figure the listing cannot show, and it would look exactly
    like a figure the run did not produce.

    The transcript's own five values are required: the index row exists only for
    a transcript that reached storage and every column behind them is NOT NULL.
    Everything from the run row is optional, because that row is written in a
    separate transaction afterwards and may never have followed.
    """

    key: str = Field(description="The transcript's own key; names its blob in the store.")
    created_at: str | None = None
    complete: bool = Field(
        description=(
            "Whether the captured feed reached its terminal `result` event. False means "
            "the stored bytes are a prefix of a run that was cut short -- a short "
            "session, not a cheap one."
        )
    )
    tool_calls: dict[str, int] = Field(
        default_factory=dict, description="Call counts keyed by tool name."
    )
    tool_result_bytes: int = 0
    repeated_read_files: int = Field(
        default=0,
        description="Distinct files read more than once; one file read three times counts once.",
    )
    repo: str | None = None
    pr: int | None = None
    seat: str | None = Field(
        default=None, description="The run's slot -- the lane label a model occupied."
    )
    provider: str | None = None
    model: str | None = None
    backend: str | None = None
    outcome: str | None = None
    started_at: str | None = None
    duration_s: float | None = None


class TranscriptListResponse(BaseModel):
    """Body returned by ``GET /transcripts``: one page, plus how many matched."""

    transcripts: list[TranscriptRunRow] = Field(default_factory=list)
    count: int = Field(
        default=0,
        description=(
            "Transcripts matching the filters across every page, carried by the rows "
            "on this one -- so it is 0 for any empty page, including an offset past "
            "the end of the window."
        ),
    )
