"""FastAPI app exposing ``/ingest`` ``/query`` ``/learnings`` ``/forget`` ``/healthz``."""

import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status

from . import circuit_breaker
from . import models
from . import review_state
from . import review_state_client as rs
from . import reviewer_health
from . import run_metrics
from . import web
from . import threads as threads_mod
from .config import settings
from .stores import current_store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run DB migrations before the sidecar serves traffic.

    Warming the pool at startup means migrations run-and-commit before any
    request, so a fresh database never 500s on its first ``/query`` / ``/cb``
    call. Best-effort and gated on a configured Postgres URL: if the database
    isn't reachable at boot the error is logged and startup proceeds, leaving
    ``/healthz`` available and the (lock-guarded) lazy ``get_pool()`` path to
    retry on first use.
    """
    if settings.database_url:
        from .db import get_pool

        try:
            get_pool()
        except Exception as e:
            print(f"fuko: startup migration deferred (database not ready?): {e}", file=sys.stderr)
    yield


app = FastAPI(title="fuko-pr sidecar", version="0.6.0", lifespan=lifespan)

app.include_router(web.router)

# The sidecar serves one store, selected by .fuko.toml (defaults to Postgres).
_store = current_store()


def _auth(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token dependency, fail-closed.

    Every protected endpoint requires a matching ``Bearer <FUKO_AUTH_TOKEN>``.
    When no token is configured the endpoint is refused (503) rather than served
    unauthenticated, so a misconfigured deployment cannot expose the mutating
    endpoints. The unauthenticated routes are ``/healthz`` and FastAPI's
    auto-generated ``/docs``, ``/redoc``, and ``/openapi.json``, which expose
    only the API schema -- no stored data and no mutation.
    """
    if not settings.auth_token:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "server auth not configured (set FUKO_AUTH_TOKEN)",
        )
    if authorization != f"Bearer {settings.auth_token}":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe (does not touch the database)."""
    return {"ok": True}


@app.post("/ingest", response_model=models.IngestResponse, dependencies=[Depends(_auth)])
def ingest_endpoint(req: models.IngestRequest) -> dict:
    """Store learnings for a repository."""
    inserted, skipped = _store.ingest(req.repo, req.items)
    return {"inserted": inserted, "skipped": skipped}


@app.post("/query", response_model=models.QueryResponse, dependencies=[Depends(_auth)])
def query_endpoint(req: models.QueryRequest) -> dict:
    """Retrieve the most relevant learnings for a pull request."""
    results = _store.query(req.repo, req.files, req.pr_body, req.query_text, req.top_k)
    return {"results": results}


@app.get("/learnings", response_model=models.ListLearningsResponse, dependencies=[Depends(_auth)])
def list_learnings_endpoint(
    repo: str | None = None,
    source: str | None = None,
    limit: int = 100,
    offset: int = 0,
    q: str | None = None,
    include_expired: bool = False,
) -> dict:
    """List stored learnings for browsing, newest first.

    Unlike ``/query`` (semantic + file-scoped, for review-time retrieval) this is
    a plain inspection listing, optionally filtered by ``repo``, ``source``, and a
    case-insensitive substring ``q`` over text and topic. Expired learnings are
    excluded unless ``include_expired`` is set, matching what retrieval surfaces.
    ``limit`` is clamped to 500 and ``offset`` floored at 0. ``count`` is the total
    matching the filters (for paging), not the page size.
    """
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    learnings, total = _store.list_learnings(
        repo=repo,
        source=source,
        limit=limit,
        offset=offset,
        q=q,
        include_expired=include_expired,
    )
    return {"learnings": learnings, "count": total}


@app.get("/repos", response_model=models.ReposResponse, dependencies=[Depends(_auth)])
def repos_endpoint() -> dict:
    """Return every repository holding live learnings, with per-source counts."""
    return {"repos": _store.repos()}


@app.get("/learnings/{id}", response_model=models.StoredLearning, dependencies=[Depends(_auth)])
def get_learning_endpoint(id: str, repo: str) -> dict:
    """Return one learning by id within ``repo``, expired ones included."""
    learning = _store.get_learning(repo, id)
    if learning is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such learning in that repo")
    return learning


@app.patch("/learnings/{id}", response_model=models.StoredLearning, dependencies=[Depends(_auth)])
def update_learning_endpoint(id: str, req: models.UpdateLearningRequest) -> dict:
    """Apply the supplied fields to one learning and return the updated row.

    Only the fields present in the request body are written, so clearing a field
    (sending it as null) stays distinguishable from omitting it. ``text`` is the
    exception: it is ``NOT NULL`` and is what gets embedded, so a null or blank
    one is rejected rather than treated as a clear. Changing ``text`` re-embeds;
    any other change skips the embedder.
    """
    try:
        updated = _store.update_learning(req.repo, id, **req.changes())
    except models.InvalidLearningError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    except models.DuplicateLearningError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such learning in that repo")
    return updated


@app.post("/forget", dependencies=[Depends(_auth)])
def forget_endpoint(req: models.ForgetRequest) -> dict:
    """Delete learnings by id, source, or wholesale for a repository."""
    if not (req.id or req.source or req.all):
        raise HTTPException(400, "provide id, source, or all=true")
    deleted = _store.forget(req.repo, id=req.id, source=req.source, all=req.all)
    return {"deleted": deleted}


@app.post("/comment", dependencies=[Depends(_auth)])
def comment_endpoint(req: models.CommentRequest) -> dict:
    """Interpret a raw PR comment as ``/remember`` or ``/forget`` and act on it."""
    from .commands import parse_forget, parse_remember

    remembered = parse_remember(req.body)
    if remembered is not None:
        text, globs = remembered
        inserted, skipped = _store.ingest(
            req.repo,
            [
                models.IngestItem(
                    text=text,
                    source="remember",
                    source_url=req.source_url,
                    file_globs=globs,
                    origin_user=req.origin_user,
                )
            ],
        )
        return {"action": "remember", "inserted": inserted, "skipped": skipped}

    forgotten = parse_forget(req.body)
    if forgotten is not None:
        deleted = _store.forget(
            req.repo,
            id=forgotten.get("id"),
            source=forgotten.get("source"),
            all=bool(forgotten.get("all")),
        )
        return {"action": "forget", "deleted": deleted}

    return {"action": "ignored"}


@app.post(
    "/ingest-threads",
    response_model=models.IngestThreadsResponse,
    dependencies=[Depends(_auth)],
)
def ingest_threads_endpoint(req: models.IngestThreadsRequest) -> dict:
    """Mine resolved review threads for learnings and ingest a bounded batch.

    Embedding is the slow part and its cost tracks the number of *new* learnings,
    which the caller cannot see — a batch of already-stored threads is free while
    the same-sized batch of fresh ones can outrun any client timeout. So the cap
    lives here, where the post-dedup count is known: at most
    ``FUKO_INGEST_MAX_NEW`` learnings are embedded per call and the rest are
    reported as ``remaining`` for the caller to drain by re-sending the batch.

    The cap floors at 1 because it is operator-supplied: a zero or negative value
    would defer every item forever, reporting work remaining while never making
    progress, and a draining caller would spin until its own retry bound.
    """
    items = [
        it
        for it in (threads_mod.select_learning(t, req.bot_login) for t in req.threads)
        if it is not None
    ]
    inserted, skipped = _store.ingest(req.repo, items, max_new=max(1, settings.ingest_max_new))
    return {
        "considered": len(req.threads),
        "inserted": inserted,
        "skipped": skipped,
        "remaining": len(items) - inserted - skipped,
    }


@app.get("/cb/cooldowns", response_model=models.CooldownsResponse, dependencies=[Depends(_auth)])
def cb_cooldowns_endpoint() -> dict:
    """Return the providers whose circuit breaker is currently open (cooling down)."""
    return {"cooldowns": circuit_breaker.get_cooldowns()}


@app.post("/cb/trip", response_model=models.TripResponse, dependencies=[Depends(_auth)])
def cb_trip_endpoint(req: models.TripRequest) -> dict:
    """Open a provider's circuit breaker for a cooldown window (idempotent upsert)."""
    until = circuit_breaker.trip(req.provider, req.cooldown_seconds, req.reason or "")
    return {"provider": req.provider, "cooldown_until": until}


@app.get("/rh/state", response_model=models.ReviewerHealthResponse, dependencies=[Depends(_auth)])
def rh_state_endpoint(repo: str) -> dict:
    """Return the last observed state of each external reviewer for ``repo``."""
    return {"reviewers": reviewer_health.states(repo)}


@app.post("/metrics/run", dependencies=[Depends(_auth)])
def metrics_run_endpoint(req: models.RunMetricRequest) -> dict:
    """Record one review-run row reported by the runner at branch completion."""
    run_metrics.record(
        req.repo,
        req.pr,
        req.provider,
        req.model,
        slot=req.slot,
        duration_s=req.duration_s,
        attempts=req.attempts,
        outcome=req.outcome,
        findings=req.findings,
        detail=req.detail or "",
        backend=req.backend,
        endpoint=req.endpoint,
        input_tokens=req.input_tokens,
        output_tokens=req.output_tokens,
        cache_read_tokens=req.cache_read_tokens,
        cache_write_tokens=req.cache_write_tokens,
        cost_usd=req.cost_usd,
        turns=req.turns,
    )
    return {"recorded": True, "persisted": bool(settings.database_url)}


@app.get(
    "/metrics/summary", response_model=models.RunSummaryResponse, dependencies=[Depends(_auth)]
)
def metrics_summary_endpoint(repo: str | None = None, days: int = 30) -> dict:
    """Aggregate review runs per provider+model over the last ``days``."""
    return {"summary": run_metrics.summary(repo=repo, days=days)}


@app.post("/rh/observe", dependencies=[Depends(_auth)])
def rh_observe_endpoint(req: models.ObserveHealthRequest) -> dict:
    """Batch-record the reviewer states the runner observed at the end of a round."""
    for obs in req.observations:
        reviewer_health.observe(req.repo, obs.reviewer, obs.state, req.pr, obs.detail or "")
    return {"recorded": len(req.observations), "persisted": bool(settings.database_url)}


# --- The per-seat review ledgers, for runners with no connection string (#171).
#
# One endpoint per :mod:`sidecar.review_state` primitive, each handler a direct
# call to it, so the HTTP path and the local path share one semantics rather
# than resembling each other. Why not the two composite endpoints #171 sketches,
# what stays behind on the runner, and how a dead sidecar is bounded, are all in
# :mod:`sidecar.review_state_client`.
#
# WHAT THIS BOUNDARY DOES AND DOES NOT CHECK. The store-side guards travel with
# the primitives and so hold for both transports: the ``FINDING_STATUSES``
# vocabulary, ``REOPENABLE_STATUSES`` (no request can REVERSE a ``stale`` -- but
# see :func:`rs_transition_endpoint`, because a request can now create one), the
# UUID check, ``_clip``'s text bound, the read caps
# and the retention window -- plus the ``(repo, pr, seat)`` scope now matched in
# SQL, which is what keeps one seat from settling another's row (#160).
#
# The prompt-side guards deliberately stay on the runner, because the sidecar
# cannot evaluate them: ``p1..pN`` are minted per round by ``render_prior_state``
# and mean nothing here; ``PriorState.accepted_status`` drops a verdict naming an
# id THIS ROUND was not handed, which is a fact about the prompt the server never
# sees; ``_one_line``/``_indented`` bound what a stored row may do to a later
# prompt and belong at render time for the same reason. A caller holding
# ``FUKO_TOKEN`` can therefore write a ledger row that no round would have
# produced -- but that caller is the runner, and the same token already
# authorizes ``/forget`` with ``all=true``. The transport widens what a LEAKED
# token reaches, not what a reviewed pull request can reach.
#
# "The model never holds the token" is the load-bearing half of that, and it is
# now ENFORCED rather than assumed: ``FUKO_URL``, ``FUKO_TOKEN`` and
# ``FUKO_DATABASE_URL`` are stripped from the agent's environment alongside the
# GitHub credentials (:data:`sidecar.backends.agentic._SIDECAR_CRED_VARS`).
# Before that they were inherited straight into the harness subprocess by the
# review workflow's own exports, so the acceptance above rested on the agent
# having no tool that reads its own environ -- which is a denylist, and the
# denylist's own docstring says it closes the instance, not the class.


@app.post("/rs/findings", response_model=rs.LedgerCountResponse, dependencies=[Depends(_auth)])
def rs_record_findings_endpoint(req: rs.RecordFindingsRequest) -> dict:
    """Record one round's newly-opened findings for a seat."""
    return {
        "count": review_state.record_findings(
            req.repo, req.pr, req.seat, req.round, req.head_sha, req.findings
        )
    }


@app.get("/rs/findings", response_model=rs.OpenFindingsResponse, dependencies=[Depends(_auth)])
def rs_open_findings_endpoint(repo: str, pr: int, seat: str) -> dict:
    """Return a seat's still-open findings, and how many the read cap cut."""
    ledger = review_state.open_findings(repo, pr, seat)
    return {"rows": list(ledger.rows), "truncated": ledger.truncated}


@app.get("/rs/round", response_model=rs.NextRoundResponse, dependencies=[Depends(_auth)])
def rs_next_round_endpoint(repo: str, pr: int, seat: str) -> dict:
    """Return the round number a seat's next round should record under."""
    return {"round": review_state.next_round(repo, pr, seat)}


@app.get("/rs/settled", response_model=rs.SettledFindingsResponse, dependencies=[Depends(_auth)])
def rs_settled_findings_endpoint(repo: str, pr: int, seat: str) -> dict:
    """Return a seat's model-closed findings, the projection a re-raise needs."""
    return {"rows": list(review_state.settled_findings(repo, pr, seat))}


@app.post(
    "/rs/findings/transition",
    response_model=rs.LedgerChangedResponse,
    dependencies=[Depends(_auth)],
)
def rs_transition_endpoint(req: rs.TransitionRequest) -> dict:
    """Apply one verdict to one of a seat's open findings.

    ``stale`` is accepted here, deliberately, and it is the asymmetric one: a
    ``stale`` row is reversible by nothing, since :func:`review_state.reopen`
    matches only ``REOPENABLE_STATUSES`` and :func:`review_state.transition`
    matches only ``status = 'open'``, so it is out of every later prompt until
    the retention window drops it. In-process it is minted in one place --
    ``ledger._retire_missing``, after ``_is_gone`` has proved the file absent
    from the tree -- and that filesystem fact cannot travel the wire, so this
    endpoint cannot tell fuko's own retirement from any other caller's claim.

    It is accepted anyway because rejecting it would break the feature rather
    than protect it: on the remote branch ``_retire_missing`` reaches the store
    THROUGH this endpoint, so a filter here would silently disable retirement on
    exactly the deployment #171 exists to serve. The residual risk is the one the
    boundary note above already states and accepts -- a ``FUKO_TOKEN`` holder can
    write a ledger row no round would have produced -- and it is strictly smaller
    than what the same token already authorizes: ``/forget`` with ``all=true``
    discards the whole knowledge base, where this marks findings in one seat's
    lane unreadable for the retention window. Raised by
    ``qwen-anthropic/qwen3.8-max`` on #171, which asked for either an explicit
    acceptance or an alert hook; this is the acceptance.
    """
    return {
        "changed": review_state.transition(
            req.repo, req.pr, req.seat, req.finding_id, req.status, req.reason
        )
    }


@app.post(
    "/rs/findings/reopen", response_model=rs.LedgerChangedResponse, dependencies=[Depends(_auth)]
)
def rs_reopen_endpoint(req: rs.ReopenRequest) -> dict:
    """Re-raise one of a seat's findings that an earlier verdict closed."""
    return {"changed": review_state.reopen(req.repo, req.pr, req.seat, req.finding_id, req.reason)}


@app.post(
    "/rs/findings/touch", response_model=rs.LedgerCountResponse, dependencies=[Depends(_auth)]
)
def rs_touch_findings_endpoint(req: rs.TouchRequest) -> dict:
    """Refresh ``updated_at`` on the findings a seat's round re-asserted."""
    return {"count": review_state.touch_findings(req.repo, req.pr, req.seat, req.finding_ids)}


@app.post("/rs/coverage", response_model=rs.LedgerCountResponse, dependencies=[Depends(_auth)])
def rs_record_coverage_endpoint(req: rs.RecordCoverageRequest) -> dict:
    """Record one round's examined regions for a seat."""
    return {
        "count": review_state.record_coverage(
            req.repo, req.pr, req.seat, req.round, req.head_sha, req.regions
        )
    }


@app.get("/rs/coverage", response_model=rs.LiveCoverageResponse, dependencies=[Depends(_auth)])
def rs_live_coverage_endpoint(repo: str, pr: int, seat: str) -> dict:
    """Return a seat's unexpired coverage entries."""
    return {"rows": list(review_state.live_coverage(repo, pr, seat))}


@app.post(
    "/rs/coverage/expire", response_model=rs.LedgerCountResponse, dependencies=[Depends(_auth)]
)
def rs_expire_coverage_endpoint(req: rs.ExpireCoverageRequest) -> dict:
    """Expire a seat's coverage for the named files.

    ``files`` is required by the request model, so this endpoint can never
    perform the wholesale expiry that :func:`sidecar.review_state.expire_coverage`
    reads a ``None`` as -- see :class:`sidecar.review_state_client.ExpireCoverageRequest`.
    """
    return {"count": review_state.expire_coverage(req.repo, req.pr, req.seat, req.files)}
