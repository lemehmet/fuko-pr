"""FastAPI app exposing ``/ingest`` ``/query`` ``/learnings`` ``/forget`` ``/healthz``."""

import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import HTMLResponse

from . import circuit_breaker
from . import models
from . import reviewer_health
from . import run_metrics
from . import viewer
from . import threads as threads_mod
from .config import settings
from .fukoconfig import load_config
from .stores import get_store


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

# The sidecar serves one store, selected by .fuko.toml (defaults to Postgres).
_store = get_store(load_config().knowledge)


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
) -> dict:
    """List stored learnings for browsing, newest first.

    Unlike ``/query`` (semantic + file-scoped, for review-time retrieval) this is
    a plain inspection listing of live (non-expired) learnings, optionally filtered
    by ``repo`` and ``source``. ``limit`` is clamped to 500 and ``offset`` floored
    at 0. ``count`` is the total matching the filters (for paging), not the page
    size.
    """
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    learnings, total = _store.list_learnings(repo=repo, source=source, limit=limit, offset=offset)
    return {"learnings": learnings, "count": total}


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


@app.get("/metrics/view", response_class=HTMLResponse)
def metrics_view_endpoint(repo: str | None = None, days: int = 30) -> str:
    """Serve the human-facing metrics page (deliberately unauthenticated).

    Read-only aggregates on a LAN-only deployment (decision in #71); the
    ``/healthz`` probe set the unauthenticated precedent. Every API endpoint
    keeps its bearer auth -- only this HTML view is open, and it can reach
    nothing mutating.
    """
    days = min(max(1, days), 3650)
    data: dict = {"summary": [], "slots": [], "recent": [], "health": [], "cooldowns": {}}
    db_error = False
    try:
        data = {
            "summary": run_metrics.summary(repo=repo, days=days),
            "slots": run_metrics.slot_summary(repo=repo, days=days),
            "recent": run_metrics.recent_runs(repo=repo),
            "health": reviewer_health.all_states(),
            "cooldowns": circuit_breaker.get_cooldowns(),
        }
    except Exception as e:
        print(f"fuko: metrics view degraded (database unreachable?): {e}", file=sys.stderr)
        db_error = True
    return viewer.render_page(
        **data,
        repo=repo,
        days=days,
        db_enabled=bool(settings.database_url),
        db_error=db_error,
    )


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
