"""FastAPI app exposing ``/ingest`` ``/query`` ``/learnings`` ``/forget`` ``/healthz``."""

import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status

from . import circuit_breaker
from . import models
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


app = FastAPI(title="fuko-pr sidecar", version="0.2.0", lifespan=lifespan)

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


@app.post("/ingest-threads", dependencies=[Depends(_auth)])
def ingest_threads_endpoint(req: models.IngestThreadsRequest) -> dict:
    """Mine resolved review threads for learnings and ingest them."""
    items = [
        it
        for it in (threads_mod.select_learning(t, req.bot_login) for t in req.threads)
        if it is not None
    ]
    inserted, skipped = _store.ingest(req.repo, items)
    return {"considered": len(req.threads), "inserted": inserted, "skipped": skipped}


@app.get("/cb/cooldowns", response_model=models.CooldownsResponse, dependencies=[Depends(_auth)])
def cb_cooldowns_endpoint() -> dict:
    """Return the providers whose circuit breaker is currently open (cooling down)."""
    return {"cooldowns": circuit_breaker.get_cooldowns()}


@app.post("/cb/trip", response_model=models.TripResponse, dependencies=[Depends(_auth)])
def cb_trip_endpoint(req: models.TripRequest) -> dict:
    """Open a provider's circuit breaker for a cooldown window (idempotent upsert)."""
    until = circuit_breaker.trip(req.provider, req.cooldown_seconds, req.reason or "")
    return {"provider": req.provider, "cooldown_until": until}
