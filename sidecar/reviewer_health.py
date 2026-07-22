"""Per-repo reviewer-health observations, shared across runs via Postgres.

The runner records what each external reviewer (CodeRabbit, Copilot) was last
seen doing at the end of every review, and reads those observations at the
start of the next one -- the persistence layer behind next-round escalation
(:func:`sidecar.status.escalation_needed`). Keyed per repo because reviewer
quotas (CodeRabbit plan limits, Copilot premium-request credits) are org/repo
scoped, unlike the per-provider circuit breaker whose API keys are shared.

Best-effort by design, mirroring :mod:`sidecar.circuit_breaker`: with no
Postgres configured (``FUKO_DATABASE_URL`` unset) these functions degrade to
no-ops, so reviews still run -- escalation just can't see past rounds.
"""

from __future__ import annotations

from .config import settings


def _enabled() -> bool:
    """Reviewer-health persistence requires the shared Postgres store."""
    return bool(settings.database_url)


def observe(repo: str, reviewer: str, state: str, pr: int | None = None, detail: str = "") -> None:
    """Upsert the latest observed ``state`` of ``reviewer`` on ``repo``.

    A repeated observation replaces the previous one -- the table holds only
    the most recent state per (repo, reviewer). No-op when persistence is
    disabled.
    """
    if not _enabled():
        return
    from .db import db

    with db() as conn:
        conn.execute(
            "INSERT INTO reviewer_health (repo, reviewer, state, observed_at, pr, detail) "
            "VALUES (%s, %s, %s, now(), %s, %s) "
            "ON CONFLICT (repo, reviewer) DO UPDATE SET "
            "state = EXCLUDED.state, observed_at = now(), "
            "pr = EXCLUDED.pr, detail = EXCLUDED.detail",
            (repo, reviewer, state, pr, (detail or "")[:500]),
        )


def all_states() -> list[dict]:
    """Return the last observed state rows for every repo (empty when disabled).

    The viewer's system-health section wants the whole fleet at a glance;
    :func:`states` stays the per-repo read the runner uses.
    """
    if not _enabled():
        return []
    from .db import db

    with db() as conn:
        rows = conn.execute(
            "SELECT repo, reviewer, state, observed_at, pr, detail "
            "FROM reviewer_health ORDER BY repo, reviewer"
        ).fetchall()
    return [
        {
            "repo": repo,
            "reviewer": reviewer,
            "state": state,
            "observed_at": observed_at.isoformat(),
            "pr": pr,
            "detail": detail,
        }
        for repo, reviewer, state, observed_at, pr, detail in rows
    ]


def states(repo: str) -> list[dict]:
    """Return the last observed state rows for ``repo`` (empty when disabled).

    Row shape matches :func:`sidecar.status.reviewer_states` closely enough for
    :func:`sidecar.status.escalation_needed` to consume either directly:
    ``{"reviewer", "state", "observed_at", "pr", "detail"}``.
    """
    if not _enabled():
        return []
    from .db import db

    with db() as conn:
        rows = conn.execute(
            "SELECT reviewer, state, observed_at, pr, detail FROM reviewer_health WHERE repo = %s",
            (repo,),
        ).fetchall()
    return [
        {
            "reviewer": reviewer,
            "state": state,
            "observed_at": observed_at.isoformat(),
            "pr": pr,
            "detail": detail,
        }
        for reviewer, state, observed_at, pr, detail in rows
    ]
