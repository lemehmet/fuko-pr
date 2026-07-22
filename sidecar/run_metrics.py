"""Per-branch review-run metrics, shared across runs via Postgres.

One row per model branch per review round (see ``migrations/004``): which
provider+model ran under which slot, how long it took, how many failover
attempts it needed, how it ended, and how many findings it produced. This is
the evidence base for promoting experimental models to known-good and for
comparing slots by cost once token/cost capture exists.

Best-effort by design, mirroring :mod:`sidecar.circuit_breaker`: with no
Postgres configured (``FUKO_DATABASE_URL`` unset) these functions degrade to
no-ops -- metrics must never block or fail a review.
"""

from __future__ import annotations

from .config import settings


def _enabled() -> bool:
    """Run-metrics persistence requires the shared Postgres store."""
    return bool(settings.database_url)


def record(
    repo: str,
    pr: int,
    provider: str,
    model: str,
    *,
    slot: str | None = None,
    duration_s: float = 0.0,
    attempts: int = 1,
    outcome: str = "ok",
    findings: int | None = None,
    detail: str = "",
) -> None:
    """Insert one review-run row (no-op when persistence is disabled)."""
    if not _enabled():
        return
    from .db import db

    with db() as conn:
        conn.execute(
            "INSERT INTO review_runs "
            "(repo, pr, provider, model, slot, duration_s, attempts, outcome, findings, detail) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                repo,
                pr,
                provider,
                model,
                slot,
                duration_s,
                attempts,
                outcome,
                findings,
                (detail or "")[:500],
            ),
        )


def summary(repo: str | None = None, days: int = 30) -> list[dict]:
    """Aggregate runs per provider+model over the last ``days`` (empty when disabled).

    The grouping the whole exercise exists for: runs, outcomes, average
    duration, and total findings per model -- filtered to one repo when given.
    """
    if not _enabled():
        return []
    from .db import db

    where = "WHERE started_at > now() - make_interval(days => %s)"
    params: list = [min(max(1, days), 3650)]
    if repo:
        where += " AND repo = %s"
        params.append(repo)

    with db() as conn:
        rows = conn.execute(
            "SELECT provider, model, count(*), "
            "count(*) FILTER (WHERE outcome = 'ok'), "
            "count(*) FILTER (WHERE outcome != 'ok'), "
            "avg(duration_s), coalesce(sum(findings), 0) "
            f"FROM review_runs {where} "
            "GROUP BY provider, model ORDER BY count(*) DESC",
            params,
        ).fetchall()
    return [
        {
            "provider": provider,
            "model": model,
            "runs": runs,
            "ok": ok,
            "not_ok": not_ok,
            "avg_duration_s": round(float(avg_duration), 1) if avg_duration is not None else None,
            "findings": int(findings),
        }
        for provider, model, runs, ok, not_ok, avg_duration, findings in rows
    ]
