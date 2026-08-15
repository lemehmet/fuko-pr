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
    backend: str = "pr-agent",
) -> None:
    """Insert one review-run row (no-op when persistence is disabled).

    ``backend`` is the driver that produced the run (#99); it defaults to
    ``"pr-agent"`` so an omitting caller writes the same value the schema backfill
    applied to pre-existing rows.
    """
    if not _enabled():
        return
    from .db import db

    with db() as conn:
        conn.execute(
            "INSERT INTO review_runs "
            "(repo, pr, provider, model, slot, duration_s, attempts, outcome, findings, "
            "detail, backend) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
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
                backend,
            ),
        )


def slot_summary(repo: str | None = None, days: int = 30) -> list[dict]:
    """Aggregate runs per SLOT over the last ``days`` (empty when disabled).

    The slot view shows lane health independent of which model currently
    occupies it (slots are model-agnostic by design); rows without a slot
    (solo configs, rescued-by-backup rows keep their branch slot) are skipped.
    """
    if not _enabled():
        return []
    from .db import db

    where = "WHERE slot IS NOT NULL AND started_at > now() - make_interval(days => %s)"
    params: list = [min(max(1, days), 3650)]
    if repo:
        where += " AND repo = %s"
        params.append(repo)

    with db() as conn:
        rows = conn.execute(
            "SELECT slot, count(*), "
            "count(*) FILTER (WHERE outcome = 'ok'), "
            "count(*) FILTER (WHERE outcome != 'ok'), "
            "avg(duration_s), coalesce(sum(findings), 0) "
            f"FROM review_runs {where} "
            "GROUP BY slot ORDER BY slot",
            params,
        ).fetchall()
    return [
        {
            "slot": slot,
            "runs": runs,
            "ok": ok,
            "not_ok": not_ok,
            "avg_duration_s": round(float(avg_duration), 1) if avg_duration is not None else None,
            "findings": int(findings),
        }
        for slot, runs, ok, not_ok, avg_duration, findings in rows
    ]


def recent_runs(repo: str | None = None, limit: int = 50) -> list[dict]:
    """Return the newest run rows, bounded (empty when disabled).

    ``limit`` is clamped to [1, 200] so the viewer can never issue an
    unbounded read as ``review_runs`` grows.
    """
    if not _enabled():
        return []
    from .db import db

    where = ""
    params: list = []
    if repo:
        where = "WHERE repo = %s"
        params.append(repo)
    params.append(min(max(1, limit), 200))

    with db() as conn:
        rows = conn.execute(
            "SELECT repo, pr, provider, model, slot, started_at, duration_s, "
            "attempts, outcome, findings, backend "
            f"FROM review_runs {where} ORDER BY started_at DESC LIMIT %s",
            params,
        ).fetchall()
    return [
        {
            "repo": repo_,
            "pr": pr,
            "provider": provider,
            "model": model,
            "slot": slot,
            "started_at": started_at.isoformat(),
            "duration_s": float(duration_s),
            "attempts": attempts,
            "outcome": outcome,
            "findings": findings,
            "backend": backend,
        }
        for (
            repo_,
            pr,
            provider,
            model,
            slot,
            started_at,
            duration_s,
            attempts,
            outcome,
            findings,
            backend,
        ) in rows
    ]


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
