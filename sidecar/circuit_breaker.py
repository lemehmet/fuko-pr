"""Per-provider circuit-breaker cooldowns, shared across repos via Postgres.

Throttling is enforced on a provider's API key (typically shared across a user's
repos), so the cooldown is keyed by provider id and is global -- if one repo's
review discovers a provider is throttled, every concurrent review skips it until
the window elapses. The state lives in the same Postgres as the knowledge base
(the ``provider_cooldown`` sibling table).

Best-effort by design: with no Postgres configured (``FUKO_DATABASE_URL`` unset,
e.g. a sqlite-vec sidecar) these functions degrade to no-ops, so per-job failover
still works -- it just can't be shared across jobs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .config import settings


def _enabled() -> bool:
    """Circuit-breaker persistence requires the shared Postgres store."""
    return bool(settings.database_url)


def get_cooldowns() -> dict[str, str]:
    """Return ``{provider: cooldown_until_iso}`` for providers still cooling down."""
    if not _enabled():
        return {}
    from .db import db

    with db() as conn:
        rows = conn.execute(
            "SELECT provider, cooldown_until FROM provider_cooldown "
            "WHERE cooldown_until > now()"
        ).fetchall()
    return {provider: until.isoformat() for provider, until in rows}


def trip(provider: str, cooldown_seconds: int, reason: str = "") -> str | None:
    """Open ``provider``'s breaker for ``cooldown_seconds``; return the ISO end time.

    Upserts so a repeated throttle just extends the window. Returns ``None`` when
    persistence is disabled.
    """
    if not _enabled():
        return None
    until = datetime.now(timezone.utc) + timedelta(seconds=max(1, cooldown_seconds))
    from .db import db

    with db() as conn:
        conn.execute(
            "INSERT INTO provider_cooldown (provider, cooldown_until, tripped_at, reason) "
            "VALUES (%s, %s, now(), %s) "
            "ON CONFLICT (provider) DO UPDATE SET "
            "cooldown_until = EXCLUDED.cooldown_until, "
            "tripped_at = now(), reason = EXCLUDED.reason",
            (provider, until, (reason or "")[:500]),
        )
    return until.isoformat()
