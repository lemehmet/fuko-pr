"""The review-metrics page: per-model and per-slot aggregates, recent runs, health.

Follows the package's route/render split -- :func:`view` fetches and degrades to
empty data when the database is unreachable, :func:`render` is pure and takes
already-queried rows, so the page is always a 200 and never depends on the
store's availability to render its own chrome.
"""

from __future__ import annotations

import sys

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .. import circuit_breaker, reviewer_health, run_metrics
from ..config import settings
from ..status import DEGRADED_STATES
from . import components as c
from .layout import document, page

PAGE = page("metrics")

router = APIRouter()

#: The cost columns both aggregate tables carry (#152), in row order.
#:
#: `in` and `cache rd` are shown side by side on purpose: fresh input beside
#: cached input IS the prompt-caching answer, and reading them apart is what
#: made a ~25x difference in the bill invisible for as long as it was.
_COST_COLUMNS: list[c.Column] = [
    ("in", True),
    ("cache rd", True),
    ("out", True),
    ("$", True),
]

_MODEL_COLUMNS: list[c.Column] = [
    ("provider", False),
    ("model", False),
    ("runs", True),
    ("ok", True),
    ("not ok", True),
    ("avg s", True),
    ("findings", True),
    *_COST_COLUMNS,
]

_SLOT_COLUMNS: list[c.Column] = [
    ("slot", False),
    ("runs", True),
    ("ok", True),
    ("not ok", True),
    ("avg s", True),
    ("findings", True),
    *_COST_COLUMNS,
]

_RECENT_COLUMNS: list[c.Column] = [
    ("when", False),
    ("repo", False),
    ("PR", True),
    ("slot", False),
    ("model", False),
    ("outcome", False),
    ("s", True),
    ("tries", True),
    ("findings", True),
    ("ledger", False),
]

_LEDGER = page("ledger")

_HEALTH_COLUMNS: list[c.Column] = [
    ("repo", False),
    ("reviewer", False),
    ("state", False),
    ("observed", False),
    ("detail", False),
]


def _pr_cell(repo: str, pr: object) -> str:
    """Render the PR-link cell, degrading to a dash when ``pr`` is absent.

    The schema requires ``pr`` today, but this function's never-raises pure
    contract must not depend on a constraint two modules away.
    """
    if pr is None:
        return c.cell(None, numeric=True)
    return c.raw_cell(
        c.link(f"https://github.com/{repo}/pull/{int(pr)}", f"#{int(pr)}"), numeric=True
    )


def _ledger_cell(repo: str, pr: object) -> str:
    """Render the cross-link into this run's review-state lanes, keyed on ``(repo, pr)``.

    Keyed on the pull request and NOT on the seat, deliberately. A run row's
    ``slot`` and a ledger lane's ``seat`` look like the same thing and are not:
    ``runner._branch_seats`` prefers the slot but falls back to
    ``provider/name``, then ``provider/name#idx``, then the default seat, so a
    join on that column would silently miss every solo config. The pull request
    is the key both sides always agree on.
    """
    if pr is None:
        return c.cell(None)
    return c.raw_cell(c.link(f"{_LEDGER.path}{c.query_string({'repo': repo, 'pr': int(pr)})}", "↦"))


def _tokens(value: object) -> str:
    """Format a token total compactly; an em dash when nothing was measured.

    ``None`` must not render as ``0``: an unmeasured lane (every pr-agent row)
    would otherwise be indistinguishable from a free one.
    """
    if value is None:
        return "—"
    count = int(value)
    for unit, scale in (("G", 1_000_000_000), ("M", 1_000_000), ("k", 1_000)):
        if count >= scale:
            return f"{count / scale:.1f}{unit}"
    return str(count)


def _cost_cells(row: dict) -> str:
    """Render the four cost cells shared by the model and slot tables."""
    cost = row.get("cost_usd")
    return (
        c.cell(_tokens(row.get("input_tokens")), numeric=True)
        + c.cell(_tokens(row.get("cache_read_tokens")), numeric=True)
        + c.cell(_tokens(row.get("output_tokens")), numeric=True)
        + c.cell("—" if cost is None else f"${float(cost):,.2f}", numeric=True)
    )


def _model_rows(summary: list[dict]) -> list[str]:
    return [
        "<tr>"
        + c.cell(m["provider"])
        + c.cell(m["model"])
        + c.cell(m["runs"], numeric=True)
        + c.cell(m["ok"], numeric=True)
        + c.cell(m["not_ok"], numeric=True)
        + c.cell(m["avg_duration_s"], numeric=True)
        + c.cell(m["findings"], numeric=True)
        + _cost_cells(m)
        + "</tr>"
        for m in summary
    ]


def _slot_rows(slots: list[dict]) -> list[str]:
    return [
        "<tr>"
        + c.cell(s["slot"])
        + c.cell(s["runs"], numeric=True)
        + c.cell(s["ok"], numeric=True)
        + c.cell(s["not_ok"], numeric=True)
        + c.cell(s["avg_duration_s"], numeric=True)
        + c.cell(s["findings"], numeric=True)
        + _cost_cells(s)
        + "</tr>"
        for s in slots
    ]


def _recent_rows(recent: list[dict]) -> list[str]:
    return [
        "<tr>"
        + c.cell(r["started_at"][:16], css="muted")
        + c.cell(r["repo"])
        + _pr_cell(r["repo"], r["pr"])
        + c.cell(r["slot"] or "—")
        + c.cell(f"{r['provider']}/{r['model']}")
        + c.cell(r["outcome"], css="ok" if r["outcome"] == "ok" else "bad")
        + c.cell(round(r["duration_s"]), numeric=True)
        + c.cell(r["attempts"], numeric=True)
        + c.cell(r["findings"], numeric=True)
        + _ledger_cell(r["repo"], r["pr"])
        + "</tr>"
        for r in recent
    ]


def _health_rows(health: list[dict]) -> list[str]:
    return [
        "<tr>"
        + c.cell(h["repo"])
        + c.cell(h["reviewer"])
        + c.cell(h["state"], css="bad" if h["state"] in DEGRADED_STATES else "ok")
        + c.cell(h["observed_at"][:16], css="muted")
        + c.cell(h.get("detail") or "", css="muted")
        + "</tr>"
        for h in health
    ]


def render(
    *,
    summary: list[dict],
    slots: list[dict],
    recent: list[dict],
    health: list[dict],
    cooldowns: dict[str, str],
    repo: str | None,
    days: int,
    db_enabled: bool,
    db_error: bool = False,
) -> str:
    """Render the metrics page from already-fetched data.

    Every argument is plain queried data (aggregate dicts from
    :mod:`sidecar.run_metrics`, health rows from :mod:`sidecar.reviewer_health`,
    the open-cooldown mapping from :mod:`sidecar.circuit_breaker`) plus the
    echoed filter values. ``db_enabled=False`` renders the unconfigured notice;
    ``db_error=True`` renders the configured-but-unreachable notice, with the
    caller having substituted empty data.
    """
    parts: list[str] = [
        "<h1>fuko review metrics</h1>",
        '<form method="get">'
        + c.field("repo", "repo", repo or "", placeholder="owner/name")
        + c.field("days", "days", days, size=4)
        + "<button>filter</button></form>",
    ]

    if not db_enabled:
        parts.append(
            c.notice(
                "No database configured (FUKO_DATABASE_URL unset) — nothing to show.", kind="warn"
            )
        )
    elif db_error:
        parts.append(c.notice("Database unreachable — showing empty data.", kind="danger"))

    parts.append(
        c.section("Models", c.table(_MODEL_COLUMNS, _model_rows(summary), "no runs in this window"))
    )
    parts.append(
        c.section(
            "Slots",
            c.table(_SLOT_COLUMNS, _slot_rows(slots), "no slot-attributed runs in this window"),
        )
    )
    parts.append(
        c.section(
            "Recent runs", c.table(_RECENT_COLUMNS, _recent_rows(recent), "no runs recorded yet")
        )
    )
    parts.append(
        c.section(
            "Reviewer health (last observed)",
            c.table(_HEALTH_COLUMNS, _health_rows(health), "no observations yet"),
        )
    )
    parts.append(
        c.section(
            "Provider cooldowns (open breakers)",
            c.table(
                [("provider", False), ("cooling until", False)],
                [
                    "<tr>" + c.cell(provider) + c.cell(until[:16]) + "</tr>"
                    for provider, until in sorted(cooldowns.items())
                ],
                "no providers cooling down",
            ),
        )
    )

    return document(title="fuko review metrics", body="".join(parts), active=PAGE.slug)


@router.get(PAGE.path, response_class=HTMLResponse)
def view(repo: str | None = None, days: int = 30) -> str:
    """Serve the metrics page (deliberately unauthenticated).

    Read-only aggregates on a LAN-only deployment (decision in #71); the
    ``/healthz`` probe set the unauthenticated precedent. Every API endpoint
    keeps its bearer auth -- this view is open, and it can reach nothing
    mutating.
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
    return render(
        **data,
        repo=repo,
        days=days,
        db_enabled=bool(settings.database_url),
        db_error=db_error,
    )
