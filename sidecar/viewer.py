"""Server-rendered HTML for the metrics viewer (pure rendering, no I/O).

One self-contained page (inline CSS, zero scripts, zero dependencies) that
makes the ``review_runs`` evidence base and the review system's health
glanceable in a browser: per provider+model aggregates, per-slot aggregates,
the newest raw runs with PR links, current reviewer health, and open provider
cooldowns. Every DB-sourced string is escaped -- the page must stay XSS-clean
even though writers are trusted.
"""

from __future__ import annotations

from html import escape

from .status import DEGRADED_STATES

_STYLE = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem auto;
       max-width: 72rem; padding: 0 1rem; line-height: 1.4; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.3rem 0.6rem;
         border-bottom: 1px solid color-mix(in srgb, currentColor 25%, transparent); }
th { font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: #2da44e; } .bad { color: #cf222e; font-weight: 600; }
.muted { opacity: 0.65; } .notice { padding: 0.8rem 1rem; border-radius: 6px;
  background: color-mix(in srgb, currentColor 8%, transparent); margin: 1rem 0; }
form { margin: 1rem 0; } input { padding: 0.2rem 0.4rem; }
"""


def _outcome_cell(outcome: str) -> str:
    cls = "ok" if outcome == "ok" else "bad"
    return f'<td class="{cls}">{escape(outcome)}</td>'


def _pr_cell(repo: str, pr: object) -> str:
    """Render the PR-link cell, degrading to a dash when ``pr`` is absent.

    The schema requires ``pr`` today, but this function's never-raises pure
    contract must not depend on a constraint two modules away.
    """
    if pr is None:
        return '<td class="num">—</td>'
    return (
        f'<td class="num"><a href="https://github.com/{escape(repo)}/pull/{int(pr)}">'
        f"#{int(pr)}</a></td>"
    )


def _num(value: object) -> str:
    return f'<td class="num">{escape(str(value if value is not None else "—"))}</td>'


def _section(title: str, headers: list[tuple[str, bool]], rows: list[str], empty: str) -> str:
    """Render one table section, or a muted empty-notice when there are no rows.

    Each header is ``(label, numeric)`` -- numericness is declared per column
    rather than inferred from the label text, so renaming a header can never
    silently change its alignment.
    """
    body = f'<p class="muted">{escape(empty)}</p>'
    if rows:
        head = "".join(
            f'<th class="num">{escape(label)}</th>' if numeric else f"<th>{escape(label)}</th>"
            for label, numeric in headers
        )
        body = f"<table><tr>{head}</tr>{''.join(rows)}</table>"
    return f"<h2>{escape(title)}</h2>{body}"


def render_page(
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
    """Render the full viewer page from already-fetched data.

    Pure rendering: every argument is plain already-queried data (aggregate
    dicts from :mod:`sidecar.run_metrics`, health rows from
    :mod:`sidecar.reviewer_health`, the open-cooldown mapping from
    :mod:`sidecar.circuit_breaker`) plus the echoed filter values -- the
    function performs no I/O and never raises on empty inputs, so the caller
    can always produce a 200. ``db_enabled=False`` renders the unconfigured
    notice; ``db_error=True`` renders the configured-but-unreachable notice
    (the caller substitutes empty data in that case).
    """
    parts: list[str] = [
        f"<style>{_STYLE}</style>",
        "<title>fuko review metrics</title>",
        "<h1>fuko review metrics</h1>",
        '<form method="get">'
        f'repo <input name="repo" value="{escape(repo or "")}" placeholder="owner/name"> '
        f'days <input name="days" value="{days}" size="4"> '
        "<button>filter</button></form>",
    ]

    if not db_enabled:
        parts.append(
            '<p class="notice">No database configured (FUKO_DATABASE_URL unset) — '
            "nothing to show.</p>"
        )
    elif db_error:
        parts.append('<p class="notice">Database unreachable — showing empty data.</p>')

    parts.append(
        _section(
            "Models",
            [
                ("provider", False),
                ("model", False),
                ("runs", True),
                ("ok", True),
                ("not ok", True),
                ("avg s", True),
                ("findings", True),
            ],
            [
                f"<tr><td>{escape(m['provider'])}</td><td>{escape(m['model'])}</td>"
                f"{_num(m['runs'])}{_num(m['ok'])}{_num(m['not_ok'])}"
                f"{_num(m['avg_duration_s'])}{_num(m['findings'])}</tr>"
                for m in summary
            ],
            "no runs in this window",
        )
    )

    parts.append(
        _section(
            "Slots",
            [
                ("slot", False),
                ("runs", True),
                ("ok", True),
                ("not ok", True),
                ("avg s", True),
                ("findings", True),
            ],
            [
                f"<tr><td>{escape(s['slot'])}</td>{_num(s['runs'])}{_num(s['ok'])}"
                f"{_num(s['not_ok'])}{_num(s['avg_duration_s'])}{_num(s['findings'])}</tr>"
                for s in slots
            ],
            "no slot-attributed runs in this window",
        )
    )

    parts.append(
        _section(
            "Recent runs",
            [
                ("when", False),
                ("repo", False),
                ("PR", True),
                ("slot", False),
                ("model", False),
                ("outcome", False),
                ("s", True),
                ("tries", True),
                ("findings", True),
            ],
            [
                f'<tr><td class="muted">{escape(r["started_at"][:16])}</td>'
                f"<td>{escape(r['repo'])}</td>"
                f"{_pr_cell(r['repo'], r['pr'])}"
                f"<td>{escape(r['slot'] or '—')}</td>"
                f"<td>{escape(r['provider'])}/{escape(r['model'])}</td>"
                f"{_outcome_cell(r['outcome'])}{_num(round(r['duration_s']))}"
                f"{_num(r['attempts'])}{_num(r['findings'])}</tr>"
                for r in recent
            ],
            "no runs recorded yet",
        )
    )

    parts.append(
        _section(
            "Reviewer health (last observed)",
            [
                ("repo", False),
                ("reviewer", False),
                ("state", False),
                ("observed", False),
                ("detail", False),
            ],
            [
                f"<tr><td>{escape(h['repo'])}</td><td>{escape(h['reviewer'])}</td>"
                f'<td class="{"bad" if h["state"] in DEGRADED_STATES else "ok"}">'
                f"{escape(h['state'])}</td>"
                f'<td class="muted">{escape(h["observed_at"][:16])}</td>'
                f'<td class="muted">{escape(h.get("detail") or "")}</td></tr>'
                for h in health
            ],
            "no observations yet",
        )
    )

    parts.append(
        _section(
            "Provider cooldowns (open breakers)",
            [("provider", False), ("cooling until", False)],
            [
                f"<tr><td>{escape(provider)}</td><td>{escape(until[:16])}</td></tr>"
                for provider, until in sorted(cooldowns.items())
            ],
            "no providers cooling down",
        )
    )

    return "".join(parts)
