"""The review-state ledgers as an operator page: which lanes exist, and what each holds.

``review_findings`` and ``review_coverage`` accumulate every round and were
write-mostly until this page (#235): the reviewer read them back into its own
prompt and nothing else could see them at all. This is the window -- the same
step ``/ui/kb`` was for the knowledge base, one tier further along.

The page keeps the two ledgers' asymmetry visible rather than flattening it. A
finding is a CLAIM that survives until a round settles it with a reason, so
every status is shown and a row a later round contradicted (``reopened > 0``) is
marked. A coverage entry is an ASSURANCE that dies when the delta touches its
file, so expired entries are shown beside live ones and labelled as what a round
EXAMINED -- never as a standing conclusion.

Read-only, by design and not by omission: nothing here closes, reopens or
expires a row, so the page needs no session and mounts no ``POST``. It is
unauthenticated for the same reason ``/ui/metrics`` is (#71).
"""

from __future__ import annotations

import sys
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .. import review_state
from ..config import settings
from . import components as c
from .layout import document, page

PAGE = page("ledger")

router = APIRouter()

PAGE_SIZE = 50

_METRICS = page("metrics")

_BODY_PREVIEW_CHARS = 160

_SECTIONS = ("findings", "coverage")

_LANE_COLUMNS: list[c.Column] = [
    ("repo", False),
    ("PR", True),
    ("seat", False),
    ("round", True),
    ("last activity", False),
    *((status, True) for status in review_state.STATUS_ORDER),
    ("not offered", True),
    ("reopened", True),
    ("coverage live/all", False),
    ("carry-fwd", True),
    ("settled", True),
]

_FINDING_COLUMNS: list[c.Column] = [
    ("round", True),
    ("seat", False),
    ("status", False),
    ("severity", False),
    ("category", False),
    ("where", False),
    ("finding", False),
    ("why it is in this state", False),
]

_COVERAGE_COLUMNS: list[c.Column] = [
    ("round", True),
    ("seat", False),
    ("state", False),
    ("file", False),
    ("region", False),
    ("checked", False),
    ("conclusion", False),
]


def _url(**params: object) -> str:
    """Build one of this page's URLs with its query string."""
    return f"{PAGE.path}{c.query_string(params)}"


def _rate(value: float | None) -> str:
    """Render a rate as a percentage, or an em dash when there is nothing to divide.

    ``None`` is not ``0%``: a lane with a single round has no finding a later
    round could have acted on, and a zero would read as a seat that settles
    nothing (see :func:`sidecar.review_state._rate`).
    """
    return "—" if value is None else f"{value:.0%}"


def _timestamp(value: str | None) -> str:
    """Trim an ISO timestamp to minutes, the way the metrics page does."""
    return value[:16] if value else "—"


def _blob_href(repo: str, sha: str, path: str, line: int | None) -> str:
    """Build the GitHub link for a stored file path, or ``""`` when it cannot be built.

    Every segment is percent-encoded here rather than left to
    :func:`sidecar.web.components.safe_href`, which allow-lists the SCHEME and
    passes the rest through untouched. These paths are model output produced
    while reading a contributor-controlled checkout, so a ``#`` or a ``?`` in one
    would otherwise silently truncate the URL into a different one, and a quote
    would land in the attribute -- ``esc`` keeps that inert, but the link would
    still be wrong. ``safe="/"`` keeps directory separators as separators and
    encodes everything else; the commit is a single segment and keeps nothing.
    """
    if not sha or not path:
        return ""
    anchor = f"#L{int(line)}" if line is not None else ""
    return (
        f"https://github.com/{quote(repo, safe='/')}/blob/"
        f"{quote(sha, safe='')}/{quote(path, safe='/')}{anchor}"
    )


def _where_cell(repo: str, sha: str, path: str, line: int | None) -> str:
    """Render the file[:line] cell, linked to the blob when a head commit is known."""
    label = f"{path}:{line}" if line is not None else path
    href = _blob_href(repo, sha, path, line)
    return c.raw_cell(c.link(href, label) if href else c.esc(label), css="nowrap")


def _detail(summary: str, full: str) -> str:
    """Render a collapsible block: an escaped one-line summary over escaped full text."""
    head = summary[:_BODY_PREVIEW_CHARS].replace("\n", " ")
    ellipsis = "…" if len(summary) > _BODY_PREVIEW_CHARS else ""
    return f"<details><summary>{c.esc(head)}{ellipsis}</summary><pre>{c.esc(full)}</pre></details>"


def _lane_row(lane: review_state.LaneStat, *, linked: bool) -> str:
    """Render one lane of the index; ``linked`` adds the drill-down on the PR number."""
    pr_cell = (
        c.raw_cell(c.link(_url(repo=lane.repo, pr=lane.pr), f"#{lane.pr}"), numeric=True)
        if linked
        else c.cell(f"#{lane.pr}", numeric=True)
    )
    counts = "".join(
        c.cell(lane.counts.get(status, 0), numeric=True) for status in review_state.STATUS_ORDER
    )
    return (
        "<tr>"
        + c.cell(lane.repo)
        + pr_cell
        + c.cell(lane.seat)
        + c.cell(lane.latest_round, numeric=True)
        + c.cell(_timestamp(lane.last_activity), css="muted")
        + counts
        + c.cell(lane.never_offered, numeric=True, css="bad" if lane.never_offered else "muted")
        + c.cell(lane.reopened, numeric=True, css="bad" if lane.reopened else "muted")
        + c.cell(f"{lane.coverage_live}/{lane.coverage_total}", css="muted")
        + c.cell(_rate(lane.carry_forward_rate), numeric=True)
        + c.cell(_rate(lane.settle_rate), numeric=True)
        + "</tr>"
    )


def _finding_row(finding: review_state.LedgerFinding, *, repo: str) -> str:
    """Render one finding, marking a row a later round contradicted."""
    status = c.badge(finding.status, css="bad" if finding.anomalous else "")
    if finding.anomalous:
        status += f' <span class="bad">reopened ×{c.esc(finding.reopened)}</span>'
    claim = _detail(finding.title, finding.body)
    if finding.evidence:
        claim += f'<p class="muted">{c.esc(finding.evidence)}</p>'
    return (
        "<tr>"
        + c.cell(finding.round, numeric=True)
        + c.cell(finding.seat)
        + c.raw_cell(status, css="nowrap")
        + c.cell(finding.severity)
        + c.cell(finding.category)
        + _where_cell(repo, finding.head_sha, finding.file, finding.line)
        + c.raw_cell(claim)
        + c.cell(finding.status_reason or "—", css="muted")
        + "</tr>"
    )


def _text_html(value: str) -> str:
    """Render free text as escaped markup, folding anything long behind a summary.

    The stored text is what a round actually said, so nothing is thrown away
    here; a long entry simply starts folded so one verbose row cannot push the
    rest of the table off the screen.
    """
    if len(value) <= _BODY_PREVIEW_CHARS:
        return c.esc(value or "—")
    return _detail(value, value)


def _text_cell(value: str, *, css: str = "") -> str:
    """Render one free-text cell through :func:`_text_html`."""
    return c.raw_cell(_text_html(value), css=css)


def _coverage_row(entry: review_state.LedgerCoverage, *, repo: str) -> str:
    """Render one coverage entry, distinguishing a live assurance from an expired one."""
    if entry.live:
        state = c.badge("live", css="ok")
    else:
        state = c.badge("expired", css="muted") + f" {c.esc(_timestamp(entry.expired_at))}"
    conclusion = _text_html(entry.conclusion)
    if entry.evidence:
        conclusion += f'<p class="muted">{c.esc(entry.evidence)}</p>'
    return (
        f"<tr{c.attrs(class_=None if entry.live else 'muted')}>"
        + c.cell(entry.round, numeric=True)
        + c.cell(entry.seat)
        + c.raw_cell(state, css="nowrap")
        + _where_cell(repo, entry.head_sha, entry.file, None)
        + c.cell(entry.region or "—", css="muted")
        + _text_cell(entry.checked)
        + c.raw_cell(conclusion)
        + "</tr>"
    )


def _notices(*, db_enabled: bool, db_error: bool) -> str:
    """Render the store's state, keeping unconfigured, unreachable and empty distinct.

    Three states rather than two: a sqlite-vec deployment has no ledger tables
    at all and never will (:mod:`sidecar.review_state` is Postgres-only), which
    is a permanent property of the deployment; an unreachable Postgres is a fault
    to go and fix; and a healthy store with nothing in it is the reviewer having
    recorded nothing yet. Collapsing any two of them would tell an operator to
    do the wrong thing.
    """
    if not db_enabled:
        return c.notice(
            "Review state needs the Postgres store (FUKO_DATABASE_URL unset). "
            "A sqlite-vec deployment keeps no ledgers and reviews statelessly.",
            kind="warn",
        )
    if db_error:
        return c.notice(
            "Review-state store unreachable — this is a fault, not an empty ledger.",
            kind="danger",
        )
    return ""


def _filter_form(repo: str | None, pr: int | None, seat: str | None) -> str:
    """Render the lane filter; a repo and a PR together open that PR's detail view."""
    return (
        '<form method="get">'
        + c.field("repo", "repo", repo or "", placeholder="owner/name")
        + c.field("PR", "pr", "" if pr is None else pr, size=6)
        + c.field("seat", "seat", seat or "", placeholder="any seat")
        + "<button>filter</button></form>"
    )


def render_index(
    *,
    index: review_state.LaneIndex,
    repo: str | None,
    pr: int | None,
    seat: str | None,
    offset: int,
    limit: int,
    db_enabled: bool,
    db_error: bool = False,
) -> str:
    """Render the lane index from already-queried rows (pure, never raises).

    One row per ``(repo, pr, seat)``: the lane is the unit because that is what
    the ledgers are keyed by and what a seat carries forward. Lanes whose pull
    request has since been closed or merged are here too -- the read asks GitHub
    nothing, so a lane simply sinks down the ordering as its activity ages.
    """
    body = [
        "<h1>Review state ledgers</h1>",
        '<p class="muted">One lane per (repo, PR, seat). A finding is a claim that '
        "survives until a round settles it; coverage is an assurance that expires.</p>",
        _filter_form(repo, pr, seat),
        _notices(db_enabled=db_enabled, db_error=db_error),
        c.section(
            "Lanes",
            c.table(
                _LANE_COLUMNS,
                [_lane_row(lane, linked=True) for lane in index.lanes],
                "no review state recorded yet",
            ),
        ),
        c.pager(
            PAGE.path,
            {"repo": repo, "pr": pr, "seat": seat},
            offset=offset,
            limit=limit,
            total=index.total,
        ),
    ]
    return document(title="fuko review state", body="".join(body), active=PAGE.slug)


def render_detail(
    *,
    repo: str,
    pr: int,
    index: review_state.LaneIndex,
    findings: review_state.FindingPage,
    coverage: review_state.CoveragePage,
    show: str,
    seat: str | None,
    offset: int,
    limit: int,
    db_enabled: bool,
    db_error: bool = False,
) -> str:
    """Render one pull request's ledgers from already-queried rows (pure, never raises).

    ``show`` selects which ledger is paged, so one ``offset`` can never mean two
    different things: the sections carry different row counts and a shared cursor
    would page one of them past its end while the other still had rows.
    """
    shown = show if show in _SECTIONS else _SECTIONS[0]
    tabs = " · ".join(
        f"<strong>{c.esc(name)}</strong>"
        if name == shown
        else c.link(_url(repo=repo, pr=pr, seat=seat, show=name), name)
        for name in _SECTIONS
    )
    body = [
        f"<h1>{c.esc(repo)} #{c.esc(pr)}</h1>",
        '<p class="pager">'
        + c.link(_url(), "← all lanes")
        + " "
        + c.link(f"https://github.com/{quote(repo, safe='/')}/pull/{int(pr)}", "pull request ↗")
        + " "
        + c.link(f"{_METRICS.path}{c.query_string({'repo': repo})}", "run metrics")
        + "</p>",
        _notices(db_enabled=db_enabled, db_error=db_error),
        c.section(
            "Seats on this pull request",
            c.table(
                _LANE_COLUMNS,
                [_lane_row(lane, linked=False) for lane in index.lanes],
                "no review state recorded for this pull request",
            ),
        ),
        f'<p class="pager">{tabs}</p>',
    ]
    if shown == "coverage":
        body.append(
            c.section(
                "Coverage — what a round examined",
                '<p class="muted">Each entry is an assurance about one round\'s head, '
                "not a standing conclusion: an expired entry means the delta has since "
                "touched that file.</p>"
                + c.table(
                    _COVERAGE_COLUMNS,
                    [_coverage_row(entry, repo=repo) for entry in coverage.rows],
                    "no coverage recorded for this pull request",
                ),
            )
        )
        total = coverage.total
    else:
        body.append(
            c.section(
                "Findings — what a round claimed",
                '<p class="muted">Every status, including rows a verdict closed and rows '
                "fuko retired. A reopened row is one a round declared settled and a later "
                "round contradicted.</p>"
                + c.table(
                    _FINDING_COLUMNS,
                    [_finding_row(finding, repo=repo) for finding in findings.rows],
                    "no findings recorded for this pull request",
                ),
            )
        )
        total = findings.total
    body.append(
        c.pager(
            PAGE.path,
            {"repo": repo, "pr": pr, "seat": seat, "show": shown},
            offset=offset,
            limit=limit,
            total=total,
        )
    )
    return document(title=f"fuko review state — {repo} #{pr}", body="".join(body), active=PAGE.slug)


@router.get(PAGE.path, response_class=HTMLResponse)
def view(
    repo: str | None = None,
    pr: str | None = None,
    seat: str | None = None,
    show: str = "findings",
    offset: int = 0,
    limit: int = PAGE_SIZE,
) -> str:
    """Serve the ledger page: the lane index, or one pull request's detail.

    Fetches and degrades, per the package's route/render split. The three
    unhealthy-or-empty states stay distinguishable because the configuration
    gate is tested HERE and not inside the read: with no ``database_url`` the
    connection would fail on the way to the pool, and an unconfigured deployment
    would be reported as a broken one.

    The reads deliberately raise (they are not ``review_state._best_effort``
    wrapped), which is what makes "unreachable" reportable at all -- a swallowed
    exception would render the same empty table a healthy store does.

    ``pr`` arrives as text and is parsed by :func:`sidecar.web.components.form_int`
    because it is bound to a form field: the filter submits an untouched PR box
    as ``pr=``, which an ``int | None`` parameter rejects with a 422 rather than
    treating as absent. Declaring it text is what makes filtering by repository
    alone -- the form's own default -- reach this function at all.
    """
    number = c.form_int(pr)
    limit = min(max(1, limit), review_state.MAX_LEDGER_ROWS)
    offset = max(0, offset)
    seat = seat or None
    db_enabled = bool(settings.database_url)
    index = review_state.LaneIndex()
    findings = review_state.FindingPage()
    coverage = review_state.CoveragePage()
    db_error = False
    detail = bool(repo) and number is not None
    if db_enabled:
        try:
            if detail:
                index = review_state.lanes(repo=repo, pr=number, limit=review_state.MAX_LANES)
                if show == "coverage":
                    coverage = review_state.pr_coverage(
                        repo, number, seat=seat, limit=limit, offset=offset
                    )
                else:
                    findings = review_state.pr_findings(
                        repo, number, seat=seat, limit=limit, offset=offset
                    )
            else:
                index = review_state.lanes(
                    repo=repo or None, pr=number, seat=seat, limit=limit, offset=offset
                )
        except Exception as e:
            print(f"fuko: ledger view degraded (database unreachable?): {e}", file=sys.stderr)
            db_error = True
    if detail:
        return render_detail(
            repo=repo or "",
            pr=int(number or 0),
            index=index,
            findings=findings,
            coverage=coverage,
            show=show,
            seat=seat,
            offset=offset,
            limit=limit,
            db_enabled=db_enabled,
            db_error=db_error,
        )
    return render_index(
        index=index,
        repo=repo,
        pr=number,
        seat=seat,
        offset=offset,
        limit=limit,
        db_enabled=db_enabled,
        db_error=db_error,
    )
