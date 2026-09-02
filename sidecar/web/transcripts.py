"""The captured-session browser: which runs left a transcript, and what one did.

The last reader epic #236 asks for, and the one that closes it: an operator
takes a pull request number, finds the runs that reviewed it, sees what each
spent its turns on, and reads a session turn by turn -- in a browser, with no
SSH and no ``jq``.

The query shapes are IMPORTED rather than restated. :mod:`sidecar.transcripts`
is the contract both readers share (``fuko transcripts`` over HTTP, this page in
process), the way :mod:`sidecar.web.ledger` imports :mod:`sidecar.review_state`:
a filter this page spelled differently from the command would be two answers to
one question.

**Escaping here is the security boundary, not the house style.** A tool result
is not model output ABOUT a repository -- it is the repository, verbatim, from a
contributor-controlled checkout, plus whatever the agent wrote about it. Every
byte this page renders is therefore assumed hostile: it goes through
:func:`_esc` on the way in, no transcript-derived value is ever used to build a
URL, an attribute or a class, and nothing is interpreted as markup, markdown or
a diff (#241 forbids all three, and the ``<pre>`` boxes are the whole of the
formatting).

**The session view is authenticated; the listing is not.** ``/ui/metrics`` and
``/ui/ledger`` are deliberately open on a LAN-only deployment (#71), and the
listing here is the same exposure class: index-row figures -- repo, PR, seat,
per-tool counts, byte totals -- carrying no file content and no file path. The
session view is not that at all, so it takes the session
:mod:`sidecar.web.security` already mints from ``FUKO_AUTH_TOKEN``: rendering a
reviewed repository's file contents to anyone who can reach the port is a
different decision from publishing aggregate counts, and #240 named this corpus
the first read path onto full reviewed-repo content for exactly that reason.

Degrading honestly needs THREE reads to be distinguishable rather than one,
because they fail independently: the Postgres index (unconfigured / unreachable
/ no row for this key -- #258's un-indexed blobs are real and fetchable), the
blob store (unconfigured / unreachable / holds nothing under this key), and the
filter values themselves (an unparseable ``since`` is a typo, and reporting it
as an outage is precisely the confusion #240 exists to prevent).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from urllib.parse import quote

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse

from .. import transcripts as corpus
from ..config import settings
from ..objectstore import validate_blob_key
from . import components as c
from . import security
from .layout import document, page

PAGE = page("transcripts")

router = APIRouter()

#: Transcripts per listing page.
PAGE_SIZE = 50

#: Stored feed lines rendered per session page. Small next to the listing's page
#: size because one line can carry a whole file: a page of events is bounded by
#: :data:`MAX_BLOCK_CHARS` per block, and 25 of those is already a long scroll.
EVENTS_PER_PAGE = 25

#: Upper bound on ``limit`` for the session view, for :data:`corpus.MAX_ROWS`'
#: reason -- the corpus grows without limit by decision, so no reader may ask
#: for an unbounded slice of one.
MAX_EVENTS = 200

#: Characters of a stored text block shown before it is folded away.
_PREVIEW_CHARS = 160

#: Characters of ONE block this page will render at all. A single tool result is
#: a whole file, and #236 keeps everything, so the cap is what stops one
#: ``Read`` of a vendored bundle from becoming a 40 MB page. Clipping is stated
#: in the output, with the command that returns the un-clipped bytes -- an
#: operator must never mistake this page's limit for the transcript's contents.
MAX_BLOCK_CHARS = 20_000

#: Tools named in a listing cell before it says "+N more", matching
#: ``fuko transcripts list``'s own cap so the two readers describe a run alike.
_TOOLS_SHOWN = 4

_METRICS = page("metrics")

_LEDGER = page("ledger")

_COLUMNS: list[c.Column] = [
    ("when", False),
    ("transcript", False),
    ("repo", False),
    ("PR", True),
    ("seat", False),
    ("model", False),
    ("state", False),
    ("calls", True),
    ("tools", False),
    ("results", True),
    ("re-read", True),
    ("outcome", False),
]


@dataclass(frozen=True)
class SessionLine:
    """One stored feed line, as far as this page could decode it.

    ``event`` is the decoded object when the line was a JSON object, and
    ``None`` when it was not -- a blank-ish tail, a line the scrubber cut
    mid-value (:meth:`sidecar.reviewer.transcript.Scrubber.scrub_partial`), or
    JSON that is not an object. ``raw`` is kept either way, because an operator
    reading a truncated session needs to see the fragment rather than a page
    that silently drops it.
    """

    number: int
    event: dict | None
    raw: str


@dataclass(frozen=True)
class SessionPage:
    """One page of a stored session, plus how many lines the whole feed holds."""

    lines: tuple[SessionLine, ...] = ()
    total: int = 0


def _sanitized(text: str) -> str:
    """Return ``text`` with anything UTF-8 cannot encode replaced.

    Not belt-and-braces: a tool result carrying an unpaired surrogate escape
    (a JSON escape naming half a surrogate pair) is LEGAL JSON, and
    :func:`json.loads` hands it back as a lone-surrogate ``str`` -- the same
    shape that already broke the capture-time
    meter (see :class:`sidecar.reviewer.transcript._Meter`). Escaping keeps it
    inert as markup but not encodable, so the render would succeed and the
    response would then raise ``UnicodeEncodeError`` on its way out: a 500 from
    a page whose whole contract is that it never fails to draw. The corpus is
    contributor-controlled, so this is reachable on purpose, not by accident.
    """
    return text.encode("utf-8", "replace").decode("utf-8", "replace")


def _esc(value: object) -> str:
    """Escape one transcript-derived value: the single door stored bytes come through.

    Everything read out of a blob is funnelled through here rather than through
    :func:`sidecar.web.components.esc` directly, so the sanitizing pass cannot
    be forgotten at one call site while the others have it.
    """
    return c.esc(_sanitized(str(value)))


def read_session(data: bytes, *, offset: int, limit: int) -> SessionPage:
    """Decode one page of a stored NDJSON feed, without parsing the rest of it.

    Lines are counted in one pass over the bytes and only the requested window
    is decoded, so a 200 MB session costs one page of parsed events rather than
    a parsed session. The bytes themselves still arrive whole -- the blob-store
    interface is whole-object (#253) -- so this bounds what is HELD as objects
    and rendered, which is the part this page controls.

    Blank lines are skipped rather than numbered: the capture drops them on the
    way in (:meth:`sidecar.reviewer.transcript.Transcript.write`), so one here
    is framing rather than an event.

    Args:
        data: The stored transcript's bytes, as :func:`sidecar.transcripts.fetch`
            returns them.
        offset: Events to skip, from the start of the feed.
        limit: Events to decode.

    Returns:
        The decoded window and the number of events the whole feed holds.
    """
    lines: list[SessionLine] = []
    total = 0
    for chunk in io.BytesIO(data):
        text = chunk.decode("utf-8", "replace").strip()
        if not text:
            continue
        total += 1
        if not offset < total <= offset + limit:
            continue
        try:
            decoded = json.loads(text)
        except (ValueError, RecursionError):
            # RecursionError rather than only ValueError: `json.loads` hits the
            # interpreter's recursion limit on a deeply nested line, and that is
            # a RuntimeError. A line this page cannot parse is drawn as its raw
            # bytes -- the contract is that the page always draws, and an
            # uncaught RecursionError here would be a 500 instead.
            decoded = None
        lines.append(
            SessionLine(
                number=total,
                event=decoded if isinstance(decoded, dict) else None,
                raw=text,
            )
        )
    return SessionPage(lines=tuple(lines), total=total)


def _clip(text: str) -> tuple[str, bool]:
    """Return ``text`` bounded by :data:`MAX_BLOCK_CHARS`, and whether it was cut."""
    if len(text) <= MAX_BLOCK_CHARS:
        return text, False
    return text[:MAX_BLOCK_CHARS], True


def _fold(label: str, text: str, *, css: str = "muted") -> str:
    """Render one labelled block of stored text, folding it when it is long.

    Short blocks stay open because the point of the page is to READ a session;
    long ones fold so a single vendored file cannot bury the turn after it.

    BOTH arguments go through :func:`_esc`. ``label`` reads like this page's own
    chrome and is not always: a block's ``type`` and an event's ``type`` are
    stored bytes that arrive here as the label, so escaping them without
    sanitizing would leave exactly one seam where a lone surrogate still reaches
    the response encoder.
    """
    body, clipped = _clip(text)
    rendered = _esc(body)
    if clipped:
        rendered += (
            f'\n<span class="bad">[clipped at {MAX_BLOCK_CHARS} characters — '
            "fetch the whole session with `fuko transcripts get`]</span>"
        )
    head = f"<span{c.attrs(class_=css or None)}>{_esc(label)}</span>"
    if len(text) <= _PREVIEW_CHARS and "\n" not in text:
        return f"<p>{head}</p><pre>{rendered}</pre>"
    preview = _esc(text[:_PREVIEW_CHARS].replace("\n", " "))
    ellipsis = "…" if len(text) > _PREVIEW_CHARS else ""
    return c.disclosure(f"{head} {preview}{ellipsis}", rendered)


def _result_text(content: object) -> str:
    """Flatten a ``tool_result``'s content to the text this page shows.

    The CLI spells one result either as a bare string or as a list of typed
    blocks, and both turn up in a single feed -- the same two shapes
    :func:`sidecar.reviewer.transcript._text_bytes` counts. A block that is not
    text is NAMED rather than rendered or dropped: an image is not something
    this page will decode (#241 rules out interpreting stored bytes as anything
    but text), and silently omitting it would make a turn look emptier than it
    was.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, indent=2, default=str)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
        else:
            kind = block.get("type") if isinstance(block, dict) else None
            parts.append(f"[non-text block: {kind if isinstance(kind, str) else 'unknown'}]")
    return "\n".join(parts)


def _block_html(block: object) -> str:
    """Render one content block of an ``assistant`` or ``user`` event."""
    if not isinstance(block, dict):
        return _fold("block", json.dumps(block, indent=2, default=str))
    kind = block.get("type")
    if kind == "text":
        return _fold("assistant", str(block.get("text", "")), css="")
    if kind == "tool_use":
        name = block.get("name")
        label = f"tool call · {name if isinstance(name, str) and name else '?'}"
        # `ensure_ascii` is left at its default on purpose: it renders a
        # lone-surrogate string as its `\ud800` escape rather than as an
        # unencodable character, which is one fewer way for stored bytes to
        # reach the response encoder intact.
        return _fold(label, json.dumps(block.get("input"), indent=2, default=str))
    if kind == "tool_result":
        label = "tool result · error" if block.get("is_error") else "tool result"
        return _fold(
            label,
            _result_text(block.get("content")),
            css="bad" if block.get("is_error") else "muted",
        )
    return _fold(
        f"{kind if isinstance(kind, str) else 'block'}", json.dumps(block, indent=2, default=str)
    )


def _line_html(line: SessionLine) -> str:
    """Render one stored feed line as a titled block of inert text."""
    if line.event is None:
        return (
            f'<h3>{line.number}. <span class="bad">unreadable line</span></h3>'
            + '<p class="muted">Not a JSON event — a feed the capture cut mid-write '
            "ends this way, so the fragment is shown rather than dropped.</p>"
            + _fold("raw", line.raw)
        )
    kind = line.event.get("type")
    title = kind if isinstance(kind, str) and kind else "event"
    message = line.event.get("message")
    blocks = message.get("content") if isinstance(message, dict) else None
    if isinstance(blocks, list):
        body = "".join(_block_html(block) for block in blocks) or '<p class="muted">no content</p>'
    else:
        body = _fold(title, json.dumps(line.event, indent=2, default=str))
    return f"<h3>{line.number}. {_esc(title)}</h3>{body}"


def _human_bytes(count: int) -> str:
    """Render a byte count in binary units, the way ``fuko transcripts`` prints it."""
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"  # pragma: no cover - the loop always returns


def _when(run: corpus.TranscriptRun) -> str:
    """The instant a run is filed under, trimmed to minutes."""
    stamp = run.started_at or run.created_at or ""
    return stamp[:16] or "—"


def _state_html(complete: bool) -> str:
    """Render the completeness badge.

    An incomplete transcript is called out rather than left to be inferred from
    a small figure: a session cut short by the ``tool_timeout`` kill must never
    read as a run that finished cheaply.
    """
    return c.badge("complete", css="ok") if complete else c.badge("INCOMPLETE", css="bad")


def _tools_label(calls: dict[str, int]) -> str:
    """Per-tool call counts, busiest first, capped at :data:`_TOOLS_SHOWN`."""
    if not calls:
        return "no tool calls"
    ordered = sorted(calls.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ordered[:_TOOLS_SHOWN]
    text = " ".join(f"{name}={count}" for name, count in shown)
    if len(ordered) > len(shown):
        text += f" +{len(ordered) - len(shown)} more"
    return text


def _url(**params: object) -> str:
    """Build one of this page's URLs with its query string."""
    return f"{PAGE.path}{c.query_string(params)}"


def _row(run: corpus.TranscriptRun) -> str:
    """Render one listed transcript.

    The repo and PR cells cross-link on ``(repo, pr)`` the way the metrics page
    does, and on nothing else: a run's ``slot`` and a ledger lane's ``seat`` are
    not reliably the same label (see
    :func:`sidecar.web.metrics._ledger_cell`), so the pull request is the key
    both sides agree on.
    """
    repo_cell = (
        c.raw_cell(c.link(f"{_METRICS.path}{c.query_string({'repo': run.repo})}", run.repo))
        if run.repo
        else c.cell("(no run row)", css="muted")
    )
    if run.repo and run.pr is not None:
        pr_cell = c.raw_cell(
            c.link(
                f"{_LEDGER.path}{c.query_string({'repo': run.repo, 'pr': run.pr})}", f"#{run.pr}"
            ),
            numeric=True,
        )
    else:
        pr_cell = c.cell(None, numeric=True)
    return (
        "<tr>"
        + c.cell(_when(run), css="muted")
        + c.raw_cell(c.link(_url(key=run.key), run.key), css="nowrap")
        + repo_cell
        + pr_cell
        + c.cell(run.seat or "—")
        + c.cell(run.model or "—", css="muted")
        + c.raw_cell(_state_html(run.complete), css="nowrap")
        + c.cell(run.tool_calls_total, numeric=True)
        + c.cell(_tools_label(run.tool_calls), css="muted")
        + c.cell(_human_bytes(run.tool_result_bytes), numeric=True)
        + c.cell(run.repeated_read_files, numeric=True)
        + c.cell(run.outcome or "—", css="ok" if run.outcome == "ok" else "muted")
        + "</tr>"
    )


def _index_notices(*, db_enabled: bool, db_error: bool, filter_error: str) -> str:
    """Render the listing's store and filter states, keeping each distinguishable."""
    parts = []
    if not db_enabled:
        parts.append(
            c.notice(
                "Transcripts are indexed in the Postgres store (FUKO_DATABASE_URL unset). "
                "A sqlite-vec deployment captures no transcript index.",
                kind="warn",
            )
        )
    elif db_error:
        parts.append(
            c.notice(
                "Transcript index unreachable — this is a fault, not an empty corpus.",
                kind="danger",
            )
        )
    if filter_error:
        # NOT "filter ignored": the bad bound is rejected before the query is
        # opened, so the listing did not run with the remaining filters -- it
        # did not run at all. Saying "ignored" over an empty table reads as a
        # corpus with nothing in it, which is the same typo-as-outage
        # confusion, one level down.
        parts.append(
            c.notice(
                f"Filter not understood: {filter_error} — the listing was not run, "
                "so the table below is empty for that reason and not because the corpus is.",
                kind="warn",
            )
        )
    return "".join(parts)


def _filter_form(
    repo: str | None, pr: int | None, seat: str | None, since: str | None, until: str | None
) -> str:
    """Render the listing filter, echoing every submitted value back into its field."""
    return (
        '<form method="get">'
        + c.field("repo", "repo", repo or "", placeholder="owner/name")
        + c.field("PR", "pr", "" if pr is None else pr, size=6)
        + c.field("seat", "seat", seat or "", placeholder="any seat")
        + c.field("since", "since", since or "", placeholder="2026-09-01", size=12)
        + c.field("until", "until", until or "", placeholder="exclusive", size=12)
        + "<button>filter</button></form>"
    )


def _open_form() -> str:
    """Render the by-key form.

    Worth its own control rather than only a link from the listing: a blob can
    be stored with no index row behind it (#258), which makes it unlistable and
    still perfectly readable, and an operator holding such a key from a runner's
    stderr has no other way in.
    """
    return (
        '<form method="get">'
        + c.field("open transcript by key", "key", "", placeholder="transcript key", size=40)
        + "<button>open</button></form>"
    )


def render_index(
    *,
    page_rows: corpus.TranscriptPage,
    repo: str | None,
    pr: int | None,
    seat: str | None,
    since: str | None,
    until: str | None,
    offset: int,
    limit: int,
    db_enabled: bool,
    db_error: bool = False,
    filter_error: str = "",
    extra_nav: str = "",
) -> str:
    """Render the transcript listing from already-queried rows (pure, never raises)."""
    body = [
        "<h1>Captured session transcripts</h1>",
        '<p class="muted">One row per stored transcript, newest first. The figures are '
        "what a run spent its turns on; opening one shows the session itself.</p>",
        _filter_form(repo, pr, seat, since, until),
        _open_form(),
        _index_notices(db_enabled=db_enabled, db_error=db_error, filter_error=filter_error),
        c.section(
            "Runs",
            c.table(
                _COLUMNS,
                [_row(run) for run in page_rows.rows],
                "nothing listed — the filter above was not understood"
                if filter_error
                else "no transcripts captured yet",
            ),
        ),
        c.pager(
            PAGE.path,
            {"repo": repo, "pr": pr, "seat": seat, "since": since, "until": until},
            offset=offset,
            limit=limit,
            total=page_rows.total,
        ),
    ]
    return document(
        title="fuko session transcripts",
        body="".join(body),
        active=PAGE.slug,
        extra_nav=extra_nav,
    )


_STORE_NOTICES = {
    "unconfigured": (
        "No transcript blob store configured (FUKO_TRANSCRIPT_STORE_BACKEND unset), "
        "so no session body can be read here — this is the off state, not an empty session.",
        "warn",
    ),
    "unreachable": (
        "Transcript store unreachable — this is a fault, not an empty session.",
        "danger",
    ),
    "invalid": (
        "That is not a well-formed transcript key, so nothing was looked up.",
        "warn",
    ),
    "missing": (
        "The store answered and holds nothing under this key.",
        "warn",
    ),
}


def _session_notices(*, store_state: str, db_enabled: bool, db_error: bool, indexed: bool) -> str:
    """Render the session view's store states: the blob's, then the index's.

    Nothing is said about the index that the reads do not license. A key
    rejected at the boundary was never looked up in either place, and the
    absence of an index row only means "stored but unindexed" (#258) when the
    store actually handed over bytes -- otherwise the page would answer "nothing
    is here" and "this is a real stored session" in the same breath.
    """
    parts = []
    message = _STORE_NOTICES.get(store_state)
    if message:
        parts.append(c.notice(message[0], kind=message[1]))
    if store_state == "invalid":
        return "".join(parts)
    if not db_enabled:
        parts.append(
            c.notice(
                "No Postgres store configured, so this session carries no run attribution "
                "or derived figures.",
                kind="warn",
            )
        )
    elif db_error:
        parts.append(
            c.notice(
                "Transcript index unreachable — the session below is shown without its "
                "run attribution, which is missing rather than absent.",
                kind="danger",
            )
        )
    elif not indexed and store_state == "ok":
        parts.append(
            c.notice(
                "No index row for this key. A transcript can reach shared storage without "
                "being indexed (#258), so this is a real stored session that the listing "
                "cannot see.",
                kind="warn",
            )
        )
    return "".join(parts)


def _facts(run: corpus.TranscriptRun) -> str:
    """Render the indexed run's identity and derived figures as one small table."""
    rows = [
        ("repo", c.esc(run.repo) if run.repo else '<span class="muted">(no run row)</span>'),
        (
            "PR",
            c.link(
                f"https://github.com/{quote(run.repo, safe='/')}/pull/{int(run.pr)}",
                f"#{run.pr}",
            )
            if run.repo and run.pr is not None
            else "—",
        ),
        ("seat", c.esc(run.seat or "—")),
        ("model", c.esc(f"{run.provider or '—'}/{run.model or '—'}")),
        ("backend", c.esc(run.backend or "—")),
        ("started", c.esc(_when(run))),
        (
            "duration",
            c.esc("—" if run.duration_s is None else f"{float(run.duration_s):.1f}s"),
        ),
        ("outcome", c.esc(run.outcome or "—")),
        ("state", _state_html(run.complete)),
        ("tool calls", c.esc(f"{run.tool_calls_total} ({_tools_label(run.tool_calls)})")),
        ("tool-result bytes", c.esc(_human_bytes(run.tool_result_bytes))),
        ("files read more than once", c.esc(run.repeated_read_files)),
    ]
    return (
        "<table>"
        + "".join(f"<tr><th>{c.esc(k)}</th>{c.raw_cell(v)}</tr>" for k, v in rows)
        + "</table>"
    )


def render_session(
    *,
    key: str,
    run: corpus.TranscriptRun | None,
    session: SessionPage,
    offset: int,
    limit: int,
    store_state: str,
    db_enabled: bool,
    db_error: bool = False,
    extra_nav: str = "",
) -> str:
    """Render one stored session from already-fetched data (pure, never raises).

    The chrome -- heading, notices, cross-links, pager -- is drawn from the
    index row and the line COUNT, never from the session body, so a page still
    says what it is looking at when the blob could not be read at all.
    """
    links = [c.link(_url(), "← all transcripts")]
    if run is not None and run.repo:
        links.append(c.link(f"{_METRICS.path}{c.query_string({'repo': run.repo})}", "run metrics"))
        if run.pr is not None:
            links.append(
                c.link(
                    f"{_LEDGER.path}{c.query_string({'repo': run.repo, 'pr': run.pr})}",
                    "review state",
                )
            )
    heading = _esc(key)
    if run is not None:
        heading += f" {_state_html(run.complete)}"
    body = [
        f"<h1>{heading}</h1>",
        f'<p class="pager">{" ".join(links)}</p>',
        _session_notices(
            store_state=store_state,
            db_enabled=db_enabled,
            db_error=db_error,
            indexed=run is not None,
        ),
    ]
    if run is not None:
        body.append(c.section("Run", _facts(run)))
    body.append(
        c.section(
            "Session",
            '<p class="muted">Every line of the stored feed, in order, as inert text. '
            "Nothing here is interpreted as markup: a tool result is the reviewed "
            "repository's own bytes.</p>"
            + (
                "".join(_line_html(line) for line in session.lines)
                if session.lines
                else '<p class="muted">no events to show</p>'
            ),
        )
    )
    body.append(c.pager(PAGE.path, {"key": key}, offset=offset, limit=limit, total=session.total))
    return document(
        title=f"fuko transcript — {key}",
        body="".join(body),
        active=PAGE.slug,
        extra_nav=extra_nav,
    )


def _session_view(request: Request, *, key: str, offset: int, limit: int) -> str:
    """Fetch and render one session, degrading each read independently.

    Gated on a browser session (:func:`sidecar.web.security.require`) while the
    listing is open. See the module docstring: the listing publishes counts, and
    this publishes a contributor-controlled repository's file contents.

    The key is validated first, so "that key is a typo" and "this deployment's
    store cannot be built" stay two answers rather than one.
    """
    security.require(request)
    limit = min(max(1, limit or EVENTS_PER_PAGE), MAX_EVENTS)
    offset = max(0, offset)
    # The key is judged BEFORE either read, the way `GET /transcripts/{key}`
    # judges it: `corpus.fetch` builds the store before it ever looks at the
    # key, and `make_blob_store` raises ValueError for every
    # configured-but-broken deployment (no ROOT, no BUCKET, an unknown
    # backend). Catching ValueError off the fetch instead would report those as
    # "your key is malformed" for EVERY key -- the fault-as-typo confusion this
    # page exists to prevent, running backwards.
    try:
        validate_blob_key(key)
    except ValueError:
        return render_session(
            key=key,
            run=None,
            session=SessionPage(),
            offset=offset,
            limit=limit,
            store_state="invalid",
            db_enabled=bool(settings.database_url),
            db_error=False,
            extra_nav=security.nav_extra(request),
        )
    db_enabled = bool(settings.database_url)
    run = None
    db_error = False
    if db_enabled:
        try:
            run = corpus.describe(key)
        except Exception as e:
            corpus.log_read_failure(f"index lookup for {key}", e)
            db_error = True
    session = SessionPage()
    store_state = "ok"
    data = None
    try:
        data = corpus.fetch(key)
    except corpus.StoreUnconfigured:
        store_state = "unconfigured"
    except Exception as e:
        corpus.log_read_failure(f"fetch of {key}", e)
        store_state = "unreachable"
    if data is None and store_state == "ok":
        store_state = "missing"
    if data is not None:
        session = read_session(data, offset=offset, limit=limit)
    return render_session(
        key=key,
        run=run,
        session=session,
        offset=offset,
        limit=limit,
        store_state=store_state,
        db_enabled=db_enabled,
        db_error=db_error,
        extra_nav=security.nav_extra(request),
    )


@router.get(PAGE.path, response_class=HTMLResponse)
def view(
    request: Request,
    response: Response,
    repo: str | None = None,
    pr: str | None = None,
    seat: str | None = None,
    since: str | None = None,
    until: str | None = None,
    key: str | None = None,
    offset: int = 0,
    limit: int = 0,
) -> str:
    """Serve the transcript listing, or one session when ``key`` names it.

    Fetches and degrades, per the package's route/render split. The
    configuration test is made HERE rather than inside the read, so an
    unconfigured deployment is never reported as a broken one.

    ``pr`` arrives as text and goes through
    :func:`sidecar.web.components.form_int` because it is bound to a form field:
    the filter submits an untouched PR box as ``pr=``, which an ``int | None``
    parameter answers with a 422 rather than treating as absent.
    """
    if key:
        # The session view is the only page here that renders stored repository
        # content, and a 200 carrying no freshness information is heuristically
        # cacheable by a shared cache (RFC 9111 §4.2.2). A LAN forward proxy in
        # front of the sidecar could therefore hold this response and later hand
        # it to a request with no session -- serving the very bytes
        # `security.require` stands in front of. The listing is left cacheable:
        # it publishes index-row figures and is open by design.
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Cookie"
        return _session_view(request, key=key, offset=offset, limit=limit)
    number = c.form_int(pr)
    limit = min(max(1, limit or PAGE_SIZE), corpus.MAX_ROWS)
    offset = max(0, offset)
    db_enabled = bool(settings.database_url)
    page_rows = corpus.TranscriptPage()
    db_error = False
    filter_error = ""
    if db_enabled:
        try:
            page_rows = corpus.list_transcripts(
                repo=repo or None,
                pr=number,
                seat=seat or None,
                since=since or None,
                until=until or None,
                limit=limit,
                offset=offset,
            )
        except ValueError as e:
            # An unparseable date bound is the operator's typo. Ordered ahead of
            # the blanket arm on purpose: reporting it as "store unreachable"
            # is the exact confusion #240 exists to prevent.
            filter_error = str(e)
        except Exception as e:
            corpus.log_read_failure("listing", e)
            db_error = True
    return render_index(
        page_rows=page_rows,
        repo=repo,
        pr=number,
        seat=seat,
        since=since,
        until=until,
        offset=offset,
        limit=limit,
        db_enabled=db_enabled,
        db_error=db_error,
        filter_error=filter_error,
        extra_nav=security.nav_extra(request),
    )
