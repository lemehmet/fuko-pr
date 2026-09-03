"""Reading the session-transcript corpus: the index rows, and the blobs (#240).

The first reader over what the rest of epic #236 built. #237 tees a scrubbed
NDJSON feed to disk, #238 stores it as a keyed write-once blob, #239 records one
``review_transcripts`` row per captured transcript; this module is how those are
asked questions, and it is the contract BOTH readers share -- ``fuko
transcripts`` over HTTP today, the ``/ui`` page (#241) importing these functions
directly, the way :mod:`sidecar.web.ledger` imports :mod:`sidecar.review_state`.
That is why the query shapes live here rather than in the CLI: a filter the page
spells differently from the command would be two answers to one question.

WHY NOT IN :mod:`sidecar.run_metrics`, which owns the WRITE side of the same
table. Its reads are ``db_best_effort`` ones that return ``[]`` with no store
configured, which is precisely the failure direction #240 forbids: "nothing
found" and "could not look" must not render identically. These reads RAISE, so
a caller can tell an outage from an empty corpus (the same split #235 had to
make for the ledger page), and the process-wide latch that makes the best-effort
contract right for a review path would make a healthy operator read a no-op for
a minute after one unrelated blip.

The listing is over ``review_transcripts``, so a run that produced no transcript
-- every pr-agent run, everything predating capture -- is absent by
construction rather than shown with empty figures. The converse gap is #258's:
a throttled failover leg ships its blob but never indexes it, so that blob is
fetchable by key and invisible to the listing until #258 lands. :func:`fetch`
deliberately does not consult the index for that reason.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from .objectstore import transcript_store

#: Rows one listing page may return. The same bound
#: :data:`sidecar.review_state.MAX_LEDGER_ROWS` puts on the ledger page, for the
#: same reason: the corpus grows without limit by decision (#236 keeps
#: everything), so no reader may issue an unbounded read.
MAX_ROWS = 200

#: Default page size when a caller asks for none.
DEFAULT_ROWS = 50

#: Connection-acquisition budget for these reads, matching
#: :data:`sidecar.review_state.UI_READ_TIMEOUT_S`. An operator waiting on a
#: listing wants an answer or an error, not psycopg_pool's 30-second default.
READ_TIMEOUT_S = 5.0

#: Largest pull-request number the ``pr`` filter may name. ``review_runs.pr`` is
#: ``INTEGER`` (``migrations/004``) and the filter binds ``%s::integer``, so a
#: larger value fails the cast SERVER-side -- a ``DataError``, not a
#: ``ValueError``, which a caller's "the store is unreachable" arm would then
#: report as an outage for what is a typo. The bound lives HERE, with the query
#: that imposes it, rather than only in the endpoint: this module is the
#: contract #241's page imports, and a reader that learned the filter from the
#: signature would otherwise have to know a rule stated somewhere else.
MAX_PR = 2**31 - 1


class StoreUnconfigured(RuntimeError):
    """Raised by :func:`fetch` when no transcript blob store is configured.

    The off state, not a fault -- but it is still the reason a fetch produced
    nothing, and it must never reach a caller as "that key holds nothing".
    """


@dataclass(frozen=True)
class TranscriptRun:
    """One captured transcript, joined to the review run that produced it.

    The transcript's own figures are always present: the row exists only for a
    transcript that reached shared storage, and every column behind these is
    ``NOT NULL`` (``migrations/013``).

    Everything from ``review_runs`` is optional, because the reference is
    written in a SEPARATE transaction after this row lands. A transcript whose
    run row never followed -- the metrics post lost, or #258's un-indexed
    failover legs once they are indexed -- is still a real stored transcript,
    and appears here with no repo, PR or seat rather than not at all.
    """

    key: str
    created_at: str | None
    complete: bool
    tool_calls: dict[str, int]
    tool_result_bytes: int
    repeated_read_files: int
    repo: str | None = None
    pr: int | None = None
    seat: str | None = None
    provider: str | None = None
    model: str | None = None
    backend: str | None = None
    outcome: str | None = None
    started_at: str | None = None
    duration_s: float | None = None

    @property
    def tool_calls_total(self) -> int:
        """Total tool calls across every tool this run used."""
        return sum(self.tool_calls.values())

    def as_dict(self) -> dict:
        """This row as the plain mapping the HTTP listing carries."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "complete": self.complete,
            "tool_calls": dict(self.tool_calls),
            "tool_result_bytes": self.tool_result_bytes,
            "repeated_read_files": self.repeated_read_files,
            "repo": self.repo,
            "pr": self.pr,
            "seat": self.seat,
            "provider": self.provider,
            "model": self.model,
            "backend": self.backend,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True)
class TranscriptPage:
    """One page of :class:`TranscriptRun` rows, plus how many matched in all."""

    rows: tuple[TranscriptRun, ...] = ()
    total: int = 0


def _instant(value: str | datetime | None) -> datetime | None:
    """Normalize a date filter to an aware UTC instant, or ``None``.

    Accepts what each reader naturally holds: a ``datetime`` from a typed query
    parameter, or the raw text a form field and a CLI flag carry. A bare date
    (``2026-08-30``) is midnight, which is what makes the half-open window in
    :func:`list_transcripts` say what it looks like it says.

    The tz-fill is the point. ``created_at`` is ``TIMESTAMPTZ``, and psycopg
    binds a NAIVE datetime as a ``timestamp`` that Postgres then resolves
    through the SESSION time zone -- so the identical ``--since 2026-08-30``
    would select a different window on a UTC server than on one left at local
    time. Filling UTC here makes the boundary a property of the filter rather
    than of whichever box happened to run the query.

    Raises:
        ValueError: The text is not an ISO-8601 date or timestamp.
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.strip())
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


_LIST_SQL = (
    "SELECT t.key, t.created_at, t.complete, t.tool_calls, t.tool_result_bytes,"
    " t.repeated_read_files, r.repo, r.pr, r.slot, r.provider, r.model, r.backend,"
    " r.outcome, r.started_at, r.duration_s, count(*) OVER () "
    "FROM review_transcripts t "
    # LATERAL rather than a plain join: `run_metrics.record` has no ON CONFLICT
    # on `review_runs`, so a re-delivered metrics post really does write a
    # second run row carrying the same transcript_key -- and a plain join would
    # then list one transcript twice and count it twice in the window total.
    # One transcript is one line here, because the key is this corpus's
    # identity; the newest run row wins.
    "LEFT JOIN LATERAL ("
    " SELECT repo, pr, slot, provider, model, backend, outcome, started_at, duration_s"
    " FROM review_runs WHERE transcript_key = t.key"
    " ORDER BY started_at DESC LIMIT 1"
    ") r ON true "
    "WHERE (%s::text IS NULL OR t.key = %s::text)"
    " AND (%s::text IS NULL OR r.repo = %s::text)"
    " AND (%s::integer IS NULL OR r.pr = %s::integer)"
    " AND (%s::text IS NULL OR r.slot = %s::text)"
    " AND (%s::timestamptz IS NULL OR t.created_at >= %s::timestamptz)"
    " AND (%s::timestamptz IS NULL OR t.created_at < %s::timestamptz) "
    # `key` breaks ties on identical timestamps so paging is a total order; two
    # transcripts minted in the same tick would otherwise be free to swap places
    # between the page that shows them and the page that skips them.
    "ORDER BY t.created_at DESC, t.key DESC LIMIT %s OFFSET %s"
)


def _iso(value) -> str | None:
    """A timestamp column as ISO-8601 text, or ``None`` when the column was NULL."""
    return None if value is None else value.isoformat()


def _calls(value) -> dict[str, int]:
    """The ``tool_calls`` JSONB column as a plain mapping.

    psycopg3 adapts ``jsonb`` to a ``dict`` already, so this is a copy and a
    guard rather than a decode: a row written before the column's contract was
    enforced -- or by anything but this repo -- must cost a figure, never the
    listing it appears in.

    The entries this admits are exactly the ones
    :func:`sidecar.run_metrics._tool_calls` will write, restated because the
    column carries no CHECK and these two guards are its only enforcement. That
    includes excluding ``bool``: ``isinstance(True, int)`` holds, so a stored
    ``{"Read": true}`` would otherwise be read back as a fabricated one call and
    summed into the totals -- and the write side documents the same spelling as
    a shape error rather than a count.
    """
    if not isinstance(value, dict):
        return {}
    return {
        str(name): int(count)
        for name, count in value.items()
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0
    }


def list_transcripts(
    repo: str | None = None,
    pr: int | None = None,
    seat: str | None = None,
    since: str | datetime | None = None,
    until: str | datetime | None = None,
    limit: int = DEFAULT_ROWS,
    offset: int = 0,
    key: str | None = None,
) -> TranscriptPage:
    """List captured transcripts, newest first, with the run that produced each.

    RAISES on an unreachable store rather than returning an empty page; the
    caller is expected to have checked that Postgres is configured at all
    (``settings.database_url``) before calling, so that "never set up" and
    "down right now" stay distinguishable. See the module docstring.

    Every filter NARROWS: they are ``AND``-ed, and each is skipped only when it
    is ``None``. ``repo``/``pr``/``seat`` come from the joined ``review_runs``
    row, so any of them also excludes a transcript that has no run row -- which
    is correct, since an unattributed transcript is not known to match.

    Args:
        repo: Restrict to one ``owner/name``, or ``None`` for every repo.
        pr: Restrict to one pull request number, or ``None`` for all of them.
            Must be in ``[0, MAX_PR]`` -- the range ``review_runs.pr`` can hold,
            checked here so an out-of-range value is this function's own
            ``ValueError`` rather than a cast failure indistinguishable from the
            store being down.
        seat: Restrict to one seat -- ``review_runs.slot``, the lane label a
            model occupies -- or ``None`` for every seat.
        since: Lower bound on the transcript's ``created_at``, INCLUSIVE.
        until: Upper bound on ``created_at``, EXCLUSIVE, so adjacent windows
            tile without double-counting a row on the boundary. A bare date is
            midnight UTC, so ``until="2026-09-01"`` ends the window at the start
            of that day rather than at its end.
        limit: Rows per page, clamped to ``[1, MAX_ROWS]``.
        offset: Rows to skip, clamped at zero.
        key: Restrict to one transcript key, or ``None`` for every key. Unlike
            the other filters this one narrows on ``review_transcripts`` itself,
            so it keeps a transcript that has no run row -- which is what makes
            :func:`describe` able to answer for #258's un-indexed legs.

    Returns:
        The page of rows and the total number of transcripts that matched.
        ``total`` is ``0`` for an empty page, including an ``offset`` past the
        end: the window count travels with the rows.

    Raises:
        ValueError: ``since`` or ``until`` is not an ISO-8601 instant, or ``pr``
            is outside the range ``review_runs.pr`` can hold.
    """
    from .db import db

    lower = _instant(since)
    upper = _instant(until)
    if pr is not None and not 0 <= pr <= MAX_PR:
        raise ValueError(f"pr is outside the range this store can hold: {pr}")
    with db(timeout=READ_TIMEOUT_S, embed_space=False) as conn:
        rows = conn.execute(
            _LIST_SQL,
            (
                key,
                key,
                repo,
                repo,
                pr,
                pr,
                seat,
                seat,
                lower,
                lower,
                upper,
                upper,
                min(max(1, limit), MAX_ROWS),
                max(0, offset),
            ),
        ).fetchall()
    runs = tuple(
        TranscriptRun(
            key=row[0],
            created_at=_iso(row[1]),
            complete=bool(row[2]),
            tool_calls=_calls(row[3]),
            tool_result_bytes=int(row[4]),
            repeated_read_files=int(row[5]),
            repo=row[6],
            pr=None if row[7] is None else int(row[7]),
            seat=row[8],
            provider=row[9],
            model=row[10],
            backend=row[11],
            outcome=row[12],
            started_at=_iso(row[13]),
            duration_s=None if row[14] is None else float(row[14]),
        )
        for row in rows
    )
    return TranscriptPage(rows=runs, total=int(rows[0][15]) if rows else 0)


def describe(key: str) -> TranscriptRun | None:
    """Return the index row for one transcript, or ``None`` when it is not indexed.

    The chrome of a single-session view: which run produced this transcript,
    what it spent, and whether the feed reached its end. It is the same query
    :func:`list_transcripts` runs -- one key predicate rather than a second join
    -- because a detail view that learned a transcript's identity from a
    differently-shaped read would be free to disagree with the listing that
    linked to it.

    ``None`` is not an error and not an empty store: a blob can exist with no
    row behind it (#258's un-indexed failover legs, or a lost metrics post), and
    :func:`fetch` deliberately serves it anyway. A caller that renders the
    session must therefore treat a missing index row as missing CHROME rather
    than as a missing transcript.

    RAISES on an unreachable store, for :func:`list_transcripts`' reason.
    """
    page = list_transcripts(key=key, limit=1)
    return page.rows[0] if page.rows else None


def fetch(key: str) -> bytes | None:
    """Return one transcript's stored bytes, or ``None`` when the key holds nothing.

    The bytes AS STORED -- the scrubbed NDJSON the run streamed -- so a caller
    can pipe them to a pager, a ``grep`` or a ``jq``. Nothing is decoded or
    reformatted on the way out; #236 made fetch-and-grep the answer for content
    search, and a formatter in this path would break it.

    The index is deliberately not consulted. A blob can exist with no row behind
    it (#258), and refusing to serve it because the listing cannot see it would
    turn a known gap in the index into missing data.

    Raises:
        StoreUnconfigured: No blob store is configured, so nothing here could
            hold a transcript. Kept distinct from ``None`` for the reason the
            whole sub-issue turns on -- "could not look" must not read as
            "nothing found".
        ValueError: ``key`` is not a well-formed blob key, OR the configured
            store cannot be built (no ROOT, no BUCKET, an unknown backend).
            The store is constructed before the key is looked at, so a caller
            that needs the two apart must validate the key itself first --
            :func:`sidecar.objectstore.validate_blob_key`, as both readers do.
    """
    store = transcript_store()
    if store is None:
        raise StoreUnconfigured(
            "no transcript store configured (set FUKO_TRANSCRIPT_STORE_BACKEND)"
        )
    return store.get(key)


def log_read_failure(what: str, error: Exception) -> None:
    """Report a degraded transcript read on stderr, for the reader that swallowed it.

    Both readers turn the exception into a status or a notice, which writes
    nothing anywhere an operator will find later; the underlying error is the
    only thing that says WHICH fault it was.
    """
    print(f"fuko: transcript {what} failed: {error}", file=sys.stderr)
