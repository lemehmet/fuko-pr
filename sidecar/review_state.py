"""Per-PR, per-seat review state: the open-findings and coverage ledgers.

The storage half of the stateful-review epic (#155, epic #160). A review round
records what it FOUND (``review_findings``) and what it EXAMINED
(``review_coverage``) for one ``(repo, pr, seat)``, and the next round reads
both back so it can settle what its predecessor left open and spend its budget
on surface nobody has looked at yet. Measured on mepro, 86% of findings are
one-shot -- reported once and never re-noticed -- which is the loss this ledger
exists to stop.

Two properties shape every function here:

* **Best-effort, always.** Mirroring :mod:`sidecar.circuit_breaker` and
  :mod:`sidecar.run_metrics`, with no Postgres configured
  (``FUKO_DATABASE_URL`` unset) every function is a no-op. Unlike those two,
  the failure path is guarded here as well (see :func:`_best_effort`): a seat
  whose store is unreachable must review exactly the way it does today, which
  is a correct and complete review. That property is what makes the epic safe
  to roll out incrementally, so it is enforced in the module that owns the
  columns rather than left to each call site to remember.
* **Primitives, not policy.** This module reads, writes and expires rows. Which
  verdict closes a row, what a missing reason means, when coverage expires
  wholesale after a force-push -- that is the wiring issue's (#156, #157)
  decision, and encoding it here would put the review's semantics in two
  places.

Scope, stated rather than left as a silent gap: **Postgres only**. The
sqlite-vec store (:mod:`sidecar.sqlite_store`) mirrors the KNOWLEDGE base, and
this is operational state with a pull request's lifetime, not knowledge -- it
carries no embedding, is never retrieved semantically, and dies with the PR. A
sqlite-vec deployment therefore reviews statelessly, which is the same
degradation an unreachable Postgres produces and is already a supported mode.

Retention is a READ WINDOW (:data:`RETENTION_DAYS`), not a sweep: rows outside
it are invisible to every read here, so a months-dead PR can never be resurrected
into a prompt. Physically reclaiming them belongs with a PR-closed hook, which
no code path has yet; it is deliberately not a sweep function nothing calls.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps

from .config import settings
from .reviewer.prompt import AgenticFinding, ExaminedRegion, PriorCoverage, PriorFinding

FINDING_STATUSES = frozenset({"open", "fixed", "rejected", "stale"})
"""The states a ledger row may hold, matching the migration's CHECK constraint.

``open`` is the initial state; ``fixed`` and ``rejected`` are a later round's
settled verdicts; ``stale`` is fuko's own retirement of a finding whose file is
gone. The agent's ``still_open`` is deliberately absent -- re-asserting a finding
is not a state change, it refreshes ``updated_at`` on a row that stays ``open``.
"""

RETENTION_DAYS = 90
"""How far back a read may reach, in days.

Generous enough that no live pull request falls out of its own ledger (mepro's
longest-running branches span weeks), tight enough that the reads stay bounded
as the tables accumulate closed PRs.

Measured from the column that records ACTIVITY, which differs per ledger: a
finding from ``updated_at``, since a later round can re-assert it and that
re-assertion is what keeps it live (:func:`open_findings`); a coverage row from
``created_at``, since coverage is never touched after it is written -- it is
expired, and expired rows are already excluded.
"""

MAX_OPEN_FINDINGS = 200
"""Hard bound on one read of the open ledger.

Every row returned here is rendered into the next prompt in full
(:func:`sidecar.reviewer.prompt.render_prior_state` caps coverage but never
findings, on purpose), so an unbounded read is an unbounded prompt. A PR whose
seat has 200 unsettled findings has a problem no ledger will fix.
"""

MAX_LIVE_COVERAGE = 500
"""Hard bound on one read of the live coverage ledger.

Higher than the finding bound because the renderer caps coverage itself
(``MAX_PRIOR_COVERAGE``); this only stops the SQL read from growing without
limit over a long-lived branch.
"""

MAX_TEXT = 4000
"""Per-column cap on the free text a round may store, in characters.

Not a database limit -- ``TEXT`` has none -- but a prompt-budget one: unlike a
review comment, a stored finding is replayed into EVERY later round on the
branch, so one runaway body is a recurring cost rather than a one-off. Truncation
loses the tail of one field; the alternative is a ledger that grows the prompt
without bound.
"""


def _enabled() -> bool:
    """Review-state persistence requires the shared Postgres store."""
    return bool(settings.database_url)


def _best_effort(default_factory: Callable[[], object]):
    """Make the wrapped function no-op instead of raising or blocking a review.

    Two gates in one place, because both mean the same thing to a caller: no
    store configured, and a store that failed. State must never fail a review --
    a seat with an unreachable ledger reviews the way it did before the ledger
    existed -- so the exception is swallowed to stderr and the neutral value is
    returned rather than propagating into the review path.

    The default is a FACTORY so a caller can never be handed a shared mutable
    list to append to.
    """

    def _decorate(fn):
        @wraps(fn)
        def _guarded(*args, **kwargs):
            if not _enabled():
                return default_factory()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"fuko: review-state {fn.__name__} failed: {e}", file=sys.stderr)
                return default_factory()

        return _guarded

    return _decorate


def _clip(text: str) -> str:
    """Bound one stored free-text field at :data:`MAX_TEXT`."""
    return (text or "")[:MAX_TEXT]


def _is_uuid(value: str) -> bool:
    """Whether ``value`` parses as a UUID, so a bad id costs no round-trip."""
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _not_a_bare_string(values: Sequence[str], param: str) -> None:
    """Reject a single string where a sequence of them is meant.

    ``str`` satisfies ``Sequence[str]``, and no type checker objects, so
    ``expire_coverage(..., "src/app.py")`` iterates CHARACTERS: the argument
    survives every other guard and the call expires nothing while reporting the
    same ``0`` as "there was no coverage for that file". That is the fail-unsafe
    direction -- a stale assurance kept -- and it is the same shape as the
    ``None``-vs-``[]`` mistake this module already warns about.

    Raising is not a break with "state never fails a review": :func:`_best_effort`
    turns it into the usual no-op, so the caller's review continues and the
    operator gets a line on stderr instead of silence.
    """
    if isinstance(values, str):
        raise TypeError(f"{param} must be a sequence of strings, not a single str: {values!r}")


@dataclass(frozen=True)
class StoredFinding:
    """One open ledger row: its database id, and the finding a round will see.

    The two halves are kept separate because :class:`PriorFinding` deliberately
    carries no id -- the id a round is shown is minted at render time by
    :func:`sidecar.reviewer.prompt.render_prior_state` and is never read out of
    stored model text. This wrapper is how a caller maps a minted ``pN`` back to
    the row to transition, without depending on the renderer's enumeration
    order.
    """

    id: str
    prior: PriorFinding


@_best_effort(lambda: 0)
def record_findings(
    repo: str,
    pr: int,
    seat: str,
    round: int,
    head_sha: str,
    findings: Sequence[AgenticFinding],
) -> int:
    """Insert this round's findings as ``open`` rows; return how many landed.

    ``end_line`` is not stored: the ledger anchors a claim, and the anchor a
    later round re-verifies against is the start line. Returns 0 when
    persistence is disabled or the write fails.
    """
    if not findings:
        return 0
    from .db import db

    with db() as conn:
        for finding in findings:
            conn.execute(
                "INSERT INTO review_findings "
                "(repo, pr, seat, round, head_sha, file, line, severity, category, "
                "title, body, evidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    repo,
                    pr,
                    seat,
                    round,
                    head_sha,
                    _clip(finding.file),
                    finding.line,
                    _clip(finding.severity),
                    _clip(finding.category),
                    _clip(finding.title),
                    _clip(finding.body),
                    _clip(finding.evidence),
                ),
            )
    return len(findings)


@_best_effort(list)
def open_findings(
    repo: str, pr: int, seat: str, limit: int = MAX_OPEN_FINDINGS
) -> list[StoredFinding]:
    """Return this seat's still-open findings, oldest round first (empty when disabled).

    Oldest first so the minted ``p1..pN`` ids stay stable as new rounds append:
    a finding keeps the id it had last round unless something ahead of it closed,
    which makes a prompt diff between two rounds readable by a human.

    Two details make that ordering and that window mean what they say:

    * the retention window is measured from ``updated_at``, not ``created_at``.
      A round that re-asserts a finding refreshes ``updated_at``
      (:func:`touch_findings`), so an unsettled finding a seat is still looking
      at stays inside the window however long the branch lives; keyed on
      ``created_at`` it would age out of its own ledger while still open, which
      is exactly the silent loss this table exists to stop.
    * ``id`` breaks the tie. One round's findings are inserted in a single
      transaction, so ``now()`` -- and therefore ``created_at`` -- is identical
      for every row of that round, and equal sort keys have no stable order in
      Postgres. Ordering on the primary key makes same-round siblings arbitrary
      but FIXED relative to each other, which is what the stability above needs;
      without it their ``pN`` ids could permute between two reads that settled
      nothing.
    """
    from .db import db

    with db() as conn:
        rows = conn.execute(
            "SELECT id, file, line, severity, category, title, body, round "
            "FROM review_findings "
            "WHERE repo = %s AND pr = %s AND seat = %s AND status = 'open' "
            "AND updated_at > now() - make_interval(days => %s) "
            "ORDER BY round, created_at, id LIMIT %s",
            (repo, pr, seat, RETENTION_DAYS, min(max(1, limit), MAX_OPEN_FINDINGS)),
        ).fetchall()
    return [
        StoredFinding(
            id=str(row_id),
            prior=PriorFinding(
                file=file,
                title=title,
                body=body,
                line=line,
                severity=severity,
                category=category,
                round=round_,
            ),
        )
        for row_id, file, line, severity, category, title, body, round_ in rows
    ]


@_best_effort(lambda: 1)
def next_round(repo: str, pr: int, seat: str) -> int:
    """Return the round number this seat's next round should record under.

    ``max(round) + 1`` over every row this seat has ever written for the pull
    request, whatever state those rows are in now. Counting SETTLED rows too is
    the point: a round whose findings were all fixed still happened, and
    re-issuing its number would put two different rounds behind one label in a
    prompt a human is expected to audit.

    Returns 1 when persistence is disabled or the read fails, which is also the
    value for a seat's first round -- both mean "nothing recorded before this".

    That fallback is honest about losing a LABEL, not a row. Disabled
    persistence writes nothing, so the number never lands anywhere. A transient
    read failure is different: the caller reads the round before the review and
    writes rows after it, so a store that recovers in between records this
    round's findings under ``1`` while rounds ``1..N`` already exist. Nothing is
    lost -- the rows are stored, carried and settled like any other -- but they
    sort into the round-1 block of :func:`open_findings`, so that read's promise
    that a finding keeps its minted id "unless something ahead of it closed"
    does not survive this brownout. The alternative, refusing to record a round
    whose label is uncertain, trades a wrong label for a lost finding, which is
    the direction this table exists to avoid.
    """
    from .db import db

    with db() as conn:
        row = conn.execute(
            "SELECT coalesce(max(round), 0) + 1 FROM review_findings "
            "WHERE repo = %s AND pr = %s AND seat = %s",
            (repo, pr, seat),
        ).fetchall()
    return int(row[0][0]) if row and row[0] and row[0][0] is not None else 1


@_best_effort(lambda: False)
def transition(finding_id: str, status: str, reason: str = "") -> bool:
    """Move one ``open`` finding to ``status``; return whether a row changed.

    Guarded three ways, all in the fail-safe direction -- a finding that does not
    transition stays open, which is noise, where one that transitions wrongly is
    the silent loss this ledger exists to prevent:

    * a ``status`` outside :data:`FINDING_STATUSES` changes nothing, so an
      unrecognised verdict cannot close a row;
    * a ``finding_id`` that is not a UUID changes nothing and costs no
      round-trip;
    * the UPDATE matches ``status = 'open'`` only, so an already-settled row
      cannot be re-settled by a later round replaying a stale id.

    Whether a given verdict is ALLOWED to close a row (a ``rejected`` with no
    reason must not, per #156) is the caller's policy, not this function's.
    """
    if status not in FINDING_STATUSES or not _is_uuid(finding_id):
        return False
    from .db import db

    with db() as conn:
        cur = conn.execute(
            "UPDATE review_findings SET status = %s, status_reason = %s, updated_at = now() "
            "WHERE id = %s AND status = 'open'",
            (status, _clip(reason), finding_id),
        )
        return bool(cur.rowcount)


@_best_effort(lambda: 0)
def touch_findings(finding_ids: Sequence[str]) -> int:
    """Refresh ``updated_at`` on findings a round re-asserted; return the count.

    The ``still_open`` verdict's only effect: the row keeps its state, but the
    ledger records that a round looked at it against the current head, so a
    later reader can tell a re-verified claim from one nobody has revisited.
    """
    _not_a_bare_string(finding_ids, "finding_ids")
    ids = [i for i in finding_ids if _is_uuid(i)]
    if not ids:
        return 0
    from .db import db

    with db() as conn:
        cur = conn.execute(
            "UPDATE review_findings SET updated_at = now() WHERE id = ANY(%s) AND status = 'open'",
            (ids,),
        )
        return cur.rowcount or 0


@_best_effort(lambda: 0)
def record_coverage(
    repo: str,
    pr: int,
    seat: str,
    round: int,
    head_sha: str,
    regions: Sequence[ExaminedRegion],
) -> int:
    """Insert this round's examined regions as live coverage; return how many landed.

    Rows are written as read: the prompt contract already forbids a clean bill of
    health, and re-judging a conclusion here would make the ledger disagree with
    what the round actually said.
    """
    if not regions:
        return 0
    from .db import db

    with db() as conn:
        for region in regions:
            conn.execute(
                "INSERT INTO review_coverage "
                "(repo, pr, seat, round, head_sha, file, region, checked, conclusion, evidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    repo,
                    pr,
                    seat,
                    round,
                    head_sha,
                    _clip(region.file),
                    _clip(region.region),
                    _clip(region.checked),
                    _clip(region.conclusion),
                    _clip(region.evidence),
                ),
            )
    return len(regions)


@_best_effort(list)
def live_coverage(
    repo: str, pr: int, seat: str, limit: int = MAX_LIVE_COVERAGE
) -> list[PriorCoverage]:
    """Return this seat's unexpired coverage, newest round first (empty when disabled).

    Newest first so a read that hits ``limit`` keeps the entries the renderer
    would have kept anyway; the renderer sorts again, since ordering the prompt
    is its decision and not the store's.

    ``id`` breaks the tie for the same reason it does in :func:`open_findings`,
    and it matters here even though coverage carries no minted ids: one round's
    rows share a transaction timestamp, and BOTH caps that can fall inside a
    round -- this ``LIMIT`` and the renderer's ``max_coverage`` -- count entries.
    Without a total tiebreaker, which same-round siblings survive a cut could
    change between two reads that changed nothing, so entries would drift in and
    out of the prompt on their own. The renderer's own sort does not repair it:
    it sorts by round with a stable sort, inheriting whatever order arrives here.
    """
    from .db import db

    with db() as conn:
        rows = conn.execute(
            "SELECT file, checked, conclusion, evidence, region, round "
            "FROM review_coverage "
            "WHERE repo = %s AND pr = %s AND seat = %s AND expired_at IS NULL "
            "AND created_at > now() - make_interval(days => %s) "
            "ORDER BY round DESC, created_at DESC, id DESC LIMIT %s",
            (repo, pr, seat, RETENTION_DAYS, min(max(1, limit), MAX_LIVE_COVERAGE)),
        ).fetchall()
    return [
        PriorCoverage(
            file=file,
            checked=checked,
            conclusion=conclusion,
            evidence=evidence,
            region=region,
            round=round_,
        )
        for file, checked, conclusion, evidence, region, round_ in rows
    ]


@_best_effort(lambda: 0)
def expire_coverage(repo: str, pr: int, seat: str, files: Sequence[str] | None = None) -> int:
    """Expire live coverage for ``files`` (all of it when ``files`` is ``None``).

    The delta between the last reviewed head and the current one is what makes a
    coverage claim obsolete: the tree it described changed, so the assurance dies
    even though a finding on the same file survives. ``None`` is the wholesale
    case a rebase or force-push needs -- the tree the whole ledger described is
    gone.

    An empty sequence expires NOTHING and is not the same as ``None``: a round
    whose delta touched no file must not silently discard the ledger.

    Each branch carries its OWN complete statement rather than concatenating a
    fragment onto a shared one: the placeholders and the parameters that fill
    them are then written together and can only drift together. It also keeps
    every string reaching ``execute`` a literal, which is what a reader (and a
    static analyser) has to prove about a query built in Python.

    The paths are put through :func:`_clip` on the way in because
    :func:`record_coverage` clipped them on the way out. ``file`` is a MATCHING
    KEY, not free text, so the two sides have to agree: a path over
    :data:`MAX_TEXT` was stored truncated, and matching it raw would find
    nothing and report the same ``0`` as "there was no coverage for that file" --
    the stale assurance kept, which is the one direction this ledger must not
    fail in.
    """
    if files is not None:
        _not_a_bare_string(files, "files")
        if not files:
            return 0
    from .db import db

    if files is None:
        sql = (
            "UPDATE review_coverage SET expired_at = now() "
            "WHERE repo = %s AND pr = %s AND seat = %s AND expired_at IS NULL"
        )
        params: tuple = (repo, pr, seat)
    else:
        sql = (
            "UPDATE review_coverage SET expired_at = now() "
            "WHERE repo = %s AND pr = %s AND seat = %s AND expired_at IS NULL "
            "AND file = ANY(%s)"
        )
        params = (repo, pr, seat, [_clip(f) for f in files])

    with db() as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount or 0
