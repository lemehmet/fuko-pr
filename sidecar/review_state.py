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
it are invisible to every PROMPT-PATH read, so a months-dead PR can never be
resurrected into a prompt. Physically reclaiming them belongs with a PR-closed
hook, which no code path has yet; it is deliberately not a sweep function
nothing calls. The operator reads at the end of this module (#235) deliberately
do NOT apply that window -- a row nobody can see is the complaint they exist to
answer -- and they carry their own contract, spelled out there.
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

REOPENABLE_STATUSES = frozenset({"fixed", "rejected"})
"""The closures a later round may reverse (:func:`reopen`), and only those.

Exactly the two a MODEL VERDICT produces. #177's subject is that such a verdict
closes a row terminally while the text that drove it can originate in the
reviewed checkout, so a closure made from model output must be answerable by a
later round that independently re-finds the claim.

``stale`` is excluded because it is not a verdict: it is fuko's own retirement
of a finding whose file the head no longer carries, made on evidence fuko read
itself. Whether that path should soften -- annotate the row rather than close it
-- is a separate question the epic tracks (#177 direction 3, interacting with
#175), and answering it here by making the retirement quietly reversible would
pre-empt it.
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
seat has 200 unsettled findings has a problem no ledger will fix -- but past
this bound the ledger at least NAMES that problem rather than absorbing it:
the read reports what it cut (:class:`OpenLedger`), because a row that is never
offered to a round is a row no round can settle (#173).
"""

MAX_SETTLED_FINDINGS = 200
"""Hard bound on one read of the settled (reopenable) ledger.

Nothing here reaches a prompt -- the settled read exists so
:func:`sidecar.reviewer.ledger.settle` can tell "a claim this seat already
closed" from "a claim nobody has seen", so its cost is one query and a dict, not
prompt budget. It is bounded anyway, and at the same number as the open read,
because a seat's settled rows only ever grow over a long-lived branch.

Overflow degrades to the pre-#177 behaviour and no further: a re-found claim
whose closed row fell outside the window is simply recorded as a new open row,
so it is still carried into the next round. That is loss of the LINK back to the
closure, not of the claim -- which is why the cut needs no reporting path of its
own the way the open read's does (#173).
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


@dataclass(frozen=True)
class SettledFinding:
    """One closed ledger row, as much of it as a re-raise decision needs.

    Not a :class:`StoredFinding`: nothing here is rendered into a prompt, so the
    row is projected down to what :func:`sidecar.reviewer.ledger.settle` matches
    on (``file``, ``title``) and what it writes into the reopen's own reason
    (``status``, ``round``, ``reason``). Keeping the body and the evidence out of
    the projection is deliberate -- a reopened row keeps the text it already has,
    and a reader of this dataclass should not be able to mistake it for the
    channel that would refresh it.
    """

    id: str
    file: str
    title: str
    status: str
    round: int = 0
    reason: str = ""


@dataclass(frozen=True)
class OpenLedger:
    """One read of the open ledger: the rows it returns, and the rows it cut.

    ``truncated`` is how many open rows the window held beyond
    :data:`MAX_OPEN_FINDINGS` -- rows that exist, are ``open``, and were NOT
    handed to the caller. They matter because of what the cap does to the ledger
    downstream: a row never offered to a round is never minted a ``pN`` id, so
    no verdict can address it and it can only leave ``open`` by ageing out of the
    retention window months later (#173). That is the silent loss this table
    exists to stop, arriving through the cap instead of through a round.

    Reporting the cut rather than raising on it is the fail-safe direction here.
    :func:`_best_effort` turns an exception into the neutral value, so a raise
    would hand the round an EMPTY ledger: partial truncation would become total
    loss, and the 200 rows that were readable would go unsettled too. Deciding
    what to DO about the cut -- warn, alarm, page -- stays with the policy half
    (:func:`sidecar.reviewer.ledger.carry_in`), per this module's charter.
    """

    rows: tuple[StoredFinding, ...] = ()
    truncated: int = 0


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
    from .db import db_best_effort

    with db_best_effort() as conn:
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


@_best_effort(OpenLedger)
def open_findings(repo: str, pr: int, seat: str, limit: int = MAX_OPEN_FINDINGS) -> OpenLedger:
    """Return this seat's still-open findings, oldest round first (empty when disabled).

    Oldest first so the minted ``p1..pN`` ids stay stable as new rounds append:
    a finding keeps the id it had last round unless something ahead of it closed,
    which makes a prompt diff between two rounds readable by a human.

    That ordering decides WHICH rows ``limit`` cuts, and it cuts the newest --
    the ones a round most needs to settle. So the count of cut rows travels back
    with them (:class:`OpenLedger`, ``truncated``), from the same read that did
    the cutting rather than from a second count that could disagree with it:
    ``count(*) OVER ()`` is evaluated after the ``WHERE`` and before the
    ``LIMIT``, so it is the size of the window this read is a prefix of.

    Two details make that ordering and that window mean what they say:

    * the retention window is measured from ``updated_at``, not ``created_at``.
      A round that re-asserts a finding refreshes ``updated_at``
      (:func:`touch_findings`), so an unsettled finding a seat is still looking
      at stays inside the window however long the branch lives; keyed on
      ``created_at`` it would age out of its own ledger while still open, which
      is exactly the silent loss this table exists to stop.
    * ``evidence`` is projected alongside the claim (#174). It was the one part
      of what a round SAID that :func:`record_findings` wrote and this read
      dropped, which handed the next round a finding LESS grounded than the one
      its predecessor published -- the citation stripped from exactly the claim
      the round is asked to re-verify. (``head_sha`` is written and still not
      projected, deliberately: it records which head a claim was published
      against, which is provenance about the round rather than grounding for the
      claim, and the renderer shows the reader nothing of it.) What evidence
      costs in prompt budget is bounded by the renderer, per row
      (:data:`sidecar.reviewer.prompt.MAX_PRIOR_EVIDENCE`), not here: the store
      keeps what the round actually said.
    * ``id`` breaks the tie. One round's findings are inserted in a single
      transaction, so ``now()`` -- and therefore ``created_at`` -- is identical
      for every row of that round, and equal sort keys have no stable order in
      Postgres. Ordering on the primary key makes same-round siblings arbitrary
      but FIXED relative to each other, which is what the stability above needs;
      without it their ``pN`` ids could permute between two reads that settled
      nothing.
    """
    from .db import db_best_effort

    with db_best_effort() as conn:
        rows = conn.execute(
            "SELECT id, file, line, severity, category, title, body, evidence, round, "
            "count(*) OVER () "
            "FROM review_findings "
            "WHERE repo = %s AND pr = %s AND seat = %s AND status = 'open' "
            "AND updated_at > now() - make_interval(days => %s) "
            "ORDER BY round, created_at, id LIMIT %s",
            (repo, pr, seat, RETENTION_DAYS, min(max(1, limit), MAX_OPEN_FINDINGS)),
        ).fetchall()
    stored = tuple(
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
                evidence=evidence,
            ),
        )
        for row_id, file, line, severity, category, title, body, evidence, round_, _total in rows
    )
    # No rows means no window column to read, and the count can only be clamped
    # upward from zero: a total BELOW the rows in hand would be a contradiction,
    # and reporting a negative truncation would be worse than reporting none.
    total = int(rows[0][9]) if rows else 0
    return OpenLedger(rows=stored, truncated=max(0, total - len(stored)))


@_best_effort(lambda: 1)
def next_round(repo: str, pr: int, seat: str) -> int:
    """Return the round number this seat's next round should record under.

    ``max(round) + 1`` over every row this seat has ever written for the pull
    request, whatever state those rows are in now. Counting SETTLED rows too is
    the point: a round whose findings were all fixed still happened, and
    re-issuing its number would put two different rounds behind one label in a
    prompt a human is expected to audit.

    Both ledgers are counted, which is not a refinement of that rule but the only
    way to keep it once a round can write coverage without writing a finding
    (#157). A round that found nothing and recorded what it examined is exactly
    that shape, and it is a common one -- reading ``review_findings`` alone would
    hand every such round the number ``1`` for as long as the streak lasted, and
    the prompt would then present N rounds of coverage under one label, each
    claiming to be the oldest (CodeRabbit, #157).

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
    from .db import db_best_effort

    with db_best_effort() as conn:
        row = conn.execute(
            "SELECT coalesce(max(round), 0) + 1 FROM ("
            "SELECT round FROM review_findings WHERE repo = %s AND pr = %s AND seat = %s "
            "UNION ALL "
            "SELECT round FROM review_coverage WHERE repo = %s AND pr = %s AND seat = %s"
            ") AS rounds",
            (repo, pr, seat, repo, pr, seat),
        ).fetchall()
    return int(row[0][0]) if row and row[0] and row[0][0] is not None else 1


@_best_effort(lambda: False)
def transition(
    repo: str, pr: int, seat: str, finding_id: str, status: str, reason: str = ""
) -> bool:
    """Move one of this seat's ``open`` findings to ``status``; return whether a row changed.

    Guarded four ways, all in the fail-safe direction -- a finding that does not
    transition stays open, which is noise, where one that transitions wrongly is
    the silent loss this ledger exists to prevent:

    * a ``status`` outside :data:`FINDING_STATUSES` changes nothing, so an
      unrecognised verdict cannot close a row;
    * a ``finding_id`` that is not a UUID changes nothing and costs no
      round-trip;
    * the UPDATE matches ``status = 'open'`` only, so an already-settled row
      cannot be re-settled by a later round replaying a stale id;
    * the UPDATE matches ``(repo, pr, seat)`` as well as the id, so the row has
      to be one of THIS seat's.

    That last guard is why the lane travels with the id rather than the id
    travelling alone. In-process it is redundant -- every id a caller holds came
    out of :func:`open_findings` for the same ``(repo, pr, seat)`` -- but the
    ledger now reaches the store over HTTP as well (#171), and there the
    provenance of an id is a claim in a request body rather than a fact about
    the call stack. One seat closing another's finding is precisely the
    cross-seat coupling the per-seat ledger exists to prevent (#160), so the
    constraint is stated in SQL, where it holds for both transports and for any
    later one.

    Whether a given verdict is ALLOWED to close a row (a ``rejected`` with no
    reason must not, per #156) is the caller's policy, not this function's.
    """
    if status not in FINDING_STATUSES or not _is_uuid(finding_id):
        return False
    from .db import db_best_effort

    with db_best_effort() as conn:
        cur = conn.execute(
            "UPDATE review_findings SET status = %s, status_reason = %s, updated_at = now() "
            "WHERE id = %s AND status = 'open' AND repo = %s AND pr = %s AND seat = %s",
            (status, _clip(reason), finding_id, repo, pr, seat),
        )
        return bool(cur.rowcount)


@_best_effort(tuple)
def settled_findings(
    repo: str, pr: int, seat: str, limit: int = MAX_SETTLED_FINDINGS
) -> tuple[SettledFinding, ...]:
    """Return this seat's model-closed findings, most recently closed first.

    The read a re-raise needs and nothing more (#177): rows a verdict closed
    (:data:`REOPENABLE_STATUSES`), inside the retention window, for exactly this
    ``(repo, pr, seat)``. The scope is load-bearing rather than incidental --
    a wider read would let one seat reopen a row another seat closed, which is
    the cross-seat coupling the per-seat ledger exists to avoid (#160), and it
    would do so through a path no verdict of that seat's own could reverse.

    ``updated_at`` is the closure time (:func:`transition` sets it), so ordering
    on it descending means a ``limit`` cut keeps the most recently settled rows
    -- the ones a round on the current head is most likely to contradict -- and
    the retention window is measured against the same column the open read uses.

    Returns an empty tuple when persistence is disabled or the read fails, which
    reads as "no closure to re-raise": the caller then records the re-found claim
    as a new open row, which is what it did before this read existed.
    """
    from .db import db_best_effort

    with db_best_effort() as conn:
        rows = conn.execute(
            "SELECT id, file, title, status, round, status_reason "
            "FROM review_findings "
            "WHERE repo = %s AND pr = %s AND seat = %s AND status = ANY(%s) "
            "AND updated_at > now() - make_interval(days => %s) "
            "ORDER BY updated_at DESC, id DESC LIMIT %s",
            (
                repo,
                pr,
                seat,
                sorted(REOPENABLE_STATUSES),
                RETENTION_DAYS,
                min(max(1, limit), MAX_SETTLED_FINDINGS),
            ),
        ).fetchall()
    return tuple(
        SettledFinding(
            id=str(row_id),
            file=file,
            title=title,
            status=status,
            round=round_,
            reason=reason,
        )
        for row_id, file, title, status, round_, reason in rows
    )


@_best_effort(lambda: False)
def reopen(repo: str, pr: int, seat: str, finding_id: str, reason: str) -> bool:
    """Return one of this seat's model-closed findings to ``open``; return whether a row changed.

    The inverse of :func:`transition`, and deliberately the narrower of the two:

    * only :data:`REOPENABLE_STATUSES` are matched, so fuko's own ``stale``
      retirement is not reversed here and an already-``open`` row is not
      disturbed (its ``reopened`` count would otherwise inflate on every round
      that re-reported it);
    * a ``finding_id`` that is not a UUID changes nothing and costs no
      round-trip, as in :func:`transition`;
    * the UPDATE matches ``(repo, pr, seat)`` as well as the id, for the reason
      :func:`transition` states: over the HTTP seam (#171) an id's lane is a
      claim in a request body, and re-raising another seat's row is the same
      cross-seat coupling as closing one;
    * ``reopened`` is incremented rather than set, so the row carries how many
      times a round settled it and a later round contradicted that -- the audit
      trail a terminal closure could not leave (#177).

    ``reason`` is composed by the caller and is expected to carry the closure it
    reverses, because this UPDATE overwrites ``status_reason``: the column holds
    one explanation, and after a reopen the explanation a reader needs is the
    whole sequence, not just its last step.

    The direction is the module's usual one. A reopen that does not happen
    leaves a claim recorded as a fresh row (noise); a closure that could not be
    answered is the silent loss the ledger exists to stop.
    """
    if not _is_uuid(finding_id):
        return False
    from .db import db_best_effort

    with db_best_effort() as conn:
        cur = conn.execute(
            "UPDATE review_findings SET status = 'open', status_reason = %s, "
            "reopened = reopened + 1, updated_at = now() "
            "WHERE id = %s AND status = ANY(%s) AND repo = %s AND pr = %s AND seat = %s",
            (_clip(reason), finding_id, sorted(REOPENABLE_STATUSES), repo, pr, seat),
        )
        return bool(cur.rowcount)


@_best_effort(lambda: 0)
def touch_findings(repo: str, pr: int, seat: str, finding_ids: Sequence[str]) -> int:
    """Refresh ``updated_at`` on this seat's re-asserted findings; return the count.

    The ``still_open`` verdict's only effect: the row keeps its state, but the
    ledger records that a round looked at it against the current head, so a
    later reader can tell a re-verified claim from one nobody has revisited.

    Scoped to ``(repo, pr, seat)`` like :func:`transition` and :func:`reopen`.
    A touch is the mildest of the three -- it changes no state, only a timestamp
    -- but the timestamp is what the retention window and the open read's
    ordering are measured from, so a cross-lane touch would still let one seat
    reach into another's ledger.
    """
    _not_a_bare_string(finding_ids, "finding_ids")
    ids = [i for i in finding_ids if _is_uuid(i)]
    if not ids:
        return 0
    from .db import db_best_effort

    with db_best_effort() as conn:
        cur = conn.execute(
            "UPDATE review_findings SET updated_at = now() "
            "WHERE id = ANY(%s) AND status = 'open' AND repo = %s AND pr = %s AND seat = %s",
            (ids, repo, pr, seat),
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
    from .db import db_best_effort

    with db_best_effort() as conn:
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
    from .db import db_best_effort

    with db_best_effort() as conn:
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
    from .db import db_best_effort

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

    with db_best_effort() as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# Operator reads (#235).
#
# Everything above answers "what does THIS seat carry into its next round" and
# is shaped for a prompt: keyed by the full (repo, pr, seat), capped for budget,
# windowed by retention, and wrapped in `_best_effort` so a dead store degrades
# into a stateless-but-correct review. The reads below answer a different
# question -- "what is in the ledgers at all" -- for the operator page under
# `/ui`, and they invert three of those choices deliberately:
#
# * NOT `_best_effort`. Its charter is that state must never fail a review, so
#   it swallows the exception and returns the neutral empty value. A UI read
#   needs the opposite: the route has to tell "store unreachable" from "the
#   reviewer found nothing", and an empty table during an outage is the
#   fail-unsafe direction for a human reading it. These raise; the route
#   catches (see `sidecar.web.ledger.view`).
# * NO retention window on which rows are shown. A row outside the window is
#   invisible to every prompt-path read, which is exactly the row an operator
#   has no other way to see. The window still decides `offerable`, because that
#   number is a claim about what `open_findings` would hand a round.
# * The existing prompt reads are left alone. Their caps and ordering are
#   load-bearing for prompt budget and for #173's truncation contract, so the
#   UI gets its own reads rather than a widened `limit` on theirs.
# ---------------------------------------------------------------------------

STATUS_ORDER: tuple[str, ...] = ("open", "fixed", "rejected", "stale")
"""Display order for :data:`FINDING_STATUSES`: lifecycle, not alphabetical.

A tuple rather than the frozenset because a page renders one column per status
and a set has no order -- iterating it would shuffle the columns between
processes. The two are kept in sync by a test, not by construction, so the
order stays a deliberate choice rather than a sort's accident.
"""

MAX_LANES = 200
"""Cap on lanes one index read returns, before paging."""

MAX_LEDGER_ROWS = 200
"""Cap on findings or coverage rows one detail read returns, before paging."""


def _iso(value: object) -> str | None:
    """Render a timestamp column as text, keeping ``None`` distinguishable.

    The page renders already-queried plain data, so the conversion happens here
    rather than in a template. ``None`` survives as ``None`` because for
    ``review_coverage.expired_at`` it is not a missing value but the live state.
    """
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _rate(numerator: int, denominator: int) -> float | None:
    """Return ``numerator / denominator``, or ``None`` when nothing is comparable.

    ``None`` rather than ``0.0`` for a zero denominator: a lane with a single
    round has no finding any later round could have acted on, and rendering that
    as a 0% settle rate would read as a seat that settles nothing.
    """
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class LaneStat:
    """One ``(repo, pr, seat)`` lane as the operator index shows it.

    ``counts`` is keyed by status and always carries every key in
    :data:`STATUS_ORDER`, so a page can render a fixed set of columns without
    testing for absence.

    ``offerable`` and ``never_offered`` split the open count the way #173's
    contract does: ``offerable`` is how many open rows are inside the retention
    window and would therefore be read back, and ``never_offered`` is how many
    of those exceed :data:`MAX_OPEN_FINDINGS` -- rows that exist, are open, and
    that no round is ever handed, so no verdict can address them and they can
    only age out unseen.

    ``eligible``/``carried``/``settled`` are the arithmetic behind the two rates
    #159 needs, on one shared denominator: findings this seat recorded in a
    round STRICTLY BEFORE its latest one, which are the findings at least one
    later round could have acted on. Of those, ``carried`` are still open (the
    ledger re-showed them and nothing closed them) and ``settled`` are the ones
    a verdict closed. ``stale`` rows count in neither -- fuko retired them
    itself, no round decided anything -- so the two rates do not sum to one.
    """

    repo: str
    pr: int
    seat: str
    latest_round: int
    last_activity: str | None
    counts: dict[str, int]
    reopened: int
    offerable: int
    never_offered: int
    coverage_total: int
    coverage_live: int
    eligible: int
    carried: int
    settled: int

    @property
    def findings_total(self) -> int:
        """Every finding row in this lane, whatever state it is in."""
        return sum(self.counts.values())

    @property
    def carry_forward_rate(self) -> float | None:
        """Share of this seat's eligible findings a later round still carries open."""
        return _rate(self.carried, self.eligible)

    @property
    def settle_rate(self) -> float | None:
        """Share of this seat's eligible findings a later round's verdict closed."""
        return _rate(self.settled, self.eligible)


@dataclass(frozen=True)
class LaneIndex:
    """One page of :class:`LaneStat` rows, plus how many lanes matched in all."""

    lanes: tuple[LaneStat, ...] = ()
    total: int = 0


@dataclass(frozen=True)
class LedgerFinding:
    """One ``review_findings`` row, whole, for an operator rather than a prompt.

    Unlike :class:`StoredFinding` and :class:`SettledFinding` -- projections
    shaped by what a round or a re-raise needs -- this carries every column,
    including the ``head_sha`` the prompt path deliberately drops: for a human
    it is the provenance that makes a file link resolvable.
    """

    id: str
    seat: str
    round: int
    head_sha: str
    file: str
    line: int | None
    severity: str
    category: str
    title: str
    body: str
    evidence: str
    status: str
    status_reason: str
    reopened: int
    created_at: str | None
    updated_at: str | None

    @property
    def anomalous(self) -> bool:
        """Whether a round declared this settled and a later round contradicted it (#177)."""
        return self.reopened > 0


@dataclass(frozen=True)
class LedgerCoverage:
    """One ``review_coverage`` row, expired ones included.

    A coverage entry records what a round EXAMINED, and it dies when the delta
    touches its file -- so ``expired_at`` is shown rather than filtered on, and
    an expired entry is history about a round, never a standing conclusion.
    """

    id: str
    seat: str
    round: int
    head_sha: str
    file: str
    region: str
    checked: str
    conclusion: str
    evidence: str
    expired_at: str | None
    created_at: str | None

    @property
    def live(self) -> bool:
        """Whether this assurance is still standing (nothing has expired it)."""
        return self.expired_at is None


@dataclass(frozen=True)
class FindingPage:
    """One page of :class:`LedgerFinding` rows, plus how many matched in all."""

    rows: tuple[LedgerFinding, ...] = ()
    total: int = 0


@dataclass(frozen=True)
class CoveragePage:
    """One page of :class:`LedgerCoverage` rows, plus how many matched in all."""

    rows: tuple[LedgerCoverage, ...] = ()
    total: int = 0


_LANE_INDEX_SQL = (
    "WITH activity AS ("
    " SELECT repo, pr, seat, round, updated_at AS at FROM review_findings"
    " UNION ALL"
    " SELECT repo, pr, seat, round, created_at AS at FROM review_coverage"
    "), lanes AS ("
    " SELECT repo, pr, seat, max(round) AS latest_round, max(at) AS last_activity"
    " FROM activity"
    " WHERE (%s::text IS NULL OR repo = %s::text)"
    " AND (%s::integer IS NULL OR pr = %s::integer)"
    " AND (%s::text IS NULL OR seat = %s::text)"
    " GROUP BY repo, pr, seat"
    "), tallies AS ("
    " SELECT l.repo, l.pr, l.seat,"
    " count(f.id) FILTER (WHERE f.status = 'open') AS n_open,"
    " count(f.id) FILTER (WHERE f.status = 'fixed') AS n_fixed,"
    " count(f.id) FILTER (WHERE f.status = 'rejected') AS n_rejected,"
    " count(f.id) FILTER (WHERE f.status = 'stale') AS n_stale,"
    " coalesce(sum(f.reopened), 0) AS n_reopened,"
    " count(f.id) FILTER ("
    " WHERE f.status = 'open' AND f.updated_at > now() - make_interval(days => %s)"
    " ) AS n_offerable,"
    " count(f.id) FILTER (WHERE f.round < l.latest_round) AS n_eligible,"
    " count(f.id) FILTER (WHERE f.round < l.latest_round AND f.status = 'open') AS n_carried,"
    " count(f.id) FILTER ("
    " WHERE f.round < l.latest_round AND f.status IN ('fixed', 'rejected')"
    " ) AS n_settled"
    " FROM lanes l"
    " LEFT JOIN review_findings f"
    " ON f.repo = l.repo AND f.pr = l.pr AND f.seat = l.seat"
    " GROUP BY l.repo, l.pr, l.seat"
    "), examined AS ("
    " SELECT repo, pr, seat, count(*) AS n_total,"
    " count(*) FILTER (WHERE expired_at IS NULL) AS n_live"
    " FROM review_coverage GROUP BY repo, pr, seat"
    ") "
    "SELECT l.repo, l.pr, l.seat, l.latest_round, l.last_activity,"
    " t.n_open, t.n_fixed, t.n_rejected, t.n_stale, t.n_reopened, t.n_offerable,"
    " t.n_eligible, t.n_carried, t.n_settled,"
    " coalesce(e.n_total, 0), coalesce(e.n_live, 0), count(*) OVER () "
    "FROM lanes l"
    " JOIN tallies t ON t.repo = l.repo AND t.pr = l.pr AND t.seat = l.seat"
    " LEFT JOIN examined e ON e.repo = l.repo AND e.pr = l.pr AND e.seat = l.seat "
    "ORDER BY l.last_activity DESC, l.repo, l.pr, l.seat LIMIT %s OFFSET %s"
)


def lanes(
    repo: str | None = None,
    pr: int | None = None,
    seat: str | None = None,
    limit: int = MAX_LANES,
    offset: int = 0,
) -> LaneIndex:
    """Return the ledger lanes, most recently active first, for an operator page.

    The one read that answers "which ``(repo, pr, seat)`` lanes exist" -- every
    read above it is keyed by a lane the caller already knows. Raises rather
    than degrading; see this section's header for why.

    Both tables define a lane. A round that found nothing and recorded only
    what it examined is a real round (:func:`next_round` counts both tables for
    the same reason), so a coverage-only lane appears here with zero findings
    rather than vanishing, and ``latest_round`` is the newest round in EITHER
    ledger -- which is also the round the rates measure "before".

    Ordering is by last activity across both ledgers, so a lane whose pull
    request was closed or merged months ago still appears and simply sinks: this
    read asks GitHub nothing about a PR's state, by design.

    Args:
        repo: Restrict to one ``owner/name``, or ``None`` for every repo.
        pr: Restrict to one pull request number, or ``None`` for all of them.
        seat: Restrict to one seat label, or ``None`` for every seat.
        limit: Lanes per page, clamped to ``[1, MAX_LANES]``.
        offset: Lanes to skip, clamped at zero.

    Returns:
        The page of lanes and the total number of lanes that matched. ``total``
        is ``0`` for an empty page, including an ``offset`` past the end: the
        window count travels with the rows, so no rows means no count to read.
    """
    from .db import db

    with db() as conn:
        rows = conn.execute(
            _LANE_INDEX_SQL,
            (
                repo,
                repo,
                pr,
                pr,
                seat,
                seat,
                RETENTION_DAYS,
                min(max(1, limit), MAX_LANES),
                max(0, offset),
            ),
        ).fetchall()
    stats = tuple(
        LaneStat(
            repo=row[0],
            pr=int(row[1]),
            seat=row[2],
            latest_round=int(row[3] or 0),
            last_activity=_iso(row[4]),
            counts={
                "open": int(row[5]),
                "fixed": int(row[6]),
                "rejected": int(row[7]),
                "stale": int(row[8]),
            },
            reopened=int(row[9]),
            offerable=int(row[10]),
            never_offered=max(0, int(row[10]) - MAX_OPEN_FINDINGS),
            coverage_total=int(row[14]),
            coverage_live=int(row[15]),
            eligible=int(row[11]),
            carried=int(row[12]),
            settled=int(row[13]),
        )
        for row in rows
    )
    return LaneIndex(lanes=stats, total=int(rows[0][16]) if rows else 0)


_PR_FINDINGS_SQL = (
    "SELECT id, seat, round, head_sha, file, line, severity, category, title, body,"
    " evidence, status, status_reason, reopened, created_at, updated_at, count(*) OVER () "
    "FROM review_findings "
    "WHERE repo = %s AND pr = %s AND (%s::text IS NULL OR seat = %s::text) "
    "ORDER BY round DESC, created_at DESC, id DESC LIMIT %s OFFSET %s"
)


def pr_findings(
    repo: str,
    pr: int,
    seat: str | None = None,
    limit: int = MAX_LEDGER_ROWS,
    offset: int = 0,
) -> FindingPage:
    """Return one pull request's findings in EVERY status, newest round first.

    Deliberately unlike :func:`open_findings` and :func:`settled_findings`,
    which each select the statuses their caller can act on: an operator is
    reading the ledger's history, so a ``stale`` retirement and a ``rejected``
    verdict are as much a part of it as an open row. Every seat is included
    unless one is named, since the page's unit is the pull request.

    Args:
        repo: The ``owner/name`` the rows belong to.
        pr: The pull request number.
        seat: Restrict to one seat, or ``None`` for every seat on the PR.
        limit: Rows per page, clamped to ``[1, MAX_LEDGER_ROWS]``.
        offset: Rows to skip, clamped at zero.

    Returns:
        The page of rows and the total that matched (``0`` on an empty page).
    """
    from .db import db

    with db() as conn:
        rows = conn.execute(
            _PR_FINDINGS_SQL,
            (repo, pr, seat, seat, min(max(1, limit), MAX_LEDGER_ROWS), max(0, offset)),
        ).fetchall()
    found = tuple(
        LedgerFinding(
            id=str(row[0]),
            seat=row[1],
            round=int(row[2]),
            head_sha=row[3],
            file=row[4],
            line=row[5],
            severity=row[6],
            category=row[7],
            title=row[8],
            body=row[9],
            evidence=row[10],
            status=row[11],
            status_reason=row[12],
            reopened=int(row[13]),
            created_at=_iso(row[14]),
            updated_at=_iso(row[15]),
        )
        for row in rows
    )
    return FindingPage(rows=found, total=int(rows[0][16]) if rows else 0)


_PR_COVERAGE_SQL = (
    "SELECT id, seat, round, head_sha, file, region, checked, conclusion, evidence,"
    " expired_at, created_at, count(*) OVER () "
    "FROM review_coverage "
    "WHERE repo = %s AND pr = %s AND (%s::text IS NULL OR seat = %s::text) "
    "ORDER BY round DESC, created_at DESC, id DESC LIMIT %s OFFSET %s"
)


def pr_coverage(
    repo: str,
    pr: int,
    seat: str | None = None,
    limit: int = MAX_LEDGER_ROWS,
    offset: int = 0,
) -> CoveragePage:
    """Return one pull request's coverage, EXPIRED entries included, newest round first.

    :func:`live_coverage` filters expired rows out because a round must never be
    handed an assurance the delta already killed. An operator needs the opposite
    view: what a round examined and when that stopped being true is the audit
    trail, so ``expired_at`` is projected instead of filtered on and the page
    marks the difference.

    Args:
        repo: The ``owner/name`` the rows belong to.
        pr: The pull request number.
        seat: Restrict to one seat, or ``None`` for every seat on the PR.
        limit: Rows per page, clamped to ``[1, MAX_LEDGER_ROWS]``.
        offset: Rows to skip, clamped at zero.

    Returns:
        The page of rows and the total that matched (``0`` on an empty page).
    """
    from .db import db

    with db() as conn:
        rows = conn.execute(
            _PR_COVERAGE_SQL,
            (repo, pr, seat, seat, min(max(1, limit), MAX_LEDGER_ROWS), max(0, offset)),
        ).fetchall()
    examined = tuple(
        LedgerCoverage(
            id=str(row[0]),
            seat=row[1],
            round=int(row[2]),
            head_sha=row[3],
            file=row[4],
            region=row[5],
            checked=row[6],
            conclusion=row[7],
            evidence=row[8],
            expired_at=_iso(row[9]),
            created_at=_iso(row[10]),
        )
        for row in rows
    )
    return CoveragePage(rows=examined, total=int(rows[0][11]) if rows else 0)
