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
    """
    from .db import db

    with db() as conn:
        rows = conn.execute(
            "SELECT id, file, line, severity, category, title, body, round "
            "FROM review_findings "
            "WHERE repo = %s AND pr = %s AND seat = %s AND status = 'open' "
            "AND created_at > now() - make_interval(days => %s) "
            "ORDER BY round, created_at LIMIT %s",
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
    """
    from .db import db

    with db() as conn:
        rows = conn.execute(
            "SELECT file, checked, conclusion, evidence, region, round "
            "FROM review_coverage "
            "WHERE repo = %s AND pr = %s AND seat = %s AND expired_at IS NULL "
            "AND created_at > now() - make_interval(days => %s) "
            "ORDER BY round DESC, created_at DESC LIMIT %s",
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
    """
    if files is not None and not files:
        return 0
    from .db import db

    where = "repo = %s AND pr = %s AND seat = %s AND expired_at IS NULL"
    params: list = [repo, pr, seat]
    if files is not None:
        where += " AND file = ANY(%s)"
        params.append(list(files))

    with db() as conn:
        cur = conn.execute(f"UPDATE review_coverage SET expired_at = now() WHERE {where}", params)
        return cur.rowcount or 0
