"""The open-findings ledger as a review round sees it: carry in, then settle.

Tier 1 of the stateful-review epic (#156, epic #160). :mod:`sidecar.review_state`
owns the rows and deliberately owns no semantics; this module owns the policy
that turns those rows into one round's behaviour, so the two questions the epic
actually argues about live in exactly one place each:

* **what a round is told** -- :func:`carry_in` reads this seat's still-open
  findings, retires the ones whose file the current head no longer has, and
  renders the rest through :func:`sidecar.reviewer.prompt.render_prior_state`;
* **what a round is allowed to conclude** -- :func:`settle` applies the verdicts
  the agent returned, then records what this round found as the next round's
  open ledger.

Every decision here leans the same way, because the two failure directions are
not symmetric. A finding that stays open when it should have closed is *noise*:
the next round is asked about it again and settles it. A finding that closes
when it should have stayed open is the 86% one-shot loss this ledger exists to
stop, arriving by a new door -- silently, and permanently. So an unrecognised
verdict, a missing reason, an unreadable path and an unreachable store all end
in "the row keeps the state it had".

Scope is Tier 1 only: the FINDINGS ledger. The coverage ledger (recording
``examined``, expiring it against the delta) is #157's, and nothing here writes
or reads :func:`sidecar.review_state.record_coverage` -- a half-wired coverage
ledger that records assurances nothing ever expires is worse than none.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .. import review_state
from ..review_state import StoredFinding
from .prompt import (
    AgenticFinding,
    PriorFindingStatus,
    PriorState,
    render_prior_state,
)

DEFAULT_SEAT = "default"
"""Seat label for a SOLO run that has no dedicated App identity.

The runner derives a seat from the branch's ``token_env``
(``FUKO_GITHUB_TOKEN_DORIAN`` -> ``dorian``), which a solo config or a laptop run
does not have. Such a deployment is genuinely ONE seat, so it gets one ledger
rather than none: falling back to a constant keeps the single-seat case working,
while the multi-seat fleets that motivated "every ledger is keyed per seat" all
name their seats.

"Solo" is the load-bearing word, and it is the runner that guarantees it: an A/B
branch is ALWAYS handed an explicit seat, falling back to its configured label
when it declares no ``token_env`` (:func:`sidecar.runner._branch_seat`). Without
that, a compare run of identity-less entries would land every branch on this one
constant and let one model close another's findings by ``fixed`` verdict -- so
this default must never be how a multi-branch run gets its seat.
"""


@dataclass(frozen=True)
class CarriedState:
    """The ledger one round carries in, plus what it needs to settle it.

    ``rows`` is the half :class:`sidecar.reviewer.prompt.PriorState` cannot
    provide: it maps each minted ``pN`` back to the ledger row it was minted
    from. The renderer deliberately shows the agent no database id -- the ids in
    the prompt are fuko's own, so a verdict from the fenced channel can only ever
    address a row this round actually offered -- and this mapping is where that
    indirection is undone, on the fuko side of the fence.
    """

    state: PriorState = field(default_factory=PriorState)
    rows: Mapping[str, str] = field(default_factory=dict)
    round: int = 1

    @property
    def text(self) -> str:
        """The rendered section, or ``""`` when there is nothing to carry."""
        return self.state.text


@dataclass(frozen=True)
class Settlement:
    """What one round's settle pass actually changed, for the log line.

    ``reasserted`` counts every row whose outcome was "stays open, with its
    ``updated_at`` refreshed to say a round looked at it against this head": the
    explicit ``still_open`` verdicts, the ``rejected``-without-a-reason ones this
    module downgrades to them, and the rows a round re-reported as a new finding
    instead of settling. All three are the same outcome, so they are one number.

    ``deduped`` counts published findings that were NOT recorded because they
    merely restated such a row -- see :func:`settle`.
    """

    closed: int = 0
    reasserted: int = 0
    recorded: int = 0
    deduped: int = 0


def _within_checkout(root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` inside ``root``, or ``None`` when it does not stay there.

    ``rel`` is a path out of a stored finding, which means it is model text about
    an untrusted checkout: absolute paths, ``..`` escapes and empty strings all
    have to be assumed. Rejecting them returns ``None`` rather than a path, and
    the one caller reads ``None`` as "cannot judge", never as "missing" -- a
    finding whose anchor this function refuses to interpret keeps its state.
    """
    if not rel or os.path.isabs(rel) or "\x00" in rel:
        return None
    normalized = os.path.normpath(rel)
    if normalized == ".." or normalized.startswith(("../", "..\\")):
        return None
    return root / normalized


def _is_gone(root: Path, real_root: str, rel: str) -> bool:
    """Whether ``rel`` is a path this checkout provably no longer carries.

    Every uncertain answer is ``False``, which keeps the finding open:

    * a checkout root that is not a directory means nothing can be judged
      (the caller checks that once, before any path);
    * a path that leaves the root is unjudgeable rather than absent. Lexical
      containment is not enough for that: ``lstat`` does not follow the FINAL
      component but the kernel still resolves every PARENT, so ``link/gone.py``
      under a ``link`` that points out of the tree is answered by the host
      filesystem, and a host path that happens not to exist would retire a live
      finding. So the parent is resolved through its symlinks first and the
      answer is only trusted while it lands back inside ``real_root``;
    * only ``FileNotFoundError`` retires anything. This is why the check is
      ``lstat`` and not ``os.path.lexists``, which swallows every ``OSError``
      into the same ``False`` -- a name too long for the filesystem, a component
      that is not a directory, a path the runner may not stat would all read as
      "deleted" and close a live finding;
    * ``lstat`` rather than ``stat``: a dangling symlink is still a path the tree
      carries, and a finding about one is not a finding about a deleted file.
      Resolving only the PARENT is what keeps that true -- the last component is
      never followed, so a broken link is still judged live.
    """
    target = _within_checkout(root, rel)
    if target is None:
        return False
    try:
        parent = os.path.realpath(target.parent)
    except (OSError, ValueError):
        return False
    if parent != real_root and not parent.startswith(real_root + os.sep):
        return False
    try:
        os.lstat(target)
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return False
    return False


def _retire_missing(
    stored: Sequence[StoredFinding], checkout_root: str, head_sha: str
) -> list[StoredFinding]:
    """Mark findings whose file is gone ``stale`` and return the ones to carry.

    "The file no longer exists" is the ONE closure this module makes on its own
    authority, and it is safe for the reason line drift is not: the anchor is
    unreachable, so no later round could re-verify the claim even in principle.
    A moved line, a renamed symbol, a rewritten function -- none of those retire
    anything here, because "the code changed" is not evidence the problem was
    fixed, and treating it as such is the most tempting way to re-create the loss
    this ledger exists to stop.

    A checkout root that is missing or is not a directory retires NOTHING: under
    a root that does not exist every path is "missing", which would silently
    close the entire ledger the first time a caller passed a bad path. The root
    is resolved through its own symlinks once, here, so that every per-path
    containment check compares like with like: on a host whose temp or work dir
    is itself a link, a real parent under a lexical root would never match.
    """
    root = Path(checkout_root) if checkout_root else None
    if root is None or not root.is_dir():
        return list(stored)
    try:
        real_root = os.path.realpath(root)
    except (OSError, ValueError):
        return list(stored)
    live: list[StoredFinding] = []
    for row in stored:
        if _is_gone(root, real_root, row.prior.file):
            review_state.transition(row.id, "stale", f"file absent from the tree at {head_sha}")
        else:
            live.append(row)
    return live


def _anchor(file: str, title: str) -> tuple[str, str]:
    """The key two claims are "the same finding" under.

    Deliberately coarse -- file plus case-folded title, not the body -- because
    it is only ever used to decide whether a round RE-REPORTED something it was
    already holding. A near-miss costs a duplicate row (noise); it can never
    close anything, so the usual asymmetry does not apply here.
    """
    return (file.strip(), title.strip().casefold())


def carry_in(
    repo: str, pr: int, seat: str, checkout_root: str = "", head_sha: str = ""
) -> CarriedState:
    """Load this seat's open findings and render the section the round carries.

    The read is per ``(repo, pr, seat)`` and never wider: a shared cross-seat
    ledger would raise fleet coverage at the cost of the independent
    second-opinion property that is the whole reason two seats exist (#160).

    Returns an empty :class:`CarriedState` when there is nothing to carry -- and
    also when the store is unconfigured or unreachable, since every
    :mod:`sidecar.review_state` call degrades to its neutral value. An empty
    state renders no ``prior-review-state`` section at all, so a round with no
    predecessor (or no store) gets byte-for-byte the prompt it got before this
    ledger existed.
    """
    live = _retire_missing(review_state.open_findings(repo, pr, seat), checkout_root, head_sha)
    state = render_prior_state([row.prior for row in live])
    # zip over the renderer's OWN minted keys rather than re-deriving `p{n}`
    # here: the id scheme is the renderer's to choose, and pairing its output
    # with the list it was given cannot drift out of step with it the way a
    # second copy of the enumeration could.
    rows = {minted: row.id for minted, row in zip(state.ids, live)}
    return CarriedState(state=state, rows=rows, round=review_state.next_round(repo, pr, seat))


def settle(
    carried: CarriedState,
    *,
    repo: str,
    pr: int,
    seat: str,
    head_sha: str,
    prior_status: Sequence[PriorFindingStatus] = (),
    findings: Sequence[AgenticFinding] = (),
) -> Settlement:
    """Apply this round's verdicts on carried findings, then record its own.

    The transition policy, and why each branch leans where it does:

    * ``fixed`` closes the row. A reason is asked for unconditionally by the
      strategy but not required here: the round asserts the problem is gone at a
      head a human can check, and the fix commit is the evidence.
    * ``rejected`` closes the row **only with a reason**. Rejection is a seat
      overruling its predecessor, and the reason is the whole audit trail; a
      bare "rejected" would let a round close any inherited finding by
      assertion. Without one the entry is downgraded to ``still_open`` (#156).
    * ``still_open`` refreshes ``updated_at`` and nothing else -- re-asserting a
      finding is not a state change, it is the record that a round re-verified
      it against this head.
    * a finding **nobody mentioned** is untouched and is offered again next
      round. The output contract lets a model omit ``prior_status`` entirely,
      and this is the fail-safe reading of that silence.

    Which verdicts are even eligible is not re-decided here:
    :meth:`sidecar.reviewer.prompt.PriorState.accepted_status` has already
    dropped off-vocabulary statuses and any id this round was not handed, so the
    fenced channel cannot address a row it was never shown.

    A round may also RE-REPORT a finding it was handed instead of settling it --
    the prompt asks it to settle each carried row, and nothing enforces that. A
    plain record would then leave two open rows for one claim, both offered next
    round, with settling one still leaving the other. Duplicates compound per
    round, and the compounding has an end: past :data:`review_state.
    MAX_OPEN_FINDINGS` the read keeps the oldest rows, so the ones cut are the
    newest -- never rendered, never minted an id, never touched, and so ageing
    out of the retention window unseen. That is this module's own loss arriving
    by volume, so a published finding whose ``(file, title)`` matches a carried
    row this round left open is treated as a re-assertion of that row: the row
    is touched (a round did look at it against this head) and the finding is not
    recorded again. Matching stays coarse on purpose -- it only ever suppresses
    a WRITE, never a close, so a near-miss costs one duplicate row.

    ``findings`` must be the findings the round actually PUBLISHED, not
    everything the model returned. The 86% loss this ledger repairs is about
    claims a developer saw and did not act on; carrying a low-confidence finding
    the confidence valve withheld into the next round's prompt would re-publish
    it through the settle path and route around that valve. The cost of
    recording at review time rather than post time is that a review whose post
    fails leaves rows behind -- they are re-offered next round, which is noise,
    and noise is the direction this module accepts.
    """
    touch: list[str] = []
    closed = 0
    settled: set[str] = set()
    for entry in carried.state.accepted_status(prior_status):
        row_id = carried.rows.get(entry.id)
        if row_id is None:
            continue
        reason = entry.reason.strip()
        if entry.status == "fixed" or (entry.status == "rejected" and reason):
            settled.add(entry.id)
            closed += int(review_state.transition(row_id, entry.status, reason))
        else:
            touch.append(row_id)
    # Rows this round leaves open, keyed by claim. A verdict that MEANT to close
    # is excluded even if the write failed: the row is then still open, so the
    # worst case is recording the duplicate anyway -- the noise direction.
    still_open = {
        _anchor(prior.file, prior.title): carried.rows[minted]
        for minted, prior in carried.state.ids.items()
        if minted in carried.rows and minted not in settled
    }
    fresh: list[AgenticFinding] = []
    for finding in findings:
        row_id = still_open.get(_anchor(finding.file, finding.title))
        if row_id is None:
            fresh.append(finding)
        elif row_id not in touch:
            touch.append(row_id)
    review_state.touch_findings(touch)
    recorded = review_state.record_findings(repo, pr, seat, carried.round, head_sha, fresh)
    return Settlement(
        closed=closed,
        reasserted=len(touch),
        recorded=recorded,
        deduped=len(findings) - len(fresh),
    )
