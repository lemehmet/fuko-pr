"""The review ledgers as a round sees them: carry in, then settle.

Tiers 1 and 2 of the stateful-review epic (#156, #157, epic #160).
:mod:`sidecar.review_state` owns the rows and deliberately owns no semantics;
this module owns the policy that turns those rows into one round's behaviour, so
the two questions the epic actually argues about live in exactly one place each:

* **what a round is told** -- :func:`carry_in` reads this seat's still-open
  findings, retires the ones whose file the current head no longer has, expires
  the coverage this round's delta invalidates, and renders what is left through
  :func:`sidecar.reviewer.prompt.render_prior_state`;
* **what a round is allowed to conclude** -- :func:`settle` applies the verdicts
  the agent returned, then records what this round found -- and, when the
  coverage ledger is on, what it examined -- as the next round's state.

Every decision here leans the same way, because the two failure directions are
not symmetric. A finding that stays open when it should have closed is *noise*:
the next round is asked about it again and settles it. A finding that closes
when it should have stayed open is the 86% one-shot loss this ledger exists to
stop, arriving by a new door -- silently, and permanently. So an unrecognised
verdict, a missing reason, an unreadable path and an unreachable store all end
in "the row keeps the state it had", and a verdict's closure is no longer the
last word: a later round that independently publishes the same claim re-raises
the row it closed (#177).

The two ledgers are deliberately NOT symmetric, and the asymmetry is the epic's
central rule rather than an implementation detail. A finding is a CLAIM and
survives: it stays open until a round settles it with a reason. A coverage entry
is an ASSURANCE and expires: the moment the delta touches its file, the tree it
described is gone and the entry dies unread. Nothing here ever carries a clean
verdict forward -- "module X is sound" is the one artifact that would turn a
round-1 mistake into a permanent blind spot -- so coverage records what was
LOOKED AT, is introduced to the round as advisory
(:data:`sidecar.reviewer.prompt.COVERAGE_ADVISORY`), and is dropped outright when
the entry is too hollow to be retraced.

Both ledgers are keyed per ``(repo, pr, seat)`` and never wider. A shared
cross-seat ledger would raise fleet coverage at the cost of the independent
second opinion that is the entire reason for running two seats (#160), and it
would do so most seductively here: the seats overlap on 45 files, so sharing
coverage looks like free coverage and is in fact the manufacture of exactly the
correlation the second seat exists to break.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .. import review_state
from ..review_state import SettledFinding, StoredFinding
from .prompt import (
    EXAMINED_REQUIRED_FIELDS,
    MAX_PRIOR_COVERAGE,
    AgenticFinding,
    ExaminedRegion,
    PriorCoverage,
    PriorFindingStatus,
    PriorState,
    render_prior_state,
)

DEFAULT_SEAT = "default"
"""Seat label for a SOLO run that has no dedicated App identity.

The runner derives a seat from the branch's ``token_env``
(``FUKO_GITHUB_TOKEN_DORIAN`` -> ``dorian``), which a laptop run and many solo
configs do not declare. Such a deployment is genuinely ONE seat, so it gets one
ledger rather than none: falling back to a constant keeps the single-seat case
working, while the multi-seat fleets that motivated "every ledger is keyed per
seat" all name their seats.

Reached only when there is NO name to prefer. A solo run that DOES declare a
``token_env`` keeps its slot -- which is the better key, since the ledger then
survives that config growing into an A/B run -- and only a solo run declaring
none lands here.

"Solo" is the other load-bearing word, and it is the runner that guarantees it:
an A/B branch is ALWAYS handed an explicit seat, and
:func:`sidecar.runner._branch_seats` guarantees those are distinct. Without that,
a compare run whose branches share a name (or declare none) would land every
branch on this one constant and let one model close another's findings by
``fixed`` verdict -- so this default must never be how a multi-branch run gets
its seat.
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

    ``truncated`` is the other half the renderer knows nothing about: how many
    open rows the store held beyond its read cap and therefore did NOT offer this
    round. It is carried rather than dropped because it is the one ledger fact a
    later reader cannot recover -- the cut rows look identical to live ones in
    the table, and nothing in this round's output mentions them (#173).

    ``coverage`` and ``expired`` are the coverage ledger's two counts, kept for
    the caller's one-line round summary: how many entries this round was actually
    SHOWN, and how many the delta invalidated on the way in. They are counts and
    not entries because, unlike a suppressed finding, both are fully recoverable
    from the store afterwards -- ``expired_at`` records exactly which rows died
    and when.

    ``coverage`` counts the RENDERED entries, not the live ones the read
    returned: the renderer caps the block at
    :data:`sidecar.reviewer.prompt.MAX_PRIOR_COVERAGE` and announces the cut only
    in-band inside the prompt, so a seat holding more live entries than that
    would otherwise have its receipt claim a number the round never saw. #157's
    rollout is scored on exactly these receipts, which is what makes the
    difference matter rather than merely being imprecise
    (``qwen-anthropic/qwen3.8-max``, #157).
    """

    state: PriorState = field(default_factory=PriorState)
    rows: Mapping[str, str] = field(default_factory=dict)
    round: int = 1
    truncated: int = 0
    coverage: int = 0
    expired: int = 0

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

    ``deduped`` names the published findings that were NOT recorded because they
    restated such a row -- see :func:`settle`. It is the claims and not a count
    because suppressing a write is the one thing this module does that a reader
    cannot reconstruct from the store afterwards: the row that survived carries
    the EARLIER body, so a silent count would leave "which claim did this round
    decide it already had?" unanswerable. The caller logs them.

    ``reopened`` names the published findings that re-raised a row an earlier
    round (or this one) had CLOSED by verdict -- #177's reversal. Named for the
    same reason ``deduped`` is, and for one more: every entry is a round
    contradicting a settled verdict, which is the anomaly signal the terminal
    closure could never produce. A closure that keeps being contradicted is
    either a seat settling claims it has not verified or a verdict that was never
    the seat's own idea, and both want an operator's eyes rather than a counter.

    ``coverage`` is how many examined regions this round recorded, and is zero
    on every seat whose entry has not turned the coverage ledger on.
    """

    closed: int = 0
    reasserted: int = 0
    recorded: int = 0
    deduped: tuple[str, ...] = ()
    reopened: tuple[str, ...] = ()
    coverage: int = 0


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


def _tree_root(checkout_root: str) -> tuple[Path, str] | None:
    """The checkout root and its resolved form, or ``None`` when it cannot be judged.

    Both ledgers ask the tree the same question -- is this path still there? --
    and both must answer "keep it" when the tree itself is unusable: under a root
    that does not exist every path is "missing", which would close the whole
    findings ledger and expire the whole coverage ledger on one bad argument. The
    resolution happens once, here, so every per-path containment check compares
    like with like on a host whose temp or work dir is itself a link.
    """
    root = Path(checkout_root) if checkout_root else None
    if root is None or not root.is_dir():
        return None
    try:
        return root, os.path.realpath(root)
    except (OSError, ValueError):
        return None


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
    resolved = _tree_root(checkout_root)
    if resolved is None:
        return list(stored)
    root, real_root = resolved
    live: list[StoredFinding] = []
    for row in stored:
        if _is_gone(root, real_root, row.prior.file):
            review_state.transition(row.id, "stale", f"file absent from the tree at {head_sha}")
        else:
            live.append(row)
    return live


def _expiry_targets(files: Sequence[str]) -> Sequence[str]:
    """The file list to expire coverage for: de-duplicated and ordered.

    Sorted so the store's parameter -- and therefore the query plan, the test and
    any log of it -- is a function of the delta rather than of a set's iteration
    order, and de-duplicated because the delta arrives as a set of paths and a
    repeated path expires nothing twice.

    A bare ``str`` is passed through UNTOUCHED rather than normalised, which is
    the one case worth stating: ``sorted(set("src/app.py"))`` is a list of
    CHARACTERS, so normalising here would defeat
    :func:`sidecar.review_state._not_a_bare_string` -- the guard whose whole
    purpose is that such a call must not quietly expire nothing while reporting
    the same ``0`` as "there was no coverage for that file". Handing the string
    on lets that guard fire, and :func:`sidecar.review_state._best_effort` turns
    it into a stderr line rather than a failed review.
    """
    if isinstance(files, str):
        return files
    return sorted(set(files))


def _keyed(examined: Sequence[ExaminedRegion]) -> Sequence[ExaminedRegion]:
    """This round's examined regions with the one NON-free-text field normalised.

    Every other field on an :class:`sidecar.reviewer.prompt.ExaminedRegion` is
    read by a model and nothing else, so what a round wrote is what should be
    stored. ``file`` is the exception and the docstring on
    :func:`sidecar.review_state.expire_coverage` says why: it is a MATCHING KEY.
    The delta arrives already stripped from the diff parser, the store's
    ``_clip`` truncates and does not strip, and the comparison is exact -- so a
    path the model padded with a space is recorded, rendered into later prompts,
    and matched by no delta that ever touches that file again. That is the
    stale-assurance direction this tier must not fail in, reached through the key
    rather than through the documented revert gap (CodeRabbit and
    ``qwen-anthropic/qwen3.8-max``, #157).

    Stripping is the whole normalisation on purpose. It is the same treatment
    :func:`_anchor` already gives the findings ledger's matching key, and it
    cannot change which file an entry names. Anything richer -- resolving
    ``./``, symlinks, case -- would be guessing at a path the round chose, and
    guessing wrong would expire the coverage of a DIFFERENT file, which is the
    one error direction stripping cannot produce.

    A blank ``file`` is deliberately NOT dropped here: the table records what the
    round said, and the judgement is applied on the way back out, where
    :data:`sidecar.reviewer.prompt.EXAMINED_REQUIRED_FIELDS` now includes
    ``file`` and :func:`_carried_coverage` drops and logs it.
    """
    return [
        entry.model_copy(update={"file": entry.file.strip()})
        if entry.file != entry.file.strip()
        else entry
        for entry in examined
    ]


def _carried_coverage(
    repo: str, pr: int, seat: str, checkout_root: str = ""
) -> tuple[list[PriorCoverage], int, int]:
    """This seat's live coverage minus the entries too hollow to carry, and the count dropped.

    The filter is the first of #157's four mitigations, and it is applied on the
    way OUT of the store rather than on the way in: the table records what a
    round actually said, and what must never reach a later prompt is a coverage
    row that cannot be retraced. The schema alone cannot enforce this -- ``""``
    satisfies a required ``str``, so an entry can pass
    :class:`sidecar.reviewer.prompt.ExaminedRegion` while recording nothing --
    and the difference matters most for exactly the shape the epic prohibits: an
    entry with a conclusion, no ``checked`` and no ``evidence`` is a bare clean
    bill of health, unfalsifiable and permanent, rendered as though a round had
    established it.

    Every one of :data:`sidecar.reviewer.prompt.EXAMINED_REQUIRED_FIELDS` is
    required to be non-blank, the same set #166 fails a whole round for omitting.
    ``file`` is in that set for a second reason on top of retraceability: it is
    the key expiry matches the delta against, so an entry with a blank one is an
    assurance no future change can ever retire. Dropping such an entry costs one
    advisory line in one prompt;
    carrying it spends budget on a claim no later round can check, which is the
    one direction this tier must not fail in.

    The tree answers the second question, and it is the one the DELTA cannot
    (`qwen-anthropic/qwen3.8-max` and `openrouter/upstage/solar-pro4`, #157). A
    deleted file emits ``+++ /dev/null`` and a renamed-away one emits nothing at
    the old path, so neither reaches ``parse_diff``'s file set -- the two shapes
    that differ MAXIMALLY from base are exactly the two the delta never names,
    and their coverage would otherwise survive every round to retention. So the
    entries are checked against the checkout with the same ``_is_gone`` the
    findings ledger uses, and one whose file the head no longer carries is
    expired in the store rather than merely skipped: the row is dead for every
    future round too, and leaving it live would re-pay this check each time.

    That pass is on the READ path, so unlike delta-expiry it does not run for a
    seat with the ledger off. It does not need to: a flag-off seat carries
    nothing, so the entry cannot reach a prompt while the flag is off, and the
    first round with it back on retires the row before rendering anything. What
    delta-expiry has to survive -- a toggle window in which entries silently
    stop being invalidated -- has no equivalent here.
    """
    rows = review_state.live_coverage(repo, pr, seat)
    kept = [
        row
        for row in rows
        if all(str(getattr(row, name, "")).strip() for name in EXAMINED_REQUIRED_FIELDS)
    ]
    hollow = len(rows) - len(kept)
    resolved = _tree_root(checkout_root)
    if resolved is None:
        return kept, hollow, 0
    root, real_root = resolved
    gone = sorted({row.file for row in kept if _is_gone(root, real_root, row.file)})
    if not gone:
        return kept, hollow, 0
    retired = review_state.expire_coverage(repo, pr, seat, gone)
    dead = set(gone)
    # The rows leave this round's prompt whatever the store said: an entry about
    # a file the head does not carry must not be rendered even if the write that
    # was meant to retire it was the call that degraded.
    return [row for row in kept if row.file not in dead], hollow, retired


def _anchor(file: str, title: str) -> tuple[str, str]:
    """The key two claims are "the same finding" under.

    Deliberately coarse -- file plus case-folded title, not the body -- because
    it is only ever used to decide whether a round RE-REPORTED something it was
    already holding. A near-miss costs a duplicate row (noise); it can never
    close anything, so the usual asymmetry does not apply here.
    """
    return (file.strip(), title.strip().casefold())


def _reopen_reason(row: SettledFinding, round_: int) -> str:
    """The ``status_reason`` a re-raise leaves behind, carrying the closure it reverses.

    :func:`sidecar.review_state.reopen` overwrites the column, so the sequence --
    settled how, on what stated reason, contradicted when -- has to be composed
    into the one string or it is lost. The old reason is model text and is
    included as such; it is stored, never rendered into a prompt, and the store
    clips it.

    The round this names for the row is the round that RECORDED it, and it says
    so, because that is the only round the table keeps:
    :func:`sidecar.review_state.transition` writes status, reason and
    ``updated_at`` and no round, so the round whose verdict closed the row is
    persisted nowhere. Reading ``row.round`` as the closure round would be wrong
    in every case rather than some -- a verdict can only close a row carried in
    from an earlier round, so the closing round is always strictly later than the
    one the row was recorded in (qwen3.8-max on #189).

    The old reason comes LAST, after every fixed part, because
    :func:`sidecar.review_state.reopen` clips what it stores at
    :data:`sidecar.review_state.MAX_TEXT` and that reason is model text already
    held at up to the same cap. Composed with the prose in the middle, a closure
    reason near the cap pushed the provenance suffix past the clip and this line
    silently lost the very thing it was composed to add. Ordered this way the
    clip can only ever cost the tail of the old prose (qwen3.8-max on #189).
    """
    line = (
        f"re-raised in round {round_}: an independent finding contradicts "
        f"{row.status}, on a finding recorded in round {row.round}"
    )
    if row.reason.strip():
        line = f"{line}; the closure stated: {row.reason.strip()}"
    return line


def _reopen_candidates(repo: str, pr: int, seat: str) -> dict[tuple[str, str], SettledFinding]:
    """This seat's closed rows keyed by claim, most recent closure winning.

    The read is deliberately taken AFTER this round's verdicts are applied, so
    the rows this round just closed are candidates too: a round that reports
    ``fixed`` on a carried finding and then publishes the same claim has
    contradicted itself, and the fail-safe reading of a contradiction is that the
    finding is open. That is also the shape an injected verdict has -- text in
    the checkout asserting everything is fixed, while the round's own reading of
    the code still finds the problem -- so resolving it toward open is what makes
    the reversal worth having rather than merely tidy.
    """
    candidates: dict[tuple[str, str], SettledFinding] = {}
    for row in review_state.settled_findings(repo, pr, seat):
        candidates.setdefault(_anchor(row.file, row.title), row)
    return candidates


def carry_in(
    repo: str,
    pr: int,
    seat: str,
    checkout_root: str = "",
    head_sha: str = "",
    *,
    touched_files: Sequence[str] = (),
    coverage_ledger: bool = False,
) -> CarriedState:
    """Load this seat's state and render the section the round carries.

    The read is per ``(repo, pr, seat)`` and never wider: a shared cross-seat
    ledger would raise fleet coverage at the cost of the independent
    second-opinion property that is the whole reason two seats exist (#160).

    ``touched_files`` is this round's delta, and it is used for ONE thing --
    expiring the coverage it invalidates. This is the single place in the epic
    where the delta is consulted at all, and the direction is deliberate: it
    INVALIDATES state, it never scopes the review. A round still reads the whole
    change; what it stops being told is that somebody already looked at a file
    the change has since moved under.

    Expiry runs BEFORE the coverage read (an entry the delta killed must not be
    rendered by the same call that killed it) and runs whether or not
    ``coverage_ledger`` is on. Gating it would make a flag that only ever ADDS
    behaviour able to preserve a stale assurance: a seat that ran the ledger,
    was switched off for some rounds and was switched back on would carry
    entries describing heads that no round in between had a chance to expire.
    Expiry can only ever remove an assurance, so it is safe to run unconditionally
    and unsafe to skip.

    ``touched_files`` is the CURRENT diff (base -> head), not the delta since the
    head each entry was recorded at, which over-expires and is the safe error:

    * a file whose content differs from the base is in the diff every round, so
      its coverage never survives one -- including on a re-round of the same head
      that changed nothing. That costs re-examination of a file the round has to
      read anyway, which is the cheapest thing the ledger can lose;
    * what survives is coverage of files the head does NOT change -- the callers,
      callees and invariants a round reads to VERIFY the diff. That is the
      surface the epic measured being re-read (one 428KB file read 182 times
      across 24 runs) and the surface this tier exists to spread across rounds.

    The delta does not name every changed file, which is why it is not the only
    invalidation: a DELETED file emits ``+++ /dev/null`` and a renamed-away one
    emits nothing at its old path, so neither reaches ``parse_diff``'s file set
    even though both differ maximally from base. Those are retired against the
    checkout instead, in :func:`_carried_coverage`
    (``qwen-anthropic/qwen3.8-max`` and ``openrouter/upstage/solar-pro4``).

    The residual gap runs the other way and is bounded rather than closed: a file
    modified, examined, then reverted to its base content leaves the diff and is
    still present in the tree, so neither pass expires its entry even though the
    head it described is gone. What keeps that survivable is the third mitigation
    rather than this one -- the entry is rendered as advisory
    (:data:`sidecar.reviewer.prompt.COVERAGE_ADVISORY`), so at worst it
    deprioritises a region a round remains free to re-open.

    Returns an empty :class:`CarriedState` when there is nothing to carry -- and
    also when the store is unconfigured or unreachable, since every
    :mod:`sidecar.review_state` call degrades to its neutral value. An empty
    state renders no ``prior-review-state`` section at all, so a round with no
    predecessor (or no store) gets byte-for-byte the prompt it got before this
    ledger existed.

    A read the store had to cut is announced here and nowhere else (#173). Two
    choices in that, both deliberate:

    * it is reported at CARRY time, not folded into the settle summary the
      caller prints when the round completes. The cut is a fact about the read,
      and a round that crashes, times out or is throttled never reaches settle --
      exactly the rounds whose seat is most likely to be the one drowning in
      unsettled rows;
    * the line leads with the TOTAL, because the count is the finding. Past this
      cap a seat is not merely losing its newest rows, it is holding hundreds of
      open claims on one pull request, which says its rounds are settling
      nothing. Every value in the line is fuko's own (seat, repo, pr, integers) --
      no model text reaches it, so it needs none of the flattening the settle
      log applies to a claim's title.

    That total is the size of the window the READ saw, and the line says so:
    :func:`_retire_missing` runs between the read and the print, so on a head
    that deleted files the table already holds fewer open rows by the time the
    line appears. Reporting the read's window is the accurate choice rather than
    the tidy one -- the cap acted on that window, and the rows it cut were never
    checked against the checkout at all, so netting off only the retirements
    among the rows in hand would produce a number describing neither state.
    """
    expired = review_state.expire_coverage(repo, pr, seat, _expiry_targets(touched_files))
    opened = review_state.open_findings(repo, pr, seat)
    live = _retire_missing(opened.rows, checkout_root, head_sha)
    coverage, hollow, retired = (
        _carried_coverage(repo, pr, seat, checkout_root) if coverage_ledger else ([], 0, 0)
    )
    expired += retired
    state = render_prior_state([row.prior for row in live], coverage)
    # zip over the renderer's OWN minted keys rather than re-deriving `p{n}`
    # here: the id scheme is the renderer's to choose, and pairing its output
    # with the list it was given cannot drift out of step with it the way a
    # second copy of the enumeration could.
    rows = {minted: row.id for minted, row in zip(state.ids, live)}
    round_ = review_state.next_round(repo, pr, seat)
    if opened.truncated:
        print(
            f"fuko: review-state seat {seat} round {round_}: {repo}#{pr} held "
            f"{len(opened.rows) + opened.truncated} open findings for this seat at read time, "
            f"over the {review_state.MAX_OPEN_FINDINGS}-row read cap -- "
            f"the {opened.truncated} NEWEST "
            "are not in this round's prompt and cannot be settled until earlier rows close",
            file=sys.stderr,
        )
    if hollow:
        # An exception report, not a per-round line: every entry here was written
        # by this seat's own earlier round, so a seat that keeps producing them is
        # spending budget recording coverage that can never be carried -- a
        # failure that is otherwise perfectly silent, since the round it damages
        # is a LATER one and the store looks healthy from either end.
        print(
            f"fuko: review-state seat {seat} round {round_}: dropped {hollow} coverage "
            f"entr{'y' if hollow == 1 else 'ies'} with a blank "
            f"{'/'.join(EXAMINED_REQUIRED_FIELDS)} -- an entry nothing can retrace "
            "is the unfalsifiable record this ledger must not carry",
            file=sys.stderr,
        )
    return CarriedState(
        state=state,
        rows=rows,
        round=round_,
        truncated=opened.truncated,
        # The renderer's cap, not the read's: see `CarriedState.coverage`.
        coverage=min(len(coverage), MAX_PRIOR_COVERAGE),
        expired=expired,
    )


def settle(
    carried: CarriedState,
    *,
    repo: str,
    pr: int,
    seat: str,
    head_sha: str,
    prior_status: Sequence[PriorFindingStatus] = (),
    findings: Sequence[AgenticFinding] = (),
    examined: Sequence[ExaminedRegion] = (),
    coverage_ledger: bool = False,
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
    out of the retention window unsettled -- announced on stderr by
    :func:`carry_in` when it happens (#173), which makes the end state visible to
    an operator but no more reachable by a round. That is this module's own loss
    arriving by volume, so a published finding whose ``(file, title)`` matches a
    carried row this round left open is treated as a re-assertion of that row: the row
    is touched (a round did look at it against this head) and the finding is not
    recorded again.

    ``(file, title)`` is therefore this module's DEFINITION of claim identity,
    not a guess at one: two findings naming the same file under the same headline
    are one claim here, and a round that means to record a distinct claim has to
    title it distinctly. That definition has a cost, and it is the only one in
    this module that runs toward loss rather than noise -- a genuinely new claim
    sharing a file and a case-folded title with a carried row is not written, and
    the row that survives carries the EARLIER body and the EARLIER evidence. It
    is accepted because the
    alternative is worse in kind, not in degree: keying on the body as well would
    miss every reworded re-report, which is most of them, and restore the
    unbounded growth above, whose end state is rows cut, unreachable and aged out
    unseen. A miss here costs one round's phrasing of a claim that is still open,
    still titled the same and still in the next prompt; a miss there costs the
    claim. Every suppression is named in :attr:`Settlement.deduped` and logged by
    the caller, so it is never the silent kind.

    A published finding that matches no open row may still be one this seat has
    seen: a round can name a claim an earlier round CLOSED by verdict. Such a
    finding **re-raises the closed row** (:func:`sidecar.review_state.reopen`)
    instead of minting a second one, which is #177's reversal -- a ``fixed`` or
    ``rejected`` verdict is no longer the last word on a claim, because the seat's
    own later reading of the code can answer it. Three properties make that a
    control rather than a loophole:

    * the trigger is a PUBLISHED finding -- a claim on the pull request that a
      human can read -- not a line in the fenced verdict channel. Nothing the
      reviewed content can say opens a row it could not already have opened by
      being a real problem;
    * a re-raised row is NOT the fenced channel's to close again for free: it
      re-enters the carried ledger like any open row, and its ``reopened`` count
      records that a round settled it and a later round disagreed;
    * the reversal runs toward noise, like every other lean here. A reopen that
      does not happen (store unreachable, row outside the settled read's window)
      records the claim as a fresh row, so the claim survives either way -- what
      is lost is the LINK between the closure and its contradiction, which is
      exactly the thing an operator, not a round, needs.

    ``stale`` rows are not re-raised: that closure is fuko's own (see
    :func:`sidecar.review_state.REOPENABLE_STATUSES`), and softening it is a
    separate question (#177 direction 3, #175).

    ``findings`` must be the findings the round actually PUBLISHED, not
    everything the model returned. The 86% loss this ledger repairs is about
    claims a developer saw and did not act on; carrying a low-confidence finding
    the confidence valve withheld into the next round's prompt would re-publish
    it through the settle path and route around that valve. The cost of
    recording at review time rather than post time is that a review whose post
    fails leaves rows behind -- they are re-offered next round, which is noise,
    and noise is the direction this module accepts.

    ``examined`` is the other half, and it is treated in the opposite way to
    ``findings`` on purpose. It is written UNFILTERED and unjudged: every entry
    the round returned is recorded as this seat's coverage, because the table's
    job is to record what a round actually said and re-deciding that here would
    make the ledger disagree with the round it describes. Nothing about a
    coverage row is published, so there is no valve to route around -- the
    judgement that matters is applied where the damage would happen, on the way
    back INTO a prompt (:func:`_carried_coverage`).

    ``coverage_ledger`` gates the write as well as the read (#157's staged
    rollout: default off, scored on receipts before it reaches a gating seat). It
    gates the WRITE too, rather than recording quietly for a later flip, because
    an entry written now can only be expired by a delta that touches its file
    later; a seat that accumulated coverage while switched off would, on being
    switched on, be handed entries about heads no round in between could expire.
    A seat with the flag off therefore behaves exactly as it did before this
    tier: no read, no write, byte-identical prompt.
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
    deduped: list[str] = []
    reopened: list[str] = []
    # Only read the closed ledger when there is something that could re-raise it.
    candidates = _reopen_candidates(repo, pr, seat) if findings else {}
    for finding in findings:
        anchor = _anchor(finding.file, finding.title)
        row_id = still_open.get(anchor)
        if row_id is not None:
            deduped.append(f"{finding.file}: {finding.title}")
            if row_id not in touch:
                touch.append(row_id)
            continue
        # `pop`, so two published findings on one claim re-raise ONE row -- and
        # the re-raised row joins `still_open`, so the SECOND of them takes the
        # dedup path above rather than being recorded fresh. Either half alone
        # still leaves two open rows for one claim, which is the compounding the
        # dedup exists to stop, arriving by the new door: a reopened row is an
        # open row, and the rest of this round has to see it as one.
        closed_row = candidates.pop(anchor, None)
        if closed_row is not None and review_state.reopen(
            closed_row.id, _reopen_reason(closed_row, carried.round)
        ):
            reopened.append(f"{finding.file}: {finding.title}")
            still_open[anchor] = closed_row.id
            continue
        fresh.append(finding)
    review_state.touch_findings(touch)
    recorded = review_state.record_findings(repo, pr, seat, carried.round, head_sha, fresh)
    coverage = (
        review_state.record_coverage(repo, pr, seat, carried.round, head_sha, _keyed(examined))
        if coverage_ledger
        else 0
    )
    return Settlement(
        closed=closed,
        reasserted=len(touch),
        recorded=recorded,
        deduped=tuple(deduped),
        reopened=tuple(reopened),
        coverage=coverage,
    )
