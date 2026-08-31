"""Repeatable scoring for the stateful-vs-stateless review A/B (#159, epic #160).

Every number motivating the stateful-review epic -- 4.7% cross-seat agreement,
86% of findings one-shot, ~26% estimated pool coverage, 64% of file-touches on
three paths -- came out of throwaway scripts that no longer exist. That is the
problem this module exists to fix, and the fix is not "write the scripts down":
it is to state ONE claim-identity rule and ONE estimator per metric, in code that
is tested, so a later run means the same thing as an earlier one.

Two consequences follow, and both are deliberate:

* **The published baselines are not reproduced by tuning.** A matching rule
  loose enough to hit 4.7% exactly is a rule fitted to its answer. This module
  fixes the rule first -- claim identity is
  :func:`sidecar.reviewer.ledger.claim_anchor`, the same ``(file, casefolded
  title)`` the findings ledger already uses to decide two claims are one -- and
  then reports whatever that rule yields. Regenerating the baselines under it
  (:mod:`scripts.ab_metrics` over the pre-A/B window) is what makes before and
  after comparable; agreeing with the old figures is not.
* **Nothing here decides anything.** The functions return counts and estimates.
  #159's decision rule was written before the data arrived -- ship on a material
  coverage gain with no quality regression, never on a token saving alone -- and
  a module that scored the arms would be the obvious place to quietly soften it.

Every function here is pure -- :func:`collect_claims` takes raw comment and
review dicts and returns claims, the rest take claims and return numbers. What
lives in the script is only what needs credentials or is likely to change:
fetching those dicts from GitHub, naming the arms, reading token cost out of
``review_runs``, and formatting. The join in :func:`collect_claims` is on this
side of that line deliberately -- its failure mode is a claim that silently never
arrives, so it has to be exercisable without a network.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .normalizers import collect_review_signals, collect_signals
from .reviewer.ledger import claim_anchor
from .signals import extract_run_receipts

_LABEL_RE = re.compile(r"^\s*🤖\s*`[^`]*`\s*$")
"""The visible model label a comment carries in A/B mode. Decoration, not a title."""

TOP_PATHS = 3
"""How many paths the concentration metric reports, matching the epic's ``64%``.

A fixed N rather than a parameter with a default, because the number is only
ever compared against a published baseline computed at 3; a caller free to ask
for 5 would produce a figure that reads like the same metric and is not.
"""


def claim_title(title: str, body: str) -> str:
    """Recover a claim's title from what was published, or ``""`` if nothing usable is.

    Lives here rather than in the fetching script because it is half of claim
    identity: :func:`claim_anchor` decides when two titles are one claim, and this
    decides what counts as a title at all. Both halves belong to the same rule.

    The stored title is preferred and the body's lines are the fallback, and both
    are filtered for the two shapes that are publisher decoration rather than
    content:

    * the visible ``🤖 `model``` label prefixed to every comment in A/B mode,
      which is IDENTICAL across arms whenever they run the same model -- exactly
      #159's design -- so keying on it would collapse one arm's claims onto the
      other's and report cross-arm agreement neither reviewer reached;
    * the ``**...**`` wrapper a rendered finding title arrives in, which degrades
      to ``****`` when the title was empty -- non-empty, and so silently keyed on
      as if it were content.

    Returning ``""`` is a real answer and callers must treat it as one: anchoring
    on the empty string would collapse every untitled finding in a file onto one
    claim. Findings published in a review body arrive with neither title nor body,
    so this is their normal outcome until they are rehydrated (#142).

    Stripping the bold wrapper also moves the recovered title closer to the clean
    one the ledger anchored on, but does not make them equal -- see
    :attr:`Claim.anchor`.
    """
    for candidate in (title, *body.splitlines()):
        if _LABEL_RE.match(candidate):
            continue
        cleaned = candidate.strip().strip("*").strip()
        if cleaned:
            return cleaned
    return ""


@dataclass(frozen=True)
class Claim:
    """One finding a seat published, reduced to what the metrics need.

    ``scope`` is the window a claim's IDENTITY is bounded by -- one PR.
    :func:`claim_anchor` carries no PR of its own, because the ledger it comes
    from is already scoped per ``(repo, pr, seat)``. Left unbounded across the
    multi-PR window this tool is meant to be run over, one ``(file, title)``
    recurring on two different PRs would count as a re-report, deflating the
    one-shot rate -- and neither arm's carried state spans PRs, so that movement
    could not be attributed to the variable under test. It is a stored field
    rather than a slice of ``round_key`` so nothing depends on that key's shape.

    ``round_key`` groups claims into the unit two arms are compared WITHIN --
    the head they reviewed, which for an inline comment is its
    ``original_commit_id`` and NOT the ``commit_id`` GitHub rewrites when it
    re-anchors an outdated thread (see :func:`collect_claims`). Agreement is only
    meaningful inside one: two seats naming the same problem on two different
    heads did not agree, they were asked twice.
    """

    arm: str
    round_key: str
    file: str
    title: str
    scope: str = ""

    @property
    def anchor(self) -> tuple[str, str]:
        """This claim's identity, under the findings ledger's own rule.

        The RULE is shared; the inputs need not be. A caller scoring published
        receipts re-derives ``title`` from rendered markdown, because the signal
        marker carries neither title nor body -- so this anchor will not equal
        the one the ledger stored for the same finding, and a join against ledger
        rows on it would silently match nothing. Every comparison in this module
        is between claims derived the same way, which is what makes it sound.
        """
        return claim_anchor(self.file, self.title)


def collect_claims(
    comments: Sequence[dict],
    reviews: Sequence[dict],
    arms: Mapping[str, str],
    pr_num: int,
) -> tuple[list[Claim], set[tuple[str, str]], int]:
    """Reduce one PR's raw receipts to claims, review receipts, and untitled drops.

    Pure, so the join it performs is testable without credentials -- which is the
    point of it living here rather than in the fetching script. Its failure mode
    is a claim that silently never arrives, and a join that can only be exercised
    against live GitHub is a join nothing regression-tests.

    BOTH publication channels are read. Anchorable findings become inline
    comments; everything else rides the review BODY -- ``overflow`` under
    "Findings without a diff anchor", and, on a 422 anchoring failure, a whole
    round demoted body-only. Reading only the first would drop precisely the
    off-diff class a stateful reviewer is hypothesized to add, so the exclusion
    would not be direction-neutral: it would bias the coverage metric toward the
    null.

    Fetching that channel DISCLOSES that bias; it does not yet close it. Markers
    carry neither title nor body and nothing rehydrates them from the prose they
    sit beside (#142), so every body-carried finding currently lands in the
    untitled drop and is reported as a count rather than scored. What reading the
    channel buys today is the receipt -- the round is known to have happened --
    and a number naming what is missing, instead of an omission with no trace in
    the output. Whoever reads a coverage figure while that count is non-zero is
    reading it with the treatment's hypothesized gain class outside the numerator,
    and should say so.

    A signal reaches an arm only through the comment or review it came from: the
    normalized signal carries no author, so it is rejoined by ``thread_url``.
    A finding whose author is not one of the named arms is skipped.

    **A round is the head the arm actually reviewed, so an inline comment is keyed
    on ``original_commit_id``, not ``commit_id``.** GitHub re-anchors an outdated
    review comment onto a newer head, rewriting ``commit_id`` while
    ``original_commit_id`` keeps the commit the comment was created against.
    Keying on the mutable field merges every earlier round forward into the
    latest head -- collapsing distinct rounds, inflating the one-shot rate,
    intersecting claims the arms made against DIFFERENT heads inside one
    "shared round", and, worst for a tool whose charter is repeatability, giving
    a different answer each time the same PR is rescored. Not hypothetical: 18 of
    28 top-level comments on ``lemehmet/fuko-pr#197`` carry a rewritten
    ``commit_id``. A review's own ``commit_id`` is written once at submission and
    stays put, so receipts keep using it.

    A review RECEIPT -- ``(arm, round_key)`` -- is emitted for every head a named
    arm reviewed, whether or not it published anything there, which is what lets
    :func:`pair_metrics` score a one-versus-zero round rather than dropping the
    most disagreeing round in the window. Each claim's own key is unioned in too,
    so a receipt whose commit differs from its comments' can never exclude a
    claim that was actually published.

    A claim with no usable title is DROPPED and counted, never anchored on the
    empty string -- see :func:`claim_title`.
    """
    login_to_arm = {login: arm for arm, login in arms.items()}
    by_thread: dict[str, tuple[str, str]] = {
        c.get("html_url"): (
            (c.get("user") or {}).get("login", ""),
            c.get("original_commit_id") or c.get("commit_id") or "",
        )
        for c in comments
    }
    # Review bodies join the same way: `collect_review_signals` stamps each
    # marker's `thread_url` with the REVIEW's html_url when it carries none.
    by_thread.update(
        {
            r.get("html_url"): ((r.get("user") or {}).get("login", ""), r.get("commit_id") or "")
            for r in reviews
        }
    )
    receipts: set[tuple[str, str]] = set()
    for review in reviews:
        arm = login_to_arm.get((review.get("user") or {}).get("login", ""))
        if arm is not None:
            receipts.add((arm, f"{pr_num}@{review.get('commit_id') or ''}"))
    claims: list[Claim] = []
    untitled = 0
    for signal in collect_signals(list(comments)) + collect_review_signals(list(reviews)):
        login, commit = by_thread.get(signal.thread_url or "", ("", ""))
        arm = login_to_arm.get(login)
        if arm is None or not signal.file:
            continue
        title = claim_title(signal.title, signal.body)
        if not title:
            untitled += 1
            continue
        claim = Claim(
            arm=arm,
            round_key=f"{pr_num}@{commit}",
            file=signal.file,
            title=title,
            scope=str(pr_num),
        )
        receipts.add((claim.arm, claim.round_key))
        claims.append(claim)
    return claims, receipts, untitled


def backup_served_rounds(
    issue_comments: Sequence[dict],
    arms: Mapping[str, str],
    pr_num: int,
) -> set[tuple[str, str]]:
    """The ``(arm, round_key)`` rounds a FAILOVER backup answered, not the seat's primary.

    Every metric above assigns a round to an arm by who POSTED it, which is the
    rescued seat's identity whether its own model answered or a shared backup
    did. That is correct attribution and a silent confound at the same time: the
    arms are meant to differ in one variable, and a rescued round differs in the
    model too. The discriminator needs no new plumbing -- the branch header's
    ``fuko-run:v1`` receipt already records ``label`` (the branch's configured
    primary) beside ``model`` (the entry that actually answered), and they differ
    when a backup answered under a different ``provider/name``. That is a
    sufficient test, not a biconditional: two entries may legally share a
    ``provider/name`` and differ only by ``token_env``/endpoint, and a rescue
    between those is invisible here. The error direction is under-reporting a
    confound, never inventing one.

    Reported, never subtracted. Dropping such a round would change the
    denominator of a metric whose charter is repeatability, and #204's runner fix
    already removed the part that made a rescue *wrong* rather than merely
    different: the ledger configuration a rescued round runs is the branch's, so
    what is left is a model confound the reader should weigh, not a mislabelled
    arm. A caller that wants them excluded can subtract this set itself.

    Only a ``done`` receipt counts. ``in_progress`` names no answering model
    yet, and ``failed`` means the branch's pool was EXHAUSTED -- primary and
    backups alike -- so its ``model`` is whichever entry was tried LAST, not one
    that answered. A round nobody answered produces no claims, so it is in no
    metric above, and naming it in a report that says these rounds are
    "INCLUDED in every metric" would be false. That is not hypothetical here:
    an exhausted pool is the shape a dead backup key produces on every branch.

    An empty ``head_sha`` is skipped for the same reason. It is a designed
    degradation, not a bug -- :func:`sidecar.runner._head_for_receipts` returns
    ``""`` when HEAD cannot be resolved, so the receipt still records that the
    instance ran, on an unknown commit -- and its own contract is that "a
    consumer treats an empty ``head_sha`` as not covering the current HEAD".
    This is such a consumer. Keying one anyway yields ``<pr>@``, which joins to
    no round: the claims and review receipts are keyed by the review's
    ``commit_id``, so the round itself sits under ``<pr>@<sha>`` and a reader
    subtracting the printed key would subtract nothing while being told the
    round is included.

    The arm comes from the comment's own author. The receipt's ``app`` is a
    fallback only for a payload read WITHOUT its envelope -- no usable author
    login. An author that is present and names no arm is skipped rather than
    re-attributed from the body, so a receipt quoted or copied by a third party
    cannot mint a round for the seat it happens to name.
    """
    login_to_arm = {login: arm for arm, login in arms.items()}
    out: set[tuple[str, str]] = set()
    for comment in issue_comments:
        login = (comment.get("user") or {}).get("login") or ""
        for receipt in extract_run_receipts(comment.get("body") or ""):
            arm = login_to_arm.get(login) if login else login_to_arm.get(receipt.app)
            if arm is None or receipt.state != "done" or not receipt.head_sha:
                continue
            if not receipt.model or receipt.model == receipt.label:
                continue
            out.add((arm, f"{pr_num}@{receipt.head_sha}"))
    return out


@dataclass(frozen=True)
class ArmMetrics:
    """What one arm did on its own, independent of the other.

    ``one_shot_rate`` is the epic's headline loss restated as a fraction of
    DISTINCT claims: a claim published in exactly one round, never re-noticed.
    It is not a quality measure in either direction on its own -- a claim the
    author fixed immediately is also one-shot -- which is why #159 reads it
    beside coverage rather than instead of it.

    ``top_paths_share`` counts file-TOUCHES, not distinct paths: a seat that
    reports six findings on one file has touched it six times, and the
    concentration this measures is precisely a seat spending its budget in the
    same place.
    """

    arm: str
    rounds: int
    findings: int
    distinct_claims: int
    distinct_paths: int
    re_reported: int
    one_shot_rate: float | None
    top_paths_share: float | None
    top_paths: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PairMetrics:
    """What two arms did relative to each other, over the rounds both reviewed.

    ``rounds`` counts the heads where BOTH arms left a review RECEIPT. The
    confound to keep out is a round an arm sat out -- throttled, timed out, App
    token unminted -- which folded in as a zero would report a fleet that
    disagreed when it was really a fleet that was half absent. Membership was
    first keyed on published claims for that reason, but that key cannot tell an
    absent arm from a present one that found nothing, and it excludes the
    one-versus-zero round: the maximum-disagreement case, dropped from a pooled
    ratio, biases agreement UPWARD on the metric #159 ships on (CodeRabbit,
    #203). A review receipt separates the two cases directly -- an arm that
    reviewed a head and published nothing left one; an arm that never ran did
    not -- so it is the honest key, and the confound stays out.

    With no receipts supplied the claim-derived key stands in, which is the older
    behaviour and still the only thing available to a caller holding claims
    alone.

    ``coverage`` is the Chapman estimate: the union of what the two arms saw,
    over the pool their overlap implies exists. It inherits capture-recapture's
    assumptions and violates one of them by construction -- two seats on the same
    model are not independent samplers -- which biases the estimate toward
    OPTIMISM (correlated seats overlap more, implying a smaller pool). The
    baseline it is compared against was computed the same way, so the comparison
    survives what the absolute number does not.
    """

    rounds: int
    a_claims: int
    b_claims: int
    shared: int
    union: int
    agreement: float | None
    pool_estimate: float | None
    coverage: float | None


def chapman_pool_size(n1: int, n2: int, shared: int) -> float | None:
    """Chapman's bias-corrected capture-recapture estimate of the pool size.

    ``(n1 + 1)(n2 + 1) / (shared + 1) - 1``. Chapman rather than plain
    Lincoln-Petersen (``n1 * n2 / shared``) for the reason that matters at these
    sample sizes: the naive form is undefined at zero overlap and wildly biased
    just above it, and zero overlap is a routine outcome for two seats agreeing
    4.7% of the time. The ``+1`` terms make it defined everywhere, at the cost of
    an estimate that is merely very large rather than infinite when two arms
    share nothing.

    Returns ``None`` when either arm saw nothing, because a pool cannot be
    estimated from one capture -- not ``0``, which would read as "there was
    nothing to find".
    """
    if n1 <= 0 or n2 <= 0:
        return None
    return (n1 + 1) * (n2 + 1) / (shared + 1) - 1


def _by_round(claims: Iterable[Claim]) -> dict[str, set[tuple[str, str]]]:
    """Group claim anchors by round key."""
    out: dict[str, set[tuple[str, str]]] = {}
    for claim in claims:
        out.setdefault(claim.round_key, set()).add(claim.anchor)
    return out


def _receipt_rounds(arm: str, receipts: Sequence[tuple[str, str]]) -> set[str]:
    """The round keys ``arm`` left a review receipt on."""
    return {key for owner, key in receipts if owner == arm}


def arm_metrics(
    arm: str, claims: Sequence[Claim], receipts: Sequence[tuple[str, str]] = ()
) -> ArmMetrics:
    """Score one arm's own claims.

    ``claims`` may hold other arms' rows; they are filtered here so a caller can
    pass the whole window without partitioning it first.

    ``receipts`` are ``(arm, round_key)`` pairs for every head an arm reviewed,
    including the heads it reviewed and published nothing on. Supplying them
    makes ``rounds`` count rounds REVIEWED rather than rounds published on, which
    is both the honest figure and the one :func:`pair_metrics` scores against;
    without them the claim-derived count stands in.
    """
    mine = [c for c in claims if c.arm == arm]
    rounds = _receipt_rounds(arm, receipts) if receipts else {c.round_key for c in mine}
    # Keyed by (PR, anchor): a re-report is the SAME claim seen in more than one
    # round of one PR's review cycle, not one headline recurring across PRs.
    seen: dict[tuple[str, tuple[str, str]], set[str]] = {}
    for claim in mine:
        seen.setdefault((claim.scope, claim.anchor), set()).add(claim.round_key)
    re_reported = sum(1 for keys in seen.values() if len(keys) > 1)
    touches = Counter(c.file for c in mine)
    top = touches.most_common(TOP_PATHS)
    return ArmMetrics(
        arm=arm,
        rounds=len(rounds),
        findings=len(mine),
        distinct_claims=len(seen),
        distinct_paths=len(touches),
        re_reported=re_reported,
        one_shot_rate=(len(seen) - re_reported) / len(seen) if seen else None,
        top_paths_share=sum(n for _, n in top) / len(mine) if mine else None,
        top_paths=tuple(top),
    )


def pair_metrics(
    claims: Sequence[Claim], a: str, b: str, receipts: Sequence[tuple[str, str]] = ()
) -> PairMetrics:
    """Score arm ``a`` against arm ``b`` over the rounds both of them reviewed.

    ``receipts`` are ``(arm, round_key)`` pairs for every head an arm reviewed.
    They decide round membership: a head both arms reviewed is scored even if one
    of them published nothing there, which is the round that carries the most
    disagreement and the one a claim-derived key silently drops. Pass them
    whenever they can be observed; without them membership falls back to the
    heads both arms published on, and the reported agreement is an upper bound.

    Agreement is pooled rather than averaged per round: the intersections and
    unions are summed across shared rounds and divided once. A mean of per-round
    Jaccards would weight a round with two findings the same as a round with
    thirty, which at these sample sizes is dominated by the small rounds -- and
    the small rounds are the ones where a single coincidence moves the number
    most.
    """
    left, right = (
        _by_round(c for c in claims if c.arm == a),
        _by_round(c for c in claims if c.arm == b),
    )
    if receipts:
        shared_rounds = sorted(_receipt_rounds(a, receipts) & _receipt_rounds(b, receipts))
    else:
        shared_rounds = sorted(set(left) & set(right))
    n1 = n2 = shared = union = 0
    for key in shared_rounds:
        mine, theirs = left.get(key, set()), right.get(key, set())
        n1 += len(mine)
        n2 += len(theirs)
        shared += len(mine & theirs)
        union += len(mine | theirs)
    return PairMetrics(
        rounds=len(shared_rounds),
        a_claims=n1,
        b_claims=n2,
        shared=shared,
        union=union,
        agreement=shared / union if union else None,
        pool_estimate=chapman_pool_size(n1, n2, shared),
        coverage=_coverage(union, chapman_pool_size(n1, n2, shared)),
    )


def _coverage(union: int, pool: float | None) -> float | None:
    """Observed union over the estimated pool, or ``None`` when undefined."""
    if pool is None or pool <= 0:
        return None
    return union / pool
