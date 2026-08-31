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

Every function is pure: it takes claims and returns numbers. Fetching them from
GitHub, mapping bot logins onto arms, and reading token cost out of
``review_runs`` all live in the script, because those are the parts that need
credentials and the parts most likely to change.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .reviewer.ledger import claim_anchor

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

    ``round_key`` groups claims into the unit two arms are compared WITHIN --
    the head they reviewed, in practice the inline comment's ``commit_id``.
    Agreement is only meaningful inside one: two seats naming the same problem on
    two different heads did not agree, they were asked twice.
    """

    arm: str
    round_key: str
    file: str
    title: str

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

    ``rounds`` counts only the heads where BOTH arms published at least one
    claim. A round one arm sat out (throttled, timed out, or genuinely found
    nothing) carries no information about agreement, and folding it in as a zero
    would report a fleet that disagreed when it was really a fleet that was half
    absent -- the exact confound a receipts experiment has to keep out.

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


def arm_metrics(arm: str, claims: Sequence[Claim]) -> ArmMetrics:
    """Score one arm's own claims.

    ``claims`` may hold other arms' rows; they are filtered here so a caller can
    pass the whole window without partitioning it first.
    """
    mine = [c for c in claims if c.arm == arm]
    rounds = {c.round_key for c in mine}
    seen: dict[tuple[str, str], set[str]] = {}
    for claim in mine:
        seen.setdefault(claim.anchor, set()).add(claim.round_key)
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


def pair_metrics(claims: Sequence[Claim], a: str, b: str) -> PairMetrics:
    """Score arm ``a`` against arm ``b`` over the rounds both of them reviewed.

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
    shared_rounds = sorted(set(left) & set(right))
    n1 = n2 = shared = union = 0
    for key in shared_rounds:
        n1 += len(left[key])
        n2 += len(right[key])
        shared += len(left[key] & right[key])
        union += len(left[key] | right[key])
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
