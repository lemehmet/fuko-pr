"""Per-reviewer review STATE on a PR's current HEAD — the normalized "done" signal.

`fuko status` answers *has each reviewer finished reviewing the current HEAD?*
for the bots a review loop gates on. It is the state counterpart to
`fuko signals` (which answers *what did they find?*). Only **observable**
artifacts are read — fuko makes no time judgments like "unresponsive"; a
consumer applies its own timeout to a `pending` state.

Two kinds of reviewer are reported. **External** bots (CodeRabbit, Copilot) are
read from the artifacts they happen to leave behind, which is why each needs its
own heuristic below. **fuko's own instances** are read from the run receipts
they write deliberately (`fuko_states`), so their coverage is a recorded fact
rather than an inference — closing the gap where an instance that never started
was indistinguishable from one that reviewed and found nothing.

CodeRabbit's completion is taken from its **check-run** on the HEAD commit when one
is present ("Review in progress" → completed) — the only signal that doesn't race
the inline comments (issue #17). Its walkthrough issue comment is the fallback when
no check-run is observable, and only a *terminal* walkthrough marker (a completion
line, not merely the "Reviewing files … between …" range that CR posts up front)
counts as done there. Copilot's state is its latest review's `commit_id`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal

from .signals import RunReceipt, extract_run_receipts

State = Literal[
    "done", "degraded", "pending", "in_progress", "rate_limited", "paused", "unavailable", "none"
]

DEGRADED_STATES: frozenset[str] = frozenset({"rate_limited", "paused", "unavailable", "degraded"})

_CR_LOGIN = "coderabbitai[bot]"
_COPILOT_LOGINS = {"copilot", "copilot-pull-request-reviewer[bot]"}
_COPILOT_QUOTA = re.compile(
    r"wasn't able to review"
    r"|quota (?:limit|exceeded|exhausted|reached)"
    r"|exceeded your .{0,30}?(?:quota|premium requests?|monthly limit)"
    r"|monthly limit of premium requests"
    r"|out of (?:premium )?(?:credits|requests)",
    re.I,
)

_CR_IN_PROGRESS = re.compile(r"review in progress|Currently processing new changes", re.I)
_CR_RATE_LIMIT = re.compile(r"Rate limit exceeded", re.I)
_CR_PAUSED = re.compile(r"Reviews paused|review paused by coderabbit\.ai", re.I)
_CR_DONE_ZERO = re.compile(r"(?im)^[>\s*_]*No actionable comments(?: were generated)?\b")
_CR_DONE_MARKER = re.compile(
    r"(?im)^[>\s*_]*(?:Actionable comments posted:\s*\d+"
    r"|No actionable comments(?: were generated)?)\b"
)
_CR_REVIEWING = re.compile(r"between\s+`?([0-9a-f]{7,40})`?\s+and\s+`?([0-9a-f]{7,40})`?", re.I)

_CR_CHECK_NAMES = re.compile(r"coderabbit", re.I)

# Identities a run receipt may legitimately come from. ANCHORED, and requiring
# the `[bot]` suffix, deliberately: the loose `fuko` substring used for finding
# TRIAGE is wrong here. Over-matching a triage pattern only pulls in extra
# findings, but over-matching here admits forged COVERAGE -- and `fuko` as a
# substring is claimable by any account calling itself `fukoo-imposter`.
# GitHub forbids `[` and `]` in user names, so no human account can satisfy this;
# minting a matching App instead requires repo-admin installation, a trusted act.
# The `fuko-<slot>[bot]` shape still covers every instance (fuko-dorian[bot],
# fuko-gray[bot], ...) so a new slot or an App rename cannot silently drop
# coverage. `github-actions[bot]` is fuko's documented App-less fallback and MUST
# STAY accepted: it is the identity of the sequential-fallback path taken whenever
# App tokens are unavailable, so removing it as "over-permissive" would void
# coverage for exactly the degraded runs this reporting exists to catch. Assuming
# it requires repo write access, which an untrusted PR author does not have.
_FUKO_RECEIPT_AUTHORS = re.compile(r"^(?:fuko-[\w.-]+\[bot\]|github-actions\[bot\])$", re.I)


def _coderabbit_check(check_runs: list[dict] | None) -> dict | None:
    """Return CodeRabbit's check-run from ``check_runs`` (the review check), if present.

    Matches by the check's ``name`` and, defensively, its app slug, so a rename of the
    visible check title ("CodeRabbit" / "Review") still resolves as long as either the
    name or the owning app mentions coderabbit. The caller has already fetched the
    check-runs for the specific HEAD SHA, so any match here is on-HEAD by construction.
    """
    for c in check_runs or []:
        name = c.get("name", "") or ""
        slug = ((c.get("app") or {}).get("slug")) or ""
        if _CR_CHECK_NAMES.search(name) or _CR_CHECK_NAMES.search(slug):
            return c
    return None


def _receipt_author_allowed(login: str, allowed: set[str] | None) -> bool:
    """Return whether ``login`` may assert run coverage.

    An empty login is refused outright: it carries no identity to check, and the
    unsafe direction here is accepting coverage, not rejecting it.
    """
    if not login:
        return False
    if allowed is not None:
        return login.lower() in allowed
    return bool(_FUKO_RECEIPT_AUTHORS.search(login))


def _range_heads(body: str) -> list[str]:
    """Return the end-sha of every "between X and Y" range line in ``body``.

    ``finditer`` rather than one ``search``: CodeRabbit rewrites its summary
    comment in place and a single body can carry more than one range line, so
    asking only about the first can miss the range that covers HEAD.
    """
    return [m.group(2) for m in _CR_REVIEWING.finditer(body or "")]


def _row(backend: str, state: State, head_reviewed: str | None, detail: str) -> dict:
    return {"backend": backend, "state": state, "head_reviewed": head_reviewed, "detail": detail}


def _sha_match(a: str | None, b: str | None) -> bool:
    """Prefix-compare two commit shas (CodeRabbit may abbreviate)."""
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def coderabbit_state(
    head_sha: str,
    issue_comments: list[dict],
    reviews: list[dict],
    check_runs: list[dict] | None = None,
) -> dict:
    """Derive CodeRabbit's state on ``head_sha``.

    The authoritative signal is CodeRabbit's **check-run** on the HEAD commit (issue
    #17): if present, its ``status`` decides — ``in_progress``/``queued`` means CR is
    still scanning (regardless of what its walkthrough already says), and only a
    ``completed`` status is ``done``. This removes the race where CR posts its
    walkthrough range line up front and then streams inline comments 1–2 min later.

    When no check-run is observable (older PRs, or a token without checks access),
    fall back to the comment/review heuristic — but require a *terminal* marker
    ("Actionable comments posted: N" / "No actionable comments") rather than the mere
    "Reviewing files … between … and <HEAD>" range line, which CR posts before it has
    finished. The range line and the terminal marker live in CR's walkthrough issue
    comment *or* its review body depending on CR's mode, so both are searched. A
    submitted CR review whose ``commit_id`` is HEAD is itself terminal — its inline
    comments are posted atomically with the review — so it reports ``done`` directly,
    even with an empty body (an APPROVED review often carries no marker). The issue #17
    race is the *up-front walkthrough* issue comment, which CR posts before it has
    finished; so when only that range line covers HEAD and CR has not yet submitted a
    review, a terminal marker is still required to report ``done``. Marker text is
    scoped to the current HEAD (a range line ending at HEAD, an on-HEAD review body, or
    — once CR has reviewed HEAD — its in-place summary) so a prior HEAD's stale marker
    can no longer report done early.

    When only the walkthrough *range line* covers HEAD — i.e. the ``_CR_REVIEWING``
    "Reviewing … between … and <HEAD>" line names HEAD, with no CR check-run, no
    submitted review on HEAD, no terminal marker, and no explicit "review in progress"
    text — this is **not** asserted as ``in_progress`` but reported as ``pending``
    (issue #34). Under a rapid-push skip CR can bump that range line to a new HEAD and
    then never engage it, so the range line alone is not evidence of an active scan;
    ``pending`` lets the consumer's own unresponsive timeout govern instead of an implied
    never-ending scan. ``in_progress`` is reserved for demonstrable engagement: a CR
    check-run that is still running, or an explicit in-progress notice in CR's live
    status *issue comment* (the in-progress text is scoped to ``cr_issue_bodies`` so a
    stale phrase in an older submitted review body for a prior HEAD cannot spuriously
    flip the current HEAD back to ``in_progress``).
    """
    cr_issue_bodies = [
        c.get("body", "") or ""
        for c in issue_comments
        if (c.get("user") or {}).get("login", "").lower() == _CR_LOGIN
    ]
    cr_reviews = [r for r in reviews if (r.get("user") or {}).get("login", "").lower() == _CR_LOGIN]
    bodies = cr_issue_bodies + [r.get("body", "") or "" for r in cr_reviews]
    check = _coderabbit_check(check_runs)
    if not bodies and not cr_reviews and check is None:
        return _row("coderabbit", "none", None, "no CodeRabbit activity")

    if check is not None:
        status = (check.get("status") or "").lower()
        if status != "completed":
            return _row(
                "coderabbit", "in_progress", head_sha, f"check-run still {status or 'pending'}"
            )
        conclusion = (check.get("conclusion") or "").lower()
        zero = bool(_CR_DONE_ZERO.search("\n".join(bodies)))
        return _row(
            "coderabbit",
            "done",
            head_sha,
            (
                "no actionable comments"
                if zero
                else f"check-run completed ({conclusion or 'neutral'})"
            ),
        )

    blob = "\n".join(bodies)
    issue_blob = "\n".join(cr_issue_bodies)
    # Every range end-sha, in body order. `walk_on_head` used to scan ALL bodies
    # while `walk_head` was bound from the FIRST one; once a PR has more than one
    # CR review those disagree, and the reported `head_reviewed` was the oldest
    # range rather than the one that matched HEAD (#101).
    walk_heads = [h for b in bodies for h in _range_heads(b)]
    walk_on_head = any(_sha_match(h, head_sha) for h in walk_heads)
    # Prefer the range that actually covers HEAD; otherwise the MOST RECENT range
    # seen, since a later body is a later review.
    walk_head = next(
        (h for h in walk_heads if _sha_match(h, head_sha)),
        walk_heads[-1] if walk_heads else None,
    )
    review_on_head = any(r.get("commit_id") == head_sha for r in cr_reviews)

    if not (walk_on_head or review_on_head):
        if _CR_RATE_LIMIT.search(blob):
            return _row(
                "coderabbit", "rate_limited", walk_head, "rate-limit notice; HEAD not yet scanned"
            )
        if _CR_PAUSED.search(blob):
            return _row("coderabbit", "paused", walk_head, "reviews paused; HEAD not yet scanned")
        if _CR_IN_PROGRESS.search(issue_blob):
            return _row("coderabbit", "in_progress", walk_head, "review in progress")
        return _row(
            "coderabbit", "pending", walk_head, "neither walkthrough nor review covers the HEAD"
        )

    head_blob = "\n".join(
        # Any range line in the body may be the one covering HEAD, so this asks
        # about all of them rather than only the first (#101). Checking just the
        # first would drop a body whose later range covers HEAD, and with it the
        # terminal marker that body carries.
        [b for b in bodies if any(_sha_match(h, head_sha) for h in _range_heads(b))]
        + [r.get("body", "") or "" for r in cr_reviews if r.get("commit_id") == head_sha]
        + (cr_issue_bodies if review_on_head else [])
    )
    if not review_on_head and not _CR_DONE_MARKER.search(head_blob):
        if _CR_IN_PROGRESS.search(issue_blob):
            return _row(
                "coderabbit",
                "in_progress",
                head_sha,
                "review in progress (CR named HEAD and reports an active scan)",
            )
        return _row(
            "coderabbit",
            "pending",
            head_sha,
            "walkthrough range line covers HEAD but CR shows no completion marker, "
            "submitted review, or in-progress check-run yet",
        )

    zero = bool(_CR_DONE_ZERO.search(head_blob))
    return _row(
        "coderabbit",
        "done",
        walk_head if walk_on_head else head_sha,
        "no actionable comments" if zero else "scanned HEAD (any findings are inline)",
    )


def copilot_state(
    head_sha: str, reviews: list[dict], issue_comments: list[dict] | None = None
) -> dict:
    """Derive Copilot's state from its latest review's commit id (reliable for Copilot).

    Quota exhaustion (no auto top-up of premium requests) surfaces as a notice
    in a Copilot-authored review body or issue comment rather than as a normal
    review -- "wasn't able to review", "quota", "monthly limit" and similar.
    When such a notice exists on this PR and no review covers HEAD, the state is
    ``unavailable`` (a degraded state, see :data:`DEGRADED_STATES`). A review on
    HEAD always wins over an older notice: credits were evidently topped up.
    """
    cps = [r for r in reviews if (r.get("user") or {}).get("login", "").lower() in _COPILOT_LOGINS]
    on_head = [r for r in cps if r.get("commit_id") == head_sha]
    if on_head:
        return _row("copilot", "done", head_sha, f"review on HEAD ({on_head[-1].get('state')})")

    copilot_bodies = [r.get("body", "") or "" for r in cps] + [
        c.get("body", "") or ""
        for c in issue_comments or []
        if (c.get("user") or {}).get("login", "").lower() in _COPILOT_LOGINS
    ]
    if any(_COPILOT_QUOTA.search(b) for b in copilot_bodies):
        return _row(
            "copilot",
            "unavailable",
            cps[-1].get("commit_id") if cps else None,
            "quota/unable-to-review notice on this PR and no review on HEAD",
        )

    if not cps:
        return _row("copilot", "none", None, "no Copilot review")
    return _row(
        "copilot",
        "pending",
        cps[-1].get("commit_id"),
        "latest Copilot review is on an older commit",
    )


def fuko_states(
    head_sha: str,
    issue_comments: list[dict],
    *,
    allowed_authors: Iterable[str] | None = None,
) -> list[dict]:
    """Return one state row per fuko instance, from the run receipts on this PR.

    fuko's own instances were historically invisible to ``fuko status``: they are
    not external reviewers, so nothing reported whether they had run. That left a
    consumer unable to distinguish "this instance reviewed HEAD and found nothing"
    from "this instance never started" -- both show up as zero signals -- and the
    ambiguity resolves in the unsafe direction, merging unreviewed code.

    Each branch writes a :class:`~sidecar.signals.RunReceipt` into its header
    comment (see :func:`sidecar.runner._post_branch_header`), so this reads that
    back. Rows carry the extra ``role`` key -- a ``trial`` instance is reported
    but must not gate -- and use backend ``fuko:<label>`` so they never collide
    with the CodeRabbit/Copilot rows.

    States map so that only a receipt finalized as ``done`` *on this HEAD* reads
    as ``done``:

    - ``done`` -- reviewed this HEAD on every channel it publishes.
    - ``degraded`` -- reviewed this HEAD, but at least one channel did not finish
      (see below). A DEGRADED state, so :func:`escalation_needed` fires on it.
    - ``in_progress`` -- started this HEAD, no outcome recorded yet. Also what a
      branch that died mid-run leaves behind, so a consumer's own timeout governs
      rather than an implied never-ending run (the same choice
      :func:`coderabbit_state` makes).
    - ``unavailable`` -- the branch finalized as failed: every model in its pool
      was exhausted. A DEGRADED state, so :func:`escalation_needed` fires on it.
    - ``pending`` -- the newest receipt is for an older commit.

    ``degraded`` is a distinct STATE rather than an extra key on a ``done`` row
    on purpose. A seat publishes on several channels (the PR-level guide from
    ``review``, inline suggestions from ``improve``), and an optional tool's
    death leaves the branch's return code at zero -- so the branch reports
    ``done`` while one of its channels produced nothing. Signalling that only
    through a new ``channels`` key would leave every consumer that gates on
    ``state == "done"`` -- the common case -- reading a half-dead seat as a clean
    pass, which is the defect itself (#108). An unrecognized state value instead
    reads as not-done, withholding a merge rather than granting one.

    A receipt with an EMPTY channel map reports its branch-level state unchanged:
    an empty map means NOT REPORTED, so there is nothing to judge. Emptiness does
    not identify one situation -- an in-flight receipt carries none before any
    tool finishes, a backend that does not report per-channel outcomes (the
    agentic one today) produces none, and receipts written before this field
    existed have none. Treating empty as degraded would flag all three, including
    every pre-upgrade receipt; treating it as healthy is the compatible reading
    and the honest limit of it is that a dead channel on such a receipt is simply
    invisible here. Closing that is per-backend work, not a claim that those
    backends have no channels to lose.

    Only the newest receipt per instance is reported. Receipts are rewritten in
    place, but a force-push or a re-run can leave an older one behind, and later
    comments are the later runs.

    Receipts are **author-scoped**, like the CodeRabbit and Copilot paths above.
    A receipt is the answer to "did this reviewer actually run?", so anyone who
    can comment on the PR could otherwise assert coverage for a review that never
    happened -- on a public repo, including the PR author -- and a consumer would
    stop waiting on an instance that never started. Only comments authored by a
    fuko App identity (or the App-less ``github-actions[bot]`` fallback the
    workflow posts under) are read; ``allowed_authors`` overrides that default
    with an exact-login allowlist for deployments using different identities.
    Rejection is silent by design: an unrecognized author yields no row, which
    reads as "this instance never ran" and withholds a merge rather than
    granting one.
    """
    allowed = None if allowed_authors is None else {a.strip().lower() for a in allowed_authors}
    latest: dict[str, RunReceipt] = {}
    for comment in issue_comments:
        login = ((comment.get("user") or {}).get("login") or "").strip()
        if not _receipt_author_allowed(login, allowed):
            continue
        for receipt in extract_run_receipts(comment.get("body", "") or ""):
            latest[receipt.label] = receipt

    rows: list[dict] = []
    for label, receipt in latest.items():
        on_head = _sha_match(receipt.head_sha, head_sha) if receipt.head_sha else False
        # Staleness is asked FIRST, ahead of the outcome. A receipt is anchored to
        # the commit it describes, so a `failed` one for an older commit says
        # nothing about HEAD -- reporting it as `unavailable` would keep firing
        # `escalation_needed` on every later round until that instance happened to
        # succeed, a failure that sticks long after the push that outdated it. This
        # is the opposite order from :func:`copilot_state`, deliberately: Copilot's
        # quota notice carries no commit anchor and so cannot go stale this way.
        if not on_head:
            state: State = "pending"
            detail = f"latest run covers {receipt.head_sha[:7] or 'an unknown commit'}, not HEAD"
        elif receipt.state == "failed":
            state = "unavailable"
            detail = receipt.detail or "every model in the branch pool was exhausted"
        elif receipt.state == "done":
            # A promoted backup answered under a different model than the branch
            # is named for; say so, since it changes whose findings these are.
            promoted = receipt.model and receipt.model != label
            reviewed = f"reviewed HEAD as {receipt.model}" if promoted else "reviewed HEAD"
            dead = sorted(
                f"{name} {value}" for name, value in receipt.channels.items() if value != "done"
            )
            if dead:
                state = "degraded"
                detail = f"{reviewed}, but reduced coverage: {', '.join(dead)}"
            else:
                state = "done"
                detail = reviewed
        else:
            state = "in_progress"
            detail = "started on HEAD, no outcome recorded yet"
        row = _row(f"fuko:{label}", state, receipt.head_sha or None, detail)
        row["role"] = receipt.role
        if receipt.channels:
            row["channels"] = dict(receipt.channels)
        rows.append(row)
    return sorted(rows, key=lambda r: r["backend"])


def reviewer_states(
    head_sha: str,
    issue_comments: list[dict],
    reviews: list[dict],
    check_runs: list[dict] | None = None,
    *,
    include_fuko: bool = True,
    allowed_authors: Iterable[str] | None = None,
) -> list[dict]:
    """Return the normalized state of each reviewer on ``head_sha``.

    Covers the external bots (CodeRabbit, Copilot) and, when ``include_fuko`` is
    set, fuko's own instances via :func:`fuko_states`. ``check_runs`` are the
    check-runs fetched for ``head_sha``; CodeRabbit's completion is read from its
    own check there when present (issue #17). Both are optional so existing
    callers and tests that only have comment/review data still work.

    ``include_fuko`` exists for one caller: :func:`sidecar.runner._observe_reviewer_health`
    records *external* reviewer health to decide backup promotion, and folding
    fuko's own instances into that would let fuko escalate in response to itself.

    ``allowed_authors`` is forwarded to :func:`fuko_states` for deployments whose
    reviewer identities don't match the default fuko-App pattern.
    """
    rows = [
        coderabbit_state(head_sha, issue_comments, reviews, check_runs),
        copilot_state(head_sha, reviews, issue_comments),
    ]
    if include_fuko:
        rows.extend(fuko_states(head_sha, issue_comments, allowed_authors=allowed_authors))
    return rows


def escalation_needed(rows: Iterable[Mapping]) -> bool:
    """Decide whether the external-reviewer situation warrants model escalation.

    THE single policy shared by every consumer (the runner's next-round backup
    promotion today; anything reading ``fuko status`` can apply the same set),
    so detection and policy can never drift apart across consumers: a reviewer
    row in any :data:`DEGRADED_STATES` state -- explicitly throttled, paused,
    quota-exhausted, or finished with a dead channel -- means the PR is losing
    review coverage and fuko's backup models should join the round. Including
    ``degraded`` cannot make fuko escalate in response to itself: the runner's
    promotion path feeds this from :mod:`sidecar.reviewer_health` rows (external
    reviewers only), never from :func:`fuko_states`. ``pending``/``none`` are NOT
    degraded: they are normal early-round states and a single snapshot cannot
    distinguish "slow" from "silent-because-broke"; a staleness-aware policy
    can tighten this once run metrics exist. Accepts both
    :func:`reviewer_states` rows and :func:`sidecar.reviewer_health.states`
    rows (only the ``state`` key is read).
    """
    return any((row.get("state") or "") in DEGRADED_STATES for row in rows)
