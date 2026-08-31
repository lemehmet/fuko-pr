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

Both of CodeRabbit's completion signals are emitted under a Fair-Usage limit too,
on a commit CR never read, so a live limit or pause notice in CR's in-place-edited
summary DEMOTES a would-be `done` unless CR left review content on HEAD (#137).
That content verdict is reported in its own right as `reviewed_head_with_content`,
which is what separates "reviewed HEAD and found nothing" from "acknowledged HEAD
and did not read it" — the two the state alone collapses together.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal

from .signals import RunReceipt, extract_run_receipts

State = Literal[
    "done",
    "degraded",
    "pending",
    "in_progress",
    "rate_limited",
    "paused",
    "unavailable",
    "superseded",
    "none",
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
# CodeRabbit throttles under at least two different notices with different prose:
# the hourly "Rate limit exceeded", and the Fair Usage ADAPTIVE limit, whose body
# says "Review limit reached" and never contains the hourly wording. Matching only
# the hourly text read a throttled CR as `pending` for the whole window.
#
# The machine-emitted HTML marker is listed FIRST because it is the stable anchor:
# CR emits `<!-- ... rate limited by coderabbit.ai -->` around BOTH variants (the
# same class of anchor the code already trusts for "summarize by coderabbit.ai"),
# while the visible prose differs per variant and is plainly subject to change.
# Bodies are stored raw, so HTML comments are present in the blob. The prose
# alternatives stay as belt-and-braces if CR ever drops the marker.
#
# Those prose alternatives are LINE-ANCHORED, for the reason `_CR_DONE_MARKER`
# below already is. Unanchored they only chose among non-done transient states;
# #137 makes them decisional over whole bodies -- stripping marker evidence and
# demoting a completion -- so a body merely *quoting* the wording could withhold a
# genuine review, and re-withhold it on every push while the quote persisted. The
# accepted prefix is the set of decorations CR actually emits around the phrase --
# blockquote and heading marks, whitespace, and a non-ASCII lead-in -- and nothing
# else. The recorded layouts are `> ## Review limit reached`, `## Reviews paused`,
# `⚠️ Rate limit exceeded. Try again in 8 minutes`, and the bare phrase. Markdown
# LIST markers and quote characters are excluded on purpose, so a line of prose
# *about* the notice (`- Reviews paused`, `> "Review limit reached"`) is not read
# as one -- which a diff touching these very patterns invites. The machine marker
# stays unanchored: CR emits it as a bare HTML comment, not as quotable prose.
_CR_RATE_LIMIT = re.compile(
    r"auto-generated comment: rate limited by coderabbit\.ai"
    r"|^(?:[>#\s]|[^\x00-\x7f])*Rate limit exceeded\b"
    r"|^(?:[>#\s]|[^\x00-\x7f])*Review limit reached\b",
    re.I | re.M,
)
_CR_PAUSED = re.compile(
    r"review paused by coderabbit\.ai"
    r"|^(?:[>#\s]|[^\x00-\x7f])*Reviews paused\b",
    re.I | re.M,
)
_CR_DONE_ZERO = re.compile(r"(?im)^[>\s*_]*No actionable comments(?: were generated)?\b")
_CR_DONE_MARKER = re.compile(
    r"(?im)^[>\s*_]*(?:Actionable comments posted:\s*\d+"
    r"|No actionable comments(?: were generated)?)\b"
)
_CR_REVIEWING = re.compile(r"between\s+`?([0-9a-f]{7,40})`?\s+and\s+`?([0-9a-f]{7,40})`?", re.I)

_CR_CHECK_NAMES = re.compile(r"coderabbit", re.I)
# The marker on CR's summary comment -- the one comment CR rewrites IN PLACE on
# every push. That is what makes a notice inside it current rather than historical,
# and it is the only body allowed to DEMOTE a completed read (#137). CR's one-off
# replies are never rewritten, so a notice in one of those would stick forever.
_CR_SUMMARY = re.compile(r"auto-generated comment: summarize by coderabbit\.ai", re.I)

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


def receipt_is_valid(receipt: RunReceipt) -> bool:
    """Whether ``receipt`` is a review validly attributable to the model it names.

    Written as a **conjunction**, never as a negation, and that is load-bearing
    rather than stylistic -- do not "simplify" it into a mismatch check:

        state == "done"  AND  model is non-empty  AND  model == label

    A negative formulation ("no mismatch detected") PASSES when ``model`` is
    empty, because an absent value has nothing to mismatch against -- and empty
    is exactly what a receipt carries when the branch never recorded which model
    answered, i.e. one of the cases this exists to catch. An affirmative
    condition must be met rather than merely not-violated, so a broken or missing
    measurement fails closed. The general rule: a gate fires on an affirmative
    success condition, never only on an affirmative failure condition, because
    broken measurements are inevitable and the only controllable variable is what
    they say when they break.

    ``label != model`` is a HARD INVALIDATION, not an annotation (#106). A
    substituted seat's output is byte-indistinguishable from a genuine clean
    review -- the substitute emits the ordinary "no major issues" guide -- so
    "kimi-k3 reviewed and found nothing" and "glm-5.2 was silently substituted,
    lost its inline channel, and found nothing" read identically. Such a round
    should be RE-RUN rather than recorded, and a consumer must not count it as
    that seat's coverage.
    """
    return receipt.state == "done" and bool(receipt.model) and receipt.model == receipt.label


def _row(backend: str, state: State, head_reviewed: str | None, detail: str) -> dict:
    return {"backend": backend, "state": state, "head_reviewed": head_reviewed, "detail": detail}


def _sha_match(a: str | None, b: str | None) -> bool:
    """Prefix-compare two commit shas (CodeRabbit may abbreviate)."""
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def _cr_row(state: State, head_reviewed: str | None, detail: str, with_content: bool) -> dict:
    """Build a CodeRabbit row, always carrying the ``reviewed_head_with_content`` verdict.

    The key rides on EVERY CodeRabbit row, including the ones where it is trivially
    false, so a consumer can read it unconditionally. A key that appears only on
    interesting rows is one a consumer learns to skip.
    """
    row = _row("coderabbit", state, head_reviewed, detail)
    row["reviewed_head_with_content"] = with_content
    return row


def _cr_is_notice_body(body: str) -> bool:
    """Whether ``body`` is one of CodeRabbit's throttle notices rather than a review."""
    return bool(_CR_RATE_LIMIT.search(body) or _CR_PAUSED.search(body))


def _cr_demotion(notice_blob: str, head_reviewed: str | None, with_content: bool) -> dict | None:
    """Return the degraded row a live CodeRabbit notice imposes on a would-be ``done``.

    ``None`` when nothing demotes: either CR left content proving it read HEAD, or
    no live notice says the window was closed.

    The notice is allowed to OVERRIDE a completion signal, not merely to fill in a
    state when nothing else does (#137). A check-run that completes, or a review row
    submitted on HEAD, is emitted by CodeRabbit even when the Fair-Usage window
    denied it the review itself -- an empty acknowledgement is *positive* evidence
    of a review that did not happen, which is strictly worse than silence, because
    a gate that opens on ``done`` then starts a fix round (and merges) against a
    reviewer that never read the commit.

    ``with_content`` is the escape hatch that keeps this from being sticky: a CR
    that genuinely reviewed HEAD leaves a terminal marker or a non-empty review
    body, and that outranks any notice.
    """
    if with_content:
        return None
    if _CR_RATE_LIMIT.search(notice_blob):
        return _cr_row(
            "rate_limited",
            head_reviewed,
            "CR signalled completion on HEAD but its live summary still carries a "
            "limit notice and nothing on HEAD carries review content",
            with_content,
        )
    if _CR_PAUSED.search(notice_blob):
        return _cr_row(
            "paused",
            head_reviewed,
            "CR signalled completion on HEAD but its live summary still carries a "
            "pause notice and nothing on HEAD carries review content",
            with_content,
        )
    return None


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

    A throttle notice is surfaced whether or not HEAD has been scanned (#19). CR's
    rate-limit notice lives in its *summary* comment, which also carries the
    walkthrough range line and is rewritten to the newest HEAD on every push, so
    ``walk_on_head`` stays true for the entire throttle window: masking the notice
    once HEAD is named is the steady state while throttled, not a transient race.
    The gate stays closed either way, but ``rate_limited``/``paused`` carry a
    recovery path (a resume nudge, a longer wait) that ``in_progress``/``pending``
    do not, and a consumer that cannot tell them apart burns its full unresponsive
    timeout instead. ``_CR_RATE_LIMIT`` matches CR's machine-emitted
    "rate limited by coderabbit.ai" marker as well as the prose, because the
    hourly and Fair-Usage-adaptive notices word themselves differently (#86) and
    matching only the hourly text read a throttled CR as ``pending``.

    A live notice can also **demote** a completion signal, not only supply a state
    where none exists (#137). Under Fair Usage, CodeRabbit still completes its
    check-run and still submits a review row on HEAD -- with an empty body and no
    walkthrough -- so both of this function's ``done`` doors open on a commit it
    never read. Worse, the summary comment is rewritten in place, so its range line
    is bumped to the new HEAD while the *previous* round's terminal marker is still
    sitting in it, which opens a third door through the marker itself. The
    discriminator is therefore CONTENT, reported as ``reviewed_head_with_content``:
    a terminal marker in a HEAD-scoped body that is not itself a notice, or a
    non-empty CR review body on HEAD. When that is absent and CR's live summary
    still carries a limit or pause notice, the row reports ``rate_limited`` /
    ``paused`` instead of ``done``. The two notices keep their own states because
    they need different recoveries (credits or a wait vs. ``@coderabbitai resume``).

    Demotion is scoped to CR's **summary** comment because that is the one comment
    CR rewrites in place, so a notice in it describes the window now; a one-off CR
    reply is never rewritten and a notice there would demote every later round
    forever. When no summary comment is observable at all, CR's live issue comments
    are used instead -- with no in-place-edited anchor, trusting the notice is the
    fail-safe direction.

    Content only overrides the notice; the notice does not gate content. An empty
    APPROVED review on HEAD with NO live notice stays ``done`` (issue #18), just
    with ``reviewed_head_with_content`` false: the key exposes the ambiguity on
    every row, while the state changes only when a notice positively says the
    window was closed. An in-progress check-run likewise still reports
    ``in_progress`` rather than being demoted -- the observed recovery sequence is
    ``rate_limited -> in_progress -> done``, so an active scan outranks the notice
    that preceded it.
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
        return _cr_row("none", None, "no CodeRabbit activity", False)

    # No all-bodies blob: every transient-state read is scoped to CR's live issue
    # comments (see below), and the completion reads use `head_blob`/`bodies`
    # directly. Reintroducing one would re-open the staleness hole.
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

    head_bodies = (
        # Any range line in the body may be the one covering HEAD, so this asks
        # about all of them rather than only the first (#101). Checking just the
        # first would drop a body whose later range covers HEAD, and with it the
        # terminal marker that body carries.
        [b for b in bodies if any(_sha_match(h, head_sha) for h in _range_heads(b))]
        + [r.get("body", "") or "" for r in cr_reviews if r.get("commit_id") == head_sha]
        # A review on HEAD makes CR's LIVE summary describe HEAD, so it is admitted
        # even when its own range line does not name HEAD. Only the summary: it is
        # the one comment CR rewrites in place, and this list is the sole source of
        # the `with_content` escape hatch that disables the #137 demotion. A stale
        # one-off CR comment carrying an old terminal marker would otherwise vouch
        # for a HEAD CR never opened, which is the exact failure the demotion exists
        # to catch (CodeRabbit finding, round 1). The scoping mirrors `notice_blob`
        # below: absent an in-place-edited anchor, neither side trusts a one-off.
        + ([b for b in cr_issue_bodies if _CR_SUMMARY.search(b)] if review_on_head else [])
    )
    head_blob = "\n".join(head_bodies)
    # A body that IS a limit/pause notice cannot also be evidence that CR read
    # HEAD. The summary comment is rewritten in place, so under Fair Usage it can
    # carry a range line already bumped to the new HEAD, the PREVIOUS round's
    # terminal marker, and the live notice, all at once -- and the marker would
    # then vouch for a commit CR never opened (#137). Excluding notice bodies
    # costs nothing on a genuine review: a real post-review summary reports its
    # allowance as a "Limit details:" line, which these patterns do not match.
    marker_on_head = any(
        _CR_DONE_MARKER.search(b) for b in head_bodies if not _cr_is_notice_body(b)
    )
    # A submitted review whose body has actual prose is independent evidence: CR
    # writes one when it reviewed, and the empty acknowledgement is precisely the
    # artifact of a Fair-Usage round. The notice exclusion applies here for the
    # same reason it applies to the marker above -- a body that IS a notice cannot
    # also be evidence that CR read HEAD -- and this suite already establishes that
    # CR submits reviews whose body is the throttle notice (the stale-notice tests
    # below are built on exactly that shape). Anchored to HEAD at submission time,
    # the live form of it is a notice-body review on the CURRENT head, which would
    # otherwise satisfy `with_content`, short-circuit `_cr_demotion`, and report
    # `done` for a commit CR never read -- through the hatch that is supposed to
    # be the discriminator (agentic reviewer, round 2).
    body_on_head = any(
        (r.get("body") or "").strip()
        for r in cr_reviews
        if r.get("commit_id") == head_sha and not _cr_is_notice_body(r.get("body") or "")
    )
    with_content = marker_on_head or body_on_head
    summary_blob = "\n".join(b for b in cr_issue_bodies if _CR_SUMMARY.search(b))
    notice_blob = summary_blob or issue_blob

    if check is not None:
        status = (check.get("status") or "").lower()
        if status != "completed":
            return _cr_row(
                "in_progress", head_sha, f"check-run still {status or 'pending'}", with_content
            )
        demoted = _cr_demotion(notice_blob, head_sha, with_content)
        if demoted is not None:
            return demoted
        conclusion = (check.get("conclusion") or "").lower()
        zero = bool(_CR_DONE_ZERO.search("\n".join(bodies)))
        return _cr_row(
            "done",
            head_sha,
            (
                "no actionable comments"
                if zero
                else f"check-run completed ({conclusion or 'neutral'})"
            ),
            with_content,
        )

    if not (walk_on_head or review_on_head):
        # `issue_blob`, not `blob`: every transient state is read from CR's LIVE
        # status comments only. A submitted review body is never rewritten, so a
        # throttle notice inside one is permanently stale -- sourcing these from
        # `blob` let a PR that was throttled once keep reporting `rate_limited`
        # on every later HEAD that had no walkthrough. The in-progress check
        # below was already scoped this way; the other two were not, and the
        # inconsistency predates #19/#86.
        if _CR_RATE_LIMIT.search(issue_blob):
            return _cr_row(
                "rate_limited", walk_head, "rate-limit notice; HEAD not yet scanned", with_content
            )
        if _CR_PAUSED.search(issue_blob):
            return _cr_row(
                "paused", walk_head, "reviews paused; HEAD not yet scanned", with_content
            )
        if _CR_IN_PROGRESS.search(issue_blob):
            return _cr_row("in_progress", walk_head, "review in progress", with_content)
        return _cr_row(
            "pending", walk_head, "neither walkthrough nor review covers the HEAD", with_content
        )

    if not review_on_head and not marker_on_head:
        # A throttle notice is asked about BEFORE `in_progress`/`pending`, and it
        # is asked here rather than only in the not-yet-scanned branch above
        # (#19). CR's throttle notice lives in its summary comment, which also
        # carries the walkthrough range line and is rewritten to the newest HEAD
        # on every push -- so `walk_on_head` stays true for the whole throttle
        # window and this branch is the STEADY STATE while throttled, not a
        # transient race. Reporting `in_progress`/`pending` here loses the one
        # fact a consumer can act on: the wait is a cooldown with its own
        # recovery path (a resume nudge, a longer window), not an active scan.
        #
        # The sticky-guard still holds: this is reached only when CR has NOT
        # completed HEAD (no terminal marker, no submitted review on HEAD), so an
        # earlier throttle cannot mask a later completed scan. Scoped to
        # `issue_blob` -- CR's live status comments -- like the in-progress check
        # below, so a stale notice in an old review body cannot resurrect it.
        if _CR_RATE_LIMIT.search(issue_blob):
            return _cr_row(
                "rate_limited",
                head_sha,
                "rate-limit notice on this PR and HEAD not yet completed",
                with_content,
            )
        if _CR_PAUSED.search(issue_blob):
            return _cr_row(
                "paused",
                head_sha,
                "reviews paused and HEAD not yet completed",
                with_content,
            )
        if _CR_IN_PROGRESS.search(issue_blob):
            return _cr_row(
                "in_progress",
                head_sha,
                "review in progress (CR named HEAD and reports an active scan)",
                with_content,
            )
        return _cr_row(
            "pending",
            head_sha,
            "walkthrough range line covers HEAD but CR shows no completion marker, "
            "submitted review, or in-progress check-run yet",
            with_content,
        )

    head_reviewed = walk_head if walk_on_head else head_sha
    demoted = _cr_demotion(notice_blob, head_reviewed, with_content)
    if demoted is not None:
        return demoted
    zero = bool(_CR_DONE_ZERO.search(head_blob))
    return _cr_row(
        "done",
        head_reviewed,
        "no actionable comments" if zero else "scanned HEAD (any findings are inline)",
        with_content,
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
    configured_labels: Iterable[str] | None = None,
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
    - ``superseded`` -- the receipt's label is not among ``configured_labels``,
      so the seat that produced it was renamed or removed from ``.fuko.toml``.

    ``superseded`` inverts the usual fail-safe direction, and that is the whole
    point of it. A receipt is anchored to the commit it reviewed, so a seat that
    no longer exists keeps producing a row anchored to a HEAD that can never
    recur -- ``pending`` forever, because no future run will ever post a receipt
    under a label that is no longer configured (#116). ``pending`` is the state a
    consumer WAITS on, so a strict "proceed when every fuko row is ``done``" gate
    would wait forever on a deleted seat. The danger here is therefore waiting
    forever, not merging early: a label absent from the current config must NOT
    gate. ``superseded`` is deliberately kept OUT of :data:`DEGRADED_STATES` too,
    so it neither blocks a merge nor triggers escalation -- a retired seat is not
    a coverage loss to react to, it is a row to disregard.

    The cross-reference happens ONLY when the caller supplies ``configured_labels``
    (the CLI, which already loads ``.fuko.toml``, passes the ``provider/name`` of
    every configured entry). When it is ``None`` -- config unreadable, or a caller
    that has no config -- today's behavior is kept and no receipt is reclassified,
    because dropping a genuinely-pending row on a failed config read would be the
    unsafe direction: it would let a merge proceed past a seat that really has not
    reviewed HEAD. Absent config, err toward the gating ``pending``; only a
    positively-confirmed absence from a loaded config downgrades to ``superseded``.

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
    # Normalize an empty or blank-only collection to None. An empty `configured`
    # set makes `label not in configured` true for EVERY receipt, so it would
    # supersede all of them -- dropping genuinely-pending gate rows and letting a
    # merge proceed past an unreviewed seat, the one unsafe direction this feature
    # exists to avoid. Only a positively-loaded, non-empty label set may
    # cross-reference; an empty one is unavailable-config, treated like None.
    configured = None
    if configured_labels is not None:
        configured = {c.strip() for c in configured_labels if c.strip()} or None
    latest: dict[str, RunReceipt] = {}
    for comment in issue_comments:
        login = ((comment.get("user") or {}).get("login") or "").strip()
        if not _receipt_author_allowed(login, allowed):
            continue
        for receipt in extract_run_receipts(comment.get("body", "") or ""):
            latest[receipt.label] = receipt

    rows: list[dict] = []
    for label, receipt in latest.items():
        dead: list[str] = []
        on_head = _sha_match(receipt.head_sha, head_sha) if receipt.head_sha else False
        # Config membership is asked BEFORE staleness (#116). A removed/renamed
        # seat's receipt is always off-HEAD -- no future run posts under its
        # label -- so the staleness branch below would report it `pending`
        # forever. Reclassify it as `superseded` first so a deleted seat can
        # never masquerade as a not-yet-reviewed one. Guarded on `configured`
        # being non-None: an absent OR empty config (both normalized to None
        # above) keeps every row, erring toward the gating `pending` rather than
        # silently dropping a real seat.
        if configured is not None and label not in configured:
            row = _row(
                f"fuko:{label}",
                "superseded",
                receipt.head_sha or None,
                "label is no longer configured (seat renamed or removed)",
            )
            row["role"] = receipt.role
            row["valid"] = False
            row["label"] = label
            row["model"] = receipt.model
            # `review_backend`, NOT `backend`: `_row` already owns `backend` for the
            # display id (`fuko:<label>`), which this function also sorts on (#99).
            row["review_backend"] = receipt.backend
            if receipt.channels:
                row["channels"] = dict(receipt.channels)
            rows.append(row)
            continue
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
            dead[:] = sorted(
                f"{name} {value}" for name, value in receipt.channels.items() if value != "done"
            )
            if not receipt_is_valid(receipt):
                # VOID, not an annotation: this branch did not review as the model
                # it is named for, so its output is not that seat's coverage and
                # the round wants re-running (#106). Reported as a degraded state
                # rather than a flag on a `done` row, because a consumer gating on
                # `state == "done"` would otherwise merge on a review that never
                # validly happened.
                state = "degraded"
                answered = receipt.model or "an unrecorded model"
                detail = f"VOID: branch is labelled {label} but {answered} answered"
            elif dead:
                state = "degraded"
                detail = f"reviewed HEAD, but reduced coverage: {', '.join(dead)}"
            else:
                state = "done"
                detail = "reviewed HEAD"
        else:
            state = "in_progress"
            detail = "started on HEAD, no outcome recorded yet"
        row = _row(f"fuko:{label}", state, receipt.head_sha or None, detail)
        row["role"] = receipt.role
        # BOTH fields are printed, and the verdict is precomputed. A consumer told
        # to compare two fields itself will eventually stop doing so (#106); one
        # told to read `valid` cannot forget to.
        row["valid"] = on_head and receipt_is_valid(receipt) and not dead
        row["label"] = label
        row["model"] = receipt.model
        # `review_backend`, NOT `backend`: `_row` owns `backend` (the display id
        # `fuko:<label>`, also this function's sort key). Surfaces which driver
        # produced the run so two harnesses are distinguishable receipts-only (#99).
        row["review_backend"] = receipt.backend
        if receipt.promoted:
            row["promoted"] = True
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
    configured_labels: Iterable[str] | None = None,
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
    ``configured_labels`` is likewise forwarded so a receipt whose seat was
    renamed or removed reports ``superseded`` rather than an unreachable
    ``pending`` (#116); it is ``None`` when the caller has no config to supply.
    """
    rows = [
        coderabbit_state(head_sha, issue_comments, reviews, check_runs),
        copilot_state(head_sha, reviews, issue_comments),
    ]
    if include_fuko:
        rows.extend(
            fuko_states(
                head_sha,
                issue_comments,
                allowed_authors=allowed_authors,
                configured_labels=configured_labels,
            )
        )
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
