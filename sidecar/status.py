"""Per-reviewer review STATE on a PR's current HEAD — the normalized "done" signal.

`fuko status` answers *has each external reviewer finished reviewing the current
HEAD?* for the bots a review loop gates on (CodeRabbit, Copilot). It is the state
counterpart to `fuko signals` (which answers *what did they find?*). Only
**observable** artifacts are read — fuko makes no time judgments like
"unresponsive"; a consumer applies its own timeout to a `pending` state.

CodeRabbit's state lives in its in-place-edited **walkthrough** issue comment (the
reviews endpoint is unreliable for CR — stale or absent commit ids on zero-finding
runs); Copilot's state is its latest review's `commit_id`.
"""

from __future__ import annotations

import re
from typing import Literal

State = Literal["done", "pending", "in_progress", "rate_limited", "paused", "none"]

_CR_LOGIN = "coderabbitai[bot]"
_COPILOT_LOGINS = {"copilot", "copilot-pull-request-reviewer[bot]"}

_CR_IN_PROGRESS = re.compile(r"review in progress|Currently processing new changes", re.I)
_CR_RATE_LIMIT = re.compile(r"Rate limit exceeded", re.I)
_CR_PAUSED = re.compile(r"Reviews paused|review paused by coderabbit\.ai", re.I)
_CR_DONE_ZERO = re.compile(r"No actionable comments were generated", re.I)
# The walkthrough's "Reviewing files … between <base> and <HEAD>" line; group 2 is
# the commit CodeRabbit actually scanned.
_CR_REVIEWING = re.compile(r"between\s+`?([0-9a-f]{7,40})`?\s+and\s+`?([0-9a-f]{7,40})`?", re.I)


def _row(backend: str, state: State, head_reviewed: str | None, detail: str) -> dict:
    return {"backend": backend, "state": state, "head_reviewed": head_reviewed, "detail": detail}


def _sha_match(a: str | None, b: str | None) -> bool:
    """Prefix-compare two commit shas (CodeRabbit may abbreviate)."""
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def coderabbit_state(head_sha: str, issue_comments: list[dict], reviews: list[dict]) -> dict:
    """Derive CodeRabbit's state on ``head_sha`` from its walkthrough comment and reviews.

    Two independent "scanned HEAD" signals are combined, because neither alone is
    reliable: the walkthrough's ``Reviewing files … between … and <HEAD>`` range line
    (absent on some runs) OR a submitted CR review whose ``commit_id`` is the HEAD
    (stale or absent on zero-finding runs).
    """
    bodies = [
        c.get("body", "") or ""
        for c in issue_comments
        if (c.get("user") or {}).get("login", "").lower() == _CR_LOGIN
    ]
    cr_reviews = [r for r in reviews if (r.get("user") or {}).get("login", "").lower() == _CR_LOGIN]
    if not bodies and not cr_reviews:
        return _row("coderabbit", "none", None, "no CodeRabbit activity")

    blob = "\n".join(bodies)
    walkthrough = next((b for b in bodies if _CR_REVIEWING.search(b)), "")
    m = _CR_REVIEWING.search(walkthrough)
    walk_head = m.group(2) if m else None
    walk_on_head = _sha_match(walk_head, head_sha)
    review_on_head = any(r.get("commit_id") == head_sha for r in cr_reviews)

    # Transient states only matter while the current HEAD hasn't been scanned —
    # the markers are sticky, so an earlier rate-limit/pause must not mask a later
    # completed scan.
    if not (walk_on_head or review_on_head):
        if _CR_RATE_LIMIT.search(blob):
            return _row(
                "coderabbit", "rate_limited", walk_head, "rate-limit notice; HEAD not yet scanned"
            )
        if _CR_PAUSED.search(blob):
            return _row("coderabbit", "paused", walk_head, "reviews paused; HEAD not yet scanned")
        if _CR_IN_PROGRESS.search(blob):
            return _row("coderabbit", "in_progress", walk_head, "review in progress")
        return _row(
            "coderabbit", "pending", walk_head, "neither walkthrough nor review covers the HEAD"
        )

    zero = bool(_CR_DONE_ZERO.search(blob))
    return _row(
        "coderabbit",
        "done",
        walk_head if walk_on_head else head_sha,
        "no actionable comments" if zero else "scanned HEAD (any findings are inline)",
    )


def copilot_state(head_sha: str, reviews: list[dict]) -> dict:
    """Derive Copilot's state from its latest review's commit id (reliable for Copilot)."""
    cps = [r for r in reviews if (r.get("user") or {}).get("login", "").lower() in _COPILOT_LOGINS]
    if not cps:
        return _row("copilot", "none", None, "no Copilot review")
    on_head = [r for r in cps if r.get("commit_id") == head_sha]
    if on_head:
        return _row("copilot", "done", head_sha, f"review on HEAD ({on_head[-1].get('state')})")
    return _row(
        "copilot",
        "pending",
        cps[-1].get("commit_id"),
        "latest Copilot review is on an older commit",
    )


def reviewer_states(head_sha: str, issue_comments: list[dict], reviews: list[dict]) -> list[dict]:
    """Return the normalized state of each gated reviewer (CodeRabbit, Copilot)."""
    return [
        coderabbit_state(head_sha, issue_comments, reviews),
        copilot_state(head_sha, reviews),
    ]
