"""Per-reviewer review STATE on a PR's current HEAD — the normalized "done" signal.

`fuko status` answers *has each external reviewer finished reviewing the current
HEAD?* for the bots a review loop gates on (CodeRabbit, Copilot). It is the state
counterpart to `fuko signals` (which answers *what did they find?*). Only
**observable** artifacts are read — fuko makes no time judgments like
"unresponsive"; a consumer applies its own timeout to a `pending` state.

CodeRabbit's completion is taken from its **check-run** on the HEAD commit when one
is present ("Review in progress" → completed) — the only signal that doesn't race
the inline comments (issue #17). Its walkthrough issue comment is the fallback when
no check-run is observable, and only a *terminal* walkthrough marker (a completion
line, not merely the "Reviewing files … between …" range that CR posts up front)
counts as done there. Copilot's state is its latest review's `commit_id`.
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
_CR_DONE_ZERO = re.compile(r"(?im)^[>\s*_]*No actionable comments(?: were generated)?\b")
_CR_DONE_MARKER = re.compile(
    r"(?im)^[>\s*_]*(?:Actionable comments posted:\s*\d+"
    r"|No actionable comments(?: were generated)?)\b"
)
# The walkthrough's "Reviewing files … between <base> and <HEAD>" line; group 2 is
# the commit CodeRabbit actually scanned.
_CR_REVIEWING = re.compile(r"between\s+`?([0-9a-f]{7,40})`?\s+and\s+`?([0-9a-f]{7,40})`?", re.I)

_CR_CHECK_NAMES = re.compile(r"coderabbit", re.I)


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
    comment *or* its review body depending on CR's mode, so both are searched. Two
    independent "scanned HEAD" signals satisfy the SHA match: a range line whose end is
    HEAD (absent on some runs) OR a submitted CR review whose ``commit_id`` is HEAD
    (stale or absent on zero-finding runs). The marker is required on text tied to the
    *current* HEAD — a range line ending at HEAD, an on-HEAD review body, or (once CR
    has reviewed HEAD) its in-place-edited summary — so a prior HEAD's stale marker can
    no longer report done early.
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
    walkthrough = next((b for b in bodies if _CR_REVIEWING.search(b)), "")
    m = _CR_REVIEWING.search(walkthrough)
    walk_head = m.group(2) if m else None
    walk_on_head = any(
        (mm := _CR_REVIEWING.search(b)) and _sha_match(mm.group(2), head_sha) for b in bodies
    )
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

    head_blob = "\n".join(
        [b for b in bodies if (mm := _CR_REVIEWING.search(b)) and _sha_match(mm.group(2), head_sha)]
        + [r.get("body", "") or "" for r in cr_reviews if r.get("commit_id") == head_sha]
        + (cr_issue_bodies if review_on_head else [])
    )
    if not _CR_DONE_MARKER.search(head_blob):
        return _row(
            "coderabbit",
            "in_progress",
            walk_head if walk_on_head else head_sha,
            "HEAD scanned but no completion marker yet (inline comments may still be posting)",
        )

    zero = bool(_CR_DONE_ZERO.search(head_blob))
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


def reviewer_states(
    head_sha: str,
    issue_comments: list[dict],
    reviews: list[dict],
    check_runs: list[dict] | None = None,
) -> list[dict]:
    """Return the normalized state of each gated reviewer (CodeRabbit, Copilot).

    ``check_runs`` are the check-runs fetched for ``head_sha``; CodeRabbit's completion
    is read from its own check there when present (issue #17). Optional so existing
    callers and tests that only have comment/review data still work via the fallback.
    """
    return [
        coderabbit_state(head_sha, issue_comments, reviews, check_runs),
        copilot_state(head_sha, reviews),
    ]
