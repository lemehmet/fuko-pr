"""Map each vendor's PR comments into canonical Review Signals (egress).

Pure parsing functions, one family per vendor, kept separate from the GitHub I/O
that fetches/edits comments. A consumer reads one schema (:class:`ReviewSignal`)
instead of sniffing each vendor's ad-hoc markdown.

PR-Agent declares structured metadata (a ``[label, importance: N]`` tag), so its
severity/category are ``declared``; free-form reviewers (e.g. Copilot) get a
best-effort ``inferred`` mapping. Detection is by comment *format*, not author --
PR-Agent posts under whatever token runs it (an app bot in CI, a human PAT locally).
"""

from __future__ import annotations

import re

from .signals import Category, ReviewSignal, make_id

_PRAGENT_PREFIX = "**Suggestion:**"
_LABEL_RE = re.compile(r"\[([^,\]]+),\s*importance:\s*(\d+)\]")

_PRAGENT_CATEGORY: dict[str, Category] = {
    "security": "security",
    "performance": "perf",
    "possible issue": "bug",
    "possible bug": "bug",
    "bug": "bug",
    "best practice": "style",
    "maintainability": "design",
    "enhancement": "design",
    "typo": "docs",
}


def _severity_from_importance(n: int) -> str:
    """Map PR-Agent's 1-10 importance onto the Review Signal severity scale."""
    if n >= 9:
        return "critical"
    if n >= 7:
        return "high"
    if n >= 4:
        return "medium"
    return "low"


def is_pragent_comment(body: str) -> bool:
    """Return whether ``body`` looks like a PR-Agent inline suggestion."""
    return (body or "").lstrip().startswith(_PRAGENT_PREFIX)


def pragent_signal(comment: dict, model: str = "") -> ReviewSignal:
    """Map one PR-Agent inline review comment (GitHub API shape) to a Review Signal."""
    body = comment.get("body", "") or ""
    path = comment.get("path")
    line = comment.get("start_line") or comment.get("line")
    end_line = comment.get("line") if comment.get("start_line") else None

    match = _LABEL_RE.search(body)
    if match:
        category = _PRAGENT_CATEGORY.get(match.group(1).strip().lower(), "bug")
        severity = _severity_from_importance(int(match.group(2)))
        severity_source = "declared"
    else:
        category, severity, severity_source = "bug", "medium", "inferred"

    head = body.split(_PRAGENT_PREFIX, 1)[-1]
    title = _LABEL_RE.split(head)[0].split("```")[0].strip()[:200]

    return ReviewSignal(
        id=make_id(path or "", str(line or ""), title),
        file=path,
        line=line,
        end_line=end_line,
        severity=severity,
        severity_source=severity_source,
        category=category,
        title=title,
        body=body,
        suggestion="```suggestion" in body,
        thread_url=comment.get("html_url"),
        backend="pr-agent",
        model=model,
    )


def pragent_signals(comments: list[dict], model: str = "") -> list[dict]:
    """Return ``(comment, signal)`` pairs for every PR-Agent-formatted comment."""
    return [
        {"comment": c, "signal": pragent_signal(c, model)}
        for c in comments
        if is_pragent_comment(c.get("body", ""))
    ]


_COPILOT_LOGINS = {"copilot", "copilot-pull-request-reviewer[bot]"}
_SECURITY_RE = re.compile(r"secur|vulnerab|inject|xss|csrf|secret|password|creden", re.I)
_PERF_RE = re.compile(r"perform|latency|memory leak|n\+1|\bslow\b|\bO\(", re.I)


def is_copilot_comment(comment: dict) -> bool:
    """Return whether ``comment`` was authored by GitHub Copilot's reviewer."""
    login = (comment.get("user") or {}).get("login", "")
    return login.lower() in _COPILOT_LOGINS


def _infer_category(text: str) -> Category:
    """Best-effort category from free-form text (used when none is declared)."""
    if _SECURITY_RE.search(text):
        return "security"
    if _PERF_RE.search(text):
        return "perf"
    return "bug"


def copilot_signal(comment: dict) -> ReviewSignal:
    """Map one Copilot inline review comment (free-form) to a Review Signal."""
    body = comment.get("body", "") or ""
    path = comment.get("path")
    line = comment.get("start_line") or comment.get("line")
    end_line = comment.get("line") if comment.get("start_line") else None
    title = body.strip().split("\n", 1)[0][:200]
    return ReviewSignal(
        id=make_id(path or "", str(line or ""), title),
        file=path,
        line=line,
        end_line=end_line,
        severity="medium",
        severity_source="inferred",
        category=_infer_category(body),
        title=title,
        body=body,
        suggestion="```suggestion" in body,
        thread_url=comment.get("html_url"),
        backend="copilot",
        model="",
    )


def collect_signals(comments: list[dict], model: str = "") -> list[ReviewSignal]:
    """Normalize a PR's comments across every recognized reviewer into one list.

    Dispatch is per comment: PR-Agent by format, Copilot by author. Comments from
    no recognized reviewer are skipped. (CodeRabbit support lands once real samples
    are available.)
    """
    out: list[ReviewSignal] = []
    for c in comments:
        if is_pragent_comment(c.get("body", "")):
            out.append(pragent_signal(c, model))
        elif is_copilot_comment(c):
            out.append(copilot_signal(c))
    return out
