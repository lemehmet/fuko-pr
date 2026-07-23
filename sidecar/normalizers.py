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

import html
import re

from .signals import Category, ReviewSignal, extract_markers, make_id

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


_CODERABBIT_LOGIN = "coderabbitai[bot]"
# A finding's first classification line, e.g. "_⚠️ Potential issue_ | _🔴 Critical_"
# (an optional effort token may follow). Author alone is not enough -- CodeRabbit
# also posts chat replies and rate-limit notices, which carry no classification.
_CR_CLASS_RE = re.compile(r"^_[^_]+_\s*\|\s*_[^_]+_.*$", re.M)
_CR_TOKEN_RE = re.compile(r"_([^_]+)_")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
_CR_SEVERITY = (
    ("critical", "critical"),
    ("major", "high"),
    ("minor", "medium"),
    ("trivial", "low"),
)


def is_coderabbit_comment(comment: dict) -> bool:
    """Return whether ``comment`` was authored by CodeRabbit."""
    return (comment.get("user") or {}).get("login", "").lower() == _CODERABBIT_LOGIN


def _cr_classification(body: str) -> list[str] | None:
    """Return the ``[category, severity, ...]`` tokens of a CodeRabbit finding, if any."""
    m = _CR_CLASS_RE.search(body or "")
    return _CR_TOKEN_RE.findall(m.group(0)) if m else None


def is_coderabbit_finding(body: str) -> bool:
    """Return whether a CodeRabbit comment body is an actual finding (vs chat/notice)."""
    return _cr_classification(body) is not None


def _cr_category(token: str) -> Category:
    t = token.lower()
    if "security" in t:
        return "security"
    if "perf" in t:
        return "perf"
    if "nitpick" in t:
        return "style"
    if "refactor" in t:
        return "design"
    if "typo" in t:
        return "docs"
    return "bug"


def coderabbit_signal(comment: dict) -> ReviewSignal:
    """Map one CodeRabbit inline finding to a Review Signal (severity/category declared)."""
    body = comment.get("body", "") or ""
    path = comment.get("path")
    line = comment.get("start_line") or comment.get("line")
    end_line = comment.get("line") if comment.get("start_line") else None

    tokens = _cr_classification(body) or []
    category = _cr_category(tokens[0]) if tokens else "bug"
    severity = next(
        (s for kw, s in _CR_SEVERITY if len(tokens) > 1 and kw in tokens[1].lower()), None
    )
    severity_source = "declared" if severity else "inferred"
    severity = severity or "medium"

    bold = _BOLD_RE.search(body)
    title = (bold.group(1).strip() if bold else body.strip().split("\n", 1)[0])[:200]

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
        suggestion="```suggestion" in body or "Suggested fix" in body,
        thread_url=comment.get("html_url"),
        backend="coderabbit",
        model="",
    )


_GUIDE_HEADER = "PR Reviewer Guide"
_GUIDE_SECURITY_CELL = "<strong>Security concerns</strong>"
_GUIDE_FOCUS_CELL = "<strong>Recommended focus areas for review</strong>"
_TD_RE = re.compile(r"<td>(.*?)</td>", re.S | re.I)
_DETAILS_RE = re.compile(r"<details>(.*?)</details>", re.S | re.I)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S | re.I)
_STRONG_RE = re.compile(r"<strong>(.*?)</strong>", re.S | re.I)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_PATH_LINE_RE = re.compile(r"\b([\w./-]+\.[A-Za-z]\w*):(\d+)\b")


def _html_to_text(fragment: str) -> str:
    """Flatten an HTML fragment to plain text (``<br>`` -> newline, tags dropped)."""
    text = _BR_RE.sub("\n", fragment)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_guide_comment(body: str) -> bool:
    """Return whether ``body`` looks like PR-Agent's "PR Reviewer Guide" issue comment."""
    return _GUIDE_HEADER in (body or "")


def guide_signals(comment: dict, model: str = "") -> list[ReviewSignal]:
    """Parse a PR-Agent "PR Reviewer Guide" issue comment into Review Signals.

    The guide is an HTML table with an optional "Security concerns" cell (one
    ``security`` signal when it carries free text; the "No security concerns
    identified" variant emits nothing) and an optional "Recommended focus areas
    for review" cell holding one ``<details>`` block per focus area (one ``bug``
    signal each; the "No major issues detected" variant emits nothing). Severity
    is always ``medium``/``inferred`` — the guide declares none.

    Ids are deterministic (``make_id("guide", <comment id>, title)``), so
    re-parsing the same comment re-derives the same signals — that is what keeps
    marker injection into the guide body idempotent. Tolerant by contract: a
    malformed or unexpected body yields ``[]`` and never raises.
    """
    try:
        body = comment.get("body", "") or ""
        if _GUIDE_HEADER not in body:
            return []
        cid = str(comment.get("id", ""))
        url = comment.get("html_url")
        out: list[ReviewSignal] = []
        for cell in _TD_RE.findall(body):
            if _GUIDE_SECURITY_CELL in cell:
                out.extend(_guide_security_signals(cell, cid, url, model))
            elif _GUIDE_FOCUS_CELL in cell:
                out.extend(_guide_focus_signals(cell, cid, url, model))
        return out
    except Exception:
        return []


def _guide_security_signals(cell: str, cid: str, url: str | None, model: str):
    """Map the guide's "Security concerns" cell to (at most) one security signal."""
    content = cell.split(_GUIDE_SECURITY_CELL, 1)[1]
    text = _html_to_text(content)
    if not text:
        return []
    strong = _STRONG_RE.search(content)
    title = _html_to_text(strong.group(1)).rstrip(":") if strong else ""
    title = title or "Security concern"
    return [
        ReviewSignal(
            id=make_id("guide", cid, title),
            severity="medium",
            severity_source="inferred",
            category="security",
            title=title[:200],
            body=text,
            thread_url=url,
            backend="pr-agent",
            model=model,
        )
    ]


def _guide_focus_signals(cell: str, cid: str, url: str | None, model: str):
    """Map each ``<details>`` focus area in the guide's focus cell to a signal."""
    out: list[ReviewSignal] = []
    for block in _DETAILS_RE.findall(cell):
        summary = _SUMMARY_RE.search(block)
        if not summary:
            continue
        summary_html = summary.group(1)
        strong = _STRONG_RE.search(summary_html)
        if strong:
            title = _html_to_text(strong.group(1))
            desc_html = re.sub(r"^\s*</a>", "", summary_html[strong.end() :])
        else:
            title = _html_to_text(summary_html).split("\n", 1)[0]
            desc_html = summary_html
        if not title:
            continue
        desc = _html_to_text(desc_html)
        # The diff-anchor href fragment cannot yield a path; only a literal
        # ``path:line`` in the text can locate the finding.
        loc = _PATH_LINE_RE.search(desc)
        out.append(
            ReviewSignal(
                id=make_id("guide", cid, title),
                file=loc.group(1) if loc else None,
                line=int(loc.group(2)) if loc else None,
                severity="medium",
                severity_source="inferred",
                category="bug",
                title=title[:200],
                body=desc,
                thread_url=url,
                backend="pr-agent",
                model=model,
            )
        )
    return out


def _prefer_marker(base: ReviewSignal, body: str) -> ReviewSignal:
    """If ``body`` carries a fuko-signal marker, trust it over the fresh parse.

    The marker was written at review time and is authoritative for the machine
    fields (notably ``model``, ``id``, ``severity``) — re-deriving them here would
    instead reflect whoever runs ``fuko signals`` and with which config. The marker
    excludes the human-facing ``title``/``body``, so those are kept from ``base``.
    """
    markers = extract_markers(body)
    if not markers:
        return base
    marked = markers[0]
    marked.title = base.title
    marked.body = base.body
    return marked


def collect_signals(comments: list[dict], model: str = "") -> list[ReviewSignal]:
    """Normalize a PR's comments across every recognized reviewer into one list.

    Dispatch is per comment: PR-Agent by format, Copilot by author, CodeRabbit by
    author *and* the presence of a finding classification (its chat replies and
    rate-limit notices are skipped). Unrecognized comments are skipped. When a
    comment carries a fuko-signal marker, its review-time fields take precedence
    (see :func:`_prefer_marker`).
    """
    out: list[ReviewSignal] = []
    for c in comments:
        body = c.get("body", "") or ""
        if is_pragent_comment(body):
            out.append(_prefer_marker(pragent_signal(c, model), body))
        elif is_copilot_comment(c):
            out.append(_prefer_marker(copilot_signal(c), body))
        elif is_coderabbit_comment(c) and is_coderabbit_finding(body):
            out.append(_prefer_marker(coderabbit_signal(c), body))
    return out


def collect_issue_comment_signals(comments: list[dict], model: str = "") -> list[ReviewSignal]:
    """Collect signals from PR *issue* comments carrying embedded fuko markers.

    Marker-driven, mirroring :func:`_prefer_marker`: the markers written at review
    time are authoritative for the machine fields. The human-facing ``title``/``body``
    (excluded from markers) are re-hydrated from a fresh guide parse when the comment
    is PR-Agent's "PR Reviewer Guide" — ``make_id`` is deterministic, so each marker
    matches its freshly parsed signal by id. Markers in other issue comments are
    yielded as-is. Comments without markers yield nothing.
    """
    out: list[ReviewSignal] = []
    for c in comments:
        body = c.get("body", "") or ""
        markers = extract_markers(body)
        if not markers:
            continue
        fresh = {s.id: s for s in guide_signals(c, model)}
        for marker in markers:
            parsed = fresh.get(marker.id)
            if parsed is not None:
                marker.title = parsed.title
                marker.body = parsed.body
            if not marker.thread_url:
                marker.thread_url = c.get("html_url")
            out.append(marker)
    return out
