"""The canonical fuko Review Signal schema (v1) and its comment marker.

Every backend normalizes its reviewer's output into this shape, so a consumer
(e.g. an address-PR-reviews tool) reads one deterministic schema instead of
sniffing each vendor's ad-hoc format. A signal travels inside an *invisible* HTML
comment marker (``<!-- fuko-signal:v1 {json} -->``) appended to the PR comment it
describes: it renders as nothing on GitHub/GitLab and survives round-trips, so the
consumer can ``grep`` the marker and parse the JSON deterministically.

The marker carries only machine fields -- the human-facing ``title``/``body`` stay
in the visible comment and are excluded, which also guarantees the JSON can never
contain ``-->`` and break the comment.
"""

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "low", "medium", "high", "critical"]
Category = Literal["bug", "security", "perf", "style", "test", "docs", "design"]

_MARKER_TAG = "fuko-signal:v1"
_MARKER_RE = re.compile(r"<!--\s*fuko-signal:v1\s+(.*?)\s*-->")
_MARKER_STRIP_RE = re.compile(r"\n*<!--\s*fuko-signal:v1\s+.*?\s*-->\n*")


class ReviewSignal(BaseModel):
    """A single normalized review finding."""

    v: int = 1
    id: str
    file: str | None = None
    line: int | None = None
    end_line: int | None = None
    severity: Severity = "medium"
    severity_source: Literal["declared", "inferred"] = "inferred"
    category: Category = "bug"
    title: str = ""
    body: str = ""
    suggestion: bool = False
    thread_url: str | None = None
    backend: str = ""
    model: str = ""
    kb_refs: list[str] = Field(default_factory=list)


def make_id(*parts: str) -> str:
    """Return a stable ``fk_`` id derived from ``parts`` (same inputs -> same id)."""
    digest = hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:10]
    return "fk_" + digest


def encode_marker(signal: ReviewSignal) -> str:
    """Render ``signal`` as an invisible HTML-comment marker (machine fields only)."""
    payload = signal.model_dump_json(exclude={"title", "body"})
    return f"<!-- {_MARKER_TAG} {payload} -->"


def extract_markers(text: str) -> list[ReviewSignal]:
    """Parse all fuko-signal markers from ``text``, skipping malformed ones."""
    out: list[ReviewSignal] = []
    for m in _MARKER_RE.finditer(text or ""):
        try:
            out.append(ReviewSignal.model_validate_json(m.group(1)))
        except ValueError:
            continue
    return out


def strip_markers(text: str) -> str:
    """Remove any fuko-signal markers (and their surrounding blank lines) from ``text``."""
    return _MARKER_STRIP_RE.sub("", text or "")


def with_marker(body: str, signal: ReviewSignal) -> str:
    """Return ``body`` with ``signal``'s marker appended, replacing any existing marker."""
    return strip_markers(body).rstrip() + "\n\n" + encode_marker(signal)
