"""The canonical fuko Review Signal schema (v1).

Every backend normalizes its reviewer's output into this shape, so a consumer
(e.g. an address-PR-reviews tool) reads one deterministic schema instead of
sniffing each vendor's ad-hoc format. The marker encode/decode for embedding a
signal invisibly in a PR comment lands with the egress work.
"""

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "low", "medium", "high", "critical"]
Category = Literal["bug", "security", "perf", "style", "test", "docs", "design"]


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
