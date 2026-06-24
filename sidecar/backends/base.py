"""Contracts for review backends and knowledge stores.

These ``Protocol`` definitions are the seams of the abstraction. A review
backend translates the unified config into its reviewer's native config
(ingress) and maps its reviewer's output back into Review Signals (egress); a
store persists and retrieves learnings. Implementations live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..fukoconfig import ModelConfig
from ..models import IngestItem
from ..presets import ProviderPreset
from ..signals import ReviewSignal


@dataclass(frozen=True)
class PRRef:
    """A pull request a backend should review or read back."""

    repo: str
    number: int
    url: str


@dataclass(frozen=True)
class InvokeResult:
    """The outcome of invoking a review backend.

    ``throttled`` is set when the failure looks like provider throttling (429,
    quota, overload, or a container timeout) -- the signal the runner uses to
    fail over to the next provider and trip that provider's breaker. ``provider``
    records which pool entry produced this result.
    """

    returncode: int
    detail: str = ""
    throttled: bool = False
    provider: str | None = None


@runtime_checkable
class ReviewBackend(Protocol):
    """A pluggable PR review engine (e.g. PR-Agent)."""

    name: str
    supports_inline_suggestions: bool
    injection: Literal["extra_instructions", "api", "prompt"]

    def build_env(
        self,
        preset: ProviderPreset,
        model: ModelConfig,
        knowledge: str,
        tools: list[str],
    ) -> dict[str, str]:
        """Translate the unified config into the backend's native config (ingress)."""
        ...

    def invoke(self, pr: PRRef, env: dict[str, str], tools: list[str]) -> InvokeResult:
        """Run the backend's ``tools`` against ``pr`` with the translated environment."""
        ...

    def normalize_output(
        self,
        pr: PRRef,
        model: str = "",
        *,
        compare_label: str | None = None,
        token: str | None = None,
        api_url: str | None = None,
    ) -> list[ReviewSignal]:
        """Read the backend's posted review and map it to Review Signals (egress).

        When ``compare_label`` is set (A/B mode) the backend additionally prepends
        that visible label to each inline comment it newly marks, so a human reading
        the diff can tell which branch produced which suggestion. The label is the
        configured ``provider/name`` (matching the per-branch summary header), kept
        distinct from ``model`` (the litellm-prefixed id used in the invisible
        marker) so the visible tag never shows a provider's litellm alias.

        ``token``/``api_url`` pin the GitHub identity that reads and edits the
        comments; in concurrent A/B mode each branch passes its own so marking
        happens under that branch's identity. When unset the backend falls back to
        the process ``GITHUB_TOKEN``/``GITHUB_API_URL`` (the sequential path).
        """
        ...


@runtime_checkable
class Store(Protocol):
    """A pluggable knowledge store (e.g. Postgres/pgvector, sqlite-vec)."""

    def ingest(self, repo: str, items: list[IngestItem]) -> tuple[int, int]:
        """Persist ``items`` for ``repo``; return ``(inserted, skipped)``."""
        ...

    def query(
        self,
        repo: str,
        files: list[str],
        pr_body: str | None,
        query_text: str | None,
        top_k: int | None,
    ) -> list[dict]:
        """Return the learnings most relevant to the given PR context."""
        ...

    def forget(
        self,
        repo: str,
        *,
        id: str | None = None,
        source: str | None = None,
        all: bool = False,
    ) -> int:
        """Delete learnings matching the given selector; return the count removed."""
        ...

    def list_learnings(
        self,
        repo: str | None = None,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Return a page of live learnings (newest-first) plus the total match count."""
        ...
