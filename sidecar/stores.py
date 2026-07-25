"""Knowledge-store factory and the Postgres/pgvector implementation.

The ``Store`` protocol (:class:`sidecar.backends.base.Store`) lets the runner and
the HTTP server stay agnostic to *where* learnings live. ``PostgresStore`` is the
default (sidecar / homelab); a sqlite-vec + object-storage store plugs in via
:func:`get_store` for the server-free deployment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import ingest as _ingest
from . import retrieve as _retrieve
from .fukoconfig import KnowledgeConfig
from .models import UNSET, IngestItem

if TYPE_CHECKING:
    from .backends.base import Store


class PostgresStore:
    """Store backed by pgvector (delegates to the ingest/retrieve modules)."""

    def ingest(
        self, repo: str, items: list[IngestItem], *, max_new: int | None = None
    ) -> tuple[int, int]:
        """Embed and insert learnings, skipping exact duplicates."""
        return _ingest.ingest(repo, items, max_new=max_new)

    def query(
        self,
        repo: str,
        files: list[str],
        pr_body: str | None = None,
        query_text: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Return the learnings most relevant to the given PR context."""
        return _retrieve.query(repo, files, pr_body, query_text, top_k)

    def forget(
        self,
        repo: str,
        *,
        id: str | None = None,
        source: str | None = None,
        all: bool = False,
    ) -> int:
        """Delete learnings by id, source, or wholesale; return the count removed."""
        return _ingest.forget(repo, id=id, source=source, all_=all)

    def list_learnings(
        self,
        repo: str | None = None,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
        q: str | None = None,
        include_expired: bool = False,
    ) -> tuple[list[dict], int]:
        """Return a page of learnings (newest-first) plus the total match count."""
        return _retrieve.list_learnings(repo, source, limit, offset, q, include_expired)

    def get_learning(self, repo: str, id: str) -> dict | None:
        """Return one learning (expired included), or ``None`` when absent from ``repo``."""
        return _retrieve.get_learning(repo, id)

    def update_learning(
        self,
        repo: str,
        id: str,
        *,
        text: str = UNSET,
        source: str = UNSET,
        source_url: str | None = UNSET,
        file_globs: list[str] = UNSET,
        topic: str | None = UNSET,
        expires_at: str | None = UNSET,
    ) -> dict | None:
        """Apply the supplied fields to one learning; re-embeds only on a text change."""
        return _ingest.update(
            repo,
            id,
            text=text,
            source=source,
            source_url=source_url,
            file_globs=file_globs,
            topic=topic,
            expires_at=expires_at,
        )

    def repos(self) -> list[dict]:
        """Return the per-repository footprint of live learnings."""
        return _retrieve.repos()


class UnknownStoreError(ValueError):
    """Raised when ``.fuko.toml`` names a knowledge store that is not implemented."""


_current: Store | None = None


def current_store() -> Store:
    """Return the process-wide store, built once from ``.fuko.toml``.

    The HTTP API and the browser console must operate on the *same* store
    instance -- two independently constructed sqlite-vec stores would each carry
    their own probed embedding dimension and object-store sync state.
    """
    global _current
    if _current is None:
        from .fukoconfig import load_config

        _current = get_store(load_config().knowledge)
    return _current


def get_store(knowledge: KnowledgeConfig) -> Store:
    """Return the store implementation selected by ``knowledge.store``."""
    if knowledge.store == "postgres":
        return PostgresStore()
    if knowledge.store == "sqlite-vec":
        from .sqlite_store import SqliteVecStore

        return SqliteVecStore(knowledge)
    raise UnknownStoreError(f"unknown knowledge store '{knowledge.store}'")
