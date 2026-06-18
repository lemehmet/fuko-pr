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
from .models import IngestItem

if TYPE_CHECKING:
    from .backends.base import Store


class PostgresStore:
    """Store backed by pgvector (delegates to the ingest/retrieve modules)."""

    def ingest(self, repo: str, items: list[IngestItem]) -> tuple[int, int]:
        """Embed and insert learnings, skipping exact duplicates."""
        return _ingest.ingest(repo, items)

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


class UnknownStoreError(ValueError):
    """Raised when ``.fuko.toml`` names a knowledge store that is not implemented."""


def get_store(knowledge: KnowledgeConfig) -> Store:
    """Return the store implementation selected by ``knowledge.store``."""
    if knowledge.store == "postgres":
        return PostgresStore()
    if knowledge.store == "sqlite-vec":
        from .sqlite_store import SqliteVecStore

        return SqliteVecStore(knowledge)
    raise UnknownStoreError(f"unknown knowledge store '{knowledge.store}'")
