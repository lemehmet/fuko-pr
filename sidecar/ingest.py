"""Ingestion of learnings with idempotent dedup via ON CONFLICT."""

from datetime import datetime
from uuid import UUID

from .db import db, vector_literal
from .embed import get_embedder
from .models import IngestItem

_INSERT_SQL = """
    INSERT INTO learnings
        (repo, text, source, source_url, file_globs, topic, embedding, origin_user, expires_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
    ON CONFLICT (repo, text, source) DO NOTHING
"""


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def ingest(repo: str, items: list[IngestItem]) -> tuple[int, int]:
    """Embed and insert learnings for ``repo``, skipping exact duplicates.

    Returns:
        A ``(inserted, skipped)`` tuple.
    """
    if not items:
        return 0, 0
    embeddings = get_embedder().embed([it.text for it in items])
    inserted = 0
    skipped = 0
    with db() as conn:
        for item, emb in zip(items, embeddings, strict=True):
            cur = conn.execute(
                _INSERT_SQL,
                (
                    repo,
                    item.text,
                    item.source,
                    item.source_url,
                    item.file_globs,
                    item.topic,
                    vector_literal(emb),
                    item.origin_user,
                    _parse_dt(item.expires_at),
                ),
            )
            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1
    return inserted, skipped


def forget(
    repo: str, *, id: str | None = None, source: str | None = None, all_: bool = False
) -> int:
    """Delete learnings for ``repo`` by id, source, or wholesale; returns the count deleted."""
    if id:
        try:
            UUID(id)
        except ValueError:
            return 0
        stmt, params = "DELETE FROM learnings WHERE repo = %s AND id = %s", (repo, id)
    elif source:
        stmt, params = "DELETE FROM learnings WHERE repo = %s AND source = %s", (repo, source)
    elif all_:
        stmt, params = "DELETE FROM learnings WHERE repo = %s", (repo,)
    else:
        return 0
    with db() as conn:
        cur = conn.execute(stmt, params)
    return cur.rowcount
