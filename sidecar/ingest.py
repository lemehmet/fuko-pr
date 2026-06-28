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


def _existing_keys(repo: str, items: list[IngestItem]) -> set[tuple[str, str]]:
    candidates = {(it.text, it.source) for it in items}
    texts = list({it.text for it in items})
    with db() as conn:
        rows = conn.execute(
            "SELECT text, source FROM learnings WHERE repo = %s AND text = ANY(%s)",
            (repo, texts),
        ).fetchall()
    return {(text, source) for text, source in rows if (text, source) in candidates}


def ingest(repo: str, items: list[IngestItem]) -> tuple[int, int]:
    """Embed and insert learnings for ``repo``, skipping exact duplicates.

    Duplicates of the ``(repo, text, source)`` key are filtered out *before*
    embedding, so re-sweeping an already-ingested backlog costs no embed calls;
    only genuinely new learnings reach the (potentially slow) embedder. The
    ``ON CONFLICT`` insert remains as a backstop for races.

    Returns:
        A ``(inserted, skipped)`` tuple.
    """
    if not items:
        return 0, 0
    existing = _existing_keys(repo, items)
    to_embed: list[IngestItem] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    for item in items:
        key = (item.text, item.source)
        if key in existing or key in seen:
            skipped += 1
            continue
        seen.add(key)
        to_embed.append(item)
    if not to_embed:
        return 0, skipped
    embeddings = get_embedder().embed([it.text for it in to_embed])
    inserted = 0
    with db() as conn:
        for item, emb in zip(to_embed, embeddings, strict=True):
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
