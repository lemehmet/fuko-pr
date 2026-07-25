"""Ingestion of learnings with idempotent dedup via ON CONFLICT."""

from datetime import datetime
from uuid import UUID

from psycopg.errors import UniqueViolation

from .db import db, vector_literal
from .dedup import partition
from .embed import get_embedder
from .models import UNSET, DuplicateLearningError, IngestItem, check_source
from .retrieve import _ROW_COLUMNS, _row_to_dict, get_learning

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


def ingest(repo: str, items: list[IngestItem], *, max_new: int | None = None) -> tuple[int, int]:
    """Embed and insert learnings for ``repo``, skipping exact duplicates.

    Duplicates of the ``(repo, text, source)`` key are filtered out *before*
    embedding, so re-sweeping an already-ingested backlog costs no embed calls;
    only genuinely new learnings reach the (potentially slow) embedder. The
    ``ON CONFLICT`` insert remains as a backstop for races.

    Args:
        repo: Repository the learnings belong to.
        items: Candidate learnings.
        max_new: Cap on how many new learnings are embedded in this call. Items
            past the cap are neither inserted nor skipped, so a caller can detect
            them as ``len(items) - inserted - skipped`` and re-send the same batch
            to drain the rest. ``None`` embeds everything new.

    Returns:
        A ``(inserted, skipped)`` tuple.
    """
    if not items:
        return 0, 0
    existing = _existing_keys(repo, items)
    to_embed, skipped, _deferred = partition(items, existing, max_new)
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


_UPDATABLE = ("text", "source", "source_url", "file_globs", "topic", "expires_at")


def update(
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
    """Apply the supplied fields to one learning in ``repo`` and return the updated row.

    Arguments left at :data:`~sidecar.models.UNSET` are not written, so clearing
    a field and leaving it alone stay distinguishable. ``text`` is what gets
    embedded, so changing it re-embeds and every other change skips the embedder
    entirely -- a topic fix must not cost an embedding call.

    Returns ``None`` when ``id`` is not a learning in ``repo``.

    Raises:
        DuplicateLearningError: The result would collide with the
            ``(repo, text, source)`` unique key.
        UnknownSourceError: ``source`` is outside ``SOURCES``.
    """
    try:
        UUID(id)
    except ValueError:
        return None
    supplied = {
        name: value
        for name, value in zip(
            _UPDATABLE, (text, source, source_url, file_globs, topic, expires_at), strict=True
        )
        if value is not UNSET
    }
    if "source" in supplied:
        check_source(supplied["source"])

    if not supplied:
        return get_learning(repo, id)

    with db() as conn:
        current = conn.execute(
            "SELECT text FROM learnings WHERE repo = %s AND id = %s", (repo, id)
        ).fetchone()
        if current is None:
            return None

        assignments: list[str] = []
        params: list = []
        for name, value in supplied.items():
            if name == "expires_at":
                assignments.append("expires_at = %s")
                params.append(_parse_dt(value))
            else:
                assignments.append(f"{name} = %s")
                params.append(value)
        if supplied.get("text", current[0]) != current[0]:
            assignments.append("embedding = %s::vector")
            params.append(vector_literal(get_embedder().embed_one(supplied["text"])))

        try:
            row = conn.execute(
                f"UPDATE learnings SET {', '.join(assignments)} "
                f"WHERE repo = %s AND id = %s RETURNING {_ROW_COLUMNS}",
                (*params, repo, id),
            ).fetchone()
        except UniqueViolation as e:
            raise DuplicateLearningError(
                "another learning in this repo already has that (text, source)"
            ) from e
    return _row_to_dict(row) if row else None


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
