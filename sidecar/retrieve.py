"""Hybrid retrieval: semantic cosine search plus explicit file-glob filtering."""

import fnmatch

from .config import settings
from .db import db, vector_literal
from .embed import get_embedder


def _build_query(files: list[str], pr_body: str | None, query_text: str | None) -> str:
    parts: list[str] = []
    if query_text:
        parts.append(query_text.strip())
    if pr_body:
        parts.append(pr_body.strip())
    if files:
        parts.append("Changed files:\n" + "\n".join(files))
    return "\n".join(p for p in parts if p).strip()


def query(
    repo: str,
    files: list[str],
    pr_body: str | None = None,
    query_text: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Return up to ``top_k`` relevant learnings for ``repo`` given changed files.

    Combines a semantic cosine pass with explicitly file-scoped learnings, then
    keeps scoped learnings only where their globs match a changed path.
    """
    q = _build_query(files, pr_body, query_text)
    if not q:
        return []
    vec = vector_literal(get_embedder().embed_one(q))
    k = top_k or settings.top_k
    cand_k = settings.candidate_k

    sql = """
        SELECT id, text, source, source_url, file_globs, topic,
               1 - (embedding <=> %s::vector) AS score
        FROM learnings
        WHERE repo = %s AND (expires_at IS NULL OR expires_at > now())
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with db() as conn:
        semantic = conn.execute(sql, (vec, repo, vec, cand_k)).fetchall()
        scoped = _fetch_scoped(conn, vec, repo, cand_k)

    seen: dict[str, tuple] = {}
    for row in (*semantic, *scoped):
        seen[row[0]] = row

    results: list[dict] = []
    for row in seen.values():
        globs = row[4] or []
        if globs and not any(fnmatch.fnmatch(f, pat) for f in files for pat in globs):
            continue
        results.append(
            {
                "id": str(row[0]),
                "text": row[1],
                "source": row[2],
                "source_url": row[3],
                "file_globs": list(globs),
                "topic": row[5],
                "score": float(row[6]),
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def list_learnings(
    repo: str | None = None,
    source: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Return a page of live learnings (newest-first) plus the total match count.

    Unlike :func:`query`, this is neither semantic nor file-scoped -- it lists
    rows for inspection, optionally narrowed by ``repo`` and ``source``, and
    excludes expired learnings to match what retrieval would surface. Embeddings
    are not returned. The second element is the total matching the filters,
    independent of ``limit``/``offset``, for pagination.
    """
    where = ["(expires_at IS NULL OR expires_at > now())"]
    params: list = []
    if repo:
        where.append("repo = %s")
        params.append(repo)
    if source:
        where.append("source = %s")
        params.append(source)
    clause = " AND ".join(where)
    page_sql = f"""
        SELECT id, repo, text, source, source_url, file_globs, topic, created_at
        FROM learnings
        WHERE {clause}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """
    with db() as conn:
        total = conn.execute(f"SELECT count(*) FROM learnings WHERE {clause}", params).fetchall()[
            0
        ][0]
        rows = conn.execute(page_sql, [*params, limit, offset]).fetchall()
    items = [
        {
            "id": str(row[0]),
            "repo": row[1],
            "text": row[2],
            "source": row[3],
            "source_url": row[4],
            "file_globs": list(row[5] or []),
            "topic": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
        }
        for row in rows
    ]
    return items, int(total)


def _fetch_scoped(conn, vec: str, repo: str, cand_k: int) -> list[tuple]:
    sql = """
        SELECT id, text, source, source_url, file_globs, topic,
               1 - (embedding <=> %s::vector) AS score
        FROM learnings
        WHERE repo = %s AND file_globs <> '{}'
              AND (expires_at IS NULL OR expires_at > now())
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    return conn.execute(sql, (vec, repo, vec, cand_k)).fetchall()
