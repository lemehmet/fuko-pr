"""Hybrid retrieval: semantic cosine search plus explicit file-glob filtering."""

import fnmatch
from uuid import UUID

from .config import settings
from .db import db, vector_literal
from .digest import DIGEST_SOURCE
from .embed import get_embedder


# The PR body is the one unbounded part of a query: descriptions carry pasted
# logs, stack traces and generated changelogs, and bodies of 12k-20k tokens are
# what used to 500 the embedder outright. Bound it *before* assembly, not by
# capping the finished string -- a tail cut would drop the "Changed files:"
# block, the most discriminative part of the query, and keep the middle of a
# rambling body instead. Retrieval loses nothing worth having: one CLS-pooled
# 1024-dim vector over 20k tokens is nearer noise than a query.
_PR_BODY_CHARS = 6000


def _trim_files_block(block: str, room: int) -> str:
    """Return ``block`` cut to ``room`` characters on a path boundary."""
    if room <= 0:
        return ""
    if len(block) <= room:
        return block
    # A cut landing exactly on a path boundary already ends on a whole path;
    # rewinding to the previous newline there would throw away a complete path
    # for nothing, and with no body to absorb the freed room it can empty the
    # query outright.
    cut = block[:room] if block[room] == "\n" else block[:room].rpartition("\n")[0]
    # A block whose first line alone overruns the room leaves nothing whole to
    # keep; half a path matches no glob and embeds as noise, so keep none.
    return cut if "\n" in cut else ""


def _build_query(files: list[str], pr_body: str | None, query_text: str | None) -> str:
    # Bounding the body alone is not enough: the transport cap in
    # ``embed._fit`` cuts the *assembled* string, so a body sitting at
    # _PR_BODY_CHARS leaves the trailing "Changed files:" block whatever is
    # left of embed_max_chars -- and a PR with enough changed files overruns
    # that, losing exactly the block this ordering exists to protect. So the
    # budget is allocated here, by discriminative value rather than by position:
    # an explicit query_text is the caller's own question, the files block is
    # what makes one PR's query differ from another's, and the body yields
    # first. Assembled order is unchanged; only the allocation is prioritised.
    limit = max(0, settings.embed_max_chars)
    head = (query_text or "").strip()
    body = (pr_body or "").strip()[:_PR_BODY_CHARS]
    files_block = "Changed files:\n" + "\n".join(files) if files else ""

    used = 0
    kept: dict[str, str] = {}
    for name, part in (("head", head), ("files", files_block), ("body", body)):
        if not part:
            continue
        sep = 1 if used else 0
        room = limit - used - sep
        part = _trim_files_block(part, room) if name == "files" else part[: max(0, room)]
        if not part:
            continue
        kept[name] = part
        used += len(part) + sep

    return "\n".join(kept[n] for n in ("head", "body", "files") if kept.get(n)).strip()


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

    ``digest`` rows are excluded unless ``FUKO_DIGEST_RETRIEVAL`` is set: file
    digests (#158) ship dark, so a populated store reaches no review until a
    deployment opts in.
    """
    q = _build_query(files, pr_body, query_text)
    if not q:
        return []
    vec = vector_literal(get_embedder().embed_query(q))
    k = top_k or settings.top_k
    cand_k = settings.candidate_k
    # Excluded in SQL rather than filtered afterwards: a disabled source that
    # still competes for the candidate window would quietly degrade what a
    # review sees, which is exactly what shipping dark must not do.
    digests = bool(settings.digest_retrieval)

    sql = """
        SELECT id, text, source, source_url, file_globs, topic,
               1 - (embedding <=> %s::vector) AS score
        FROM learnings
        WHERE repo = %s AND (expires_at IS NULL OR expires_at > now())
              AND (%s OR source <> %s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with db() as conn:
        semantic = conn.execute(sql, (vec, repo, digests, DIGEST_SOURCE, vec, cand_k)).fetchall()
        scoped = _fetch_scoped(conn, vec, repo, cand_k, digests)

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


_ROW_COLUMNS = "id, repo, text, source, source_url, file_globs, topic, created_at, expires_at"


def like_escape(term: str) -> str:
    r"""Escape ``LIKE``/``ILIKE`` metacharacters so a search term matches literally.

    Without this, searching ``100%`` or ``a_b`` silently over-matches: ``%`` and
    ``_`` are pattern syntax, not text. Callers pair the escaped term with an
    explicit ``ESCAPE '\'`` clause. Shared by both stores so their search
    semantics cannot drift.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_to_dict(row: tuple) -> dict:
    return {
        "id": str(row[0]),
        "repo": row[1],
        "text": row[2],
        "source": row[3],
        "source_url": row[4],
        "file_globs": list(row[5] or []),
        "topic": row[6],
        "created_at": row[7].isoformat() if row[7] else None,
        "expires_at": row[8].isoformat() if row[8] else None,
    }


def list_learnings(
    repo: str | None = None,
    source: str | None = None,
    limit: int = 100,
    offset: int = 0,
    q: str | None = None,
    include_expired: bool = False,
) -> tuple[list[dict], int]:
    """Return a page of learnings (newest-first) plus the total match count.

    Unlike :func:`query`, this is neither semantic nor file-scoped -- it lists
    rows for inspection, optionally narrowed by ``repo``, ``source``, and a
    case-insensitive substring ``q`` over text and topic. Expired learnings are
    excluded by default, matching what retrieval would surface; ``include_expired``
    is the only way to see them. Embeddings are not returned. The second element
    is the total matching the filters, independent of ``limit``/``offset``.

    The ordering carries ``id`` as a tie-breaker because ``created_at`` is not
    unique: every row a single ``ingest`` call writes shares one transaction
    timestamp, so a page boundary falling inside such a block would otherwise
    return some rows twice and skip others across the separate page queries.
    Callers that page to completion -- ``fuko digest``'s supersession sweep is
    one -- would then silently leave rows behind.
    """
    where: list[str] = []
    params: list = []
    if not include_expired:
        where.append("(expires_at IS NULL OR expires_at > now())")
    if repo:
        where.append("repo = %s")
        params.append(repo)
    if source:
        where.append("source = %s")
        params.append(source)
    if q:
        where.append(r"(text ILIKE %s ESCAPE '\' OR coalesce(topic, '') ILIKE %s ESCAPE '\')")
        pattern = f"%{like_escape(q)}%"
        params.extend([pattern, pattern])
    clause = " AND ".join(where) if where else "TRUE"
    page_sql = f"""
        SELECT {_ROW_COLUMNS}
        FROM learnings
        WHERE {clause}
        ORDER BY created_at DESC, id DESC
        LIMIT %s OFFSET %s
    """
    with db() as conn:
        total = conn.execute(f"SELECT count(*) FROM learnings WHERE {clause}", params).fetchone()[0]
        rows = conn.execute(page_sql, [*params, limit, offset]).fetchall()
    return [_row_to_dict(row) for row in rows], int(total)


def get_learning(repo: str, id: str) -> dict | None:
    """Return one learning by id within ``repo``, or ``None`` when there is no such row.

    Expired learnings are returned. They are invisible to :func:`query` and to
    the default listing, which is precisely why an operator needs a way to look
    at one before deciding whether to revive or delete it.
    """
    try:
        UUID(id)
    except ValueError:
        return None
    with db() as conn:
        row = conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM learnings WHERE repo = %s AND id = %s", (repo, id)
        ).fetchone()
    return _row_to_dict(row) if row else None


def repos() -> list[dict]:
    """Return the per-repository footprint of live learnings, repo-sorted.

    One grouped query instead of paging the whole knowledge base client-side,
    which is what ``fuko kb count`` used to do.
    """
    sql = """
        SELECT repo, source, count(*)
        FROM learnings
        WHERE expires_at IS NULL OR expires_at > now()
        GROUP BY repo, source
    """
    with db() as conn:
        rows = conn.execute(sql).fetchall()
    return fold_repo_counts(rows)


def fold_repo_counts(rows: list[tuple]) -> list[dict]:
    """Fold ``(repo, source, count)`` triples into per-repo summaries, repo-sorted.

    Shared by both stores so their ``repos()`` output cannot drift apart.
    """
    summaries: dict[str, dict] = {}
    for repo, source, count in rows:
        entry = summaries.setdefault(repo, {"repo": repo, "count": 0, "sources": {}})
        entry["count"] += int(count)
        entry["sources"][source] = entry["sources"].get(source, 0) + int(count)
    return [summaries[key] for key in sorted(summaries)]


def _fetch_scoped(conn, vec: str, repo: str, cand_k: int, digests: bool) -> list[tuple]:
    sql = """
        SELECT id, text, source, source_url, file_globs, topic,
               1 - (embedding <=> %s::vector) AS score
        FROM learnings
        WHERE repo = %s AND file_globs <> '{}'
              AND (expires_at IS NULL OR expires_at > now())
              AND (%s OR source <> %s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    return conn.execute(sql, (vec, repo, digests, DIGEST_SOURCE, vec, cand_k)).fetchall()
