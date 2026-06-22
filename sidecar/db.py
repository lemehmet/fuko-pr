"""pgvector connection pool with auto-migration and vector helpers."""

import atexit
import re
import threading
from contextlib import contextmanager
from pathlib import Path

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _resolve_embed_dim() -> int:
    """Determine the vector column dimension from the live embedding model.

    Probes the embedder so the schema matches whatever the configured model
    returns; falls back to ``FUKO_EMBED_DIM`` if the probe fails (e.g. the
    embeddings backend is down but we still need the pool for ``/forget``).
    """
    from .embed import get_embedder

    try:
        return get_embedder().probe_dim()
    except Exception:
        return settings.embed_dim


def _migration_sql(dim: int) -> list[str]:
    """All ``migrations/*.sql`` statements in filename order.

    The ``vector(N)`` substitution sets the embedding column to the live model's
    dimension; it is a no-op on migrations without a vector column. ``--`` line
    comments are stripped before splitting on ``;`` so a semicolon inside a
    comment cannot truncate a statement. Every migration is idempotent
    (``IF NOT EXISTS``), so applying them on each pool creation is safe.
    """
    mig_dir = Path(__file__).resolve().parent.parent / "migrations"
    stmts: list[str] = []
    for path in sorted(mig_dir.glob("*.sql")):
        sql = re.sub(r"vector\(\d+\)", f"vector({dim})", path.read_text())
        sql = re.sub(r"--[^\n]*", "", sql)
        stmts.extend(s.strip() for s in sql.split(";") if s.strip())
    return stmts


def _existing_embed_dim(conn) -> int | None:
    """Return the current ``embedding`` column dimension (pgvector ``atttypmod``)."""
    row = conn.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'learnings' AND a.attname = 'embedding'
        """
    ).fetchone()
    if not row or row[0] in (None, -1):
        return None
    return row[0]


def _ensure_embed_dim(conn, dim: int) -> None:
    """Migrate the ``embedding`` column to ``dim`` if the model's dimension changed.

    pgvector encodes the dimension in ``atttypmod``. When it differs, the stored
    vectors were produced by a different model and cannot be reused, so every
    learning is re-embedded with the current model and the column + HNSW index are
    rebuilt at the new dimension (a one-time, potentially slow startup cost).
    """
    existing = _existing_embed_dim(conn)
    if existing is None or existing == dim:
        return
    _migrate_embed_dim(conn, dim)


def _migrate_embed_dim(conn, dim: int) -> None:
    """Re-embed every learning and rebuild the ``embedding`` column + index at ``dim``."""
    from .embed import get_embedder

    rows = conn.execute("SELECT id, text FROM learnings").fetchall()
    embeddings = get_embedder().embed([text for _, text in rows]) if rows else []

    conn.execute("DROP INDEX IF EXISTS learnings_embedding_idx")
    conn.execute("ALTER TABLE learnings DROP COLUMN embedding")
    conn.execute(f"ALTER TABLE learnings ADD COLUMN embedding vector({dim})")
    for (row_id, _text), emb in zip(rows, embeddings, strict=True):
        conn.execute(
            "UPDATE learnings SET embedding = %s::vector WHERE id = %s",
            (vector_literal(emb), row_id),
        )
    conn.execute("ALTER TABLE learnings ALTER COLUMN embedding SET NOT NULL")
    conn.execute(
        "CREATE INDEX learnings_embedding_idx ON learnings USING hnsw (embedding vector_cosine_ops)"
    )


def _close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None


def vector_literal(vec: list[float]) -> str:
    """Render an embedding as a Postgres vector literal, e.g. ``[0.1,0.2]``."""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


def get_pool() -> ConnectionPool:
    """Return the shared connection pool, creating it and running migrations once.

    The pool is published to the module global only after migrations have
    committed, under a lock, so a concurrent first request can never observe a
    pool whose schema isn't ready yet (the fresh-DB first-request race). Callers
    should prefer warming this at startup (see ``main.lifespan``) so the very
    first request is never the one paying the migration cost.
    """
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            open=True,
        )
        dim = _resolve_embed_dim()
        with pool.connection() as conn:
            for stmt in _migration_sql(dim):
                conn.execute(stmt)
            _ensure_embed_dim(conn, dim)
            register_vector(conn)
        atexit.register(_close_pool)
        _pool = pool
    return _pool


@contextmanager
def db():
    """Yield a pooled connection with the pgvector adapter registered."""
    pool = get_pool()
    with pool.connection() as conn:
        register_vector(conn)
        yield conn
