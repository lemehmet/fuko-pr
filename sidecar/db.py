"""pgvector connection pool with auto-migration and vector helpers."""

import atexit
import re
from contextlib import contextmanager
from pathlib import Path

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings

_pool: ConnectionPool | None = None


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
    path = Path(__file__).resolve().parent.parent / "migrations" / "001_init.sql"
    sql = re.sub(r"vector\(\d+\)", f"vector({dim})", path.read_text())
    return [s.strip() for s in sql.split(";") if s.strip()]


def _verify_embed_dim(conn, dim: int) -> None:
    """Raise if the existing ``embedding`` column dimension differs from ``dim``.

    pgvector encodes the dimension in ``atttypmod``; a mismatch means the table
    predates the current model and inserts would fail at runtime, so we fail
    loudly at startup instead with a clear remediation.
    """
    row = conn.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'learnings' AND a.attname = 'embedding'
        """
    ).fetchone()
    if row and row[0] not in (None, -1) and row[0] != dim:
        raise RuntimeError(
            f"learnings.embedding is vector({row[0]}) but the model returns {dim}-dim "
            "vectors. The embedding model changed; drop & recreate the learnings table "
            "(the migration is IF NOT EXISTS and won't alter an existing column)."
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
    """Return the shared connection pool, creating it and running the migration on first use."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            open=True,
        )
        dim = _resolve_embed_dim()
        with _pool.connection() as conn:
            for stmt in _migration_sql(dim):
                conn.execute(stmt)
            _verify_embed_dim(conn, dim)
            register_vector(conn)
        atexit.register(_close_pool)
    return _pool


@contextmanager
def db():
    """Yield a pooled connection with the pgvector adapter registered."""
    pool = get_pool()
    with pool.connection() as conn:
        register_vector(conn)
        yield conn
