"""pgvector connection pool with auto-migration and vector helpers."""

import atexit
from contextlib import contextmanager
from pathlib import Path

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings

_pool: ConnectionPool | None = None


def _migration_sql() -> list[str]:
    path = Path(__file__).resolve().parent.parent / "migrations" / "001_init.sql"
    return [s.strip() for s in path.read_text().split(";") if s.strip()]


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
        with _pool.connection() as conn:
            for stmt in _migration_sql():
                conn.execute(stmt)
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
