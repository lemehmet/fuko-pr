"""pgvector connection pool with auto-migration and vector helpers.

Two classes of caller share this pool and they want opposite things from an
unreachable database. The knowledge-base paths (:mod:`sidecar.ingest`,
:mod:`sidecar.retrieve`) are the REASON a request was made, so waiting is the
right answer for them and psycopg_pool's 30s default stands. The best-effort
state paths -- the review ledgers, the circuit breaker, run metrics, reviewer
health -- are optimizations layered onto a review that must complete without
them, so for those a wait IS the failure (#170): every guarded call paid the
full 30s before its guard could convert the timeout into a no-op, and a degraded
round makes up to five of them plus one per settled finding.

:func:`db_best_effort` is that second contract, and it is a separate entry point
rather than a smaller number on the pool precisely because the two classes share
the pool: retuning it globally would silently shorten the knowledge base's wait
too. It bounds connection ACQUISITION and latches the process off after the
first failure, so a dead Postgres costs a round one timeout rather than one per
call.
"""

import atexit
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from pgvector.psycopg import register_vector
from psycopg import OperationalError
from psycopg_pool import ConnectionPool, PoolTimeout

from .config import settings
from .logfmt import flatten_for_log

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()

BEST_EFFORT_TIMEOUT_S = 2.0
"""How long a best-effort call may wait for a connection, in seconds.

Bounds ACQUISITION only -- once a connection is in hand, statement execution is
unbounded, the same as for every other caller. That is the right split: this
number exists to detect a database that is not there, and a query that is slow
once connected is a different problem with a different answer.

Sized for the deployments that take the direct-Postgres path, where the database
is on localhost or the LAN and a pooled acquisition is sub-millisecond. Two
seconds is therefore already far outside the healthy distribution while staying
clear of a cold pool's first TCP handshake; a smaller number would start
reporting a live database as dead.
"""

BEST_EFFORT_COOLDOWN_S = 60.0
"""How long the fast-fail latch holds before one call is allowed to re-probe.

:mod:`sidecar.review_state_client`'s equivalent latch is process-wide and never
reset, and says so deliberately: a runner process is one review run. This one
cannot borrow that lifetime, because the same pool serves the long-lived sidecar
whose HTTP handlers ARE the ledger for the rest of the fleet -- latching it
permanently would turn one restart of Postgres into a ledger that stays dead
until someone restarts the sidecar.

So the latch expires, and the cost of it expiring is one
:data:`BEST_EFFORT_TIMEOUT_S` per minute per process while the database is down:
two seconds a minute, against the ~2.5 minutes a round paid per #170. A round's
calls arrive in bursts (carry-in, then settle), so in practice a degraded round
pays the probe once or twice, not once per minute of its wall clock.
"""

_unreachable_until = 0.0
_latch_gen = 0
_latch_lock = threading.Lock()


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
    """Re-embed the store when the embedding model, or its dimension, changed.

    Two things can invalidate the stored vectors, and only one of them is
    visible in the schema:

    * The **dimension** changed. pgvector encodes it in ``atttypmod``, so a
      mismatch is unambiguous and the column and HNSW index have to be rebuilt
      as well as the vectors.
    * The **model** changed at the same dimension. bge-m3 and
      Qwen3-Embedding-0.6B are both 1024-wide, so nothing in the schema tells
      them apart — but their vectors are not comparable, and a store holding
      both retrieves badly rather than failing. ``meta.embed_model`` is the
      only record of which one produced what is stored.

    An absent marker is treated as a model change. Provenance cannot be proven
    for vectors written before this table existed, and paying one re-embed is
    cheaper than serving a silently mixed store.
    """
    existing = _existing_embed_dim(conn)
    stored_model = _stored_embed_model(conn)
    if existing is not None and (existing != dim or stored_model != settings.embed_model):
        _migrate_embed_dim(conn, dim)
    _record_embed_model(conn)


def _stored_embed_model(conn) -> str | None:
    """Return the model recorded as the source of the stored vectors, if any."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    return row[0] if row else None


def _record_embed_model(conn) -> None:
    """Record the configured model as the source of what is now stored."""
    conn.execute(
        """
        INSERT INTO meta (key, value) VALUES ('embed_model', %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        (settings.embed_model,),
    )


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


def get_pool(*, timeout: float | None = None) -> ConnectionPool:
    """Return the shared connection pool, creating it and running migrations once.

    The pool is published to the module global only after migrations have
    committed, under a lock, so a concurrent first request can never observe a
    pool whose schema isn't ready yet (the fresh-DB first-request race). Callers
    should prefer warming this at startup (see ``main.lifespan``) so the very
    first request is never the one paying the migration cost.

    Args:
        timeout: Seconds to wait for the migration connection, or ``None`` for
            psycopg_pool's own default. It has to be plumbed here and not only
            into :func:`db`: when the pool does not exist yet -- the case for
            every call in a process whose first database contact is a
            best-effort one -- the wait that #170 is about happens on THIS
            connection, and a bound applied only afterwards would never be
            reached. It does not bound the migration statements themselves,
            which run once the connection is in hand, nor the wait for
            ``_pool_lock``, which is held across the whole creation -- so a
            caller can still queue behind someone else's, and concurrent
            first-time callers against a dead store each run their own attempt.
            Both are residuals of the acquisition-only bound; tracked in #207.
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
        try:
            dim = _resolve_embed_dim()
            with pool.connection(timeout=timeout) as conn:
                for stmt in _migration_sql(dim):
                    conn.execute(stmt)
                _ensure_embed_dim(conn, dim)
                register_vector(conn)
        except Exception:
            pool.close()
            raise
        atexit.register(_close_pool)
        _pool = pool
    return _pool


@contextmanager
def db(*, timeout: float | None = None):
    """Yield a pooled connection with the pgvector adapter registered.

    Args:
        timeout: Seconds to wait for a connection, or ``None`` (the default) for
            psycopg_pool's own 30s. Callers that must not block a review pass
            through :func:`db_best_effort` instead of naming a number here.
    """
    pool = get_pool(timeout=timeout)
    with pool.connection(timeout=timeout) as conn:
        register_vector(conn)
        yield conn


class StoreUnavailable(PoolTimeout):
    """Raised instead of waiting, while the fast-fail latch is closed.

    A :class:`~psycopg_pool.PoolTimeout` subclass on purpose. Of the four
    best-effort modules only :mod:`sidecar.review_state` guards its own failure
    path; the other three let the exception reach a caller that catches what a
    dead pool already raises. Inheriting keeps every one of those handlers
    correct without being re-read, so the latch changes only how long a caller
    waits for the same outcome -- which is the whole of #170.
    """


def _enter_probe() -> int:
    """Refuse a latched call, or return the generation to unlatch against later.

    Both halves under one acquisition of :data:`_latch_lock`, because they are
    one decision. Reading the generation *after* a separate rejection check
    would leave a window in between: a concurrent failure latching there would
    hand this call the NEW generation, and its own success would then clear a
    failure it never tested -- the same mistake :func:`_unlatch` guards against,
    one step earlier. Taking the lock once makes that interleaving
    unrepresentable rather than merely unlikely.

    Returns:
        :data:`_latch_gen` as of the check, for :func:`_unlatch`.

    Raises:
        StoreUnavailable: The store was found unreachable within the cooldown.
    """
    with _latch_lock:
        if _unreachable_until and time.monotonic() < _unreachable_until:
            raise StoreUnavailable(
                f"postgres was unreachable within the last {BEST_EFFORT_COOLDOWN_S:.0f}s; "
                "this best-effort call was skipped rather than waited out"
            )
        return _latch_gen


def _latch(exc: Exception) -> None:
    """Close the latch for :data:`BEST_EFFORT_COOLDOWN_S`, announcing it once.

    Announced only on the transition, not per suppressed call: the calls this
    latch exists to skip can number in the hundreds in one round (``transition``
    is per finding), and each of those already gets its own line from its own
    guard. What an operator needs from this stream is the one line that says the
    ledger stopped being consulted, and when it will be tried again.
    """
    global _unreachable_until, _latch_gen
    with _latch_lock:
        already = bool(_unreachable_until) and time.monotonic() < _unreachable_until
        _unreachable_until = time.monotonic() + BEST_EFFORT_COOLDOWN_S
        _latch_gen += 1
    if already:
        return
    print(
        f"fuko: postgres unreachable ({flatten_for_log(str(exc))}); "
        f"best-effort state calls are skipped for {BEST_EFFORT_COOLDOWN_S:.0f}s",
        file=sys.stderr,
    )


def _unlatch(gen: int) -> None:
    """Reopen the latch, unless a newer failure closed it while this probe ran.

    ``gen`` is :data:`_latch_gen` as read before the probe started, and every
    :func:`_latch` bumps it. A mismatch therefore means some concurrent call
    recorded a connection failure that this probe's success says nothing about:
    the two overlapped, and the later evidence is the failure. Clearing it
    anyway would put every following best-effort call back on its own
    :data:`BEST_EFFORT_TIMEOUT_S` against a store just found unreachable, which
    is the per-call wait the latch exists to remove.

    The comparison has to happen under the lock for the same reason the write
    does -- reading the generation first and clearing afterwards would just move
    the race rather than close it.
    """
    global _unreachable_until
    with _latch_lock:
        if _latch_gen != gen:
            return
        _unreachable_until = 0.0


def _looks_saturated() -> bool:
    """True when a :class:`~psycopg_pool.PoolTimeout` reads as a full pool, not a dead one.

    A pool that holds connections and has none free was waited on because every
    one of them was in use. The sidecar serves the whole fleet's ledger traffic
    plus the knowledge base from a single ``max_size=10`` pool, so a burst can
    exhaust it while Postgres answers normally; latching on that would take the
    ledger away from every seat for :data:`BEST_EFFORT_COOLDOWN_S` on the
    strength of a queue, and print an outage line about a healthy database.

    The two misreadings are not symmetric, and that is what settles which way to
    lean. Reading a dead store as merely busy costs one
    :data:`BEST_EFFORT_TIMEOUT_S` per call -- still bounded, still an order of
    magnitude better than the 30s of #170 -- and ends as soon as the pool
    discards its stale connections. Reading a busy store as dead costs
    fleet-wide ledger silence for a minute. So this deliberately keeps the latch
    open in the ambiguous window right after a database dies with its
    connections still checked out.

    Reads the module global instead of calling :func:`get_pool`, because this
    runs on the failure path where creating a pool is precisely the wait being
    bounded; ``None`` (no pool, or one whose creation just failed) is not
    saturation. Stats that cannot be read are treated the same way -- the latch
    is the safe default once nothing supports the queue explanation.
    """
    pool = _pool
    if pool is None:
        return False
    try:
        stats = pool.get_stats()
    except Exception:
        return False
    return stats.get("pool_size", 0) > 0 and stats.get("pool_available", 0) == 0


@contextmanager
def db_best_effort():
    """Yield a pooled connection for a caller that must never block a review.

    The bounded counterpart to :func:`db`, for the state paths a review can
    complete without: :mod:`sidecar.review_state`, :mod:`sidecar.circuit_breaker`,
    :mod:`sidecar.run_metrics` and :mod:`sidecar.reviewer_health`. Two bounds,
    because one is not enough. The :data:`BEST_EFFORT_TIMEOUT_S` acquisition
    budget caps a single call; the latch caps the ROUND, since a budget alone
    still costs its own wait on every one of a degraded round's calls.

    Latched by connection failures -- :class:`~psycopg_pool.PoolTimeout` and
    :class:`~psycopg.OperationalError`, either of which means "this database is
    not answering", including a connection lost part-way through a statement. An
    error the statement itself earned arrives as a sibling class instead
    (:class:`~psycopg.ProgrammingError` for a column that is not there,
    :class:`~psycopg.IntegrityError` for a constraint) and leaves the latch open,
    for the same reason :func:`sidecar.review_state_client._mark_down` does not
    trip on an HTTP status: the latch exists to bound time, not to route around
    errors. That boundary is psycopg's rather than ours, so it is not exact -- a
    server-side cancellation (``statement_timeout``, an admin shutdown) is also
    an :class:`~psycopg.OperationalError` and does latch. That is the right side
    to be wrong on: those conditions mean the database is not usefully answering
    either, and the cost of the latch is one minute of no-ops.

    The one :class:`~psycopg_pool.PoolTimeout` that does NOT latch is a
    saturated pool -- connections held, none free -- which is a queue behind a
    healthy database rather than an outage; see :func:`_looks_saturated` for why
    that asymmetry is deliberate. A successful acquisition after the cooldown
    expired reopens the latch, so a database that comes back is used again
    without a restart.

    Yields:
        A pooled connection with the pgvector adapter registered.

    Raises:
        StoreUnavailable: The latch is closed; nothing was attempted.
    """
    gen = _enter_probe()
    try:
        with db(timeout=BEST_EFFORT_TIMEOUT_S) as conn:
            _unlatch(gen)
            yield conn
    except PoolTimeout as e:
        if not _looks_saturated():
            _latch(e)
        raise
    except OperationalError as e:
        _latch(e)
        raise
