"""Tests for the bounded best-effort store path (#170): timeout + fast-fail latch."""

import contextlib

import pytest
from psycopg import OperationalError
from psycopg_pool import PoolTimeout

import sidecar.db as db
from sidecar import review_state

_UUID = "0f9d1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b"


class _FakeConn:
    """A handed-out connection, optionally one that dies under ``execute``."""

    def __init__(self, error=None):
        self.error = error

    def execute(self, *_a, **_k):
        if self.error is not None:
            raise self.error
        return None


class _FakePool:
    """Records every acquisition and its timeout; optionally refuses to connect.

    ``error`` fails the acquisition itself; ``conn_error`` lets the acquisition
    succeed and fails the statement afterwards. The two are separate because the
    latch has to treat both as "not answering" and only the first was reachable
    through a pool that raises before it yields.
    """

    def __init__(self, error=None, conn_error=None, stats=None):
        self.error = error
        self.conn_error = conn_error
        self.stats = stats
        self.timeouts: list[float | None] = []

    @contextlib.contextmanager
    def connection(self, timeout=None):
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        yield _FakeConn(self.conn_error)

    def get_stats(self) -> dict[str, int]:
        return self.stats or {}

    def close(self) -> None:
        self.closed = True

    @property
    def attempts(self) -> int:
        return len(self.timeouts)


@pytest.fixture(autouse=True)
def _open_latch(monkeypatch):
    """The latch is process state; no test may inherit or leak a closed one."""
    monkeypatch.setattr(db, "_unreachable_until", 0.0)
    monkeypatch.setattr(db, "_latch_gen", 0)
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "register_vector", lambda conn: None)


@pytest.fixture
def pool(monkeypatch):
    """Install a fake pool in place of the real one; return it for assertions."""

    def install(error=None, conn_error=None):
        fake = _FakePool(error, conn_error)
        monkeypatch.setattr(db, "get_pool", lambda *, timeout=None: fake)
        return fake

    return install


@pytest.fixture
def creating_pool(monkeypatch):
    """Let :func:`sidecar.db.get_pool` actually build a pool, with a fake class.

    The other fixture replaces ``get_pool`` wholesale, which is what makes the
    bound on pool CREATION -- the half the fix argues matters most, since a
    process whose first database contact is best-effort spends the wait on the
    migration connection -- invisible to those tests.
    """

    def install(error=None):
        made = _FakePool(error)
        made.closed = False
        monkeypatch.setattr(db, "ConnectionPool", lambda **_k: made)
        monkeypatch.setattr(db, "_resolve_embed_dim", lambda: 8)
        monkeypatch.setattr(db, "_migration_sql", lambda dim: ())
        monkeypatch.setattr(db, "_ensure_embed_dim", lambda conn, dim: None)
        monkeypatch.setattr(db.atexit, "register", lambda fn: fn)
        return made

    return install


def test_store_unavailable_is_a_pool_timeout():
    """Callers that already handle a dead pool keep handling the latch's refusal."""
    assert issubclass(db.StoreUnavailable, PoolTimeout)


def test_a_best_effort_call_bounds_its_acquisition(pool):
    fake = pool()
    with db.db_best_effort():
        pass
    assert fake.timeouts == [db.BEST_EFFORT_TIMEOUT_S]


def test_a_plain_call_keeps_the_pool_default(pool):
    """The knowledge-base paths must not inherit the best-effort bound."""
    fake = pool()
    with db.db():
        pass
    assert fake.timeouts == [None]


def test_the_second_call_after_a_timeout_never_touches_the_pool(pool, capsys):
    """#170's shape: N calls against a dead database cost ONE timeout, not N."""
    fake = pool(PoolTimeout("couldn't get a connection after 2.00 sec"))

    for _ in range(5):
        with pytest.raises(PoolTimeout), db.db_best_effort():
            pass  # pragma: no cover

    assert fake.attempts == 1
    assert capsys.readouterr().err.count("postgres unreachable") == 1


def test_a_refused_connection_latches_too(pool):
    """``OperationalError`` is the other way a failed ACQUISITION arrives."""
    fake = pool(OperationalError("connection failed"))

    with pytest.raises(OperationalError), db.db_best_effort():
        pass  # pragma: no cover
    with pytest.raises(db.StoreUnavailable), db.db_best_effort():
        pass  # pragma: no cover

    assert fake.attempts == 1


def test_a_connection_lost_mid_statement_latches(pool):
    """The store can also stop answering AFTER a connection is in hand.

    Nothing above reaches this path: a pool that raises before it yields only
    ever exercises acquisition. Here the acquisition succeeds and ``execute``
    raises, which reaches the same guard through the generator.
    """
    fake = pool(conn_error=OperationalError("server closed the connection unexpectedly"))

    with pytest.raises(OperationalError):
        with db.db_best_effort() as conn:
            conn.execute("SELECT 1")
    with pytest.raises(db.StoreUnavailable), db.db_best_effort():
        pass  # pragma: no cover

    assert fake.attempts == 1


def test_a_probe_does_not_clear_a_newer_failure(pool, monkeypatch):
    """Overlapping success and failure: the later evidence is the failure.

    Interleaved deterministically rather than with threads -- the race window is
    between reading the generation and clearing the latch, so the test opens it
    exactly, by latching from inside the unlatch call the probe is about to
    make.
    """
    fake = pool()
    unlatch = db._unlatch

    def latch_then_unlatch(gen):
        db._latch(PoolTimeout("a concurrent call found it down"))
        unlatch(gen)

    monkeypatch.setattr(db, "_unlatch", latch_then_unlatch)

    with db.db_best_effort():
        pass

    assert db._unreachable_until != 0.0
    with pytest.raises(db.StoreUnavailable), db.db_best_effort():
        pass  # pragma: no cover
    assert fake.attempts == 1


def test_a_saturated_pool_does_not_latch(pool, monkeypatch, capsys):
    """Connections held and none free is a queue, not an outage."""
    fake = pool(PoolTimeout("couldn't get a connection after 2.00 sec"))
    monkeypatch.setattr(db, "_pool", _FakePool(stats={"pool_size": 10, "pool_available": 0}))

    for _ in range(2):
        with pytest.raises(PoolTimeout), db.db_best_effort():
            pass  # pragma: no cover

    assert fake.attempts == 2
    assert db._unreachable_until == 0.0
    assert "postgres unreachable" not in capsys.readouterr().err


def test_a_pool_holding_no_connections_latches(pool, monkeypatch):
    """Nothing established means nothing to be queued behind: the store is down."""
    fake = pool(PoolTimeout("couldn't get a connection after 2.00 sec"))
    monkeypatch.setattr(db, "_pool", _FakePool(stats={"pool_size": 0, "pool_available": 0}))

    with pytest.raises(PoolTimeout), db.db_best_effort():
        pass  # pragma: no cover
    with pytest.raises(db.StoreUnavailable), db.db_best_effort():
        pass  # pragma: no cover

    assert fake.attempts == 1


def test_a_query_error_leaves_the_latch_open(pool):
    """The latch bounds time, not errors: a failing statement is not a dead store."""
    fake = pool()

    for _ in range(2):
        with pytest.raises(ValueError):
            with db.db_best_effort():
                raise ValueError("column does not exist")

    assert fake.attempts == 2


def test_the_cooldown_lets_one_call_re_probe_and_reopens_on_success(pool, monkeypatch):
    """A database that comes back is used again without restarting the process."""
    fake = pool()
    monkeypatch.setattr(db, "_unreachable_until", db.time.monotonic() - 1)

    with db.db_best_effort():
        pass

    assert fake.attempts == 1
    assert db._unreachable_until == 0.0


def test_an_expired_latch_that_still_fails_closes_again(pool, monkeypatch):
    fake = pool(PoolTimeout("still down"))
    monkeypatch.setattr(db, "_unreachable_until", db.time.monotonic() - 1)

    with pytest.raises(PoolTimeout), db.db_best_effort():
        pass  # pragma: no cover
    with pytest.raises(db.StoreUnavailable), db.db_best_effort():
        pass  # pragma: no cover

    assert fake.attempts == 1


def test_an_unreachable_postgres_costs_a_round_one_wait(pool, monkeypatch):
    """The acceptance criterion, at the layer that owns the guarantee.

    Every primitive still returns its neutral value, and the whole degraded round
    pays a single bounded acquisition instead of one per call.
    """
    monkeypatch.setattr(review_state.settings, "database_url", "postgres://x")
    fake = pool(PoolTimeout("couldn't get a connection after 2.00 sec"))

    assert review_state.open_findings("o/r", 7, "henry") == review_state.OpenLedger()
    assert review_state.live_coverage("o/r", 7, "henry") == []
    assert review_state.settled_findings("o/r", 7, "henry") == ()
    for _ in range(200):
        assert review_state.transition("o/r", 7, "henry", _UUID, "fixed") is False

    assert fake.timeouts == [db.BEST_EFFORT_TIMEOUT_S]


def test_pool_creation_bounds_the_migration_connection(creating_pool):
    """The half of the bound that #170 actually turns on.

    In a process whose first database contact is a best-effort call, the wait
    happens on the migration connection inside pool creation -- a bound applied
    only to the acquisition afterwards would never be reached. Both connections
    must carry the budget, so dropping ``timeout=`` from either one fails here.
    """
    made = creating_pool()

    with db.db_best_effort():
        pass

    assert made.timeouts == [db.BEST_EFFORT_TIMEOUT_S, db.BEST_EFFORT_TIMEOUT_S]


def test_a_timeout_creating_the_pool_closes_it_and_latches(creating_pool):
    """A failed creation must not publish a pool, and must still close the latch."""
    made = creating_pool(PoolTimeout("couldn't get a connection after 2.00 sec"))

    with pytest.raises(PoolTimeout), db.db_best_effort():
        pass  # pragma: no cover

    assert made.closed is True
    assert db._pool is None
    with pytest.raises(db.StoreUnavailable), db.db_best_effort():
        pass  # pragma: no cover
    assert made.attempts == 1


def test_the_latch_is_announced_once_even_if_two_branches_trip_it(capsys):
    """A/B branches are threads of one process, so both can fail before either
    has latched; the operator still gets one line, not one per branch."""
    db._latch(PoolTimeout("down"))
    db._latch(PoolTimeout("down"))

    assert capsys.readouterr().err.count("postgres unreachable") == 1
