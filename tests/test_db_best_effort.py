"""Tests for the bounded best-effort store path (#170): timeout + fast-fail latch."""

import contextlib

import pytest
from psycopg import OperationalError
from psycopg_pool import PoolTimeout

import sidecar.db as db
from sidecar import review_state

_UUID = "0f9d1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b"


class _FakePool:
    """Records every acquisition and its timeout; optionally refuses to connect."""

    def __init__(self, error=None):
        self.error = error
        self.timeouts: list[float | None] = []

    @contextlib.contextmanager
    def connection(self, timeout=None):
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        yield object()

    @property
    def attempts(self) -> int:
        return len(self.timeouts)


@pytest.fixture(autouse=True)
def _open_latch(monkeypatch):
    """The latch is process state; no test may inherit or leak a closed one."""
    monkeypatch.setattr(db, "_unreachable_until", 0.0)
    monkeypatch.setattr(db, "register_vector", lambda conn: None)


@pytest.fixture
def pool(monkeypatch):
    """Install a fake pool in place of the real one; return it for assertions."""

    def install(error=None):
        fake = _FakePool(error)
        monkeypatch.setattr(db, "get_pool", lambda *, timeout=None: fake)
        return fake

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


def test_a_dead_connection_mid_query_latches_too(pool):
    """``OperationalError`` is the other way "not answering" arrives."""
    fake = pool(OperationalError("connection failed"))

    with pytest.raises(OperationalError), db.db_best_effort():
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


def test_the_latch_is_announced_once_even_if_two_branches_trip_it(capsys):
    """A/B branches are threads of one process, so both can fail before either
    has latched; the operator still gets one line, not one per branch."""
    db._latch(PoolTimeout("down"))
    db._latch(PoolTimeout("down"))

    assert capsys.readouterr().err.count("postgres unreachable") == 1
