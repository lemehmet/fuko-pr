"""Where the embedding-provenance pass runs relative to ``_pool_lock`` (#217).

``_ensure_embed_dim`` can call ``_migrate_embed_dim``, which re-embeds the whole
store synchronously -- minutes on a real store, and guaranteed once on the first
startup after #214 because an absent ``meta.embed_model`` marker counts as a
model change. While that ran inside ``get_pool``'s lock, a runner whose first
database contact is a best-effort write (the direct-Postgres path, which has no
``main.lifespan`` warm) queued behind all of it, breaking the "never block a
review" contract ``review_state`` and ``run_metrics`` both document.

``tests/test_embed_provenance.py`` pins *what* the pass decides; this file pins
*when* it runs and *who* waits for it. Nothing else in the suite exercises
``get_pool`` with a pass pending -- ``test_db_best_effort``'s ``creating_pool``
stubs ``_ensure_embed_dim`` out precisely so the lock question stays invisible
to it.
"""

import contextlib
import threading

import pytest

import sidecar.db as db


class _FakeConn:
    def execute(self, *_a, **_k):
        return None


class _FakePool:
    """Records every acquisition; never fails."""

    def __init__(self):
        self.acquisitions = 0
        self.closed = False

    @contextlib.contextmanager
    def connection(self, timeout=None):
        self.acquisitions += 1
        yield _FakeConn()

    def get_stats(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Pool identity and pass state are process globals; no test may inherit one."""
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_embed_space_checked", False)
    monkeypatch.setattr(db, "_pending_embed_space", None)
    monkeypatch.setattr(db, "_unreachable_until", 0.0)
    monkeypatch.setattr(db, "_latch_gen", 0)
    monkeypatch.setattr(db, "register_vector", lambda conn: None)


@pytest.fixture
def creating(monkeypatch):
    """Let ``get_pool`` build a fake pool; return it and the pass's call log.

    ``_ensure_embed_dim`` is recorded rather than stubbed away, because the
    whole question here is how many times it runs and on whose call.
    """

    def install(pass_error=None):
        made = _FakePool()
        calls: list[tuple[int, bool]] = []

        def record(_conn, dim, *, probed=True):
            calls.append((dim, probed))
            if pass_error is not None:
                raise pass_error

        monkeypatch.setattr(db, "ConnectionPool", lambda **_k: made)
        monkeypatch.setattr(db, "_resolve_embed_dim", lambda: (1024, True))
        monkeypatch.setattr(db, "_migration_sql", lambda dim: ("SELECT 1",))
        monkeypatch.setattr(db, "_ensure_embed_dim", record)
        monkeypatch.setattr(db.atexit, "register", lambda fn: fn)
        return made, calls

    return install


def test_a_best_effort_first_contact_skips_the_pass(creating):
    """The acceptance criterion, at the layer that owns it.

    A runner's first best-effort write creates the pool and replays the
    idempotent DDL -- milliseconds -- and does not pay for a re-embed it has no
    use for: none of the four best-effort modules touches ``learnings`` or a
    vector.
    """
    made, calls = creating()

    with db.db_best_effort():
        pass

    assert calls == []
    assert db._pool is made


def test_the_pass_stays_pending_for_whoever_needs_the_space(creating):
    """Skipped is not cancelled: the next knowledge-base caller still runs it.

    This is what keeps the split from turning into "serve the stale space" --
    the marker's whole purpose. The best-effort caller published the pool, so
    the pass now runs behind it rather than inside its creation.
    """
    _, calls = creating()

    with db.db_best_effort():
        pass
    with db.db():
        pass

    assert calls == [(1024, True)]


def test_a_startup_warm_still_pays_it_up_front(creating):
    """``main.lifespan`` calls ``get_pool()`` bare; that must keep migrating.

    ``embed_space`` defaults to ``True`` so the sidecar's startup warm -- the
    reason #217 says "the sidecar is fine" -- is unchanged by the split.
    """
    _, calls = creating()

    db.get_pool()

    assert calls == [(1024, True)]


def test_the_pass_runs_once_however_many_callers_arrive(creating):
    """Once per process, not once per connection."""
    _, calls = creating()

    for _ in range(3):
        with db.db():
            pass

    assert calls == [(1024, True)]


def test_the_deferred_pass_reuses_the_dimension_creation_resolved(creating, monkeypatch):
    """A probe that failed at creation stays failed for the pass (#218 unmoved).

    Re-probing here would convert ``_ensure_embed_dim``'s documented "deferred
    to the next process start" into an in-process retry against an embedder
    that is by definition unreachable.
    """
    _, calls = creating()
    monkeypatch.setattr(db, "_resolve_embed_dim", lambda: (8, False))

    with db.db_best_effort():
        pass
    monkeypatch.setattr(db, "_resolve_embed_dim", lambda: (1024, True))
    with db.db():
        pass

    assert calls == [(8, False)]


def test_a_failing_pass_leaves_the_pool_up_and_the_pass_pending(creating):
    """Unlike a failed creation, a failed pass must not take the pool down.

    By then the pool is published and serving the best-effort traffic that
    never needed the pass, so the failure is left to the next caller that does.
    """
    made, calls = creating(pass_error=RuntimeError("embeddings backend refused"))

    with pytest.raises(RuntimeError):
        with db.db():
            pass  # pragma: no cover

    assert made.closed is False
    assert db._pool is made
    with db.db_best_effort():
        pass
    with pytest.raises(RuntimeError):
        with db.db():
            pass  # pragma: no cover
    assert len(calls) == 2


def test_a_best_effort_call_does_not_queue_behind_a_running_pass(creating, monkeypatch):
    """The minutes-long wait #217 is about, reproduced and shown to be gone.

    A thread, not an interleave, because the property under test is precisely
    that one caller is inside the pass while another is not blocked by it. The
    pass is held open on an event; the best-effort call must complete while it
    is still held. A regression that puts the pass back under ``_pool_lock``
    fails on the timeout rather than hanging the suite.
    """
    running = threading.Event()
    release = threading.Event()
    made, calls = creating()

    outer_ensure = db._ensure_embed_dim

    def blocking(conn, dim, *, probed=True):
        outer_ensure(conn, dim, probed=probed)
        running.set()
        release.wait(timeout=5)

    monkeypatch.setattr(db, "_ensure_embed_dim", blocking)
    finished = threading.Event()

    def best_effort_caller():
        with db.db_best_effort():
            pass
        finished.set()

    worker = threading.Thread(target=db.get_pool)
    try:
        worker.start()
        assert running.wait(timeout=5), "the pass never started"
        caller = threading.Thread(target=best_effort_caller)
        caller.start()
        assert finished.wait(timeout=5), "the best-effort call queued behind the re-embed"
        caller.join(timeout=5)
    finally:
        release.set()
        worker.join(timeout=5)

    assert calls == [(1024, True)]
    assert made.closed is False
