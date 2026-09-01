"""The Postgres half of the embedding-provenance guard (``db._ensure_embed_dim``).

The decision this file pins is the one the ``meta.embed_model`` marker exists
for: re-embed when the model changed at the same dimension, because pgvector's
``atttypmod`` cannot tell 1024-wide bge-m3 from 1024-wide Qwen3-Embedding-0.6B
and a store holding both retrieves badly instead of failing.

The sqlite twin has ``test_same_dimension_model_swap_still_re_embeds``; without
this file the *default* store's branch is exercised by nothing — the only other
references to ``_ensure_embed_dim`` in the suite stub it out
(``test_db_best_effort``) or reach past it to ``_migrate_embed_dim`` behind a
live-database skip (``test_integration``). An inverted condition there would
keep mixed-space stores with every test green.

Driven through a fake connection rather than a live one on purpose: the
statement stream is fully deterministic, so the decision is testable without
pgvector, and the test runs in the default suite where a live-DB one would skip.
"""

import pytest

import sidecar.db as db

_DIM_SQL = "atttypmod"
_MODEL_SQL = "FROM meta WHERE key = 'embed_model'"
_ROWS_SQL = "FROM learnings LIMIT 1"


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    """Answers the three reads ``_ensure_embed_dim`` makes; records every write.

    Routed on SQL substrings rather than call order so a reordering of the
    reads inside ``_ensure_embed_dim`` cannot silently change what the test
    thinks it is answering.
    """

    def __init__(self, *, dim, model, has_rows):
        self._dim = dim
        self._model = model
        self._has_rows = has_rows
        self.writes: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        if _DIM_SQL in sql:
            return _Result(None if self._dim is None else (self._dim,))
        if _MODEL_SQL in sql:
            return _Result(None if self._model is None else (self._model,))
        if _ROWS_SQL in sql:
            return _Result((1,) if self._has_rows else None)
        self.writes.append((sql, params))
        return _Result(None)

    @property
    def recorded_model(self):
        """The model written to ``meta``, or ``None`` if nothing was written."""
        for sql, params in self.writes:
            if "INSERT INTO meta" in sql:
                return params[0]
        return None


@pytest.fixture
def migrations(monkeypatch):
    """Record ``_migrate_embed_dim`` calls instead of running them."""
    calls: list[int] = []
    monkeypatch.setattr(db, "_migrate_embed_dim", lambda conn, dim: calls.append(dim))
    monkeypatch.setattr(db.settings, "embed_model", "qwen3-embedding-0.6b")
    return calls


def test_same_dimension_model_swap_re_embeds(migrations):
    # The case the whole marker exists for: the dimension check sees nothing.
    conn = _FakeConn(dim=1024, model="bge-m3", has_rows=True)
    db._ensure_embed_dim(conn, 1024)
    assert migrations == [1024]
    assert conn.recorded_model == "qwen3-embedding-0.6b"


def test_absent_marker_on_a_populated_store_re_embeds(migrations):
    # Provenance cannot be proven for vectors written before the meta table, so
    # an absent marker counts as a change rather than as agreement.
    conn = _FakeConn(dim=1024, model=None, has_rows=True)
    db._ensure_embed_dim(conn, 1024)
    assert migrations == [1024]
    assert conn.recorded_model == "qwen3-embedding-0.6b"


def test_matching_marker_and_dimension_does_nothing(migrations):
    conn = _FakeConn(dim=1024, model="qwen3-embedding-0.6b", has_rows=True)
    db._ensure_embed_dim(conn, 1024)
    assert migrations == []
    assert conn.recorded_model == "qwen3-embedding-0.6b"


def test_dimension_change_re_embeds_even_on_an_empty_store(migrations):
    # A dimension change has to rebuild the column and index whether or not
    # there is anything to re-embed -- unlike a model change, which does not.
    conn = _FakeConn(dim=768, model="qwen3-embedding-0.6b", has_rows=False)
    db._ensure_embed_dim(conn, 1024)
    assert migrations == [1024]


def test_absent_marker_on_an_empty_store_is_marked_not_rebuilt(migrations):
    # Every fresh database lands here: the migrations just created the column
    # and index at the right dimension and an empty meta table, so the
    # absent-marker rule would otherwise drop and re-add schema created
    # moments earlier. There are no vectors to be wrong about.
    conn = _FakeConn(dim=1024, model=None, has_rows=False)
    db._ensure_embed_dim(conn, 1024)
    assert migrations == []
    assert conn.recorded_model == "qwen3-embedding-0.6b"


def test_an_unprobed_dimension_defers_everything(migrations):
    # `dim` came from the FUKO_EMBED_DIM fallback, which means the embedder is
    # unreachable. Re-embedding needs it, so the pool would fail to open at all
    # -- taking /forget and the best-effort state paths down with it, the exact
    # outage the fallback exists to avoid. The marker is withheld too: it must
    # only ever describe vectors this process actually produced.
    conn = _FakeConn(dim=1024, model="bge-m3", has_rows=True)
    db._ensure_embed_dim(conn, 1024, probed=False)
    assert migrations == []
    assert conn.recorded_model is None


def test_resolve_embed_dim_reports_a_failed_probe(monkeypatch):
    class _Down:
        def probe_dim(self):
            raise RuntimeError("embeddings backend unreachable")

    monkeypatch.setattr("sidecar.embed.get_embedder", lambda: _Down())
    monkeypatch.setattr(db.settings, "embed_dim", 1024)
    assert db._resolve_embed_dim() == (1024, False)


def test_resolve_embed_dim_reports_a_live_probe(monkeypatch):
    class _Up:
        def probe_dim(self):
            return 768

    monkeypatch.setattr("sidecar.embed.get_embedder", lambda: _Up())
    assert db._resolve_embed_dim() == (768, True)
