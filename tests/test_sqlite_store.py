"""Round-trip tests for SqliteVecStore (real sqlite-vec; in-memory object sync)."""

import pytest

from sidecar import sqlite_store as ss
from sidecar.fukoconfig import KnowledgeConfig, ObjectStoreConfig
from sidecar.models import IngestItem
from sidecar.objectstore import PreconditionFailed
from sidecar.stores import get_store

DIM = 3


def _vec(text: str) -> list[float]:
    t = text.lower()
    return [float("auth" in t), float("db" in t), float("ui" in t)]


class _FakeEmbedder:
    def embed(self, texts):
        return [_vec(t) for t in texts]

    def embed_one(self, text):
        return _vec(text)

    def probe_dim(self):
        return DIM


class _MemObj:
    """In-memory object store with controllable conflicts, for the sync layer."""

    def __init__(self):
        self.data = None
        self.token = None
        self.fail_next_saves = 0
        self._n = 0

    def load(self):
        return self.data, self.token

    def save(self, data, token):
        if token != self.token:
            raise PreconditionFailed("stale")
        if self.fail_next_saves > 0:
            self.fail_next_saves -= 1
            raise PreconditionFailed("simulated race")
        self._n += 1
        self.data, self.token = data, str(self._n)
        return self.token


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "get_embedder", lambda: _FakeEmbedder())
    cfg = KnowledgeConfig(
        store="sqlite-vec",
        object_store=ObjectStoreConfig(backend="file", key=str(tmp_path / "kb.db")),
    )
    s = ss.SqliteVecStore(cfg)
    s._obj = _MemObj()  # swap the file sync for the in-memory one
    return s


def test_requires_object_store():
    with pytest.raises(ValueError):
        ss.SqliteVecStore(KnowledgeConfig(store="sqlite-vec", object_store=None))


def test_get_store_dispatches_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "get_embedder", lambda: _FakeEmbedder())
    cfg = KnowledgeConfig(
        store="sqlite-vec",
        object_store=ObjectStoreConfig(backend="file", key=str(tmp_path / "kb.db")),
    )
    assert isinstance(get_store(cfg), ss.SqliteVecStore)


def test_list_learnings(store):
    store.ingest(
        "o/r",
        [
            IngestItem(text="auth login flow notes", source="remember", file_globs=["src/auth/**"]),
            IngestItem(text="db migration notes here", source="docs"),
        ],
    )
    store.ingest("other/repo", [IngestItem(text="ui spacing convention", source="remember")])

    all_for_repo, total = store.list_learnings(repo="o/r")
    assert {r["text"] for r in all_for_repo} == {"auth login flow notes", "db migration notes here"}
    assert total == 2

    docs_only, docs_total = store.list_learnings(repo="o/r", source="docs")
    assert [r["text"] for r in docs_only] == ["db migration notes here"]
    assert docs_only[0]["file_globs"] == []
    assert docs_total == 1

    page, page_total = store.list_learnings(repo="o/r", limit=1)
    assert len(page) == 1 and page_total == 2  # one row returned, two total


def test_ingest_dedup_and_query_scoping(store):
    ins, skip = store.ingest(
        "o/r",
        [
            IngestItem(text="auth login flow", source="remember", file_globs=["src/auth/**"]),
            IngestItem(text="db migration notes", source="docs"),
            IngestItem(text="auth login flow", source="remember"),  # dup (repo,text,source)
        ],
    )
    assert (ins, skip) == (2, 1)

    # auth query with a matching changed file: scoped learning passes and ranks first
    res = store.query("o/r", ["src/auth/login.py"], pr_body="fixing auth")
    assert res[0]["text"] == "auth login flow"
    assert res[0]["score"] == pytest.approx(1.0)

    # the scoped auth learning is filtered out when no changed file matches its glob
    res2 = store.query("o/r", ["db/schema.sql"], query_text="database")
    assert [r["text"] for r in res2] == ["db migration notes"]


def test_ingest_empty(store):
    assert store.ingest("o/r", []) == (0, 0)


def test_query_empty_when_no_context(store):
    store.ingest("o/r", [IngestItem(text="auth", source="docs")])
    assert store.query("o/r", [], pr_body=None, query_text=None) == []


def test_query_on_empty_store(store):
    # non-empty query text, but nothing ingested -> KNN returns nothing
    assert store.query("o/r", [], query_text="auth") == []


class _Embedder4:
    """A different embedding model: dimension 4 (adds a constant 4th component)."""

    def embed(self, texts):
        return [[*_vec(t), 0.5] for t in texts]

    def embed_one(self, text):
        return self.embed([text])[0]

    def probe_dim(self):
        return 4


def _file_cfg(tmp_path):
    return KnowledgeConfig(
        store="sqlite-vec",
        object_store=ObjectStoreConfig(backend="file", key=str(tmp_path / "kb.db")),
    )


def test_dim_migration_reembeds_and_persists(tmp_path, monkeypatch):
    cfg = _file_cfg(tmp_path)
    monkeypatch.setattr(ss, "get_embedder", lambda: _FakeEmbedder())  # dim 3
    ss.SqliteVecStore(cfg).ingest("o/r", [IngestItem(text="auth flow", source="docs")])

    # the embedding model changes (dim 3 -> 4): a query must auto re-embed + rebuild
    monkeypatch.setattr(ss, "get_embedder", lambda: _Embedder4())
    res = ss.SqliteVecStore(cfg).query("o/r", [], query_text="auth")
    assert [r["text"] for r in res] == ["auth flow"]

    # the re-embed was persisted: a fresh store at dim 4 works without re-migrating,
    # and writes still succeed afterward
    s3 = ss.SqliteVecStore(cfg)
    assert s3.ingest("o/r", [IngestItem(text="db notes", source="docs")]) == (1, 0)
    assert {r["text"] for r in s3.query("o/r", [], query_text="auth db")} >= {
        "auth flow",
        "db notes",
    }


def test_dim_migration_on_empty_store(tmp_path, monkeypatch):
    cfg = _file_cfg(tmp_path)
    monkeypatch.setattr(ss, "get_embedder", lambda: _FakeEmbedder())  # dim 3
    s = ss.SqliteVecStore(cfg)
    s.ingest("o/r", [IngestItem(text="auth", source="docs")])  # persist a dim-3 file
    s.forget("o/r", all=True)  # 0 rows remain, but the file + meta(dim=3) persist
    monkeypatch.setattr(ss, "get_embedder", lambda: _Embedder4())  # dim 4, nothing to re-embed
    assert ss.SqliteVecStore(cfg).query("o/r", [], query_text="auth") == []


def test_read_migration_persist_tolerates_race(tmp_path, monkeypatch):
    mem = _MemObj()
    cfg = _file_cfg(tmp_path)
    monkeypatch.setattr(ss, "get_embedder", lambda: _FakeEmbedder())  # dim 3
    s1 = ss.SqliteVecStore(cfg)
    s1._obj = mem
    s1.ingest("o/r", [IngestItem(text="auth", source="docs")])

    # a later process sees the new model (dim 4); its read migrates, and the
    # migration-persist loses a race -> swallowed, results still correct
    monkeypatch.setattr(ss, "get_embedder", lambda: _Embedder4())
    s2 = ss.SqliteVecStore(cfg)  # fresh instance -> re-probes dim 4
    s2._obj = mem
    mem.fail_next_saves = 1
    res = s2.query("o/r", [], query_text="auth")
    assert [r["text"] for r in res] == ["auth"]


def test_expires_at_normalization_and_filtering(store):
    from datetime import datetime, timedelta, timezone

    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    store.ingest(
        "o/r",
        [
            IngestItem(text="auth expired", source="docs", expires_at=past),
            IngestItem(text="auth fresh", source="docs", expires_at=future),
            IngestItem(text="auth garbage", source="docs", expires_at="not-a-date"),
        ],
    )
    texts = {r["text"] for r in store.query("o/r", [], query_text="auth")}
    assert "auth expired" not in texts  # past expiry filtered out
    assert "auth fresh" in texts  # future expiry kept
    assert "auth garbage" in texts  # unparseable -> stored NULL -> never expires


def test_forget_by_source_id_all(store):
    store.ingest(
        "o/r",
        [
            IngestItem(text="auth one", source="remember"),
            IngestItem(text="db two", source="docs"),
        ],
    )
    assert store.forget("o/r", source="docs") == 1
    # the deleted 'db two' is gone (the store has no score threshold, so an
    # unrelated low-score learning may still come back — just not the deleted one)
    assert all(r["text"] != "db two" for r in store.query("o/r", [], query_text="db two"))

    rows = store.query("o/r", [], query_text="auth one")
    assert store.forget("o/r", id=rows[0]["id"]) == 1
    assert store.forget("o/r", all=True) == 0  # nothing left
    assert store.forget("o/r") == 0  # no selector


def test_mutate_retries_then_succeeds(store):
    store._obj.fail_next_saves = 2  # lose two races, then win
    ins, _ = store.ingest("o/r", [IngestItem(text="auth", source="docs")])
    assert ins == 1
    assert [r["text"] for r in store.query("o/r", [], query_text="auth")] == ["auth"]


def test_mutate_gives_up_after_max_retries(store):
    store._obj.fail_next_saves = 99
    with pytest.raises(PreconditionFailed):
        store.ingest("o/r", [IngestItem(text="auth", source="docs")])
