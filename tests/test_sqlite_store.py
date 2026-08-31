"""Round-trip tests for SqliteVecStore (real sqlite-vec; in-memory object sync)."""

import pytest

from sidecar import sqlite_store as ss
from sidecar.fukoconfig import KnowledgeConfig, ObjectStoreConfig
from sidecar.models import (
    DuplicateLearningError,
    IngestItem,
    InvalidLearningError,
    UnknownSourceError,
)
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


def test_reingest_skips_embedding_known_threads(store, monkeypatch):
    items = [
        IngestItem(text="auth login flow", source="review_thread"),
        IngestItem(text="db migration notes", source="review_thread"),
    ]
    assert store.ingest("o/r", items) == (2, 0)

    embedded: list[list[str]] = []

    class _CountingEmbedder(_FakeEmbedder):
        def embed(self, texts):
            embedded.append(list(texts))
            return super().embed(texts)

    monkeypatch.setattr(ss, "get_embedder", lambda: _CountingEmbedder())

    new = IngestItem(text="ui spacing rule", source="review_thread")
    assert store.ingest("o/r", items + [new]) == (1, 2)
    assert embedded == [["ui spacing rule"]]


def test_max_new_bounds_embedding_and_leaves_the_rest_unaccounted(store, monkeypatch):
    embedded: list[list[str]] = []

    class _CountingEmbedder(_FakeEmbedder):
        def embed(self, texts):
            embedded.append(list(texts))
            return super().embed(texts)

    monkeypatch.setattr(ss, "get_embedder", lambda: _CountingEmbedder())

    items = [IngestItem(text=f"learning {i}", source="review_thread") for i in range(5)]
    inserted, skipped = store.ingest("o/r", items, max_new=2)

    assert (inserted, skipped) == (2, 0)
    assert embedded == [["learning 0", "learning 1"]]
    assert len(items) - inserted - skipped == 3


def test_resending_the_same_batch_drains_the_backlog(store):
    items = [IngestItem(text=f"learning {i}", source="review_thread") for i in range(5)]

    passes = []
    for _ in range(10):
        inserted, skipped = store.ingest("o/r", items, max_new=2)
        remaining = len(items) - inserted - skipped
        passes.append((inserted, skipped, remaining))
        if not remaining:
            break

    assert passes == [(2, 0, 3), (2, 2, 1), (1, 4, 0)]
    assert store.list_learnings("o/r")[1] == 5


def test_max_new_none_embeds_everything(store):
    items = [IngestItem(text=f"learning {i}", source="review_thread") for i in range(5)]
    assert store.ingest("o/r", items, max_new=None) == (5, 0)


def test_existing_keys_batches_over_sqlite_var_limit(store, monkeypatch):
    monkeypatch.setattr(ss, "_VAR_BATCH", 50)
    big = [IngestItem(text=f"learning number {i}", source="review_thread") for i in range(120)]
    assert store.ingest("o/r", big) == (120, 0)

    extra = IngestItem(text="brand new learning", source="review_thread")
    assert store.ingest("o/r", big + [extra]) == (1, 120)


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


def _seed(store):
    store.ingest(
        "o/r",
        [
            IngestItem(text="auth login flow", source="remember", file_globs=["src/auth/**"]),
            IngestItem(text="db migration notes", source="docs", topic="Migrations"),
        ],
    )
    store.ingest("o/s", [IngestItem(text="ui spacing rule", source="docs")])
    return {r["text"]: r for r in store.list_learnings(repo="o/r")[0]}


def test_get_learning_is_repo_scoped(store):
    rows = _seed(store)
    lid = rows["auth login flow"]["id"]
    assert store.get_learning("o/r", lid)["text"] == "auth login flow"
    assert store.get_learning("o/s", lid) is None
    assert store.get_learning("o/r", "no-such-id") is None


def test_repos_counts_match_a_full_listing(store):
    _seed(store)
    summary = {e["repo"]: e for e in store.repos()}
    assert [e["repo"] for e in store.repos()] == ["o/r", "o/s"]
    assert summary["o/r"]["count"] == store.list_learnings(repo="o/r")[1]
    assert summary["o/r"]["sources"] == {"remember": 1, "docs": 1}
    assert summary["o/s"]["sources"] == {"docs": 1}


def test_list_learnings_searches_text_and_topic_case_insensitively(store):
    _seed(store)
    assert [r["text"] for r in store.list_learnings(repo="o/r", q="LOGIN")[0]] == [
        "auth login flow"
    ]
    assert [r["text"] for r in store.list_learnings(repo="o/r", q="migrations")[0]] == [
        "db migration notes"
    ]
    assert store.list_learnings(repo="o/r", q="nothing here")[1] == 0


def test_expired_learnings_are_hidden_unless_asked_for(store):
    store.ingest(
        "o/r",
        [
            IngestItem(text="auth still valid", source="docs"),
            IngestItem(text="auth long gone", source="docs", expires_at="2020-01-01T00:00:00Z"),
        ],
    )
    live, live_total = store.list_learnings(repo="o/r")
    assert [r["text"] for r in live] == ["auth still valid"] and live_total == 1

    everything, total = store.list_learnings(repo="o/r", include_expired=True)
    assert {r["text"] for r in everything} == {"auth still valid", "auth long gone"}
    assert total == 2

    expired_id = next(r["id"] for r in everything if r["text"] == "auth long gone")
    assert store.get_learning("o/r", expired_id)["text"] == "auth long gone"
    assert store.repos()[0]["count"] == 1


def test_update_writes_only_the_supplied_fields(store):
    lid = _seed(store)["db migration notes"]["id"]
    updated = store.update_learning("o/r", lid, topic="Schema", file_globs=["migrations/*.sql"])
    assert updated["topic"] == "Schema"
    assert updated["file_globs"] == ["migrations/*.sql"]
    assert updated["text"] == "db migration notes"
    assert updated["source"] == "docs"
    assert store.get_learning("o/r", lid)["topic"] == "Schema"


def test_update_can_clear_a_field(store):
    lid = _seed(store)["db migration notes"]["id"]
    assert store.update_learning("o/r", lid, topic=None)["topic"] is None
    assert store.update_learning("o/r", lid, file_globs=[])["file_globs"] == []


def test_update_re_embeds_only_on_a_text_change(store, monkeypatch):
    lid = _seed(store)["auth login flow"]["id"]
    embedded: list[str] = []

    class _Counting(_FakeEmbedder):
        def embed_one(self, text):
            embedded.append(text)
            return _vec(text)

    monkeypatch.setattr(ss, "get_embedder", lambda: _Counting())

    store.update_learning("o/r", lid, topic="Auth")
    assert embedded == []

    store.update_learning("o/r", lid, text="ui spacing instead")
    assert embedded == ["ui spacing instead"]
    hits = store.query("o/r", ["src/auth/login.py"], query_text="ui spacing")
    assert hits[0]["text"] == "ui spacing instead"


def test_update_rejects_an_unknown_source(store):
    lid = _seed(store)["auth login flow"]["id"]
    with pytest.raises(UnknownSourceError):
        store.update_learning("o/r", lid, source="resolved_thread")
    assert store.get_learning("o/r", lid)["source"] == "remember"


def test_update_reports_a_unique_collision(store):
    rows = _seed(store)
    lid = rows["auth login flow"]["id"]
    with pytest.raises(DuplicateLearningError):
        store.update_learning("o/r", lid, text="db migration notes", source="docs")
    assert store.get_learning("o/r", lid)["text"] == "auth login flow"


def test_update_returns_none_for_a_missing_or_foreign_row(store):
    lid = _seed(store)["auth login flow"]["id"]
    assert store.update_learning("o/r", "no-such-id", topic="x") is None
    assert store.update_learning("o/s", lid, topic="x") is None


def test_update_with_nothing_supplied_is_a_read(store):
    lid = _seed(store)["auth login flow"]["id"]
    assert store.update_learning("o/r", lid) == store.get_learning("o/r", lid)


def test_update_rejects_a_null_or_blank_text(store):
    lid = _seed(store)["auth login flow"]["id"]
    for bad in (None, "  "):
        with pytest.raises(InvalidLearningError):
            store.update_learning("o/r", lid, text=bad)
    assert store.get_learning("o/r", lid)["text"] == "auth login flow"


def test_update_embeds_once_across_lost_races(store, monkeypatch):
    lid = _seed(store)["auth login flow"]["id"]
    embedded: list[str] = []

    class _Counting(_FakeEmbedder):
        def embed_one(self, text):
            embedded.append(text)
            return _vec(text)

    monkeypatch.setattr(ss, "get_embedder", lambda: _Counting())
    store._obj.fail_next_saves = 2  # lose two races, then win

    updated = store.update_learning("o/r", lid, text="db notes instead")
    assert updated["text"] == "db notes instead"
    assert embedded == ["db notes instead"]


def test_update_reads_the_stored_text_inside_the_mutation(store, monkeypatch):
    """A concurrent writer must not leave this write's text beside its embedding."""
    lid = _seed(store)["auth login flow"]["id"]
    seen: list[str] = []
    real_open = store._open

    def spy_open(path, dim):
        conn, migrated = real_open(path, dim)
        row = conn.execute("SELECT text FROM learnings WHERE lid = ?", (lid,)).fetchone()
        if row:
            seen.append(row[0])
        return conn, migrated

    monkeypatch.setattr(store, "_open", spy_open)
    store.update_learning("o/r", lid, topic="Auth")
    # exactly one connection is opened for the mutation: the decision cannot be
    # based on a snapshot from an earlier, separate read
    assert seen == ["auth login flow"]


def test_search_treats_like_metacharacters_literally(store):
    store.ingest(
        "o/r",
        [
            IngestItem(text="auth coverage is 100% on this path", source="docs"),
            IngestItem(text="auth coverage is 55 percent elsewhere", source="docs"),
            IngestItem(text="db a_b naming convention", source="docs"),
            IngestItem(text="db axb naming convention", source="docs"),
        ],
    )
    hits, total = store.list_learnings(repo="o/r", q="100%")
    assert total == 1 and hits[0]["text"].startswith("auth coverage is 100%")

    hits, total = store.list_learnings(repo="o/r", q="a_b")
    assert total == 1 and "a_b" in hits[0]["text"]

    # a bare % must not become a match-everything wildcard
    assert store.list_learnings(repo="o/r", q="%")[1] == 1


def test_update_rejects_an_unparseable_expiry_instead_of_clearing_it(store):
    lid = _seed(store)["auth login flow"]["id"]
    store.update_learning("o/r", lid, expires_at="2099-01-01T00:00:00Z")
    assert store.get_learning("o/r", lid)["expires_at"].startswith("2099-01-01")

    with pytest.raises(InvalidLearningError):
        store.update_learning("o/r", lid, expires_at="next tuesday")
    assert store.get_learning("o/r", lid)["expires_at"].startswith("2099-01-01")


def test_update_still_clears_an_expiry_with_an_empty_value(store):
    lid = _seed(store)["auth login flow"]["id"]
    store.update_learning("o/r", lid, expires_at="2099-01-01T00:00:00Z")
    assert store.update_learning("o/r", lid, expires_at=None)["expires_at"] is None


def test_search_is_case_insensitive_beyond_ascii(store):
    store.ingest(
        "o/r",
        [
            IngestItem(text="auth ÄPFEL naming in the German docs", source="docs"),
            IngestItem(text="db STRASSE unrelated", source="docs"),
        ],
    )
    hits, total = store.list_learnings(repo="o/r", q="äpfel")
    assert total == 1 and "ÄPFEL" in hits[0]["text"]
    assert store.list_learnings(repo="o/r", q="ÄPFEL")[1] == 1


def test_digests_are_dark_until_retrieval_is_enabled(store, monkeypatch):
    """A populated digest reaches no review until the deployment opts in (#158)."""
    store.ingest(
        "o/r",
        [
            IngestItem(
                text="auth login flow notes", source="remember", file_globs=["src/auth/a.rs"]
            ),
            IngestItem(
                text="Structural index of src/auth/a.rs -- auth",
                source="digest",
                file_globs=["src/auth/a.rs"],
                topic="file-index:src/auth/a.rs@abc123abc123",
            ),
        ],
    )

    dark = store.query("o/r", ["src/auth/a.rs"], query_text="auth")
    assert {r["source"] for r in dark} == {"remember"}

    monkeypatch.setattr(ss.settings, "digest_retrieval", True)
    lit = store.query("o/r", ["src/auth/a.rs"], query_text="auth")
    assert {r["source"] for r in lit} == {"remember", "digest"}


def test_a_dark_digest_does_not_displace_a_learning_from_the_candidate_window(store, monkeypatch):
    """Filtering digests after the KNN must not cost a real learning its slot.

    ``vec_learnings`` has no ``source`` column, so the exclusion can only happen
    post-KNN; the window is widened by the digest count to compensate. With a
    candidate window of one and a digest scoring at least as well, the un-widened
    query returned nothing at all.
    """
    monkeypatch.setattr(ss.settings, "candidate_k", 1)
    store.ingest(
        "o/r",
        [
            IngestItem(text="auth digest index", source="digest", file_globs=["src/auth/a.rs"]),
            IngestItem(text="auth convention", source="remember", file_globs=["src/auth/a.rs"]),
        ],
    )

    dark = store.query("o/r", ["src/auth/a.rs"], query_text="auth")
    assert [r["source"] for r in dark] == ["remember"]


def test_digest_survives_a_source_round_trip(store):
    store.ingest(
        "o/r",
        [IngestItem(text="auth index", source="digest", file_globs=["src/auth/a.rs"])],
    )
    rows, total = store.list_learnings(repo="o/r", source="digest")
    assert total == 1 and rows[0]["source"] == "digest"
