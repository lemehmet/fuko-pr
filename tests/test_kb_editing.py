"""Tests for the knowledge-base editing primitives: get, partial update, repos (#88).

The Postgres store's SQL runs against a fake connection that records statements —
enough to pin the decision logic that lives in Python (which fields are written,
when the embedder is called, how a unique collision surfaces) without a database.
The sqlite-vec store's equivalents run for real in ``test_sqlite_store.py``.
"""

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from psycopg.errors import UniqueViolation

from sidecar import ingest, main, retrieve
from sidecar.models import (
    UNSET,
    DuplicateLearningError,
    InvalidLearningError,
    UnknownSourceError,
    UpdateLearningRequest,
    check_source,
    check_text,
)

_ID = "11111111-1111-1111-1111-111111111111"
_NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)
_TOKEN = "tok"


def _row(**over):
    base = {
        "id": _ID,
        "repo": "o/r",
        "text": "original text",
        "source": "docs",
        "source_url": None,
        "file_globs": ["a.py"],
        "topic": "t",
        "created_at": _NOW,
        "expires_at": None,
    }
    base.update(over)
    return tuple(base.values())


class _FakeCursor:
    def __init__(self, result):
        self._result = result

    def fetchone(self):
        return self._result


class _FakeConn:
    """Records executed statements; replays a queued result (or an exception) per call."""

    def __init__(self, results):
        self._results = list(results)
        self.statements: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.statements.append((" ".join(sql.split()), tuple(params)))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return _FakeCursor(result)


class _Embedder:
    def __init__(self):
        self.calls: list[str] = []

    def embed_one(self, text):
        self.calls.append(text)
        return [0.1, 0.2]


@pytest.fixture
def pg(monkeypatch):
    """Patch ``ingest.db`` with a fake connection factory and stub the embedder."""
    embedder = _Embedder()
    monkeypatch.setattr(ingest, "get_embedder", lambda: embedder)

    def install(results):
        conn = _FakeConn(results)

        @contextmanager
        def fake_db():
            yield conn

        monkeypatch.setattr(ingest, "db", fake_db)
        return conn

    install.embedder = embedder
    return install


def test_check_source_accepts_known_and_rejects_unknown():
    assert check_source("review_thread") == "review_thread"
    with pytest.raises(UnknownSourceError):
        check_source("resolved_thread")


def test_update_rejects_an_unknown_source_before_touching_the_database(pg):
    conn = pg([])
    with pytest.raises(UnknownSourceError):
        ingest.update("o/r", _ID, source="nope")
    assert conn.statements == []


def test_update_returns_none_for_a_malformed_id(pg):
    conn = pg([])
    assert ingest.update("o/r", "not-a-uuid", topic="x") is None
    assert conn.statements == []


def test_update_returns_none_when_the_row_is_not_in_that_repo(pg):
    conn = pg([None])
    assert ingest.update("o/r", _ID, topic="x") is None
    assert len(conn.statements) == 1


def test_update_writes_only_the_supplied_fields(pg):
    conn = pg([("original text",), _row(topic="new")])
    updated = ingest.update("o/r", _ID, topic="new")
    assert updated["topic"] == "new"
    set_sql, params = conn.statements[1]
    assert set_sql.startswith("UPDATE learnings SET topic = %s WHERE repo = %s AND id = %s")
    assert params == ("new", "o/r", _ID)


def test_update_can_clear_a_field_to_null(pg):
    conn = pg([("original text",), _row(topic=None)])
    ingest.update("o/r", _ID, topic=None, source_url=None)
    set_sql, params = conn.statements[1]
    assert "SET source_url = %s, topic = %s WHERE" in set_sql
    assert params == (None, None, "o/r", _ID)


def test_update_re_embeds_when_the_text_actually_changes(pg):
    conn = pg([("original text",), _row(text="different text")])
    ingest.update("o/r", _ID, text="different text")
    assert pg.embedder.calls == ["different text"]
    assert "embedding = %s::vector" in conn.statements[1][0]


def test_update_skips_the_embedder_when_the_text_is_resent_unchanged(pg):
    conn = pg([("original text",), _row()])
    ingest.update("o/r", _ID, text="original text")
    assert pg.embedder.calls == []
    assert "embedding" not in conn.statements[1][0]


def test_update_skips_the_embedder_for_a_metadata_only_change(pg):
    pg([("original text",), _row(topic="new")])
    ingest.update("o/r", _ID, topic="new", file_globs=["b.py"])
    assert pg.embedder.calls == []


def test_update_maps_a_unique_collision_onto_duplicate_learning_error(pg):
    pg([("original text",), UniqueViolation("duplicate key")])
    with pytest.raises(DuplicateLearningError):
        ingest.update("o/r", _ID, text="collides")


def test_update_with_nothing_supplied_reads_the_row_back(pg, monkeypatch):
    conn = pg([])
    monkeypatch.setattr(ingest, "get_learning", lambda repo, id: {"id": id, "repo": repo})
    assert ingest.update("o/r", _ID) == {"id": _ID, "repo": "o/r"}
    assert conn.statements == []


def test_get_learning_returns_none_for_a_malformed_id():
    assert retrieve.get_learning("o/r", "not-a-uuid") is None


def test_row_to_dict_renders_timestamps_as_iso_strings():
    shaped = retrieve._row_to_dict(_row(expires_at=_NOW))
    assert shaped["created_at"] == "2026-07-01T00:00:00+00:00"
    assert shaped["expires_at"] == "2026-07-01T00:00:00+00:00"
    assert shaped["file_globs"] == ["a.py"]


def test_row_to_dict_tolerates_absent_timestamps():
    shaped = retrieve._row_to_dict(_row(created_at=None, file_globs=None))
    assert shaped["created_at"] is None and shaped["expires_at"] is None
    assert shaped["file_globs"] == []


def test_fold_repo_counts_sums_per_repo_and_sorts():
    folded = retrieve.fold_repo_counts(
        [("o/s", "docs", 5), ("o/r", "remember", 1), ("o/r", "review_thread", 2)]
    )
    assert [e["repo"] for e in folded] == ["o/r", "o/s"]
    assert folded[0] == {"repo": "o/r", "count": 3, "sources": {"remember": 1, "review_thread": 2}}


def test_update_request_reports_only_the_supplied_fields():
    req = UpdateLearningRequest(repo="o/r", topic=None)
    assert req.changes() == {"topic": None}
    assert UpdateLearningRequest(repo="o/r").changes() == {}
    assert UpdateLearningRequest(repo="o/r", text="x").changes() == {"text": "x"}


class _FakeStore:
    def __init__(self):
        self.learning = {
            "id": _ID,
            "repo": "o/r",
            "text": "t",
            "source": "docs",
            "source_url": None,
            "file_globs": [],
            "topic": None,
            "created_at": None,
            "expires_at": None,
        }
        self.raises = None
        self.seen = {}

    def get_learning(self, repo, id):
        self.seen["get"] = (repo, id)
        return self.learning if (repo, id) == ("o/r", _ID) else None

    def update_learning(self, repo, id, **changes):
        self.seen["update"] = (repo, id, changes)
        if self.raises:
            raise self.raises
        return {**self.learning, **changes} if (repo, id) == ("o/r", _ID) else None

    def repos(self):
        return [{"repo": "o/r", "count": 2, "sources": {"docs": 2}}]


@pytest.fixture
def api(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(main, "_store", store)
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    client = TestClient(main.app)
    client.headers.update({"Authorization": f"Bearer {_TOKEN}"})
    return store, client


def test_repos_endpoint_returns_the_store_summary(api):
    _, client = api
    resp = client.get("/repos")
    assert resp.status_code == 200
    assert resp.json() == {"repos": [{"repo": "o/r", "count": 2, "sources": {"docs": 2}}]}


def test_get_learning_endpoint_returns_the_row(api):
    store, client = api
    resp = client.get(f"/learnings/{_ID}?repo=o/r")
    assert resp.status_code == 200
    assert resp.json()["id"] == _ID
    assert store.seen["get"] == ("o/r", _ID)


def test_get_learning_endpoint_404s_for_another_repo(api):
    _, client = api
    assert client.get(f"/learnings/{_ID}?repo=other/repo").status_code == 404


def test_patch_forwards_only_the_supplied_fields(api):
    store, client = api
    resp = client.patch(f"/learnings/{_ID}", json={"repo": "o/r", "topic": "new"})
    assert resp.status_code == 200
    assert store.seen["update"] == ("o/r", _ID, {"topic": "new"})
    assert resp.json()["topic"] == "new"


def test_patch_forwards_an_explicit_null_as_a_clear(api):
    store, client = api
    client.patch(f"/learnings/{_ID}", json={"repo": "o/r", "source_url": None})
    assert store.seen["update"][2] == {"source_url": None}


def test_patch_404s_when_the_learning_is_not_in_that_repo(api):
    _, client = api
    assert client.patch(f"/learnings/{_ID}", json={"repo": "nope/x"}).status_code == 404


def test_patch_409s_on_a_unique_collision(api):
    store, client = api
    store.raises = DuplicateLearningError("collides")
    resp = client.patch(f"/learnings/{_ID}", json={"repo": "o/r", "text": "x"})
    assert resp.status_code == 409
    assert "collides" in resp.json()["detail"]


def test_patch_422s_on_an_unknown_source(api):
    store, client = api
    store.raises = UnknownSourceError("unknown source 'nope'")
    resp = client.patch(f"/learnings/{_ID}", json={"repo": "o/r", "source": "nope"})
    assert resp.status_code == 422


def test_new_endpoints_require_auth(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    client = TestClient(main.app)
    assert client.get("/repos").status_code == 401
    assert client.get(f"/learnings/{_ID}?repo=o/r").status_code == 401
    assert client.patch(f"/learnings/{_ID}", json={"repo": "o/r"}).status_code == 401


def test_unset_sentinel_reads_as_unset():
    assert repr(UNSET) == "UNSET"


def test_check_text_rejects_null_and_blank():
    assert check_text("real text") == "real text"
    for bad in (None, "", "   ", "\n\t", 5):
        with pytest.raises(InvalidLearningError):
            check_text(bad)


def test_unknown_source_is_an_invalid_learning():
    assert issubclass(UnknownSourceError, InvalidLearningError)


def test_update_rejects_a_null_text_before_reaching_the_embedder(pg):
    conn = pg([])
    with pytest.raises(InvalidLearningError):
        ingest.update("o/r", _ID, text=None)
    assert conn.statements == []
    assert pg.embedder.calls == []


def test_update_rejects_a_blank_text(pg):
    conn = pg([])
    with pytest.raises(InvalidLearningError):
        ingest.update("o/r", _ID, text="   ")
    assert conn.statements == []


def test_update_locks_the_row_it_reads_for_the_embed_decision(pg):
    conn = pg([("original text",), _row(topic="new")])
    ingest.update("o/r", _ID, topic="new")
    assert conn.statements[0][0].endswith("FOR UPDATE")


def test_patch_422s_on_a_null_text(api):
    store, client = api
    store.raises = InvalidLearningError("text must be a non-empty string")
    resp = client.patch(f"/learnings/{_ID}", json={"repo": "o/r", "text": None})
    assert resp.status_code == 422
    assert "non-empty" in resp.json()["detail"]


def test_like_escape_neutralizes_pattern_syntax():
    assert retrieve.like_escape("100%") == r"100\%"
    assert retrieve.like_escape("a_b") == r"a\_b"
    assert retrieve.like_escape("back\\slash") == "back\\\\slash"
    assert retrieve.like_escape("plain") == "plain"


def test_list_learnings_escapes_the_search_term_and_declares_an_escape_char(monkeypatch):
    seen = {}

    class _Cursor:
        def fetchone(self):
            return (0,)

        def fetchall(self):
            return []

    class _Conn:
        def execute(self, sql, params=()):
            seen.setdefault("sql", " ".join(sql.split()))
            seen.setdefault("params", list(params))
            return _Cursor()

    @contextmanager
    def fake_db():
        yield _Conn()

    monkeypatch.setattr(retrieve, "db", fake_db)
    retrieve.list_learnings(repo="o/r", q="100%")
    assert "ESCAPE '\\'" in seen["sql"]
    assert seen["params"][1:] == [r"%100\%%", r"%100\%%"]


def test_checked_expires_clears_on_empty_and_rejects_a_typo():
    assert ingest.checked_expires(None) is None
    assert ingest.checked_expires("") is None
    assert ingest.checked_expires("2027-01-01T00:00:00Z").year == 2027
    for bad in ("next tuesday", "2027-13-45", "01/01/2027"):
        with pytest.raises(InvalidLearningError):
            ingest.checked_expires(bad)


def test_update_rejects_a_bad_expiry_before_locking_the_row(pg):
    conn = pg([])
    with pytest.raises(InvalidLearningError):
        ingest.update("o/r", _ID, expires_at="not-a-date")
    assert conn.statements == []


def test_update_passes_a_parsed_expiry_through_to_the_write(pg):
    conn = pg([("original text",), _row()])
    ingest.update("o/r", _ID, expires_at="2027-01-01T00:00:00Z")
    set_sql, params = conn.statements[1]
    assert "SET expires_at = %s" in set_sql
    assert params[0] == datetime(2027, 1, 1, tzinfo=timezone.utc)
