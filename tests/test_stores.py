"""Unit tests for the store factory and PostgresStore delegation."""

import pytest

from sidecar import stores
from sidecar.fukoconfig import KnowledgeConfig
from sidecar.models import UNSET, IngestItem
from sidecar.stores import PostgresStore, UnknownStoreError, get_store


def test_get_store_returns_postgres_by_default():
    assert isinstance(get_store(KnowledgeConfig()), PostgresStore)


def test_get_store_rejects_unknown():
    with pytest.raises(UnknownStoreError):
        get_store(KnowledgeConfig(store="nope"))


def test_postgres_store_delegates(monkeypatch):
    calls = {}

    def fake_ingest(repo, items, *, max_new=None):
        calls["ingest"] = (repo, items, max_new)
        return (1, 0)

    def fake_query(repo, files, pr_body, query_text, top_k):
        calls["query"] = (repo, files, pr_body, query_text, top_k)
        return [{"text": "x"}]

    def fake_forget(repo, *, id, source, all_):
        calls["forget"] = (id, source, all_)
        return 3

    monkeypatch.setattr(stores._ingest, "ingest", fake_ingest)
    monkeypatch.setattr(stores._retrieve, "query", fake_query)
    monkeypatch.setattr(stores._ingest, "forget", fake_forget)

    s = PostgresStore()
    assert s.ingest("o/r", [IngestItem(text="t", source="docs")], max_new=4) == (1, 0)
    assert calls["ingest"][2] == 4
    assert s.query("o/r", ["a.py"]) == [{"text": "x"}]
    # the protocol's `all` maps onto ingest.forget's `all_`
    assert s.forget("o/r", all=True) == 3
    assert calls["forget"] == (None, None, True)
    assert calls["query"] == ("o/r", ["a.py"], None, None, None)


def test_postgres_store_delegates_the_browsing_calls(monkeypatch):
    calls = {}

    def fake_list(repo, source, limit, offset, q, include_expired):
        calls["list"] = (repo, source, limit, offset, q, include_expired)
        return ([{"id": "1"}], 1)

    monkeypatch.setattr(stores._retrieve, "list_learnings", fake_list)
    monkeypatch.setattr(stores._retrieve, "get_learning", lambda repo, id: {"id": id, "repo": repo})
    monkeypatch.setattr(stores._retrieve, "repos", lambda: [{"repo": "o/r", "count": 1}])

    s = PostgresStore()
    assert s.list_learnings(repo="o/r", q="glob", include_expired=True) == ([{"id": "1"}], 1)
    assert calls["list"] == ("o/r", None, 100, 0, "glob", True)
    assert s.get_learning("o/r", "abc") == {"id": "abc", "repo": "o/r"}
    assert s.repos() == [{"repo": "o/r", "count": 1}]


def test_postgres_store_forwards_only_supplied_update_fields(monkeypatch):
    calls = {}

    def fake_update(repo, id, **kw):
        calls["update"] = (repo, id, kw)
        return {"id": id}

    monkeypatch.setattr(stores._ingest, "update", fake_update)

    assert PostgresStore().update_learning("o/r", "abc", topic="new") == {"id": "abc"}
    repo, id_, kw = calls["update"]
    assert (repo, id_) == ("o/r", "abc")
    assert kw["topic"] == "new"
    # everything the caller did not touch stays UNSET, so ingest.update skips it
    assert {name for name, value in kw.items() if value is not UNSET} == {"topic"}
