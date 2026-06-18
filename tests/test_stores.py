"""Unit tests for the store factory and PostgresStore delegation."""

import pytest

from sidecar import stores
from sidecar.fukoconfig import KnowledgeConfig
from sidecar.models import IngestItem
from sidecar.stores import PostgresStore, UnknownStoreError, get_store


def test_get_store_returns_postgres_by_default():
    assert isinstance(get_store(KnowledgeConfig()), PostgresStore)


def test_get_store_rejects_unknown():
    with pytest.raises(UnknownStoreError):
        get_store(KnowledgeConfig(store="nope"))


def test_postgres_store_delegates(monkeypatch):
    calls = {}

    def fake_ingest(repo, items):
        calls["ingest"] = (repo, items)
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
    assert s.ingest("o/r", [IngestItem(text="t", source="docs")]) == (1, 0)
    assert s.query("o/r", ["a.py"]) == [{"text": "x"}]
    # the protocol's `all` maps onto ingest.forget's `all_`
    assert s.forget("o/r", all=True) == 3
    assert calls["forget"] == (None, None, True)
    assert calls["query"] == ("o/r", ["a.py"], None, None, None)
