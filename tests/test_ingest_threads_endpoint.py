"""Tests for the bounded POST /ingest-threads endpoint (the store is faked)."""

from fastapi.testclient import TestClient

from sidecar import main

_TOKEN = "test-token"


def _thread(body, path="src/a.py", login="alice"):
    return {
        "isResolved": True,
        "path": path,
        "comments": {"nodes": [{"author": {"login": login}, "body": body, "url": "u"}]},
    }


def _decline(n):
    return f"Declining — the {n} ordering here is intentional; the alternative reorders effects."


class _CappingStore:
    """Minimal store that honours ``max_new`` and dedups on text, like the real ones."""

    def __init__(self):
        self.stored: set[str] = set()
        self.embedded: list[int] = []

    def ingest(self, repo, items, *, max_new=None):
        fresh = [it for it in items if it.text not in self.stored]
        skipped = len(items) - len(fresh)
        batch = fresh if max_new is None else fresh[:max_new]
        self.embedded.append(len(batch))
        self.stored.update(it.text for it in batch)
        return len(batch), skipped


def _client(monkeypatch, store, max_new=2):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    monkeypatch.setattr(main.settings, "ingest_max_new", max_new)
    monkeypatch.setattr(main, "_store", store)
    return TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})


def test_reports_remaining_when_the_cap_bites(monkeypatch):
    store = _CappingStore()
    threads = [_thread(_decline(i)) for i in range(5)]

    body = (
        _client(monkeypatch, store)
        .post("/ingest-threads", json={"repo": "o/r", "threads": threads})
        .json()
    )

    assert body == {"considered": 5, "inserted": 2, "skipped": 0, "remaining": 3}
    assert store.embedded == [2]


def test_resending_the_same_batch_drains_to_zero(monkeypatch):
    store = _CappingStore()
    client = _client(monkeypatch, store)
    threads = [_thread(_decline(i)) for i in range(5)]

    seen = []
    for _ in range(10):
        body = client.post("/ingest-threads", json={"repo": "o/r", "threads": threads}).json()
        seen.append((body["inserted"], body["skipped"], body["remaining"]))
        if not body["remaining"]:
            break

    assert seen == [(2, 0, 3), (2, 2, 1), (1, 4, 0)]
    assert store.embedded == [2, 2, 1]


def test_remaining_is_zero_when_nothing_qualifies(monkeypatch):
    store = _CappingStore()
    threads = [_thread("Fixed in abc1234."), _thread("thanks!")]

    body = (
        _client(monkeypatch, store)
        .post("/ingest-threads", json={"repo": "o/r", "threads": threads})
        .json()
    )

    assert body == {"considered": 2, "inserted": 0, "skipped": 0, "remaining": 0}


def test_remaining_counts_only_unprocessed_not_duplicates(monkeypatch):
    store = _CappingStore()
    client = _client(monkeypatch, store, max_new=10)
    threads = [_thread(_decline(i)) for i in range(3)]

    client.post("/ingest-threads", json={"repo": "o/r", "threads": threads})
    body = client.post("/ingest-threads", json={"repo": "o/r", "threads": threads}).json()

    assert body == {"considered": 3, "inserted": 0, "skipped": 3, "remaining": 0}


def test_requires_auth(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    resp = TestClient(main.app).post("/ingest-threads", json={"repo": "o/r", "threads": []})
    assert resp.status_code == 401
