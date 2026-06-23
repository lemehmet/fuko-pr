"""Tests for the GET /learnings listing endpoint (the store is faked)."""

from fastapi.testclient import TestClient

from sidecar import main

_TOKEN = "test-token"

_ROW = {
    "id": "1",
    "repo": "o/r",
    "text": "Declining — vitest hoists vi.mock above the imports here.",
    "source": "resolved_thread",
    "source_url": "https://example/pull/1#discussion_r1",
    "file_globs": ["a.py"],
    "topic": "review decision",
    "created_at": "2026-06-23T00:00:00+00:00",
}


def _client(monkeypatch, fake):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    monkeypatch.setattr(main._store, "list_learnings", fake)
    return TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})


def test_returns_items_and_total_count(monkeypatch):
    resp = _client(monkeypatch, lambda **kw: ([_ROW], 42)).get("/learnings?repo=o/r")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 42  # total matching, not the page size
    assert len(body["learnings"]) == 1
    assert body["learnings"][0]["text"].startswith("Declining")


def test_clamps_limit_and_offset(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return [], 0

    _client(monkeypatch, fake).get("/learnings?limit=9999&offset=-5")
    assert seen["limit"] == 500
    assert seen["offset"] == 0


def test_passes_filters_through(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return [], 0

    _client(monkeypatch, fake).get("/learnings?repo=o/r&source=resolved_thread&limit=10&offset=20")
    assert seen == {"repo": "o/r", "source": "resolved_thread", "limit": 10, "offset": 20}


def test_requires_auth(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    assert TestClient(main.app).get("/learnings").status_code == 401
