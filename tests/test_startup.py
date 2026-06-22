"""Tests for the startup lifespan that warms (migrates) the pool before serving."""

from fastapi.testclient import TestClient

from sidecar import db, main


def test_startup_warms_pool_when_db_configured(monkeypatch):
    calls = []
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x")
    monkeypatch.setattr(db, "get_pool", lambda: calls.append("warm"))
    with TestClient(main.app):
        pass
    assert calls == ["warm"]


def test_startup_skips_when_no_db_configured(monkeypatch):
    calls = []
    monkeypatch.setattr(main.settings, "database_url", "")
    monkeypatch.setattr(db, "get_pool", lambda: calls.append("warm"))
    with TestClient(main.app):
        pass
    assert calls == []


def test_startup_db_error_does_not_block_serving(monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x")

    def boom():
        raise RuntimeError("database not ready")

    monkeypatch.setattr(db, "get_pool", boom)
    with TestClient(main.app) as client:
        assert client.get("/healthz").json() == {"ok": True}
