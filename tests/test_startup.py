"""Tests for the startup lifespan that warms (migrates) the pool before serving."""

import pytest
from fastapi.testclient import TestClient

from sidecar import db, embed, main


@pytest.fixture(autouse=True)
def _no_model_check(monkeypatch):
    """Neutralise the embedding-endpoint check for the pool-warming tests.

    Without this these tests reach the network: whichever embedder happens to
    listen on the configured base URL decides whether startup succeeds, so a
    developer running Ollama locally fails a test about database warming. The
    check has its own tests in test_embed_model_verification.py, and the one
    below covers its effect on startup.
    """
    monkeypatch.setattr(embed.Embedder, "verify_model", lambda self: None)


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


def test_startup_refuses_when_the_endpoint_serves_another_model(monkeypatch):
    # The one startup failure that is deliberately fatal: everything else here
    # degrades and serves /healthz, but a wrong-model endpoint fails no request
    # and silently poisons retrieval, so it has to stop the process (#220).
    def mismatch(self):
        raise embed.EmbedModelMismatch("serves 'other', not 'configured'")

    monkeypatch.setattr(embed.Embedder, "verify_model", mismatch)
    monkeypatch.setattr(main.settings, "database_url", "")
    with pytest.raises(embed.EmbedModelMismatch):
        with TestClient(main.app):
            pass
