"""Tests for the circuit-breaker HTTP endpoints (store/db are not touched)."""

from fastapi.testclient import TestClient

from sidecar import circuit_breaker, main


def _client(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", None)  # disable bearer auth
    return TestClient(main.app)


def test_cb_cooldowns_endpoint(monkeypatch):
    monkeypatch.setattr(
        circuit_breaker, "get_cooldowns", lambda: {"zai-coding": "2026-06-21T20:00:00+00:00"}
    )
    resp = _client(monkeypatch).get("/cb/cooldowns")
    assert resp.status_code == 200
    assert resp.json() == {"cooldowns": {"zai-coding": "2026-06-21T20:00:00+00:00"}}


def test_cb_trip_endpoint(monkeypatch):
    seen = {}

    def fake_trip(provider, cooldown_seconds, reason):
        seen.update(provider=provider, cooldown_seconds=cooldown_seconds, reason=reason)
        return "2026-06-21T20:05:00+00:00"

    monkeypatch.setattr(circuit_breaker, "trip", fake_trip)
    resp = _client(monkeypatch).post(
        "/cb/trip",
        json={"provider": "zai-coding", "cooldown_seconds": 300, "reason": "429"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"provider": "zai-coding", "cooldown_until": "2026-06-21T20:05:00+00:00"}
    assert seen == {"provider": "zai-coding", "cooldown_seconds": 300, "reason": "429"}
