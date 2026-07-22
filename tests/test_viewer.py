"""Tests for the unauthenticated metrics viewer page (issue #71)."""

from fastapi.testclient import TestClient

from sidecar import circuit_breaker, main, reviewer_health, run_metrics

_SUMMARY = [
    {
        "provider": "openrouter",
        "model": "x-ai/grok-4.5",
        "runs": 4,
        "ok": 4,
        "not_ok": 0,
        "avg_duration_s": 25.1,
        "findings": 2,
    }
]
_SLOTS = [{"slot": "sybil", "runs": 4, "ok": 4, "not_ok": 0, "avg_duration_s": 25.1, "findings": 2}]
_RECENT = [
    {
        "repo": "lemehmet/shuanda",
        "pr": 74,
        "provider": "lemonade",
        "model": "Qwen3-Coder-Next-GGUF",
        "slot": "gray",
        "started_at": "2026-07-22T20:15:00+00:00",
        "duration_s": 81.2,
        "attempts": 1,
        "outcome": "ok",
        "findings": 1,
    }
]
_HEALTH = [
    {
        "repo": "lemehmet/shuanda",
        "reviewer": "coderabbit",
        "state": "rate_limited",
        "observed_at": "2026-07-22T20:16:00+00:00",
        "pr": 74,
        "detail": "rate-limit notice",
    }
]


def _wire(monkeypatch, *, summary=None, slots=None, recent=None, health=None, cooldowns=None):
    seen = {}
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")

    def fake_summary(repo=None, days=30):
        seen["summary"] = (repo, days)
        return summary or []

    monkeypatch.setattr(run_metrics, "summary", fake_summary)
    monkeypatch.setattr(run_metrics, "slot_summary", lambda repo=None, days=30: slots or [])
    monkeypatch.setattr(run_metrics, "recent_runs", lambda repo=None, limit=50: recent or [])
    monkeypatch.setattr(reviewer_health, "all_states", lambda: health or [])
    monkeypatch.setattr(circuit_breaker, "get_cooldowns", lambda: cooldowns or {})
    return seen, TestClient(main.app)


def test_view_is_open_without_auth_and_renders_all_sections(monkeypatch):
    _, client = _wire(
        monkeypatch,
        summary=_SUMMARY,
        slots=_SLOTS,
        recent=_RECENT,
        health=_HEALTH,
        cooldowns={"zai-coding": "2026-07-22T21:00:00+00:00"},
    )
    resp = client.get("/metrics/view")
    assert resp.status_code == 200
    page = resp.text
    assert "x-ai/grok-4.5" in page and "25.1" in page
    assert "sybil" in page
    assert 'href="https://github.com/lemehmet/shuanda/pull/74"' in page
    assert "rate_limited" in page and 'class="bad"' in page
    assert "zai-coding" in page and "2026-07-22T21:00" in page


def test_view_passes_filters_through(monkeypatch):
    seen, client = _wire(monkeypatch, summary=_SUMMARY)
    resp = client.get("/metrics/view", params={"repo": "lemehmet/mepro", "days": "7"})
    assert resp.status_code == 200
    assert seen["summary"] == ("lemehmet/mepro", 7)


def test_view_clamps_days(monkeypatch):
    seen, client = _wire(monkeypatch)
    assert client.get("/metrics/view", params={"days": "999999"}).status_code == 200
    assert seen["summary"] == (None, 3650)


def test_view_empty_data_renders_notices(monkeypatch):
    _, client = _wire(monkeypatch)
    resp = client.get("/metrics/view")
    assert resp.status_code == 200
    assert "no runs in this window" in resp.text
    assert "no providers cooling down" in resp.text


def test_view_without_database_renders_notice(monkeypatch):
    _, client = _wire(monkeypatch)
    monkeypatch.setattr(main.settings, "database_url", "")
    resp = client.get("/metrics/view")
    assert resp.status_code == 200
    assert "No database configured" in resp.text


def test_view_escapes_db_sourced_strings(monkeypatch):
    evil = dict(_RECENT[0], repo='lemehmet/x"><script>alert(1)</script>')
    _, client = _wire(monkeypatch, recent=[evil])
    page = client.get("/metrics/view").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_api_endpoints_remain_authed(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", "t")
    client = TestClient(main.app)
    for path in ("/metrics/summary", "/rh/state?repo=o%2Fr", "/cb/cooldowns"):
        assert client.get(path).status_code == 401


def test_view_with_unreachable_database_renders_notice(monkeypatch):
    _, client = _wire(monkeypatch)

    def boom(repo=None, days=30):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(run_metrics, "summary", boom)
    resp = client.get("/metrics/view")
    assert resp.status_code == 200
    assert "Database unreachable" in resp.text
