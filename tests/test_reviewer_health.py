"""Tests for reviewer-health persistence, the escalation policy, and its endpoints."""

from fastapi.testclient import TestClient

from sidecar import main, reviewer_health, runner
from sidecar.backends.base import InvokeResult
from sidecar.status import escalation_needed

_TOKEN = "test-token"

_REAL_RH_STATES = runner._rh_states
_REAL_OBSERVE = runner._observe_reviewer_health


def _client(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    return TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})


def test_escalation_needed_on_each_degraded_state():
    for state in ("rate_limited", "paused", "unavailable"):
        assert escalation_needed([{"backend": "coderabbit", "state": state}])


def test_escalation_not_needed_on_healthy_states():
    rows = [
        {"backend": "coderabbit", "state": "done"},
        {"backend": "copilot", "state": "pending"},
        {"backend": "x", "state": "in_progress"},
        {"backend": "y", "state": "none"},
    ]
    assert not escalation_needed(rows)
    assert not escalation_needed([])


def test_escalation_accepts_stored_row_shape():
    rows = [
        {"reviewer": "copilot", "state": "unavailable", "observed_at": "x", "pr": 7, "detail": ""}
    ]
    assert escalation_needed(rows)


def test_reviewer_health_no_ops_without_database(monkeypatch):
    monkeypatch.setattr(reviewer_health.settings, "database_url", "")
    assert reviewer_health.observe("o/r", "coderabbit", "rate_limited", 7, "429") is None
    assert reviewer_health.states("o/r") == []


def test_rh_state_endpoint(monkeypatch):
    rows = [
        {
            "reviewer": "coderabbit",
            "state": "rate_limited",
            "observed_at": "2026-07-22T12:00:00+00:00",
            "pr": 7,
            "detail": "rate-limit notice",
        }
    ]
    monkeypatch.setattr(reviewer_health, "states", lambda repo: rows if repo == "o/r" else [])
    resp = _client(monkeypatch).get("/rh/state", params={"repo": "o/r"})
    assert resp.status_code == 200
    assert resp.json() == {"reviewers": rows}


def test_rh_observe_endpoint(monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", "")
    seen = []
    monkeypatch.setattr(
        reviewer_health,
        "observe",
        lambda repo, rev, state, pr, detail: seen.append((repo, rev, state, pr, detail)),
    )
    resp = _client(monkeypatch).post(
        "/rh/observe",
        json={
            "repo": "o/r",
            "pr": 7,
            "observations": [
                {"reviewer": "coderabbit", "state": "done", "detail": "scanned HEAD"},
                {"reviewer": "copilot", "state": "unavailable", "detail": "quota"},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"recorded": 2, "persisted": False}
    assert seen == [
        ("o/r", "coderabbit", "done", 7, "scanned HEAD"),
        ("o/r", "copilot", "unavailable", 7, "quota"),
    ]


def test_rh_observe_endpoint_reports_persisted_with_database(monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    monkeypatch.setattr(reviewer_health, "observe", lambda *a: None)
    resp = _client(monkeypatch).post(
        "/rh/observe",
        json={"repo": "o/r", "pr": 7, "observations": [{"reviewer": "copilot", "state": "done"}]},
    )
    assert resp.json() == {"recorded": 1, "persisted": True}


def test_rh_states_reads_sidecar_over_http(monkeypatch):
    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")
    monkeypatch.setenv("FUKO_TOKEN", "t")

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"reviewers": [{"reviewer": "coderabbit", "state": "paused"}]}

    monkeypatch.setattr(
        runner.httpx, "get", lambda url, params=None, headers=None, timeout=None: _Resp()
    )
    assert _REAL_RH_STATES("o/r") == [{"reviewer": "coderabbit", "state": "paused"}]


def test_rh_states_empty_on_http_error(monkeypatch):
    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")

    def boom(*a, **k):
        raise RuntimeError("sidecar down")

    monkeypatch.setattr(runner.httpx, "get", boom)
    assert _REAL_RH_STATES("o/r") == []


def _wire_review(monkeypatch, cfg_path, backend):
    monkeypatch.setattr(runner, "build_knowledge", lambda *a: "")
    monkeypatch.setattr(runner, "_cb_cooldowns", lambda: set())
    monkeypatch.setattr(runner, "_estimate_required_context", lambda *a: None)
    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: backend)
    monkeypatch.setattr(runner, "_post_branch_header", lambda *a: None)


def _one_active_one_backup(tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "anthropic"\nname = "a"\n'
        "[[review.models]]\n"
        'provider = "ollama"\nname = "b"\nrole = "backup"\n',
        encoding="utf-8",
    )
    return cfg


class _CountingBackend:
    def __init__(self):
        self.models = []

    def build_env(self, preset, model, knowledge, tools):
        self.models.append(f"{model.provider}/{model.name}")
        return {}

    def invoke(self, pr, env, tools):
        return InvokeResult(returncode=0)

    def normalize_output(self, pr, model="", *, compare_label=None, **_kw):
        return []


def test_review_promotes_backups_when_degraded(monkeypatch, tmp_path, capsys):
    cfg = _one_active_one_backup(tmp_path)
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    backend = _CountingBackend()
    _wire_review(monkeypatch, cfg, backend)
    monkeypatch.setattr(
        runner, "_rh_states", lambda repo: [{"reviewer": "coderabbit", "state": "rate_limited"}]
    )
    observed = []
    monkeypatch.setattr(
        runner, "_observe_reviewer_health", lambda pr, token, api_url: observed.append(pr.repo)
    )

    result = runner.review("https://github.com/o/r/pull/7", str(cfg))

    assert result.returncode == 0
    assert backend.models == ["anthropic/a", "ollama/b"]
    assert observed == ["o/r"]
    err = capsys.readouterr().err
    assert "promoting backup" in err


def test_review_keeps_backups_in_reserve_when_healthy(monkeypatch, tmp_path):
    cfg = _one_active_one_backup(tmp_path)
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    backend = _CountingBackend()
    _wire_review(monkeypatch, cfg, backend)
    monkeypatch.setattr(
        runner, "_rh_states", lambda repo: [{"reviewer": "coderabbit", "state": "done"}]
    )
    monkeypatch.setattr(runner, "_observe_reviewer_health", lambda pr, token, api_url: None)

    result = runner.review("https://github.com/o/r/pull/7", str(cfg))

    assert result.returncode == 0
    assert backend.models == ["anthropic/a"]


def test_observe_reviewer_health_persists_rows_locally(monkeypatch):
    from sidecar import reviewer_health as rh
    from sidecar.backends.base import PRRef

    monkeypatch.delenv("FUKO_URL", raising=False)
    monkeypatch.setattr(runner, "fetch_pr_head", lambda pr, token, api: "headsha")
    monkeypatch.setattr(runner, "fetch_issue_comments", lambda pr, token, api: [])
    monkeypatch.setattr(runner, "fetch_reviews", lambda pr, token, api: [])
    monkeypatch.setattr(runner, "fetch_check_runs", lambda pr, ref, token, api: [])
    seen = []
    monkeypatch.setattr(
        rh, "observe", lambda repo, rev, state, pr, detail: seen.append((repo, rev, state, pr))
    )

    _REAL_OBSERVE(PRRef("o/r", 7, "u"), "tok", "https://api.github.com")

    assert ("o/r", "coderabbit", "none", 7) in seen
    assert ("o/r", "copilot", "none", 7) in seen


def test_observe_reviewer_health_swallows_fetch_errors(monkeypatch, capsys):
    from sidecar.backends.base import PRRef

    def boom(*a, **k):
        raise RuntimeError("github down")

    monkeypatch.setattr(runner, "fetch_pr_head", boom)
    _REAL_OBSERVE(PRRef("o/r", 7, "u"), "tok", "https://api.github.com")
    assert "observation skipped" in capsys.readouterr().err
