"""Tests for review-run metrics: module, endpoints, and runner recording."""

from fastapi.testclient import TestClient

from sidecar import main, run_metrics, runner
from sidecar.backends.base import InvokeResult
from sidecar.fukoconfig import ModelConfig, ReviewModel

_TOKEN = "test-token"

_REAL_RECORD_RUN = runner._record_run


def _client(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    return TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})


def test_migration_006_backfills_backend_idempotently():
    """#99: the backend column is added NOT NULL DEFAULT 'pr-agent' so existing
    review_runs rows backfill; ADD COLUMN IF NOT EXISTS keeps re-apply a no-op
    (migrations re-run on every pool creation). Guards the text without a live DB,
    matching how the repo already checks migrations."""
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parent.parent / "migrations" / "006_review_run_backend.sql"
    ).read_text(encoding="utf-8")
    # Strip line comments exactly as db._migration_sql does, then split on ';'.
    stripped = "\n".join(ln for ln in sql.splitlines() if not ln.lstrip().startswith("--"))
    stmts = [s.strip() for s in stripped.split(";") if s.strip()]
    assert len(stmts) == 2
    assert "ADD COLUMN IF NOT EXISTS backend" in stmts[0]
    assert "DEFAULT 'pr-agent'" in stmts[0] and "NOT NULL" in stmts[0]
    assert stmts[1].startswith("CREATE INDEX IF NOT EXISTS")


def test_run_metrics_no_ops_without_database(monkeypatch):
    monkeypatch.setattr(run_metrics.settings, "database_url", "")
    assert run_metrics.record("o/r", 7, "openrouter", "m") is None
    assert run_metrics.summary() == []


def test_metrics_run_endpoint(monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", "")
    seen = {}

    def fake_record(repo, pr, provider, model, **kw):
        seen.update(repo=repo, pr=pr, provider=provider, model=model, **kw)

    monkeypatch.setattr(run_metrics, "record", fake_record)
    resp = _client(monkeypatch).post(
        "/metrics/run",
        json={
            "repo": "o/r",
            "pr": 7,
            "provider": "openrouter",
            "model": "x-ai/grok-4.5",
            "slot": "sybil",
            "duration_s": 43.2,
            "attempts": 2,
            "outcome": "ok",
            "findings": 3,
            "detail": "failed over once",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"recorded": True, "persisted": False}
    assert seen["slot"] == "sybil" and seen["attempts"] == 2 and seen["findings"] == 3
    # #99: an omitted backend defaults to 'pr-agent' at the request model.
    assert seen["backend"] == "pr-agent"


def test_metrics_run_endpoint_carries_backend(monkeypatch):
    """#99: an explicit backend rides the /metrics/run body through to record()."""
    monkeypatch.setattr(main.settings, "database_url", "")
    seen = {}
    monkeypatch.setattr(
        run_metrics, "record", lambda repo, pr, provider, model, **kw: seen.update(kw)
    )
    resp = _client(monkeypatch).post(
        "/metrics/run",
        json={"repo": "o/r", "pr": 7, "provider": "p", "model": "m", "backend": "agentic"},
    )
    assert resp.status_code == 200
    assert seen["backend"] == "agentic"


def test_metrics_summary_endpoint(monkeypatch):
    rows = [
        {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-5",
            "runs": 12,
            "ok": 11,
            "not_ok": 1,
            "avg_duration_s": 51.0,
            "findings": 9,
        }
    ]
    monkeypatch.setattr(run_metrics, "summary", lambda repo=None, days=30: rows)
    resp = _client(monkeypatch).get("/metrics/summary", params={"repo": "o/r"})
    assert resp.status_code == 200
    assert resp.json() == {"summary": rows}


def test_slot_of_derives_from_token_env():
    assert (
        runner._slot_of(ReviewModel(provider="p", name="m", token_env="FUKO_GITHUB_TOKEN_DORIAN"))
        == "dorian"
    )
    assert runner._slot_of(ReviewModel(provider="p", name="m")) is None
    assert runner._slot_of(ModelConfig(provider="p", name="m")) is None
    assert runner._slot_of(ReviewModel(provider="p", name="m", token_env="MY_CUSTOM_TOKEN")) is None


def test_record_run_swallows_http_errors(monkeypatch, capsys):
    from sidecar.backends.base import PRRef

    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")

    def boom(*a, **k):
        raise RuntimeError("sidecar down")

    monkeypatch.setattr(runner.httpx, "post", boom)
    _REAL_RECORD_RUN(
        PRRef("o/r", 7, "u"),
        ModelConfig(provider="zai-coding", name="glm-5.2"),
        slot="dorian",
        duration_s=10.0,
        attempts=1,
        outcome="ok",
        findings=2,
        detail="",
    )
    assert "run-metrics record failed" in capsys.readouterr().err


def test_review_records_metrics_with_failover(monkeypatch, tmp_path):
    """A throttled primary that fails over to the backup records ONE row for the
    branch: winner model, attempts=2, the solo slot from the active's token_env."""
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "zai-coding"\nname = "glm-5.2"\ntoken_env = "FUKO_GITHUB_TOKEN_DORIAN"\n'
        "[[review.models]]\n"
        'provider = "anthropic"\nname = "claude-sonnet-4-6"\nrole = "backup"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ZAI_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    monkeypatch.setattr(runner, "build_knowledge", lambda *a: "")
    monkeypatch.setattr(runner, "_cb_cooldowns", lambda: set())
    monkeypatch.setattr(runner, "_cb_trip", lambda *a: None)
    monkeypatch.setattr(runner, "_estimate_required_context", lambda *a: None)

    results = iter(
        [
            InvokeResult(returncode=1, detail="429 rate limit", throttled=True),
            InvokeResult(returncode=0),
        ]
    )

    class FakeBackend:
        def build_env(self, preset, model, knowledge, tools):
            return {}

        def invoke(self, pr, env, tools):
            return next(results)

        def normalize_output(self, pr, model="", *, compare_label=None, **_kw):
            return ["s1", "s2", "s3"]

    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: FakeBackend())
    recorded = []
    monkeypatch.setattr(runner, "_record_run", lambda pr, model, **kw: recorded.append((model, kw)))

    result = runner.review("https://github.com/o/r/pull/7", str(cfg))

    assert result.returncode == 0
    assert len(recorded) == 1
    model, kw = recorded[0]
    assert model.provider == "anthropic"
    assert kw["attempts"] == 2
    assert kw["outcome"] == "ok"
    assert kw["findings"] == 3
    assert kw["slot"] == "dorian"
    assert kw["duration_s"] >= 0
    # #99 golden: a pr-agent-shaped config attributes its run to 'pr-agent'.
    assert kw["backend"] == "pr-agent"


def test_sequential_compare_records_per_branch_slots(monkeypatch, tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "anthropic"\nname = "a"\ntoken_env = "FUKO_GITHUB_TOKEN_BASIL"\n'
        "[[review.models]]\n"
        'provider = "ollama"\nname = "b"\ntoken_env = "FUKO_GITHUB_TOKEN_SYBIL"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    monkeypatch.delenv("FUKO_GITHUB_TOKEN_BASIL", raising=False)
    monkeypatch.delenv("FUKO_GITHUB_TOKEN_SYBIL", raising=False)
    monkeypatch.setattr(runner, "build_knowledge", lambda *a: "")
    monkeypatch.setattr(runner, "_cb_cooldowns", lambda: set())
    monkeypatch.setattr(runner, "_estimate_required_context", lambda *a: None)
    monkeypatch.setattr(runner, "_post_branch_header", lambda *a, **k: (None, None, None))

    class FakeBackend:
        def build_env(self, preset, model, knowledge, tools):
            return {}

        def invoke(self, pr, env, tools):
            return InvokeResult(returncode=0)

        def normalize_output(self, pr, model="", *, compare_label=None, **_kw):
            return []

    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: FakeBackend())
    recorded = []
    monkeypatch.setattr(runner, "_record_run", lambda pr, model, **kw: recorded.append(kw["slot"]))

    assert runner.review("https://github.com/o/r/pull/7", str(cfg)).returncode == 0
    assert recorded == ["basil", "sybil"]
