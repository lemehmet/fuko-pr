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


def test_migration_007_backfills_endpoint_idempotently():
    """The endpoint column mirrors 006's shape: NOT NULL DEFAULT '' backfills
    existing rows to the same "no explicit endpoint recorded" value an omitting
    caller writes, and ADD COLUMN IF NOT EXISTS keeps re-apply a no-op
    (migrations re-run on every pool creation)."""
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parent.parent / "migrations" / "007_review_run_endpoint.sql"
    ).read_text(encoding="utf-8")
    stripped = "\n".join(ln for ln in sql.splitlines() if not ln.lstrip().startswith("--"))
    stmts = [s.strip() for s in stripped.split(";") if s.strip()]
    assert len(stmts) == 1
    assert "ADD COLUMN IF NOT EXISTS endpoint" in stmts[0]
    assert "DEFAULT ''" in stmts[0] and "NOT NULL" in stmts[0]


def test_metrics_run_endpoint_carries_endpoint(monkeypatch):
    """An explicit endpoint rides the /metrics/run body through to record(),
    same guard as the backend field's."""
    monkeypatch.setattr(main.settings, "database_url", "")
    seen = {}
    monkeypatch.setattr(
        run_metrics, "record", lambda repo, pr, provider, model, **kw: seen.update(kw)
    )
    resp = _client(monkeypatch).post(
        "/metrics/run",
        json={"repo": "o/r", "pr": 7, "provider": "p", "model": "m", "endpoint": "https://e"},
    )
    assert resp.status_code == 200
    assert seen["endpoint"] == "https://e"


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


def test_migration_008_adds_nullable_token_columns():
    """#152: unlike 006/007 these columns are NOT backfilled. There is no honest
    historical value — every pre-existing row and every pr-agent row has an
    unknown cost — and a 0 would read as "this review was free"."""
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parent.parent / "migrations" / "008_review_run_tokens.sql"
    ).read_text(encoding="utf-8")
    stripped = "\n".join(ln for ln in sql.splitlines() if not ln.lstrip().startswith("--"))
    stmts = [s.strip() for s in stripped.split(";") if s.strip()]
    assert len(stmts) == 5
    added = {s.split("IF NOT EXISTS")[1].split()[0] for s in stmts}
    assert added == {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "turns",
    }
    assert all(s.startswith("ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS") for s in stmts)
    # Nullable is the whole point, and cost_usd already exists from 004.
    assert "NOT NULL" not in stripped and "DEFAULT" not in stripped
    assert "cost_usd" not in {s.split("IF NOT EXISTS")[1].split()[0] for s in stmts}


def test_metrics_run_endpoint_carries_token_costs(monkeypatch):
    """#152: the accounting rides the same body the runner already posts."""
    monkeypatch.setattr(main.settings, "database_url", "")
    seen = {}
    monkeypatch.setattr(
        run_metrics, "record", lambda repo, pr, provider, model, **kw: seen.update(kw)
    )
    resp = _client(monkeypatch).post(
        "/metrics/run",
        json={
            "repo": "o/r",
            "pr": 7,
            "provider": "p",
            "model": "m",
            "input_tokens": 29000,
            "output_tokens": 4100,
            "cache_read_tokens": 339000,
            "cache_write_tokens": 12000,
            "cost_usd": 1.23,
            "turns": 57,
        },
    )
    assert resp.status_code == 200
    assert seen["input_tokens"] == 29000 and seen["cache_read_tokens"] == 339000
    assert seen["cost_usd"] == 1.23 and seen["turns"] == 57


def test_metrics_run_endpoint_defaults_costs_to_unmeasured(monkeypatch):
    """A body from a runner that predates #152 (or from a backend with no usage
    feed) records "not measured" — never a zero that reads as free."""
    monkeypatch.setattr(main.settings, "database_url", "")
    seen = {}
    monkeypatch.setattr(
        run_metrics, "record", lambda repo, pr, provider, model, **kw: seen.update(kw)
    )
    resp = _client(monkeypatch).post(
        "/metrics/run", json={"repo": "o/r", "pr": 7, "provider": "p", "model": "m"}
    )
    assert resp.status_code == 200
    assert seen["input_tokens"] is None and seen["cost_usd"] is None and seen["turns"] is None


def test_record_run_posts_the_costs_it_was_given(monkeypatch):
    """The runner's HTTP transport must carry the accounting, not drop it."""
    from sidecar.backends.base import PRRef

    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")
    posted = {}

    class _Resp:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        runner.httpx, "post", lambda url, **kw: (posted.update(kw["json"]), _Resp())[1]
    )
    _REAL_RECORD_RUN(
        PRRef("o/r", 7, "u"),
        ModelConfig(provider="anthropic", name="claude-sonnet-5"),
        slot="dorian",
        duration_s=10.0,
        attempts=1,
        outcome="ok",
        findings=2,
        detail="",
        costs={"input_tokens": 5, "cost_usd": 0.5},
    )
    assert posted["input_tokens"] == 5 and posted["cost_usd"] == 0.5


def test_costs_of_reads_only_the_known_fields():
    """The key set is fixed by the runner, never taken from the result, so this
    can only ever produce arguments record() accepts."""
    costs = runner._costs_of(InvokeResult(returncode=0, input_tokens=7, cost_usd=1.5))
    assert costs == {
        "input_tokens": 7,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "cost_usd": 1.5,
        "turns": None,
    }


def test_costs_of_tolerates_an_older_result_shape():
    """A third-party backend returning a pre-#152 InvokeResult records "not
    measured" rather than crashing the metrics write."""

    class OldResult:
        returncode = 0

    assert set(runner._costs_of(OldResult()).values()) == {None}


def test_summary_row_mapping_preserves_unmeasured_groups():
    """sum() over an all-NULL group is NULL and must stay None: a fleet of
    unmeasured pr-agent runs is not a fleet that cost nothing."""
    from decimal import Decimal

    assert run_metrics._costs((None, None, None, None, None, None)) == {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "cost_usd": None,
        "turns": None,
    }
    # NUMERIC arrives as Decimal, which does not survive JSON serialization.
    measured = run_metrics._costs((29000, 4100, 339000, 12000, Decimal("1.2345"), 57))
    assert measured["cost_usd"] == 1.2345 and isinstance(measured["cost_usd"], float)
    assert measured["cache_read_tokens"] == 339000


def test_cost_aggregates_are_not_coalesced():
    """Guards the one line that would quietly turn "unmeasured" into "free"."""
    assert "coalesce" not in run_metrics._COST_AGGREGATES.lower()


def test_metrics_summary_endpoint(monkeypatch):
    """The response model declares the cost aggregates too (#152) — an undeclared
    key is silently DROPPED by FastAPI, so the endpoint could not return one."""
    rows = [
        {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-5",
            "runs": 12,
            "ok": 11,
            "not_ok": 1,
            "avg_duration_s": 51.0,
            "findings": 9,
            "input_tokens": 348000,
            "output_tokens": 49200,
            "cache_read_tokens": 4068000,
            "cache_write_tokens": 144000,
            "cost_usd": 14.82,
            "turns": 684,
        }
    ]
    monkeypatch.setattr(run_metrics, "summary", lambda repo=None, days=30: rows)
    resp = _client(monkeypatch).get("/metrics/summary", params={"repo": "o/r"})
    assert resp.status_code == 200
    assert resp.json() == {"summary": rows}


def test_metrics_summary_endpoint_reports_unmeasured_rows_as_null(monkeypatch):
    """A pr-agent row has no honest cost; it must come back null, not zero."""
    monkeypatch.setattr(
        run_metrics,
        "summary",
        lambda repo=None, days=30: [
            {
                "provider": "openrouter",
                "model": "m",
                "runs": 1,
                "ok": 1,
                "not_ok": 0,
                "avg_duration_s": 5.0,
                "findings": 0,
            }
        ],
    )
    row = _client(monkeypatch).get("/metrics/summary").json()["summary"][0]
    assert row["cost_usd"] is None and row["input_tokens"] is None


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
