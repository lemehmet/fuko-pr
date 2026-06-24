"""Tests for the multi-provider failover loop and circuit-breaker plumbing."""

from sidecar import circuit_breaker, runner
from sidecar.backends import pragent
from sidecar.backends.base import InvokeResult, PRRef
from sidecar.fukoconfig import FukoConfig, ModelConfig, ReviewConfig


class _FakeBackend:
    """Records which providers it was asked to build env for and returns canned results."""

    def __init__(self, results):
        self._results = list(results)
        self.models = []
        self.normalized = None

    def build_env(self, preset, model, knowledge, tools):
        self.models.append(model)
        return {}

    def invoke(self, pr, env, tools):
        return self._results.pop(0)

    def normalize_output(self, pr, model="", *, compare_label=None):
        self.normalized = model
        return []


def _two_provider_cfg(**review_kw):
    return FukoConfig(
        review=ReviewConfig(
            providers=[
                ModelConfig(provider="zai-coding", name="glm-5.2"),
                ModelConfig(provider="anthropic", name="claude-sonnet-4-6"),
            ],
            cooldown_seconds=120,
            **review_kw,
        )
    )


def _wire(monkeypatch, cfg, backend, cooled=frozenset(), required=None):
    monkeypatch.setattr(runner, "load_config", lambda p: cfg)
    monkeypatch.setattr(runner, "build_knowledge", lambda *a, **k: "")
    monkeypatch.setattr(runner, "get_backend", lambda name, c: backend)
    monkeypatch.setattr(runner, "_cb_cooldowns", lambda: set(cooled))
    monkeypatch.setattr(runner, "_estimate_required_context", lambda *a, **k: required)
    trips = []
    monkeypatch.setattr(runner, "_cb_trip", lambda prov, secs, reason: trips.append((prov, secs)))
    return trips


def test_review_fails_over_on_throttle(monkeypatch):
    backend = _FakeBackend(
        [
            InvokeResult(returncode=1, detail="429 rate limit", throttled=True),
            InvokeResult(returncode=0),
        ]
    )
    trips = _wire(monkeypatch, _two_provider_cfg(), backend)

    result = runner.review("https://github.com/o/r/pull/1")

    assert result.returncode == 0
    assert [m.provider for m in backend.models] == ["zai-coding", "anthropic"]
    assert trips == [("zai-coding", 120)]
    assert backend.normalized == "anthropic/claude-sonnet-4-6"


def test_review_non_throttle_error_does_not_fail_over(monkeypatch):
    backend = _FakeBackend([InvokeResult(returncode=2, detail="pr-agent bug", throttled=False)])
    trips = _wire(monkeypatch, _two_provider_cfg(), backend)

    result = runner.review("https://github.com/o/r/pull/1")

    assert result.returncode == 2
    assert [m.provider for m in backend.models] == ["zai-coding"]
    assert trips == []


def test_review_skips_cooled_provider(monkeypatch):
    backend = _FakeBackend([InvokeResult(returncode=0)])
    _wire(monkeypatch, _two_provider_cfg(), backend, cooled={"zai-coding"})

    result = runner.review("https://github.com/o/r/pull/1")

    assert result.returncode == 0
    assert [m.provider for m in backend.models] == ["anthropic"]


def test_review_exhausts_pool_when_all_throttle(monkeypatch):
    backend = _FakeBackend(
        [
            InvokeResult(returncode=1, detail="429", throttled=True),
            InvokeResult(returncode=1, detail="overloaded", throttled=True),
        ]
    )
    trips = _wire(monkeypatch, _two_provider_cfg(), backend)

    result = runner.review("https://github.com/o/r/pull/1")

    assert result.throttled is True
    assert {p for p, _ in trips} == {"zai-coding", "anthropic"}
    assert backend.normalized is None


def test_review_demotes_provider_that_cannot_hold_the_job(monkeypatch):
    cfg = FukoConfig(
        review=ReviewConfig(
            providers=[
                ModelConfig(provider="ollama", name="kimi", max_context=8000),
                ModelConfig(provider="anthropic", name="claude-sonnet-4-6", max_context=200000),
            ]
        )
    )
    backend = _FakeBackend([InvokeResult(returncode=0)])
    _wire(monkeypatch, cfg, backend, required=50000)

    result = runner.review("https://github.com/o/r/pull/1")

    assert result.returncode == 0
    assert [m.provider for m in backend.models] == ["anthropic"]


def test_legacy_single_model_still_runs(monkeypatch):
    cfg = FukoConfig(review=ReviewConfig(model=ModelConfig(provider="ollama", name="kimi")))
    backend = _FakeBackend([InvokeResult(returncode=0)])
    _wire(monkeypatch, cfg, backend)

    result = runner.review("https://github.com/o/r/pull/1")

    assert result.returncode == 0
    assert [m.provider for m in backend.models] == ["ollama"]
    assert backend.normalized == "ollama/kimi"


def test_invoke_detects_throttle_and_stops_early(monkeypatch):
    ran = []

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "litellm.RateLimitError: 429 Too Many Requests"

    def fake_run(cmd, env=None, check=False, timeout=None, **kw):
        ran.append(cmd[-1])
        return _Proc()

    monkeypatch.setattr(pragent.subprocess, "run", fake_run)
    pr = PRRef(repo="o/r", number=1, url="https://github.com/o/r/pull/1")
    result = pragent.PrAgentBackend().invoke(pr, {}, ["review", "improve"])

    assert result.throttled is True
    assert result.returncode == 1
    assert ran == ["review"]


def test_circuit_breaker_no_ops_without_database(monkeypatch):
    monkeypatch.setattr(circuit_breaker.settings, "database_url", "")
    assert circuit_breaker.get_cooldowns() == {}
    assert circuit_breaker.trip("zai-coding", 300, "429") is None


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_cb_cooldowns_reads_sidecar_over_http(monkeypatch):
    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")
    monkeypatch.setenv("FUKO_TOKEN", "t")
    monkeypatch.setattr(
        runner.httpx,
        "get",
        lambda url, headers=None, timeout=None: _Resp({"cooldowns": {"zai-coding": "x"}}),
    )
    assert runner._cb_cooldowns() == {"zai-coding"}


def test_cb_cooldowns_empty_on_http_error(monkeypatch):
    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")

    def boom(*a, **k):
        raise RuntimeError("sidecar down")

    monkeypatch.setattr(runner.httpx, "get", boom)
    assert runner._cb_cooldowns() == set()


def test_cb_trip_uses_local_breaker_without_url(monkeypatch):
    monkeypatch.delenv("FUKO_URL", raising=False)
    calls = []
    monkeypatch.setattr(circuit_breaker, "trip", lambda p, s, r: calls.append((p, s, r)))
    runner._cb_trip("zai-coding", 300, "429")
    assert calls == [("zai-coding", 300, "429")]
