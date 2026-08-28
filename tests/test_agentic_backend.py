"""Unit tests for the agentic review backend driver."""

import json

import httpx
import pytest

from sidecar.backends import agentic as agentic_mod
from sidecar.backends import get_backend
from sidecar.backends.agentic import AgenticBackend
from sidecar.backends.base import PRRef
from sidecar.fukoconfig import ModelConfig, ReviewConfig
from sidecar.presets import get_preset
from sidecar.reviewer.checkout import PRContext
from sidecar.reviewer.harness import HarnessResult
from sidecar.signals import extract_markers

PR = PRRef(repo="o/r", number=9, url="https://github.com/o/r/pull/9")

REVIEW_JSON = json.dumps(
    {
        "summary": "looked closely",
        "findings": [
            {
                "file": "src/app.py",
                "line": 4,
                "severity": "high",
                "category": "bug",
                "title": "leak",
                "body": "closes nothing",
                "evidence": "read src/app.py:1-40",
                "confidence": "high",
            },
            {
                "file": "docs/other.md",
                "line": None,
                "severity": "low",
                "category": "docs",
                "title": "stale doc",
                "body": "update it",
                "confidence": "medium",
            },
            {
                "file": "src/app.py",
                "line": 8,
                "title": "hunch",
                "body": "maybe",
                "confidence": "low",
            },
        ],
    }
)


def _ctx() -> PRContext:
    return PRContext(
        title="T",
        body="B",
        head_sha="beef",
        base_ref="main",
        diff="d",
        diff_files=frozenset({"src/app.py"}),
    )


def _invoke(monkeypatch, backend: AgenticBackend, harness_result: HarnessResult, env=None):
    monkeypatch.setattr(agentic_mod, "fetch_pr_context", lambda *a, **k: _ctx())
    monkeypatch.setattr(agentic_mod, "checkout_pr_head", lambda *a, **k: "/tmp/nowhere")
    monkeypatch.setattr(agentic_mod, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(agentic_mod, "strip_agent_config", lambda *a, **k: [])
    monkeypatch.setattr(agentic_mod, "check_auth", lambda *a, **k: {"loggedIn": True})
    captured = {}

    def fake_run_review(prompt, checkout, *, cwd, model, env, timeout, max_turns):
        captured.update(
            prompt=prompt, checkout=checkout, cwd=cwd, model=model, env=env, timeout=timeout
        )
        return harness_result

    monkeypatch.setattr(agentic_mod, "run_review", fake_run_review)
    result = backend.invoke(PR, env or {"FUKO_AGENTIC_MODEL": "claude-x"}, ["review"])
    return result, captured


def test_registered_in_backend_registry():
    assert isinstance(get_backend("agentic"), AgenticBackend)


def _model(**kw) -> ModelConfig:
    return ModelConfig(provider="anthropic", name="claude-sonnet-5", **kw)


def test_build_env_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant")
    env = AgenticBackend().build_env(
        get_preset("anthropic"),
        _model(extra_instructions="Hunt races."),
        knowledge="- learn this",
        tools=["review", "improve"],
    )
    assert env["FUKO_AGENTIC_MODEL"] == "claude-sonnet-5"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant"
    assert env["FUKO_AGENTIC_AUTH"] == "api-key"
    # Operator steering and repo-mined knowledge travel separately: the prompt
    # gives them different trust levels, so they must not arrive pre-joined.
    assert env["FUKO_AGENTIC_INSTRUCTIONS"] == "Hunt races."
    assert env["FUKO_AGENTIC_KNOWLEDGE"] == "- learn this"


def test_build_env_auto_falls_back_to_subscription(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_KEY", raising=False)
    env = AgenticBackend().build_env(get_preset("anthropic"), _model(), "", ["review"])
    assert env["FUKO_AGENTIC_AUTH"] == "subscription"
    assert "ANTHROPIC_API_KEY" not in env


def test_build_env_explicit_subscription_ignores_ambient_key(monkeypatch):
    """A key in the environment must not silently move billing off the plan."""
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant")
    env = AgenticBackend().build_env(
        get_preset("anthropic"), _model(auth="subscription"), "", ["review"]
    )
    assert env["FUKO_AGENTIC_AUTH"] == "subscription"
    assert "ANTHROPIC_API_KEY" not in env


def test_build_env_explicit_api_key_without_key_fails_fast(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_KEY"):
        AgenticBackend().build_env(get_preset("anthropic"), _model(auth="api-key"), "", ["review"])


def test_build_env_rejects_non_anthropic():
    with pytest.raises(ValueError, match="agentic"):
        AgenticBackend().build_env(
            get_preset("openrouter"),
            ModelConfig(provider="openrouter", name="x-ai/grok-4.5"),
            knowledge="",
            tools=["review"],
        )


def test_build_env_accepts_anthropic_compatible_gateway(monkeypatch):
    """An `anthropic/`-prefixed gateway preset (qwen-anthropic) passes the gate.

    The gate is on the PROTOCOL the harness speaks, not on who serves it: the
    Token Plan endpoint answers the Anthropic API with a Qwen model behind it.
    The gateway's base URL and key must be injected, and the background-model
    routing vars must map haiku-class and subagent calls onto slugs the gateway
    actually serves -- without them those calls request `claude-haiku-*` from an
    endpoint that has never heard of it.
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    env = AgenticBackend().build_env(
        get_preset("qwen-anthropic"),
        ModelConfig(provider="qwen-anthropic", name="qwen3.8-max", auth="api-key"),
        knowledge="",
        tools=["review"],
    )
    assert env["FUKO_AGENTIC_MODEL"] == "qwen3.8-max"
    assert env["ANTHROPIC_API_KEY"] == "sk-sp-test"
    assert env["ANTHROPIC_BASE_URL"].startswith("https://token-plan.")
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen3.6-flash"  # preset small_model
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "qwen3.8-max"


def test_zai_anthropic_preset_builds_gateway_env(monkeypatch):
    """The z.ai Coding Plan preset (2026-08-24, henry's migration off the
    QwenCloud Token Plan) wires the same gateway shape: anthropic/ prefix,
    plan base URL, ZAI_KEY, and the plan's fast model for the harness's
    auxiliary calls."""
    monkeypatch.setenv("ZAI_KEY", "zk-test")
    env = AgenticBackend().build_env(
        get_preset("zai-anthropic"),
        ModelConfig(provider="zai-anthropic", name="glm-5.3", auth="api-key"),
        knowledge="",
        tools=["review"],
    )
    assert env["FUKO_AGENTIC_MODEL"] == "glm-5.3"
    assert env["ANTHROPIC_API_KEY"] == "zk-test"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "glm-4.5-air"  # preset small_model
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.3"


def test_build_env_plain_anthropic_leaves_model_routing_alone(monkeypatch):
    """No base_url = real Anthropic: the harness's own `claude-*` defaults are
    correct there, so the routing vars must NOT be injected."""
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant")
    env = AgenticBackend().build_env(
        get_preset("anthropic"), _model(auth="api-key"), knowledge="", tools=["review"]
    )
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in env
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env


def test_configured_endpoint_is_auth_aware(monkeypatch):
    """Attribution must follow the traffic: api-key mode goes to the preset's
    gateway; subscription mode deliberately injects no base URL and talks to
    the SDK default — claiming the gateway URL there would be the very
    substitution class the receipt's endpoint field exists to expose."""
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend()
    preset = get_preset("qwen-anthropic")
    keyed = ModelConfig(provider="qwen-anthropic", name="qwen3.8-max", auth="api-key")
    assert backend.configured_endpoint(preset, keyed).startswith("https://token-plan.")
    subscribed = ModelConfig(provider="qwen-anthropic", name="qwen3.8-max", auth="subscription")
    assert backend.configured_endpoint(preset, subscribed) == ""


def test_invoke_reinjects_per_entry_context_window(monkeypatch):
    """The entry's `max_context` reaches the harness subprocess; ambient dies.

    Two active agentic seats with different windows made the workflow-global
    CLAUDE_CODE_MAX_CONTEXT_TOKENS export unable to be right for both, so the
    window is per-entry now: build_env derives it from `max_context` and
    invoke() re-injects exactly that value, while an ambient export (the old
    single-seat mechanism, or a stale one) is scrubbed like the other routing
    vars. Asserted at the seam that matters — the captured subprocess env.
    """
    monkeypatch.setenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "12345")
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    env = backend.build_env(
        get_preset("qwen-anthropic"),
        ModelConfig(
            provider="qwen-anthropic",
            name="qwen3.8-max",
            auth="api-key",
            max_context=1_000_000,
        ),
        knowledge="",
        tools=["review"],
    )
    _, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON), env=env)
    assert captured["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000000"


def test_invoke_logs_the_delivered_window(monkeypatch, capsys):
    """The delivery-side receipt (mepro#2012 r2): the run log must carry the
    window the spawned harness ACTUALLY gets — a config-side validator
    cannot see a delivery failure, so this line is the only observable for
    it. Both directions: a set window logs its value; an absent one logs
    ABSENT loudly instead of silently reviewing at the harness default."""
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    windowed = backend.build_env(
        get_preset("qwen-anthropic"),
        ModelConfig(
            provider="qwen-anthropic",
            name="qwen3.8-max",
            auth="api-key",
            max_context=1_000_000,
        ),
        knowledge="",
        tools=["review"],
    )
    _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON), env=windowed)
    assert (
        "agentic harness qwen3.8-max: CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000"
        in capsys.readouterr().err
    )
    bare = backend.build_env(
        get_preset("qwen-anthropic"),
        ModelConfig(provider="qwen-anthropic", name="qwen3.8-max", auth="api-key"),
        knowledge="",
        tools=["review"],
    )
    _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON), env=bare)
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS=ABSENT" in capsys.readouterr().err


def test_invoke_entry_without_max_context_gets_no_window(monkeypatch):
    """No `max_context` on the entry = no window var at all — the ambient
    export must NOT leak through as a fallback (config decides; an entry that
    wants a window states one, and an unstated one fails loudly downstream as
    the harness's unrecognized-model refusal rather than silently reviewing
    at whatever window the runner's environment happened to carry)."""
    monkeypatch.setenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "12345")
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    env = backend.build_env(
        get_preset("qwen-anthropic"),
        ModelConfig(provider="qwen-anthropic", name="qwen3.8-max", auth="api-key"),
        knowledge="",
        tools=["review"],
    )
    _, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON), env=env)
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in captured["env"]


def test_invoke_reinjects_model_routing_for_gateway(monkeypatch):
    """Ambient wrapper-shell routing vars are stripped; the config-built ones land.

    A claude-qwen-style wrapper on the RUNNER exports ANTHROPIC_MODEL &co for a
    human's shell; carried into the harness they would redirect this seat's
    calls to whatever the wrapper was aimed at. Config decides: the subprocess
    env must hold exactly the values build_env derived, and the ambient ones
    must be gone.
    """
    monkeypatch.setenv("ANTHROPIC_MODEL", "ambient-model")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_HAIKU_MODEL", "ambient-haiku")
    monkeypatch.setenv("ANTHROPIC_SMALL_FAST_MODEL", "ambient-legacy-haiku")
    monkeypatch.setenv("CLAUDE_CODE_SUBAGENT_MODEL", "ambient-sub")
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    env = backend.build_env(
        get_preset("qwen-anthropic"),
        ModelConfig(provider="qwen-anthropic", name="qwen3.8-max", auth="api-key"),
        knowledge="",
        tools=["review"],
    )
    _, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON), env=env)
    assert captured["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen3.6-flash"
    assert captured["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "qwen3.8-max"
    assert "ANTHROPIC_MODEL" not in captured["env"]
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in captured["env"]


def test_invoke_stashes_and_filters(monkeypatch):
    backend = AgenticBackend(ReviewConfig(tool_timeout=222))
    result, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON))
    assert result.returncode == 0
    assert "2 findings" in result.detail
    stash = backend._pending[(PR.url, "claude-x", "")]
    assert [f.title for f in stash.findings] == ["leak", "stale doc"]  # low-conf dropped
    assert stash.withheld_low == 1
    assert stash.over_cap == 0
    assert captured["timeout"] == 222


def test_invoke_strips_github_tokens_from_harness_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    monkeypatch.setenv("FUKO_GITHUB_TOKEN_DORIAN", "app-secret")
    backend = AgenticBackend()
    _, captured = _invoke(
        monkeypatch,
        backend,
        HarnessResult(0, REVIEW_JSON),
        env={
            "FUKO_AGENTIC_MODEL": "m",
            "FUKO_AGENTIC_AUTH": "api-key",
            "ANTHROPIC_API_KEY": "sk",
        },
    )
    assert "GITHUB_TOKEN" not in captured["env"]
    assert "FUKO_GITHUB_TOKEN_DORIAN" not in captured["env"]
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk"


def test_invoke_strips_gh_cli_credentials(monkeypatch):
    """`gh`'s own spellings are exported by many runner images and are just as live."""
    monkeypatch.setenv("GH_TOKEN", "gh-cli-secret")
    monkeypatch.setenv("GH_ENTERPRISE_TOKEN", "ghe-secret")
    monkeypatch.setenv("GITHUB_ENTERPRISE_TOKEN", "ghe2-secret")
    backend = AgenticBackend()
    _, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON))
    assert "GH_TOKEN" not in captured["env"]
    assert "GH_ENTERPRISE_TOKEN" not in captured["env"]
    assert "GITHUB_ENTERPRISE_TOKEN" not in captured["env"]


def test_invoke_without_a_model_fails_before_cloning(monkeypatch):
    backend = AgenticBackend()
    called = {"checkout": False}
    monkeypatch.setattr(agentic_mod, "fetch_pr_context", lambda *a, **k: _ctx())

    def boom(*a, **k):
        called["checkout"] = True
        raise AssertionError("must not clone without a model")

    monkeypatch.setattr(agentic_mod, "checkout_pr_head", boom)
    result = backend.invoke(PR, {"FUKO_AGENTIC_AUTH": "api-key"}, ["review"])
    assert result.returncode == 1
    assert "FUKO_AGENTIC_MODEL" in result.detail
    assert not called["checkout"]


def test_invoke_subscription_mode_drops_ambient_anthropic_creds(monkeypatch):
    """Ambient keys outrank the subscription login, so they must not survive."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ambient")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-ambient")
    monkeypatch.setenv("HOME", "/home/runner")
    backend = AgenticBackend()
    _, captured = _invoke(
        monkeypatch,
        backend,
        HarnessResult(0, REVIEW_JSON),
        env={"FUKO_AGENTIC_MODEL": "m", "FUKO_AGENTIC_AUTH": "subscription"},
    )
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]
    # HOME must survive: the subscription login lives there.
    assert captured["env"]["HOME"] == "/home/runner"


def test_invoke_subscription_mode_drops_ambient_base_url(monkeypatch):
    """An inherited endpoint would aim the runner's own session at a foreign host."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://attacker.example/v1")
    backend = AgenticBackend()
    _, captured = _invoke(
        monkeypatch,
        backend,
        HarnessResult(0, REVIEW_JSON),
        env={"FUKO_AGENTIC_MODEL": "m", "FUKO_AGENTIC_AUTH": "subscription"},
    )
    assert "ANTHROPIC_BASE_URL" not in captured["env"]


def test_invoke_api_key_mode_uses_configured_base_url_over_ambient(monkeypatch):
    """Config decides the endpoint; the ambient environment never does."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://attacker.example/v1")
    backend = AgenticBackend()
    _, captured = _invoke(
        monkeypatch,
        backend,
        HarnessResult(0, REVIEW_JSON),
        env={
            "FUKO_AGENTIC_MODEL": "m",
            "FUKO_AGENTIC_AUTH": "api-key",
            "ANTHROPIC_API_KEY": "sk",
            # What build_env derives from `model.base_url or preset.base_url`.
            "ANTHROPIC_BASE_URL": "https://gateway.internal/v1",
        },
    )
    assert captured["env"]["ANTHROPIC_BASE_URL"] == "https://gateway.internal/v1"


def test_invoke_subscription_mode_forwards_ci_oauth_token(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-tok")
    backend = AgenticBackend()
    _, captured = _invoke(
        monkeypatch,
        backend,
        HarnessResult(0, REVIEW_JSON),
        env={"FUKO_AGENTIC_MODEL": "m", "FUKO_AGENTIC_AUTH": "subscription"},
    )
    assert captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-tok"


def test_invoke_api_key_mode_drops_oauth_token(monkeypatch):
    """Each mode injects exactly one credential, so billing is never ambiguous."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-tok")
    backend = AgenticBackend()
    _, captured = _invoke(
        monkeypatch,
        backend,
        HarnessResult(0, REVIEW_JSON),
        env={
            "FUKO_AGENTIC_MODEL": "m",
            "FUKO_AGENTIC_AUTH": "api-key",
            "ANTHROPIC_API_KEY": "sk",
        },
    )
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in captured["env"]


def test_invoke_subscription_preflight_fails_fast_when_logged_out(monkeypatch):
    monkeypatch.setattr(agentic_mod, "check_auth", lambda *a, **k: {"loggedIn": False})
    monkeypatch.setattr(
        agentic_mod, "fetch_pr_context", lambda *a, **k: pytest.fail("should not clone")
    )
    result = AgenticBackend().invoke(
        PR, {"FUKO_AGENTIC_MODEL": "m", "FUKO_AGENTIC_AUTH": "subscription"}, ["review"]
    )
    assert result.returncode == 1
    assert not result.throttled
    assert "setup-token" in result.detail


def test_invoke_runs_outside_the_checkout(monkeypatch):
    """The agent's cwd must never be the untrusted checkout (repo hooks)."""
    backend = AgenticBackend()
    _, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON))
    assert str(captured["cwd"]) != str(captured["checkout"])
    assert "/tmp/nowhere" in captured["prompt"]  # the root is named in the prompt


def test_invoke_auth_failure_is_not_throttled(monkeypatch):
    """Failing over on bad credentials would burn every provider in the pool."""
    backend = AgenticBackend()
    result, _ = _invoke(
        monkeypatch, backend, HarnessResult(1, "", stderr="Not logged in · Please run /login")
    )
    assert result.returncode == 1
    assert not result.throttled
    assert "authenticate" in result.detail


def test_invoke_subscription_limit_is_throttled(monkeypatch):
    """An exhausted plan window should fail over to a backup, not fail the run."""
    backend = AgenticBackend()
    result, _ = _invoke(
        monkeypatch, backend, HarnessResult(1, "", stderr="You've hit your weekly limit")
    )
    assert result.throttled


def test_invoke_throttle_classification(monkeypatch):
    backend = AgenticBackend()
    result, _ = _invoke(monkeypatch, backend, HarnessResult(1, "", stderr="429 too many requests"))
    assert result.throttled
    result, _ = _invoke(monkeypatch, backend, HarnessResult(124, "", timed_out=True))
    assert result.throttled


def test_invoke_parse_failure_is_not_throttle(monkeypatch):
    backend = AgenticBackend()
    result, _ = _invoke(monkeypatch, backend, HarnessResult(0, "not json at all"))
    assert result.returncode == 1
    assert not result.throttled
    assert "reviewer output" in result.detail


def test_invoke_completed_run_reports_its_channel_done(monkeypatch):
    """#113: a COMPLETED agentic run must state its channel finished, not stay empty.

    An empty channel map reads as "not reported" in `fuko_states`, so a `done`
    receipt with no channels would pass as a clean review even if the channel had
    in fact failed. Mirrors the pr-agent side's populated-map assertion.
    """
    backend = AgenticBackend()
    result, _ = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON))
    assert result.returncode == 0
    assert result.channels == {"agentic-review": "done"}


def test_invoke_timed_out_run_names_the_dead_channel(monkeypatch):
    """A killed run names its channel `killed:timeout`, mirroring the pr-agent side."""
    backend = AgenticBackend()
    result, _ = _invoke(monkeypatch, backend, HarnessResult(124, "", timed_out=True))
    assert result.channels == {"agentic-review": "killed:timeout"}
    assert result.throttled  # a timeout still fails over


def test_invoke_throttled_run_names_the_channel_throttled(monkeypatch):
    """A 429/overload names the channel `throttled:exit N` (not a bare timeout)."""
    backend = AgenticBackend()
    result, _ = _invoke(monkeypatch, backend, HarnessResult(1, "", stderr="429 too many requests"))
    assert result.channels == {"agentic-review": "throttled:exit 1"}


def test_invoke_plain_failure_names_the_channel_failed(monkeypatch):
    """A non-throttle, non-timeout failure names the channel `failed:exit N`."""
    backend = AgenticBackend()
    result, _ = _invoke(monkeypatch, backend, HarnessResult(3, "", stderr="boom"))
    assert result.channels == {"agentic-review": "failed:exit 3"}


def test_invoke_auth_failure_names_the_channel_failed_not_throttled(monkeypatch):
    """Auth failure is a plain channel failure — it must not read as throttle/timeout."""
    backend = AgenticBackend()
    result, _ = _invoke(
        monkeypatch, backend, HarnessResult(1, "", stderr="Not logged in · Please run /login")
    )
    assert result.channels == {"agentic-review": "failed:exit 1"}
    assert not result.throttled


def test_invoke_precondition_failure_marks_the_channel_failed(monkeypatch):
    """A failure before the run (no model) still reports the channel, never empty."""
    backend = AgenticBackend()
    monkeypatch.setattr(agentic_mod, "fetch_pr_context", lambda *a, **k: _ctx())
    monkeypatch.setattr(agentic_mod, "checkout_pr_head", lambda *a, **k: "/tmp/nowhere")
    result = backend.invoke(PR, {"FUKO_AGENTIC_AUTH": "api-key"}, ["review"])
    assert result.returncode == 1
    assert result.channels == {"agentic-review": "failed:exit 1"}


class _FakeResponse:
    def __init__(self, status_code, text="", json_body=None):
        self.status_code = status_code
        self.text = text
        self._json = json_body

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 300:
            raise httpx.HTTPError(f"status {self.status_code}")


class _FakeHttpx:
    HTTPError = httpx.HTTPError

    def __init__(self, statuses, reviews=None):
        self.statuses = list(statuses)
        self.posts = []
        self.gets = []
        self.reviews = reviews or []

    def Client(self, **kwargs):  # noqa: N802 - mimics httpx.Client
        fake = self

        class _C:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None, **kw):
                fake.posts.append((url, json))
                outcome = fake.statuses.pop(0)
                # An entry may be an exception instance, to model a transport
                # failure rather than an HTTP status.
                if isinstance(outcome, BaseException):
                    raise outcome
                return _FakeResponse(outcome)

            def get(self, url, params=None, **kw):
                """The read-back GET that decides whether an ambiguous retry is safe."""
                fake.gets.append(url)
                return _FakeResponse(200, json_body=getattr(fake, "reviews", None) or [])

        return _C()


# Tests post with token="t"; the stash is keyed by that token's fingerprint.
TOKEN_ID = agentic_mod._identity("t")


def _seed(backend: AgenticBackend, model_key="claude-x", identity=TOKEN_ID):
    from sidecar.backends.agentic import _PendingReview
    from sidecar.reviewer.prompt import AgenticFinding

    backend._pending[(PR.url, model_key, identity)] = _PendingReview(
        findings=[
            AgenticFinding(
                file="src/app.py",
                line=4,
                severity="high",
                title="leak",
                body="b1",
                evidence="read it",
            ),
            AgenticFinding(file="docs/other.md", title="stale doc", body="b2"),
        ],
        summary="s",
        head_sha="beef",
        diff_files=frozenset({"src/app.py"}),
        withheld_low=1,
    )


def test_normalize_posts_review_with_markers(monkeypatch):
    backend = AgenticBackend()
    _seed(backend, identity=agentic_mod._identity("tok"))
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(
        PR, "anthropic/claude-x", token="tok", api_url="https://api.x", role="trial"
    )
    assert [s.role for s in signals] == ["trial", "trial"]
    assert all(s.backend == "agentic" for s in signals)
    url, payload = fake.posts[0]
    assert url.endswith("/repos/o/r/pulls/9/reviews")
    assert payload["commit_id"] == "beef"
    (comment,) = payload["comments"]
    assert comment["path"] == "src/app.py" and comment["line"] == 4
    (marker,) = extract_markers(comment["body"])
    assert marker.role == "trial" and marker.backend == "agentic"
    assert "Verified against:" in comment["body"]
    assert "stale doc" in payload["body"]  # unanchored finding lands in the body
    assert "withheld" in payload["body"]


def test_normalize_visible_label_in_ab_mode(monkeypatch):
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(
        PR, "anthropic/claude-x", compare_label="anthropic/claude-x", token="t"
    )
    (comment,) = fake.posts[0][1]["comments"]
    assert comment["body"].startswith("🤖 `anthropic/claude-x`")


def test_normalize_retries_body_only_on_422(monkeypatch):
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([422, 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")
    assert len(signals) == 2
    assert len(fake.posts) == 2
    retry_payload = fake.posts[1][1]
    assert retry_payload["comments"] == []
    assert "leak" in retry_payload["body"]
    assert "src/app.py" in retry_payload["body"]  # the anchor it could not attach to


# --- transient-failure retry around the NON-idempotent review POST (#103) -----


@pytest.fixture
def _no_sleep(monkeypatch):
    """Retries are real code paths; their backoff is not worth real seconds."""
    monkeypatch.setattr(agentic_mod.time, "sleep", lambda _s: None)


def _posted_review(head="beef", label="claude-x"):
    """A review already on the PR, as the read-back would see it."""
    return {"commit_id": head, "body": f"{agentic_mod._review_header(label)}\n\nlooked closely"}


@pytest.mark.parametrize("status", [502, 503, 504])
def test_normalize_retries_transient_5xx(monkeypatch, _no_sleep, status):
    """A completed review must survive a blip, not be discarded with it."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([status, 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")
    assert len(signals) == 2
    assert len(fake.posts) == 2


@pytest.mark.parametrize("status", [502, 503, 504])
def test_a_landed_5xx_does_not_repost(monkeypatch, _no_sleep, status):
    """A gateway status is not evidence the request was refused.

    502/504 mean the upstream may have committed the review and only the
    response was lost — indistinguishable from a request that never arrived. So
    these consult the read-back too, or the retry duplicates the review.
    """
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([status, 200], reviews=[_posted_review()])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    assert len(fake.posts) == 1  # NOT re-posted
    assert len(fake.gets) == 1  # the read-back ran
    assert len(signals) == 2  # findings still reported


def test_normalize_does_not_retry_a_4xx(monkeypatch, _no_sleep):
    """A 403 will not improve by repetition, and a 422 has its own path."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([403])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    assert backend.normalize_output(PR, "claude-x", token="t") == []
    assert len(fake.posts) == 1


def test_normalize_retries_a_connect_error_without_reading_back(monkeypatch, _no_sleep):
    """A connection that never opened cannot have committed a review."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([httpx.ConnectError("refused"), 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")
    assert len(signals) == 2
    assert len(fake.posts) == 2
    assert fake.gets == []  # no read-back needed for the provably-safe class


def test_normalize_does_not_repost_when_the_lost_response_had_landed(monkeypatch, _no_sleep):
    """THE hazard: POST /reviews is not idempotent.

    A read timeout after the server committed is indistinguishable from one
    before, so a naive retry would post the whole review twice — duplicating
    every inline comment and every marker, which downstream extraction would
    read as two findings where there is one.
    """
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([httpx.ReadTimeout("lost"), 200], reviews=[_posted_review()])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    assert len(fake.posts) == 1  # NOT re-posted
    assert len(fake.gets) == 1  # the read-back happened
    assert len(signals) == 2  # and the findings are still reported


def test_normalize_retries_a_lost_response_that_did_not_land(monkeypatch, _no_sleep):
    """Ambiguous, but the read-back proves nothing was committed: retry is safe."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([httpx.ReadTimeout("lost"), 200], reviews=[])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")
    assert len(fake.posts) == 2
    assert len(signals) == 2


def test_read_back_ignores_a_review_for_another_commit(monkeypatch, _no_sleep):
    """Our review of an EARLIER head must not be mistaken for this one."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([httpx.ReadTimeout("lost"), 200], reviews=[_posted_review(head="older")])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(PR, "claude-x", token="t")
    assert len(fake.posts) == 2  # not ours for this head -> retry


def test_read_back_ignores_a_sibling_ab_branch_review(monkeypatch, _no_sleep):
    """A/B branches post for the SAME commit through the same backend.

    If the fingerprint were the bare header, branch A's committed review would
    satisfy branch B's read-back — branch B would report its signals as posted
    while its review never reached the PR. Losing a review silently is worse than
    the duplicate the read-back exists to prevent.
    """
    backend = AgenticBackend()
    _seed(backend)
    sibling = _posted_review(head="beef", label="some-other-model")
    fake = _FakeHttpx([httpx.ReadTimeout("lost"), 200], reviews=[sibling])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    assert len(fake.posts) == 2  # the sibling's review is not ours -> retry
    assert len(signals) == 2


def test_read_back_matches_our_own_branch_in_ab_mode(monkeypatch, _no_sleep):
    """The converse: our OWN branch's review must still be recognised."""
    backend = AgenticBackend()
    _seed(backend)
    ours = _posted_review(head="beef", label="anthropic/claude-x")
    fake = _FakeHttpx([httpx.ReadTimeout("lost"), 200], reviews=[ours])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(
        PR, "anthropic/claude-x", compare_label="anthropic/claude-x", token="t"
    )
    assert len(fake.posts) == 1  # recognised as already posted


def test_read_back_walks_past_the_first_page(monkeypatch, _no_sleep):
    """Reviews come back oldest-first, so ours is on the LAST page.

    Reading only page 1 of a busy PR would miss our own review and retry into
    exactly the duplicate this guard exists to prevent.
    """
    backend = AgenticBackend()
    _seed(backend)

    class _Paged(_FakeHttpx):
        def Client(self, **kwargs):  # noqa: N802
            fake = self

            class _C:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def post(self, url, json=None, **kw):
                    fake.posts.append((url, json))
                    outcome = fake.statuses.pop(0)
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return _FakeResponse(outcome)

                def get(self, url, params=None, **kw):
                    page = (params or {}).get("page", 1)
                    fake.gets.append(page)
                    # Page 1 is a full page of unrelated reviews; ours is on page 2.
                    if page == 1:
                        return _FakeResponse(200, json_body=[{"commit_id": "old"}] * 100)
                    return _FakeResponse(200, json_body=[_posted_review()])

            return _C()

    fake = _Paged([httpx.ReadTimeout("lost"), 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(PR, "claude-x", token="t")

    assert fake.gets == [1, 2]
    assert len(fake.posts) == 1  # found on page 2 -> not re-posted


def test_read_back_ignores_another_reviewers_review(monkeypatch, _no_sleep):
    """CodeRabbit reviewing the same commit is not evidence that we posted."""
    backend = AgenticBackend()
    _seed(backend)
    other = {"commit_id": "beef", "body": "**Actionable comments posted: 2**"}
    fake = _FakeHttpx([httpx.ReadTimeout("lost"), 200], reviews=[other])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(PR, "claude-x", token="t")
    assert len(fake.posts) == 2


def test_unreadable_read_back_declines_to_retry(monkeypatch, _no_sleep):
    """Unknown is not "no": a duplicate review is worse than a missing one.

    The read-back itself failing leaves us unable to tell whether the review
    landed, so the safe move is to stop rather than risk posting twice.
    """
    backend = AgenticBackend()
    _seed(backend)

    class _Broken(_FakeHttpx):
        def Client(self, **kwargs):  # noqa: N802
            outer = super().Client(**kwargs)

            class _C:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def post(self, url, json=None, **kw):
                    return outer.__enter__().post(url, json=json)

                def get(self, url, params=None, **kw):
                    raise httpx.HTTPError("read-back down")

            return _C()

    fake = _Broken([httpx.ReadTimeout("lost"), 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    assert len(fake.posts) == 1  # declined to retry — a duplicate is unacceptable
    # ...and declined to CLAIM it posted. Both directions are unsafe when the
    # outcome is unknown: retrying risks a duplicate, reporting success reports a
    # review nobody confirmed. The branch fails instead, which reads as not-done.
    assert signals == []


def test_posting_deadline_stops_the_flow_and_reports_failure(monkeypatch):
    """The whole posting flow shares ONE budget, independent of tool_timeout.

    `tool_timeout` is passed to `run_review()` inside `invoke()`; the runner calls
    `normalize_output()` afterwards, so nothing bounded this path. Unbounded, the
    worst case was 3 attempts x (1 POST + 20 read-back GETs) x 60s ~= 63 minutes —
    long enough to blow the CI job's cap and get the run killed mid-review, which
    is the starved round that reads as clean.
    """
    monkeypatch.setattr(agentic_mod, "_POST_DEADLINE_SECONDS", 0.0)
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([httpx.ReadTimeout("lost"), 200], reviews=[])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    assert fake.posts == []  # never even started, the budget was already spent
    assert signals == []  # and did not claim success


def test_read_back_stops_paging_when_the_deadline_expires(monkeypatch, _no_sleep):
    """A deep pagination walk must not outlive the budget either."""
    backend = AgenticBackend()
    _seed(backend)

    class _SlowPages(_FakeHttpx):
        def Client(self, **kwargs):  # noqa: N802
            fake = self

            class _C:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def post(self, url, json=None, **kw):
                    fake.posts.append((url, json))
                    outcome = fake.statuses.pop(0)
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return _FakeResponse(outcome)

                def get(self, url, params=None, **kw):
                    fake.gets.append((params or {}).get("page", 1))
                    # Always a full page, so the walk would run to the page cap.
                    return _FakeResponse(200, json_body=[{"commit_id": "other"}] * 100)

            return _C()

    # A controllable clock: every reading advances 10s, so the walk consumes the
    # budget the way real requests would. Without this the fake is instantaneous
    # and the page cap, not the deadline, is what stops it.
    class _Clock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            self.now += 10.0
            return self.now

        def sleep(self, _s):
            pass

    fake = _SlowPages([httpx.ReadTimeout("lost"), 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    monkeypatch.setattr(agentic_mod, "time", _Clock())
    monkeypatch.setattr(agentic_mod, "_READ_BACK_MAX_PAGES", 200)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    # Bounded by the 120s deadline (~12 clock ticks), not the 200-page cap.
    assert 0 < len(fake.gets) < 200
    assert signals == []


def test_retry_budget_is_bounded(monkeypatch, _no_sleep):
    """A persistent outage must fail, not spin — the budget stays inside tool_timeout."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([503, 503, 503])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    assert backend.normalize_output(PR, "claude-x", token="t") == []
    assert len(fake.posts) == agentic_mod._POST_ATTEMPTS


def test_normalize_body_only_fallback_keeps_markers(monkeypatch):
    """A finding demoted to the body must stay recoverable as a Review Signal."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([422, 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    body = fake.posts[1][1]["body"]
    # Every finding is recoverable from the body: the demoted inline one and the
    # unanchored one that was already rendered there.
    assert sorted(m.id for m in extract_markers(body)) == sorted(s.id for s in signals)
    assert "read it" in body  # evidence survives the demotion too


def test_normalize_body_only_fallback_keeps_visible_label(monkeypatch):
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([422, 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(
        PR, "anthropic/claude-x", compare_label="anthropic/claude-x", token="t"
    )
    assert "🤖 `anthropic/claude-x`" in fake.posts[1][1]["body"]


def test_review_body_separates_low_confidence_from_cap(monkeypatch):
    """'Withheld' is the agent's own call; the cap is ours. Do not conflate them."""
    from sidecar.backends.agentic import _PendingReview
    from sidecar.reviewer.prompt import AgenticFinding

    backend = AgenticBackend()
    backend._pending[(PR.url, "claude-x", TOKEN_ID)] = _PendingReview(
        findings=[AgenticFinding(file="docs/a.md", title="t", body="b")],
        summary="s",
        head_sha="beef",
        diff_files=frozenset(),
        withheld_low=2,
        over_cap=3,
    )
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(PR, "claude-x", token="t")

    body = fake.posts[0][1]["body"]
    assert "2 low-confidence finding(s) withheld" in body
    assert "3 further finding(s) cut by the" in body


def test_normalize_post_failure_returns_no_signals(monkeypatch, capsys):
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([500])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    assert backend.normalize_output(PR, "claude-x", token="t") == []
    assert "post failed" in capsys.readouterr().err


def test_claim_does_not_steal_another_models_review(monkeypatch):
    """In A/B mode several branches share one backend; a claim must not cross models."""
    backend = AgenticBackend()
    _seed(backend, model_key="claude-opus")  # another branch's pending review
    fake = _FakeHttpx([])
    monkeypatch.setattr(agentic_mod, "httpx", fake)

    assert backend.normalize_output(PR, "anthropic/claude-sonnet", token="t") == []
    assert fake.posts == []
    # Still there for its rightful owner.
    assert (PR.url, "claude-opus", TOKEN_ID) in backend._pending


def test_claim_tolerates_prefixed_spelling_of_the_same_model(monkeypatch):
    backend = AgenticBackend()
    _seed(backend, model_key="openai/claude-x")  # a different provider spelling
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    assert len(backend.normalize_output(PR, "anthropic/claude-x", token="t")) == 2


def test_a_hallucinated_line_does_not_demote_every_other_finding(monkeypatch):
    """One out-of-hunk line 422s the whole review, so filter before posting."""
    from sidecar.backends.agentic import _PendingReview
    from sidecar.reviewer.prompt import AgenticFinding

    backend = AgenticBackend()
    backend._pending[(PR.url, "claude-x", TOKEN_ID)] = _PendingReview(
        findings=[
            AgenticFinding(file="src/app.py", line=11, title="real", body="b"),
            AgenticFinding(file="src/app.py", line=900, title="hallucinated", body="b"),
        ],
        summary="s",
        head_sha="beef",
        diff_files=frozenset({"src/app.py"}),
        diff_positions=frozenset({("src/app.py", 10), ("src/app.py", 11)}),
    )
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(PR, "claude-x", token="t")

    payload = fake.posts[0][1]
    (comment,) = payload["comments"]  # only the in-hunk one is anchored
    assert comment["line"] == 11
    assert "hallucinated" in payload["body"]  # the other still reaches the reader
    assert len(fake.posts) == 1  # and no 422 round-trip was needed


def test_anchoring_falls_back_to_file_membership_without_positions(monkeypatch):
    """An unparsed diff must not silently send every finding to the body."""
    backend = AgenticBackend()
    _seed(backend)  # seeded with no diff_positions
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(PR, "claude-x", token="t")
    assert len(fake.posts[0][1]["comments"]) == 1


def test_pending_stash_is_bounded(monkeypatch):
    """A branch that dies before egress must not pin its findings forever."""
    backend = AgenticBackend()
    monkeypatch.setattr(agentic_mod, "_MAX_PENDING", 3)
    for i in range(6):
        _invoke(
            monkeypatch,
            backend,
            HarnessResult(0, REVIEW_JSON),
            env={"FUKO_AGENTIC_MODEL": f"model-{i}", "FUKO_AGENTIC_AUTH": "api-key"},
        )
    assert len(backend._pending) <= 3
    # The survivors are the most recent arrivals.
    assert any(k[1] == "model-5" for k in backend._pending)
    assert not any(k[1] == "model-0" for k in backend._pending)


def test_normalize_token_fallback_matches_invoke(monkeypatch):
    """A runner with only GITHUB__USER_TOKEN must still be able to claim its stash."""
    monkeypatch.setenv("GITHUB__USER_TOKEN", "runner-tok")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    backend = AgenticBackend()
    _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON))
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)

    # token=None -> normalize must resolve it the same way invoke did.
    assert len(backend.normalize_output(PR, "claude-x")) == 2


def test_two_identities_of_the_same_model_do_not_collide(monkeypatch):
    """Two [[review.models]] entries may share provider/name and differ by token_env."""
    backend = AgenticBackend()
    _seed(backend, identity=agentic_mod._identity("tok-a"))
    _seed(backend, identity=agentic_mod._identity("tok-b"))
    assert len(backend._pending) == 2  # the second must not overwrite the first

    fake = _FakeHttpx([200, 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)

    assert len(backend.normalize_output(PR, "claude-x", token="tok-a")) == 2
    # The other identity's stash is untouched and still claimable by its owner.
    assert (PR.url, "claude-x", agentic_mod._identity("tok-b")) in backend._pending
    assert len(backend.normalize_output(PR, "claude-x", token="tok-b")) == 2
    assert backend._pending == {}


def test_claim_rejects_a_model_that_merely_ends_with_the_stashed_name(monkeypatch):
    """Suffix matching looks equivalent to exact bare equality and is not."""
    backend = AgenticBackend()
    _seed(backend, model_key="sonnet-4")
    fake = _FakeHttpx([])
    monkeypatch.setattr(agentic_mod, "httpx", fake)

    assert backend.normalize_output(PR, "anthropic/claude-sonnet-4", token="t") == []
    assert fake.posts == []
    assert (PR.url, "sonnet-4", TOKEN_ID) in backend._pending


def test_normalize_marks_unanchored_findings_in_the_body(monkeypatch):
    """Findings outside the diff are this reviewer's specialty -- they need markers too."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    body = fake.posts[0][1]["body"]
    unanchored = next(s for s in signals if s.file == "docs/other.md")
    assert unanchored.id in [m.id for m in extract_markers(body)]
    assert "docs/other.md" in body


def test_normalize_unanchored_findings_carry_the_visible_label(monkeypatch):
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    backend.normalize_output(
        PR, "anthropic/claude-x", compare_label="anthropic/claude-x", token="t"
    )
    assert "🤖 `anthropic/claude-x`" in fake.posts[0][1]["body"]


@pytest.mark.parametrize("spelling", ["low", "Low", "LOW", " low "])
def test_invoke_withholds_low_confidence_regardless_of_spelling(monkeypatch, spelling):
    """The pressure valve must meet the model where it writes, not where we hope."""
    review = json.dumps(
        {
            "summary": "s",
            "findings": [
                {"file": "src/app.py", "line": 4, "title": "keep", "body": "b"},
                {
                    "file": "src/app.py",
                    "line": 8,
                    "title": "hedged",
                    "body": "b",
                    "confidence": spelling,
                },
            ],
        }
    )
    backend = AgenticBackend()
    _invoke(monkeypatch, backend, HarnessResult(0, review))
    stash = backend._pending[(PR.url, "claude-x", "")]
    assert [f.title for f in stash.findings] == ["keep"]
    assert stash.withheld_low == 1


def test_normalize_transport_failure_returns_no_signals(monkeypatch, capsys):
    """A connection error means nothing reached the PR, same as a rejected post."""
    backend = AgenticBackend()
    _seed(backend)

    class _Boom(_FakeHttpx):
        def Client(self, **kwargs):  # noqa: N802 - mimics httpx.Client
            class _C:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def post(self, url, json=None, **kw):
                    raise httpx.ConnectError("no route to host")

            return _C()

    monkeypatch.setattr(agentic_mod, "httpx", _Boom([]))
    assert backend.normalize_output(PR, "claude-x", token="t") == []
    assert "transport" in capsys.readouterr().err


def test_normalize_without_stash_is_empty(monkeypatch):
    backend = AgenticBackend()
    fake = _FakeHttpx([])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    assert backend.normalize_output(PR, "claude-x", token="t") == []
    assert fake.posts == []


def test_concurrent_api_key_branches_get_distinct_claude_config_dirs(monkeypatch):
    """Two concurrent agentic branches must NOT share one Claude state dir.

    Branches run as threads in ONE process (runner.py's ThreadPoolExecutor)
    and every spawned harness used to inherit the same ambient HOME, so two
    headless Claude Code processes starting milliseconds apart contended on a
    single `~/.claude`. The first-spawned one lost and exited 1 — measured on
    mepro PR #2064, 3/3 rounds, tracking spawn order rather than model or
    endpoint.

    The existing suite could not see this because every other test invokes a
    single branch; the defect only exists BETWEEN branches. So assert the
    property that matters: distinct, non-empty, per-branch config dirs.
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    seen = []
    for name in ("qwen3.8-max", "qwen3.8-max"):
        env = backend.build_env(
            get_preset("qwen-anthropic"),
            ModelConfig(provider="qwen-anthropic", name=name, auth="api-key"),
            knowledge="",
            tools=["review"],
        )
        _, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON), env=env)
        seen.append(captured["env"].get("CLAUDE_CONFIG_DIR"))

    assert all(seen), f"api-key branches must get their own CLAUDE_CONFIG_DIR, got {seen}"
    assert seen[0] != seen[1], f"concurrent branches share a config dir: {seen}"


def test_subscription_branch_keeps_the_shared_claude_config_dir(monkeypatch):
    """The isolation above must NOT apply in subscription mode.

    A subscription login LIVES in HOME/CLAUDE_CONFIG_DIR. Pointing the harness
    at a fresh empty directory would discard the credential and every such
    branch would fail to authenticate — so subscription branches keep the
    ambient directory (and, knowingly, the race).
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    env = backend.build_env(
        get_preset("qwen-anthropic"),
        ModelConfig(provider="qwen-anthropic", name="qwen3.8-max", auth="subscription"),
        knowledge="",
        tools=["review"],
    )
    _, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON), env=env)
    assert "CLAUDE_CONFIG_DIR" not in captured["env"]


def test_api_key_branch_still_denies_the_ambient_claude_config_dir(monkeypatch):
    """Replacing CLAUDE_CONFIG_DIR must not drop the deny rule that covered it.

    The read denylist keys on CLAUDE_CONFIG_DIR. Redirecting it at a private
    per-branch directory would otherwise silently un-deny the RUNNER's real
    config dir — an operator credential store, readable by an agent whose
    findings are published verbatim to an untrusted PR author. The displaced
    value must reach the harness so `_permission_settings` can deny both.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/runner/ambient-claude")
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    env = backend.build_env(
        get_preset("qwen-anthropic"),
        ModelConfig(provider="qwen-anthropic", name="qwen3.8-max", auth="api-key"),
        knowledge="",
        tools=["review"],
    )
    _, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON), env=env)
    assert captured["env"]["CLAUDE_CONFIG_DIR"] != "/runner/ambient-claude"
    assert captured["env"]["FUKO_AMBIENT_CLAUDE_CONFIG_DIR"] == "/runner/ambient-claude"


def test_failure_prints_full_stderr_and_leads_the_detail_with_the_verdict(monkeypatch, capsys):
    """A failing branch must leave an unabridged copy of stderr in the log.

    `detail` is capped for the receipt, and headless Claude Code opens stderr
    with a benign `[claude-code:unrecognized_model]` warning on any non-catalog
    gateway — so the cap is filled by noise and the real error is discarded.
    That cost mepro #2064 several wrong fixes: the published reason named a
    correctly-configured model as "unrecognized" while the actual failure was
    never visible anywhere.
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    noise = '[claude-code:unrecognized_model] {"model":"qwen3.8-max"}\n' * 12
    real = "RealError: the thing that actually broke"
    result, _ = _invoke(monkeypatch, backend, HarnessResult(1, "", stderr=noise + real), env=None)

    err = capsys.readouterr().err
    assert real in err, "the real error must reach the log unabridged"
    assert "stderr ends" in err, "the dump must be delimited so it is greppable"
    # And the receipt leads with the verdict rather than the noise.
    assert result.detail.startswith("failed:exit 1"), result.detail[:80]


def test_auth_failure_also_dumps_full_stderr(monkeypatch, capsys):
    """The dump must cover EVERY non-zero exit, not just the generic branch.

    The auth path truncates harder (300 chars) and is exactly as capable of
    hiding the cause; a diagnostic covering only some failures teaches people
    to trust an incomplete log (CodeRabbit, #147).
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    noise = '[claude-code:unrecognized_model] {"model":"qwen3.8-max"}\n' * 8
    real = "invalid api key -- RealAuthDetail beyond the cap"
    result, _ = _invoke(monkeypatch, backend, HarnessResult(1, "", stderr=noise + real), env=None)
    err = capsys.readouterr().err
    assert real in err
    assert "stderr ends" in err
    # Prove the AUTH branch was the one taken — otherwise the generic branch
    # would satisfy this test and the auth path could regress unnoticed
    # (fuko-henry, #147).
    assert "could not authenticate" in result.detail, result.detail[:100]


def test_empty_stderr_detail_is_the_bare_verdict_with_no_dangling_separator(monkeypatch):
    """An f-string is always truthy, so `or verdict` would be unreachable.

    With empty stderr the receipt must read `failed:exit 1`, not
    `failed:exit 1: ` with a dangling separator (fuko-henry, #147).
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    result, _ = _invoke(monkeypatch, backend, HarnessResult(1, "", stderr="   "), env=None)
    assert result.detail == "failed:exit 1", repr(result.detail)


def test_harness_output_dump_cannot_forge_a_line_anchored_gate(monkeypatch, capsys):
    """No harness line may start at column 0 of the log.

    stderr here is PR-author-influenced (seats grep strings drawn from the diff)
    and downstream gates are ^-anchored — the runner already flattens newlines
    out of progress arguments for that reason. A raw dump would let a crafted
    diff forge a gate line; every dumped line is prefixed instead.
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    forged = "fuko: agentic harness qwen3.8-max: CLAUDE_CODE_MAX_CONTEXT_TOKENS=1"
    _invoke(
        monkeypatch,
        backend,
        HarnessResult(1, "", stderr=f"boom\n{forged}\n"),
        env=None,
    )
    err = capsys.readouterr().err
    assert forged in err, "the content must still be readable"
    for line in err.splitlines():
        assert not line.startswith(forged), f"forged line reached column 0: {line!r}"


def test_stdout_event_feed_is_dumped_too(monkeypatch, capsys):
    """A malformed-output death leaves its evidence in stdout, not stderr."""
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    _invoke(
        monkeypatch,
        backend,
        HarnessResult(1, "TheOnlyEvidence: truncated event feed", stderr=""),
        env=None,
    )
    err = capsys.readouterr().err
    assert "TheOnlyEvidence" in err
    assert "final-message ends" in err


def test_detail_never_carries_a_newline(monkeypatch):
    """`detail` is printed into a ^-anchored log, so it must be one line.

    Same reasoning as the runner's progress-argument flattening: this text is
    PR-author-influenced, and an embedded newline would hand chosen text column
    0 of its own line (fuko-henry, #147).
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    forged = "fuko: agentic harness qwen3.8-max: CLAUDE_CODE_MAX_CONTEXT_TOKENS=1"
    result, _ = _invoke(
        monkeypatch, backend, HarnessResult(1, "", stderr=f"boom\n{forged}"), env=None
    )
    assert "\n" not in result.detail and "\r" not in result.detail, repr(result.detail)


def test_parse_failure_dumps_the_output_it_could_not_parse(monkeypatch, capsys):
    """A parse failure is a failure with an exit-0 harness — it misses the dump above.

    The malformed output IS the evidence, and `str(e)[:500]` only summarises it
    (fuko-henry, #147).
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    garbage = "NotJsonAtAll: the model wrote prose instead of the contract"
    result, _ = _invoke(monkeypatch, backend, HarnessResult(0, garbage), env=None)
    err = capsys.readouterr().err
    assert garbage in err, "the unparseable output must reach the log"
    assert result.returncode == 1
    assert "\n" not in result.detail
    # Same receipt contract as every other failure path: verdict first, so a
    # reader distinguishes a crash from a timeout from a throttle at a glance
    # (CodeRabbit, #147).
    assert result.detail.startswith("failed:exit 1"), result.detail[:80]


def test_auth_failure_detail_is_flattened_and_verdict_led(monkeypatch):
    """The auth path kept the column-0 vector open after the others were closed.

    `strip()` removes only leading/trailing whitespace, so internal newlines
    survived into a detail printed to a ^-anchored log (fuko-henry, #147).
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    forged = "fuko: agentic harness qwen3.8-max: CLAUDE_CODE_MAX_CONTEXT_TOKENS=1"
    result, _ = _invoke(
        monkeypatch,
        backend,
        HarnessResult(1, "", stderr=f"invalid api key\n{forged}"),
        env=None,
    )
    assert "could not authenticate" in result.detail
    assert "\n" not in result.detail and "\r" not in result.detail, repr(result.detail)
    assert result.detail.startswith("failed:exit 1"), result.detail[:60]


def test_dump_docstring_names_the_prefix_the_code_emits():
    """Greppability depends on the documented prefix being the real one."""
    doc = agentic_mod._dump_harness_output.__doc__ or ""
    assert "final-message|" in doc
    assert "stdout|" not in doc


def test_flatten_covers_every_character_splitlines_breaks_on(monkeypatch):
    """Flattening must use the splitter's own definition of a line.

    Python breaks lines on eight characters beyond \\r and \\n. A hand-rolled
    replace leaves a crafted payload looking flat in `detail` while still
    splitting downstream, reopening the column-0 forgery (fuko-henry, #147).
    """
    exotic = "\x0b\x0c\x1c\x1d\x1e\x85  "
    flat = agentic_mod._flatten_for_log(f"head{exotic}forged")
    assert len(flat.splitlines()) == 1, repr(flat)

    # And end to end, through a real failure detail.
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    forged = "fuko: agentic harness qwen3.8-max: CLAUDE_CODE_MAX_CONTEXT_TOKENS=1"
    result, _ = _invoke(
        monkeypatch, backend, HarnessResult(1, "", stderr=f"boom\x85{forged}"), env=None
    )
    assert len(result.detail.splitlines()) == 1, repr(result.detail)


def test_parse_failure_dump_header_does_not_claim_exit_zero(monkeypatch, capsys):
    """The parse path passes a LABEL, not a returncode that would read `exit 0`.

    It is a failure with an exit-0 harness, so printing `exit 0` in the header
    of a failure dump is actively misleading (fuko-henry, #147).
    """
    monkeypatch.setenv("QWEN_TOKEN_PLAN_KEY", "sk-sp-test")
    backend = AgenticBackend(ReviewConfig(tool_timeout=5))
    _invoke(monkeypatch, backend, HarnessResult(0, "not json"), env=None)
    err = capsys.readouterr().err
    assert "parse-failure" in err
    assert "exit 0 — full harness" not in err
