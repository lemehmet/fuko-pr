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


def test_invoke_stashes_and_filters(monkeypatch):
    backend = AgenticBackend(ReviewConfig(tool_timeout=222))
    result, captured = _invoke(monkeypatch, backend, HarnessResult(0, REVIEW_JSON))
    assert result.returncode == 0
    assert "2 findings" in result.detail
    stash = backend._pending[(PR.url, "claude-x")]
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


class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class _FakeHttpx:
    HTTPError = httpx.HTTPError

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.posts = []

    def Client(self, **kwargs):  # noqa: N802 - mimics httpx.Client
        fake = self

        class _C:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None):
                fake.posts.append((url, json))
                return _FakeResponse(fake.statuses.pop(0))

        return _C()


def _seed(backend: AgenticBackend, model_key="claude-x"):
    from sidecar.backends.agentic import _PendingReview
    from sidecar.reviewer.prompt import AgenticFinding

    backend._pending[(PR.url, model_key)] = _PendingReview(
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
    _seed(backend)
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


def test_normalize_body_only_fallback_keeps_markers(monkeypatch):
    """A finding demoted to the body must stay recoverable as a Review Signal."""
    backend = AgenticBackend()
    _seed(backend)
    fake = _FakeHttpx([422, 200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    signals = backend.normalize_output(PR, "claude-x", token="t")

    body = fake.posts[1][1]["body"]
    markers = extract_markers(body)
    assert [m.id for m in markers] == [s.id for s in signals if s.file == "src/app.py"]
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
    backend._pending[(PR.url, "claude-x")] = _PendingReview(
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
    assert (PR.url, "claude-opus") in backend._pending


def test_claim_tolerates_prefixed_spelling_of_the_same_model(monkeypatch):
    backend = AgenticBackend()
    _seed(backend, model_key="claude-x")
    fake = _FakeHttpx([200])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    assert len(backend.normalize_output(PR, "anthropic/claude-x", token="t")) == 2


def test_normalize_without_stash_is_empty(monkeypatch):
    backend = AgenticBackend()
    fake = _FakeHttpx([])
    monkeypatch.setattr(agentic_mod, "httpx", fake)
    assert backend.normalize_output(PR, "claude-x", token="t") == []
    assert fake.posts == []
