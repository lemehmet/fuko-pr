"""Unit tests for the fuko review runner (GitHub + subprocess are mocked)."""

import httpx
import pytest

from sidecar import runner
from sidecar.backends import pragent
from sidecar.backends.base import InvokeResult, PRRef
from sidecar.fukoconfig import KnowledgeConfig


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, handler):
        self._handler = handler

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        return self._handler(url, params)


def test_parse_pr_url_ok():
    pr = runner.parse_pr_url("https://github.com/owner/repo/pull/42")
    assert pr.repo == "owner/repo"
    assert pr.number == 42
    assert pr.url.endswith("/pull/42")


def test_parse_pr_url_rejects_non_pr():
    with pytest.raises(ValueError):
        runner.parse_pr_url("https://github.com/owner/repo/issues/42")


def test_github_env_maps_token():
    env = runner._github_env("tok")
    assert env == {"GITHUB__USER_TOKEN": "tok", "GITHUB__DEPLOYMENT_TYPE": "user"}


def test_github_env_empty_without_token():
    assert runner._github_env("") == {}


def test_invoke_runs_docker_per_tool(monkeypatch):
    calls = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, env=None, check=False, timeout=None, **kw):
        calls.append((cmd, env))
        return _Proc()

    monkeypatch.setattr(pragent.subprocess, "run", fake_run)
    pr = PRRef(repo="o/r", number=1, url="https://github.com/o/r/pull/1")
    result = pragent.PrAgentBackend().invoke(
        pr, {"CONFIG__MODEL": "x", "ANTHROPIC__KEY": "secret"}, ["review", "improve"]
    )

    assert result.returncode == 0
    assert [c[0][-1] for c in calls] == ["review", "improve"]
    cmd = calls[0][0]
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[-4:] == [pragent.PrAgentBackend.DEFAULT_IMAGE, "--pr_url", pr.url, "review"]
    # secrets are forwarded by name, never placed in argv
    assert "-e" in cmd and "ANTHROPIC__KEY" in cmd
    assert "secret" not in cmd
    # the value is passed through the subprocess environment instead
    assert calls[0][1]["ANTHROPIC__KEY"] == "secret"


def test_invoke_uses_configured_image_and_extra_args(monkeypatch):
    from sidecar.fukoconfig import ReviewConfig

    captured = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, env=None, check=False, timeout=None, **kw):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(pragent.subprocess, "run", fake_run)
    backend = pragent.PrAgentBackend(
        ReviewConfig(image="ghcr.io/me/pr-agent:0.36.1", docker_extra_args=["--network", "host"])
    )
    pr = PRRef(repo="o/r", number=1, url="u")
    backend.invoke(pr, {}, ["review"])

    cmd = captured["cmd"]
    assert "ghcr.io/me/pr-agent:0.36.1" in cmd
    assert cmd[cmd.index("--network") + 1] == "host"


def test_invoke_reports_failure(monkeypatch):
    class _Proc:
        returncode = 3
        stdout = ""
        stderr = "review failed: boom"

    monkeypatch.setattr(pragent.subprocess, "run", lambda cmd, **kw: _Proc())
    pr = PRRef(repo="o/r", number=1, url="https://github.com/o/r/pull/1")
    result = pragent.PrAgentBackend().invoke(pr, {}, ["review"])

    assert result.returncode == 3
    assert "review exited 3" in result.detail


def test_invoke_times_out_and_kills_container(monkeypatch):
    from sidecar.fukoconfig import ReviewConfig

    killed = []

    class _Killed:
        returncode = 0

    def fake_run(cmd, env=None, check=False, timeout=None, **kw):
        if cmd[:2] == ["docker", "kill"]:
            killed.append(cmd[2])
            return _Killed()
        raise pragent.subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(pragent.subprocess, "run", fake_run)
    pr = PRRef(repo="o/r", number=1, url="https://github.com/o/r/pull/1")
    result = pragent.PrAgentBackend(ReviewConfig(tool_timeout=5)).invoke(pr, {}, ["review"])

    assert result.returncode == 124
    assert "timed out after 5s" in result.detail
    assert killed and killed[0].startswith("fuko-pragent-")  # the container was reaped


def test_invoke_optional_tool_timeout_is_nonfatal(monkeypatch):
    from sidecar.fukoconfig import ReviewConfig

    class _Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, env=None, check=False, timeout=None, **kw):
        if cmd[:2] == ["docker", "kill"]:
            return _Ok()
        if cmd[-1] == "review":  # the primary tool succeeds
            return _Ok()
        raise pragent.subprocess.TimeoutExpired(cmd, timeout)  # `improve` hangs

    monkeypatch.setattr(pragent.subprocess, "run", fake_run)
    pr = PRRef(repo="o/r", number=1, url="https://github.com/o/r/pull/1")
    cfg = ReviewConfig(tool_timeout=5, optional_tools=["improve"])
    result = pragent.PrAgentBackend(cfg).invoke(pr, {}, ["review", "improve"])

    # review ok + improve timed out, but improve is optional -> overall success
    assert result.returncode == 0
    assert "improve timed out after 5s" in result.detail and "[optional]" in result.detail


def test_invoke_optional_tool_nonzero_exit_is_nonfatal(monkeypatch):
    from sidecar.fukoconfig import ReviewConfig

    class _Proc:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, env=None, check=False, timeout=None, **kw):
        return _Proc(0) if cmd[-1] == "review" else _Proc(5)  # improve exits non-zero

    monkeypatch.setattr(pragent.subprocess, "run", fake_run)
    pr = PRRef(repo="o/r", number=1, url="https://github.com/o/r/pull/1")
    cfg = ReviewConfig(optional_tools=["improve"])
    result = pragent.PrAgentBackend(cfg).invoke(pr, {}, ["review", "improve"])

    assert result.returncode == 0  # optional non-zero exit is a warning, not a failure
    assert "improve exited 5 [optional]" in result.detail


def test_build_env_disables_ticket_analysis():
    from sidecar.fukoconfig import ModelConfig
    from sidecar.presets import get_preset

    env = pragent.PrAgentBackend().build_env(get_preset("ollama"), ModelConfig(), "", ["review"])
    assert env["PR_REVIEWER__REQUIRE_TICKET_ANALYSIS_REVIEW"] == "false"


class _FakeStore:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def query(self, repo, files, pr_body, query_text, top_k):
        self.calls.append((repo, files))
        return self.results


def test_build_knowledge_uses_sidecar_when_url_set(monkeypatch):
    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")
    monkeypatch.setattr(runner, "_fetch_pr_context", lambda pr, t, a: (["src/a.py"], "body"))

    captured = {}

    def fake_sidecar(url, token, repo, files, pr_body):
        captured.update(url=url, repo=repo, files=files)
        return [
            {
                "text": "always validate input",
                "source": "remember",
                "source_url": None,
                "file_globs": [],
                "topic": None,
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(runner, "_sidecar_query", fake_sidecar)

    def _no_store(knowledge):
        raise AssertionError("the local store must not be built when a sidecar is used")

    monkeypatch.setattr(runner, "get_store", _no_store)  # lazy: never consulted here
    pr = PRRef(repo="o/r", number=1, url="u")
    md = runner.build_knowledge(pr, "tok", runner._DEFAULT_API, KnowledgeConfig())

    assert captured["url"] == "http://fuko.internal:8000"
    assert "always validate input" in md


def test_build_knowledge_uses_local_store_without_url(monkeypatch):
    monkeypatch.delenv("FUKO_URL", raising=False)
    monkeypatch.setattr(runner, "_fetch_pr_context", lambda pr, t, a: (["x.py"], ""))
    store = _FakeStore(results=[])
    monkeypatch.setattr(runner, "get_store", lambda knowledge: store)
    pr = PRRef(repo="o/r", number=1, url="u")
    assert runner.build_knowledge(pr, "", runner._DEFAULT_API, KnowledgeConfig()) == ""
    assert store.calls == [("o/r", ["x.py"])]


def test_build_knowledge_degrades_on_error(monkeypatch):
    monkeypatch.delenv("FUKO_URL", raising=False)

    def boom(pr, t, a):
        raise RuntimeError("github down")

    monkeypatch.setattr(runner, "_fetch_pr_context", boom)
    monkeypatch.setattr(runner, "get_store", lambda knowledge: _FakeStore())
    pr = PRRef(repo="o/r", number=1, url="u")
    assert runner.build_knowledge(pr, "", runner._DEFAULT_API, KnowledgeConfig()) == ""


def test_cmd_review_success(monkeypatch):
    import argparse

    from sidecar import cli

    monkeypatch.setattr(runner, "review", lambda url, cfg: InvokeResult(returncode=0))
    cli._cmd_review(argparse.Namespace(pr_url="u", config="c"))


def test_cmd_review_exits_on_failure(monkeypatch):
    import argparse

    from sidecar import cli

    monkeypatch.setattr(
        runner, "review", lambda url, cfg: InvokeResult(returncode=2, detail="boom")
    )
    with pytest.raises(SystemExit):
        cli._cmd_review(argparse.Namespace(pr_url="u", config="c"))


def test_fetch_inline_comments_paginates(monkeypatch):
    def handler(url, params=None):
        if params["page"] == 1:
            return _Resp([{"id": i} for i in range(100)])
        return _Resp([{"id": 999}])

    monkeypatch.setattr(runner.httpx, "Client", lambda *a, **k: _FakeClient(handler))
    pr = PRRef(repo="o/r", number=8, url="u")
    out = runner.fetch_inline_comments(pr, "tok", runner._DEFAULT_API)
    assert len(out) == 101 and out[-1]["id"] == 999


def test_fetch_inline_comments_empty(monkeypatch):
    monkeypatch.setattr(
        runner.httpx, "Client", lambda *a, **k: _FakeClient(lambda u, p=None: _Resp([]))
    )
    out = runner.fetch_inline_comments(PRRef("o/r", 8, "u"), "", runner._DEFAULT_API)
    assert out == []


def test_cmd_signals_emits_json(monkeypatch, tmp_path, capsys):
    import argparse
    import json

    from sidecar import cli

    cfg = tmp_path / ".fuko.toml"
    cfg.write_text('[review.model]\nprovider = "anthropic"\nname = "claude"\n', encoding="utf-8")
    comments = [
        {"user": {"login": "Copilot"}, "body": "Use strict equality.", "path": "a.ts", "line": 3},
    ]
    monkeypatch.setattr(runner, "fetch_inline_comments", lambda pr, token, api: comments)
    cli._cmd_signals(argparse.Namespace(pr_url="https://github.com/o/r/pull/8", config=str(cfg)))

    out = json.loads(capsys.readouterr().out)
    assert len(out) == 1
    assert out[0]["backend"] == "copilot"
    assert out[0]["file"] == "a.ts"


def _http_error(status):
    req = httpx.Request("GET", "https://api.github.com/x")
    return httpx.HTTPStatusError("e", request=req, response=httpx.Response(status, request=req))


def test_cmd_signals_friendly_auth_error(monkeypatch, tmp_path, capsys):
    import argparse

    from sidecar import cli

    cfg = tmp_path / ".fuko.toml"
    cfg.write_text('[review.model]\nprovider = "ollama"\nname = "x"\n', encoding="utf-8")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        runner, "fetch_inline_comments", lambda *a: (_ for _ in ()).throw(_http_error(404))
    )

    with pytest.raises(SystemExit) as e:
        cli._cmd_signals(
            argparse.Namespace(pr_url="https://github.com/o/r/pull/8", config=str(cfg))
        )
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "cannot read o/r#8" in err
    assert "Pull requests: Read" in err
    assert "GITHUB_TOKEN is not set" in err


def test_cmd_signals_reraises_non_auth_error(monkeypatch, tmp_path):
    import argparse

    from sidecar import cli

    cfg = tmp_path / ".fuko.toml"
    cfg.write_text('[review.model]\nprovider = "ollama"\nname = "x"\n', encoding="utf-8")
    monkeypatch.setattr(
        runner, "fetch_inline_comments", lambda *a: (_ for _ in ()).throw(_http_error(500))
    )
    with pytest.raises(httpx.HTTPStatusError):
        cli._cmd_signals(
            argparse.Namespace(pr_url="https://github.com/o/r/pull/8", config=str(cfg))
        )


def test_fetch_issue_comments_and_reviews_and_head(monkeypatch):
    def handler(url, params=None):
        if url.endswith("/issues/8/comments"):
            return _Resp([{"id": 1, "body": "walkthrough"}])
        if url.endswith("/pulls/8/reviews"):
            return _Resp([{"id": 2, "state": "COMMENTED"}])
        if url.endswith("/pulls/8"):
            return _Resp({"head": {"sha": "deadbeef"}})
        raise AssertionError(url)

    monkeypatch.setattr(runner.httpx, "Client", lambda *a, **k: _FakeClient(handler))
    pr = PRRef(repo="o/r", number=8, url="u")
    assert runner.fetch_issue_comments(pr, "t", runner._DEFAULT_API)[0]["body"] == "walkthrough"
    assert runner.fetch_reviews(pr, "t", runner._DEFAULT_API)[0]["state"] == "COMMENTED"
    assert runner.fetch_pr_head(pr, "t", runner._DEFAULT_API) == "deadbeef"


def test_cmd_status_emits_json(monkeypatch, capsys):
    import argparse
    import json

    from sidecar import cli

    head = "abcdef1234567890"
    walk = (
        "📝 Walkthrough\n\nReviewing files ... between `1111111` and "
        "`abcdef1`.\nNo actionable comments were generated."
    )
    monkeypatch.setattr(runner, "fetch_pr_head", lambda pr, t, a: head)
    monkeypatch.setattr(
        runner,
        "fetch_issue_comments",
        lambda pr, t, a: [{"user": {"login": "coderabbitai[bot]"}, "body": walk}],
    )
    monkeypatch.setattr(
        runner,
        "fetch_reviews",
        lambda pr, t, a: [{"user": {"login": "Copilot"}, "commit_id": head, "state": "APPROVED"}],
    )
    # No CodeRabbit check-run present -> falls back to the zero-finding walkthrough marker.
    monkeypatch.setattr(runner, "fetch_check_runs", lambda pr, ref, t, a: [])
    cli._cmd_status(argparse.Namespace(pr_url="https://github.com/o/r/pull/8"))
    out = {r["backend"]: r["state"] for r in json.loads(capsys.readouterr().out)}
    assert out == {"coderabbit": "done", "copilot": "done"}


def test_cmd_status_uses_check_run_to_gate_coderabbit(monkeypatch, capsys):
    import argparse
    import json

    from sidecar import cli

    head = "abcdef1234567890"
    # Walkthrough already covers HEAD, but CR's check-run is still in progress: the
    # premature-done bug (#17). The check-run must win -> coderabbit "in_progress".
    walk = "📝 Walkthrough\n\nReviewing files ... between `1111111` and `abcdef1`."
    monkeypatch.setattr(runner, "fetch_pr_head", lambda pr, t, a: head)
    monkeypatch.setattr(
        runner,
        "fetch_issue_comments",
        lambda pr, t, a: [{"user": {"login": "coderabbitai[bot]"}, "body": walk}],
    )
    monkeypatch.setattr(runner, "fetch_reviews", lambda pr, t, a: [])
    monkeypatch.setattr(
        runner,
        "fetch_check_runs",
        lambda pr, ref, t, a: [{"name": "CodeRabbit", "status": "in_progress", "conclusion": None}],
    )
    cli._cmd_status(argparse.Namespace(pr_url="https://github.com/o/r/pull/8"))
    out = {r["backend"]: r["state"] for r in json.loads(capsys.readouterr().out)}
    assert out["coderabbit"] == "in_progress"


def test_cmd_status_degrades_when_check_runs_forbidden(monkeypatch, capsys):
    import argparse
    import json

    from sidecar import cli

    head = "abcdef1234567890"
    walk = (
        "📝 Walkthrough\n\nReviewing files ... between `1111111` and "
        "`abcdef1`.\nNo actionable comments were generated."
    )
    monkeypatch.setattr(runner, "fetch_pr_head", lambda pr, t, a: head)
    monkeypatch.setattr(
        runner,
        "fetch_issue_comments",
        lambda pr, t, a: [{"user": {"login": "coderabbitai[bot]"}, "body": walk}],
    )
    monkeypatch.setattr(runner, "fetch_reviews", lambda pr, t, a: [])
    # A token without checks access -> fetch raises; status must still resolve via fallback.
    monkeypatch.setattr(
        runner, "fetch_check_runs", lambda *a: (_ for _ in ()).throw(_http_error(403))
    )
    cli._cmd_status(argparse.Namespace(pr_url="https://github.com/o/r/pull/8"))
    out = {r["backend"]: r["state"] for r in json.loads(capsys.readouterr().out)}
    assert out["coderabbit"] == "done"


def test_fetch_check_runs_paginates(monkeypatch):
    def handler(url, params=None):
        assert url.endswith("/commits/deadbeef/check-runs")
        if params["page"] == 1:
            return _Resp({"total_count": 101, "check_runs": [{"id": i} for i in range(100)]})
        return _Resp({"total_count": 101, "check_runs": [{"id": 999}]})

    monkeypatch.setattr(runner.httpx, "Client", lambda *a, **k: _FakeClient(handler))
    pr = PRRef(repo="o/r", number=8, url="u")
    out = runner.fetch_check_runs(pr, "deadbeef", "t", runner._DEFAULT_API)
    assert len(out) == 101 and out[-1]["id"] == 999


def test_fetch_check_runs_empty(monkeypatch):
    monkeypatch.setattr(
        runner.httpx,
        "Client",
        lambda *a, **k: _FakeClient(lambda u, p=None: _Resp({"total_count": 0, "check_runs": []})),
    )
    out = runner.fetch_check_runs(PRRef("o/r", 8, "u"), "ref", "", runner._DEFAULT_API)
    assert out == []


def test_cmd_status_friendly_auth_error(monkeypatch, capsys):
    import argparse

    from sidecar import cli

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(runner, "fetch_pr_head", lambda *a: (_ for _ in ()).throw(_http_error(404)))
    with pytest.raises(SystemExit) as e:
        cli._cmd_status(argparse.Namespace(pr_url="https://github.com/o/r/pull/8"))
    assert e.value.code == 1
    assert "cannot read o/r#8" in capsys.readouterr().err


def test_gh_headers():
    assert "Authorization" not in runner._gh_headers("")
    assert runner._gh_headers("t")["Authorization"] == "Bearer t"


def test_fetch_pr_context_paginates(monkeypatch):
    def handler(url, params=None):
        if url.endswith("/pulls/5"):
            return _Resp({"body": "PR body"})
        if url.endswith("/files"):
            if params["page"] == 1:
                return _Resp([{"filename": f"f{i}.py"} for i in range(100)])
            return _Resp([{"filename": "last.py"}])
        raise AssertionError(url)

    monkeypatch.setattr(runner.httpx, "Client", lambda *a, **k: _FakeClient(handler))
    pr = PRRef(repo="o/r", number=5, url="u")
    files, body = runner._fetch_pr_context(pr, "tok", runner._DEFAULT_API)
    assert body == "PR body"
    assert len(files) == 101
    assert files[-1] == "last.py"


def test_sidecar_query_extracts_results(monkeypatch):
    monkeypatch.setattr(runner.httpx, "post", lambda *a, **k: _Resp({"results": [{"text": "x"}]}))
    out = runner._sidecar_query("http://f", "tok", "o/r", ["a.py"], "body")
    assert out == [{"text": "x"}]


def test_sidecar_query_handles_missing_results(monkeypatch):
    monkeypatch.setattr(runner.httpx, "post", lambda *a, **k: _Resp({}))
    assert runner._sidecar_query("http://f", "", "o/r", [], "") == []


def test_review_wires_config_to_backend(monkeypatch, tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        '[review]\nbackend = "pr-agent"\ntools = ["review"]\n'
        '[review.model]\nprovider = "anthropic"\nname = "claude-sonnet-4-6"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_TOKEN", "ghtok")
    monkeypatch.setenv("ANTHROPIC_KEY", "antkey")
    monkeypatch.setattr(runner, "build_knowledge", lambda pr, token, api, store: "- kb item")

    seen = {}

    class FakeBackend:
        def build_env(self, preset, model, knowledge, tools):
            seen.update(model=model.name, knowledge=knowledge, prefix=preset.litellm_prefix)
            return {"CONFIG__MODEL": preset.litellm_prefix + model.name}

        def invoke(self, pr, env, tools):
            seen.update(env=env, tools=tools)
            return InvokeResult(returncode=0)

        def normalize_output(self, pr, model=""):
            return []

    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: FakeBackend())
    result = runner.review("https://github.com/o/r/pull/7", str(cfg))

    assert result.returncode == 0
    assert seen["model"] == "claude-sonnet-4-6"
    assert seen["knowledge"] == "- kb item"
    assert seen["env"]["CONFIG__MODEL"] == "anthropic/claude-sonnet-4-6"
    assert seen["env"]["GITHUB__USER_TOKEN"] == "ghtok"
    assert seen["tools"] == ["review"]


def test_review_swallows_unimplemented_normalize(monkeypatch, tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text('[review.model]\nprovider = "ollama"\nname = "x"\n', encoding="utf-8")
    monkeypatch.setattr(runner, "build_knowledge", lambda pr, token, api, store: "")

    class FakeBackend:
        def build_env(self, preset, model, knowledge, tools):
            return {}

        def invoke(self, pr, env, tools):
            return InvokeResult(returncode=0)

        def normalize_output(self, pr, model=""):
            raise NotImplementedError

    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: FakeBackend())
    assert runner.review("https://github.com/o/r/pull/7", str(cfg)).returncode == 0


def _stub_compare_io(monkeypatch):
    """Neutralize knowledge, cooldown, and sizing I/O for A/B runner tests.

    The per-branch header (`_post_branch_header`) is patched separately by each
    test that needs it, so it is intentionally left untouched here.
    """
    monkeypatch.setattr(runner, "build_knowledge", lambda *a: "")
    monkeypatch.setattr(runner, "_cb_cooldowns", lambda: set())
    monkeypatch.setattr(runner, "_estimate_required_context", lambda *a: None)


def test_review_compare_runs_each_model_fresh_without_describe(monkeypatch, tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        '[review]\ntools = ["review", "improve", "describe"]\n'
        '[[review.compare]]\nprovider = "anthropic"\nname = "claude-sonnet-4-6"\n'
        '[[review.compare]]\nprovider = "ollama"\nname = "qwen2.5-coder"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_TOKEN", "ghtok")
    monkeypatch.setenv("ANTHROPIC_KEY", "antkey")
    _stub_compare_io(monkeypatch)

    headers = []
    monkeypatch.setattr(
        runner, "_post_branch_header", lambda pr, token, api, label: headers.append(label)
    )

    calls = []

    class FakeBackend:
        def build_env(self, preset, model, knowledge, tools):
            return {"CONFIG__MODEL": preset.litellm_prefix + model.name}

        def invoke(self, pr, env, tools):
            calls.append({"model": env["CONFIG__MODEL"], "tools": list(tools), "env": env})
            return InvokeResult(returncode=0)

        def normalize_output(self, pr, model=""):
            return []

    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: FakeBackend())
    result = runner.review("https://github.com/o/r/pull/7", str(cfg))

    assert result.returncode == 0
    assert headers == ["anthropic/claude-sonnet-4-6", "ollama/qwen2.5-coder"]
    assert [c["model"] for c in calls] == ["anthropic/claude-sonnet-4-6", "ollama/qwen2.5-coder"]
    assert all("describe" not in c["tools"] for c in calls)
    assert all(c["env"]["PR_REVIEWER__PERSISTENT_COMMENT"] == "false" for c in calls)
    assert all(c["env"]["PR_CODE_SUGGESTIONS__PERSISTENT_COMMENT"] == "false" for c in calls)
    assert "anthropic/claude-sonnet-4-6" in result.detail


@pytest.mark.parametrize("returncodes,expected", [([1, 0], 0), ([0, 1], 0), ([1, 1], 1)])
def test_review_compare_is_green_when_any_branch_posts(
    monkeypatch, tmp_path, returncodes, expected
):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        '[[review.compare]]\nprovider = "anthropic"\nname = "a"\n'
        '[[review.compare]]\nprovider = "ollama"\nname = "b"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    _stub_compare_io(monkeypatch)
    monkeypatch.setattr(runner, "_post_branch_header", lambda *a: None)

    rcs = iter(returncodes)

    class FakeBackend:
        def build_env(self, preset, model, knowledge, tools):
            return {}

        def invoke(self, pr, env, tools):
            return InvokeResult(returncode=next(rcs), detail="d")

        def normalize_output(self, pr, model=""):
            return []

    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: FakeBackend())
    assert runner.review("https://github.com/o/r/pull/7", str(cfg)).returncode == expected


def test_review_compare_fails_when_describe_is_only_tool(monkeypatch, tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        '[review]\ntools = ["describe"]\n[[review.compare]]\nprovider = "anthropic"\nname = "a"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    _stub_compare_io(monkeypatch)
    monkeypatch.setattr(runner, "_post_branch_header", lambda *a: None)

    class FakeBackend:
        def build_env(self, preset, model, knowledge, tools):
            raise AssertionError("no branch may run when the tool list is empty")

        def invoke(self, pr, env, tools):
            raise AssertionError("no branch may run when the tool list is empty")

        def normalize_output(self, pr, model=""):
            return []

    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: FakeBackend())
    result = runner.review("https://github.com/o/r/pull/7", str(cfg))
    assert result.returncode == 1
    assert "describe" in result.detail


def test_post_branch_header_skips_without_token(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not POST without a token")

    monkeypatch.setattr(runner.httpx, "post", boom)
    runner._post_branch_header(PRRef("o/r", 8, "u"), "", "https://api.github.com", "anthropic/x")


def test_post_branch_header_posts_labelled_comment(monkeypatch):
    posted = {}

    def fake_post(url, json, headers, timeout):
        posted.update(url=url, body=json["body"])
        return _Resp({})

    monkeypatch.setattr(runner.httpx, "post", fake_post)
    runner._post_branch_header(
        PRRef("o/r", 8, "u"), "ghtok", "https://api.github.com", "anthropic/claude"
    )
    assert posted["url"].endswith("/repos/o/r/issues/8/comments")
    assert "anthropic/claude" in posted["body"]
