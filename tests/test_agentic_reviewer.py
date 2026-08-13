"""Unit tests for the reviewer core: checkout/context, prompt, and harness."""

import json
import subprocess

import httpx
import pytest

from sidecar.reviewer import checkout as checkout_mod
from sidecar.reviewer import harness as harness_mod
from sidecar.reviewer.checkout import (
    CheckoutError,
    PRContext,
    checkout_pr_head,
    fetch_pr_context,
)
from sidecar.reviewer.harness import HarnessNotAvailableError, run_review
from sidecar.reviewer.prompt import (
    MAX_FINDINGS,
    ReviewParseError,
    build_prompt,
    parse_review,
)

DIFF = """\
diff --git a/src/app.py b/src/app.py
index 111..222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,4 @@
+import os
diff --git a/src/util.py b/src/util.py
index 333..444 100644
--- a/src/util.py
+++ b/src/util.py
@@ -10,2 +10,3 @@
+def helper():
"""


def _ctx(**overrides) -> PRContext:
    base = dict(
        title="t",
        body="b",
        head_sha="abc123",
        base_ref="main",
        diff=DIFF,
        diff_files=frozenset({"src/app.py", "src/util.py"}),
        truncated=False,
    )
    base.update(overrides)
    return PRContext(**base)


class _Namespace:
    """A stand-in for the ``httpx`` name inside a module under test."""

    def __init__(self, client_factory):
        self.Client = client_factory
        self.HTTPError = httpx.HTTPError


def _mock_httpx(monkeypatch, module, handler):
    transport = httpx.MockTransport(handler)

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return httpx.Client(transport=transport, **kwargs)

    monkeypatch.setattr(module, "httpx", _Namespace(factory))


def test_fetch_pr_context_meta_diff_and_files(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Accept") == "application/vnd.github.v3.diff":
            return httpx.Response(200, text=DIFF)
        return httpx.Response(
            200,
            json={
                "title": "T",
                "body": "B",
                "head": {"sha": "abc123"},
                "base": {"ref": "main"},
            },
        )

    _mock_httpx(monkeypatch, checkout_mod, handler)
    ctx = fetch_pr_context("o/r", 5, token="tok")
    assert ctx.title == "T"
    assert ctx.head_sha == "abc123"
    assert ctx.diff_files == frozenset({"src/app.py", "src/util.py"})
    assert not ctx.truncated


def test_fetch_pr_context_truncates_at_file_boundary(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Accept") == "application/vnd.github.v3.diff":
            return httpx.Response(200, text=DIFF)
        return httpx.Response(
            200,
            json={"title": "T", "body": "", "head": {"sha": "s"}, "base": {"ref": "m"}},
        )

    _mock_httpx(monkeypatch, checkout_mod, handler)
    ctx = fetch_pr_context("o/r", 5, token="tok", diff_budget=len(DIFF) // 2)
    assert ctx.truncated
    assert "src/util.py" not in ctx.diff
    # The anchor set still covers the FULL diff, not just the kept prefix.
    assert "src/util.py" in ctx.diff_files


def test_checkout_pr_head_steps_and_token_hygiene(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("env")))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checkout_mod.subprocess, "run", fake_run)
    dest = checkout_pr_head("o/r", 7, "beef", token="sekret", workdir=str(tmp_path / "co"))
    assert dest == tmp_path / "co"
    joined = " ".join(" ".join(cmd) for cmd, _ in calls)
    assert "pull/7/head" in joined
    assert "beef" in joined
    assert "sekret" not in joined
    fetch_env = calls[2][1]
    assert fetch_env["GIT_CONFIG_COUNT"] == "1"
    assert "sekret" not in fetch_env["GIT_CONFIG_VALUE_0"]  # base64d, not raw
    assert fetch_env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")


def test_checkout_pr_head_failure_raises(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: nope")

    monkeypatch.setattr(checkout_mod.subprocess, "run", fake_run)
    with pytest.raises(CheckoutError, match="nope"):
        checkout_pr_head("o/r", 7, "beef", token="t", workdir=str(tmp_path / "co"))


def test_build_prompt_sections():
    prompt = build_prompt(_ctx(), instructions="- prefer httpx")
    assert "<diff>" in prompt and DIFF.strip() in prompt
    assert "<operator-guidance>" in prompt and "- prefer httpx" in prompt
    assert "UNTRUSTED" in prompt
    assert f"at most {MAX_FINDINGS} findings" in prompt
    assert "truncated" not in prompt.split("<diff>")[0].split("Unified diff")[1]


def test_build_prompt_omits_guidance_and_flags_truncation():
    prompt = build_prompt(_ctx(truncated=True))
    assert "<operator-guidance>" not in prompt
    assert "truncated to fit" in prompt


def test_parse_review_bare_and_wrapped():
    payload = {
        "summary": "ok",
        "findings": [{"file": "a.py", "line": 3, "title": "t", "body": "b"}],
    }
    text = json.dumps(payload)
    for wrapped in (text, f"```json\n{text}\n```", f"Here you go:\n{text}\nDone."):
        review = parse_review(wrapped)
        assert review.summary == "ok"
        assert review.findings[0].file == "a.py"
        assert review.findings[0].severity == "medium"


def test_parse_review_rejects_garbage_and_bad_schema():
    with pytest.raises(ReviewParseError):
        parse_review("no json here")
    with pytest.raises(ReviewParseError):
        parse_review('{"findings": [{"file": "a.py", "severity": "apocalyptic"}]}')


def test_run_review_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(harness_mod.shutil, "which", lambda *a, **k: None)
    with pytest.raises(HarnessNotAvailableError):
        run_review("p", tmp_path, model="m", env={}, timeout=5)


def test_run_review_invocation_shape(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout='{"findings": []}', stderr="")

    monkeypatch.setattr(harness_mod.shutil, "which", lambda *a, **k: "/bin/claude")
    monkeypatch.setattr(harness_mod.subprocess, "run", fake_run)
    result = run_review(
        "the prompt", tmp_path, model="claude-x", env={"A": "1"}, timeout=9, max_turns=7
    )
    assert result.returncode == 0 and result.text == '{"findings": []}'
    cmd = seen["cmd"]
    assert cmd[0] == "/bin/claude" and "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-x"
    assert cmd[cmd.index("--allowedTools") + 1] == "Read,Grep,Glob"
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    assert seen["kwargs"]["input"] == "the prompt"
    assert seen["kwargs"]["cwd"] == str(tmp_path)
    assert seen["kwargs"]["env"] == {"A": "1"}
    assert seen["kwargs"]["timeout"] == 9


def test_run_review_timeout_maps_to_throttle_returncode(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 9)

    monkeypatch.setattr(harness_mod.shutil, "which", lambda *a, **k: "/bin/claude")
    monkeypatch.setattr(harness_mod.subprocess, "run", fake_run)
    result = run_review("p", tmp_path, model="m", env={}, timeout=9)
    assert result.timed_out and result.returncode == 124
