"""Tests for egress normalization: vendor comments -> Review Signals.

Fixtures are real comment bodies captured from a live PR (pr-agent + Copilot).
"""

import httpx
import pytest

from sidecar.backends import pragent
from sidecar.backends.base import PRRef
from sidecar.backends.pragent import PrAgentBackend
from sidecar.normalizers import (
    coderabbit_signal,
    collect_signals,
    copilot_signal,
    is_coderabbit_comment,
    is_coderabbit_finding,
    is_copilot_comment,
    is_pragent_comment,
    pragent_signal,
    pragent_signals,
)
from sidecar.signals import ReviewSignal, encode_marker

PRAGENT = {
    "id": 111,
    "path": "src/lib/breakLogic.ts",
    "line": 4,
    "start_line": None,
    "html_url": "https://github.com/o/r/pull/8#discussion_r111",
    "user": {"login": "lemehmet"},
    "body": (
        "**Suggestion:** When `completedFocusCount` is 0, the modulo condition is "
        "true (0 % 4 === 0), so the very first call returns a long break of 15 "
        "minutes instead of a short break. Add a guard to ensure the count is "
        "positive before applying the long-break rule. [possible issue, importance: 7]\n"
        "```suggestion\n  if (completedFocusCount > 0 && completedFocusCount % 4 === 0) "
        "return 15\n```"
    ),
}

COPILOT = {
    "id": 222,
    "path": "src/lib/breakLogic.ts",
    "line": 6,
    "html_url": "https://github.com/o/r/pull/8#discussion_r222",
    "user": {"login": "Copilot"},
    "body": "`completedFocusCount % 4 == 0` treats `0` as a long break, likely incorrect.",
}


def test_is_pragent_comment():
    assert is_pragent_comment(PRAGENT["body"])
    assert not is_pragent_comment(COPILOT["body"])
    assert not is_pragent_comment("")


def test_pragent_signal_maps_declared_fields():
    sig = pragent_signal(PRAGENT, model="anthropic/claude-sonnet-4-6")
    assert sig.file == "src/lib/breakLogic.ts"
    assert (sig.line, sig.end_line) == (4, None)
    assert sig.category == "bug"  # "possible issue"
    assert sig.severity == "high"  # importance 7
    assert sig.severity_source == "declared"
    assert sig.suggestion is True
    assert sig.backend == "pr-agent"
    assert sig.model == "anthropic/claude-sonnet-4-6"
    assert sig.title.startswith("When `completedFocusCount` is 0")
    assert "[possible issue, importance: 7]" not in sig.title  # label trimmed off
    assert sig.thread_url == PRAGENT["html_url"]


@pytest.mark.parametrize(
    "imp,expected",
    [(2, "low"), (5, "medium"), (7, "high"), (9, "critical")],
)
def test_severity_from_importance(imp, expected):
    c = dict(PRAGENT, body=f"**Suggestion:** x [possible issue, importance: {imp}]")
    assert pragent_signal(c).severity == expected


@pytest.mark.parametrize(
    "label,expected",
    [("security", "security"), ("performance", "perf"), ("best practice", "style")],
)
def test_category_mapping(label, expected):
    c = dict(PRAGENT, body=f"**Suggestion:** x [{label}, importance: 5]")
    assert pragent_signal(c).category == expected


def test_pragent_signal_without_label_is_inferred():
    c = dict(PRAGENT, body="**Suggestion:** tighten this type")
    sig = pragent_signal(c)
    assert sig.severity_source == "inferred"
    assert sig.severity == "medium"
    assert sig.category == "bug"


def test_pragent_signal_multiline_range():
    c = dict(PRAGENT, start_line=4, line=8)
    sig = pragent_signal(c)
    assert sig.line == 4
    assert sig.end_line == 8


def test_pragent_signals_filters_foreign_comments():
    pairs = pragent_signals([PRAGENT, COPILOT], model="m")
    assert len(pairs) == 1
    assert pairs[0]["comment"]["id"] == 111


def test_is_copilot_comment():
    assert is_copilot_comment(COPILOT)
    assert is_copilot_comment({"user": {"login": "copilot-pull-request-reviewer[bot]"}})
    assert not is_copilot_comment(PRAGENT)
    assert not is_copilot_comment({})


def test_copilot_signal_inferred_fields():
    sig = copilot_signal(COPILOT)
    assert sig.backend == "copilot"
    assert sig.model == ""
    assert sig.severity_source == "inferred"
    assert sig.severity == "medium"
    assert sig.file == "src/lib/breakLogic.ts"
    assert sig.line == 6
    assert sig.suggestion is False
    assert sig.title.startswith("`completedFocusCount % 4 == 0`")


def test_copilot_category_inference():
    sec = dict(COPILOT, body="This is a SQL injection vulnerability.")
    perf = dict(COPILOT, body="This causes an N+1 query and is slow.")
    plain = dict(COPILOT, body="Rename this variable for clarity.")
    assert copilot_signal(sec).category == "security"
    assert copilot_signal(perf).category == "perf"
    assert copilot_signal(plain).category == "bug"


def test_collect_signals_dispatches_per_vendor():
    other = {"user": {"login": "some-human"}, "body": "lgtm", "path": "x", "line": 1}
    signals = collect_signals([PRAGENT, COPILOT, other], model="anthropic/claude")
    assert [s.backend for s in signals] == ["pr-agent", "copilot"]
    assert signals[0].severity_source == "declared"
    assert signals[1].severity_source == "inferred"


CODERABBIT = {
    "id": 333,
    "path": "apps/web/src/hooks/use-webrtc.ts",
    "line": 87,
    "start_line": None,
    "html_url": "https://github.com/o/r/pull/9#discussion_r333",
    "user": {"login": "coderabbitai[bot]"},
    "body": (
        "_⚠️ Potential issue_ | _🔴 Critical_\n\n"
        "**Make media-resolution gating reactive, or calls can still stall.**\n\n"
        "`localMediaResolvedRef` updates won't re-run Effect 3.\n\n"
        "<details>\n<summary>Suggested fix</summary>\n\n```diff\n- a\n+ b\n```\n</details>\n\n"
        "<!-- This is an auto-generated comment by CodeRabbit -->"
    ),
}

CODERABBIT_CHAT = {
    "id": 334,
    "user": {"login": "coderabbitai[bot]"},
    "body": "`@lemehmet`, you're right — my concern was incorrect. Thanks for the clarification.",
}


def test_is_coderabbit_comment_and_finding():
    assert is_coderabbit_comment(CODERABBIT)
    assert is_coderabbit_finding(CODERABBIT["body"])
    # author matches but it's a chat reply, not a finding
    assert is_coderabbit_comment(CODERABBIT_CHAT)
    assert not is_coderabbit_finding(CODERABBIT_CHAT["body"])
    assert not is_coderabbit_comment(COPILOT)


def test_coderabbit_signal_declared_fields():
    sig = coderabbit_signal(CODERABBIT)
    assert sig.backend == "coderabbit"
    assert sig.severity == "critical"
    assert sig.severity_source == "declared"
    assert sig.category == "bug"  # "Potential issue"
    assert sig.suggestion is True  # "Suggested fix"
    assert sig.file == "apps/web/src/hooks/use-webrtc.ts"
    assert sig.line == 87
    assert sig.title == "Make media-resolution gating reactive, or calls can still stall."


@pytest.mark.parametrize(
    "cls,severity,category",
    [
        ("_⚠️ Potential issue_ | _🟠 Major_ | _⚡ Quick win_", "high", "bug"),
        ("_⚠️ Potential issue_ | _🟡 Minor_", "medium", "bug"),
        ("_🧹 Nitpick_ | _🔵 Trivial_", "low", "style"),
        ("_🛠️ Refactor suggestion_ | _🟠 Major_", "high", "design"),
        ("_🔒 Security_ | _🔴 Critical_", "critical", "security"),
        ("_🐢 Performance issue_ | _🟡 Minor_", "medium", "perf"),
        ("_✏️ Typo_ | _🔵 Trivial_", "low", "docs"),
    ],
)
def test_coderabbit_severity_and_category_mapping(cls, severity, category):
    c = dict(CODERABBIT, body=f"{cls}\n\n**t**\n\nbody")
    sig = coderabbit_signal(c)
    assert sig.severity == severity
    assert sig.category == category
    assert sig.severity_source == "declared"


def test_collect_signals_includes_coderabbit_findings_only():
    signals = collect_signals([PRAGENT, COPILOT, CODERABBIT, CODERABBIT_CHAT], model="m")
    assert [s.backend for s in signals] == ["pr-agent", "copilot", "coderabbit"]


def test_collect_signals_prefers_embedded_marker():
    # A fuko-pr comment carrying a review-time marker (model glm-5.2, severity high)
    marker = encode_marker(
        ReviewSignal(
            id="fk_reviewtime",
            file="x.py",
            line=10,
            severity="high",
            severity_source="declared",
            category="security",
            backend="pr-agent",
            model="openai/glm-5.2",
        )
    )
    body = (
        "**Suggestion:** tighten this input handling [possible issue, importance: 4]\n"
        "```suggestion\nfix\n```\n\n" + marker
    )
    c = {
        "path": "x.py",
        "line": 10,
        "html_url": "u",
        "user": {"login": "fuko-pr-review[bot]"},
        "body": body,
    }

    # run with the WRONG local default model — the marker must win
    [sig] = collect_signals([c], model="ollama/qwen2.5-coder")
    assert sig.model == "openai/glm-5.2"  # from the marker, not the local config
    assert sig.id == "fk_reviewtime"
    assert sig.severity == "high"  # marker, not importance-4-derived "medium"
    assert sig.category == "security"
    # human fields are kept from the live parse (the marker excludes them)
    assert sig.title.startswith("tighten this input handling")
    assert "**Suggestion:**" in sig.body


def test_collect_signals_without_marker_uses_local_model():
    # no marker -> model comes from the passed config (unchanged behavior)
    [sig] = collect_signals([PRAGENT], model="anthropic/claude")
    assert sig.model == "anthropic/claude"


def test_normalize_output_returns_only_pragent_signals(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghtok")
    monkeypatch.setattr(
        PrAgentBackend, "_fetch_review_comments", lambda self, a, p, h: [PRAGENT, COPILOT]
    )
    injected = []
    monkeypatch.setattr(
        PrAgentBackend, "_inject_markers", lambda self, a, p, h, pairs: injected.extend(pairs)
    )
    sigs = PrAgentBackend().normalize_output(PRRef("o/r", 8, "u"), model="anthropic/claude")
    assert [s.severity for s in sigs] == ["high"]
    assert len(injected) == 1


def test_normalize_output_degrades_when_fetch_fails(monkeypatch):
    def boom(self, a, p, h):
        raise httpx.HTTPError("nope")

    monkeypatch.setattr(PrAgentBackend, "_fetch_review_comments", boom)
    assert PrAgentBackend().normalize_output(PRRef("o/r", 8, "u")) == []


class _PatchClient:
    """Fake httpx.Client capturing PATCH calls."""

    calls: list = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def patch(self, url, json):
        _PatchClient.calls.append((url, json))

        class _R:
            def raise_for_status(self):
                return None

        return _R()


_AUTH = {"Authorization": "Bearer t"}


def test_inject_markers_patches_unmarked(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    pairs = pragent_signals([PRAGENT], model="m")
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, pairs)
    assert len(_PatchClient.calls) == 1
    url, payload = _PatchClient.calls[0]
    assert url.endswith("/pulls/comments/111")
    assert "fuko-signal:v1" in payload["body"]


def test_inject_markers_skips_already_marked(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    sig = pragent_signal(PRAGENT, model="m")
    marked = dict(PRAGENT, body=PRAGENT["body"] + "\n\n" + encode_marker(sig))
    PrAgentBackend()._inject_markers(
        "https://api", PRRef("o/r", 8, "u"), _AUTH, [{"comment": marked, "signal": sig}]
    )
    assert _PatchClient.calls == []


def test_inject_markers_empty_is_noop(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, [])
    assert _PatchClient.calls == []


def test_inject_markers_skips_when_unauthenticated(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    pairs = pragent_signals([PRAGENT], model="m")
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), {}, pairs)
    assert _PatchClient.calls == []


def test_inject_markers_skips_on_patch_error(monkeypatch):
    class _ErrClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def patch(self, url, json):
            raise httpx.HTTPError("403 not your comment")

    monkeypatch.setattr(pragent.httpx, "Client", _ErrClient)
    pairs = pragent_signals([PRAGENT], model="m")
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, pairs)


class _GetClient:
    def __init__(self, pages):
        self._pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params):
        batch = self._pages.get(params["page"], [])

        class _R:
            def raise_for_status(self):
                return None

            def json(self):
                return batch

        return _R()


def test_fetch_review_comments_paginates(monkeypatch):
    pages = {1: [{"id": i} for i in range(100)], 2: [{"id": 999}]}
    monkeypatch.setattr(pragent.httpx, "Client", lambda *a, **k: _GetClient(pages))
    out = PrAgentBackend()._fetch_review_comments("https://api", PRRef("o/r", 8, "u"), {})
    assert len(out) == 101
    assert out[-1]["id"] == 999


def test_fetch_review_comments_empty(monkeypatch):
    monkeypatch.setattr(pragent.httpx, "Client", lambda *a, **k: _GetClient({1: []}))
    out = PrAgentBackend()._fetch_review_comments("https://api", PRRef("o/r", 8, "u"), {})
    assert out == []
