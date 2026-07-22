"""Tests for author-scoped marker injection in concurrent A/B mode (issue #66).

A repo-write token can successfully edit a sibling branch's comments — GitHub
does not enforce authorship — so ``_inject_markers`` must filter candidates to
the acting identity's own comments, and must refuse to mark at all in A/B mode
when that identity cannot be resolved.
"""

import httpx
import pytest

from sidecar.backends.base import PRRef
from sidecar.backends.pragent import PrAgentBackend
from sidecar.backends import pragent
from sidecar.normalizers import pragent_signals

_PR = PRRef(repo="o/r", number=8, url="https://github.com/o/r/pull/8")
_HEADERS = {"Authorization": "Bearer tok"}


def _comment(comment_id, user_id):
    return {
        "id": comment_id,
        "path": "src/lib/breakLogic.ts",
        "line": 4,
        "start_line": None,
        "html_url": f"https://github.com/o/r/pull/8#discussion_r{comment_id}",
        "user": {"login": f"bot-{user_id}[bot]", "id": user_id},
        "body": (
            "**Suggestion:** Guard the zero case before applying the long-break "
            "rule. [possible issue, importance: 7]\n"
            "```suggestion\n  if (n > 0 && n % 4 === 0) return 15\n```"
        ),
    }


class _FakeClient:
    """Stands in for ``httpx.Client``: scripted /user answer, recorded PATCHes."""

    def __init__(self, user_response="403"):
        self.user_response = user_response
        self.patched = []
        self.user_calls = 0

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **_kw):
        self.user_calls += 1
        if self.user_response == "403":
            request = httpx.Request("GET", url)
            raise httpx.HTTPStatusError(
                "403", request=request, response=httpx.Response(403, request=request)
            )
        resp = httpx.Response(200, json={"id": int(self.user_response)})
        resp.request = httpx.Request("GET", url)
        return resp

    def patch(self, url, json=None, **_kw):
        self.patched.append(url)
        resp = httpx.Response(200, json={})
        resp.request = httpx.Request("PATCH", url)
        return resp


@pytest.fixture()
def backend():
    return PrAgentBackend.__new__(PrAgentBackend)


def test_ab_marking_skips_foreign_comments_with_caller_actor(monkeypatch, backend):
    """With the actor passed in from the header post, only own comments are PATCHed
    and no /user probe is made — the probe 403s for App tokens anyway."""
    client = _FakeClient(user_response="403")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = pragent_signals([_comment(111, 42), _comment(222, 99)], model="openai/m")

    backend._inject_markers("https://api.github.com", _PR, _HEADERS, pairs, label="p/m", actor="42")

    assert client.patched == ["https://api.github.com/repos/o/r/pulls/comments/111"]
    assert client.user_calls == 0


def test_ab_marking_fails_closed_when_identity_unresolvable(monkeypatch, backend, capsys):
    """A/B mode + App token whose /user probe 403s: zero PATCHes — an unmarked
    comment is recoverable, a cross-labeled one corrupts attribution (#66)."""
    client = _FakeClient(user_response="403")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = pragent_signals([_comment(111, 42)], model="openai/m")

    backend._inject_markers("https://api.github.com", _PR, _HEADERS, pairs, label="p/m")

    assert client.patched == []
    assert "marking skipped" in capsys.readouterr().err


def test_solo_marking_keeps_legacy_mark_all_without_identity(monkeypatch, backend):
    """Solo mode (no label): unresolvable identity keeps the legacy best-effort
    mark-all behavior — there is no sibling branch to cross-label."""
    client = _FakeClient(user_response="403")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = pragent_signals([_comment(111, 42)], model="openai/m")

    backend._inject_markers("https://api.github.com", _PR, _HEADERS, pairs, label=None)

    assert client.patched == ["https://api.github.com/repos/o/r/pulls/comments/111"]


def test_ab_marking_probes_user_when_no_actor_passed(monkeypatch, backend):
    """PAT path: no caller actor, /user resolves — foreign comments still skipped."""
    client = _FakeClient(user_response="42")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = pragent_signals([_comment(111, 42), _comment(222, 99)], model="openai/m")

    backend._inject_markers("https://api.github.com", _PR, _HEADERS, pairs, label="p/m")

    assert client.patched == ["https://api.github.com/repos/o/r/pulls/comments/111"]
    assert client.user_calls == 1
