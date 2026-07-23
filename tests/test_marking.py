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
from sidecar.normalizers import guide_signals, pragent_signals
from sidecar.signals import extract_markers, with_markers

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


def _guide_comment(comment_id, user_id):
    return {
        "id": comment_id,
        "html_url": f"https://github.com/o/r/pull/8#issuecomment-{comment_id}",
        "user": {"login": f"bot-{user_id}[bot]", "id": user_id},
        "body": (
            "## PR Reviewer Guide 🔍\n\n<table>\n"
            "<tr><td>🔒&nbsp;<strong>Security concerns</strong><br><br>"
            "<strong>Token leak:</strong><br> workflow tokens may reach the logs.</td></tr>\n"
            "<tr><td>⚡&nbsp;<strong>Recommended focus areas for review</strong><br><br>"
            "<details><summary><strong>Ownership race</strong>\n\n"
            "Two writers can claim the same sink.\n</summary>\n\n</details>\n"
            "</td></tr>\n</table>"
        ),
    }


def _guide_pairs(*comments, model="openai/m"):
    return [{"comment": c, "signals": guide_signals(c, model)} for c in comments]


class _FakeClient:
    """Stands in for ``httpx.Client``: scripted /user answer, recorded PATCHes."""

    def __init__(self, user_response="403"):
        self.user_response = user_response
        self.patched = []
        self.bodies = []
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
        self.bodies.append((json or {}).get("body", ""))
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


def test_guide_marking_patches_issue_comment_endpoint(monkeypatch, backend):
    """The guide is an ISSUE comment: markers PATCH /issues/comments/{id}, and
    the new body carries one marker per parsed signal (security + focus area)."""
    client = _FakeClient(user_response="42")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = _guide_pairs(_guide_comment(501, 42))

    backend._mark_guide_comments("https://api.github.com", _PR, _HEADERS, pairs, actor="42")

    assert client.patched == ["https://api.github.com/repos/o/r/issues/comments/501"]
    markers = extract_markers(client.bodies[0])
    assert len(markers) == 2
    assert {m.category for m in markers} == {"security", "bug"}
    assert client.bodies[0].startswith("## PR Reviewer Guide")


def test_guide_marking_skips_foreign_comments(monkeypatch, backend):
    """Author-scoping mirrors inline marking: a sibling branch's guide stays untouched."""
    client = _FakeClient(user_response="403")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = _guide_pairs(_guide_comment(501, 42), _guide_comment(502, 99))

    backend._mark_guide_comments(
        "https://api.github.com", _PR, _HEADERS, pairs, label="p/m", actor="42"
    )

    assert client.patched == ["https://api.github.com/repos/o/r/issues/comments/501"]
    assert client.user_calls == 0


def test_guide_marking_fails_closed_when_identity_unresolvable(monkeypatch, backend, capsys):
    """A/B mode + unresolvable identity: zero PATCHes, same #66 rule as inline."""
    client = _FakeClient(user_response="403")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = _guide_pairs(_guide_comment(501, 42))

    backend._mark_guide_comments("https://api.github.com", _PR, _HEADERS, pairs, label="p/m")

    assert client.patched == []
    assert "guide marking skipped" in capsys.readouterr().err


def test_guide_marking_solo_marks_without_identity(monkeypatch, backend):
    """Solo mode: unresolvable identity keeps the legacy best-effort mark."""
    client = _FakeClient(user_response="403")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = _guide_pairs(_guide_comment(501, 42))

    backend._mark_guide_comments("https://api.github.com", _PR, _HEADERS, pairs, label=None)

    assert client.patched == ["https://api.github.com/repos/o/r/issues/comments/501"]


def test_guide_marking_rerun_is_idempotent(monkeypatch, backend):
    """A guide already carrying the freshly re-derived marker set is NOT re-PATCHed
    (make_id is deterministic, so the rebuilt body is byte-identical)."""
    client = _FakeClient(user_response="42")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    comment = _guide_comment(501, 42)
    marked = dict(comment, body=with_markers(comment["body"], guide_signals(comment, "openai/m")))

    backend._mark_guide_comments("https://api.github.com", _PR, _HEADERS, _guide_pairs(marked))

    assert client.patched == []


def test_guide_marking_replaces_stale_markers_without_duplicating(monkeypatch, backend):
    """A guide holding markers from an older parse gets the fresh set — stripped
    then re-appended, never accumulated."""
    client = _FakeClient(user_response="42")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    comment = _guide_comment(501, 42)
    stale = guide_signals(comment, "openai/old-model")
    marked = dict(comment, body=with_markers(comment["body"], stale))

    backend._mark_guide_comments(
        "https://api.github.com", _PR, _HEADERS, _guide_pairs(marked, model="openai/new")
    )

    assert len(client.patched) == 1
    markers = extract_markers(client.bodies[0])
    assert len(markers) == 2
    assert {m.model for m in markers} == {"openai/new"}


def test_guide_marking_skips_when_unauthenticated(monkeypatch, backend):
    client = _FakeClient(user_response="42")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = _guide_pairs(_guide_comment(501, 42))

    backend._mark_guide_comments("https://api.github.com", _PR, {}, pairs)

    assert client.patched == []


def test_guide_marking_empty_pairs_is_noop(monkeypatch, backend):
    client = _FakeClient(user_response="42")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    backend._mark_guide_comments("https://api.github.com", _PR, _HEADERS, [])
    assert client.patched == []


def test_guide_marking_swallows_patch_errors(monkeypatch, backend):
    class _ErrClient(_FakeClient):
        def patch(self, url, json=None, **_kw):
            raise httpx.HTTPError("403 not your comment")

    client = _ErrClient(user_response="42")
    monkeypatch.setattr(pragent.httpx, "Client", client)
    pairs = _guide_pairs(_guide_comment(501, 42))
    backend._mark_guide_comments("https://api.github.com", _PR, _HEADERS, pairs, actor="42")
    assert client.patched == []
