"""Tests for the ``fuko kb`` HTTP-client subcommands (the network call is faked)."""

import argparse

import pytest

from sidecar import kbcli

_ROW = {
    "id": "abc-1",
    "repo": "o/r",
    "text": "Declining — this synchronous path is intentional for ordering here.",
    "source": "resolved_thread",
    "source_url": "https://example/pull/1#r1",
    "file_globs": ["a.py"],
    "topic": "review decision",
    "created_at": "2026-06-23T00:00:00+00:00",
}


def _ns(**kw):
    return argparse.Namespace(**kw)


def test_list_calls_learnings_and_prints(monkeypatch, capsys):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(method=method, path=path, params=params)
        return {"learnings": [_ROW], "count": 7}

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._list(_ns(repo="o/r", source=None, limit=100, offset=0, full=False))
    out = capsys.readouterr().out
    assert seen["method"] == "GET" and seen["path"] == "/learnings"
    assert seen["params"]["repo"] == "o/r"
    assert "1 shown · 7 total" in out
    assert "Declining" in out


def test_count_aggregates_by_repo_and_source(monkeypatch, capsys):
    rows = [
        {**_ROW, "repo": "o/r", "source": "resolved_thread"},
        {**_ROW, "repo": "o/r", "source": "remember"},
        {**_ROW, "repo": "o/r", "source": "resolved_thread"},
    ]
    monkeypatch.setattr(kbcli, "_call", lambda *a, **k: {"learnings": rows, "count": 3})
    kbcli._count(_ns(repo=None, source=None))
    out = capsys.readouterr().out
    assert "3 total" in out
    assert "resolved_thread" in out and "remember" in out


def test_query_builds_post_body(monkeypatch, capsys):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(method=method, path=path, body=body)
        return {"results": [{**_ROW, "score": 0.91}]}

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._query(_ns(repo="o/r", files=["a.py"], text="ordering", pr_body=None, top_k=3))
    out = capsys.readouterr().out
    assert seen["method"] == "POST" and seen["path"] == "/query"
    assert seen["body"] == {"repo": "o/r", "files": ["a.py"], "query_text": "ordering", "top_k": 3}
    assert "score 0.910" in out


def test_forget_by_id_posts_selector(monkeypatch, capsys):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(path=path, body=body)
        return {"deleted": 1}

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._forget(_ns(repo="o/r", id="abc-1", source=None, all=False, yes=False))
    assert seen["path"] == "/forget"
    assert seen["body"] == {"repo": "o/r", "id": "abc-1"}
    assert "deleted 1" in capsys.readouterr().out


def test_forget_all_requires_confirmation(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    monkeypatch.setattr(kbcli, "_call", lambda *a, **k: pytest.fail("must not call before confirm"))
    with pytest.raises(SystemExit):
        kbcli._forget(_ns(repo="o/r", id=None, source=None, all=True, yes=False))


def test_call_requires_token(monkeypatch):
    monkeypatch.delenv("FUKO_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        kbcli._call("GET", "/learnings")
