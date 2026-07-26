"""Tests for the ``fuko kb`` HTTP-client subcommands (the network call is faked)."""

import argparse

import pytest

from sidecar import kbcli

_ROW = {
    "id": "abc-1",
    "repo": "o/r",
    "text": "Declining — this synchronous path is intentional for ordering here.",
    "source": "review_thread",
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
    kbcli._list(
        _ns(repo="o/r", source=None, limit=100, offset=0, q=None, include_expired=False, full=False)
    )
    out = capsys.readouterr().out
    assert seen["method"] == "GET" and seen["path"] == "/learnings"
    assert seen["params"]["repo"] == "o/r"
    assert "1 shown · 7 total" in out
    assert "Declining" in out


def test_list_passes_search_and_expired_filters(monkeypatch):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(params)
        return {"learnings": [], "count": 0}

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._list(
        _ns(repo=None, source=None, limit=10, offset=0, q="glob", include_expired=True, full=False)
    )
    assert seen["q"] == "glob"
    assert seen["include_expired"] is True


def test_list_omits_include_expired_when_not_asked(monkeypatch):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(params)
        return {"learnings": [], "count": 0}

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._list(
        _ns(repo=None, source=None, limit=10, offset=0, q=None, include_expired=False, full=False)
    )
    assert seen["include_expired"] is None


_REPOS = {
    "repos": [
        {"repo": "o/r", "count": 3, "sources": {"review_thread": 2, "remember": 1}},
        {"repo": "o/s", "count": 5, "sources": {"docs": 5}},
    ]
}


def test_repos_prints_totals_and_source_mix(monkeypatch, capsys):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(method=method, path=path)
        return _REPOS

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._repos(_ns())
    out = capsys.readouterr().out
    assert seen == {"method": "GET", "path": "/repos"}
    assert "2 repo(s) · 8 learnings" in out
    assert "review_thread=2" in out and "docs=5" in out


def test_count_reads_repos_instead_of_paging(monkeypatch, capsys):
    calls = []

    def fake(method, path, params=None, body=None):
        calls.append(path)
        return _REPOS

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._count(_ns(repo=None, source=None))
    out = capsys.readouterr().out
    assert calls == ["/repos"]
    assert "8 total" in out
    assert "review_thread" in out and "docs" in out


def test_count_narrows_by_repo_and_source(monkeypatch, capsys):
    monkeypatch.setattr(kbcli, "_call", lambda *a, **k: _REPOS)
    kbcli._count(_ns(repo="o/r", source="remember"))
    out = capsys.readouterr().out
    assert "1 total" in out
    assert "o/s" not in out and "review_thread" not in out


def test_edit_patches_only_the_supplied_fields(monkeypatch, capsys):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(method=method, path=path, body=body)
        return {**_ROW, "topic": "new topic"}

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._edit(
        _ns(
            repo="o/r",
            id="abc-1",
            text=None,
            source=None,
            source_url=None,
            topic="new topic",
            expires_at=None,
            globs=None,
        )
    )
    assert seen["method"] == "PATCH" and seen["path"] == "/learnings/abc-1"
    assert seen["body"] == {"repo": "o/r", "topic": "new topic"}
    assert "updated" in capsys.readouterr().out


def test_edit_can_clear_globs_with_an_empty_list(monkeypatch):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(body=body)
        return _ROW

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._edit(
        _ns(
            repo="o/r",
            id="abc-1",
            text=None,
            source=None,
            source_url=None,
            topic=None,
            expires_at=None,
            globs=[],
        )
    )
    assert seen["body"] == {"repo": "o/r", "file_globs": []}


def test_edit_refuses_a_no_op(monkeypatch):
    monkeypatch.setattr(kbcli, "_call", lambda *a, **k: {})
    with pytest.raises(SystemExit) as exc:
        kbcli._edit(
            _ns(
                repo="o/r",
                id="abc-1",
                text=None,
                source=None,
                source_url=None,
                topic=None,
                expires_at=None,
                globs=None,
            )
        )
    assert "nothing to change" in str(exc.value)


def test_call_exits_on_value_error(monkeypatch):
    monkeypatch.setenv("FUKO_AUTH_TOKEN", "t")

    def boom(*a, **k):
        raise ValueError("unknown url type")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(SystemExit):
        kbcli._call("GET", "/learnings")


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


def _edit_ns(**over):
    base = dict(
        repo="o/r",
        id="abc-1",
        text=None,
        source=None,
        source_url=None,
        topic=None,
        expires_at=None,
        globs=None,
    )
    base.update(over)
    return _ns(**base)


@pytest.mark.parametrize("field", ["source_url", "topic", "expires_at"])
def test_edit_clears_a_nullable_field_with_an_empty_string(monkeypatch, field):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(body=body)
        return _ROW

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._edit(_edit_ns(**{field: ""}))
    assert seen["body"] == {"repo": "o/r", field: None}


def test_edit_does_not_clear_text_or_source_on_an_empty_string(monkeypatch):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(body=body)
        return _ROW

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._edit(_edit_ns(text=""))
    assert seen["body"] == {"repo": "o/r", "text": ""}


def test_edit_omits_fields_that_were_not_passed(monkeypatch):
    seen = {}

    def fake(method, path, params=None, body=None):
        seen.update(body=body)
        return _ROW

    monkeypatch.setattr(kbcli, "_call", fake)
    kbcli._edit(_edit_ns(topic="Migrations"))
    assert seen["body"] == {"repo": "o/r", "topic": "Migrations"}
