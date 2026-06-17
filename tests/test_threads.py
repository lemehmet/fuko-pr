"""Unit tests for resolved-thread learning selection."""

from sidecar.threads import select_learning


def _thread(resolved=True, path="src/a.py", comments=None):
    return {
        "isResolved": resolved,
        "path": path,
        "comments": {"nodes": comments if comments is not None else []},
    }


def _comment(login, body, url="u"):
    return {"author": {"login": login}, "body": body, "url": url}


def test_selects_last_human_comment():
    t = _thread(
        comments=[_comment("github-actions[bot]", "consider X"), _comment("alice", "we use Y")]
    )
    item = select_learning(t)
    assert item is not None
    assert item.text == "we use Y"
    assert item.source == "resolved_thread"
    assert item.file_globs == ["src/a.py"]
    assert item.source_url == "u"


def test_unresolved_thread_ignored():
    t = _thread(resolved=False, comments=[_comment("alice", "note")])
    assert select_learning(t) is None


def test_bot_only_thread_ignored():
    t = _thread(comments=[_comment("github-actions[bot]", "bot only")])
    assert select_learning(t) is None


def test_empty_human_body_ignored():
    t = _thread(comments=[_comment("alice", "   ")])
    assert select_learning(t) is None


def test_missing_author_treated_as_non_human():
    t = _thread(comments=[{"author": None, "body": "ghost", "url": "u"}])
    assert select_learning(t) is None


def test_custom_bot_login_excluded():
    t = _thread(comments=[_comment("my-reviewer-app", "app comment")])
    assert select_learning(t, bot_login="my-reviewer-app") is None


def test_no_path_yields_global_learning():
    t = _thread(path=None, comments=[_comment("alice", "global rule")])
    item = select_learning(t)
    assert item.file_globs == []
