"""Unit tests for review-thread learning selection (decline capture)."""

from sidecar.threads import select_learning

_DECLINE = (
    "Not changing this — in this codebase we intentionally keep the synchronous "
    "path because the queue guarantees ordering."
)


def _thread(resolved=True, path="src/a.py", comments=None):
    return {
        "isResolved": resolved,
        "path": path,
        "comments": {"nodes": comments if comments is not None else []},
    }


def _comment(login, body, url="u"):
    return {"author": {"login": login}, "body": body, "url": url}


def test_selects_substantive_decline():
    t = _thread(comments=[_comment("coderabbitai[bot]", "consider X"), _comment("alice", _DECLINE)])
    item = select_learning(t)
    assert item is not None
    assert item.text == _DECLINE
    assert item.source == "resolved_thread"
    assert item.topic == "review decision"
    assert item.file_globs == ["src/a.py"]
    assert item.source_url == "u"


def test_unresolved_decline_is_kept():
    t = _thread(resolved=False, comments=[_comment("alice", _DECLINE)])
    assert select_learning(t) is not None


def test_fix_ack_is_dropped():
    t = _thread(
        comments=[
            _comment("coderabbitai[bot]", "bug here"),
            _comment("alice", "Fixed in 98df6b4 — the confirmation now resets on new input."),
        ]
    )
    assert select_learning(t) is None


def test_deferral_is_dropped():
    t = _thread(
        comments=[
            _comment("copilot", "edge case"),
            _comment("alice", "Filed as #1344 — paginate reviewThreads beyond 100 for large PRs."),
        ]
    )
    assert select_learning(t) is None


def test_short_comment_is_dropped():
    t = _thread(comments=[_comment("alice", "good catch, agreed")])
    assert select_learning(t) is None


def test_bot_only_thread_ignored():
    t = _thread(
        comments=[_comment("github-actions[bot]", "bot only finding, fairly long body here")]
    )
    assert select_learning(t) is None


def test_missing_author_treated_as_non_human():
    t = _thread(comments=[{"author": None, "body": _DECLINE, "url": "u"}])
    assert select_learning(t) is None


def test_custom_bot_login_excluded():
    t = _thread(comments=[_comment("my-reviewer-app", _DECLINE)])
    assert select_learning(t, bot_login="my-reviewer-app") is None


def test_no_path_yields_global_learning():
    t = _thread(path=None, comments=[_comment("alice", _DECLINE)])
    item = select_learning(t)
    assert item.file_globs == []
