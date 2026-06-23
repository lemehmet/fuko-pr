"""Unit tests for review-thread learning selection (decline allowlist)."""

import pytest

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


@pytest.mark.parametrize(
    "body",
    [
        "Declining — vitest hoists vi.mock and vi.hoisted above the file imports here.",
        "Not adding this one: DeckLink open_sink requires a real device to reach it.",
        "Not applicable — salePrice and soldAt are not part of the OpenSearch projection.",
        "Intentional — committing the SealedSecret in base is the sealed-secrets design.",
        "Verified false positive — eslint-config-next 16.2.4 DOES export a flat config.",
        "Good eye — but it's actually required, not redundant, for the skill bash blocks.",
        "Moot now — the RUNNER_TEMP venv was removed when this switched to uvx entirely.",
        "Premises here are off: this repo is private, so there is no public exposure here.",
        "No change needed — saved-searches is a middleware-protected server shell anyway.",
    ],
)
def test_decline_stances_kept(body):
    assert select_learning(_thread(comments=[_comment("alice", body)])) is not None


@pytest.mark.parametrize(
    "body",
    [
        "Fixed in 98df6b4 — the confirmation now resets on new input as requested.",
        "Added in 1aa7f907: a new test 'renders the error status indicator' for the panel.",
        "Added `drains_a_buffered_frame_while_still_syncing` — pushes a single frame here.",
        "Good catch — fixed. Set tool_timeout = 600 in .fuko.toml for the slow review path.",
        "Good catch — updated the PR description to match the current uvx implementation.",
        "Strengthened in 1aa7f907: the assertion now matches the full leg-prefixed string.",
        "Switched to the JSON form in 3b3f809a so the parser stops tripping on tart output.",
        "Filed as #1344 — paginate reviewThreads beyond 100 for very large pull requests.",
    ],
)
def test_non_declines_dropped(body):
    t = _thread(comments=[_comment("coderabbitai[bot]", "finding"), _comment("alice", body)])
    assert select_learning(t) is None


@pytest.mark.parametrize(
    "body",
    [
        "Deferring this to a follow-up; not addressing it in this PR for scope reasons.",
        "Deferred — not changing it here; tracked in #1344 for a dedicated pass later on.",
        "Not adding this now — filed as #1290 to handle the device-claim path separately.",
    ],
)
def test_deferrals_dropped_even_when_they_read_as_declines(body):
    assert select_learning(_thread(comments=[_comment("alice", body)])) is None


def test_short_decline_is_dropped():
    t = _thread(comments=[_comment("alice", "Declining.")])
    assert select_learning(t) is None


def test_bot_only_thread_ignored():
    t = _thread(comments=[_comment("github-actions[bot]", "Declining — long enough bot body here")])
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
