"""Unit tests for review-thread learning selection (decline capture)."""

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
        "Fixed in 98df6b4 — the confirmation now resets on new input as requested.",
        "Already addressed (fbffa0b, round 1) — README line 25 reads the right thing.",
        "Not addressing this — already in place since fbffa0b (round 1); see README.",
        "Done in abc1234 — switched to the async path for ordering as suggested.",
        "Resolved in def5678 — added the missing null check on the request body.",
        "Resolve handled in 1a2b3c4 by clamping the radius before the query runs.",
    ],
)
def test_fix_ack_variants_dropped(body):
    t = _thread(comments=[_comment("coderabbitai[bot]", "bug here"), _comment("alice", body)])
    assert select_learning(t) is None


def test_sha_free_decline_is_kept():
    body = "Not addressing this — verified it is not an issue for this opensearch image."
    assert select_learning(_thread(comments=[_comment("alice", body)])) is not None


@pytest.mark.parametrize(
    "body",
    [
        "Addressed: the script skips gracefully when FUKO_URL is unset (exits 0).",
        "Addressed in the current workflow: the trust filter excludes bots three ways.",
        "Done — switched to the async path so ordering is preserved across the queue.",
        "Already addressed: README line 25 now states the Docker Compose requirement.",
    ],
)
def test_verb_led_acks_dropped_without_sha(body):
    assert select_learning(_thread(comments=[_comment("alice", body)])) is None


@pytest.mark.parametrize(
    "body",
    [
        "Declining — vitest hoists vi.mock and vi.hoisted above the file imports here.",
        "Intentional — committing the SealedSecret in base is the sealed-secrets design.",
        "Verified false positive — eslint-config-next 16.2.4 DOES export a flat config.",
        "Not applicable — salePrice and soldAt are not part of the OpenSearch projection.",
    ],
)
def test_genuine_declines_kept(body):
    assert select_learning(_thread(comments=[_comment("alice", body)])) is not None


def test_deferral_is_dropped():
    t = _thread(
        comments=[
            _comment("copilot", "edge case"),
            _comment("alice", "Filed as #1344 — paginate reviewThreads beyond 100 for large PRs."),
        ]
    )
    assert select_learning(t) is None


@pytest.mark.parametrize(
    "body",
    [
        "Deferring this to a follow-up; not addressing it in this PR for scope reasons.",
        "Deferred — this is out of scope for the current change, tracking separately.",
        "This is a valid deferral, will address in a dedicated change later on here.",
    ],
)
def test_deferral_word_forms_dropped(body):
    t = _thread(comments=[_comment("alice", body)])
    assert select_learning(t) is None


@pytest.mark.parametrize(
    "body",
    [
        "This change will open a security hole if we drop the lock around the cache.",
        "We will address your concern below: the queue already guarantees ordering.",
        "Deferential to that point, but the synchronous path is intentional here.",
    ],
)
def test_substantive_decisions_not_dropped_as_deferrals(body):
    t = _thread(comments=[_comment("alice", body)])
    assert select_learning(t) is not None


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
