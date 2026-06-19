"""Tests for per-reviewer state detection (fuko status), grounded in survey forms."""

from sidecar.status import coderabbit_state, copilot_state, reviewer_states

HEAD = "def5678abc0000000000000000000000000000aa"


def _cr(body):
    return {"user": {"login": "coderabbitai[bot]"}, "body": body}


def _walk(reviewed_sha, *, zero=False, extra=""):
    body = (
        "📝 Walkthrough\n\nReviewing files that changed from the base of the PR "
        f"and between `abc1234` and `{reviewed_sha}`.\n{extra}"
    )
    if zero:
        body += "\nNo actionable comments were generated in the recent review. 🎉"
    return _cr(body)


def _cr_review(commit_id, state="COMMENTED"):
    return {"user": {"login": "coderabbitai[bot]"}, "commit_id": commit_id, "state": state}


def test_coderabbit_done_zero_via_walkthrough():
    s = coderabbit_state(HEAD, [_walk("def5678", zero=True)], [])  # abbreviated sha prefixes HEAD
    assert s["state"] == "done"
    assert "no actionable comments" in s["detail"]


def test_coderabbit_done_with_findings_via_walkthrough():
    s = coderabbit_state(HEAD, [_walk(HEAD)], [])
    assert s["state"] == "done"
    assert "inline" in s["detail"]


def test_coderabbit_done_via_review_commit_id_when_walkthrough_has_no_range():
    # #1333/#1326 shape: walkthrough lacks the "between … and …" line, but the CR
    # review object is on HEAD — that alone must resolve to done.
    plain_walkthrough = _cr("📝 Walkthrough\n\nIntroduces a new file. No range line here.")
    s = coderabbit_state(HEAD, [plain_walkthrough], [_cr_review(HEAD)])
    assert s["state"] == "done"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_pending_when_neither_signal_covers_head():
    s = coderabbit_state(HEAD, [_walk("0000aaa")], [_cr_review("0000aaa")])
    assert s["state"] == "pending"


def test_coderabbit_in_progress():
    cs = [_walk("0000aaa"), _cr("🔬 review in progress — Currently processing new changes")]
    assert coderabbit_state(HEAD, cs, [])["state"] == "in_progress"


def test_coderabbit_rate_limited():
    cs = [_walk("0000aaa"), _cr("⚠️ Rate limit exceeded. Try again in 8 minutes and 9 seconds.")]
    assert coderabbit_state(HEAD, cs, [])["state"] == "rate_limited"


def test_coderabbit_paused():
    cs = [_walk("0000aaa"), _cr("## Reviews paused\n<!-- review paused by coderabbit.ai -->")]
    assert coderabbit_state(HEAD, cs, [])["state"] == "paused"


def test_coderabbit_transient_masked_once_head_scanned():
    # an earlier rate-limit notice must NOT override a later completed scan of HEAD
    cs = [_cr("Rate limit exceeded earlier"), _walk("def5678", zero=True)]
    assert coderabbit_state(HEAD, cs, [])["state"] == "done"


def test_coderabbit_none():
    assert coderabbit_state(HEAD, [{"user": {"login": "x"}, "body": "hi"}], [])["state"] == "none"


def test_copilot_done_on_head():
    reviews = [{"user": {"login": "Copilot"}, "commit_id": HEAD, "state": "COMMENTED"}]
    s = copilot_state(HEAD, reviews)
    assert s["state"] == "done" and s["head_reviewed"] == HEAD


def test_copilot_pending_on_older_commit():
    reviews = [{"user": {"login": "Copilot"}, "commit_id": "old111", "state": "COMMENTED"}]
    assert copilot_state(HEAD, reviews)["state"] == "pending"


def test_copilot_none():
    assert copilot_state(HEAD, [{"user": {"login": "coderabbitai[bot]"}}])["state"] == "none"


def test_reviewer_states_returns_both():
    rows = reviewer_states(
        HEAD,
        [_walk(HEAD)],
        [
            {"user": {"login": "Copilot"}, "commit_id": HEAD, "state": "APPROVED"},
            _cr_review(HEAD),
        ],
    )
    assert [r["backend"] for r in rows] == ["coderabbit", "copilot"]
    assert all(r["state"] == "done" for r in rows)
