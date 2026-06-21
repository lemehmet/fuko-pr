"""Tests for per-reviewer state detection (fuko status), grounded in survey forms."""

from sidecar.status import coderabbit_state, copilot_state, reviewer_states

HEAD = "def5678abc0000000000000000000000000000aa"


def _cr(body):
    return {"user": {"login": "coderabbitai[bot]"}, "body": body}


def _walk(reviewed_sha, *, zero=False, posted=None, extra=""):
    """A CodeRabbit walkthrough comment.

    ``zero`` adds the zero-finding completion line; ``posted`` adds the
    "Actionable comments posted: N" terminal marker. Without either, the comment has
    only the up-front "Reviewing files … between …" range line (no completion marker),
    which models the in-flight window where CR is still streaming inline comments.
    """
    body = (
        "📝 Walkthrough\n\nReviewing files that changed from the base of the PR "
        f"and between `abc1234` and `{reviewed_sha}`.\n{extra}"
    )
    if zero:
        body += "\nNo actionable comments were generated in the recent review. 🎉"
    if posted is not None:
        body += f"\n**Actionable comments posted: {posted}**"
    return _cr(body)


def _cr_review(commit_id, state="COMMENTED"):
    return {"user": {"login": "coderabbitai[bot]"}, "commit_id": commit_id, "state": state}


def _check(status, conclusion=None, name="CodeRabbit", slug=None):
    c = {"name": name, "status": status, "conclusion": conclusion}
    if slug is not None:
        c["app"] = {"slug": slug}
    return c


# --- check-run path (issue #17, authoritative) --------------------------------


def test_coderabbit_in_progress_when_check_pending_despite_walkthrough():
    # The premature-done bug: walkthrough already covers HEAD, but CR's check is still
    # "in_progress" (inline comments not yet posted) — must NOT be done.
    s = coderabbit_state(HEAD, [_walk(HEAD)], [], [_check("in_progress")])
    assert s["state"] == "in_progress"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_done_when_check_completed():
    s = coderabbit_state(HEAD, [_walk(HEAD)], [], [_check("completed", "neutral")])
    assert s["state"] == "done"
    assert s["head_reviewed"] == HEAD
    assert "completed" in s["detail"]


def test_coderabbit_done_when_check_completed_zero_findings():
    s = coderabbit_state(HEAD, [_walk(HEAD, zero=True)], [], [_check("completed", "success")])
    assert s["state"] == "done"
    assert "no actionable comments" in s["detail"]


def test_coderabbit_check_matches_by_app_slug():
    chk = _check("in_progress", name="Review", slug="coderabbitai")
    assert coderabbit_state(HEAD, [_walk(HEAD)], [], [chk])["state"] == "in_progress"


def test_coderabbit_check_queued_is_in_progress():
    assert coderabbit_state(HEAD, [_walk(HEAD)], [], [_check("queued")])["state"] == "in_progress"


def test_coderabbit_unrelated_checks_ignored_falls_back_to_comments():
    # Only non-CR checks present -> ignore them, use the comment fallback.
    chk = _check("in_progress", name="ci/build", slug="github-actions")
    s = coderabbit_state(HEAD, [_walk(HEAD, posted=2)], [], [chk])
    assert s["state"] == "done"  # marker present, no CR check to gate on


# --- comment fallback path (no observable CR check-run) -----------------------


def test_coderabbit_done_zero_via_walkthrough():
    s = coderabbit_state(HEAD, [_walk("def5678", zero=True)], [])  # abbreviated sha prefixes HEAD
    assert s["state"] == "done"
    assert "no actionable comments" in s["detail"]


def test_coderabbit_done_with_findings_via_walkthrough_marker():
    # With findings, fallback requires the "Actionable comments posted" terminal marker.
    s = coderabbit_state(HEAD, [_walk(HEAD, posted=3)], [])
    assert s["state"] == "done"
    assert "inline" in s["detail"]


def test_coderabbit_in_progress_via_walkthrough_without_marker():
    # HEAD scanned (range line matches) but no completion marker yet -> still in progress.
    s = coderabbit_state(HEAD, [_walk(HEAD)], [])
    assert s["state"] == "in_progress"
    assert "completion marker" in s["detail"]


def test_coderabbit_done_via_review_commit_id_with_marker():
    # #1333/#1326 shape: walkthrough lacks the "between … and …" line, but the CR
    # review object is on HEAD. With a terminal marker present this resolves to done.
    plain = _cr(
        "📝 Walkthrough\n\nIntroduces a new file. No range line here.\n"
        "**Actionable comments posted: 0**"
    )
    s = coderabbit_state(HEAD, [plain], [_cr_review(HEAD)])
    assert s["state"] == "done"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_in_progress_via_review_commit_id_without_marker():
    plain = _cr("📝 Walkthrough\n\nIntroduces a new file. No range line here.")
    s = coderabbit_state(HEAD, [plain], [_cr_review(HEAD)])
    assert s["state"] == "in_progress"


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


# --- copilot ------------------------------------------------------------------


def test_copilot_done_on_head():
    reviews = [{"user": {"login": "Copilot"}, "commit_id": HEAD, "state": "COMMENTED"}]
    s = copilot_state(HEAD, reviews)
    assert s["state"] == "done" and s["head_reviewed"] == HEAD


def test_copilot_pending_on_older_commit():
    reviews = [{"user": {"login": "Copilot"}, "commit_id": "old111", "state": "COMMENTED"}]
    assert copilot_state(HEAD, reviews)["state"] == "pending"


def test_copilot_none():
    assert copilot_state(HEAD, [{"user": {"login": "coderabbitai[bot]"}}])["state"] == "none"


# --- reviewer_states ----------------------------------------------------------


def test_reviewer_states_returns_both():
    rows = reviewer_states(
        HEAD,
        [_walk(HEAD, posted=1)],
        [
            {"user": {"login": "Copilot"}, "commit_id": HEAD, "state": "APPROVED"},
            _cr_review(HEAD),
        ],
    )
    assert [r["backend"] for r in rows] == ["coderabbit", "copilot"]
    assert all(r["state"] == "done" for r in rows)


def test_reviewer_states_threads_check_runs():
    rows = reviewer_states(
        HEAD,
        [_walk(HEAD)],
        [{"user": {"login": "Copilot"}, "commit_id": HEAD, "state": "APPROVED"}],
        [_check("in_progress")],
    )
    cr = next(r for r in rows if r["backend"] == "coderabbit")
    assert cr["state"] == "in_progress"
