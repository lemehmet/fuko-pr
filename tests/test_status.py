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


def _cr_review(commit_id, state="COMMENTED", body=""):
    return {
        "user": {"login": "coderabbitai[bot]"},
        "commit_id": commit_id,
        "state": state,
        "body": body,
    }


def _review_body(reviewed_sha, *, posted=None, zero=False):
    """A CodeRabbit review body — where CR posts the range line and terminal marker."""
    body = (
        "Reviewing files that changed from the base of the PR "
        f"and between `abc1234` and `{reviewed_sha}`."
    )
    if zero:
        body += "\nNo actionable comments"
    if posted is not None:
        body += f"\n**Actionable comments posted: {posted}**"
    return body


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


def test_coderabbit_pending_via_walkthrough_range_line_only():
    """issue #34: walkthrough range line covers HEAD, no completion marker/review/check.

    The ``_CR_REVIEWING`` range line names HEAD but there is no terminal marker, no
    submitted review on HEAD, and no in-progress check-run or notice. Under a rapid-push
    skip CR can bump that range line to a new HEAD and then never engage it, so the range
    line alone is not evidence of an active scan -> pending (the consumer's unresponsive
    timeout governs), not a never-ending in_progress.
    """
    s = coderabbit_state(HEAD, [_walk(HEAD)], [])
    assert s["state"] == "pending"
    assert s["head_reviewed"] == HEAD
    assert "walkthrough range line covers HEAD" in s["detail"]


def test_coderabbit_in_progress_when_header_names_head_and_active_notice():
    """issue #34: range line covering HEAD plus an explicit in-progress notice -> in_progress.

    Demonstrable engagement (an active-scan notice) alongside the range line is enough to
    report in_progress rather than pending.
    """
    cs = [_walk(HEAD), _cr("🔬 review in progress — Currently processing new changes")]
    s = coderabbit_state(HEAD, cs, [])
    assert s["state"] == "in_progress"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_in_progress_when_header_names_head_and_check_running():
    """issue #34: a still-running CR check-run on HEAD is authoritative engagement.

    The check-run overrides the range-line-only pending classification.
    """
    s = coderabbit_state(HEAD, [_walk(HEAD)], [], [_check("in_progress")])
    assert s["state"] == "in_progress"


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


def test_coderabbit_done_via_review_on_head_without_marker():
    # A submitted CR review on HEAD is terminal even with no marker — its inline comments
    # post atomically with it. (issue #18 round 2: CR APPROVED with an empty review body
    # and fuko was wrongly stuck in_progress for the full timeout.)
    plain = _cr("📝 Walkthrough\n\nIntroduces a new file. No range line here.")
    s = coderabbit_state(HEAD, [plain], [_cr_review(HEAD)])
    assert s["state"] == "done"


def test_coderabbit_done_on_approved_review_with_empty_body():
    s = coderabbit_state(HEAD, [], [_cr_review(HEAD, state="APPROVED", body="")])
    assert s["state"] == "done"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_done_when_marker_in_review_body():
    # PR #18 deadlock shape: CR posts the range line AND the terminal marker in its
    # REVIEW body, while the summary issue comment carries neither. Searching review
    # bodies (not just issue comments) is what resolves it to done.
    summary = _cr("📝 Walkthrough\n\nSummary text, no marker here.")
    review = _cr_review(HEAD, state="CHANGES_REQUESTED", body=_review_body(HEAD, posted=2))
    s = coderabbit_state(HEAD, [summary], [review])
    assert s["state"] == "done"
    assert "inline" in s["detail"]
    assert s["head_reviewed"] == HEAD


def test_coderabbit_stale_marker_in_old_review_does_not_satisfy_head():
    """A stale terminal marker on an old review must not satisfy done for the new HEAD.

    Current HEAD is covered only by the up-front walkthrough range line (no submitted
    review yet); an older review for a previous HEAD carries the terminal marker. That
    stale marker must NOT satisfy done for the new HEAD (CodeRabbit finding, round 1).
    With only the range line covering HEAD and no in-progress signal, this is pending
    (#34), not done. ``head_reviewed`` must be the current HEAD, not the older scan's sha.
    """
    old = _cr_review("0000aaa", body=_review_body("0000aaa", posted=3))
    walk = _walk(HEAD)
    s = coderabbit_state(HEAD, [walk], [old])
    assert s["state"] == "pending"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_rapid_push_skip_reports_pending_not_indefinite_in_progress():
    """issue #34 acceptance: rapid-push skip reports pending, not indefinite in_progress.

    CR reviewed the previous commit (CHANGES_REQUESTED), then its walkthrough range line
    was bumped to the latest HEAD after >=3 quick pushes, but CR never engaged that HEAD
    (no check-run, no review on HEAD, no terminal marker for HEAD). fuko must report
    pending, not a never-ending in_progress that burns the consumer's full per-bot timeout.
    """
    prev = _cr_review("0000aaa", state="CHANGES_REQUESTED", body=_review_body("0000aaa", posted=2))
    range_line_on_head = _walk(HEAD)
    s = coderabbit_state(HEAD, [range_line_on_head], [prev])
    assert s["state"] == "pending"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_done_short_no_actionable_marker():
    # The shorter "No actionable comments" terminal marker (named in the fallback
    # contract) must count, not only "No actionable comments were generated".
    s = coderabbit_state(HEAD, [], [_cr_review(HEAD, body=_review_body(HEAD, zero=True))])
    assert s["state"] == "done"
    assert "no actionable comments" in s["detail"]


def test_coderabbit_marker_ignores_inline_prose_quoting_the_phrase():
    # CR's review body can quote "No actionable comments" mid-line in prose (e.g. a
    # finding about the marker regex). That must NOT be read as the zero marker — only
    # the real line-anchored "Actionable comments posted: N" counts (issue #18 round 1).
    body = (
        _review_body(HEAD, posted=2)
        + '\nIt also names the shorter "No actionable comments" variant in the contract.'
    )
    s = coderabbit_state(HEAD, [], [_cr_review(HEAD, body=body)])
    assert s["state"] == "done"
    assert "inline" in s["detail"]
    assert "no actionable comments" not in s["detail"]


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
