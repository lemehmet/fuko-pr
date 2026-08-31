"""Tests for per-reviewer state detection (fuko status), grounded in survey forms."""

import pytest

from sidecar.signals import RunReceipt, with_run_receipt
from sidecar.status import (
    DEGRADED_STATES,
    coderabbit_state,
    copilot_state,
    escalation_needed,
    fuko_states,
    reviewer_states,
)

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


def test_coderabbit_in_progress_when_range_line_covers_head_and_active_notice():
    """issue #34: range line covering HEAD plus an explicit in-progress notice -> in_progress.

    Demonstrable engagement (an active-scan notice) alongside the range line is enough to
    report in_progress rather than pending.
    """
    cs = [_walk(HEAD), _cr("🔬 review in progress — Currently processing new changes")]
    s = coderabbit_state(HEAD, cs, [])
    assert s["state"] == "in_progress"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_pending_when_in_progress_phrase_only_in_stale_review_body():
    """The in-progress notice is scoped to CR's live status issue comments.

    A stale "review in progress" phrase carried in an older submitted review body (for a
    prior HEAD) must NOT flip the range-line-only HEAD back to in_progress — otherwise the
    rapid-push-skip case (#34) would never settle to pending. Only the live status issue
    comment counts as an active-scan notice.
    """
    stale = _cr_review("0000aaa", body="🔬 review in progress — Currently processing new changes")
    s = coderabbit_state(HEAD, [_walk(HEAD)], [stale])
    assert s["state"] == "pending"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_head_reviewed_is_the_range_that_covers_head():
    """#101, reproduced from PR #100: two CR reviews, each with its own range line.

    `walk_on_head` scanned every body while `walk_head` was bound from the FIRST
    one, so once a PR had more than one CR review the reported `head_reviewed`
    was review 1's OLD end-sha while `state` was correctly `done`.
    """
    older = _cr_review("11617069", body=_review_body("11617069", posted=1))
    newer = _cr_review(HEAD, body=_review_body(HEAD, posted=2))
    s = coderabbit_state(HEAD, [], [older, newer])
    assert s["state"] == "done"
    assert s["head_reviewed"] == HEAD
    assert s["head_reviewed"] != "11617069"


def test_coderabbit_head_reviewed_prefers_head_regardless_of_body_order():
    """Order must not decide it — the range that covers HEAD wins wherever it sits."""
    newer = _cr_review(HEAD, body=_review_body(HEAD, posted=2))
    older = _cr_review("11617069", body=_review_body("11617069", posted=1))
    assert coderabbit_state(HEAD, [], [newer, older])["head_reviewed"] == HEAD


def test_coderabbit_head_reviewed_falls_back_to_the_newest_range_not_the_first():
    """With no range covering HEAD, report the most recent one seen, not the oldest."""
    first = _cr(_review_body("aaaaaaa"))
    second = _cr(_review_body("bbbbbbb"))
    s = coderabbit_state(HEAD, [first, second], [])
    assert s["head_reviewed"] == "bbbbbbb"


def test_coderabbit_reads_every_range_line_within_one_body():
    """A single rewritten summary can carry several range lines; the last covers HEAD."""
    body = _cr(
        "Reviewing files that changed from the base of the PR and between `aaa1111` "
        "and `bbb2222`.\nReviewing files that changed from the base of the PR and "
        f"between `bbb2222` and `{HEAD}`.\n**Actionable comments posted: 0**"
    )
    s = coderabbit_state(HEAD, [body], [])
    assert s["state"] == "done"
    assert s["head_reviewed"] == HEAD


# --- throttle visibility (#19 control flow + #86 pattern coverage) ------------

_FAIR_USAGE = (
    "<!-- This is an auto-generated comment: summarize by coderabbit.ai -->\n"
    "<!-- This is an auto-generated comment: rate limited by coderabbit.ai -->\n\n"
    "> [!WARNING]\n> ## Review limit reached\n>\n"
    "> You've reached a temporary PR review limit under our Fair Usage Limits Policy.\n"
    "> Your recent review volume is higher than typical usage, so adaptive limits "
    "are currently applied.\n>\n"
    "> **Next review available in:** **28 minutes**\n"
    "<!-- end of auto-generated comment: rate limited by coderabbit.ai -->"
)

_PAUSE_NOTICE = (
    "<!-- This is an auto-generated comment: summarize by coderabbit.ai -->\n"
    "> [!WARNING]\n> ## Reviews paused\n>\n"
    "> Use the following command to resume: `@coderabbitai resume`"
)


def test_coderabbit_fair_usage_limit_with_head_walkthrough_is_rate_limited():
    """#19 + #86 together, from mepro #1584.

    The adaptive-limit notice contains NO "Rate limit exceeded" text (#86) and
    rides in the summary comment that also carries the HEAD range line, so the
    old code took the head-scanned path and reported `pending` (#19). CR never
    reviewed that PR at all; `rate_limited` was accurate for two hours.
    """
    body = _walk(HEAD)["body"] + "\n" + _FAIR_USAGE
    s = coderabbit_state(HEAD, [_cr(body)], [])
    assert s["state"] == "rate_limited"
    assert s["state"] in DEGRADED_STATES
    assert escalation_needed([s]) is True


def test_coderabbit_hourly_rate_limit_with_head_walkthrough_is_rate_limited():
    """#19 on its own: the hourly wording, which the pattern already matched."""
    body = _walk(HEAD)["body"] + "\nRate limit exceeded. Try again later."
    assert coderabbit_state(HEAD, [_cr(body)], [])["state"] == "rate_limited"


def test_coderabbit_paused_with_head_walkthrough_is_paused():
    body = _walk(HEAD)["body"] + "\nReviews paused by coderabbit.ai"
    assert coderabbit_state(HEAD, [_cr(body)], [])["state"] == "paused"


def test_coderabbit_fair_usage_marker_alone_is_enough():
    """The machine marker is the stable anchor; prose may change under us."""
    body = _walk(HEAD)["body"] + (
        "\n<!-- This is an auto-generated comment: rate limited by coderabbit.ai -->"
    )
    assert coderabbit_state(HEAD, [_cr(body)], [])["state"] == "rate_limited"


def test_coderabbit_throttle_outranks_a_stale_in_progress_phrase():
    """A throttled CR is not scanning, whatever an older phrase in the same body says."""
    body = _walk(HEAD)["body"] + "\nreview in progress\n" + _FAIR_USAGE
    assert coderabbit_state(HEAD, [_cr(body)], [])["state"] == "rate_limited"


def test_coderabbit_earlier_rate_limit_does_not_mask_a_later_completed_scan():
    """The sticky-guard, preserved: a completed scan on HEAD still wins.

    This is the assertion that keeps the #19 fix from making throttle states
    sticky — the throttle check sits INSIDE the not-yet-completed branch.
    """
    throttled = _cr("Rate limit exceeded\n" + _FAIR_USAGE)
    done = _walk(HEAD, posted=2)
    assert coderabbit_state(HEAD, [throttled, done], [])["state"] == "done"


def test_coderabbit_empty_review_on_head_under_a_live_notice_is_demoted():
    """#137 — INVERTS the old `a review on HEAD outranks a live throttle notice`.

    Under Fair Usage CR still submits a review row on HEAD, with an empty body and
    no walkthrough. That empty acknowledgement used to read as `done`, so the gate
    opened on a commit CR never read. With a live notice in CR's in-place summary
    and no review content anywhere on HEAD, the notice demotes it.
    """
    throttled = _cr(_FAIR_USAGE)
    s = coderabbit_state(HEAD, [throttled], [_cr_review(HEAD, state="APPROVED")])
    assert s["state"] == "rate_limited"
    assert s["reviewed_head_with_content"] is False
    assert escalation_needed([s]) is True


def test_coderabbit_check_run_completed_under_a_live_notice_is_demoted():
    """#137, the check-run door: PR #172 rounds 4/8 and PR #178 on this repo.

    CR's check-run completes under Fair Usage just as it does after a real review,
    so the authoritative-signal path (#17) reported `done` on heads CR never read.
    """
    s = coderabbit_state(HEAD, [_cr(_FAIR_USAGE)], [], [_check("completed", "success")])
    assert s["state"] == "rate_limited"
    assert s["state"] in DEGRADED_STATES
    assert s["reviewed_head_with_content"] is False


def test_coderabbit_check_run_completed_with_a_notice_naming_head_is_demoted():
    """The observed field shape: the notice's range line is already bumped to HEAD."""
    body = _walk(HEAD)["body"] + "\n" + _FAIR_USAGE
    s = coderabbit_state(HEAD, [_cr(body)], [], [_check("completed", "success")])
    assert s["state"] == "rate_limited"
    assert s["head_reviewed"] == HEAD


def test_coderabbit_stale_marker_in_a_bumped_notice_summary_is_not_evidence():
    """The third door: one in-place-edited comment carrying all three artifacts.

    CR rewrites its summary on every push, so a throttled round leaves a body whose
    range line already names the NEW head while the terminal marker in it is the
    PREVIOUS round's. Reading that marker as coverage vouches for a commit CR never
    opened, so a body that is itself a notice cannot supply marker evidence.
    """
    body = _walk(HEAD, posted=2)["body"] + "\n" + _FAIR_USAGE
    s = coderabbit_state(HEAD, [_cr(body)], [])
    assert s["state"] == "rate_limited"
    assert s["reviewed_head_with_content"] is False


def test_coderabbit_stale_one_off_marker_is_not_content_for_head():
    """The fourth door: a stale one-off comment feeding the `with_content` escape hatch.

    A review on HEAD admits CR's live summary as describing HEAD. Admitting CR's
    OTHER issue comments alongside it lets a one-off reply from an earlier round --
    never rewritten, so permanently stale -- supply the terminal marker that cancels
    the demotion, and the live limit notice in the summary is overruled by evidence
    about a different commit (CodeRabbit finding, round 1).
    """
    stale_reply = _cr(
        "<!-- This is an auto-generated reply by CodeRabbit -->\n**Actionable comments posted: 3**"
    )
    s = coderabbit_state(
        HEAD, [_cr(_FAIR_USAGE), stale_reply], [_cr_review(HEAD, state="APPROVED")]
    )
    assert s["state"] == "rate_limited"
    assert s["reviewed_head_with_content"] is False


def test_coderabbit_notice_body_review_on_head_is_not_content():
    """The fifth door: `body_on_head` accepted a review body that IS a notice.

    CR submits reviews whose body is the throttle notice -- the stale-notice tests
    below are built on that shape, and a review is anchored to the head it was
    submitted against, so the live form is a notice-body review on the CURRENT
    head. Counting it as content satisfied the escape hatch, short-circuited the
    demotion, and reported `done` with `reviewed_head_with_content` TRUE for a
    commit CR never read -- #137's failure, through its own discriminator.
    """
    s = coderabbit_state(HEAD, [_cr(_FAIR_USAGE)], [_cr_review(HEAD, body=_FAIR_USAGE)])
    assert s["state"] == "rate_limited"
    assert s["reviewed_head_with_content"] is False


def test_coderabbit_summary_quoting_notice_prose_is_not_a_notice():
    """The false-demote half: notice prose QUOTED mid-line must not be decisional.

    #137 promotes the notice patterns from "choose among transient states" to
    "strip marker evidence and demote a completion", so an unanchored substring
    hit could withhold a genuine review -- and re-withhold it on every push while
    the quote persisted. CR renders the real notice as a heading in its warning
    blockquote; a body discussing one quotes it mid-sentence. This is the same
    false positive `_CR_DONE_MARKER` is line-anchored to avoid.
    """
    body = (
        _walk(HEAD, posted=2)["body"]
        + '\nThe matcher is taught the "Review limit reached" and "Rate limit '
        'exceeded" wordings here.'
    )
    s = coderabbit_state(HEAD, [_cr(body)], [], [_check("completed", "success")])
    assert s["state"] == "done"
    assert s["reviewed_head_with_content"] is True


def test_coderabbit_pause_notice_demotes_to_paused_not_rate_limited():
    """The two notices need different recoveries, so they keep different states.

    A pause clears with `@coderabbitai resume`; a Fair-Usage limit needs credits or
    a wait. Collapsing both into one state costs the consumer that distinction.
    """
    s = coderabbit_state(HEAD, [_cr(_PAUSE_NOTICE)], [], [_check("completed", "success")])
    assert s["state"] == "paused"


def test_coderabbit_demote_falls_back_to_live_comments_without_a_summary():
    """No in-place-edited anchor to scope to — trust the notice, the safe direction."""
    s = coderabbit_state(HEAD, [_cr("Rate limit exceeded")], [], [_check("completed")])
    assert s["state"] == "rate_limited"


def test_coderabbit_notice_in_a_one_off_reply_does_not_demote():
    """The anti-stickiness scoping: only the in-place-edited summary may demote.

    CR's one-off replies are never rewritten, so a notice in one describes a window
    that closed at some point in the past, not the window now. Letting it demote
    would leave the PR degraded for the rest of its life.
    """
    reply = _cr("<!-- This is an auto-generated reply by CodeRabbit -->\nRate limit exceeded")
    summary = _cr(
        "<!-- This is an auto-generated comment: summarize by coderabbit.ai -->\n📝 Walkthrough"
    )
    s = coderabbit_state(
        HEAD, [summary, reply], [_cr_review(HEAD, state="APPROVED")], [_check("completed")]
    )
    assert s["state"] == "done"


def test_coderabbit_real_post_review_summary_with_limit_details_is_done():
    """Over-match regression, from PR #178's actual body after a genuine review.

    A completed CR review reports its remaining allowance as a "Limit details:"
    line inside the walkthrough. That is not a notice, and reading it as one would
    demote every review CR performs while near its limit.
    """
    body = (
        "<!-- This is an auto-generated comment: summarize by coderabbit.ai -->\n"
        "<!-- recent_review_start -->\n\n"
        "No actionable comments were generated in the recent review. 🎉\n\n"
        "Reviewing files that changed from the base of the PR and between "
        f"4a60100d921054f4cd58499bf0aede7bff11e730 and {HEAD}.\n\n"
        "**Limit details:** You've used the included review currently available. "
        "Your 93 included PR review attempts over the past 7 days set your current "
        "allowance at 1 review per hour."
    )
    s = coderabbit_state(HEAD, [_cr(body)], [], [_check("completed", "success")])
    assert s["state"] == "done"
    assert s["reviewed_head_with_content"] is True


def test_coderabbit_empty_approved_review_without_a_notice_stays_done():
    """Issue #18 survives #137: only a live notice changes the STATE.

    The new key still reports the ambiguity — CR acknowledged HEAD without leaving
    anything that proves it read the diff — but with nothing saying the window was
    closed, an empty APPROVED review remains a completion.
    """
    s = coderabbit_state(HEAD, [], [_cr_review(HEAD, state="APPROVED")])
    assert s["state"] == "done"
    assert s["reviewed_head_with_content"] is False


def test_coderabbit_rows_always_carry_the_content_verdict():
    """Every CodeRabbit row exposes the key, so a consumer can read it blind."""
    assert coderabbit_state(HEAD, [], [])["reviewed_head_with_content"] is False
    assert coderabbit_state(HEAD, [_walk(HEAD)], [])["reviewed_head_with_content"] is False
    assert coderabbit_state(HEAD, [_walk(HEAD, posted=2)], [])["reviewed_head_with_content"] is True
    with_body = _cr_review(HEAD, body="**Actionable comments posted: 1**")
    assert coderabbit_state(HEAD, [], [with_body])["reviewed_head_with_content"] is True


def test_reviewer_states_carries_the_content_verdict_on_the_coderabbit_row():
    rows = reviewer_states(HEAD, [_cr(_FAIR_USAGE)], [], [_check("completed", "success")])
    cr = next(r for r in rows if r["backend"] == "coderabbit")
    assert cr["state"] == "rate_limited"
    assert cr["reviewed_head_with_content"] is False


@pytest.mark.parametrize(
    "notice,label",
    [(_FAIR_USAGE, "rate_limited"), ("Rate limit exceeded", "rate_limited")],
)
def test_stale_review_body_throttle_does_not_resurrect_on_an_unscanned_head(notice, label):
    """The unscanned-HEAD path was sourcing throttles from review bodies too.

    A submitted review body is never rewritten, so a notice inside one is
    permanently stale — a PR throttled once would keep reporting `rate_limited`
    on every later HEAD with no walkthrough. Only CR's live status comments count.
    """
    stale = _cr_review("0000aaa", body=notice)
    s = coderabbit_state(HEAD, [], [stale])
    assert s["state"] == "pending"
    assert s["state"] != label


def test_stale_review_body_pause_does_not_resurrect_on_an_unscanned_head():
    stale = _cr_review("0000aaa", body="Reviews paused by coderabbit.ai")
    assert coderabbit_state(HEAD, [], [stale])["state"] == "pending"


def test_a_live_throttle_notice_still_reports_on_an_unscanned_head():
    """The scoping must not make the unscanned path blind to a real throttle."""
    s = coderabbit_state(HEAD, [_cr(_FAIR_USAGE)], [])
    assert s["state"] == "rate_limited"


def test_coderabbit_throttle_notice_in_a_stale_review_body_does_not_throttle_head():
    """Scoped to CR's live issue comments, matching the in-progress rule."""
    stale = _cr_review("0000aaa", body=_FAIR_USAGE)
    assert coderabbit_state(HEAD, [_walk(HEAD)], [stale])["state"] == "pending"


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


def test_copilot_quota_notice_in_review_body_is_unavailable():
    reviews = [
        {
            "user": {"login": "Copilot"},
            "commit_id": "old111",
            "state": "COMMENTED",
            "body": "Copilot wasn't able to review this pull request because of a quota limit.",
        }
    ]
    s = copilot_state(HEAD, reviews)
    assert s["state"] == "unavailable"
    assert s["head_reviewed"] == "old111"


def test_copilot_quota_notice_in_issue_comment_is_unavailable():
    comments = [
        {
            "user": {"login": "copilot-pull-request-reviewer[bot]"},
            "body": "You have exceeded your monthly limit of premium requests.",
        }
    ]
    s = copilot_state(HEAD, [], comments)
    assert s["state"] == "unavailable"
    assert s["head_reviewed"] is None


def test_copilot_review_on_head_beats_stale_quota_notice():
    reviews = [
        {
            "user": {"login": "Copilot"},
            "commit_id": "old111",
            "body": "quota exceeded",
            "state": "COMMENTED",
        },
        {"user": {"login": "Copilot"}, "commit_id": HEAD, "state": "COMMENTED"},
    ]
    assert copilot_state(HEAD, reviews)["state"] == "done"


def test_copilot_incidental_quota_words_stay_pending():
    reviews = [
        {
            "user": {"login": "Copilot"},
            "commit_id": "old111",
            "state": "COMMENTED",
            "body": (
                "This change adjusts the API quota handling and is unable to review binary files."
            ),
        }
    ]
    assert copilot_state(HEAD, reviews)["state"] == "pending"


def test_copilot_non_quota_bodies_stay_pending():
    reviews = [
        {
            "user": {"login": "Copilot"},
            "commit_id": "old111",
            "state": "COMMENTED",
            "body": "Here are my review comments on the changes.",
        }
    ]
    assert copilot_state(HEAD, reviews)["state"] == "pending"


# --- fuko's own instances (run receipts) -------------------------------------
#
# The gap these close: an instance that reviewed HEAD and found nothing looks
# identical to one that never started — both produce zero signals. Every case
# below asserts that the two are now distinguishable, and that the ambiguous
# ones resolve AWAY from "done" (which would merge unreviewed code).


def _receipt_comment(**kw):
    """An instance's header comment carrying one run receipt.

    ``model`` defaults to ``label`` because that is what a real finalized receipt
    carries: `_finalize_branch_header` records `result.provider or label`, so a
    healthy primary run always attributes itself. A receipt whose `model` is
    empty or differs from `label` is a genuine anomaly and every test that wants
    one says so explicitly.
    """
    fields = {"label": "openrouter/x-ai/grok-4.5", "head_sha": HEAD, "state": "done"}
    fields.update(kw)
    fields.setdefault("model", fields["label"] if fields["state"] == "done" else "")
    return {
        "user": {"login": "fuko-sybil[bot]"},
        "body": with_run_receipt("🤖 **fuko A/B** — model `x`", RunReceipt(**fields)),
    }


def test_fuko_instance_reports_done_on_head():
    rows = fuko_states(HEAD, [_receipt_comment()])
    assert [r["backend"] for r in rows] == ["fuko:openrouter/x-ai/grok-4.5"]
    assert rows[0]["state"] == "done"
    assert rows[0]["head_reviewed"] == HEAD
    assert rows[0]["role"] == "active"


def test_fuko_instance_that_never_ran_is_absent_not_done():
    """No receipt at all must not read as a clean pass.

    This is the whole point: silence from an instance that never started is not
    evidence of a clean review, so it yields no row rather than a `done` one.
    """
    assert fuko_states(HEAD, [{"user": {"login": "someone"}, "body": "hi"}]) == []


def test_fuko_instance_still_running_is_in_progress():
    rows = fuko_states(HEAD, [_receipt_comment(state="in_progress")])
    assert rows[0]["state"] == "in_progress"


def test_fuko_branch_that_died_mid_run_stays_in_progress_not_done():
    """A branch killed before finalizing leaves an `in_progress` receipt.

    It must not decay to `done`; the consumer's own timeout governs instead.
    """
    rows = fuko_states(HEAD, [_receipt_comment(state="in_progress")])
    assert rows[0]["state"] != "done"


def test_fuko_failed_branch_is_degraded_and_escalates():
    rows = fuko_states(HEAD, [_receipt_comment(state="failed", detail="all providers throttled")])
    assert rows[0]["state"] == "unavailable"
    assert rows[0]["state"] in DEGRADED_STATES
    assert escalation_needed(rows) is True
    assert "throttled" in rows[0]["detail"]


def test_fuko_receipt_for_an_older_head_is_pending():
    rows = fuko_states(HEAD, [_receipt_comment(head_sha="0" * 40)])
    assert rows[0]["state"] == "pending"


def test_fuko_receipt_without_a_head_never_reads_as_done():
    """An un-anchored receipt (HEAD unresolvable at run time) withholds, not grants."""
    rows = fuko_states(HEAD, [_receipt_comment(head_sha="")])
    assert rows[0]["state"] == "pending"


def test_fuko_trial_role_is_reported_so_a_consumer_can_skip_gating():
    rows = fuko_states(HEAD, [_receipt_comment(role="trial")])
    assert rows[0]["role"] == "trial"
    assert rows[0]["state"] == "done"


def test_fuko_substituted_model_is_void_not_an_annotation():
    """#106: a seat that did not review as the model it names has NOT covered HEAD.

    Previously reported `done` with an attribution note in `detail`. A substituted
    seat's output is byte-indistinguishable from a genuine clean review, so a
    consumer gating on `state == "done"` merged on a review that never validly
    happened. The round wants re-running, not recording.
    """
    rows = fuko_states(HEAD, [_receipt_comment(model="ollama-cloud/glm-5.2:cloud")])
    assert rows[0]["state"] == "degraded"
    assert rows[0]["valid"] is False
    assert "VOID" in rows[0]["detail"]
    # BOTH fields are printed, so the comparison is not left to the consumer.
    assert rows[0]["label"] == "openrouter/x-ai/grok-4.5"
    assert rows[0]["model"] == "ollama-cloud/glm-5.2:cloud"


def test_fuko_empty_model_is_void_because_the_test_is_a_conjunction():
    """The formulation matters: an empty model passes "no mismatch detected".

    There is nothing to mismatch against, so a negated test would call this
    valid. The conjunction requires `model` to be non-empty affirmatively.
    """
    rows = fuko_states(HEAD, [_receipt_comment(model="")])
    assert rows[0]["state"] == "degraded"
    assert rows[0]["valid"] is False


def test_fuko_valid_seat_reports_valid_true():
    rows = fuko_states(HEAD, [_receipt_comment()])
    assert rows[0]["valid"] is True
    assert rows[0]["label"] == rows[0]["model"]


def test_fuko_stale_receipt_is_never_valid():
    """`valid` must not outlive the commit it describes."""
    rows = fuko_states(HEAD, [_receipt_comment(head_sha="0" * 40)])
    assert rows[0]["state"] == "pending"
    assert rows[0]["valid"] is False


def test_fuko_dead_channel_seat_is_not_valid_either():
    """Two distinct faults, still machine-distinguishable.

    Both a substitution and a dead channel report `degraded`, which is right —
    both are lost coverage — but a consumer must be able to tell them apart, so
    the substitution names VOID and the channel case carries the `channels` map.
    """
    rows = fuko_states(
        HEAD, [_receipt_comment(channels={"review": "done", "improve": "killed:timeout"})]
    )
    assert rows[0]["state"] == "degraded"
    assert rows[0]["valid"] is False
    assert "VOID" not in rows[0]["detail"]
    assert rows[0]["channels"]["improve"] == "killed:timeout"


def test_fuko_promoted_backup_is_marked_as_promoted():
    """#106: `slot: null` beside `role: active` was unattributable on its own."""
    rows = fuko_states(HEAD, [_receipt_comment(promoted=True, slot=None)])
    assert rows[0]["promoted"] is True


def test_fuko_dead_channel_is_degraded_not_done():
    """The #108 defect: a seat with a working guide and a dead suggestions channel.

    `improve` is optional, so its container being killed leaves the branch's
    return code at zero and the receipt at `state: done`. A seat that published
    only half its channels is NOT a clean pass, and must not read as one.
    """
    rows = fuko_states(
        HEAD, [_receipt_comment(channels={"review": "done", "improve": "killed:timeout"})]
    )
    assert rows[0]["state"] == "degraded"
    assert rows[0]["state"] != "done"
    assert "improve killed:timeout" in rows[0]["detail"]


def test_fuko_dead_channel_escalates_like_any_lost_coverage():
    rows = fuko_states(
        HEAD, [_receipt_comment(channels={"review": "done", "improve": "killed:timeout"})]
    )
    assert rows[0]["state"] in DEGRADED_STATES
    assert escalation_needed(rows) is True


def test_fuko_all_channels_done_is_still_done():
    """The map must not make a genuinely healthy seat look degraded."""
    rows = fuko_states(HEAD, [_receipt_comment(channels={"review": "done", "improve": "done"})])
    assert rows[0]["state"] == "done"
    assert rows[0]["channels"] == {"review": "done", "improve": "done"}


def test_fuko_receipt_without_channels_keeps_its_old_meaning():
    """A receipt written before channel reporting has nothing to judge.

    Empty must mean "not reported", never "every channel was healthy" — but it
    also must not retroactively degrade every pre-upgrade receipt.
    """
    rows = fuko_states(HEAD, [_receipt_comment()])
    assert rows[0]["state"] == "done"
    assert "channels" not in rows[0]


def test_fuko_in_flight_receipt_has_no_channels_and_is_not_degraded():
    """An empty map does not identify "old receipt" — an in-flight one is empty too.

    `_post_branch_header` writes the opening receipt before any tool has run, so
    it carries no channels. That must report `in_progress` on its own terms, not
    be mistaken for either a healthy seat or a degraded one.
    """
    rows = fuko_states(HEAD, [_receipt_comment(state="in_progress")])
    assert rows[0]["state"] == "in_progress"
    assert "channels" not in rows[0]


def test_fuko_channel_that_never_ran_is_degraded():
    """`skipped` is a dead channel too — an unreached tool produced nothing."""
    rows = fuko_states(HEAD, [_receipt_comment(channels={"review": "done", "improve": "skipped"})])
    assert rows[0]["state"] == "degraded"


def test_fuko_dead_channel_on_an_older_head_is_still_pending():
    """Staleness outranks the channel map, as it already outranks the outcome."""
    rows = fuko_states(
        HEAD,
        [_receipt_comment(head_sha="0" * 40, channels={"review": "done", "improve": "skipped"})],
    )
    assert rows[0]["state"] == "pending"


def test_fuko_row_surfaces_review_backend():
    """#99: the row names the driver under `review_backend`, leaving `backend`
    (the display id) untouched so the sort key and consumers are unaffected."""
    rows = fuko_states(HEAD, [_receipt_comment(backend="agentic")])
    assert rows[0]["review_backend"] == "agentic"
    assert rows[0]["backend"] == "fuko:openrouter/x-ai/grok-4.5"


def test_fuko_removed_seat_is_superseded_not_a_stuck_pending():
    """#116: a receipt whose seat was renamed/removed must not gate forever.

    Its receipt is anchored to a HEAD that can never recur, so without the
    cross-reference it would report `pending` on every later round — a gate that
    can never open. Given the current seat labels, the absent one downgrades to
    `superseded`, the one deliberately-non-gating state.
    """
    rows = fuko_states(
        HEAD,
        [_receipt_comment(label="openrouter/z-ai/glm-5.2", model="openrouter/z-ai/glm-5.2")],
        configured_labels=["openrouter/qwen/qwen3.8-max"],
    )
    assert rows[0]["state"] == "superseded"
    assert rows[0]["valid"] is False


def test_fuko_superseded_does_not_gate_or_escalate():
    """`superseded` is neither `pending` (waited on) nor a DEGRADED state (escalated)."""
    rows = fuko_states(
        HEAD,
        [_receipt_comment(label="openrouter/z-ai/glm-5.2", model="openrouter/z-ai/glm-5.2")],
        configured_labels=["openrouter/qwen/qwen3.8-max"],
    )
    assert rows[0]["state"] not in DEGRADED_STATES
    assert escalation_needed(rows) is False


def test_fuko_still_configured_seat_that_has_not_run_stays_pending():
    """A genuinely not-yet-reviewed seat is untouched by the cross-reference.

    The receipt is on an older HEAD but its label IS still configured, so it must
    keep gating as `pending` rather than being mistaken for a removed seat.
    """
    rows = fuko_states(
        HEAD,
        [_receipt_comment(head_sha="0" * 40)],
        configured_labels=["openrouter/x-ai/grok-4.5"],
    )
    assert rows[0]["state"] == "pending"


def test_fuko_absent_configured_labels_keeps_todays_behavior():
    """The fail-safe guard: with no config to cross-reference, no row is dropped.

    A config-read failure passes `configured_labels=None`; the removed seat then
    still reports its stale `pending` rather than being silently reclassified —
    erring toward the gating state, never past an unreviewed seat.
    """
    rows = fuko_states(
        HEAD,
        [_receipt_comment(label="openrouter/z-ai/glm-5.2", model="openrouter/z-ai/glm-5.2")],
    )
    assert rows[0]["state"] != "superseded"


def test_fuko_empty_configured_labels_keeps_pending_not_superseded():
    """#116 follow-up: an EMPTY configured-label collection must not supersede all.

    An empty set would make every receipt's label 'not configured' and drop every
    gating `pending` row -- the unsafe direction. `fuko_states` normalizes an empty
    (or blank-only) collection to None, so a stale receipt stays `pending`, exactly
    as it would when config is unreadable. Flagged by CodeRabbit + Copilot, and by
    the glm-5.2 reviewer guide.
    """
    rows = fuko_states(
        HEAD,
        [
            _receipt_comment(
                head_sha="0" * 40, label="openrouter/z-ai/glm-5.2", model="openrouter/z-ai/glm-5.2"
            )
        ],
        configured_labels=[],
    )
    assert rows[0]["state"] == "pending"
    rows_blank = fuko_states(
        HEAD,
        [
            _receipt_comment(
                head_sha="0" * 40, label="openrouter/z-ai/glm-5.2", model="openrouter/z-ai/glm-5.2"
            )
        ],
        configured_labels=["  ", ""],
    )
    assert rows_blank[0]["state"] == "pending"


def test_fuko_configured_seat_still_reports_done():
    """A seat whose label is still configured reviews and reports `done` as before."""
    rows = fuko_states(
        HEAD,
        [_receipt_comment()],
        configured_labels=["openrouter/x-ai/grok-4.5"],
    )
    assert rows[0]["state"] == "done"


def test_fuko_receipt_from_a_non_fuko_author_is_not_coverage():
    """The #105 spoof: a well-formed receipt posted by anyone who can comment.

    A receipt asserts that a reviewer RAN. If any commenter can assert that, a
    consumer stops waiting on an instance that never started — so coverage would
    be assertable by a party that never reviewed, including the PR author on a
    public repo.
    """
    spoof = dict(_receipt_comment(), user={"login": "random-contributor"})
    assert fuko_states(HEAD, [spoof]) == []


@pytest.mark.parametrize(
    "login",
    [
        "fukoo-imposter",  # merely CONTAINS "fuko" — a substring match would admit it
        "fuko-dorian",  # right prefix, not a bot account
        "not-fuko-dorian[bot]",  # right shape, wrong start — must be anchored
        "fuko-dorian[bot]-evil",  # right start, trailing junk — must be anchored
        "github-actions",  # fallback identity without the [bot] suffix
    ],
)
def test_fuko_receipt_from_a_lookalike_author_is_not_coverage(login):
    """Scoping is on the author, not on the receipt looking plausible.

    The loose `fuko` substring used for finding triage is unsafe here: an extra
    triaged finding is harmless, forged coverage is not. `fukoo-imposter` is the
    case that motivated anchoring — GitHub user names cannot contain `[` or `]`,
    so requiring the `[bot]` suffix puts every match out of a human's reach.
    """
    spoof = dict(_receipt_comment(label="zai-coding/glm-5.2"), user={"login": login})
    assert fuko_states(HEAD, [spoof]) == []


def test_fuko_receipt_without_an_author_is_refused():
    """No identity to check means no coverage — the unsafe direction is accepting."""
    assert fuko_states(HEAD, [{"user": {}, "body": _receipt_comment()["body"]}]) == []
    assert fuko_states(HEAD, [{"body": _receipt_comment()["body"]}]) == []


def test_fuko_receipt_from_any_fuko_app_instance_is_accepted():
    """A new slot or an App rename must not silently drop that instance's coverage."""
    for login in ("fuko-dorian[bot]", "fuko-gray[bot]", "fuko-basil[bot]"):
        rows = fuko_states(HEAD, [dict(_receipt_comment(), user={"login": login})])
        assert rows and rows[0]["state"] == "done", login


def test_fuko_receipt_from_the_app_less_fallback_identity_is_accepted():
    """With no App configured the workflow posts as github-actions[bot]."""
    rows = fuko_states(HEAD, [dict(_receipt_comment(), user={"login": "github-actions[bot]"})])
    assert rows[0]["state"] == "done"


def test_fuko_allowed_authors_overrides_the_default_pattern():
    comment = dict(_receipt_comment(), user={"login": "my-own-reviewer[bot]"})
    assert fuko_states(HEAD, [comment]) == []
    rows = fuko_states(HEAD, [comment], allowed_authors=["My-Own-Reviewer[bot]"])
    assert rows[0]["state"] == "done"


def test_fuko_allowed_authors_excludes_the_default_fuko_apps():
    """An explicit allowlist REPLACES the default; it does not widen it."""
    comment = dict(_receipt_comment(), user={"login": "fuko-dorian[bot]"})
    assert fuko_states(HEAD, [comment], allowed_authors=["other[bot]"]) == []


def test_reviewer_states_forwards_allowed_authors():
    comment = dict(_receipt_comment(), user={"login": "my-own-reviewer[bot]"})
    rows = reviewer_states(HEAD, [comment], [], allowed_authors=["my-own-reviewer[bot]"])
    assert any(r["backend"].startswith("fuko:") for r in rows)


def test_fuko_newest_receipt_per_instance_wins():
    """A re-run leaves an older receipt behind; later comments are later runs."""
    stale = _receipt_comment(head_sha="0" * 40, state="done")
    fresh = _receipt_comment(head_sha=HEAD, state="done")
    rows = fuko_states(HEAD, [stale, fresh])
    assert len(rows) == 1
    assert rows[0]["state"] == "done"


def test_fuko_instances_are_separate_rows():
    rows = fuko_states(
        HEAD,
        [_receipt_comment(label="a/one"), _receipt_comment(label="b/two", role="trial")],
    )
    assert [r["backend"] for r in rows] == ["fuko:a/one", "fuko:b/two"]


def test_malformed_receipt_is_skipped_not_fatal():
    bad = {"user": {"login": "x"}, "body": "<!-- fuko-run:v1 {not json} -->"}
    assert fuko_states(HEAD, [bad, _receipt_comment()]) != []


def test_reviewer_states_includes_fuko_rows_by_default():
    rows = reviewer_states(HEAD, [_receipt_comment()], [])
    assert [r["backend"] for r in rows] == [
        "coderabbit",
        "copilot",
        "fuko:openrouter/x-ai/grok-4.5",
    ]


def test_reviewer_states_can_exclude_fuko_to_avoid_self_escalation():
    """The runner's health probe must not let fuko escalate in response to itself."""
    rows = reviewer_states(HEAD, [_receipt_comment(state="failed")], [], include_fuko=False)
    assert all(not r["backend"].startswith("fuko:") for r in rows)
    assert escalation_needed(rows) is False


def test_stale_failed_receipt_reports_pending_not_unavailable():
    """A failure on an OLD commit says nothing about HEAD.

    Reporting it `unavailable` would keep firing `escalation_needed` every later
    round until that instance happened to succeed — a failure that sticks long
    after the push that outdated it.
    """
    rows = fuko_states(HEAD, [_receipt_comment(state="failed", head_sha="0" * 40)])
    assert rows[0]["state"] == "pending"
    assert escalation_needed(rows) is False


def test_failure_on_the_current_head_still_escalates():
    rows = fuko_states(HEAD, [_receipt_comment(state="failed", head_sha=HEAD)])
    assert rows[0]["state"] == "unavailable"
    assert escalation_needed(rows) is True
