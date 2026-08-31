"""Tests for the A/B estimators (#159).

These are arithmetic with a published baseline riding on them, so the cases are
worked by hand rather than round-tripped: an estimator that agrees with itself
is exactly the failure mode ad-hoc measurement scripts have.
"""

from sidecar.abmetrics import (
    Claim,
    arm_metrics,
    chapman_pool_size,
    claim_title,
    backup_served_rounds,
    collect_claims,
    pair_metrics,
)
from sidecar.signals import RunReceipt, with_run_receipt


def _c(arm, round_key, file, title, scope=""):
    return Claim(arm=arm, round_key=round_key, file=file, title=title, scope=scope)


def test_claim_identity_is_the_ledgers_own_rule():
    assert (
        _c("a", "r1", " src/app.py ", "Unchecked None").anchor
        == _c("b", "r2", "src/app.py", "unchecked none").anchor
    )


def test_arm_metrics_counts_re_reports_not_repeats_within_a_round():
    claims = [
        _c("a", "r1", "src/app.py", "leak"),
        _c("a", "r1", "src/app.py", "leak"),
        _c("a", "r2", "src/app.py", "leak"),
        _c("a", "r2", "src/util.py", "race"),
        _c("b", "r1", "src/other.py", "ignored"),
    ]
    m = arm_metrics("a", claims)

    assert m.rounds == 2 and m.findings == 4
    assert m.distinct_claims == 2 and m.distinct_paths == 2
    # "leak" appears in two rounds; "race" in one.
    assert m.re_reported == 1 and m.one_shot_rate == 0.5
    # Touches, not distinct paths: src/app.py was touched three times of four.
    assert m.top_paths_share == 1.0
    assert m.top_paths[0] == ("src/app.py", 3)


def test_an_arm_with_no_claims_reports_none_not_zero():
    m = arm_metrics("a", [_c("b", "r1", "f", "t")])

    assert m.findings == 0
    assert m.one_shot_rate is None and m.top_paths_share is None


def test_chapman_is_defined_at_zero_overlap():
    # (3+1)(3+1)/(0+1) - 1
    assert chapman_pool_size(3, 3, 0) == 15
    # (2+1)(2+1)/(2+1) - 1 -- perfect agreement implies the pool is what was seen.
    assert chapman_pool_size(2, 2, 2) == 2
    assert chapman_pool_size(0, 5, 0) is None


def test_pair_metrics_only_scores_rounds_both_arms_reviewed():
    claims = [
        _c("a", "r1", "src/app.py", "leak"),
        _c("b", "r1", "src/app.py", "leak"),
        _c("b", "r1", "src/util.py", "race"),
        # r2: only the control published, so it carries no agreement information.
        _c("a", "r2", "src/solo.py", "alone"),
    ]
    p = pair_metrics(claims, "a", "b")

    assert p.rounds == 1
    assert (p.a_claims, p.b_claims, p.shared, p.union) == (1, 2, 1, 2)
    assert p.agreement == 0.5
    # (1+1)(2+1)/(1+1) - 1 = 2, and the union of 2 covers all of it.
    assert p.pool_estimate == 2 and p.coverage == 1.0


def test_pair_metrics_are_none_when_the_arms_never_overlapped():
    p = pair_metrics([_c("a", "r1", "f", "t")], "a", "b")

    assert p.rounds == 0 and p.union == 0
    assert p.agreement is None and p.pool_estimate is None and p.coverage is None


def test_claim_title_prefers_the_stored_title_and_unwraps_its_rendering():
    assert claim_title("Unchecked None", "body") == "Unchecked None"
    assert claim_title("**Unchecked None**", "body") == "Unchecked None"


def test_claim_title_falls_back_past_the_shared_model_label():
    """The label is identical across arms by design, so keying on it fakes agreement."""
    body = "\U0001f916 `qwen-anthropic/qwen3.8-max`\n\n**Unchecked None**\n\ndetail"

    assert claim_title("", body) == "Unchecked None"


def test_claim_title_is_empty_when_only_decoration_survives():
    """`****` is a rendered EMPTY title; treating it as content collapses a file's claims."""
    assert claim_title("****", "\U0001f916 `qwen-anthropic/qwen3.8-max`") == ""
    assert claim_title("", "") == ""
    assert claim_title("  ", "\n  \n**  **\n") == ""


def test_two_untitled_claims_on_one_file_do_not_become_one():
    """The guarantee the empty return buys: callers must drop, not anchor on ``""``."""
    assert claim_title("****", "") == claim_title("", "")


def test_receipts_score_the_round_one_arm_reviewed_and_found_nothing_in():
    """The maximum-disagreement round a claim-derived key silently drops."""
    claims = [_c("a", "r1", "src/app.py", "leak"), _c("a", "r2", "src/util.py", "race")]
    # Both arms reviewed both heads; b published on neither.
    receipts = [("a", "r1"), ("a", "r2"), ("b", "r1"), ("b", "r2")]

    scored = pair_metrics(claims, "a", "b", receipts)
    dropped = pair_metrics(claims, "a", "b")

    assert (scored.rounds, scored.a_claims, scored.b_claims) == (2, 2, 0)
    assert scored.shared == 0 and scored.union == 2 and scored.agreement == 0.0
    # Without receipts the same data reports NO shared rounds at all, so the
    # disagreement never enters the pooled ratio.
    assert dropped.rounds == 0 and dropped.agreement is None


def test_receipts_still_exclude_a_round_an_arm_never_reviewed():
    """An absent arm is not a disagreeing one -- that confound stays out."""
    claims = [_c("a", "r1", "src/app.py", "leak"), _c("a", "r2", "src/util.py", "race")]
    receipts = [("a", "r1"), ("a", "r2"), ("b", "r1")]

    p = pair_metrics(claims, "a", "b", receipts)

    assert p.rounds == 1 and p.a_claims == 1 and p.union == 1


def test_arm_rounds_count_heads_reviewed_when_receipts_are_supplied():
    claims = [_c("a", "r1", "src/app.py", "leak")]

    assert arm_metrics("a", claims).rounds == 1
    assert arm_metrics("a", claims, [("a", "r1"), ("a", "r2")]).rounds == 2


ARMS = {"control": "fuko-dorian[bot]", "treatment": "fuko-gray[bot]"}


def _comment(login, url, title, path="src/app.py", commit="head9", original="head1"):
    """An inline review comment shaped the way the agentic backend posts one."""
    return {
        "html_url": url,
        "user": {"login": login},
        "commit_id": commit,
        "original_commit_id": original,
        "path": path,
        "body": (
            f"\U0001f916 `qwen-anthropic/qwen3.8-max`\n\n**{title}**\n\ndetail\n\n"
            '<!-- fuko-signal:v1 {"v":1,"id":"fk_1","file":"' + path + '","line":4,'
            '"severity":"high","severity_source":"declared","category":"bug",'
            '"suggestion":false,"suppressed":false,"thread_url":null,'
            '"backend":"agentic","model":"m","role":"active","kb_refs":[]} -->'
        ),
    }


def test_collect_claims_joins_each_signal_to_its_author_and_head():
    claims, receipts, untitled = collect_claims(
        [_comment("fuko-dorian[bot]", "u1", "unchecked None")], [], ARMS, 7
    )

    assert untitled == 0
    assert [(c.arm, c.round_key, c.file, c.title) for c in claims] == [
        ("control", "7@head1", "src/app.py", "unchecked None")
    ]
    assert receipts == {("control", "7@head1")}


def test_collect_claims_keys_a_round_on_the_head_the_comment_was_created_against():
    """`commit_id` is rewritten when GitHub re-anchors an outdated thread."""
    claims, _, _ = collect_claims(
        [
            _comment("fuko-dorian[bot]", "u1", "one", commit="head9", original="head1"),
            _comment("fuko-dorian[bot]", "u2", "two", commit="head9", original="head2"),
        ],
        [],
        ARMS,
        7,
    )

    # Two heads stay two rounds; keying on `commit_id` would merge them into one.
    assert {c.round_key for c in claims} == {"7@head1", "7@head2"}


def test_collect_claims_skips_authors_outside_the_named_arms():
    claims, receipts, _ = collect_claims(
        [_comment("coderabbitai[bot]", "u1", "not in this experiment")], [], ARMS, 7
    )

    assert claims == [] and receipts == set()


def test_collect_claims_emits_a_receipt_for_a_review_that_published_nothing():
    reviews = [{"html_url": "r1", "user": {"login": "fuko-gray[bot]"}, "commit_id": "head1"}]

    claims, receipts, _ = collect_claims([], reviews, ARMS, 7)

    assert claims == [] and receipts == {("treatment", "7@head1")}


def test_collect_claims_drops_a_body_carried_finding_that_carries_no_title():
    """Marker-only signals arrive titleless (#142); they must not anchor on ``""``."""
    reviews = [
        {
            "html_url": "r1",
            "user": {"login": "fuko-gray[bot]"},
            "commit_id": "head1",
            "body": (
                '<!-- fuko-signal:v1 {"v":1,"id":"fk_2","file":"src/x.py","line":1,'
                '"severity":"high","severity_source":"declared","category":"bug",'
                '"suggestion":false,"suppressed":false,"thread_url":null,'
                '"backend":"agentic","model":"m","role":"active","kb_refs":[]} -->'
            ),
        }
    ]

    claims, receipts, untitled = collect_claims([], reviews, ARMS, 7)

    assert claims == [] and untitled == 1
    # The round still counts: the arm reviewed that head.
    assert receipts == {("treatment", "7@head1")}


def test_collect_claims_drops_a_signal_whose_author_is_not_a_named_arm():
    """The author filter, not the join -- CodeRabbit reviews the same PRs."""
    claims, _, _ = collect_claims(
        [_comment("fuko-dorian[bot]", "u1", "orphan")], [], {"control": "someone-else[bot]"}, 7
    )

    assert claims == []


def test_collect_claims_drops_a_signal_whose_thread_matches_no_fetched_receipt():
    """The join-miss branch, exercised with the author deliberately IN the arms map.

    A marker that carries its own ``thread_url`` keeps it -- the review stamp only
    fills an empty one -- so a posting-side change that started setting the field
    would route signals past the join and shrink an arm's claim set with nothing
    counting the loss. That is the silent undercount this tool exists to prevent,
    so it gets a test even though no current path produces it.
    """
    marker = (
        '<!-- fuko-signal:v1 {"v":1,"id":"fk_9","file":"src/z.py","line":1,'
        '"severity":"high","severity_source":"declared","category":"bug",'
        '"suggestion":false,"suppressed":false,'
        '"thread_url":"https://elsewhere/#discussion_r1",'
        '"backend":"agentic","model":"m","role":"active","kb_refs":[]} -->'
    )
    reviews = [
        {
            "html_url": "r1",
            "user": {"login": "fuko-gray[bot]"},
            "commit_id": "head1",
            "body": f"**a real title**\n\n{marker}",
        }
    ]

    claims, receipts, untitled = collect_claims([], reviews, ARMS, 7)

    # The arm IS mapped and the finding IS titled, so only the join can drop it.
    assert claims == [] and untitled == 0
    assert receipts == {("treatment", "7@head1")}


def test_one_headline_recurring_on_two_prs_is_not_a_re_report():
    """`claim_anchor` carries no PR; a multi-PR window must not merge across them."""
    claims = [
        _c("a", "7@h1", "src/app.py", "leak", scope="7"),
        _c("a", "8@h1", "src/app.py", "leak", scope="8"),
    ]

    m = arm_metrics("a", claims)

    assert m.re_reported == 0 and m.distinct_claims == 2 and m.one_shot_rate == 1.0


def test_the_same_claim_in_two_rounds_of_one_pr_is_a_re_report():
    claims = [
        _c("a", "7@h1", "src/app.py", "leak", scope="7"),
        _c("a", "7@h2", "src/app.py", "leak", scope="7"),
    ]

    m = arm_metrics("a", claims)

    assert m.re_reported == 1 and m.distinct_claims == 1 and m.one_shot_rate == 0.0


def _header(login, label, model, head="head1", state="done"):
    """A branch-header issue comment carrying one finalized run receipt."""
    receipt = RunReceipt(
        label=label, model=model, head_sha=head, state=state, app=login, slot="dorian"
    )
    return {"user": {"login": login}, "body": with_run_receipt("\U0001f916 **fuko A/B**", receipt)}


def test_a_round_the_primary_answered_is_not_backup_served():
    """`label == model` is the clean case, and the common one."""
    headers = [_header("fuko-dorian[bot]", "qwen/qwen3.8-max", "qwen/qwen3.8-max")]

    assert backup_served_rounds(headers, ARMS, 7) == set()


def test_a_rescued_round_is_named_under_the_arm_that_posted_it():
    """The seat is the poster; the model that answered is the backup's (#204)."""
    headers = [
        _header("fuko-gray[bot]", "qwen/qwen3.8-max", "zai/glm-5.3", head="head2"),
        _header("fuko-dorian[bot]", "qwen/qwen3.8-max", "qwen/qwen3.8-max"),
    ]

    assert backup_served_rounds(headers, ARMS, 7) == {("treatment", "7@head2")}


def test_an_unfinished_receipt_claims_no_rescue_either_way():
    """An in-progress receipt names no answering model; a dead branch is not evidence."""
    headers = [_header("fuko-dorian[bot]", "qwen/qwen3.8-max", "", state="in_progress")]

    assert backup_served_rounds(headers, ARMS, 7) == set()


def test_a_header_from_outside_the_named_arms_is_skipped():
    """Same rule the claim join follows: an unnamed author is not an arm."""
    headers = [_header("fuko-henry[bot]", "qwen/qwen3.8-max", "zai/glm-5.3")]

    assert backup_served_rounds(headers, ARMS, 7) == set()


def test_an_exhausted_pool_is_not_a_rescue():
    """`failed` means primary AND backups died: `model` is the last entry TRIED."""
    headers = [_header("fuko-gray[bot]", "qwen/qwen3.8-max", "zai/glm-5.3", state="failed")]

    assert backup_served_rounds(headers, ARMS, 7) == set()


def test_an_unknown_author_cannot_mint_a_round_from_the_receipt_body():
    """A quoted receipt is not a run: `app` stands in for a MISSING author, not a wrong one."""
    quoted = _header("fuko-gray[bot]", "qwen/qwen3.8-max", "zai/glm-5.3", head="head2")
    quoted["user"] = {"login": "lemehmet"}

    assert backup_served_rounds([quoted], ARMS, 7) == set()


def test_an_unanchored_receipt_names_no_round():
    """`_head_for_receipts` degrades to "": an unknown commit joins to no round."""
    headers = [_header("fuko-gray[bot]", "qwen/qwen3.8-max", "zai/glm-5.3", head="")]

    assert backup_served_rounds(headers, ARMS, 7) == set()


def test_a_receipt_read_without_its_envelope_still_names_its_arm():
    """The `app` fallback the docstring promises: no author field at all."""
    payload = _header("fuko-gray[bot]", "qwen/qwen3.8-max", "zai/glm-5.3", head="head2")
    del payload["user"]

    assert backup_served_rounds([payload], ARMS, 7) == {("treatment", "7@head2")}
