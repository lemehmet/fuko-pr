"""Tests for the A/B estimators (#159).

These are arithmetic with a published baseline riding on them, so the cases are
worked by hand rather than round-tripped: an estimator that agrees with itself
is exactly the failure mode ad-hoc measurement scripts have.
"""

from sidecar.abmetrics import Claim, arm_metrics, chapman_pool_size, claim_title, pair_metrics


def _c(arm, round_key, file, title):
    return Claim(arm=arm, round_key=round_key, file=file, title=title)


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
