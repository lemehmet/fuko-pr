"""Unit tests for the context-size estimator."""

from sidecar.sizing import estimate_tokens, required_context


def test_estimate_tokens_is_chars_over_four():
    assert estimate_tokens(0) == 0
    assert estimate_tokens(400) == 100


def test_required_context_includes_reserve():
    assert required_context(0, 0, reserve_tokens=8000) == 8000
    assert required_context(400, 400, reserve_tokens=1000) == 1200


def test_required_context_scales_with_diff_and_knowledge():
    small = required_context(1_000, 0)
    large = required_context(1_000_000, 0)
    assert large > small
