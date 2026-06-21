"""Unit tests for the throttle classifier."""

import pytest

from sidecar.throttle import is_throttle


@pytest.mark.parametrize(
    "output",
    [
        "litellm.RateLimitError: 429 Too Many Requests",
        "Error code: 429",
        "anthropic: Overloaded (529)",
        "You exceeded your current quota",
        "RESOURCE_EXHAUSTED",
        "rate limit reached for requests",
        "insufficient_quota",
    ],
)
def test_throttle_signatures_classified(output):
    assert is_throttle(1, output) is True


def test_timeout_returncode_is_throttle_regardless_of_output():
    assert is_throttle(124, "") is True


def test_non_throttle_error_is_not_throttle():
    assert is_throttle(1, "Traceback: KeyError 'config'") is False


def test_empty_output_non_timeout_is_not_throttle():
    assert is_throttle(1, "") is False
