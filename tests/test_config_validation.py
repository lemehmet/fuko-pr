"""Tests for review-pool config validation."""

import pytest
from pydantic import ValidationError

from sidecar.fukoconfig import ReviewConfig


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValidationError):
        ReviewConfig(strategy="round-robin")


def test_non_positive_cooldown_is_rejected():
    with pytest.raises(ValidationError):
        ReviewConfig(cooldown_seconds=0)


def test_failover_strategy_and_positive_cooldown_accepted():
    cfg = ReviewConfig(strategy="failover", cooldown_seconds=120)
    assert cfg.strategy == "failover"
    assert cfg.cooldown_seconds == 120
