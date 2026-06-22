"""Tests for review-pool config validation."""

import pytest
from pydantic import ValidationError

from sidecar.fukoconfig import ModelConfig, ReviewConfig


def test_non_positive_max_model_tokens_is_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(provider="zai-coding", name="glm-5.2", max_model_tokens=0)


def test_positive_max_model_tokens_accepted():
    m = ModelConfig(provider="zai-coding", name="glm-5.2", max_model_tokens=256000)
    assert m.max_model_tokens == 256000


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
