"""Tests for review-pool config validation."""

import pytest
from pydantic import ValidationError

from sidecar.fukoconfig import FukoConfig, ModelConfig, ReviewConfig, load_config


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


def test_compare_defaults_to_empty():
    assert ReviewConfig().compare == []


def test_compare_entries_parse_with_token_env(tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.compare]]\n"
        'provider = "anthropic"\n'
        'name = "claude-sonnet-4-6"\n'
        "[[review.compare]]\n"
        'provider = "ollama"\n'
        'name = "qwen2.5-coder"\n'
        'token_env = "FUKO_GITHUB_TOKEN_B"\n',
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert isinstance(loaded, FukoConfig)
    assert [(c.provider, c.name) for c in loaded.review.compare] == [
        ("anthropic", "claude-sonnet-4-6"),
        ("ollama", "qwen2.5-coder"),
    ]
    assert loaded.review.compare[0].token_env is None
    assert loaded.review.compare[1].token_env == "FUKO_GITHUB_TOKEN_B"
