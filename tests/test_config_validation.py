"""Tests for review-pool config validation."""

import pytest
from pydantic import ValidationError

from sidecar.fukoconfig import FukoConfig, ModelConfig, ReviewConfig, ReviewModel, load_config


def test_unknown_backend_on_review_is_rejected():
    """#99: a whole-config backend no driver is registered for fails at parse time."""
    with pytest.raises(ValidationError, match="unknown review backend"):
        ReviewConfig(backend="nope")


def test_unknown_backend_on_model_entry_is_rejected():
    """A per-entry backend is validated too, not just [review].backend."""
    with pytest.raises(ValidationError, match="unknown review backend"):
        ReviewConfig(models=[ReviewModel(provider="p", name="m", role="active", backend="nope")])


def test_empty_string_backend_on_model_entry_is_rejected():
    """An explicit `backend = ""` is a mistake, not an inherit request.

    Truthiness-based filtering would drop "" from validation and let
    `_backend_for`'s `entry.backend or review.backend` silently treat it as
    unset; validation must reject it instead.
    """
    with pytest.raises(ValidationError, match="unknown review backend"):
        ReviewConfig(models=[ReviewModel(provider="p", name="m", role="active", backend="")])


def test_known_backends_accepted():
    """Both registered drivers parse, whole-config and per-entry."""
    assert ReviewConfig(backend="agentic").backend == "agentic"
    cfg = ReviewConfig(
        models=[ReviewModel(provider="p", name="m", role="active", backend="pr-agent")]
    )
    assert cfg.models[0].backend == "pr-agent"


def test_model_backend_defaults_to_none():
    """An entry without an explicit backend inherits [review].backend (None)."""
    assert ReviewModel(provider="p", name="m").backend is None


def test_non_positive_max_model_tokens_is_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(provider="zai-coding", name="glm-5.2", max_model_tokens=0)


def test_positive_max_model_tokens_accepted():
    m = ModelConfig(provider="zai-coding", name="glm-5.2", max_model_tokens=256000)
    assert m.max_model_tokens == 256000


def test_non_positive_max_turns_is_rejected_at_both_levels():
    """#229: a turn cap of 0 or less is a config error, as the sibling knobs are."""
    with pytest.raises(ValidationError):
        ModelConfig(provider="anthropic", name="m", max_turns=0)
    with pytest.raises(ValidationError):
        ReviewConfig(max_turns=-1)


def test_max_turns_defaults_to_none_at_both_levels():
    """Unset at both levels means the harness default, not a number of our own."""
    assert ModelConfig(provider="anthropic", name="m").max_turns is None
    assert ReviewConfig().max_turns is None


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


def test_models_defaults_to_empty():
    assert ReviewConfig().models == []


def test_models_roles_parse_from_toml(tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "zai-coding"\n'
        'name = "glm-5.2"\n'
        'token_env = "FUKO_GITHUB_TOKEN_DORIAN"\n'
        "[[review.models]]\n"
        'provider = "openrouter"\n'
        'name = "deepseek/deepseek-v4-pro"\n'
        'role = "backup"\n',
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert [(m.provider, m.role) for m in loaded.review.models] == [
        ("zai-coding", "active"),
        ("openrouter", "backup"),
    ]
    assert loaded.review.models[0].token_env == "FUKO_GITHUB_TOKEN_DORIAN"
    assert loaded.review.models[1].token_env is None


def test_max_turns_parses_from_toml_at_both_levels(tmp_path):
    """#229: the knob is reachable from a real .fuko.toml, fleet-wide and per seat.

    Hand-built model objects would not prove the TOML surface exists, which is
    the only surface a downstream fleet has.
    """
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[review]\n"
        "max_turns = 250\n"
        "[[review.models]]\n"
        'provider = "anthropic"\n'
        'name = "claude-x"\n'
        "max_turns = 100\n"
        "[[review.models]]\n"
        'provider = "openrouter"\n'
        'name = "deepseek/deepseek-v4-pro"\n',
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.review.max_turns == 250
    assert [m.max_turns for m in loaded.review.models] == [100, None]


def test_models_unknown_role_is_rejected():
    with pytest.raises(ValidationError):
        ReviewModel(provider="ollama", name="x", role="standby")


def test_models_all_backup_is_rejected():
    with pytest.raises(ValidationError):
        ReviewConfig(models=[ReviewModel(provider="ollama", name="x", role="backup")])


def test_models_trial_role_parses(tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "zai-coding"\n'
        'name = "glm-5.2"\n'
        "[[review.models]]\n"
        'provider = "openrouter"\n'
        'name = "meta/muse-spark-1.1"\n'
        'role = "trial"\n',
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert [(m.provider, m.role) for m in loaded.review.models] == [
        ("zai-coding", "active"),
        ("openrouter", "trial"),
    ]


def test_models_trial_only_is_rejected():
    # A trial is non-gating, so a config with no active has nothing that gates —
    # the same "needs an active" rule that rejects all-backup rejects all-trial.
    with pytest.raises(ValidationError):
        ReviewConfig(models=[ReviewModel(provider="ollama", name="x", role="trial")])


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
