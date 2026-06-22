"""Unit tests for the ingress core: presets, config loading, and build_env."""

import pytest

from sidecar.backends import UnknownBackendError, get_backend
from sidecar.backends.pragent import PrAgentBackend
from sidecar.fukoconfig import FukoConfig, ModelConfig, load_config
from sidecar.presets import UnknownPresetError, get_preset


def test_get_preset_known():
    p = get_preset("zai-coding")
    assert p.litellm_prefix == "openai/"
    assert p.base_url == "https://api.z.ai/api/coding/paas/v4"
    assert p.key_env == "ZAI_KEY"
    assert p.quirks["custom_model_max_tokens"] == 1000000


def test_get_preset_ollama_cloud():
    p = get_preset("ollama-cloud")
    assert p.litellm_prefix == "openai/"
    assert p.base_url == "https://ollama.com/v1"
    assert p.key_env == "OLLAMA_API_KEY"
    assert p.quirks["custom_model_max_tokens"] == 976000


def test_build_env_ollama_cloud(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ol-secret")
    env = PrAgentBackend().build_env(
        get_preset("ollama-cloud"),
        ModelConfig(provider="ollama-cloud", name="glm-5.2:cloud"),
        knowledge="",
        tools=["review"],
    )
    assert env["CONFIG__MODEL"] == "openai/glm-5.2:cloud"
    assert env["OPENAI__API_BASE"] == "https://ollama.com/v1"
    assert env["OPENAI__KEY"] == "ol-secret"
    assert env["CONFIG__CUSTOM_MODEL_MAX_TOKENS"] == "976000"
    assert env["CONFIG__AI_TIMEOUT"] == "300"


def test_get_preset_unknown():
    with pytest.raises(UnknownPresetError) as e:
        get_preset("no-such-provider")
    assert "known presets" in str(e.value)


def test_load_config_defaults_when_missing(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    assert isinstance(cfg, FukoConfig)
    assert cfg.review.backend == "pr-agent"
    assert cfg.knowledge.store == "postgres"
    assert cfg.embedding.provider == "ollama"


def test_load_config_parses_toml(tmp_path):
    p = tmp_path / ".fuko.toml"
    p.write_text(
        "\n".join(
            [
                "[review]",
                'backend = "pr-agent"',
                'tools = ["review"]',
                "[review.model]",
                'provider = "zai-coding"',
                'name = "glm-5.2"',
                "[knowledge]",
                'store = "sqlite-vec"',
                "[knowledge.object_store]",
                'bucket = "my-kb"',
                'key = "owner/repo.db"',
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.review.model.provider == "zai-coding"
    assert cfg.review.model.name == "glm-5.2"
    assert cfg.review.tools == ["review"]
    assert cfg.knowledge.store == "sqlite-vec"
    assert cfg.knowledge.object_store.bucket == "my-kb"


def test_get_backend_known():
    assert isinstance(get_backend("pr-agent"), PrAgentBackend)


def test_get_backend_unknown():
    with pytest.raises(UnknownBackendError):
        get_backend("no-such-backend")


def test_build_env_zai_coding_matches_known_good(monkeypatch):
    monkeypatch.setenv("ZAI_KEY", "secret-123")
    env = PrAgentBackend().build_env(
        get_preset("zai-coding"),
        ModelConfig(provider="zai-coding", name="glm-5.2"),
        knowledge="- learn this",
        tools=["review", "improve"],
    )
    assert env["CONFIG__MODEL"] == "openai/glm-5.2"
    assert env["CONFIG__FALLBACK_MODELS"] == '["openai/glm-5.2"]'
    assert env["OPENAI__API_BASE"] == "https://api.z.ai/api/coding/paas/v4"
    assert env["OPENAI__KEY"] == "secret-123"
    assert env["CONFIG__CUSTOM_MODEL_MAX_TOKENS"] == "1000000"
    assert env["CONFIG__AI_TIMEOUT"] == "300"
    assert env["PR_REVIEWER__EXTRA_INSTRUCTIONS"] == "- learn this"
    assert env["PR_CODE_SUGGESTIONS__EXTRA_INSTRUCTIONS"] == "- learn this"
    assert env["github_action_config.auto_review"] == "true"
    assert env["github_action_config.auto_improve"] == "true"
    assert env["github_action_config.auto_describe"] == "false"


def test_build_env_ollama_no_key_no_quirks():
    env = PrAgentBackend().build_env(
        get_preset("ollama"),
        ModelConfig(provider="ollama", name="qwen2.5-coder:32b"),
        knowledge="",
        tools=["review"],
    )
    assert env["CONFIG__MODEL"] == "ollama/qwen2.5-coder:32b"
    assert env["OLLAMA__API_BASE"] == "http://localhost:11434"
    assert "OPENAI__KEY" not in env
    assert "CONFIG__CUSTOM_MODEL_MAX_TOKENS" not in env
    assert "CONFIG__AI_TIMEOUT" not in env
    assert "PR_REVIEWER__EXTRA_INSTRUCTIONS" not in env
    assert env["github_action_config.auto_review"] == "true"
    assert env["github_action_config.auto_improve"] == "false"


def test_build_env_anthropic_key_routes_to_anthropic_section(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant")
    env = PrAgentBackend().build_env(
        get_preset("anthropic"),
        ModelConfig(provider="anthropic", name="claude-sonnet-4-6"),
        knowledge="",
        tools=[],
    )
    assert env["CONFIG__MODEL"] == "anthropic/claude-sonnet-4-6"
    assert env["ANTHROPIC__KEY"] == "sk-ant"
    assert "ANTHROPIC__API_BASE" not in env


def test_build_env_model_base_url_overrides_preset():
    env = PrAgentBackend().build_env(
        get_preset("ollama"),
        ModelConfig(
            provider="ollama",
            name="qwen2.5-coder:32b",
            base_url="http://host.docker.internal:11434",
        ),
        knowledge="",
        tools=["review"],
    )
    assert env["OLLAMA__API_BASE"] == "http://host.docker.internal:11434"


def test_build_env_key_omitted_when_env_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    env = PrAgentBackend().build_env(
        get_preset("openai"),
        ModelConfig(provider="openai", name="gpt-4o"),
        knowledge="",
        tools=[],
    )
    assert "OPENAI__KEY" not in env
