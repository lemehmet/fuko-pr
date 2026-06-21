"""Model-provider presets.

A preset maps a short provider name (used in ``.fuko.toml``) to the endpoint,
LiteLLM model prefix, key env var, and any known per-provider quirks a backend
must account for. Adding a provider is adding an entry here -- data, not code.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderPreset:
    """Connection details and quirks for one model provider."""

    litellm_prefix: str
    base_url: str | None = None
    key_env: str | None = None
    quirks: dict[str, object] = field(default_factory=dict)


PRESETS: dict[str, ProviderPreset] = {
    "zai-coding": ProviderPreset(
        litellm_prefix="openai/",
        base_url="https://api.z.ai/api/coding/paas/v4",
        key_env="ZAI_KEY",
        quirks={"custom_model_max_tokens": 128000, "ai_timeout": 300},
    ),
    "ollama": ProviderPreset(
        litellm_prefix="ollama/",
        base_url="http://localhost:11434",
    ),
    "ollama-cloud": ProviderPreset(
        litellm_prefix="openai/",
        base_url="https://ollama.com/v1",
        key_env="OLLAMA_API_KEY",
        quirks={"custom_model_max_tokens": 128000, "ai_timeout": 300},
    ),
    "openai": ProviderPreset(
        litellm_prefix="openai/",
        key_env="OPENAI_KEY",
    ),
    "anthropic": ProviderPreset(
        litellm_prefix="anthropic/",
        key_env="ANTHROPIC_KEY",
    ),
}


class UnknownPresetError(KeyError):
    """Raised when a ``.fuko.toml`` names a provider preset that is not registered."""


def get_preset(name: str) -> ProviderPreset:
    """Return the registered preset for ``name`` or raise ``UnknownPresetError``."""
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise UnknownPresetError(
            f"unknown model provider '{name}'; known presets: {known}"
        ) from None
