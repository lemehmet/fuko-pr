"""Model-provider presets.

A preset maps a short provider name (used in ``.fuko.toml``) to the endpoint,
LiteLLM model prefix, key env var, and any known per-provider quirks a backend
must account for. Adding a provider is adding an entry here -- data, not code.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderPreset:
    """Connection details and quirks for one model provider.

    ``requires_base_url`` marks presets with no meaningful default endpoint
    (e.g. rented GPU boxes, whose address changes per rental): the model entry
    in ``.fuko.toml`` must supply ``base_url``, and a backend fails fast if it
    doesn't — otherwise the preset's key would silently go to the SDK's
    default endpoint.
    """

    litellm_prefix: str
    base_url: str | None = None
    key_env: str | None = None
    quirks: dict[str, object] = field(default_factory=dict)
    requires_base_url: bool = False


PRESETS: dict[str, ProviderPreset] = {
    "zai-coding": ProviderPreset(
        litellm_prefix="openai/",
        base_url="https://api.z.ai/api/coding/paas/v4",
        key_env="ZAI_KEY",
        quirks={
            "custom_model_max_tokens": 1000000,
            "max_model_tokens": 512000,
            "ai_timeout": 300,
        },
    ),
    "ollama": ProviderPreset(
        litellm_prefix="ollama/",
        base_url="http://localhost:11434",
    ),
    "ollama-cloud": ProviderPreset(
        litellm_prefix="openai/",
        base_url="https://ollama.com/v1",
        key_env="OLLAMA_API_KEY",
        quirks={
            "custom_model_max_tokens": 976000,
            "max_model_tokens": 512000,
            "ai_timeout": 300,
        },
    ),
    "lemonade": ProviderPreset(
        litellm_prefix="openai/",
        base_url="http://localhost:8000/api/v1",
        key_env="LEMONADE_API_KEY",
        quirks={
            "custom_model_max_tokens": 262144,
            "max_model_tokens": 131072,
            "ai_timeout": 540,
        },
    ),
    "openrouter": ProviderPreset(
        litellm_prefix="openai/",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_KEY",
        quirks={
            "custom_model_max_tokens": 1048576,
            "max_model_tokens": 512000,
            "ai_timeout": 300,
        },
    ),
    "prodia": ProviderPreset(
        litellm_prefix="openai/",
        key_env="PRODIA_KEY",
        quirks={
            "custom_model_max_tokens": 1048576,
            "max_model_tokens": 512000,
            "ai_timeout": 300,
        },
        requires_base_url=True,
    ),
    "openai": ProviderPreset(
        litellm_prefix="openai/",
        key_env="OPENAI_KEY",
    ),
    "anthropic": ProviderPreset(
        litellm_prefix="anthropic/",
        key_env="ANTHROPIC_KEY",
    ),
    # QwenCloud's Anthropic-compatible gateway (Token Plan). The `anthropic/`
    # prefix is what admits it to the agentic backend (headless Claude Code
    # speaks the Anthropic API; the gateway answers it) -- the model behind the
    # endpoint is Qwen, not Claude. Key: the Token Plan key (sk-sp-…); do NOT
    # point a DashScope/PAYG key here, plans and keys must not be mixed.
    # `small_model` maps Claude Code's background haiku-class calls to a slug
    # this gateway actually serves -- without it those calls request
    # `claude-haiku-*` from an endpoint that has never heard of it.
    "qwen-anthropic": ProviderPreset(
        litellm_prefix="anthropic/",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic",
        key_env="QWEN_TOKEN_PLAN_KEY",
        quirks={"small_model": "qwen3.6-flash"},
    ),
    # z.ai's Anthropic-compatible endpoint (the Coding Plan surface Claude
    # Code itself uses). Added 2026-08-24 for the henry seat's migration off
    # the QwenCloud Token Plan: that gateway degraded to unrecognized_model
    # with ~8000 units still on balance, and its pack purchases are capped.
    # `small_model` maps the harness's auxiliary calls; glm-4.5-air is the
    # plan's documented fast model — VERIFY against the account on first run
    # (a wrong value reproduces the exact generate_session_title failure the
    # qwen gateway showed).
    "zai-anthropic": ProviderPreset(
        litellm_prefix="anthropic/",
        base_url="https://api.z.ai/api/anthropic",
        key_env="ZAI_KEY",
        quirks={"small_model": "glm-4.5-air"},
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
