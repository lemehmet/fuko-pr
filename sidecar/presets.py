"""Model-provider presets.

A preset maps a short provider name (used in ``.fuko.toml``) to the endpoint,
LiteLLM model prefix, key env var, and any known per-provider quirks a backend
must account for. Adding a provider is adding an entry here -- data, not code.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderPreset:
    """Connection details and quirks for one model provider.

    ``requires_base_url`` marks presets that reach their model ONLY through
    ``base_url``. Usually that is because there is no meaningful default
    endpoint (e.g. rented GPU boxes, whose address changes per rental), and
    then the model entry in ``.fuko.toml`` must supply one; a backend fails
    fast if it doesn't, because otherwise the preset's key would silently go to
    the SDK's default endpoint. A preset may carry BOTH a default and this flag
    (``codex-proxy``): the default answers "which endpoint", the flag is what
    makes the agentic backend refuse subscription auth, which injects no
    endpoint at all and would therefore reach Anthropic rather than the
    gateway.

    ``key_env`` names the FIRST credential. A preset may be backed by several,
    each an independent quota bucket, supplied as numbered siblings of that
    name -- see :func:`resolve_credentials`.
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
    # ChatGPT/Codex, through an Anthropic-to-Codex translator on the runner.
    # Added 2026-09-07: OpenAI serves no Anthropic-compatible /v1/messages (and
    # neither does OpenRouter), so unlike every gateway preset above this one is
    # not a base-URL swap -- something on the box has to translate Anthropic
    # Messages into Codex's Responses API. That something is claude-code-proxy
    # (raine, MIT), installed and pinned by the runner playbook; see
    # runner-setup.md.
    #
    # WHAT THE SUBSCRIPTION BUYS AND WHAT IT DOES NOT: this is the ChatGPT plan,
    # reached through the proxy's own stored OAuth session. It is NOT the OpenAI
    # API, which the `openai` preset above reaches with OPENAI_KEY and bills as
    # separate credits -- a ChatGPT subscription does not include them.
    #
    # The base URL is the unit's pinned loopback listener, so the entry does not
    # have to spell it -- but `requires_base_url` is set anyway, and it is the
    # tooth that matters here: the proxy is the ONLY thing that can serve this
    # entry's `gpt-*` slugs, and subscription auth injects no endpoint, so
    # without the flag an entry that left `auth` at its `auto` default with
    # CODEX_PROXY_KEY unexported would run against api.anthropic.com under the
    # runner's own Claude login -- a real Claude review published under a
    # `gpt-…` label.
    #
    # `small_model` is what keeps the harness's background haiku-class and
    # subagent calls off the expensive tier; the plan does have a cheap one,
    # unlike the single-model deployments `anthropic-compatible` serves.
    # VERIFY it against the account when flipping the main model -- a slug the
    # plan does not serve fails only on the auxiliary calls, which is the
    # quietest way this class breaks.
    #
    # STANDING (recorded because it is asked every time): the operator's own
    # subscription, on the operator's own repositories, no resale and no second
    # user. The reverse direction -- an Anthropic subscription token through a
    # third-party proxy -- does violate Anthropic's terms, and nothing here does
    # it.
    "codex-proxy": ProviderPreset(
        litellm_prefix="anthropic/",
        base_url="http://127.0.0.1:18765",
        key_env="CODEX_PROXY_KEY",
        quirks={"small_model": "gpt-5.6-luna"},
        requires_base_url=True,
    ),
    # Kimi Code -- Moonshot's Kimi membership (the Plus/Pro/Max/Ultra tiers
    # with "Kimi Code benefits"), reached at its Anthropic-compatible endpoint.
    # Added 2026-10-25 for the basil seat's return as a Kimi K3.1 Pro trial.
    #
    # A base-URL swap like `zai-anthropic`, NOT a translator like
    # `codex-proxy`: the endpoint answers /v1/messages natively (verified
    # 2026-10-25 -- the response `model` echoes the requested slug verbatim,
    # a tool definition answers `stop_reason: tool_use`, prompt caching
    # reports cache_read_input_tokens). The key is minted in the Kimi Code
    # Console (kimi.ai -> Kimi Code -> Console, up to five, shown once) and is
    # NOT a platform.moonshot.ai key: that is the pay-per-token API, a
    # different product the `openrouter` preset's `moonshotai/kimi-k3` also
    # reaches, billed separately. Plans and keys must not be mixed.
    #
    # `api.kimi.ai` is the overseas host; `api.kimi.com` serves the same
    # product for China. Both accept the same key.
    #
    # Every model this endpoint serves is THINKING-ONLY (`/v1/models` reports
    # `supports_thinking_type: only`): a bare call returns a `thinking` block
    # unrequested, exactly as the QwenCloud gateway did, so a seat here pays
    # reasoning tokens on every tool-call turn and wants the same generous
    # per-entry `tool_timeout` that lesson produced. Kimi's own Claude Code
    # setup points every model-tier variable at the main model and names no
    # cheap tier (the `-highspeed` variants cost MORE quota), so there is no
    # `small_model` quirk: the entry's own model serves every slot, as for
    # `anthropic-compatible`. Name the plain slug (`k3.1-pro`), never the
    # `[1m]` spelling: that suffix is a Claude Code hint stripped before the
    # call, and the receipt's label would then disagree with the answering
    # model on every round. `max_context` on the entry sizes the window.
    "kimi-code": ProviderPreset(
        litellm_prefix="anthropic/",
        base_url="https://api.kimi.ai/coding",
        key_env="KIMI_CODE_KEY",
    ),
    # Any gateway that speaks the Anthropic Messages API and is not one of the
    # named vendors above -- a self-hosted LiteLLM or vLLM, a rented box, a
    # provider we have not earned a preset for yet. The endpoint is the
    # deployment's, never ours, hence `requires_base_url`: without it the key
    # would go to api.anthropic.com, which for a local gateway's throwaway key
    # is a confusing 401 and for a fleet that also holds a real Anthropic key
    # is a silently billed review against the wrong model.
    #
    # No `small_model` quirk, and that is a choice rather than an omission. The
    # sibling gateway presets name a cheap tier so the harness's background
    # haiku-class calls do not run on the expensive model; a single-model
    # deployment has no cheap tier, and pointing those calls at a second slug
    # would make it swap models mid-review. Absent the quirk the backend falls
    # back to the entry's own model for every slot, which is the correct
    # default here.
    "anthropic-compatible": ProviderPreset(
        litellm_prefix="anthropic/",
        key_env="ANTHROPIC_COMPAT_KEY",
        requires_base_url=True,
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


class CredentialGapError(ValueError):
    """Raised when a preset's numbered sibling keys are not contiguous from 1."""


def credential_env(key_env: str, index: int) -> str:
    """Return the env var holding credential ``index`` (1-based) of ``key_env``.

    Credential 1 is the bare name, so a single-key deployment is spelled exactly
    as it was before siblings existed; credential N is ``<key_env>_<N>``.
    """
    return key_env if index == 1 else f"{key_env}_{index}"


def _sibling_indexes(key_env: str, environ: Mapping[str, str]) -> list[int]:
    """Indexes (>= 2) of every non-empty numbered sibling of ``key_env``."""
    pattern = re.compile(re.escape(key_env) + r"_([1-9]\d*)")
    found = []
    for name, value in environ.items():
        match = pattern.fullmatch(name)
        if match and value and int(match.group(1)) >= 2:
            found.append(int(match.group(1)))
    return sorted(found)


def resolve_credentials(
    preset: ProviderPreset, environ: Mapping[str, str] | None = None
) -> list[str]:
    """Return the preset's credential VALUES in drain order (credential 1 first).

    Numbered siblings rather than one list-valued secret (#298): GitHub masks a
    secret, and the transcript scrubber redacts one, only as its whole value, so
    each token inside a ``k1 k2`` secret would be printed unmasked.

    An empty value reads as absent, because GitHub renders an unset secret as
    an empty string. The siblings must be contiguous from the bare name: a gap
    (``_2`` set without the bare name, ``_3`` without ``_2``) raises rather than
    silently leaving a paid-for credential unused.

    Raises:
        CredentialGapError: A numbered sibling is set past an unset one.
    """
    if not preset.key_env:
        return []
    environ = os.environ if environ is None else environ
    values: list[str] = []
    while value := environ.get(credential_env(preset.key_env, len(values) + 1)):
        values.append(value)
    stranded = [i for i in _sibling_indexes(preset.key_env, environ) if i > len(values)]
    if stranded:
        missing = credential_env(preset.key_env, len(values) + 1)
        names = ", ".join(credential_env(preset.key_env, i) for i in stranded)
        raise CredentialGapError(
            f"{names} is set but {missing} is not; provider keys must be numbered "
            f"contiguously ({preset.key_env}, {preset.key_env}_2, {preset.key_env}_3, ...)"
        )
    return values


def provider_key_vars(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """Every env var name that holds a provider key, numbered siblings included.

    The bare name of every preset, plus whatever ``environ`` carries that is
    SHAPED like a sibling of one. Matched by shape rather than taken from
    :func:`resolve_credentials` on purpose: this set is what gets stripped from
    an agent's environment and scrubbed from its transcript, so a name the
    resolver refuses to use must still be covered. The shape is therefore
    WIDER than the resolver's -- any ``<key_env>_<digits>``, which takes in a
    sibling past a gap and the natural misnamings of the first key (``_1``,
    ``_01``) that are never read as credentials but hold one all the same.
    """
    environ = os.environ if environ is None else environ
    names: set[str] = set()
    for preset in PRESETS.values():
        if preset.key_env:
            names.add(preset.key_env)
            shape = re.compile(re.escape(preset.key_env) + r"_\d+")
            names.update(name for name in environ if shape.fullmatch(name))
    return frozenset(names)
