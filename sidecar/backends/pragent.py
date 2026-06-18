"""The PR-Agent review backend.

Translates the unified fuko config into PR-Agent's dynaconf settings, which must
be passed as DUNDER env vars (``SECTION__KEY``) -- dotted keys are silently
ignored. This is the tested home for every provider/model landmine that was
previously hand-tuned in the GitHub workflow: the coding-vs-paas endpoint,
``CONFIG__CUSTOM_MODEL_MAX_TOKENS`` for models absent from PR-Agent's table, and
the raised ``CONFIG__AI_TIMEOUT`` for slow reasoning models.

PR-Agent is invoked via its Docker image rather than pip: the package's pinned
dependencies are mutually unsatisfiable (e.g. ``google-cloud-storage==2.10.0``
vs ``google-cloud-aiplatform==1.154.0`` needing ``>=3.10.0``), so the official
image is the only reliable way to run it. The image is configurable; point it at
a pinned tag in your own registry once you publish one.
"""

from __future__ import annotations

import os
import subprocess

from ..fukoconfig import ModelConfig, ReviewConfig
from ..presets import ProviderPreset
from .base import InvokeResult, PRRef
from ..signals import ReviewSignal

_TOOL_FLAGS = {
    "review": "github_action_config.auto_review",
    "improve": "github_action_config.auto_improve",
    "describe": "github_action_config.auto_describe",
}


class PrAgentBackend:
    """Drive PR-Agent over any LiteLLM-supported model selected by a preset."""

    name = "pr-agent"
    supports_inline_suggestions = True
    injection = "extra_instructions"

    DEFAULT_IMAGE = "codiumai/pr-agent:latest"

    def __init__(self, config: ReviewConfig | None = None) -> None:
        """Configure the runtime image and extra ``docker run`` args from config."""
        self.image = config.image if config and config.image else self.DEFAULT_IMAGE
        self.docker_extra_args = list(config.docker_extra_args) if config else []

    def build_env(
        self,
        preset: ProviderPreset,
        model: ModelConfig,
        knowledge: str,
        tools: list[str],
    ) -> dict[str, str]:
        """Build the PR-Agent dunder-env mapping for the given provider and model.

        The LiteLLM model id is ``<prefix><name>`` (e.g. ``openai/glm-5.2``).
        PR-Agent's dynaconf has a section per provider family, so the API base
        and key route to ``<FAMILY>__API_BASE`` / ``<FAMILY>__KEY`` derived from
        the preset's prefix (``OPENAI``, ``OLLAMA``, ``ANTHROPIC``, ...).
        """
        model_id = preset.litellm_prefix + model.name
        family = preset.litellm_prefix.rstrip("/").upper()
        env: dict[str, str] = {
            "CONFIG__MODEL": model_id,
            "CONFIG__FALLBACK_MODELS": f'["{model_id}"]',
            "PR_CODE_SUGGESTIONS__COMMITABLE_CODE_SUGGESTIONS": "true",
        }

        base_url = model.base_url or preset.base_url
        if base_url:
            env[f"{family}__API_BASE"] = base_url
        if preset.key_env:
            key = os.environ.get(preset.key_env)
            if key:
                env[f"{family}__KEY"] = key

        quirks = preset.quirks
        if "custom_model_max_tokens" in quirks:
            env["CONFIG__CUSTOM_MODEL_MAX_TOKENS"] = str(quirks["custom_model_max_tokens"])
        if "ai_timeout" in quirks:
            env["CONFIG__AI_TIMEOUT"] = str(quirks["ai_timeout"])

        if knowledge:
            env["PR_REVIEWER__EXTRA_INSTRUCTIONS"] = knowledge
            env["PR_CODE_SUGGESTIONS__EXTRA_INSTRUCTIONS"] = knowledge

        for tool, flag in _TOOL_FLAGS.items():
            env[flag] = "true" if tool in tools else "false"

        return env

    def invoke(self, pr: PRRef, env: dict[str, str], tools: list[str]) -> InvokeResult:
        """Run PR-Agent's Docker image once per tool against the PR URL.

        Each translated env var is forwarded by name (``-e KEY``), so Docker reads
        its value from this process's environment -- keeping secrets and multiline
        ``extra_instructions`` out of the command line. The image runs exactly the
        named tool (``review``, ``improve``, ...); no GitHub event payload is
        required, so the runner works from any CI or a laptop.
        """
        full_env = {**os.environ, **env}
        forward: list[str] = []
        for key in env:
            forward += ["-e", key]
        docker_base = ["docker", "run", "--rm", *self.docker_extra_args, *forward]

        rc = 0
        details: list[str] = []
        for tool in tools:
            proc = subprocess.run(
                [*docker_base, self.image, "--pr_url", pr.url, tool],
                env=full_env,
                check=False,
            )
            if proc.returncode != 0:
                rc = proc.returncode
                details.append(f"{tool} exited {proc.returncode}")
        return InvokeResult(returncode=rc, detail="; ".join(details))

    def normalize_output(self, pr: PRRef) -> list[ReviewSignal]:
        """Map PR-Agent's posted review to Review Signals (implemented in egress phase)."""
        raise NotImplementedError("normalize_output lands with the egress work (task 8)")
