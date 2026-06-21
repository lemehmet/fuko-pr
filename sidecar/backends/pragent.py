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
import sys

import httpx

from ..fukoconfig import ModelConfig, ReviewConfig
from ..normalizers import pragent_signals
from ..presets import ProviderPreset
from ..throttle import is_throttle
from .base import InvokeResult, PRRef
from ..signals import ReviewSignal, encode_marker, with_marker


def _echo(stdout: str | None, stderr: str | None) -> None:
    """Re-emit a captured tool's output so CI logs still show PR-Agent's run.

    Output is captured (not inherited) so it can be scanned for a throttle
    signature; echoing keeps the logs intact at the cost of buffering until each
    tool finishes.
    """
    if stdout:
        print(stdout, end="", file=sys.stdout)
    if stderr:
        print(stderr, end="", file=sys.stderr)


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
        self.tool_timeout = config.tool_timeout if config else 900
        self.optional_tools = set(config.optional_tools) if config else set()

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

        Ticket-compliance analysis is disabled
        (``PR_REVIEWER__REQUIRE_TICKET_ANALYSIS_REVIEW=false``): it fetches the
        sub-issues of ``#<n>`` refs in the PR body and throws on a ref that
        resolves to a PR rather than an issue, and it is irrelevant to a review.
        """
        model_id = preset.litellm_prefix + model.name
        family = preset.litellm_prefix.rstrip("/").upper()
        env: dict[str, str] = {
            "CONFIG__MODEL": model_id,
            "CONFIG__FALLBACK_MODELS": f'["{model_id}"]',
            "PR_CODE_SUGGESTIONS__COMMITABLE_CODE_SUGGESTIONS": "true",
            "PR_REVIEWER__REQUIRE_TICKET_ANALYSIS_REVIEW": "false",
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

        Output is captured and re-echoed so it can be scanned for a throttle
        signature. A throttle (or timeout) on a required tool returns early with
        ``throttled=True`` so the runner fails over to the next provider without
        running the remaining tools; the same on an optional tool is a non-fatal
        skip.
        """
        full_env = {**os.environ, **env}
        forward: list[str] = []
        for key in env:
            forward += ["-e", key]
        docker_base = ["docker", "run", "--rm", *self.docker_extra_args, *forward]

        rc = 0
        details: list[str] = []

        def _record(tool: str, code: int, what: str) -> None:
            """Record a tool failure — fatal unless the tool is marked optional."""
            nonlocal rc
            if tool in self.optional_tools:
                details.append(f"{what} [optional]")
            else:
                rc = rc or code
                details.append(what)

        for index, tool in enumerate(tools):
            name = f"fuko-pragent-{os.getpid()}-{index}"
            optional = tool in self.optional_tools
            try:
                proc = subprocess.run(
                    [*docker_base, "--name", name, self.image, "--pr_url", pr.url, tool],
                    env=full_env,
                    check=False,
                    timeout=self.tool_timeout,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired as exc:
                # Reap the container so a hung tool can't outlive the killed
                # subprocess on a persistent self-hosted runner.
                _echo(exc.stdout, exc.stderr)
                subprocess.run(
                    ["docker", "kill", name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                what = f"{tool} timed out after {self.tool_timeout}s (container killed)"
                if not optional:
                    return InvokeResult(
                        returncode=124, detail="; ".join([*details, what]), throttled=True
                    )
                _record(tool, 124, what)
                continue

            _echo(proc.stdout, proc.stderr)
            if proc.returncode != 0:
                blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
                throttled = is_throttle(proc.returncode, blob)
                if throttled and not optional:
                    return InvokeResult(
                        returncode=proc.returncode,
                        detail="; ".join([*details, f"{tool} throttled (exit {proc.returncode})"]),
                        throttled=True,
                    )
                suffix = " (throttled)" if throttled else ""
                _record(tool, proc.returncode, f"{tool} exited {proc.returncode}{suffix}")
        return InvokeResult(returncode=rc, detail="; ".join(details))

    def normalize_output(self, pr: PRRef, model: str = "") -> list[ReviewSignal]:
        """Read PR-Agent's inline comments, map them to Review Signals, and mark them.

        Detection is by comment *format* (PR-Agent posts under whatever token ran
        it), so this matches its ``**Suggestion:**`` shape rather than an author.
        Marker injection is best-effort: GitHub only allows editing comments the
        current token authored, so foreign comments simply stay unmarked. Failure
        to read comments degrades to an empty list -- the review itself already ran.
        """
        token = os.environ.get("GITHUB_TOKEN", "")
        api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = "Bearer " + token

        try:
            comments = self._fetch_review_comments(api, pr, headers)
        except httpx.HTTPError as e:
            print(f"fuko: could not read comments for normalization: {e}", file=sys.stderr)
            return []

        pairs = pragent_signals(comments, model)
        self._inject_markers(api, pr, headers, pairs)
        return [p["signal"] for p in pairs]

    def _fetch_review_comments(self, api: str, pr: PRRef, headers: dict[str, str]) -> list[dict]:
        """Return all inline review comments on the PR (paginated)."""
        out: list[dict] = []
        page = 1
        with httpx.Client(timeout=30.0, headers=headers) as client:
            while True:
                resp = client.get(
                    f"{api}/repos/{pr.repo}/pulls/{pr.number}/comments",
                    params={"page": page, "per_page": 100},
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                out.extend(batch)
                if len(batch) < 100:
                    break
                page += 1
        return out

    def _inject_markers(
        self, api: str, pr: PRRef, headers: dict[str, str], pairs: list[dict]
    ) -> None:
        """Best-effort: append each signal's marker to its comment (skip on any error).

        Skips entirely when unauthenticated -- every PATCH would 401, so there is no
        point generating the API traffic.
        """
        if not pairs or "Authorization" not in headers:
            return
        with httpx.Client(timeout=30.0, headers=headers) as client:
            for pair in pairs:
                comment, signal = pair["comment"], pair["signal"]
                body = comment.get("body") or ""
                if encode_marker(signal) in body:
                    continue
                try:
                    resp = client.patch(
                        f"{api}/repos/{pr.repo}/pulls/comments/{comment['id']}",
                        json={"body": with_marker(body, signal)},
                    )
                    resp.raise_for_status()
                except httpx.HTTPError:
                    continue
