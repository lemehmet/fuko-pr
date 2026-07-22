"""The PR-Agent review backend.

Translates the unified fuko config into PR-Agent's dynaconf settings, which must
be passed as DUNDER env vars (``SECTION__KEY``) -- dotted keys are silently
ignored. This is the tested home for every provider/model landmine that was
previously hand-tuned in the GitHub workflow: the coding-vs-paas endpoint,
``CONFIG__CUSTOM_MODEL_MAX_TOKENS`` for models absent from PR-Agent's table, and
the raised ``CONFIG__AI_TIMEOUT`` for slow reasoning models.

Output relays live: each tool's merged stdout/stderr streams line-by-line into
the runner's stderr as it is produced (``CONFIG__VERBOSITY_LEVEL=1`` +
``PYTHONUNBUFFERED=1`` make PR-Agent chatty and prompt), so CI logs show
progress in real time while still being scanned for throttle signatures.

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
import threading

import httpx

from ..fukoconfig import ModelConfig, ReviewConfig
from ..normalizers import pragent_signals
from ..presets import ProviderPreset
from ..throttle import is_throttle
from .base import InvokeResult, PRRef
from ..signals import ReviewSignal, extract_markers, with_marker, with_visible_label


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
            "CONFIG__VERBOSITY_LEVEL": "1",
            "PR_CODE_SUGGESTIONS__COMMITABLE_CODE_SUGGESTIONS": "true",
            "PR_REVIEWER__REQUIRE_TICKET_ANALYSIS_REVIEW": "false",
            "PYTHONUNBUFFERED": "1",
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
        max_model_tokens = (
            model.max_model_tokens
            if model.max_model_tokens is not None
            else quirks.get("max_model_tokens")
        )
        if max_model_tokens is not None:
            env["CONFIG__MAX_MODEL_TOKENS"] = str(max_model_tokens)

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

        Output streams LIVE: stdout+stderr are merged and relayed line by line
        to this process's stderr as the tool produces them, so a CI log shows
        PR-Agent's progress in real time instead of one buffered dump when the
        tool exits (the old ``subprocess.run`` capture). The relayed lines are
        also accumulated and scanned for a throttle signature on a non-zero
        exit. A throttle (or timeout) on a required tool returns early with
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
            name = f"fuko-pragent-{os.getpid()}-{threading.get_ident()}-{index}"
            optional = tool in self.optional_tools
            code, blob, timed_out = self._stream_tool(
                [*docker_base, "--name", name, self.image, "--pr_url", pr.url, tool],
                full_env,
                name,
            )
            if timed_out:
                what = f"{tool} timed out after {self.tool_timeout}s (container killed)"
                if not optional:
                    return InvokeResult(
                        returncode=124, detail="; ".join([*details, what]), throttled=True
                    )
                _record(tool, 124, what)
                continue

            if code != 0:
                throttled = is_throttle(code, blob)
                if throttled and not optional:
                    return InvokeResult(
                        returncode=code,
                        detail="; ".join([*details, f"{tool} throttled (exit {code})"]),
                        throttled=True,
                    )
                suffix = " (throttled)" if throttled else ""
                _record(tool, code, f"{tool} exited {code}{suffix}")
        return InvokeResult(returncode=rc, detail="; ".join(details))

    def _stream_tool(
        self, cmd: list[str], full_env: dict[str, str], container: str
    ) -> tuple[int, str, bool]:
        """Run one docker command, relaying its merged output live.

        Returns ``(returncode, captured_output, timed_out)``. stdout and stderr
        are merged (one pipe preserves interleaving order) and each line is
        printed to this process's stderr with an immediate flush -- on a GitHub
        runner that is what makes the review branch's progress visible while it
        runs. A reader thread drains the pipe so a chatty tool can never fill
        the pipe buffer and deadlock against ``wait()``. On the normal path the
        reader is joined WITHOUT a timeout: the exited child was the pipe's
        only writer, so EOF is guaranteed and the join is what guarantees
        ``captured`` is complete before the throttle scan reads it (a timed
        join could truncate the blob and miss a throttle signature). On
        timeout the container is reaped (a hung tool must not outlive the
        killed subprocess on a persistent self-hosted runner); if the docker
        client itself won't exit after the kill (unresponsive daemon) it is
        killed directly so it can't leak, the reader join IS bounded (a stuck
        writer may never EOF), and the possibly-partial capture is acceptable
        because a timeout already reports ``throttled=True`` unconditionally.
        The docker kill itself is bounded too -- it is reached precisely when
        the daemon may be unresponsive. On every path the pipe is explicitly
        closed before returning (one leaked fd per tool adds up on a
        persistent runner), which also unblocks a reader stuck mid-read.
        """
        captured: list[str] = []
        proc = subprocess.Popen(
            cmd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _relay() -> None:
            for line in proc.stdout:
                print(line, end="", file=sys.stderr, flush=True)
                captured.append(line)

        reader = threading.Thread(target=_relay, daemon=True)
        reader.start()
        timed_out = False
        try:
            code = proc.wait(timeout=self.tool_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            code = 124
            try:
                subprocess.run(
                    ["docker", "kill", container],
                    check=False,
                    timeout=30,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            reader.join(timeout=10)
        else:
            reader.join()

        if reader.is_alive():
            try:
                proc.stdout.close()
            except Exception:
                pass
            reader.join(timeout=2)
        try:
            proc.stdout.close()
        except Exception:
            pass
        return code, "".join(captured), timed_out

    def normalize_output(
        self,
        pr: PRRef,
        model: str = "",
        *,
        compare_label: str | None = None,
        token: str | None = None,
        api_url: str | None = None,
        actor: str | None = None,
    ) -> list[ReviewSignal]:
        """Read PR-Agent's inline comments, map them to Review Signals, and mark them.

        Detection is by comment *format* (PR-Agent posts under whatever token ran
        it), so this matches its ``**Suggestion:**`` shape rather than an author.
        Marker injection is best-effort: GitHub only allows editing comments the
        current token authored, so foreign comments simply stay unmarked. Failure
        to read comments degrades to an empty list -- the review itself already ran.

        When ``compare_label`` is set (A/B mode) the newly marked comments also get
        a compact visible tag of that label, so the producing branch is legible on
        the diff. The label is the configured ``provider/name`` (matching the branch
        header), distinct from the litellm-prefixed ``model`` in the marker.

        ``token``/``api_url`` pin the GitHub identity that reads and edits comments;
        when unset they fall back to the process ``GITHUB_TOKEN``/``GITHUB_API_URL``.
        ``actor`` is that identity's user id when the caller already knows it (the
        branch header post reveals it, even for App installation tokens whose
        ``GET /user`` 403s). It is what keeps marking author-scoped: a repo-write
        token CAN edit a sibling branch's comments -- GitHub does not stop it
        (#66) -- so in A/B mode marking is refused outright when no identity can
        be resolved rather than risk relabeling another branch's output.
        """
        token = os.environ.get("GITHUB_TOKEN", "") if token is None else token
        if api_url is None:
            api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        api = api_url.rstrip("/")
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
        self._inject_markers(api, pr, headers, pairs, label=compare_label, actor=actor)
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

    def _resolve_actor(self, api: str, client: httpx.Client) -> str | None:
        """Return the GitHub actor id ``client``'s token authenticates as, or ``None``.

        Used to skip PATCHing comments this identity didn't author: in concurrent
        A/B mode every branch sees the *whole* PR's comments, and a repo-write
        token WILL successfully edit a sibling's comment (#66) -- authorship is
        not enforced by GitHub, so it must be filtered here. ``GET /user`` 403s
        for App installation tokens (#57), so App-token callers should pass the
        identity in from the branch-header post instead of relying on this probe.
        Any lookup failure returns ``None``; the caller decides what that means
        (fail-closed in A/B mode, legacy mark-all in solo mode).
        """
        try:
            resp = client.get(f"{api}/user")
            resp.raise_for_status()
            actor_id = resp.json().get("id")
        except (httpx.HTTPError, ValueError):
            return None
        return str(actor_id) if actor_id is not None else None

    def _inject_markers(
        self,
        api: str,
        pr: PRRef,
        headers: dict[str, str],
        pairs: list[dict],
        label: str | None = None,
        actor: str | None = None,
    ) -> None:
        """Best-effort: append each signal's marker to its comment (skip on any error).

        Skips entirely when unauthenticated -- every PATCH would 401, so there is no
        point generating the API traffic. A comment that already carries *any*
        fuko-signal marker is left untouched: re-running a review must not relabel a
        comment, and in A/B compare mode it keeps each branch from overwriting the
        marker an earlier branch wrote on its own suggestions.

        Comments authored by a *different* GitHub identity are skipped before any
        PATCH: in concurrent A/B mode this identity sees every branch's comments,
        and a repo-write token successfully edits a sibling's comment -- GitHub
        does not enforce authorship (#66: fuko-gray relabeled a fuko-basil
        suggestion) -- so this filter is correctness-critical, not an optimization.
        ``actor`` comes from the caller when known (branch-header post); otherwise
        a ``GET /user`` probe is attempted. When no identity can be resolved in
        A/B mode (``label`` set), marking is skipped entirely -- an unmarked
        comment is recoverable, a cross-labeled one silently corrupts per-model
        attribution. Solo mode keeps the legacy best-effort mark-all.

        When ``label`` is given (A/B compare mode), the comment is also prefixed with
        a compact visible model tag so the producing branch is legible on the diff;
        only the comments this branch newly marks get tagged.
        """
        if not pairs or "Authorization" not in headers:
            return
        with httpx.Client(timeout=30.0, headers=headers) as client:
            if actor is None:
                actor = self._resolve_actor(api, client)
            if actor is None and label is not None:
                print(
                    "fuko: A/B marking skipped — branch identity unresolved; marking "
                    "could relabel a sibling branch's comments (#66)",
                    file=sys.stderr,
                )
                return
            for pair in pairs:
                comment, signal = pair["comment"], pair["signal"]
                if actor is not None and str((comment.get("user") or {}).get("id")) != actor:
                    continue
                body = comment.get("body") or ""
                if extract_markers(body):
                    continue
                new_body = (
                    with_visible_label(body, label, signal) if label else with_marker(body, signal)
                )
                try:
                    resp = client.patch(
                        f"{api}/repos/{pr.repo}/pulls/comments/{comment['id']}",
                        json={"body": new_body},
                    )
                    resp.raise_for_status()
                except httpx.HTTPError:
                    continue
