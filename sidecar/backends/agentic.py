"""The agentic review backend: fuko's own reviewer as a drop-in driver.

Where :mod:`sidecar.backends.pragent` drives an external single-shot harness
and re-parses its published markdown, this backend drives
:mod:`sidecar.reviewer` -- an agent with a real checkout and read-only
navigation tools -- and OWNS its output end to end: findings are born as
Review Signals and posted with their markers already attached, so the egress
scrape/PATCH machinery PR-Agent needs does not exist here.

Split of responsibilities across the protocol:

* :meth:`AgenticBackend.invoke` runs the review (checkout, agent, parse) and
  stashes the structured findings in memory. It posts nothing.
* :meth:`AgenticBackend.normalize_output` turns the stash into signals and
  posts them as ONE pull-request review (inline comments for anchored
  findings, the summary plus unanchored findings in the review body) under the
  calling branch's token. This is why posting lives on the egress side: only
  ``normalize_output`` receives the branch's ``role``, ``compare_label``, and
  identity ``token``, and the marker must carry the true role from birth.

Current limits, on purpose: only the ``anthropic`` preset is accepted (the
headless-Claude harness authenticates via ``ANTHROPIC_API_KEY``; other model
families arrive with an OSS harness behind the same seam), and only the
``review`` tool exists -- ``improve``/``describe`` entries are ignored rather
than errors so a shared ``[review].tools`` list keeps working.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import httpx

from ..fukoconfig import ModelConfig, ReviewConfig
from ..presets import ProviderPreset
from ..reviewer.checkout import (
    CheckoutError,
    checkout_pr_head,
    fetch_pr_context,
    strip_agent_config,
)
from ..reviewer.harness import (
    DEFAULT_MAX_TURNS,
    HarnessNotAvailableError,
    check_auth,
    is_auth_failure,
    run_review,
)
from ..reviewer.prompt import (
    MAX_FINDINGS,
    AgenticFinding,
    ReviewParseError,
    build_prompt,
    parse_review,
)
from ..signals import ReviewSignal, make_id, with_marker, with_visible_label
from ..throttle import is_throttle
from .base import InvokeResult, PRRef

_ENV_MODEL = "FUKO_AGENTIC_MODEL"
_ENV_INSTRUCTIONS = "FUKO_AGENTIC_INSTRUCTIONS"
# Kept separate from the instructions on purpose: the knowledge blob is mined
# from the repository's own review threads, so it carries the repository's trust
# level and must not enter the prompt as operator instruction (see
# `sidecar.reviewer.prompt.build_prompt`).
_ENV_KNOWLEDGE = "FUKO_AGENTIC_KNOWLEDGE"
_ENV_AUTH = "FUKO_AGENTIC_AUTH"
# Runner-merged GitHub credential names (PR-Agent dunder shape until #99 moves
# them behind the driver contract); the process fallbacks keep laptop runs working.
_ENV_GH_TOKEN = "GITHUB__USER_TOKEN"

# Every GitHub credential the review process may carry, stripped from the agent
# subprocess: it has no use for them (its tools are read-only and networkless),
# so carrying them is pure blast radius. `GH_TOKEN`/`GH_ENTERPRISE_TOKEN` are
# the `gh` CLI's own spellings -- easy to miss because nothing in this module
# sets them, and just as easy for a runner image to export.
_GITHUB_CRED_VARS = (
    "GITHUB_TOKEN",
    _ENV_GH_TOKEN,
    "GH_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)

_AUTH_API_KEY = "api-key"
_AUTH_SUBSCRIPTION = "subscription"
# Everything ambient that can decide WHO PAYS or WHERE THE TRAFFIC GOES. The
# credential three: Claude Code's precedence is ANTHROPIC_AUTH_TOKEN >
# ANTHROPIC_API_KEY > apiKeyHelper > CLAUDE_CODE_OAUTH_TOKEN > the runner's
# interactive login, so an ambient API key silently moves billing OFF a
# subscription (verified: with ANTHROPIC_API_KEY set, `claude auth status`
# reports apiKeySource=ANTHROPIC_API_KEY and subscriptionType=null). And the
# endpoint: an ambient ANTHROPIC_BASE_URL would aim the runner's authenticated
# session at a non-Anthropic host, which is credential exfiltration rather than
# a routing quirk -- so it is stripped too and re-injected in api-key mode ONLY
# from configuration (`model.base_url or preset.base_url`). All of these are
# stripped from the ambient environment and each mode injects exactly what it
# means to use: config decides, never the ambient environment.
_ANTHROPIC_INHERITED_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)


@dataclass
class _PendingReview:
    """One completed-but-unposted review, keyed until egress claims it."""

    findings: list[AgenticFinding]
    summary: str
    head_sha: str
    diff_files: frozenset[str]
    #: Findings the agent itself rated low-confidence -- the pressure valve the
    #: strategy promises it, reported as "withheld".
    withheld_low: int = 0
    #: Findings cut purely by :data:`MAX_FINDINGS`. A different thing entirely:
    #: these were confident enough to report and the cap is ours, so they are
    #: reported separately rather than as "low-confidence".
    over_cap: int = 0


class AgenticBackend:
    """Drive fuko's agentic reviewer (headless Claude Code) over a PR."""

    name = "agentic"
    supports_inline_suggestions = False
    injection = "prompt"

    def __init__(self, config: ReviewConfig | None = None) -> None:
        """Take the per-tool timeout from ``[review]``; other knobs are constants."""
        self.tool_timeout = config.tool_timeout if config else 900
        self.max_turns = DEFAULT_MAX_TURNS
        self._pending: dict[tuple[str, str], _PendingReview] = {}
        self._lock = threading.Lock()

    def build_env(
        self,
        preset: ProviderPreset,
        model: ModelConfig,
        knowledge: str,
        tools: list[str],
    ) -> dict[str, str]:
        """Translate the model entry into the harness environment.

        Only the ``anthropic`` preset family is accepted: the harness is
        headless Claude Code. It authenticates two ways, chosen by the entry's
        ``auth`` field (see :attr:`sidecar.fukoconfig.ModelConfig.auth`):

        * ``api-key`` -- the preset's key env var is passed through as
          ``ANTHROPIC_API_KEY`` (plus ``ANTHROPIC_BASE_URL`` for a gateway).
          A missing key is a config error raised here, not a runtime surprise.
        * ``subscription`` -- NO key is passed; the agent authenticates as the
          runner's own logged-in Claude session (``claude setup-token`` for CI,
          or an interactive login under ``HOME``/``CLAUDE_CONFIG_DIR``).

        ``auto`` resolves to ``api-key`` when the preset's key env var is set,
        else ``subscription``.

        The per-entry ``extra_instructions`` and the shared ``knowledge`` blob
        travel in SEPARATE variables (:data:`_ENV_INSTRUCTIONS` and
        :data:`_ENV_KNOWLEDGE`) rather than pre-joined as the pr-agent backend
        joins them, because the prompt gives them different trust levels: the
        operator wrote the first, the reviewed repository produced the second.
        ``tools`` is accepted for protocol parity; anything besides ``review``
        is ignored (one tool here).
        """
        if preset.litellm_prefix != "anthropic/":
            raise ValueError(
                f"backend 'agentic' currently supports only the 'anthropic' preset "
                f"(its harness is headless Claude Code); got provider "
                f"'{model.provider}'. Other model families arrive with an OSS "
                f"harness -- until then run them on the pr-agent backend."
            )
        auth = self._resolve_auth(preset, model)
        env: dict[str, str] = {_ENV_MODEL: model.name, _ENV_AUTH: auth}
        if auth == _AUTH_API_KEY:
            key = os.environ.get(preset.key_env or "", "")
            if not key:
                raise ValueError(
                    f"model entry '{model.provider}/{model.name}' asks for "
                    f"auth = 'api-key' but {preset.key_env or '<no key env>'} is "
                    f"not set; export it, or use auth = 'subscription' to run as "
                    f"the runner's own logged-in Claude session."
                )
            env["ANTHROPIC_API_KEY"] = key
            base_url = model.base_url or preset.base_url
            if base_url:
                env["ANTHROPIC_BASE_URL"] = base_url
        if model.extra_instructions:
            env[_ENV_INSTRUCTIONS] = model.extra_instructions
        if knowledge:
            env[_ENV_KNOWLEDGE] = knowledge
        return env

    @staticmethod
    def _resolve_auth(preset: ProviderPreset, model: ModelConfig) -> str:
        """Resolve the entry's ``auth`` setting to a concrete mode.

        ``auto`` prefers an available API key because that is the mode a
        key-holding user almost always means; a runner with no key is a
        subscription runner. Pin the field explicitly when both are present --
        that is the case where the default would quietly charge per token.
        """
        if model.auth != "auto":
            return model.auth
        has_key = bool(preset.key_env and os.environ.get(preset.key_env))
        return _AUTH_API_KEY if has_key else _AUTH_SUBSCRIPTION

    def invoke(self, pr: PRRef, env: dict[str, str], tools: list[str]) -> InvokeResult:
        """Check out the PR head, run the agent, and stash the parsed findings.

        The harness subprocess gets the ambient environment MINUS every GitHub
        credential and MINUS everything that decides who pays or where the
        traffic goes (see :data:`_ANTHROPIC_INHERITED_VARS`), plus exactly what
        its auth mode means to use. **Config decides, never the ambient
        environment**: api-key mode re-injects ``ANTHROPIC_BASE_URL`` only from
        ``model.base_url or preset.base_url``, so a gateway user sets
        ``base_url`` on the model entry rather than exporting it, and
        subscription mode never gets a base URL at all -- an inherited one
        would point the runner's own authenticated session at a foreign host.
        Everything else passes through untouched -- notably
        ``HOME``/``CLAUDE_CONFIG_DIR``, which is where a subscription login
        lives.

        The agent runs from a clean scratch directory with the checkout mounted
        read-only beside it, never *inside* the checkout: project config in a
        reviewed repository would otherwise execute on this runner (see
        :mod:`sidecar.reviewer.harness`). The checkout is additionally stripped
        of agent-config files before the run.

        A throttle-shaped failure (timeout, 429/overloaded, an exhausted
        subscription window) reports ``throttled=True`` so the pool fails over
        exactly as it does for PR-Agent; an authentication failure explicitly
        does NOT, because failing over would burn every remaining provider on
        what is a one-line runner fix.
        """
        token = env.get(_ENV_GH_TOKEN) or os.environ.get("GITHUB_TOKEN", "")
        api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        model_name = env.get(_ENV_MODEL, "")
        if not model_name:
            # An empty --model would reach the CLI as a confusing runtime error
            # after a full clone; say what is actually wrong, before paying for it.
            return InvokeResult(
                returncode=1,
                detail=f"no model name in the harness environment ({_ENV_MODEL} unset or empty)",
            )
        auth = env.get(_ENV_AUTH, _AUTH_SUBSCRIPTION)

        harness_env = {
            k: v
            for k, v in os.environ.items()
            if k not in _GITHUB_CRED_VARS
            and not k.startswith("FUKO_GITHUB_")
            and k not in _ANTHROPIC_INHERITED_VARS
        }
        if auth == _AUTH_API_KEY:
            for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
                if key in env:
                    harness_env[key] = env[key]
        else:
            # The CI form of a subscription login: a long-lived token from
            # `claude setup-token`. An interactive login instead lives under
            # HOME/CLAUDE_CONFIG_DIR, which never left the environment.
            oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
            if oauth:
                harness_env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth
            status = check_auth(harness_env)
            if status is not None and not status.get("loggedIn"):
                return InvokeResult(
                    returncode=1,
                    detail=(
                        "agentic backend is in subscription auth mode but this "
                        "runner has no logged-in Claude session; run `claude "
                        "setup-token` and export CLAUDE_CODE_OAUTH_TOKEN, or set "
                        "auth = 'api-key' on this model entry"
                    ),
                )

        try:
            ctx = fetch_pr_context(pr.repo, pr.number, token=token, api_url=api_url)
        except httpx.HTTPError as e:
            return InvokeResult(returncode=1, detail=f"could not fetch PR context: {e}")
        try:
            checkout = checkout_pr_head(
                pr.repo, pr.number, ctx.head_sha, token=token, server_url=server_url
            )
        except CheckoutError as e:
            return InvokeResult(returncode=1, detail=str(e))

        # Everything from here owns the checkout, so every exit path -- including
        # a failure to create the scratch cwd or to build the prompt -- goes
        # through the `finally` that removes it.
        workdir: Path | None = None
        try:
            workdir = Path(mkdtemp(prefix="fuko-agentic-cwd-"))
            prompt = build_prompt(
                ctx,
                env.get(_ENV_INSTRUCTIONS, ""),
                checkout_root=str(checkout),
                knowledge=env.get(_ENV_KNOWLEDGE, ""),
            )
            strip_agent_config(Path(checkout))
            result = run_review(
                prompt,
                Path(checkout),
                cwd=workdir,
                model=model_name,
                env=harness_env,
                timeout=self.tool_timeout,
                max_turns=self.max_turns,
            )
        except HarnessNotAvailableError as e:
            return InvokeResult(returncode=1, detail=str(e))
        except OSError as e:
            return InvokeResult(returncode=1, detail=f"could not prepare the review sandbox: {e}")
        finally:
            rmtree(checkout, ignore_errors=True)
            if workdir is not None:
                rmtree(workdir, ignore_errors=True)

        if result.returncode != 0:
            output = result.stderr + "\n" + result.text
            if is_auth_failure(output):
                return InvokeResult(
                    returncode=result.returncode,
                    detail=(
                        f"agent could not authenticate in {auth} mode: "
                        f"{result.stderr.strip()[:300]}"
                    ),
                )
            return InvokeResult(
                returncode=result.returncode,
                detail=result.stderr.strip()[:500] or "agent run failed",
                throttled=is_throttle(result.returncode, output),
            )
        try:
            review = parse_review(result.text)
        except ReviewParseError as e:
            return InvokeResult(returncode=1, detail=str(e)[:500])

        confident = [f for f in review.findings if f.confidence != "low"]
        kept = confident[:MAX_FINDINGS]
        with self._lock:
            self._pending[(pr.url, model_name)] = _PendingReview(
                findings=kept,
                summary=review.summary,
                head_sha=ctx.head_sha,
                diff_files=ctx.diff_files,
                withheld_low=len(review.findings) - len(confident),
                over_cap=len(confident) - len(kept),
            )
        return InvokeResult(returncode=0, detail=f"{len(kept)} findings")

    def normalize_output(
        self,
        pr: PRRef,
        model: str = "",
        *,
        compare_label: str | None = None,
        token: str | None = None,
        api_url: str | None = None,
        actor: str | None = None,
        role: str = "active",
    ) -> list[ReviewSignal]:
        """Post the stashed review under the branch identity and return its signals.

        Signals are built here -- not in :meth:`invoke` -- because the marker
        must carry the branch's true ``role`` and the post must happen under
        the branch's ``token``, and only this call receives them. Findings
        anchored to a diff file/line become inline review comments (marker and,
        in A/B mode, the visible label attached at creation); unanchored ones
        are listed in the review body. If GitHub rejects the inline set (e.g. a
        line the API will not anchor), the whole review is retried body-only,
        so a placement quirk degrades presentation rather than losing the
        review. A post that still fails returns ``[]`` -- signals that never
        reached the PR must not count as posted findings.

        ``actor`` is unused (protocol parity): output is born marked under this
        branch's own identity, so there is no foreign-comment scoping problem.
        """
        stash = self._claim(pr.url, model)
        if stash is None:
            return []
        token = os.environ.get("GITHUB_TOKEN", "") if token is None else token
        if api_url is None:
            api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")

        signals: list[ReviewSignal] = []
        inline: list[tuple[AgenticFinding, dict]] = []
        overflow: list[AgenticFinding] = []
        for f in stash.findings:
            signal = ReviewSignal(
                id=make_id(pr.url, f.file, str(f.line or 0), f.title),
                file=f.file,
                line=f.line,
                end_line=f.end_line,
                severity=f.severity,
                severity_source="declared",
                category=f.category,
                title=f.title,
                body=f.body,
                backend=self.name,
                model=model,
                role=role,
            )
            signals.append(signal)
            body = f"**{f.title}**\n\n{f.body}"
            if f.evidence:
                body += f"\n\n*Verified against:* {f.evidence}"
            if f.line is not None and f.file in stash.diff_files:
                if compare_label is not None:
                    body = with_visible_label(body, compare_label, signal)
                else:
                    body = with_marker(body, signal)
                comment = {"path": f.file, "line": f.line, "side": "RIGHT", "body": body}
                if f.end_line is not None and f.end_line > f.line:
                    comment.update(
                        {"start_line": f.line, "start_side": "RIGHT", "line": f.end_line}
                    )
                inline.append((f, comment))
            else:
                overflow.append(f)

        posted = self._post_review(pr, token, api_url, stash, inline, overflow)
        return signals if posted else []

    def _claim(self, pr_url: str, model: str) -> _PendingReview | None:
        """Pop the pending review for ``(pr_url, model)``, tolerating id spelling.

        ``invoke`` keys the stash by the bare harness model name while the
        runner hands egress the litellm-prefixed id, so the prefixed spelling
        is normalized before lookup.

        The last-resort claim is deliberately NOT "the only pending entry for
        this PR": in A/B mode several branches review the same PR through the
        same backend instance, so that rule lets whichever branch normalizes
        first walk off with another model's findings and post them under its own
        identity and role. The fallback therefore requires the remaining entry to
        be the *same model under a different spelling* (one side a suffix of the
        other), which still covers the id-drift case it exists for.
        """
        bare = model.rsplit("/", 1)[-1]
        with self._lock:
            for key in ((pr_url, bare), (pr_url, model)):
                if key in self._pending:
                    return self._pending.pop(key)
            mine = [k for k in self._pending if k[0] == pr_url]
            if len(mine) == 1:
                stashed = mine[0][1]
                stashed_bare = stashed.rsplit("/", 1)[-1]
                if (
                    bare
                    and stashed_bare
                    and (bare.endswith(stashed_bare) or stashed_bare.endswith(bare))
                ):
                    return self._pending.pop(mine[0])
        return None

    @staticmethod
    def _finding_line(f: AgenticFinding) -> str:
        """Render one finding as a review-body bullet (the unanchored fallback).

        Carries ``evidence`` through: the strategy requires every finding to
        cite what was read to verify it, and a reader of the review body needs
        that just as much as a reader of an inline comment does.
        """
        where = f"`{f.file}`" + (f":{f.line}" if f.line is not None else "")
        line = f"- **{f.title}** ({where}, {f.severity}): {f.body}"
        if f.evidence:
            line += f" *Verified against:* {f.evidence}"
        return line

    def _post_review(
        self,
        pr: PRRef,
        token: str,
        api_url: str,
        stash: _PendingReview,
        inline: list[tuple[AgenticFinding, dict]],
        overflow: list[AgenticFinding],
    ) -> bool:
        """POST one PR review; retry body-only on a 422; report success."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = "Bearer " + token
        body_parts = ["## fuko agentic review", "", stash.summary or "(no summary)"]
        if overflow:
            body_parts += ["", "### Findings without a diff anchor"]
            body_parts += [self._finding_line(f) for f in overflow]
        if stash.withheld_low:
            body_parts += ["", f"*{stash.withheld_low} low-confidence finding(s) withheld.*"]
        if stash.over_cap:
            body_parts += [
                "",
                f"*{stash.over_cap} further finding(s) cut by the {MAX_FINDINGS}-finding cap.*",
            ]
        payload = {
            "commit_id": stash.head_sha,
            "event": "COMMENT",
            "body": "\n".join(body_parts),
            "comments": [comment for _, comment in inline],
        }
        url = f"{api_url.rstrip('/')}/repos/{pr.repo}/pulls/{pr.number}/reviews"
        with httpx.Client(timeout=60.0, headers=headers) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 422 and inline:
                # A single unanchorable line 422s the whole review; degrade to
                # body-only rather than dropping the run's findings. Re-use each
                # comment's ALREADY-MARKED body rather than re-rendering it: the
                # marker (and, in A/B mode, the visible model label) is what
                # makes a finding recoverable as a Review Signal later, and a
                # plain bullet would strip exactly that.
                payload["comments"] = []
                blocks = [
                    f"**Location:** `{f.file}`" + (f":{f.line}" if f.line is not None else "")
                    for f, _ in inline
                ]
                payload["body"] += "\n\n### Inline findings (anchoring failed)\n\n" + (
                    "\n\n---\n\n".join(
                        f"{where}\n\n{comment['body']}"
                        for where, (_, comment) in zip(blocks, inline, strict=True)
                    )
                )
                resp = client.post(url, json=payload)
            if resp.status_code >= 300:
                print(
                    f"fuko: agentic review post failed ({resp.status_code}): {resp.text[:300]}",
                    file=sys.stderr,
                )
                return False
        return True
