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

import httpx

from ..fukoconfig import ModelConfig, ReviewConfig
from ..presets import ProviderPreset
from ..reviewer.checkout import CheckoutError, checkout_pr_head, fetch_pr_context
from ..reviewer.harness import (
    DEFAULT_MAX_TURNS,
    HarnessNotAvailableError,
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
# Runner-merged GitHub credential names (PR-Agent dunder shape until #99 moves
# them behind the driver contract); the process fallbacks keep laptop runs working.
_ENV_GH_TOKEN = "GITHUB__USER_TOKEN"


@dataclass
class _PendingReview:
    """One completed-but-unposted review, keyed until egress claims it."""

    findings: list[AgenticFinding]
    summary: str
    head_sha: str
    diff_files: frozenset[str]
    dropped: int = 0


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

        Only the ``anthropic`` preset family is accepted: the headless-Claude
        harness reads ``ANTHROPIC_API_KEY`` (mapped here from the preset's
        ``key_env``) and optionally ``ANTHROPIC_BASE_URL``. The per-entry
        ``extra_instructions`` and the shared ``knowledge`` blob are combined
        exactly as the pr-agent backend combines them (steering first) and
        carried under :data:`_ENV_INSTRUCTIONS` into the review prompt's
        operator-guidance section. ``tools`` is accepted for protocol parity;
        anything besides ``review`` is ignored (this backend has one tool).
        """
        if preset.litellm_prefix != "anthropic/":
            raise ValueError(
                f"backend 'agentic' currently supports only the 'anthropic' preset "
                f"(its harness is headless Claude Code); got provider "
                f"'{model.provider}'. Other model families arrive with an OSS "
                f"harness -- until then run them on the pr-agent backend."
            )
        env: dict[str, str] = {_ENV_MODEL: model.name}
        if preset.key_env:
            key = os.environ.get(preset.key_env)
            if key:
                env["ANTHROPIC_API_KEY"] = key
        base_url = model.base_url or preset.base_url
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
        instructions = "\n\n".join(part for part in (model.extra_instructions, knowledge) if part)
        if instructions:
            env[_ENV_INSTRUCTIONS] = instructions
        return env

    def invoke(self, pr: PRRef, env: dict[str, str], tools: list[str]) -> InvokeResult:
        """Check out the PR head, run the agent, and stash the parsed findings.

        The harness subprocess gets the ambient environment MINUS GitHub
        credentials (the agent's tools are read-only and networkless, but the
        review process still should not carry tokens it has no use for), plus
        the translated Anthropic credentials. A throttle-shaped failure
        (timeout, 429/overloaded signature) reports ``throttled=True`` so the
        pool fails over exactly as it does for PR-Agent.
        """
        token = env.get(_ENV_GH_TOKEN) or os.environ.get("GITHUB_TOKEN", "")
        api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        model_name = env.get(_ENV_MODEL, "")

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

        harness_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("GITHUB_TOKEN", _ENV_GH_TOKEN) and not k.startswith("FUKO_GITHUB_")
        }
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
            if key in env:
                harness_env[key] = env[key]

        prompt = build_prompt(ctx, env.get(_ENV_INSTRUCTIONS, ""))
        try:
            result = run_review(
                prompt,
                Path(checkout),
                model=model_name,
                env=harness_env,
                timeout=self.tool_timeout,
                max_turns=self.max_turns,
            )
        except HarnessNotAvailableError as e:
            return InvokeResult(returncode=1, detail=str(e))
        finally:
            rmtree(checkout, ignore_errors=True)

        if result.returncode != 0:
            output = result.stderr + "\n" + result.text
            return InvokeResult(
                returncode=result.returncode,
                detail=result.stderr.strip()[:500] or "agent run failed",
                throttled=is_throttle(result.returncode, output),
            )
        try:
            review = parse_review(result.text)
        except ReviewParseError as e:
            return InvokeResult(returncode=1, detail=str(e)[:500])

        kept = [f for f in review.findings if f.confidence != "low"][:MAX_FINDINGS]
        dropped = len(review.findings) - len(kept)
        with self._lock:
            self._pending[(pr.url, model_name)] = _PendingReview(
                findings=kept,
                summary=review.summary,
                head_sha=ctx.head_sha,
                diff_files=ctx.diff_files,
                dropped=dropped,
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
        is normalized before lookup; a lone pending entry for the PR is claimed
        as a last resort (solo runs where the ids drift).
        """
        bare = model.rsplit("/", 1)[-1]
        with self._lock:
            for key in ((pr_url, bare), (pr_url, model)):
                if key in self._pending:
                    return self._pending.pop(key)
            mine = [k for k in self._pending if k[0] == pr_url]
            if len(mine) == 1:
                return self._pending.pop(mine[0])
        return None

    @staticmethod
    def _finding_line(f: AgenticFinding) -> str:
        """Render one finding as a review-body bullet (the unanchored fallback)."""
        where = f"`{f.file}`" + (f":{f.line}" if f.line is not None else "")
        return f"- **{f.title}** ({where}, {f.severity}): {f.body}"

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
        if stash.dropped:
            body_parts += ["", f"*{stash.dropped} low-confidence finding(s) withheld.*"]
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
                # body-only rather than dropping the run's findings.
                payload["comments"] = []
                payload["body"] += "\n\n### Inline findings (anchoring failed)\n" + "\n".join(
                    self._finding_line(f) for f, _ in inline
                )
                resp = client.post(url, json=payload)
            if resp.status_code >= 300:
                print(
                    f"fuko: agentic review post failed ({resp.status_code}): {resp.text[:300]}",
                    file=sys.stderr,
                )
                return False
        return True
