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
import random
import sys
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
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

#: The heading every agentic review body opens with.
_REVIEW_HEADER = "## fuko agentic review"

#: The one output channel this backend publishes on today: a single holistic
#: review (a PR-level summary plus inline findings). Named explicitly rather than
#: borrowing PR-Agent's per-tool names (`review`/`improve`/`describe`) because it
#: is a DIFFERENT surface -- the agent produces one combined verdict, not those
#: separable tool outputs. Populating :attr:`InvokeResult.channels` with it (#113)
#: is what lets a COMPLETED agentic receipt state its channel finished rather than
#: leaving an empty map, which :func:`sidecar.status.fuko_states` reads as "not
#: reported" and cannot tell from a dead channel.
_CHANNEL = "agentic-review"


def _review_header(label: str = "") -> str:
    """The review body's opening line, carrying the branch label when there is one.

    The label is what makes the read-back fingerprint BRANCH-specific. In A/B mode
    several branches review the same PR through the same backend and post for the
    same ``commit_id``, so a header alone would let branch A's committed review
    satisfy branch B's read-back -- and branch B would then report its signals as
    posted while its review never reached the PR. Losing a review silently is
    worse than the duplicate the read-back exists to prevent, so the fingerprint
    has to name the branch.
    """
    return f"{_REVIEW_HEADER} — `{label}`" if label else _REVIEW_HEADER


#: Statuses worth another attempt. All of them are treated as AMBIGUOUS about
#: whether the review was committed: 502/504 are gateway failures, so the
#: upstream may have processed the request and only the response was lost. 4xx is
#: deliberately excluded -- a 422 has its own body-only degrade path, and a
#: 401/403 will not improve by repetition.
_TRANSIENT_STATUSES = frozenset({502, 503, 504})

#: Errors that provably never reached the server, so a retry cannot duplicate a
#: committed review. Anything else (a read timeout, a connection broken
#: mid-response) is AMBIGUOUS and goes through the read-back first.
_SAFE_TO_RETRY_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)

#: Attempt count and backoff base. These bound the RETRIES, not the wall clock --
#: an earlier version of this comment claimed the budget "stays far inside
#: `[review].tool_timeout`", which was false twice over: that timeout does not
#: cover this code path at all (it is passed to `run_review()` inside `invoke()`,
#: while posting happens later in `normalize_output()`), and the attempt count
#: alone bounds nothing when each attempt can issue many requests. The actual
#: wall-clock bound is `_POST_DEADLINE_SECONDS`, shared across every request and
#: sleep in the flow. This exists to survive a blip, not to wait out an outage.
_POST_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.5

#: Bound on the read-back pagination walk, so a pathological PR cannot spin.
_READ_BACK_MAX_PAGES = 20

#: Wall-clock ceiling on the ENTIRE posting flow -- every POST attempt, every
#: read-back GET, and every backoff sleep, sharing one deadline.
#:
#: It has to be its own budget because `[review].tool_timeout` does NOT cover
#: this code. That timeout is passed to `run_review()` inside `invoke()`; the
#: runner calls `normalize_output()` afterwards, so the posting flow runs
#: unbounded by it. Without a deadline the worst case is
#: `_POST_ATTEMPTS * (1 POST + _READ_BACK_MAX_PAGES GETs) * _HTTP_TIMEOUT`
#: = 3 * 21 * 60s ~= 63 minutes -- long enough to blow the CI job's
#: `timeout-minutes` and get the run killed mid-review, which is the starved
#: round that reads as a clean one. The resilience path must not become the
#: thing that kills the job it was added to protect.
_POST_DEADLINE_SECONDS = 120.0

#: Per-request ceiling. Deliberately below the whole-flow deadline so one hung
#: request cannot consume the entire budget on its own.
_HTTP_TIMEOUT = 30.0


def _remaining_timeout(deadline: float) -> float:
    """Per-request timeout clamped to what is left of the shared budget.

    Never returns zero or negative: the callers already refuse to start a request
    once the deadline has passed, and this only guards the sliver between that
    check and the call, where a non-positive timeout would be an error rather
    than a fast failure.
    """
    return max(0.001, min(_HTTP_TIMEOUT, deadline - time.monotonic()))


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, in seconds, for retry ``attempt``."""
    return _BACKOFF_BASE_SECONDS * (2**attempt) * (0.5 + random.random() / 2)


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


#: How many completed-but-unclaimed reviews to retain. Generous next to any real
#: fleet (a PR runs one branch per active model), small enough that a leak stays
#: a bounded one.
_MAX_PENDING = 32


def _identity(token: str) -> str:
    """Fingerprint the branch's posting token, to key its stash without holding it.

    A hash, not the token: this value lives in a dict key that could surface in
    a traceback or a debugger, and a GitHub App token there would be a
    credential in a log. Truncation is fine -- this only has to separate the
    handful of branches on one PR, not resist collision attacks.
    """
    return sha256(token.encode()).hexdigest()[:12] if token else ""


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
    #: (path, new-side line) the API will anchor; empty means "unknown, fall back".
    diff_positions: frozenset[tuple[str, int]] = frozenset()
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
        # Keyed (pr_url, model, identity): two [[review.models]] entries may
        # legally share a provider/name and differ only by `token_env` (the same
        # model run under two App identities) -- nothing validates uniqueness --
        # so a (pr_url, model) key collides in concurrent A/B and one branch's
        # invoke silently overwrites the other's stash. The identity component
        # is a fingerprint of the posting token, which is exactly what makes the
        # branches distinct.
        #
        # Bounded: entries leave on a successful claim, but a branch that dies
        # between invoke and normalize_output, or a claim that misses on an id
        # we did not anticipate, would otherwise pin its findings for the life
        # of the sidecar process. Insertion order is arrival order, so the
        # oldest unclaimed entry is the one to shed.
        self._pending: dict[tuple[str, str, str], _PendingReview] = {}
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

        Every return path sets :attr:`InvokeResult.channels` for the single
        :data:`_CHANNEL` this backend publishes (#113): ``done`` on success, and
        the pr-agent driver's own failure vocabulary otherwise (``killed:timeout``
        / ``throttled:exit N`` / ``failed:exit N``). A COMPLETED run must state its
        channel finished rather than leave an empty map -- an empty map is what
        :func:`sidecar.status.fuko_states` reads as "not reported", which it cannot
        tell from a dead channel, so a ``done`` receipt with no channels would read
        as a clean pass even if the one channel had failed.
        """
        # Must stay in lockstep with normalize_output's fallback: this value
        # fingerprints the stash key, so any divergence makes a completed review
        # unclaimable. Branch env first (the runner sets it per identity), then
        # the same two process vars egress falls back to.
        token = (
            env.get(_ENV_GH_TOKEN)
            or os.environ.get(_ENV_GH_TOKEN)
            or os.environ.get("GITHUB_TOKEN", "")
        )
        api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        model_name = env.get(_ENV_MODEL, "")
        if not model_name:
            # An empty --model would reach the CLI as a confusing runtime error
            # after a full clone; say what is actually wrong, before paying for it.
            return InvokeResult(
                returncode=1,
                detail=f"no model name in the harness environment ({_ENV_MODEL} unset or empty)",
                channels={_CHANNEL: "failed:exit 1"},
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
                    channels={_CHANNEL: "failed:exit 1"},
                )

        try:
            ctx = fetch_pr_context(pr.repo, pr.number, token=token, api_url=api_url)
        except httpx.HTTPError as e:
            return InvokeResult(
                returncode=1,
                detail=f"could not fetch PR context: {e}",
                channels={_CHANNEL: "failed:exit 1"},
            )
        try:
            checkout = checkout_pr_head(
                pr.repo, pr.number, ctx.head_sha, token=token, server_url=server_url
            )
        except CheckoutError as e:
            return InvokeResult(returncode=1, detail=str(e), channels={_CHANNEL: "failed:exit 1"})

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
            return InvokeResult(returncode=1, detail=str(e), channels={_CHANNEL: "failed:exit 1"})
        except OSError as e:
            return InvokeResult(
                returncode=1,
                detail=f"could not prepare the review sandbox: {e}",
                channels={_CHANNEL: "failed:exit 1"},
            )
        finally:
            rmtree(checkout, ignore_errors=True)
            if workdir is not None:
                rmtree(workdir, ignore_errors=True)

        if result.returncode != 0:
            output = result.stderr + "\n" + result.text
            if is_auth_failure(output):
                # Auth is neither a timeout nor a throttle -- failing over would burn
                # the pool on a one-line runner fix -- so the channel is a plain fail.
                return InvokeResult(
                    returncode=result.returncode,
                    detail=(
                        f"agent could not authenticate in {auth} mode: "
                        f"{result.stderr.strip()[:300]}"
                    ),
                    channels={_CHANNEL: f"failed:exit {result.returncode}"},
                )
            throttled = is_throttle(result.returncode, output)
            # Same vocabulary the pr-agent driver records per tool, so a consumer
            # reads one channel map regardless of backend: a hung run is
            # `killed:timeout`, a 429/overload is `throttled:exit N`, anything else
            # is `failed:exit N`.
            if result.timed_out:
                verdict = "killed:timeout"
            elif throttled:
                verdict = f"throttled:exit {result.returncode}"
            else:
                verdict = f"failed:exit {result.returncode}"
            return InvokeResult(
                returncode=result.returncode,
                detail=result.stderr.strip()[:500] or "agent run failed",
                throttled=throttled,
                channels={_CHANNEL: verdict},
            )
        try:
            review = parse_review(result.text)
        except ReviewParseError as e:
            return InvokeResult(
                returncode=1, detail=str(e)[:500], channels={_CHANNEL: "failed:exit 1"}
            )

        # Case/whitespace-normalized: `confidence` is deliberately a free-form
        # str so an off-vocabulary value degrades to filtering rather than
        # failing the parse, but that only works if the comparison meets the
        # model where it writes -- "Low" and "LOW" must reach the pressure valve
        # too, or a finding the agent hedged on gets posted as a confident one.
        confident = [f for f in review.findings if f.confidence.strip().lower() != "low"]
        kept = confident[:MAX_FINDINGS]
        with self._lock:
            key = (pr.url, model_name, _identity(token))
            while len(self._pending) >= _MAX_PENDING and key not in self._pending:
                # dicts preserve insertion order, so the first key is the oldest
                # unclaimed review -- the one whose branch is least likely to
                # still be coming back for it.
                stale = next(iter(self._pending))
                del self._pending[stale]
                print(
                    f"fuko: dropped an unclaimed agentic review for {stale[1]} "
                    f"(pending cap {_MAX_PENDING} reached)",
                    file=sys.stderr,
                )
            self._pending[key] = _PendingReview(
                findings=kept,
                summary=review.summary,
                head_sha=ctx.head_sha,
                diff_files=ctx.diff_files,
                diff_positions=ctx.diff_positions,
                withheld_low=len(review.findings) - len(confident),
                over_cap=len(confident) - len(kept),
            )
        # A COMPLETED run states its channel finished, rather than leaving an empty
        # map that `fuko_states` cannot tell from a dead channel (#113). This is the
        # one path where the difference bites: a receipt finalized `done` with no
        # channels reads as a clean pass even if the channel had in fact failed.
        return InvokeResult(
            returncode=0, detail=f"{len(kept)} findings", channels={_CHANNEL: "done"}
        )

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
        # Resolve the token FIRST: it fingerprints the branch, and the stash is
        # keyed by that fingerprint (see __init__), so it is an input to the
        # claim rather than something needed only for posting.
        if token is None:
            # Same precedence invoke uses. A runner that supplies only
            # GITHUB__USER_TOKEN would otherwise fingerprint differently here
            # than at stash time, and the review would be silently unclaimable.
            token = os.environ.get(_ENV_GH_TOKEN) or os.environ.get("GITHUB_TOKEN", "")
        if api_url is None:
            api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        stash = self._claim(pr.url, model, _identity(token))
        if stash is None:
            return []

        signals: list[ReviewSignal] = []
        inline: list[tuple[AgenticFinding, dict]] = []
        overflow: list[tuple[AgenticFinding, str]] = []
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
            # Decorate BEFORE choosing inline vs body. The marker is what makes a
            # finding recoverable as a Review Signal, and the body is not the rare
            # path here: findings about callers, cleanup paths and missing tests
            # legitimately land outside the diff, which is exactly the class this
            # reviewer exists to produce.
            if compare_label is not None:
                body = with_visible_label(body, compare_label, signal)
            else:
                body = with_marker(body, signal)
            if f.line is not None and self._anchorable(stash, f):
                comment = {"path": f.file, "line": f.line, "side": "RIGHT", "body": body}
                if f.end_line is not None and f.end_line > f.line:
                    comment.update(
                        {"start_line": f.line, "start_side": "RIGHT", "line": f.end_line}
                    )
                inline.append((f, comment))
            else:
                overflow.append((f, body))

        posted = self._post_review(
            pr, token, api_url, stash, inline, overflow, label=compare_label or model
        )
        return signals if posted else []

    def _claim(self, pr_url: str, model: str, identity: str = "") -> _PendingReview | None:
        """Pop the pending review for this PR, model and branch identity.

        ``invoke`` keys the stash by the bare harness model name while the
        runner hands egress the litellm-prefixed id, so the prefixed spelling
        is normalized before lookup.

        The last-resort claim is deliberately NOT "the only pending entry for
        this PR": in A/B mode several branches review the same PR through the
        same backend instance, so that rule lets whichever branch normalizes
        first walk off with another model's findings and post them under its own
        identity and role. The fallback therefore requires the remaining entry to
        be the *same model under a different spelling*: the bare names (after
        dropping any provider prefix) must match EXACTLY. A suffix match looks
        equivalent and is not -- two genuinely different models can end in each
        other (``sonnet-4`` and ``claude-sonnet-4`` are the same model, but the
        rule does not know that, and nothing stops a pair where they differ) --
        while exact bare equality already covers the litellm-prefix drift this
        fallback exists for, since both sides are reduced the same way.

        ``identity`` narrows every lookup to the branch that produced the stash,
        so two entries sharing a provider/name and differing only by
        ``token_env`` cannot claim each other's findings.
        """
        bare = model.rsplit("/", 1)[-1]
        with self._lock:
            for key in ((pr_url, bare, identity), (pr_url, model, identity)):
                if key in self._pending:
                    return self._pending.pop(key)
            mine = [k for k in self._pending if k[0] == pr_url and k[2] == identity]
            if len(mine) == 1:
                stashed = mine[0][1]
                stashed_bare = stashed.rsplit("/", 1)[-1]
                if bare and bare == stashed_bare:
                    return self._pending.pop(mine[0])
        return None

    @staticmethod
    def _anchorable(stash: _PendingReview, f: AgenticFinding) -> bool:
        """Whether GitHub will accept ``f`` as an inline comment on this diff.

        File membership is too weak: the API rejects any line outside a hunk,
        and ONE rejected comment 422s the whole review, so a single hallucinated
        line number would demote every other finding to the review body. Check
        the actual hunk positions instead, and check the end line too, since a
        multi-line comment must span anchorable lines.

        Falls back to file membership when positions are empty -- an unparsed
        diff should not silently send every finding to the body.
        """
        if not stash.diff_positions:
            return f.file in stash.diff_files
        if (f.file, f.line) not in stash.diff_positions:
            return False
        if f.end_line is not None and f.end_line > (f.line or 0):
            return (f.file, f.end_line) in stash.diff_positions
        return True

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
        overflow: list[tuple[AgenticFinding, str]],
        label: str = "",
    ) -> bool:
        """POST one PR review; retry body-only on a 422; report success."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = "Bearer " + token
        body_parts = [_review_header(label), "", stash.summary or "(no summary)"]
        if overflow:
            body_parts += ["", "### Findings without a diff anchor", ""]
            body_parts += [
                "\n\n---\n\n".join(
                    f"**Location:** `{f.file}`"
                    + (f":{f.line}" if f.line is not None else "")
                    + f" ({f.severity})\n\n{body}"
                    for f, body in overflow
                )
            ]
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
        try:
            return self._post(url, headers, payload, inline, label)
        except httpx.HTTPError as e:
            # A transport failure that survived the retry budget. Report it as a
            # failed post so normalize_output returns no signals rather than
            # phantom findings.
            print(f"fuko: agentic review post failed (transport): {e}", file=sys.stderr)
            return False

    def _review_already_posted(
        self, client: httpx.Client, url: str, payload: dict, label: str = "", deadline: float = 0.0
    ) -> bool | None:
        """Whether OUR review for this commit is already on the PR.

        Tri-state on purpose. ``True``/``False`` are answers; ``None`` means the
        question could not be answered -- the read-back errored, ran out of pages,
        or ran out of deadline. ``None`` is NOT folded into either answer, because
        the two failure directions are both real: answering "yes" would report a
        review we never confirmed, and answering "no" would retry into a duplicate.
        The caller declines to do either.

        The read-back that makes a retry safe. ``POST /pulls/{n}/reviews`` is NOT
        idempotent, and the failure a retry exists to fix -- a response lost after
        the server committed -- is exactly the one where you cannot tell whether
        the first attempt landed. Retrying blindly would duplicate every inline
        comment and every marker, which downstream signal extraction would then
        read as two findings where there is one (#103).

        Matching is on ``commit_id`` plus our own review header, so it cannot
        mistake another reviewer's review (or our own review of an earlier HEAD)
        for this one. Costs one request, and only on the failure path.
        """
        head = payload.get("commit_id")
        # The fingerprint is the FULL header including this branch's label, so a
        # sibling A/B branch's review for the same commit cannot satisfy it.
        fingerprint = _review_header(label)
        page = 1
        try:
            while page <= _READ_BACK_MAX_PAGES:
                if time.monotonic() >= deadline:
                    print(
                        "fuko: review read-back ran out of its posting deadline",
                        file=sys.stderr,
                    )
                    return None
                resp = client.get(
                    url,
                    params={"per_page": 100, "page": page},
                    # Clamp to what is left: a request we cannot afford to finish
                    # must not be issued with a timeout that outlives the budget.
                    timeout=_remaining_timeout(deadline),
                )
                resp.raise_for_status()
                batch = resp.json()
                if not isinstance(batch, list) or not batch:
                    return False
                if any(
                    r.get("commit_id") == head and fingerprint in (r.get("body") or "")
                    for r in batch
                    if isinstance(r, dict)
                ):
                    return True
                # Reviews come back oldest-first, so ours -- the newest -- is on
                # the LAST page. Reading only page 1 of a busy PR would miss it
                # and retry into a duplicate, which is the outcome this guards.
                if len(batch) < 100:
                    return False
                page += 1
        except (httpx.HTTPError, ValueError) as e:
            print(f"fuko: could not read back agentic review state: {e}", file=sys.stderr)
            return None
        print("fuko: review read-back exceeded its page budget", file=sys.stderr)
        return None

    def _send_review(
        self,
        client: httpx.Client,
        url: str,
        payload: dict,
        label: str = "",
        deadline: float = 0.0,
    ) -> httpx.Response | None:
        """POST the review with a bounded retry for transient failures only.

        Only ONE class is retried blind: connection errors that provably never
        reached the server, so nothing can have been committed. Everything else
        that is retried at all -- 502/503/504, a read timeout, a connection broken
        mid-response -- is AMBIGUOUS about whether the review landed, and consults
        :meth:`_review_already_posted` first. A gateway status is not evidence the
        request was refused; it is evidence the response did not come back.

        Not retried: any other 4xx. A 422 has its own body-only degrade path and a
        401/403 will not improve by repetition.

        ``deadline`` is a monotonic timestamp shared with the read-back, so POSTs,
        GETs and backoff all draw on ONE budget.

        Returns ``None`` when the review is confirmed already on the PR, so the
        caller reports success without posting twice. Raises when the outcome
        cannot be established -- an unverified review must not be reported as
        posted.
        """

        def _unverified(reason: str) -> None:
            raise httpx.HTTPError(f"review post outcome could not be established: {reason}")

        last = _POST_ATTEMPTS - 1
        for attempt in range(_POST_ATTEMPTS):
            if time.monotonic() >= deadline:
                _unverified("posting deadline exhausted")
            try:
                resp = client.post(url, json=payload, timeout=_remaining_timeout(deadline))
            except _SAFE_TO_RETRY_ERRORS as e:
                # Never reached the server: nothing can have been committed.
                if attempt == last:
                    raise
                print(
                    f"fuko: review post attempt {attempt + 1} failed ({e}); retrying",
                    file=sys.stderr,
                )
            except httpx.HTTPError as e:
                landed = self._review_already_posted(client, url, payload, label, deadline)
                if landed is True:
                    print(
                        f"fuko: review post response lost ({e}) but the review is "
                        "on the PR; not re-posting",
                        file=sys.stderr,
                    )
                    return None
                if landed is None:
                    # Could not determine. Retrying risks a duplicate; claiming
                    # success would report a review we never confirmed. Do
                    # neither -- the branch fails, which reads as NOT done.
                    _unverified(f"read-back inconclusive after {e}")
                if attempt == last:
                    raise
                print(
                    f"fuko: review post attempt {attempt + 1} lost ({e}); retrying", file=sys.stderr
                )
            else:
                if resp.status_code in _TRANSIENT_STATUSES:
                    # AMBIGUOUS, not safe. A 502/504 is a GATEWAY failure: the
                    # upstream may have committed the review and only the
                    # response was lost, which is indistinguishable from a
                    # request that never got there. 503 is treated the same way
                    # rather than reasoned about per-status -- one rule beats
                    # three fragile ones, and the read-back costs a request only
                    # on a path that has already failed.
                    landed = self._review_already_posted(client, url, payload, label, deadline)
                    if landed is True:
                        print(
                            f"fuko: review post got {resp.status_code} but the review "
                            "is on the PR; not re-posting",
                            file=sys.stderr,
                        )
                        return None
                    if landed is None:
                        _unverified(f"read-back inconclusive after {resp.status_code}")
                    if attempt == last:
                        return resp
                    print(
                        f"fuko: review post attempt {attempt + 1} got {resp.status_code}; retrying",
                        file=sys.stderr,
                    )
                else:
                    return resp
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _unverified("posting deadline exhausted")
            time.sleep(min(_backoff_delay(attempt), remaining))
        raise httpx.HTTPError("review post retries exhausted")  # pragma: no cover

    def _post(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
        inline: list[tuple[AgenticFinding, dict]],
        label: str = "",
    ) -> bool:
        """Issue the review POST, with the one body-only retry; report success."""
        deadline = time.monotonic() + _POST_DEADLINE_SECONDS
        with httpx.Client(timeout=_HTTP_TIMEOUT, headers=headers) as client:
            resp = self._send_review(client, url, payload, label, deadline)
            if resp is None:
                return True
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
                resp = self._send_review(client, url, payload, label, deadline)
                if resp is None:
                    return True
            if resp.status_code >= 300:
                print(
                    f"fuko: agentic review post failed ({resp.status_code}): {resp.text[:300]}",
                    file=sys.stderr,
                )
                return False
        return True
