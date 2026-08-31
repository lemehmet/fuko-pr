"""The agentic review backend: fuko's own reviewer as a drop-in driver.

Where :mod:`sidecar.backends.pragent` drives an external single-shot harness
and re-parses its published markdown, this backend drives
:mod:`sidecar.reviewer` -- an agent with a real checkout and read-only
navigation tools -- and OWNS its output end to end: findings are born as
Review Signals and posted with their markers already attached, so the egress
scrape/PATCH machinery PR-Agent needs does not exist here.

Split of responsibilities across the protocol:

* :meth:`AgenticBackend.invoke` runs the review (checkout, agent, parse) and
  stashes the structured findings in memory. It posts nothing. It is also where
  the per-seat open-findings ledger is carried in and settled
  (:mod:`sidecar.reviewer.ledger`), because both ends of that -- the prompt and
  the parsed verdicts -- exist only here.
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
    HarnessResult,
    check_auth,
    is_auth_failure,
    run_review,
    usage_tokens,
)
from ..reviewer.ledger import DEFAULT_SEAT, CarriedState, carry_in, settle
from ..reviewer.prompt import (
    MAX_FINDINGS,
    AgenticFinding,
    ReviewParseError,
    build_prompt,
    parse_review,
)
from ..signals import ReviewSignal, make_id, with_marker, with_visible_label
from ..throttle import TIMEOUT_RETURNCODE, is_throttle
from .base import ENV_SEAT, InvokeResult, PRRef

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

#: Serialises the failure dump. Concurrent branches are threads in one process
#: sharing one stderr, so an unlocked multi-line dump can interleave with a
#: sibling's and splice a line (fuko-henry, #147).
_DUMP_LOCK = threading.Lock()

#: How much of a failure message reaches ``InvokeResult.detail``. Truncation is
#: from the END, so anything a message needs the reader to see has to fit --
#: named rather than inlined so the reviewer-side budgets that size themselves
#: against it (``sidecar.reviewer.prompt``'s runbook) can be pinned to the same
#: number instead of a copy of it (fuko-henry, #178).
DETAIL_CAP = 460


def _run_costs(result: HarnessResult | None) -> dict:
    """The token/cost fields one harness run contributes to its result (#152).

    Kept as a single mapping splatted into :class:`InvokeResult` so every return
    path -- success, parse failure, timeout, throttle -- carries the accounting
    the same way. ``None`` (a failure that never reached the harness) yields the
    all-unreported mapping, which is the honest value: no run, no bill.
    """
    if result is None:
        return {}
    return {
        **usage_tokens(result.usage),
        "cost_usd": result.cost_usd,
        "turns": result.turns,
    }


def _failure_result(
    verdict: str,
    message: str = "",
    *,
    throttled: bool = False,
    costs: dict | None = None,
) -> InvokeResult:
    """Build EVERY failure return, so the receipt invariants cannot drift apart.

    Three properties, each of which was independently violated by at least one
    hand-rolled failure return in this module before it was centralised
    (fuko-henry, #147):

    * **Verdict-led.** `failed:exit 1` beats prose at telling a crash from a
      timeout from a throttle, and a consumer should not parse English to learn
      which happened.
    * **Flattened.** Every message here is repo- or PR-author-influenced --
      harness stderr, a git error carrying remote output, a parser complaint
      quoting model text -- and `detail` is printed into a log whose gates are
      ^-anchored. An embedded line break would hand chosen text column 0.
    * **No dangling separator.** An empty message publishes the bare verdict,
      never `failed:exit 1: `.

    The returncode is derived from the verdict rather than passed separately:
    they disagreed in an earlier draft, which is exactly the drift this exists
    to prevent.

    ``costs`` (from :func:`_run_costs`) is what the failed run still spent. A
    failure is not a refund -- a run killed at the timeout, or one that walked
    its whole turn budget and then emitted unparseable output, is among the most
    expensive shapes this fleet produces -- so the accounting rides the failure
    returns too, whenever the harness got far enough to report it.
    """
    body = _flatten_for_log(message)[:DETAIL_CAP]
    return InvokeResult(
        returncode=TIMEOUT_RETURNCODE if verdict == "killed:timeout" else 1,
        detail=f"{verdict}: {body}" if body else verdict,
        throttled=throttled,
        channels={_CHANNEL: verdict},
        **(costs or {}),
    )


def _flatten_for_log(value: str) -> str:
    r"""Collapse to ONE physical line: log gates downstream are ^-anchored.

    Flattens using ``splitlines()`` -- the SAME rule the dump splits on -- rather
    than replacing ``\r``/``\n``. Python breaks lines on eight more characters
    than those two (``\x0b``, ``\x0c``, ``\x1c``-``\x1e``, ``\x85``,
    ``\u2028``, ``\u2029``), so a hand-rolled replace leaves a crafted payload
    looking flat here while still splitting downstream -- reopening the column-0
    forgery this exists to close (fuko-henry, #147). Defining "one line" by the
    splitter's own definition makes the two agree by construction.
    """
    return " ".join(value.strip().splitlines())


def _dump_harness_output(model: str, label: str, stderr: str, text: str) -> None:
    """Print the harness's unabridged output on failure, one PREFIXED line at a time.

    Why prefixed rather than dumped verbatim: this content is
    PR-author-influenced (seats grep for strings drawn from the diff, and the
    harness echoes those arguments), and downstream log gates are ^-anchored --
    the runner already flattens newlines out of progress arguments for exactly
    that reason. A raw multi-line dump would hand arbitrary text column 0 of its
    own line and let a crafted diff forge a gate line. Every line therefore
    carries a `fuko: <model> stderr|` / `fuko: <model> final-message|` prefix, so
    nothing from the harness can start a line, while the content stays fully
    readable and greppable. The MODEL is on every line, not just the header,
    because seats run concurrently by design -- two failing branches interleave
    on one stderr and a header-only attribution leaves the mixed lines
    unassignable (fuko-henry, #147).

    Emitted as ONE locked write rather than a sequence of prints: branches are
    threads sharing this stream, and a per-line print can be spliced mid-line by
    a concurrently failing seat -- which would both scramble the attribution and
    let harness content start a spliced line.

    Both channels, because the cause does not always live in stderr. Be precise
    about what the second one IS, though: ``text`` is the harness's LIFTED
    FINAL MESSAGE (the terminal ``result`` event, or the last assistant text on
    schema drift) — NOT the raw NDJSON event feed, which ``_consume_stream``
    folds away and no one retains. So a malformed final message is captured
    here; a mid-stream protocol death still leaves nothing behind. Retaining
    the raw feed is a separate change with its own volume trade-off
    (fuko-henry, #147).
    """
    chunks: list[str] = []
    for stream, body in (("stderr", stderr), ("final-message", text)):
        body = (body or "").rstrip()
        if not body:
            continue
        chunks.append(f"fuko: agentic {model} {label} — full harness {stream} follows")
        chunks.extend(f"fuko: {model} {stream}| {line}" for line in body.splitlines())
        chunks.append(f"fuko: agentic {model} {stream} ends")
    if not chunks:
        return
    # ONE write, under a lock. Branches are threads sharing this stderr, so a
    # per-line `print()` can be spliced mid-line by a concurrently failing seat
    # -- which would defeat the per-line attribution above AND could place
    # harness content at column 0 of a spliced line, reopening the very forgery
    # the prefix exists to prevent (fuko-henry, #147). Composing the whole dump
    # and emitting it under `_DUMP_LOCK` makes interleaving impossible between
    # branches of this process.
    with _DUMP_LOCK:
        sys.stderr.write("\n".join(chunks) + "\n")
        sys.stderr.flush()


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
    # Model-routing overrides: ambient values (e.g. a claude-qwen-style wrapper
    # shell) would silently redirect the harness's main, background-haiku, and
    # subagent calls to whatever model the RUNNER's shell was aimed at. Config
    # decides here too: `_MODEL_ROUTING_VARS` re-injects them for gateway
    # presets only.
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    # Legacy spelling of the haiku-class override — deprecated but still
    # honored by the CLI as a fallback, so an ambient one routes background
    # calls exactly like the var that replaced it.
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    # Context-window override. One ambient value cannot be right for a fleet
    # with more than one agentic seat (qwen at 1M next to glm at 1M is luck,
    # not design) — config decides per entry: build_env derives it from the
    # entry's `max_context` and invoke() re-injects that, never the ambient.
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
)

# Injected in api-key mode when the preset routes to a NON-Anthropic gateway
# (base_url set): Claude Code makes background haiku-class and subagent calls
# with `claude-*` slugs, which a gateway serving another model family has never
# heard of. The main model already travels via `--model`; these cover the calls
# that don't.
_MODEL_ROUTING_VARS = (
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
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
                # A gateway serves its own model family, so the harness's
                # background haiku-class and subagent calls must not ask it for
                # `claude-*` slugs. `small_model` names the gateway's cheap
                # tier; without one the entry's own model does every job.
                small = str(preset.quirks.get("small_model") or model.name)
                env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = small
                env["CLAUDE_CODE_SUBAGENT_MODEL"] = model.name
        # Per-entry context window, BOTH auth modes. The harness sizes its
        # auto-compact window from CLAUDE_CODE_MAX_CONTEXT_TOKENS and refuses
        # a model it does not recognize when the var is unset; a workflow-
        # global export could only ever serve ONE window, which stopped being
        # enough the day a second agentic seat went active. `max_context`
        # already drives the runner's context-fit routing, so this keeps one
        # source of truth for both consumers.
        if model.max_context:
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(model.max_context)
        if model.extra_instructions:
            env[_ENV_INSTRUCTIONS] = model.extra_instructions
        if knowledge:
            env[_ENV_KNOWLEDGE] = knowledge
        return env

    def configured_endpoint(self, preset: ProviderPreset, model: ModelConfig) -> str:
        """The base URL this entry's traffic ACTUALLY goes to, for attribution.

        Subscription auth deliberately injects no base URL -- the runner's own
        logged-in session talks to Anthropic's default endpoint -- so
        attributing the preset's gateway URL to such a run would claim the
        traffic went somewhere it never did (the same substitution class the
        receipt's endpoint field exists to expose). Empty string = the SDK's
        own default endpoint, matching the receipt field's convention.
        """
        if self._resolve_auth(preset, model) == _AUTH_SUBSCRIPTION:
            return ""
        return model.base_url or preset.base_url or ""

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
        Everything else passes through untouched, with ONE exception:
        ``HOME`` is inherited as-is (a subscription login lives there), but in
        **api-key mode** ``CLAUDE_CONFIG_DIR`` is REPLACED with a private
        per-branch directory under this invocation's workdir, so concurrent
        branches -- which are threads in one process -- cannot contend on a
        single ``~/.claude``. The displaced value is forwarded as
        ``FUKO_AMBIENT_CLAUDE_CONFIG_DIR`` purely so the read denylist can
        still cover it. Subscription mode is untouched, because that is where
        its credential lives.

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

        The round is STATEFUL when a review-state store is configured (#156):
        this seat's still-open findings from earlier rounds are rendered into the
        prompt behind their own fence, the verdicts the agent returns on them are
        applied, and this round's published findings become the next round's open
        ledger. All of it is best-effort -- with no store, or an unreachable one,
        every ledger call degrades to a no-op and the prompt is byte-for-byte the
        one this backend built before the ledger existed.

        Every path that reached the harness also carries what the run spent --
        tokens, dollars, turns (#152) -- lifted from the CLI's terminal event by
        :func:`_run_costs`. This is the only backend that can report it today;
        PR-Agent exposes no equivalent feed and leaves the fields null.
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
            return _failure_result(
                "failed:exit 1",
                f"no model name in the harness environment ({_ENV_MODEL} unset or empty)",
            )
        auth = env.get(_ENV_AUTH, _AUTH_SUBSCRIPTION)

        harness_env = {
            k: v
            for k, v in os.environ.items()
            if k not in _GITHUB_CRED_VARS
            and not k.startswith("FUKO_GITHUB_")
            and k not in _ANTHROPIC_INHERITED_VARS
        }
        # Auth-mode-independent: the entry's context window rides along
        # whenever build_env derived one (from `max_context`). The ambient
        # spelling was scrubbed above with the other routing vars — config
        # decides, and a stale workflow-global export must not reach a seat
        # whose entry says otherwise.
        if "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in env:
            harness_env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"]
        # DELIVERY-side receipt (mepro#2012 r2, both gating seats converged):
        # a workflow validator can only prove the CONFIG carries a window;
        # this line is the one place that knows what the spawned harness
        # will actually read, so log it — the absent case especially, since
        # with unknown-model enforcement disabled that seat silently reviews
        # at the harness's ~200k default while its receipt reads done.
        print(
            "fuko: agentic harness %s: CLAUDE_CODE_MAX_CONTEXT_TOKENS=%s"
            % (
                env.get(_ENV_MODEL, "?"),
                harness_env.get(
                    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
                    "ABSENT (harness default ~200k)",
                ),
            ),
            file=sys.stderr,
        )
        if auth == _AUTH_API_KEY:
            for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", *_MODEL_ROUTING_VARS):
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
                return _failure_result(
                    "failed:exit 1",
                    "agentic backend is in subscription auth mode but this "
                    "runner has no logged-in Claude session; run `claude "
                    "setup-token` and export CLAUDE_CODE_OAUTH_TOKEN, or set "
                    "auth = 'api-key' on this model entry",
                )

        try:
            ctx = fetch_pr_context(pr.repo, pr.number, token=token, api_url=api_url)
        except httpx.HTTPError as e:
            return _failure_result("failed:exit 1", f"could not fetch PR context: {e}")
        try:
            checkout = checkout_pr_head(
                pr.repo, pr.number, ctx.head_sha, token=token, server_url=server_url
            )
        except CheckoutError as e:
            return _failure_result("failed:exit 1", str(e))

        # The lane this branch occupies, which is what per-PR review state is
        # keyed by (#156). Absent for a solo config or a laptop run, which is one
        # seat rather than none -- see `DEFAULT_SEAT`.
        seat = env.get(ENV_SEAT, "").strip() or DEFAULT_SEAT
        # Bound before the try so the settle pass below can read it on every path
        # that gets that far; an empty state is exactly what a first round (or an
        # unreachable ledger) carries.
        carried = CarriedState()

        # Everything from here owns the checkout, so every exit path -- including
        # a failure to create the scratch cwd or to build the prompt -- goes
        # through the `finally` that removes it.
        workdir: Path | None = None
        try:
            workdir = Path(mkdtemp(prefix="fuko-agentic-cwd-"))
            # PER-BRANCH CLAUDE STATE DIRECTORY.
            #
            # Concurrent agentic branches are THREADS IN ONE PROCESS
            # (runner.py's ThreadPoolExecutor), and until now every spawned
            # harness inherited the same ambient HOME. Two headless Claude Code
            # processes starting milliseconds apart therefore contended on a
            # single `~/.claude`, and the FIRST-SPAWNED one lost and exited 1.
            #
            # Measured on mepro PR #2064: with two agentic seats configured,
            # exactly one died per round and it was the earlier spawn 3/3 —
            # tracking start order, never the model, endpoint or max_context
            # (one round ran 1000000/1048576, two ran 950000/950000, same
            # one-survivor outcome; the spawns were 15 ms and 70 ms apart).
            # It went unseen until 2026-08-25, when a second seat converted to
            # this driver and gave a round two concurrent harnesses for the
            # first time.
            #
            # Everything else here was already per-branch — the checkout, this
            # workdir, the env dict — so this closes the last shared mutable
            # resource.
            #
            # API-KEY MODE ONLY, deliberately: in subscription mode the login
            # LIVES in HOME/CLAUDE_CONFIG_DIR, so redirecting it at a fresh
            # empty directory would throw the credential away and every
            # subscription branch would fail to authenticate. Those branches
            # keep the shared directory and therefore keep the race; that is
            # the safe trade, and running two concurrent SUBSCRIPTION agentic
            # seats is not a configuration this fleet uses.
            # SECOND-ORDER EFFECTS OF THE OVERRIDE, both deliberate:
            #
            # * The read denylist keys on CLAUDE_CONFIG_DIR, so replacing it
            #   would silently drop the rule protecting the RUNNER's real config
            #   dir — an operator credential store, readable by an agent whose
            #   findings are published verbatim to an untrusted PR author. The
            #   value we displace is therefore handed to the harness under
            #   `FUKO_AMBIENT_CLAUDE_CONFIG_DIR`, which `_permission_settings`
            #   denies alongside the replacement. (`~/.claude` under HOME stays
            #   covered by SENSITIVE_HOME_DIRS regardless; this closes the case
            #   where the runner points CLAUDE_CONFIG_DIR somewhere else.)
            # * `--setting-sources user` resolves against CLAUDE_CONFIG_DIR, so
            #   api-key branches now load NO user-scope settings. That is
            #   acceptable and arguably right for a CI reviewer: every setting
            #   this harness depends on is passed explicitly via `--settings`,
            #   and inheriting an operator's ambient preferences into a
            #   published review was never intended. Verified against Claude
            #   Code with a fresh empty directory: the session runs normally
            #   and populates its own state there.
            if auth == _AUTH_API_KEY:
                branch_config = workdir / "claude-config"
                branch_config.mkdir(parents=True, exist_ok=True)
                ambient_config = harness_env.get("CLAUDE_CONFIG_DIR")
                if ambient_config:
                    harness_env["FUKO_AMBIENT_CLAUDE_CONFIG_DIR"] = ambient_config
                harness_env["CLAUDE_CONFIG_DIR"] = str(branch_config)
            # Read the ledger with the checkout in hand: retiring a finding whose
            # file this head no longer carries needs the tree, and it is the one
            # closure fuko makes without the agent's verdict.
            carried = carry_in(pr.repo, pr.number, seat, str(checkout), ctx.head_sha)
            prompt = build_prompt(
                ctx,
                env.get(_ENV_INSTRUCTIONS, ""),
                checkout_root=str(checkout),
                knowledge=env.get(_ENV_KNOWLEDGE, ""),
                prior_state=carried.text,
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
            return _failure_result("failed:exit 1", str(e))
        except OSError as e:
            return _failure_result("failed:exit 1", f"could not prepare the review sandbox: {e}")
        finally:
            rmtree(checkout, ignore_errors=True)
            if workdir is not None:
                rmtree(workdir, ignore_errors=True)

        if result.returncode != 0:
            output = result.stderr + "\n" + result.text
            # Unabridged stderr for EVERY non-zero exit, before the paths below
            # split on auth/throttle/other. Placing it here rather than in one
            # branch is the point: the auth path truncates harder (300 chars)
            # and is exactly as capable of hiding the real error, and a
            # diagnostic that covers some failures is the kind of half-measure
            # that teaches people to trust an incomplete log (CodeRabbit, #147).
            #
            # Claude Code opens stderr with a benign
            # `[claude-code:unrecognized_model]` warning on any gateway whose
            # model is not in its catalog — every run here — so the head of
            # this buffer is noise and the cap in `detail` never reaches the
            # cause. This buffer is the only copy in the process.
            _dump_harness_output(
                model_name, f"exit {result.returncode}", result.stderr, result.text
            )
            if is_auth_failure(output):
                # Auth is neither a timeout nor a throttle -- failing over would burn
                # the pool on a one-line runner fix -- so the channel is a plain fail.
                auth_tail = _flatten_for_log(result.stderr)[:300]
                return _failure_result(
                    f"failed:exit {result.returncode}",
                    f"agent could not authenticate in {auth} mode"
                    + (f": {auth_tail}" if auth_tail else ""),
                    costs=_run_costs(result),
                )
            # FLATTENED, for the same reason the runner flattens progress
            # arguments (27011698) and the dump prefixes its lines: this text
            # is PR-author-influenced and `detail` is printed into a log whose
            # gates are ^-anchored, so an embedded newline would hand chosen
            # text column 0 of its own line (fuko-henry, #147).
            stderr_tail = _flatten_for_log(result.stderr)[:DETAIL_CAP]
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
            return _failure_result(
                verdict, stderr_tail, throttled=throttled, costs=_run_costs(result)
            )
        try:
            review = parse_review(result.text)
        except ReviewParseError as e:
            # A parse failure is a FAILURE with an exit-0 harness, so it never
            # reached the dump above — yet the malformed output is the entire
            # evidence, and `str(e)[:500]` is a summary of it (fuko-henry,
            # #147). Dump before returning.
            _dump_harness_output(
                model_name, "parse-failure (harness exit 0)", result.stderr, result.text
            )
            # The dump above is the evidence; THIS is the diagnosis, and it has
            # to reach the job log on its own rather than only via `detail` —
            # the caller that prints `detail` is several frames away and an A/B
            # run joins eight of them into one string (#166). Prefixed and
            # flattened like every other author-influenced line here: the
            # message quotes model-written paths and the gates are ^-anchored.
            # Carries the MODEL for the same reason `_dump_harness_output` puts
            # it on every line: seats are threads on one stderr, so two branches
            # failing in the same round -- a correlated bad payload, exactly the
            # incident this runbook is written for -- would otherwise emit two
            # indistinguishable lines and leave "re-run this seat" pointing at
            # no seat (fuko-henry, #178). Written under `_DUMP_LOCK` for the
            # other half of that hazard, which `_dump_harness_output` documents:
            # a bare `print` from one branch can be spliced into another's dump
            # mid-line, which both scrambles the attribution and hands harness
            # content column 0 of the spliced line.
            with _DUMP_LOCK:
                sys.stderr.write(f"fuko: {model_name} {_flatten_for_log(str(e))}\n")
                sys.stderr.flush()
            # Lead with the verdict here too. This path returns
            # `failed:exit 1` on the channel, so a detail opening with parser
            # prose would break the contract the other failure paths keep
            # (CodeRabbit, #147) — and the whole point of that contract is
            # that a reader can tell a crash from a timeout from a throttle
            # without parsing prose.
            return _failure_result("failed:exit 1", str(e), costs=_run_costs(result))

        # Case/whitespace-normalized: `confidence` is deliberately a free-form
        # str so an off-vocabulary value degrades to filtering rather than
        # failing the parse, but that only works if the comparison meets the
        # model where it writes -- "Low" and "LOW" must reach the pressure valve
        # too, or a finding the agent hedged on gets posted as a confident one.
        confident = [f for f in review.findings if f.confidence.strip().lower() != "low"]
        kept = confident[:MAX_FINDINGS]
        # Settle the carried ledger and record this round's own findings (#156).
        # `kept` and not `review.findings`: the ledger carries claims the author
        # was actually shown, so a finding the confidence valve withheld must not
        # re-enter through next round's prior-state section. Best-effort
        # throughout -- with no store this is three no-ops and the round is
        # indistinguishable from a pre-ledger one.
        settlement = settle(
            carried,
            repo=pr.repo,
            pr=pr.number,
            seat=seat,
            head_sha=ctx.head_sha,
            prior_status=review.prior_status,
            findings=kept,
        )
        if carried.rows or settlement.recorded or settlement.reopened:
            print(
                f"fuko: review-state seat {seat} round {carried.round}: carried "
                f"{len(carried.rows)}, closed {settlement.closed}, re-asserted "
                f"{settlement.reasserted}, recorded {settlement.recorded}, "
                f"deduped {len(settlement.deduped)}, "
                f"reopened {len(settlement.reopened)}",
                file=sys.stderr,
            )
            # A re-raise is the one settle outcome that says a PREVIOUS round was
            # wrong -- it closed a claim this round found anyway (#177) -- so it
            # is named on its own line under its own prefix, greppable across a
            # fleet's logs. Flattened for the same reason `deduped` is.
            for claim in settlement.reopened:
                print(
                    f"fuko: review-state seat {seat} re-raised a closed finding: "
                    f"{_flatten_for_log(claim)}",
                    file=sys.stderr,
                )
            # Named, not just counted: a suppressed write is the one settle
            # outcome the store cannot show afterwards, since the surviving row
            # keeps the earlier body. FLATTENED like every other
            # author-influenced value this module logs (#147): the claim is a
            # model's file and title, read out of a contributor-controlled
            # checkout, and an embedded line break would hand that text column 0
            # of its own line in a log whose gates are ^-anchored.
            for claim in settlement.deduped:
                print(
                    f"fuko: review-state seat {seat} re-asserted, not re-recorded: "
                    f"{_flatten_for_log(claim)}",
                    file=sys.stderr,
                )
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
            returncode=0,
            detail=f"{len(kept)} findings",
            channels={_CHANNEL: "done"},
            **_run_costs(result),
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
