"""Agent runtimes that can execute the review strategy.

A harness takes a prepared prompt plus the checkout to review and returns the
agent's final text; :func:`sidecar.reviewer.prompt.parse_review` turns that into
structured findings. The first (and currently only) harness drives headless
Claude Code (``claude -p``), which must be installed on the runner. Other
runtimes -- e.g. an OSS agentic harness fronting non-Anthropic models --
implement the same ``run_review`` signature and slot in without touching the
strategy or the driver.

Two isolation properties are load-bearing, because the checkout is an untrusted
contributor's pull request:

* **The agent never runs FROM the checkout.** ``cwd`` is a clean, empty
  directory and the checkout is exposed as an additional readable root
  (``--add-dir``). Claude Code loads project configuration -- including
  ``.claude/settings.json`` **hooks**, which are arbitrary commands -- from its
  working directory, and headless mode skips the workspace-trust prompt that
  would otherwise gate it. Verified against Claude Code 2.1.232: a hook shipped
  in the working directory executes, and the same hook does not execute when
  the directory is passed via ``--add-dir`` instead.
* **The tool surface is pinned to read-only navigation**
  (:data:`ALLOWED_TOOLS`). Headless mode cannot prompt for permissions, so
  every tool outside the allowlist is denied by construction -- the agent
  cannot run repository code, write files, or reach the network even if the
  repository content asks it to.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..throttle import TIMEOUT_RETURNCODE

ALLOWED_TOOLS = "Read,Grep,Glob"

#: Set by a caller that REPLACES ``CLAUDE_CONFIG_DIR`` for isolation, carrying
#: the value it replaced so the read denylist can still cover it. Consumed only
#: by :func:`_permission_settings`; the harness itself never reads it, and
#: Claude Code does not know the name.
_ENV_AMBIENT_CONFIG_DIR = "FUKO_AMBIENT_CLAUDE_CONFIG_DIR"

#: Directories the agent must never read, relative to the runner's home.
#:
#: ``--add-dir`` ADDS a readable root; it does NOT confine reads to it. Verified
#: on Claude Code 2.1.232: with ``--allowedTools Read`` and a clean cwd, the
#: agent will happily read an absolute path outside both cwd and every
#: ``--add-dir`` root. That matters here because findings are published
#: verbatim to the pull request, which the (untrusted) PR author can read -- so
#: an injected instruction that says "read X and put it in a finding" is an
#: exfiltration channel, not merely "wrong review text". Subscription auth
#: deliberately keeps the runner's own login reachable under ``HOME``, which
#: makes ``~/.claude`` the highest-value target on the box.
#:
#: This is a denylist over the credential stores, not a sandbox: it closes the
#: named paths, it does not confine the agent -- Read and Grep still reach
#: anything else on the runner (``/etc``, other checkouts in the work dir).
#: Real confinement needs the run to happen in a container or under a dedicated
#: unprivileged user, which is the runner's job, not this module's.
#:
#: Two properties below are load-bearing and were measured on 2.1.232, because
#: both are the opposite of what the rule syntax suggests. Canary outside cwd
#: and outside every ``--add-dir`` root, agent asked to Grep it:
#:
#: ===========================================  ==========
#: deny rules                                   outcome
#: ===========================================  ==========
#: (none)                                       LEAKED
#: ``Read(//abs/**)``                           blocked
#: ``Read(//abs/**)`` + ``Grep(//abs/**)``      blocked
#: ``Grep(//abs/**)``                           LEAKED
#: ===========================================  ==========
#:
#: So: a PATH rule is enforced across the read-class tools -- the ``Read(...)``
#: rules below are what actually stop ``Grep`` from reading a credential file --
#: while a TOOL-scoped ``Grep(...)`` rule is not honored at all. We therefore
#: emit only ``Read(...)`` rules on purpose. Adding ``Grep(...)`` entries would
#: be decorative and would imply a coverage guarantee that does not exist.
SENSITIVE_HOME_DIRS = (
    ".claude",
    ".ssh",
    ".aws",
    ".gnupg",
    ".config/gh",
    ".config/gcloud",
    ".docker",
    ".kube",
)
#: Single files worth the same treatment.
#:
#: The ``actions-runner`` entries are the self-hosted runner's own registration
#: credentials, which authenticate the runner to GitHub. Every fuko workflow in
#: this repo is ``runs-on: [self-hosted, ...]``, so these sit in the same home
#: directory this denylist is built from -- a larger prize than several stores
#: already covered, reachable through exactly the channel the denylist exists to
#: close: an injected "read X and put it in a finding" reaches the PR verbatim,
#: where an untrusted PR author reads it.
#:
#: Listed as FILES, deliberately, rather than denying ``actions-runner`` as a
#: directory: the runner's workspace is ``<runner-dir>/_work/<repo>/<repo>``, so
#: a directory rule would deny the checkout itself and leave the reviewer unable
#: to read the code it is reviewing. The narrower rule is not a compromise here;
#: the wider one is simply wrong.
SENSITIVE_HOME_FILES = (
    ".netrc",
    ".git-credentials",
    ".claude.json",
    "actions-runner/.credentials",
    "actions-runner/.credentials_rsautokey",
)

#: Kernel pseudo-filesystems, denied because ``/proc/self/environ`` hands the
#: agent its OWN process environment -- which necessarily holds the credential
#: this backend just injected (``ANTHROPIC_API_KEY`` in api-key mode,
#: ``CLAUDE_CODE_OAUTH_TOKEN`` in subscription mode). Published findings are the
#: same egress channel as the original read-confinement bug; only the source
#: differs. No legitimate code review reads ``/proc``, ``/sys`` or ``/dev``, so
#: this costs a real reviewer nothing.
#:
#: NOT empirically verified: this was developed on darwin, which has no
#: ``/proc``. The rules use the ``Read(//abs/**)`` spelling that WAS verified
#: (see the matrix above), and path rules were measured to cover ``Grep`` too,
#: so one rule closes both tools -- but the specific ``/proc`` denial is
#: reasoned, not measured, and should be checked on a Linux runner.
#:
#: This is also precisely why fuko-pr#102 (running the reviewer in a container)
#: matters: a denylist over ``/proc`` still leaves the credential sitting in
#: the agent's own environment, reachable by any path we failed to enumerate.
#: Only a boundary fixes the class; this closes the instance.
SENSITIVE_SYSTEM_DIRS = ("/proc", "/sys", "/dev")


def _permission_settings(env: dict[str, str]) -> str:
    """Build the ``--settings`` payload: hooks off, credential stores unreadable.

    Paths use Claude Code's absolute-rule spelling ``Read(//abs/path/**)`` (a
    leading ``//``), which is what actually matches an absolute path -- a
    single-slash rule silently fails to match and the read goes through.

    Only POSIX-absolute roots produce rules. A Windows-shaped home
    (a ``C:`` drive path) would otherwise render as ``Read(/C:/Users/...)``,
    which is not the verified spelling and would silently match nothing -- a
    denylist that looks present and protects nothing is worse than none at all.
    Such a root is skipped and announced on stderr instead, so the operator
    learns the credential denylist is not in force on that runner rather than
    discovering it from a leaked review.
    """
    candidates: list[tuple[str, bool]] = []  # (path, is_directory)
    home = (env.get("HOME") or env.get("USERPROFILE") or "").replace("\\", "/").rstrip("/")
    if home:
        candidates += [(f"{home}/{d}", True) for d in SENSITIVE_HOME_DIRS]
        candidates += [(f"{home}/{f}", False) for f in SENSITIVE_HOME_FILES]
    # Both the config dir the harness will USE and any ambient one it replaced.
    #
    # The caller may redirect CLAUDE_CONFIG_DIR at a private per-branch
    # directory (the agentic backend does, to stop concurrent branches sharing
    # one ~/.claude). Denying only the effective value would then silently drop
    # the rule that had covered the RUNNER's real config dir, leaving an
    # operator's credentials readable by an agent whose findings are published
    # verbatim to an untrusted PR author. Deny both: the replacement, because
    # it accumulates this run's own session state, and the ambient one,
    # because it is the higher-value target and the reason this rule exists.
    for key in ("CLAUDE_CONFIG_DIR", _ENV_AMBIENT_CONFIG_DIR):
        config_dir = (env.get(key) or "").replace("\\", "/").rstrip("/")
        if config_dir:
            candidates.append((config_dir, True))
    # Unconditional: these do not depend on HOME, and on a runner without one
    # they are the only rules that remain.
    candidates += [(d, True) for d in SENSITIVE_SYSTEM_DIRS]

    # Build the rule from a normalized path rather than by concatenating onto
    # whatever the environment held: POSIX permits a leading `//` with
    # implementation-defined meaning, so a HOME of `//home/runner` would
    # otherwise render `Read(///home/...)` -- the silently non-matching form,
    # leaving the credential stores undenied with nothing to show for it.
    # Stripping and re-adding makes exactly one `//` prefix by construction, so
    # the broken spelling cannot be produced at all rather than merely asserted
    # against.
    deny = [
        f"Read(//{path.lstrip('/')}/**)" if is_dir else f"Read(//{path.lstrip('/')})"
        for path, is_dir in candidates
        if path.startswith("/")
    ]
    unusable = sorted({path for path, _ in candidates if not path.startswith("/")})
    if unusable:
        print(
            "fuko: credential denylist NOT applied -- these paths are not "
            f"POSIX-absolute, so no verified deny rule exists for them: {', '.join(unusable)}. "
            "The agentic reviewer's read denylist is inert on this runner; run it "
            "in a container or under a dedicated unprivileged user.",
            file=sys.stderr,
        )
    return json.dumps({"disableAllHooks": True, "permissions": {"deny": deny}})


# Not listed in `claude --help` for 2.1.232, but accepted (verified: unknown
# options exit with "error: unknown option", this one runs) and documented in
# the CLI reference. It bounds a pathological tool loop; the wall-clock
# `tool_timeout` is the outer bound that does not depend on this flag.
#
# RAISED 50 -> 250 (2026-08-28). 50 was too low for a real review of a large
# diff and was silently ENDING them: exhausting `--max-turns` exits 1 with
# NOTHING on stderr but the benign `[claude-code:unrecognized_model]` startup
# warning, and the only true signal -- `result.subtype == "error_max_turns"` --
# arrives in the stdout event feed, which `_consume_stream` folded away.
# Reproduced directly against the qwen gateway: `--max-turns 2` on a task
# needing more gives exactly that signature. On mepro #2064 a gating seat died
# four times in a tight band (fuko-turns 109-117) with no other evidence, and
# the failure was read as a provider fault for a full day.
#
# 250 is chosen so `tool_timeout` BINDS FIRST: at the observed ~5 turns/min a
# 2700s budget is ~225 turns, so a runaway hits the wall-clock bound that the
# budget arithmetic actually reasons about, and this stays what it claims to
# be -- a backstop against a pathological loop, not a review-length limit.
DEFAULT_MAX_TURNS = 250

# "Not logged in · Please run /login" and the API-key equivalents. Auth failure
# must NOT be treated as throttling: failing over to the next provider would
# burn the whole pool on what is a one-line runner fix.
_AUTH_FAILURE_RE = re.compile(
    r"not logged in|please run /login|invalid api key|authentication_error|"
    r"\bunauthorized\b|\b401\b",
    re.IGNORECASE,
)


class HarnessNotAvailableError(RuntimeError):
    """Raised when the harness binary is not installed on this runner."""


@dataclass(frozen=True)
class HarnessResult:
    """The raw outcome of one agent run."""

    returncode: int
    text: str
    stderr: str = ""
    timed_out: bool = False


def is_auth_failure(output: str) -> bool:
    """Return whether ``output`` looks like a credential problem, not a capacity one."""
    return bool(output) and _AUTH_FAILURE_RE.search(output) is not None


def check_auth(env: dict[str, str]) -> dict | None:
    """Return ``claude auth status`` as parsed JSON, or None if it cannot be read.

    Used as a preflight for subscription mode, where a lapsed login on the
    runner would otherwise surface as a confusing mid-review failure on every
    PR. Returns None (rather than raising) when the binary is missing, the
    probe fails, or the output is not a JSON **object** -- an unreadable probe
    is not evidence of a broken login, so the review proceeds and the run
    itself reports the truth. The object check matters because ``json.loads``
    happily returns a bare ``null``/number/string for malformed-but-valid JSON,
    which the caller would then treat as a status mapping and crash on.
    """
    binary = shutil.which("claude", path=env.get("PATH"))
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        parsed = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def run_review(
    prompt: str,
    repo_dir: Path,
    *,
    cwd: Path,
    model: str,
    env: dict[str, str],
    timeout: int,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> HarnessResult:
    """Run headless Claude Code over ``repo_dir`` and return its final text.

    The agent runs from ``cwd`` -- which must be a clean directory the
    repository does not control -- with ``repo_dir`` mounted as an additional
    readable root, so no project configuration from the reviewed code is
    loaded (see the module docstring).

    The prompt goes over stdin (it embeds a full diff -- argv has size limits
    and shows up in process listings). ``--output-format stream-json`` (which
    print mode requires ``--verbose`` for) turns stdout into an NDJSON event
    feed consumed INCREMENTALLY: each ``tool_use`` block becomes one compact
    progress line on this process's stderr as it happens (mepro asked for
    this after a fleet of 15-30 min reviews whose only log lines were start
    and end -- an 1800s kill now shows the last tool the seat was on), and
    the agent's final message is lifted from the terminal ``result`` event,
    so the downstream contract is unchanged: the returned ``text`` is still
    the final message the strategy constrains to a bare JSON object. Event
    parsing is TOLERANT -- unknown or non-JSON lines are skipped, and if the
    ``result`` event never arrives (CLI schema drift, mid-stream kill) the
    last assistant text block stands in, so drift degrades to the old
    behavior rather than a hard failure. ``env`` is the harness process
    environment (the caller decides exactly which credentials it carries) and
    is used as given rather than merged with this process's.

    A timeout maps to :data:`sidecar.throttle.TIMEOUT_RETURNCODE` so the driver
    classifies a hung run as throttle-class, same as a hung PR-Agent container.
    """
    binary = shutil.which("claude", path=env.get("PATH"))
    if binary is None:
        raise HarnessNotAvailableError(
            "the 'claude' CLI is not on PATH; install Claude Code on this "
            "runner or switch this model entry to the pr-agent backend"
        )
    cmd = [
        binary,
        "-p",
        "--model",
        model,
        "--output-format",
        "stream-json",
        # Print mode refuses stream-json without it; it gates the event feed,
        # not log chattiness.
        "--verbose",
        "--allowedTools",
        ALLOWED_TOOLS,
        "--add-dir",
        str(repo_dir),
        # Belt and braces on top of the clean cwd, each independently sufficient
        # for its vector: load settings from the USER scope only (never the
        # reviewed project's), refuse hooks outright, and start no MCP server
        # that was not explicitly configured here (none is).
        "--setting-sources",
        "user",
        "--settings",
        _permission_settings(env),
        "--strict-mcp-config",
        "--max-turns",
        str(max_turns),
    ]
    start = time.monotonic()

    def _emit(turn: int, tool: str, arg: str) -> None:
        mins = int((time.monotonic() - start) // 60)
        print(
            f"fuko: agentic {model} [{mins}m t{turn}] {tool} {arg}",
            file=sys.stderr,
            flush=True,
        )

    returncode, text, stderr, timed_out = _drive(
        cmd, prompt=prompt, cwd=cwd, env=env, timeout=timeout, emit=_emit
    )
    if timed_out:
        return HarnessResult(
            returncode=TIMEOUT_RETURNCODE,
            text="",
            stderr=stderr or f"review timed out after {timeout}s",
            timed_out=True,
        )
    return HarnessResult(returncode=returncode, text=text, stderr=stderr)


def _flatten(value: str) -> str:
    r"""One PHYSICAL log line, always.

    The argument is reviewer-chosen (and PR-author-influenced — seats grep for
    strings drawn from the diff), and downstream log gates anchor on line
    starts, so an embedded newline must not let an argument place chosen text
    at column 0 of its own line (mepro PR #2014 r2).

    Flattened via ``splitlines()`` rather than by replacing ``\r``/``\n``:
    Python breaks lines on eight further characters (``\x0b``, ``\x0c``,
    ``\x1c``-``\x1e``, ``\x85``, ``\u2028``, ``\u2029``), so the replace form
    left a crafted argument looking flat while any consumer splitting by the
    normal rule still saw two lines — the very forgery this guards against
    (fuko-henry, #147). Truncation happens AFTER flattening so a cut cannot
    resurrect a break.
    """
    return " ".join(value.splitlines())[:100]


def _tool_arg(tool_input: dict) -> str:
    """The one argument worth showing for a tool call, truncated for a log line."""
    for key in ("file_path", "pattern", "query", "command", "url", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return _flatten(value)
    for value in tool_input.values():
        if isinstance(value, str) and value:
            return _flatten(value)
    return ""


def _consume_stream(lines, emit) -> tuple[str, bool]:
    """Fold the harness's NDJSON event feed into (final_text, saw_result).

    ``emit(turn, tool, arg)`` fires once per ``tool_use`` block as it streams.
    Tolerant by design: blank/non-JSON/unknown lines are skipped, because a
    progress feature must never be the thing that kills a review. When no
    ``result`` event arrives, the last assistant text block stands in -- for a
    healthy run they are the same message.
    """
    result_text: str | None = None
    last_assistant_text = ""
    turns = 0
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "assistant":
            turns += 1
            message = event.get("message") or {}
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    emit(turns, block.get("name") or "?", _tool_arg(block.get("input") or {}))
                elif block.get("type") == "text" and block.get("text"):
                    last_assistant_text = block["text"]
        elif kind == "result":
            if isinstance(event.get("result"), str):
                result_text = event["result"]
            # The terminal event carries WHY the session ended. `error_max_turns`
            # is the one that mattered: it exits 1 with an otherwise-empty
            # stderr, so without this the run is indistinguishable from a crash
            # and the only visible line names the model, sending every
            # investigation at the provider (mepro #2064, a full day).
            if event.get("is_error") or event.get("subtype") not in (None, "success"):
                subtype = event.get("subtype") or "unknown"
                print(
                    f"fuko: agentic harness ended with result subtype={subtype}"
                    + (" (is_error)" if event.get("is_error") else ""),
                    file=sys.stderr,
                    flush=True,
                )
    if result_text is not None:
        return result_text, True
    return last_assistant_text, False


def _drive(
    cmd: list[str],
    *,
    prompt: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    emit,
) -> tuple[int, str, str, bool]:
    """Run ``cmd`` streaming stdout through :func:`_consume_stream`.

    Returns ``(returncode, final_text, stderr, timed_out)``. Three pipes need
    three actors to avoid deadlock on a large prompt or chatty child: stdin is
    fed from its own thread (the child may start emitting before it finishes
    reading a multi-megabyte diff), stderr drains on a second thread, and the
    main thread consumes the stdout event feed so progress lines appear the
    moment the child writes them. The timeout is a Timer that kills the
    process outright -- the driver classifies that as throttle-class via
    TIMEOUT_RETURNCODE exactly as the old ``subprocess.run(timeout=...)`` did.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
        env=env,
    )
    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, _kill)
    timer.start()
    stderr_chunks: list[str] = []
    stderr_thread = threading.Thread(target=lambda: stderr_chunks.extend(proc.stderr), daemon=True)
    stderr_thread.start()

    def _feed() -> None:
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # child died first; its returncode tells the story

    stdin_thread = threading.Thread(target=_feed, daemon=True)
    stdin_thread.start()
    try:
        text, _saw_result = _consume_stream(proc.stdout, emit)
        proc.wait()
    finally:
        timer.cancel()
    stderr_thread.join(timeout=5)
    return proc.returncode, text, "".join(stderr_chunks), timed_out.is_set()
