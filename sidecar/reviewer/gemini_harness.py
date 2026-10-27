"""Headless Gemini CLI as a review harness (``gemini -p``), on Vertex AI.

The second runtime behind :func:`sidecar.reviewer.harness.run_review`'s
signature, for model families that do not speak the Anthropic Messages API.
The strategy prompt, the output contract, the ledgers and the driver are
unchanged; what differs is how the CLI is isolated, authenticated and read.

Isolation was verified against Gemini CLI 0.65.0 with a hostile checkout
carrying ``.gemini/settings.json`` (hooks, an MCP server, a tool-discovery
command, a context-file rename), ``.env`` and ``.gemini/.env`` (a
``GEMINI_SYSTEM_MD`` takeover), ``GEMINI.md`` at the root and in a
subdirectory, and a skill, an agent and a custom command:

=================================================  =========  ==========
vector                                             naive [1]  this module
=================================================  =========  ==========
shell command executed                             **RAN**    no tool
skill / agent / command / renamed context file     loaded     not loaded
repo ``.env`` reaches the process                  loaded     not loaded
read outside the workspace                         refused    refused
root ``GEMINI.md`` in context                      loaded     stripped
``GEMINI.md`` attached when a sibling file is read loaded     stripped
=================================================  =========  ==========

[1] run from the checkout with ``--skip-trust --yolo``.

The properties that produce the right-hand column, each load-bearing:

* **The CLI never runs from the checkout.** ``cwd`` is a clean directory and
  the checkout is an ``--include-directories`` root. Workspace settings,
  ``.env`` discovery and trust are all decided by the working directory, so a
  repository's ``.gemini/`` is never a settings source. ``--skip-trust``
  trusts that clean directory only; headless mode refuses to start in an
  untrusted one.
* **A private home per run.** ``HOME`` and ``GEMINI_CLI_HOME`` point at a
  fresh directory holding only :func:`harness_settings`, so neither the
  runner's ``~/.gemini`` (extensions, user skills, MCP servers) nor a stray
  ``~/.env`` shapes a published review, and concurrent branches -- threads in
  one process -- never share CLI state. It is created OUTSIDE ``cwd``:
  ``cwd`` is itself a workspace root the agent can read.
* **The tool registry is the boundary.** ``tools.core`` admits exactly
  :data:`GEMINI_TOOLS`; nothing else is registered, so there is no shell,
  write or network tool to approve. Headless ``default`` approval mode refuses
  any tool that would need confirmation, which is a second layer should a
  future CLI widen the registry.
* **Reads are confined by the CLI itself.** Unlike Claude Code's read tools,
  Gemini's refuse any path outside the workspace roots and the session's own
  temp directory ("Path not in workspace"), so no credential denylist is
  needed for stores outside those roots. ``GOOGLE_APPLICATION_CREDENTIALS``
  must therefore never point inside ``cwd`` or the checkout.
* **Context files are not a channel.** The CLI loads ``GEMINI.md`` from an
  included root eagerly and from any directory it reads lazily ("just in
  time"), regardless of ``context.loadMemoryFromIncludeDirectories``. The
  driver's :func:`sidecar.reviewer.checkout.strip_agent_config` removes them,
  at every depth and in any letter case, before the run. ``AGENTS.md`` and
  ``CLAUDE.md`` are not read by this CLI.

Authentication is Google Application Default Credentials only. The Vertex path
does not honour ``GOOGLE_CLOUD_ACCESS_TOKEN`` (that is Code Assist's), so a
short-lived token reaches the CLI through ADC's own sources: a credentials
file, or the metadata server. The project and location come from config, never
the ambient environment -- they decide who pays.

What this harness cannot report: the stream carries no tool OUTPUT
(``tool_result.output`` is empty), so a transcript records what the agent
asked for, not what it was shown; and no dollar cost, so ``cost_usd`` is
``None`` ("not measured"), never a zero.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from tempfile import mkdtemp

from ..throttle import TIMEOUT_RETURNCODE
from .harness import (
    DEFAULT_MAX_TURNS,
    HarnessNotAvailableError,
    HarnessResult,
    _as_int,
    _drive,
    _flatten,
    _StreamOutcome,
    _tool_arg,
)
from .transcript import Transcript

#: The read-only navigation tools the agent is given, by registry name.
GEMINI_TOOLS = ("read_file", "read_many_files", "glob", "grep_search", "list_directory")

#: Vertex location when the model entry names none. Current Gemini models are
#: served only from the global endpoint (measured 2026-10-27: regional
#: endpoints 404 for every 3.x and 4.x model).
DEFAULT_LOCATION = "global"

#: Operator-supplied admin settings, forwarded as the CLI's system settings
#: path. The CLI ignores a system settings file whose directory is not owned by
#: root, so this is for a runner image that bakes one in (e.g.
#: ``/etc/gemini-cli/settings.json``), not for a per-run file.
ENV_SYSTEM_SETTINGS = "FUKO_GEMINI_SYSTEM_SETTINGS"

#: Ambient variables that decide where the CLI's traffic goes, who pays, what
#: it runs as, or what it is told -- stripped so config decides, never the
#: runner's environment. ``GOOGLE_APPLICATION_CREDENTIALS`` and
#: ``GCE_METADATA_HOST`` are deliberately absent: they are WHERE the credential
#: comes from, which is the runner's business, not which project is billed.
_GOOGLE_INHERITED_VARS = (
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_GCA",
    "GOOGLE_GENAI_API_VERSION",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_PROJECT_ID",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_CLOUD_REGION",
    "GOOGLE_CLOUD_UNIVERSE_DOMAIN",
    "GOOGLE_CLOUD_ACCESS_TOKEN",
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    "GOOGLE_API_KEY",
    "CLOUD_SHELL",
)

#: Prefix of the CLI's own configuration surface (``GEMINI_API_KEY``,
#: ``GEMINI_MODEL``, ``GEMINI_SYSTEM_MD``, ``GEMINI_CLI_*`` ...). Every one is
#: either set by this module or must not be inherited.
_GEMINI_ENV_PREFIX = "GEMINI_"


def harness_settings(max_turns: int) -> dict:
    """The user-scope settings every run is given, in a private home.

    Only keys whose effect was observed are set. ``tools.core`` is the boundary
    (see the module docstring); the rest switch off what a reviewer must not
    load or emit -- hooks, skills, agents, checkpointing, usage statistics and
    telemetry, update checks -- and ``ignoreLocalEnv`` stops ``.env`` discovery
    from the working directory's ancestors.
    """
    return {
        "security": {
            "folderTrust": {"enabled": True},
            "auth": {"selectedType": "vertex-ai", "enforcedType": "vertex-ai"},
        },
        "hooksConfig": {"enabled": False},
        "hooks": {},
        "mcpServers": {},
        "tools": {"core": list(GEMINI_TOOLS)},
        "skills": {"enabled": False},
        "experimental": {"enableAgents": False},
        "advanced": {"ignoreLocalEnv": True},
        "privacy": {"usageStatisticsEnabled": False},
        "telemetry": {"enabled": False},
        "general": {
            "checkpointing": {"enabled": False},
            "disableAutoUpdate": True,
            "disableUpdateNag": True,
        },
        "context": {"loadMemoryFromIncludeDirectories": False},
        "model": {"maxSessionTurns": max_turns},
    }


def harness_environment(base: dict[str, str], *, project: str, location: str) -> dict[str, str]:
    """Return ``base`` minus inherited routing variables, plus Vertex routing.

    ``base`` is the driver's already-scrubbed environment (no GitHub or model
    credentials). This removes the Google and Gemini variables that would let
    the runner's environment pick the project, endpoint, model or system
    prompt, then sets exactly the three that select Vertex for ``project`` in
    ``location``. ``HOME``, ``GEMINI_CLI_HOME`` and ``TMPDIR`` are set per run
    by :func:`run_review`, which owns the private home.
    """
    env = {
        k: v
        for k, v in base.items()
        if k not in _GOOGLE_INHERITED_VARS and not k.startswith(_GEMINI_ENV_PREFIX)
    }
    env["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    env["GOOGLE_CLOUD_PROJECT"] = project
    env["GOOGLE_CLOUD_LOCATION"] = location
    return env


def vertex_endpoint(project: str, location: str) -> str:
    """The Vertex base URL a run for ``project``/``location`` actually reaches.

    For the receipt's endpoint field. The global location has no regional
    hostname prefix; every other location does.
    """
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return f"https://{host}/v1/projects/{project}/locations/{location}"


def _default_adc_file(home: str | None) -> str | None:
    """The gcloud ADC file under the runner's real ``HOME``, if one exists.

    :func:`run_review` replaces ``HOME``, which would otherwise hide a laptop
    user's ``gcloud auth application-default login`` from ADC's lookup.
    """
    if not home:
        return None
    path = Path(home) / ".config" / "gcloud" / "application_default_credentials.json"
    return str(path) if path.is_file() else None


def _private_home(env: dict[str, str], max_turns: int) -> tuple[Path, dict[str, str]]:
    """Create this run's private home and return it with the environment to use.

    The home holds ``.gemini/settings.json`` and a ``tmp`` that becomes
    ``TMPDIR``, so the CLI's own error reports -- which can quote the request,
    i.e. the reviewed code -- land inside the directory the caller removes,
    not in a shared ``/tmp``.
    """
    home = Path(mkdtemp(prefix="fuko-gemini-home-"))
    gemini_dir = home / ".gemini"
    gemini_dir.mkdir()
    (gemini_dir / "settings.json").write_text(json.dumps(harness_settings(max_turns), indent=1))
    tmp = home / "tmp"
    tmp.mkdir()
    run_env = dict(env)
    if "GOOGLE_APPLICATION_CREDENTIALS" not in run_env:
        adc = _default_adc_file(env.get("HOME"))
        if adc:
            run_env["GOOGLE_APPLICATION_CREDENTIALS"] = adc
    run_env["HOME"] = str(home)
    run_env["GEMINI_CLI_HOME"] = str(home)
    run_env["TMPDIR"] = str(tmp)
    system_settings = os.environ.get(ENV_SYSTEM_SETTINGS, "").strip()
    if system_settings:
        run_env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = system_settings
    return home, run_env


def _gemini_usage(stats: object) -> dict | None:
    """Map a ``result`` event's ``stats`` onto the Anthropic-shaped usage keys.

    :func:`sidecar.reviewer.harness.usage_tokens` reads Anthropic spellings,
    where ``input_tokens`` excludes cache reads. Gemini's ``input_tokens`` is
    the inclusive total and ``input`` the uncached part (measured: 14609 =
    10684 + 3925 cached), so ``input`` is what lands in ``input_tokens``. The
    raw stats ride along under ``gemini`` for whoever wants a column the
    mapping does not fill. Gemini reports no cache WRITES, so that key is
    absent and reads as "not measured".
    """
    if not isinstance(stats, dict):
        return None
    usage: dict = {"gemini": stats}
    for ours, theirs in (
        ("input_tokens", "input"),
        ("output_tokens", "output_tokens"),
        ("cache_read_input_tokens", "cached"),
    ):
        value = _as_int(stats.get(theirs))
        if value is not None:
            usage[ours] = value
    return usage


def _error_text(event: dict) -> str:
    """The message of a failed ``result`` event, or the empty string."""
    error = event.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return ""


def _consume_gemini_stream(lines, emit) -> _StreamOutcome:
    """Fold the Gemini CLI's ``stream-json`` feed into a :class:`_StreamOutcome`.

    The dialect: ``init``; ``message`` events (``role`` user/assistant,
    assistant text arriving as ``delta`` chunks); top-level ``tool_use`` /
    ``tool_result``; and one terminal ``result`` with ``status`` and
    ``stats``, or ``error`` on failure.

    The final message is the assistant text after the LAST tool call -- the
    same "last assistant text block" rule as Claude Code's fold -- so narration
    between tool calls never reaches the JSON parser. ``subtype`` carries the
    terminal ``status`` (``success`` / ``error``), and a failed run's error
    message becomes the text, so the driver's auth and throttle classifiers
    can read it. ``turns`` stays ``None``: the CLI reports tool calls, not
    turns, and a different quantity in that column would be a second
    definition of it.
    """
    text_since_tool = ""
    error_text = ""
    reported_usage: dict | None = None
    reported_status: str | None = None
    saw_result = False
    tool_calls = 0
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
        if kind == "tool_use":
            tool_calls += 1
            text_since_tool = ""
            params = event.get("parameters")
            emit(
                tool_calls,
                str(event.get("tool_name") or "?"),
                _tool_arg(params if isinstance(params, dict) else {}),
            )
        elif kind == "message" and event.get("role") == "assistant":
            content = event.get("content")
            if isinstance(content, str):
                text_since_tool += content
        elif kind == "result":
            saw_result = True
            if isinstance(event.get("status"), str):
                reported_status = event["status"]
            reported_usage = _gemini_usage(event.get("stats"))
            error_text = _error_text(event)
    text = text_since_tool
    if reported_status not in (None, "success") and error_text:
        text = error_text
    return _StreamOutcome(
        text=text,
        saw_result=saw_result,
        usage=reported_usage,
        cost_usd=None,
        turns=None,
        subtype=reported_status,
    )


def run_review(
    prompt: str,
    repo_dir: Path,
    *,
    cwd: Path,
    model: str,
    env: dict[str, str],
    timeout: int,
    max_turns: int = DEFAULT_MAX_TURNS,
    transcript: Transcript | None = None,
) -> HarnessResult:
    """Run headless Gemini CLI over ``repo_dir`` and return its final text.

    Same contract as :func:`sidecar.reviewer.harness.run_review`: ``cwd`` is a
    clean directory, ``repo_dir`` an additional readable root, ``env`` the
    complete process environment (see :func:`harness_environment`), and a
    timeout maps to :data:`sidecar.throttle.TIMEOUT_RETURNCODE`. The prompt goes
    over stdin with an empty ``-p``, which the CLI appends to the piped input,
    so a multi-megabyte diff never reaches argv. ``max_turns`` becomes the
    CLI's ``model.maxSessionTurns``.

    The private home (see the module docstring) is created here and removed on
    every exit path, the kill path included.
    """
    binary = shutil.which("gemini", path=env.get("PATH"))
    if binary is None:
        raise HarnessNotAvailableError(
            "the 'gemini' CLI is not on PATH; install @google/gemini-cli (Node 20+) "
            "on this runner or switch this model entry to another provider"
        )
    home, run_env = _private_home(env, max_turns)
    cmd = [
        binary,
        "-p",
        "",
        "--model",
        model,
        "--output-format",
        "stream-json",
        # Trusts the clean cwd, which headless mode requires; the checkout is
        # an included root and is never the trusted workspace.
        "--skip-trust",
        "--approval-mode",
        "default",
        "--include-directories",
        str(repo_dir),
    ]
    start = time.monotonic()

    def _emit(turn: int, tool: str, arg: str) -> None:
        mins = int((time.monotonic() - start) // 60)
        print(
            f"fuko: agentic {model} [{mins}m t{turn}] {_flatten(tool)} {arg}",
            file=sys.stderr,
            flush=True,
        )

    try:
        returncode, outcome, stderr, timed_out = _drive(
            cmd,
            prompt=prompt,
            cwd=cwd,
            env=run_env,
            timeout=timeout,
            emit=_emit,
            transcript=transcript,
            consume=_consume_gemini_stream,
        )
    finally:
        shutil.rmtree(home, ignore_errors=True)
    if outcome.subtype and outcome.subtype != "success":
        print(
            f"fuko: agentic {model} harness ended with result status={_flatten(outcome.subtype)}",
            file=sys.stderr,
            flush=True,
        )
    if timed_out:
        return HarnessResult(
            returncode=TIMEOUT_RETURNCODE,
            text="",
            stderr=stderr or f"review timed out after {timeout}s",
            timed_out=True,
            usage=outcome.usage,
        )
    # The CLI can exit 0 with a failed terminal event; trust the event.
    if returncode == 0 and outcome.subtype not in (None, "success"):
        returncode = 1
    return HarnessResult(
        returncode=returncode,
        text=outcome.text,
        stderr=stderr,
        usage=outcome.usage,
        cost_usd=None,
        turns=None,
    )
