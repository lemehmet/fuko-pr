"""Agent runtimes that can execute the review strategy.

A harness takes a prepared prompt plus a checkout directory and returns the
agent's final text; :func:`sidecar.reviewer.prompt.parse_review` turns that
into structured findings. The first (and currently only) harness drives
headless Claude Code (``claude -p``), which must be installed on the runner.
Other runtimes -- e.g. an OSS agentic harness fronting non-Anthropic models --
implement the same ``run_review`` signature and slot in without touching the
strategy or the driver.

The agent's tool surface is pinned to read-only code navigation
(:data:`ALLOWED_TOOLS`). Headless mode cannot prompt for permissions, so every
tool outside the allowlist is denied by construction -- the agent cannot run
repository code, write files, or reach the network even if the (untrusted)
repository content asks it to.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..throttle import TIMEOUT_RETURNCODE

ALLOWED_TOOLS = "Read,Grep,Glob"
DEFAULT_MAX_TURNS = 50


class HarnessNotAvailableError(RuntimeError):
    """Raised when the harness binary is not installed on this runner."""


@dataclass(frozen=True)
class HarnessResult:
    """The raw outcome of one agent run."""

    returncode: int
    text: str
    stderr: str = ""
    timed_out: bool = False


def run_review(
    prompt: str,
    checkout: Path,
    *,
    model: str,
    env: dict[str, str],
    timeout: int,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> HarnessResult:
    """Run headless Claude Code over ``checkout`` and return its final text.

    The prompt goes over stdin (it embeds a full diff -- argv has size limits
    and shows up in process listings). ``--output-format text`` keeps the
    contract simple: stdout IS the agent's final message, which the strategy
    already constrains to a bare JSON object. ``env`` is the harness process
    environment (the caller passes its translated credentials, e.g.
    ``ANTHROPIC_API_KEY``); it is used as given rather than merged here so the
    caller controls exactly what the agent process can see.

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
        "text",
        "--allowedTools",
        ALLOWED_TOOLS,
        "--max-turns",
        str(max_turns),
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=str(checkout),
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        partial = e.stderr if isinstance(e.stderr, str) else ""
        return HarnessResult(
            returncode=TIMEOUT_RETURNCODE,
            text="",
            stderr=partial or f"review timed out after {timeout}s",
            timed_out=True,
        )
    return HarnessResult(returncode=proc.returncode, text=proc.stdout, stderr=proc.stderr)
