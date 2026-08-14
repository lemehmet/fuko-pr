"""Materialize a PR head as a local checkout, plus its diff and metadata.

The reviewer needs two views of the same PR: the unified diff (what changed --
fetched from the GitHub API, so it is exactly what a human reviewer sees) and a
working tree at the head commit (the whole repo, so the agent can chase call
sites and verify claims). The checkout is a shallow (``--depth 1``) fetch of the
PR's ``pull/N/head`` ref into a temp directory -- deliberately WITHOUT a blob
filter, see :func:`checkout_pr_head`. Hooks never run (git does not execute
repo-shipped hooks on clone/fetch/checkout), LFS smudging is disabled so no
repository-specified filter or endpoint is contacted, and nothing here ever
executes repository code -- the tree is data for the reviewer to read.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx


class CheckoutError(RuntimeError):
    """Raised when the PR head cannot be fetched or checked out."""


@dataclass(frozen=True)
class PRContext:
    """Everything the reviewer needs to know about the PR under review."""

    title: str
    body: str
    head_sha: str
    base_ref: str
    diff: str
    diff_files: frozenset[str]
    truncated: bool = False


_DIFF_FILE_RE_PREFIX = "+++ b/"


def _api_headers(token: str) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def fetch_pr_context(
    repo: str,
    number: int,
    *,
    token: str,
    api_url: str = "https://api.github.com",
    diff_budget: int = 250_000,
) -> PRContext:
    """Fetch PR metadata and its unified diff from the GitHub API.

    ``diff_budget`` caps the diff text (in characters) embedded into the review
    prompt; an over-budget diff is cut at the last complete file boundary below
    the cap and flagged ``truncated`` so the prompt can say so instead of the
    model silently reviewing a partial diff. ``diff_files`` is parsed from
    ``+++ b/`` headers of the FULL diff (before truncation), because it is also
    the anchor set for inline comment placement.
    """
    api = api_url.rstrip("/")
    with httpx.Client(timeout=60.0, headers=_api_headers(token)) as client:
        meta = client.get(f"{api}/repos/{repo}/pulls/{number}")
        meta.raise_for_status()
        pull = meta.json()
        diff_resp = client.get(
            f"{api}/repos/{repo}/pulls/{number}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        diff_resp.raise_for_status()
        diff = diff_resp.text

    files = frozenset(
        line[len(_DIFF_FILE_RE_PREFIX) :].strip()
        for line in diff.splitlines()
        if line.startswith(_DIFF_FILE_RE_PREFIX)
    )
    truncated = len(diff) > diff_budget
    if truncated:
        # Prefer a file boundary; if the very first file already exceeds the
        # budget there is none, so fall back to the last complete LINE rather
        # than slicing mid-line and handing the model a malformed hunk.
        cut = diff.rfind("\ndiff --git ", 0, diff_budget)
        if cut <= 0:
            cut = diff.rfind("\n", 0, diff_budget)
        diff = diff[: cut if cut > 0 else diff_budget]

    return PRContext(
        title=pull.get("title") or "",
        body=pull.get("body") or "",
        head_sha=pull["head"]["sha"],
        base_ref=pull["base"]["ref"],
        diff=diff,
        diff_files=files,
        truncated=truncated,
    )


#: Agent-runtime config THIS harness would read, removed at ANY depth: a
#: subdirectory of a checkout is a project root in its own right, so a nested
#: ``.claude/`` (settings, skills, subagents) or ``.mcp.json`` is the same
#: vector as the top-level one. Bare names, matched against each directory
#: entry during the walk -- not paths.
AGENT_CONFIG_NAMES = (".claude", ".mcp.json")

#: Other tools' agent config, removed at the checkout ROOT only. These are not
#: read by this harness at all; clearing them just keeps the reviewer from
#: inheriting a neighbouring tool's instructions. Root-relative PATHS (note
#: ``.github/copilot-instructions.md`` has a directory component), which is why
#: they cannot be folded into the name-matched set above.
AGENT_CONFIG_ROOT_PATHS = (".cursor", ".github/copilot-instructions.md")


def _remove_path(target: Path) -> bool:
    """Best-effort delete of a file or directory; report whether it was there.

    Symlinks are unlinked, never followed: the checkout is contributor-
    controlled, so a ``.claude`` symlink pointing outside it must cost the link
    and nothing else. (``is_dir()`` follows symlinks, which is why the symlink
    test has to come first.)
    """
    try:
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
        else:
            return False
    except OSError:
        return False
    return True


def strip_agent_config(root: Path) -> list[str]:
    """Delete agent-runtime config from a checkout; return what was removed.

    Defense in depth behind the clean working directory in
    :mod:`sidecar.reviewer.harness`. The primary guarantee is that hooks and
    MCP servers do not load from an ``--add-dir`` root at all -- they load from
    the working directory, which is never the checkout. This strip exists
    because skills and subagents *are* read from additional roots, and because
    the reviewer should not inherit instructions from the code it is judging in
    any case.

    :data:`AGENT_CONFIG_NAMES` is cleared at every depth (a nested project root
    is a project root); :data:`AGENT_CONFIG_ROOT_PATHS` only at ``root``.
    Returned entries are always repository-relative, so a nested hit reads
    ``packages/x/.claude`` rather than a bare name.

    The files still appear in the diff the reviewer reads, so a PR that edits
    them remains reviewable -- and worth reporting, which the strategy prompt
    asks for explicitly. Removal is best-effort and idempotent: this operates
    on a throwaway checkout, so a failure to delete is not worth failing the
    review over.
    """
    removed: list[str] = []
    for rel in AGENT_CONFIG_ROOT_PATHS:
        if _remove_path(root / rel):
            removed.append(rel)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        for name in list(dirnames):
            # Never descend into the git database; it holds no agent config and
            # walking it on a large repo is pure cost.
            if name == ".git":
                dirnames.remove(name)
            elif name in AGENT_CONFIG_NAMES:
                if _remove_path(here / name):
                    removed.append(str((here / name).relative_to(root)))
                dirnames.remove(name)
        for name in filenames:
            if name in AGENT_CONFIG_NAMES and _remove_path(here / name):
                removed.append(str((here / name).relative_to(root)))
    return removed


def checkout_pr_head(
    repo: str,
    number: int,
    head_sha: str,
    *,
    token: str,
    server_url: str = "https://github.com",
    workdir: str | None = None,
) -> Path:
    """Fetch ``pull/N/head`` shallowly into a fresh temp directory and check it out.

    The fetch is ``--depth 1`` WITHOUT a blob filter: a filtered clone would
    lazy-fetch blobs during checkout, and those follow-up fetches would not
    carry the auth header (it is scoped to the explicit fetch step below), so
    a private repo's checkout would fail. Depth 1 alone brings exactly the head
    commit's tree in one authenticated packfile. The token rides in an
    ephemeral ``http.<url>.extraheader`` passed via the ``GIT_CONFIG_*``
    environment (not ``-c`` argv, which process listings can read; not the
    remote URL, which ``.git/config`` would persist), scoped to the fetch step
    only.

    Every step runs with ``GIT_LFS_SKIP_SMUDGE=1``. The checkout step would
    otherwise run the LFS smudge filter for any path the repository's own
    ``.gitattributes`` marks, which contacts an endpoint the repository can
    name (``.lfsconfig``) while the runner's git credentials are in scope. The
    reviewer only needs to read the pointer files.

    A checkout that fails part-way removes the directory it created, so a
    failing PR does not leave a half-populated tree behind; a caller-supplied
    ``workdir`` is left alone (it is the caller's to manage).

    Raises:
        CheckoutError: git is unavailable, or any git step fails; stderr of the
            failing step is included (git does not echo the header secret).
    """
    ours = workdir is None
    dest = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="fuko-agentic-"))
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{server_url.rstrip('/')}/{repo}.git"
    base_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    fetch_env = base_env
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        fetch_env = {
            **base_env,
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"http.{url}.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
        }
    fetch = [
        "git",
        "-C",
        str(dest),
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "origin",
        f"pull/{number}/head",
    ]
    steps: list[tuple[list[str], dict[str, str]]] = [
        (["git", "init", "--quiet", str(dest)], base_env),
        (["git", "-C", str(dest), "remote", "add", "origin", url], base_env),
        (fetch, fetch_env),
        (["git", "-C", str(dest), "checkout", "--quiet", head_sha], base_env),
    ]
    for cmd, env in steps:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            if ours:
                shutil.rmtree(dest, ignore_errors=True)
            raise CheckoutError(f"checkout step {cmd[:2]} failed: {e}") from e
        if proc.returncode != 0:
            detail = (
                f"checkout step {cmd[:2]} exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
            if ours:
                shutil.rmtree(dest, ignore_errors=True)
            raise CheckoutError(detail)
    return dest
