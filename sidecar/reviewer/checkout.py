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
import re
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
    #: (path, new-side line) pairs the API will accept as inline anchors.
    diff_positions: frozenset[tuple[str, int]] = frozenset()
    truncated: bool = False


_DIFF_FILE_RE_PREFIX = "+++ b/"
_HUNK_NEW_START_RE = re.compile(r"^@@ .*?\+(\d+)")


def parse_diff(diff: str) -> tuple[frozenset[str], frozenset[tuple[str, int]]]:
    """Parse a unified diff into its file set and its anchorable positions.

    One pass, because both answers need the same state and getting that state
    wrong corrupts both. Two properties are load-bearing:

    * **A ``+++ b/`` line is a file header only OUTSIDE a hunk.** An added
      source line whose own content starts with ``++ b/`` serializes as
      ``+++ b/...`` inside the hunk, and treating it as a header silently
      repoints every subsequent position at an attacker-chosen path. Hunk state
      is what tells the two apart.
    * **Only added (``+``) and context lines are anchorable.** Removed lines do
      not exist on the new side and consume no new-side number; the API rejects
      any line outside a hunk with a 422, and one rejection fails the WHOLE
      review, so this set is what keeps a bad line number from demoting every
      other finding to the body.
    """
    files: set[str] = set()
    positions: set[tuple[str, int]] = set()
    path: str | None = None
    new_line = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path, in_hunk = None, False
            continue
        if not in_hunk and line.startswith(_DIFF_FILE_RE_PREFIX):
            path = line[len(_DIFF_FILE_RE_PREFIX) :].strip()
            files.add(path)
            continue
        if line.startswith("@@"):
            match = _HUNK_NEW_START_RE.match(line)
            if match:
                new_line = int(match.group(1))
                in_hunk = True
            continue
        if not in_hunk or path is None:
            continue
        if line.startswith("\\"):  # "\ No newline at end of file"
            continue
        # "" is a context line whose trailing space some tools strip; counting
        # it keeps the new-side numbering aligned with the file.
        if line == "" or line.startswith(("+", " ")):
            positions.add((path, new_line))
            new_line += 1
    return frozenset(files), frozenset(positions)


def parse_diff_positions(diff: str) -> frozenset[tuple[str, int]]:
    """The anchorable ``(path, new-side line)`` pairs of ``diff``."""
    return parse_diff(diff)[1]


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

    # Parsed from the FULL diff: these are anchor sets for comment placement,
    # and a truncated tail is still real, anchorable code.
    files, positions = parse_diff(diff)
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
        diff_positions=positions,
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


def _has_symlinked_parent(target: Path, root: Path) -> bool:
    """Whether any directory component between ``root`` and ``target`` is a link.

    A symlinked *leaf* is handled by :func:`_remove_path`, but a symlinked
    parent is the same escape one level up: ``.github/copilot-instructions.md``
    has a directory component, so a PR that ships ``.github`` as a link to
    somewhere outside the checkout would make ``exists()`` follow it and
    ``unlink()`` delete the target's file rather than the checkout's.
    """
    try:
        relative = target.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _remove_path(target: Path) -> bool:
    """Best-effort delete of a file or directory; report whether it was there.

    Symlinks are unlinked, never followed: the checkout is contributor-
    controlled, so a ``.claude`` symlink pointing outside it must cost the link
    and nothing else. (``is_dir()`` follows symlinks, which is why the symlink
    test has to come first.) Callers are responsible for rejecting a target
    whose PARENT is a link -- see :func:`_has_symlinked_parent`.
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
        target = root / rel
        # A multi-component root path (`.github/copilot-instructions.md`) can be
        # reached through a symlinked directory; deleting through it would land
        # outside the checkout entirely.
        if _has_symlinked_parent(target, root):
            continue
        if _remove_path(target):
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
