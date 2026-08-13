"""Materialize a PR head as a local checkout, plus its diff and metadata.

The reviewer needs two views of the same PR: the unified diff (what changed --
fetched from the GitHub API, so it is exactly what a human reviewer sees) and a
working tree at the head commit (the whole repo, so the agent can chase call
sites and verify claims). The checkout is a shallow, blob-filtered fetch of the
PR's ``pull/N/head`` ref into a temp directory: hooks never run (git does not
execute repo-shipped hooks on clone/fetch/checkout), and nothing here ever
executes repository code -- the tree is data for the reviewer to read.
"""

from __future__ import annotations

import base64
import os
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
        cut = diff.rfind("\ndiff --git ", 0, diff_budget)
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

    Raises:
        CheckoutError: git is unavailable, or any git step fails; stderr of the
            failing step is included (git does not echo the header secret).
    """
    dest = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="fuko-agentic-"))
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{server_url.rstrip('/')}/{repo}.git"
    fetch_env = None
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        fetch_env = {
            **os.environ,
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
    steps: list[tuple[list[str], dict[str, str] | None]] = [
        (["git", "init", "--quiet", str(dest)], None),
        (["git", "-C", str(dest), "remote", "add", "origin", url], None),
        (fetch, fetch_env),
        (["git", "-C", str(dest), "checkout", "--quiet", head_sha], None),
    ]
    for cmd, env in steps:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise CheckoutError(f"checkout step {cmd[:2]} failed: {e}") from e
        if proc.returncode != 0:
            raise CheckoutError(
                f"checkout step {cmd[:2]} exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
    return dest
