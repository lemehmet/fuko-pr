"""The ``fuko review`` runner.

Orchestrates one PR review through the configured, pluggable backend:

1. resolve ``.fuko.toml`` and the provider preset,
2. build repo knowledge (from a running sidecar over HTTP, or a local store),
3. translate config -> backend env (ingress),
4. invoke the backend,
5. normalize its output into Review Signals (egress; stubbed until task 8).

Knowledge and PR-context failures degrade gracefully: the review still runs,
just without injected knowledge -- matching the original workflow's behavior.
"""

from __future__ import annotations

import os
import re
import sys

import httpx

from .backends import get_backend
from .backends.base import InvokeResult, PRRef
from .fukoconfig import DEFAULT_CONFIG_PATH, FukoConfig, load_config
from .presets import get_preset

_PR_URL = re.compile(r"https?://[^/]+/([^/]+/[^/]+)/pull/(\d+)")
_DEFAULT_API = "https://api.github.com"


def parse_pr_url(url: str) -> PRRef:
    """Parse a PR URL into a ``PRRef`` (``owner/repo`` + number)."""
    m = _PR_URL.match(url)
    if not m:
        raise ValueError(f"not a pull request URL: {url!r}")
    return PRRef(repo=m.group(1), number=int(m.group(2)), url=url)


def _gh_headers(token: str) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _fetch_pr_context(pr: PRRef, token: str, api_url: str) -> tuple[list[str], str]:
    """Fetch a PR's changed file paths (paginated) and body via the GitHub API."""
    base = api_url.rstrip("/")
    with httpx.Client(timeout=30.0, headers=_gh_headers(token)) as client:
        meta = client.get(f"{base}/repos/{pr.repo}/pulls/{pr.number}")
        meta.raise_for_status()
        body = meta.json().get("body") or ""

        files: list[str] = []
        page = 1
        while True:
            resp = client.get(
                f"{base}/repos/{pr.repo}/pulls/{pr.number}/files",
                params={"page": page, "per_page": 100},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            files.extend(f["filename"] for f in batch)
            if len(batch) < 100:
                break
            page += 1
    return files, body


def _sidecar_query(url: str, token: str, repo: str, files: list[str], pr_body: str) -> list[dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    resp = httpx.post(
        url.rstrip("/") + "/query",
        json={"repo": repo, "files": files, "pr_body": pr_body},
        headers=headers,
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") if isinstance(data, dict) else None
    return results if isinstance(results, list) else []


def _local_query(repo: str, files: list[str], pr_body: str) -> list[dict]:
    from . import retrieve

    return retrieve.query(repo, files, pr_body, None, None)


def build_knowledge(pr: PRRef, token: str, api_url: str) -> str:
    """Return the formatted knowledge block for ``pr``, or ``""`` on any failure.

    Uses a running sidecar when ``FUKO_URL`` is set (the homelab deployment),
    otherwise queries the local store directly.
    """
    fuko_url = os.environ.get("FUKO_URL", "").strip()
    fuko_token = os.environ.get("FUKO_TOKEN", "")
    try:
        files, pr_body = _fetch_pr_context(pr, token, api_url)
        if fuko_url:
            results = _sidecar_query(fuko_url, fuko_token, pr.repo, files, pr_body)
        else:
            results = _local_query(pr.repo, files, pr_body)
    except Exception as e:
        print(f"fuko: knowledge build failed, proceeding without it: {e}", file=sys.stderr)
        return ""

    from .cli import format_extra_instructions

    print(f"fuko: injected {len(results)} learnings", file=sys.stderr)
    return format_extra_instructions(results)


def _github_env(token: str) -> dict[str, str]:
    """Map a GitHub token into PR-Agent's CLI (user-token) deployment settings."""
    if not token:
        return {}
    return {"GITHUB__USER_TOKEN": token, "GITHUB__DEPLOYMENT_TYPE": "user"}


def review(pr_url: str, config_path: str = DEFAULT_CONFIG_PATH) -> InvokeResult:
    """Run a full review for ``pr_url`` using the backend named in ``config_path``."""
    cfg: FukoConfig = load_config(config_path)
    pr = parse_pr_url(pr_url)
    token = os.environ.get("GITHUB_TOKEN", "")
    api_url = os.environ.get("GITHUB_API_URL", _DEFAULT_API)

    backend = get_backend(cfg.review.backend, cfg.review)
    preset = get_preset(cfg.review.model.provider)
    knowledge = build_knowledge(pr, token, api_url)

    env = backend.build_env(preset, cfg.review.model, knowledge, cfg.review.tools)
    env.update(_github_env(token))

    result = backend.invoke(pr, env, cfg.review.tools)

    try:
        signals = backend.normalize_output(pr)
        print(f"fuko: normalized {len(signals)} review signals", file=sys.stderr)
    except NotImplementedError:
        pass

    return result
