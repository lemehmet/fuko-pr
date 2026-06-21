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
from dataclasses import replace

import httpx

from .backends import get_backend
from .backends.base import InvokeResult, PRRef
from .fukoconfig import DEFAULT_CONFIG_PATH, FukoConfig, KnowledgeConfig, ModelConfig, load_config
from .pool import order_pool, resolve_pool
from .presets import get_preset
from .sizing import required_context
from .stores import get_store

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


def _paginated_get(token: str, api_url: str, path: str) -> list[dict]:
    """Fetch all pages of a paginated GitHub list endpoint at ``path``."""
    base = api_url.rstrip("/")
    out: list[dict] = []
    page = 1
    with httpx.Client(timeout=30.0, headers=_gh_headers(token)) as client:
        while True:
            resp = client.get(f"{base}{path}", params={"page": page, "per_page": 100})
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
    return out


def fetch_inline_comments(pr: PRRef, token: str, api_url: str) -> list[dict]:
    """Fetch all inline review comments on a PR (paginated)."""
    return _paginated_get(token, api_url, f"/repos/{pr.repo}/pulls/{pr.number}/comments")


def fetch_issue_comments(pr: PRRef, token: str, api_url: str) -> list[dict]:
    """Fetch all issue-level comments on a PR (paginated) — e.g. CodeRabbit's walkthrough."""
    return _paginated_get(token, api_url, f"/repos/{pr.repo}/issues/{pr.number}/comments")


def fetch_reviews(pr: PRRef, token: str, api_url: str) -> list[dict]:
    """Fetch all submitted reviews on a PR (paginated)."""
    return _paginated_get(token, api_url, f"/repos/{pr.repo}/pulls/{pr.number}/reviews")


def fetch_pr_head(pr: PRRef, token: str, api_url: str) -> str:
    """Return the PR's current head commit sha."""
    base = api_url.rstrip("/")
    with httpx.Client(timeout=30.0, headers=_gh_headers(token)) as client:
        resp = client.get(f"{base}/repos/{pr.repo}/pulls/{pr.number}")
        resp.raise_for_status()
        return resp.json()["head"]["sha"]


def fetch_check_runs(pr: PRRef, ref: str, token: str, api_url: str) -> list[dict]:
    """Fetch all check-runs for a commit ``ref`` (paginated).

    The list-check-runs endpoint wraps its page under a ``check_runs`` key (unlike
    the bare-array list endpoints handled by :func:`_paginated_get`), and reports the
    full count in ``total_count`` — used here to know when to stop. This is the
    authoritative completion signal for reviewers that publish a check (e.g.
    CodeRabbit's "Review in progress" → "Review completed").
    """
    base = api_url.rstrip("/")
    out: list[dict] = []
    page = 1
    with httpx.Client(timeout=30.0, headers=_gh_headers(token)) as client:
        while True:
            resp = client.get(
                f"{base}/repos/{pr.repo}/commits/{ref}/check-runs",
                params={"page": page, "per_page": 100},
            )
            resp.raise_for_status()
            payload = resp.json()
            batch = payload.get("check_runs") or []
            out.extend(batch)
            total = payload.get("total_count")
            if not batch or (total is not None and len(out) >= total) or len(batch) < 100:
                break
            page += 1
    return out


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


def build_knowledge(pr: PRRef, token: str, api_url: str, knowledge: KnowledgeConfig) -> str:
    """Return the formatted knowledge block for ``pr``, or ``""`` on any failure.

    Uses a running sidecar when ``FUKO_URL`` is set (the homelab deployment),
    otherwise queries the configured local store. The local store is constructed
    lazily, so a ``sqlite-vec`` config never has to resolve when the sidecar is used.
    """
    fuko_url = os.environ.get("FUKO_URL", "").strip()
    fuko_token = os.environ.get("FUKO_TOKEN", "")
    try:
        files, pr_body = _fetch_pr_context(pr, token, api_url)
        if fuko_url:
            results = _sidecar_query(fuko_url, fuko_token, pr.repo, files, pr_body)
        else:
            results = get_store(knowledge).query(pr.repo, files, pr_body, None, None)
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


def _cb_endpoint() -> tuple[str, str]:
    """Return ``(fuko_url, fuko_token)`` for the sidecar's circuit-breaker API."""
    return os.environ.get("FUKO_URL", "").strip(), os.environ.get("FUKO_TOKEN", "")


def _cb_cooldowns() -> set[str]:
    """Provider ids currently in circuit-breaker cooldown (best-effort).

    Reads the shared state from the sidecar over HTTP when ``FUKO_URL`` is set,
    else from the local Postgres. Any failure yields an empty set -- the breaker
    is an optimization, so a read error must never block a review.
    """
    fuko_url, fuko_token = _cb_endpoint()
    try:
        if fuko_url:
            headers = {"Authorization": "Bearer " + fuko_token} if fuko_token else {}
            resp = httpx.get(fuko_url.rstrip("/") + "/cb/cooldowns", headers=headers, timeout=10.0)
            resp.raise_for_status()
            data = resp.json().get("cooldowns") if isinstance(resp.json(), dict) else None
            return set(data.keys()) if isinstance(data, dict) else set()
        from .circuit_breaker import get_cooldowns

        return set(get_cooldowns().keys())
    except Exception as e:
        print(f"fuko: circuit-breaker read failed, ignoring cooldowns: {e}", file=sys.stderr)
        return set()


def _cb_trip(provider: str, cooldown_seconds: int, reason: str) -> None:
    """Open ``provider``'s breaker (best-effort; a failure must not abort failover)."""
    fuko_url, fuko_token = _cb_endpoint()
    try:
        if fuko_url:
            headers = {"Content-Type": "application/json"}
            if fuko_token:
                headers["Authorization"] = "Bearer " + fuko_token
            resp = httpx.post(
                fuko_url.rstrip("/") + "/cb/trip",
                json={
                    "provider": provider,
                    "cooldown_seconds": cooldown_seconds,
                    "reason": (reason or "")[:500],
                },
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
        else:
            from .circuit_breaker import trip

            trip(provider, cooldown_seconds, reason)
    except Exception as e:
        print(f"fuko: circuit-breaker trip failed (continuing): {e}", file=sys.stderr)


def _normalize(backend, pr: PRRef, model: ModelConfig) -> None:
    """Map the posted review into Review Signals for the winning provider's model."""
    try:
        preset = get_preset(model.provider)
        model_id = preset.litellm_prefix + model.name
        signals = backend.normalize_output(pr, model=model_id)
        print(f"fuko: normalized {len(signals)} review signals", file=sys.stderr)
    except NotImplementedError:
        pass


def _estimate_required_context(pr: PRRef, token: str, api_url: str, knowledge: str) -> int | None:
    """Estimate the context window this review needs, or ``None`` if unsizable.

    Fetches the PR diff to size the job; any failure returns ``None`` so
    context-fit ordering is skipped rather than ever blocking a review.
    """
    try:
        base = api_url.rstrip("/")
        headers = _gh_headers(token)
        headers["Accept"] = "application/vnd.github.diff"
        with httpx.Client(timeout=30.0, headers=headers) as client:
            resp = client.get(f"{base}/repos/{pr.repo}/pulls/{pr.number}")
            resp.raise_for_status()
            diff = resp.text
        return required_context(len(diff), len(knowledge))
    except Exception as e:
        print(f"fuko: could not size PR for context-fit, not gating: {e}", file=sys.stderr)
        return None


def review(pr_url: str, config_path: str = DEFAULT_CONFIG_PATH) -> InvokeResult:
    """Run a full review for ``pr_url``, failing over across the provider pool.

    Providers are tried in priority order, with those whose context window can't
    hold the job and those in circuit-breaker cooldown demoted to last resort.
    The first provider is pinned for the whole job; a throttle (429/quota/
    overload/timeout) trips its breaker and fails the job over to the next
    provider, while any other error fails fast (no failover).
    """
    cfg: FukoConfig = load_config(config_path)
    pr = parse_pr_url(pr_url)
    token = os.environ.get("GITHUB_TOKEN", "")
    api_url = os.environ.get("GITHUB_API_URL", _DEFAULT_API)

    backend = get_backend(cfg.review.backend, cfg.review)
    knowledge = build_knowledge(pr, token, api_url, cfg.knowledge)
    gh_env = _github_env(token)

    pool = resolve_pool(cfg.review)
    cooled = _cb_cooldowns()
    required = _estimate_required_context(pr, token, api_url, knowledge)
    ordered = order_pool(pool, cooled, required)

    result = InvokeResult(returncode=1, detail="no providers configured")
    for index, model in enumerate(ordered):
        preset = get_preset(model.provider)
        env = backend.build_env(preset, model, knowledge, cfg.review.tools)
        env.update(gh_env)

        label = f"{model.provider}/{model.name}"
        cooling = " (cooling — last resort)" if model.provider in cooled else ""
        print(
            f"fuko: review attempt {index + 1}/{len(ordered)} via {label}{cooling}",
            file=sys.stderr,
        )

        result = replace(backend.invoke(pr, env, cfg.review.tools), provider=model.provider)
        if not result.throttled:
            if result.returncode == 0:
                _normalize(backend, pr, model)
            return result

        _cb_trip(model.provider, cfg.review.cooldown_seconds, result.detail)
        print(
            f"fuko: {label} throttled ({result.detail}); breaker tripped, failing over",
            file=sys.stderr,
        )

    print("fuko: provider pool exhausted; all attempts throttled", file=sys.stderr)
    return result
