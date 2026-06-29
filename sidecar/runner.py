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

import hashlib
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx

from .backends import get_backend
from .backends.base import InvokeResult, PRRef
from .fukoconfig import (
    DEFAULT_CONFIG_PATH,
    CompareModel,
    FukoConfig,
    KnowledgeConfig,
    ModelConfig,
    ReviewConfig,
    load_config,
)
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


def _resolve_actor(token: str, api_url: str) -> str | None:
    """Return the GitHub actor identity a token authenticates as, or ``None``.

    Calls ``GET /user`` and returns the numeric account id as a string -- the
    stable identity two tokens share when they belong to the same user/bot, even
    if the tokens themselves differ.

    A **GitHub App installation token** cannot call ``GET /user`` (GitHub answers
    ``403`` "Resource not accessible by integration"), yet it authors comments as
    the app's own distinct ``<slug>[bot]`` user. Returning ``None`` there would
    wrongly collapse two different apps to "unresolvable" and disable concurrent
    A/B. So on a ``403`` we return a per-token surrogate: distinct app tokens get
    distinct identities (enabling concurrency) while the same token reused stays a
    single identity. Any other failure (network, auth, unexpected payload) returns
    ``None`` so callers fall back to the sequential path rather than guess.
    """
    if not token:
        return None
    base = api_url.rstrip("/")
    try:
        resp = httpx.get(f"{base}/user", headers=_gh_headers(token), timeout=30.0)
        if resp.status_code == 403:
            return "bot:" + hashlib.sha256(token.encode()).hexdigest()[:16]
        resp.raise_for_status()
        actor_id = resp.json().get("id")
    except (httpx.HTTPError, ValueError):
        return None
    return str(actor_id) if actor_id is not None else None


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


def _normalize(
    backend,
    pr: PRRef,
    model: ModelConfig,
    *,
    compare: bool = False,
    token: str | None = None,
    api_url: str | None = None,
) -> None:
    """Map the posted review into Review Signals for the winning provider's model.

    In A/B ``compare`` mode the backend additionally tags each newly marked inline
    comment with a visible label so the producing branch is legible on the diff. That
    visible label is the configured ``provider/name`` (matching the per-branch summary
    header), kept distinct from the litellm-prefixed ``model_id`` that feeds the
    invisible marker — so a ``zai-coding`` branch reads ``zai-coding/<name>`` on the
    diff rather than its litellm alias ``openai/<name>``.

    ``token``/``api_url`` pin the GitHub identity used to read and edit the comments;
    in concurrent A/B mode each branch passes its own so marking happens under that
    branch's identity. When unset the backend falls back to the process token.
    """
    try:
        preset = get_preset(model.provider)
        model_id = preset.litellm_prefix + model.name
        compare_label = f"{model.provider}/{model.name}" if compare else None
        signals = backend.normalize_output(
            pr, model=model_id, compare_label=compare_label, token=token, api_url=api_url
        )
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


_FRESH_COMMENT_ENV = {
    "PR_REVIEWER__PERSISTENT_COMMENT": "false",
    "PR_CODE_SUGGESTIONS__PERSISTENT_COMMENT": "false",
}


def _post_branch_header(pr: PRRef, token: str, api_url: str, label: str) -> None:
    """Post a model-labelled header issue comment for one A/B branch (best-effort).

    It gives a human a visible anchor for which model produced the summary that
    follows; a failure here must never abort the branch, so it only logs.
    """
    if not token:
        return
    base = api_url.rstrip("/")
    body = f"🤖 **fuko A/B** — model `{label}`"
    try:
        resp = httpx.post(
            f"{base}/repos/{pr.repo}/issues/{pr.number}/comments",
            json={"body": body},
            headers=_gh_headers(token),
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"fuko: could not post A/B branch header for {label}: {e}", file=sys.stderr)


def _run_pool(
    backend,
    pr: PRRef,
    knowledge: str,
    gh_env: dict[str, str],
    review: ReviewConfig,
    pool: list[ModelConfig],
    cooled: set[str],
    required: int | None,
    *,
    tools: list[str] | None = None,
    fresh_comment: bool = False,
    compare: bool = False,
    token: str | None = None,
    api_url: str | None = None,
) -> InvokeResult:
    """Run one review over ``pool`` with failover, normalizing the winner's output.

    Providers are tried in priority order, with those whose context window can't
    hold the job and those in circuit-breaker cooldown demoted to last resort. The
    first provider is pinned for the whole job; a throttle (429/quota/overload/
    timeout) trips its breaker and fails over, while any other error fails fast.
    ``tools`` overrides the configured tool list (used to drop ``describe`` in A/B
    mode), ``fresh_comment`` posts a new summary instead of updating PR-Agent's
    persistent one, and ``compare`` tags each branch's inline suggestions with a
    visible model label.

    ``token``/``api_url`` pin the GitHub identity that normalization reads/edits
    comments under; concurrent A/B branches pass their own so marking is
    author-separated. When unset normalization falls back to the process token.
    """
    tools = review.tools if tools is None else tools
    ordered = order_pool(pool, cooled, required)

    result = InvokeResult(returncode=1, detail="no providers configured")
    for index, model in enumerate(ordered):
        preset = get_preset(model.provider)
        env = backend.build_env(preset, model, knowledge, tools)
        env.update(gh_env)
        if fresh_comment:
            env.update(_FRESH_COMMENT_ENV)

        label = f"{model.provider}/{model.name}"
        cooling = " (cooling — last resort)" if model.provider in cooled else ""
        print(
            f"fuko: review attempt {index + 1}/{len(ordered)} via {label}{cooling}",
            file=sys.stderr,
        )

        result = replace(backend.invoke(pr, env, tools), provider=model.provider)
        if not result.throttled:
            if result.returncode == 0:
                _normalize(backend, pr, model, compare=compare, token=token, api_url=api_url)
            return result

        _cb_trip(model.provider, review.cooldown_seconds, result.detail)
        print(
            f"fuko: {label} throttled ({result.detail}); breaker tripped, failing over",
            file=sys.stderr,
        )

    print("fuko: provider pool exhausted; all attempts throttled", file=sys.stderr)
    return result


def _resolve_branch_identities(compare: list[CompareModel], api_url: str) -> list[str] | None:
    """Return one distinct-identity GitHub token per branch, or ``None`` (sequential).

    Concurrent A/B mode is all-or-nothing and needs no config flag: it activates
    *iff* every compare entry names a ``token_env`` whose env var resolves to a
    non-empty value **and** the tokens resolve to distinct GitHub *actors*. If any
    branch lacks ``token_env``, its env var is unset/empty, an actor lookup fails,
    or two branches resolve to the same actor, the whole run falls back to the
    sequential single-token path (``None``).

    Distinctness is by resolved actor identity (``GET /user`` id), not raw token
    value: two different tokens (e.g. two PATs for the same bot user) share one
    identity and would race on each other's comments exactly as a single token
    does, so they must not enable concurrency.
    """
    tokens: list[str] = []
    actors: list[str] = []
    for entry in compare:
        if not entry.token_env:
            return None
        value = os.environ.get(entry.token_env, "")
        if not value:
            return None
        actor = _resolve_actor(value, api_url)
        if actor is None:
            return None
        tokens.append(value)
        actors.append(actor)
    if len(set(actors)) != len(actors):
        return None
    return tokens


def _run_compare_branch(
    backend,
    pr: PRRef,
    knowledge: str,
    review: ReviewConfig,
    entry: CompareModel,
    cooled: set[str],
    required: int | None,
    tools: list[str],
    api_url: str,
    token: str,
) -> tuple[str, InvokeResult]:
    """Run one A/B branch end-to-end under its own ``token`` identity.

    Posts the branch's model-labelled header, then its fresh summary + inline
    suggestions, marking and editing only under ``token`` so GitHub's permissions
    stop this branch touching another branch's comments. Returns ``(label, result)``.
    Any exception is captured as a failed result so one branch's failure can never
    abort or corrupt a sibling running concurrently.
    """
    label = f"{entry.provider}/{entry.name}"
    try:
        _post_branch_header(pr, token, api_url, label)
        result = _run_pool(
            backend,
            pr,
            knowledge,
            _github_env(token),
            review,
            [entry],
            cooled,
            required,
            tools=tools,
            fresh_comment=True,
            compare=True,
            token=token,
            api_url=api_url,
        )
    except Exception as e:
        print(f"fuko: A/B branch {label} failed in isolation: {e}", file=sys.stderr)
        return label, InvokeResult(returncode=1, detail=f"{label} errored: {e}")
    return label, result


def _review_compare(
    backend,
    pr: PRRef,
    knowledge: str,
    gh_env: dict[str, str],
    review: ReviewConfig,
    token: str,
    api_url: str,
    cooled: set[str],
    required: int | None,
) -> InvokeResult:
    """Review ``pr`` once per ``review.compare`` entry for an A/B model comparison.

    Two execution modes, auto-selected by :func:`_resolve_branch_identities`:

    - **Concurrent** (every branch has a distinct, resolvable ``token_env``): the
      branches run in a thread pool, one thread per branch, each posting and editing
      under its own GitHub identity. Total wall-clock is the slowest single branch,
      and author separation plus idempotent marking keep comments uncrossed.
    - **Sequential** (the default; any branch lacks a distinct token): branches run
      one after another under the shared token exactly as before, marker injection
      staying idempotent so a later branch never relabels an earlier one's
      suggestions.

    ``describe`` is dropped in both modes because a PR has one description the
    branches would otherwise overwrite. The overall result is green when any branch
    posted a review.
    """
    tools = [t for t in review.tools if t != "describe"]
    if "describe" in review.tools:
        print(
            "fuko: A/B compare mode — 'describe' disabled (a PR has one description)",
            file=sys.stderr,
        )
    if not tools:
        return InvokeResult(
            returncode=1,
            detail="A/B compare disables 'describe'; configure at least one non-describe tool",
        )

    identities = _resolve_branch_identities(review.compare, api_url)
    if identities is not None:
        print(
            f"fuko: A/B compare mode — running {len(review.compare)} branches concurrently "
            "under per-branch identities",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=len(review.compare)) as pool:
            futures = [
                pool.submit(
                    _run_compare_branch,
                    backend,
                    pr,
                    knowledge,
                    review,
                    entry,
                    cooled,
                    required,
                    tools,
                    api_url,
                    branch_token,
                )
                for entry, branch_token in zip(review.compare, identities)
            ]
            outcomes = [f.result() for f in futures]
    else:
        outcomes = []
        for index, entry in enumerate(review.compare):
            label = f"{entry.provider}/{entry.name}"
            print(f"fuko: A/B branch {index + 1}/{len(review.compare)}: {label}", file=sys.stderr)
            _post_branch_header(pr, token, api_url, label)
            result = _run_pool(
                backend,
                pr,
                knowledge,
                gh_env,
                review,
                [entry],
                cooled,
                required,
                tools=tools,
                fresh_comment=True,
                compare=True,
                token=token,
                api_url=api_url,
            )
            outcomes.append((label, result))

    detail = "; ".join(f"{label}: {r.detail or 'ok'}" for label, r in outcomes)
    rc = 0 if any(r.returncode == 0 for _, r in outcomes) else 1
    return InvokeResult(returncode=rc, detail=detail)


def _warn_compare_overrides(review: ReviewConfig) -> None:
    """Warn when A/B compare silently sidelines a configured failover pool or model.

    ``[[review.compare]]`` wins by strict precedence over ``[[review.providers]]``
    and ``[review.model]`` (each branch is a single model; A/B deliberately does no
    failover). That override is intended, but should not be silent: a user who has a
    working failover pool and later adds compare entries would otherwise lose
    failover with no signal. Emits a one-line stderr warning naming what is ignored.

    No-op unless ``[[review.compare]]`` is actually set, so the helper stays
    self-consistent if called outside the guarded ``review()`` dispatch.
    """
    if not review.compare:
        return
    if review.providers:
        print(
            f"fuko: A/B compare mode active — the {len(review.providers)}-provider "
            "failover pool ([[review.providers]]) is ignored (compare runs each "
            "branch as a single model with no failover)",
            file=sys.stderr,
        )
    elif "model" in review.model_fields_set:
        print(
            "fuko: A/B compare mode active — the single [review.model] is ignored "
            "(compare reviews only the listed [[review.compare]] models)",
            file=sys.stderr,
        )


def review(pr_url: str, config_path: str = DEFAULT_CONFIG_PATH) -> InvokeResult:
    """Run a full review for ``pr_url`` through the configured backend.

    With a single model (or a ``[[review.providers]]`` pool) the PR is reviewed
    once, failing over across the pool on throttling. When ``[[review.compare]]``
    is set, the PR is instead A/B'd once per listed model (see
    :func:`_review_compare`).
    """
    cfg: FukoConfig = load_config(config_path)
    pr = parse_pr_url(pr_url)
    token = os.environ.get("GITHUB_TOKEN", "")
    api_url = os.environ.get("GITHUB_API_URL", _DEFAULT_API)

    backend = get_backend(cfg.review.backend, cfg.review)
    knowledge = build_knowledge(pr, token, api_url, cfg.knowledge)
    gh_env = _github_env(token)

    cooled = _cb_cooldowns()
    required = _estimate_required_context(pr, token, api_url, knowledge)

    if cfg.review.compare:
        _warn_compare_overrides(cfg.review)
        return _review_compare(
            backend, pr, knowledge, gh_env, cfg.review, token, api_url, cooled, required
        )

    pool = resolve_pool(cfg.review)
    return _run_pool(backend, pr, knowledge, gh_env, cfg.review, pool, cooled, required)
