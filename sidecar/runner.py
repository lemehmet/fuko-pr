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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx

from .backends import get_backend
from .backends.base import InvokeResult, PRRef
from .fukoconfig import (
    DEFAULT_CONFIG_PATH,
    FukoConfig,
    KnowledgeConfig,
    ModelConfig,
    ReviewConfig,
    ReviewModel,
    load_config,
)
from .pool import order_pool, partition_roles, resolve_models
from .presets import get_preset
from .status import escalation_needed
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

    A **GitHub App installation token** cannot call ``GET /user`` -- GitHub answers
    ``403`` with the specific message "Resource not accessible by integration" --
    yet it authors comments as the app's own distinct ``<slug>[bot]`` user.
    Returning ``None`` there would wrongly collapse two different apps to
    "unresolvable" and disable concurrent A/B. So on *that* 403 we return a
    per-token surrogate: distinct app tokens get distinct identities (enabling
    concurrency) while the same token reused stays a single identity. The match is
    narrowed to that exact integration message because a bare ``403`` also covers
    rate limits, SSO/org restrictions, and under-scoped PATs -- which are real
    failures that must fall back to ``None`` (sequential) rather than fabricate a
    distinct identity for what may be one actor. Any other failure (network, auth,
    unexpected payload) likewise returns ``None``.
    """
    if not token:
        return None
    base = api_url.rstrip("/")
    try:
        resp = httpx.get(f"{base}/user", headers=_gh_headers(token), timeout=30.0)
        if resp.status_code == 403:
            try:
                payload = resp.json()
            except ValueError:
                payload = None
            message = payload.get("message", "") if isinstance(payload, dict) else ""
            if "not accessible by integration" in message.lower():
                return "bot:" + hashlib.sha256(token.encode()).hexdigest()
            return None
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


def _slot_of(model: ModelConfig) -> str | None:
    """Derive the human slot name from a model's ``token_env`` (None when absent).

    ``FUKO_GITHUB_TOKEN_DORIAN`` -> ``dorian``; an entry without a dedicated
    identity (solo configs, backups) has no slot of its own.
    """
    token_env = getattr(model, "token_env", None) or ""
    if not token_env:
        return None
    return token_env.removeprefix("FUKO_GITHUB_TOKEN_").lower() or None


def _record_run(
    pr: PRRef,
    model: ModelConfig,
    *,
    slot: str | None,
    duration_s: float,
    attempts: int,
    outcome: str,
    findings: int | None,
    detail: str,
) -> None:
    """Persist one review-run metrics row (best-effort, never raises).

    Reaches the sidecar over HTTP when ``FUKO_URL`` is set, else the local
    Postgres module; both degrade to no-ops. Metrics are an observability
    layer -- a failure here must never affect the review that produced them.
    """
    try:
        fuko_url, fuko_token = _cb_endpoint()
        payload = {
            "repo": pr.repo,
            "pr": pr.number,
            "provider": model.provider,
            "model": model.name,
            "slot": slot,
            "duration_s": round(duration_s, 1),
            "attempts": attempts,
            "outcome": outcome,
            "findings": findings,
            "detail": (detail or "")[:500],
        }
        if fuko_url:
            headers = {"Content-Type": "application/json"}
            if fuko_token:
                headers["Authorization"] = "Bearer " + fuko_token
            resp = httpx.post(
                fuko_url.rstrip("/") + "/metrics/run", json=payload, headers=headers, timeout=10.0
            )
            resp.raise_for_status()
        else:
            from .run_metrics import record

            record(
                pr.repo,
                pr.number,
                model.provider,
                model.name,
                slot=slot,
                duration_s=payload["duration_s"],
                attempts=attempts,
                outcome=outcome,
                findings=findings,
                detail=payload["detail"],
            )
    except Exception as e:
        print(f"fuko: run-metrics record failed (continuing): {e}", file=sys.stderr)


def _rh_states(repo: str) -> list[dict]:
    """Last observed reviewer-health rows for ``repo`` (best-effort).

    Reads the shared state from the sidecar over HTTP when ``FUKO_URL`` is set,
    else from the local Postgres. Any failure yields an empty list -- escalation
    is an optimization, so a read error must never block a review.
    """
    try:
        fuko_url, fuko_token = _cb_endpoint()
        if fuko_url:
            headers = {"Authorization": "Bearer " + fuko_token} if fuko_token else {}
            resp = httpx.get(
                fuko_url.rstrip("/") + "/rh/state",
                params={"repo": repo},
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("reviewers") if isinstance(payload, dict) else None
            return data if isinstance(data, list) else []
        from .reviewer_health import states

        return states(repo)
    except Exception as e:
        print(f"fuko: reviewer-health read failed, no escalation this round: {e}", file=sys.stderr)
        return []


def _observe_reviewer_health(pr: PRRef, token: str, api_url: str) -> None:
    """Record what each external reviewer did on this PR's HEAD (best-effort).

    Called at the end of a review run so the NEXT round can escalate on what
    this one saw (next-round escalation). Fetches the PR artifacts fresh --
    CodeRabbit/Copilot typically post while fuko's own review runs -- and
    persists via the sidecar or the local Postgres. Every failure is swallowed:
    an observation must never fail the review that produced it.
    """
    from .status import reviewer_states

    try:
        head = fetch_pr_head(pr, token, api_url)
        issue_comments = fetch_issue_comments(pr, token, api_url)
        reviews = fetch_reviews(pr, token, api_url)
        try:
            check_runs = fetch_check_runs(pr, head, token, api_url)
        except Exception:
            check_runs = None
        rows = reviewer_states(head, issue_comments, reviews, check_runs)
    except Exception as e:
        print(f"fuko: reviewer-health observation skipped: {e}", file=sys.stderr)
        return

    try:
        fuko_url, fuko_token = _cb_endpoint()
        if fuko_url:
            headers = {"Content-Type": "application/json"}
            if fuko_token:
                headers["Authorization"] = "Bearer " + fuko_token
            resp = httpx.post(
                fuko_url.rstrip("/") + "/rh/observe",
                json={
                    "repo": pr.repo,
                    "pr": pr.number,
                    "observations": [
                        {
                            "reviewer": r["backend"],
                            "state": r["state"],
                            "detail": (r.get("detail") or "")[:500],
                        }
                        for r in rows
                    ],
                },
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
        else:
            from .reviewer_health import observe

            for r in rows:
                observe(pr.repo, r["backend"], r["state"], pr.number, r.get("detail") or "")
        summary = ", ".join(f"{r['backend']}={r['state']}" for r in rows)
        print(f"fuko: reviewer health observed: {summary}", file=sys.stderr)
    except Exception as e:
        print(f"fuko: reviewer-health observation failed (continuing): {e}", file=sys.stderr)


def _normalize(
    backend,
    pr: PRRef,
    model: ModelConfig,
    *,
    compare: bool = False,
    token: str | None = None,
    api_url: str | None = None,
    actor: str | None = None,
) -> int | None:
    """Map the posted review into Review Signals for the winning provider's model.

    Returns the signal count, or ``None`` when the backend has no normalization
    -- the count feeds the review-run metrics row (``findings``).

    In A/B ``compare`` mode the backend additionally tags each newly marked inline
    comment with a visible label so the producing branch is legible on the diff. That
    visible label is the configured ``provider/name`` (matching the per-branch summary
    header), kept distinct from the litellm-prefixed ``model_id`` that feeds the
    invisible marker — so a ``zai-coding`` branch reads ``zai-coding/<name>`` on the
    diff rather than its litellm alias ``openai/<name>``.

    ``token``/``api_url`` pin the GitHub identity used to read and edit the comments;
    in concurrent A/B mode each branch passes its own so marking happens under that
    branch's identity. ``actor`` is that identity's user id (from the branch header
    post) so the backend can filter marking to this branch's own comments without a
    ``GET /user`` probe, which 403s for App installation tokens (#66). When unset
    the backend falls back to the process token.
    """
    try:
        preset = get_preset(model.provider)
        model_id = preset.litellm_prefix + model.name
        compare_label = f"{model.provider}/{model.name}" if compare else None
        signals = backend.normalize_output(
            pr,
            model=model_id,
            compare_label=compare_label,
            token=token,
            api_url=api_url,
            actor=actor,
        )
        print(f"fuko: normalized {len(signals)} review signals", file=sys.stderr)
        return len(signals)
    except NotImplementedError:
        return None


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


def _post_branch_header(pr: PRRef, token: str, api_url: str, label: str) -> str | None:
    """Post a model-labelled header issue comment for one A/B branch (best-effort).

    It gives a human a visible anchor for which model produced the summary that
    follows; a failure here must never abort the branch, so it only logs.

    Returns the posting identity's user id as reported by the create-comment
    response -- the one identity probe that works for App installation tokens
    (``GET /user`` 403s for them, #57) -- so marker injection can restrict
    itself to this branch's own comments (#66). ``None`` when nothing was
    posted or the id is unavailable.
    """
    if not token:
        return None
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
        actor_id = ((resp.json() or {}).get("user") or {}).get("id")
        return str(actor_id) if actor_id is not None else None
    except (httpx.HTTPError, ValueError) as e:
        print(f"fuko: could not post A/B branch header for {label}: {e}", file=sys.stderr)
        return None


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
    actor: str | None = None,
    slot: str | None = None,
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
    author-separated. ``actor`` is that identity's user id when already known
    (from the branch header post) — required for author-scoped marking under
    App installation tokens, whose ``GET /user`` probe 403s (#66). When unset
    normalization falls back to the process token.
    """
    tools = review.tools if tools is None else tools
    ordered = order_pool(pool, cooled, required)
    started = time.monotonic()

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
            findings = None
            if result.returncode == 0:
                findings = _normalize(
                    backend, pr, model, compare=compare, token=token, api_url=api_url, actor=actor
                )
            _record_run(
                pr,
                model,
                slot=slot,
                duration_s=time.monotonic() - started,
                attempts=index + 1,
                outcome="ok" if result.returncode == 0 else "failed",
                findings=findings,
                detail=result.detail or "",
            )
            return result

        _cb_trip(model.provider, review.cooldown_seconds, result.detail)
        print(
            f"fuko: {label} throttled ({result.detail}); breaker tripped, failing over",
            file=sys.stderr,
        )

    print("fuko: provider pool exhausted; all attempts throttled", file=sys.stderr)
    if ordered:
        _record_run(
            pr,
            ordered[-1],
            slot=slot,
            duration_s=time.monotonic() - started,
            attempts=len(ordered),
            outcome="throttled_out",
            findings=None,
            detail=result.detail or "",
        )
    return result


def _resolve_branch_identities(actives: list[ReviewModel], api_url: str) -> list[str] | None:
    """Return one distinct-identity GitHub token per branch, or ``None`` (sequential).

    Concurrent A/B mode is all-or-nothing and needs no config flag: it activates
    *iff* every active entry names a ``token_env`` whose env var resolves to a
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
    for entry in actives:
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
    entry: ReviewModel,
    backups: list[ReviewModel],
    cooled: set[str],
    required: int | None,
    tools: list[str],
    api_url: str,
    token: str,
) -> tuple[str, InvokeResult]:
    """Run one A/B branch end-to-end under its own ``token`` identity.

    Posts the branch's model-labelled header, then its fresh summary + inline
    suggestions, with marker injection restricted to the ``actor`` identity the
    header post revealed -- repo-write tokens CAN edit a sibling's comments
    (#66), so authorship is filtered, not assumed. The branch's pool is its
    active entry followed by the shared ``backups``, so a throttled primary fails
    over instead of losing the round. Returns ``(label, result)``. Any exception
    is captured as a failed result so one branch's failure can never abort or
    corrupt a sibling running concurrently.
    """
    label = f"{entry.provider}/{entry.name}"
    try:
        actor = _post_branch_header(pr, token, api_url, label)
        result = _run_pool(
            backend,
            pr,
            knowledge,
            _github_env(token),
            review,
            [entry, *backups],
            cooled,
            required,
            tools=tools,
            fresh_comment=True,
            compare=True,
            token=token,
            api_url=api_url,
            actor=actor,
            slot=_slot_of(entry),
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
    actives: list[ReviewModel],
    backups: list[ReviewModel],
    token: str,
    api_url: str,
    cooled: set[str],
    required: int | None,
) -> InvokeResult:
    """Review ``pr`` once per active model for an A/B comparison.

    Two execution modes, auto-selected by :func:`_resolve_branch_identities`:

    - **Concurrent** (every branch has a distinct, resolvable ``token_env``): the
      branches run in a thread pool, one thread per branch, each posting and editing
      under its own GitHub identity. Total wall-clock is the slowest single branch,
      and author separation plus idempotent marking keep comments uncrossed.
    - **Sequential** (the default; any branch lacks a distinct token): branches run
      one after another under the shared token exactly as before, marker injection
      staying idempotent so a later branch never relabels an earlier one's
      suggestions.

    Each branch's pool is its own active entry followed by the shared ``backups``,
    so a throttled primary fails over instead of losing the round. Two branches
    whose primaries both throttle may converge on the same backup model; their
    posts stay separable by identity (concurrent) or model label (sequential).

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

    identities = _resolve_branch_identities(actives, api_url)
    if identities is not None:
        print(
            f"fuko: A/B compare mode — running {len(actives)} branches concurrently "
            "under per-branch identities",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=len(actives)) as pool:
            futures = [
                pool.submit(
                    _run_compare_branch,
                    backend,
                    pr,
                    knowledge,
                    review,
                    entry,
                    backups,
                    cooled,
                    required,
                    tools,
                    api_url,
                    branch_token,
                )
                for entry, branch_token in zip(actives, identities)
            ]
            outcomes = [f.result() for f in futures]
    else:
        outcomes = []
        for index, entry in enumerate(actives):
            label = f"{entry.provider}/{entry.name}"
            print(f"fuko: A/B branch {index + 1}/{len(actives)}: {label}", file=sys.stderr)
            actor = _post_branch_header(pr, token, api_url, label)
            result = _run_pool(
                backend,
                pr,
                knowledge,
                gh_env,
                review,
                [entry, *backups],
                cooled,
                required,
                tools=tools,
                fresh_comment=True,
                compare=True,
                token=token,
                api_url=api_url,
                actor=actor,
                slot=_slot_of(entry),
            )
            outcomes.append((label, result))

    detail = "; ".join(f"{label}: {r.detail or 'ok'}" for label, r in outcomes)
    rc = 0 if any(r.returncode == 0 for _, r in outcomes) else 1
    return InvokeResult(returncode=rc, detail=detail)


def _warn_legacy_config(review: ReviewConfig) -> None:
    """Warn when deprecated model sections are in play, or silently overridden.

    ``[[review.models]]`` is the canonical surface. When it is set alongside any
    deprecated section (``[[review.compare]]``, ``[[review.providers]]``, or an
    explicitly written ``[review.model]``), the deprecated sections are ignored
    -- say so, or the user gets a silent surprise. When only deprecated sections
    are set they keep working through :func:`sidecar.pool.resolve_models`, but
    each run nudges toward migrating. Silent when nothing deprecated was written
    (the implicit default model is not a migration candidate).
    """
    legacy = []
    if review.compare:
        legacy.append("[[review.compare]]")
    if review.providers:
        legacy.append("[[review.providers]]")
    if "model" in review.model_fields_set:
        legacy.append("[review.model]")
    if not legacy:
        return
    joined = ", ".join(legacy)
    if review.models:
        print(
            f"fuko: [[review.models]] is set — deprecated {joined} ignored",
            file=sys.stderr,
        )
    else:
        print(
            f"fuko: {joined} deprecated — migrate to [[review.models]] entries "
            'with role = "active" | "backup"',
            file=sys.stderr,
        )


def review(pr_url: str, config_path: str = DEFAULT_CONFIG_PATH) -> InvokeResult:
    """Run a full review for ``pr_url`` through the configured backend.

    The unified ``[[review.models]]`` list drives the run: with one active entry
    the PR is reviewed once, failing over across the backups on throttling; with
    several actives the PR is A/B'd once per active, each branch sharing the same
    backups (see :func:`_review_compare`). Deprecated sections are mapped onto
    the unified list by :func:`sidecar.pool.resolve_models`.

    Next-round escalation: when the previous round observed an external reviewer
    in a degraded state (:func:`sidecar.status.escalation_needed` over the
    persisted :mod:`sidecar.reviewer_health` rows), every backup is promoted to
    active for this round -- the PR is losing outside review coverage, so the
    extra models run as their own branches instead of waiting for a throttle.
    A promoted backup without a distinct ``token_env`` identity makes the round
    run sequentially (the concurrency gate is all-or-nothing); escalated rounds
    are expected to be rare enough that this is an acceptable trade. At the end
    of every run the current reviewer states are observed and persisted for the
    next round (:func:`_observe_reviewer_health`).
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

    _warn_legacy_config(cfg.review)
    actives, backups = partition_roles(resolve_models(cfg.review))

    if backups and escalation_needed(_rh_states(pr.repo)):
        promoted = ", ".join(f"{m.provider}/{m.name}" for m in backups)
        print(
            "fuko: external reviewers degraded last round — promoting backup "
            f"model(s) to active for this round: {promoted}",
            file=sys.stderr,
        )
        actives, backups = [*actives, *backups], []

    if len(actives) > 1:
        result = _review_compare(
            backend,
            pr,
            knowledge,
            gh_env,
            cfg.review,
            actives,
            backups,
            token,
            api_url,
            cooled,
            required,
        )
    else:
        result = _run_pool(
            backend,
            pr,
            knowledge,
            gh_env,
            cfg.review,
            [*actives, *backups],
            cooled,
            required,
            slot=_slot_of(actives[0]) if actives else None,
        )

    _observe_reviewer_health(pr, token, api_url)
    return result
