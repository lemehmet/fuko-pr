"""Provider-pool resolution and failover ordering (pure, no I/O).

The pool is the ordered list of providers a review may use. Order in the config
is priority: the first eligible provider is pinned for the whole job, and only a
throttle fails the job over to the next one. These helpers are pure so the
selection policy is unit-testable without a backend, a network, or a database.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .fukoconfig import ModelConfig, ReviewConfig, ReviewModel


def resolve_models(review: ReviewConfig) -> list[ReviewModel]:
    """Return the unified model list, mapping deprecated sections when needed.

    ``[[review.models]]`` is the canonical surface and wins when non-empty. The
    deprecated sections map onto it losslessly: ``[[review.compare]]`` becomes
    all-active entries, ``[[review.providers]]`` becomes its first entry active
    with the rest as backups (config order preserved, so failover priority is
    unchanged), and the single ``[review.model]`` becomes one active entry.
    Precedence among the deprecated sections matches the old dispatch:
    ``compare`` over ``providers`` over ``model``.
    """
    if review.models:
        return list(review.models)
    if review.compare:
        return [ReviewModel(**m.model_dump()) for m in review.compare]
    if review.providers:
        return [
            ReviewModel(**m.model_dump(), role="active" if index == 0 else "backup")
            for index, m in enumerate(review.providers)
        ]
    return [ReviewModel(**review.model.model_dump())]


def partition_roles(
    models: Iterable[ReviewModel],
) -> tuple[list[ReviewModel], list[ReviewModel], list[ReviewModel]]:
    """Split ``models`` into ``(actives, backups, trials)``, preserving config order.

    Actives and trials both start a review branch each run (the caller runs
    ``actives + trials`` as branches); backups are the shared failover pool.
    The active/trial distinction is carried per-branch by ``ReviewModel.role``
    so downstream can gate on actives while surfacing trials non-gating.
    """
    materialized = list(models)
    actives = [m for m in materialized if m.role == "active"]
    backups = [m for m in materialized if m.role == "backup"]
    trials = [m for m in materialized if m.role == "trial"]
    return actives, backups, trials


def order_pool(
    pool: Iterable[ModelConfig],
    cooled: set[str],
    required_tokens: int | None = None,
) -> list[ModelConfig]:
    """Order ``pool`` for a failover attempt.

    Ranked first by context fit, then by cooldown, with config order (priority)
    preserved within a tier: a provider whose ``max_context`` cannot hold the job
    is ranked last (a definite truncation, only a last resort), then a provider
    in cooldown is ranked after fitting/available ones. So the order is
    fits+available > fits+cooled > too-small+available > too-small+cooled. A
    provider with no ``max_context`` is assumed to fit, and ``required_tokens=None``
    disables the fit check (cooldown-only ordering). ``cooled`` is keyed by
    :func:`breaker_key` because the cooldown is global per credential (an API
    key shared across repos), so a cooling credential does not demote its
    siblings.
    """
    pool = list(pool)

    def fits(model: ModelConfig) -> bool:
        return (
            required_tokens is None
            or model.max_context is None
            or model.max_context >= required_tokens
        )

    def rank(model: ModelConfig) -> tuple[int, int]:
        return (0 if fits(model) else 1, 0 if breaker_key(model) not in cooled else 1)

    return sorted(pool, key=rank)


def breaker_key(model: ModelConfig) -> str:
    """Return the circuit-breaker key for ``model``'s credential.

    Credential 1 keeps the bare provider id, so single-key deployments and
    existing cooldown rows are unchanged; credential N is ``provider#N``. The
    index only -- a key, its hash or a prefix of it must never land in a table
    the metrics page renders.
    """
    if model.credential == 1:
        return model.provider
    return f"{model.provider}#{model.credential}"


def expand_credentials(
    pool: Iterable[ModelConfig], count_of: Callable[[ModelConfig], int]
) -> list[ModelConfig]:
    """Expand each pool entry into one attempt per provider credential.

    ``count_of`` says how many credentials an entry can authenticate with; an
    entry with zero or one stays a single attempt, untouched. The copies sit
    next to each other in credential order, which makes the existing failover
    loop drain them SEQUENTIALLY: credential 2 is never sent a request until
    credential 1 throttles. That is the point rather than a side effect -- a
    quota window's clock starts at first use, so round-robin would start every
    window together and exhaust them together (#298).
    """
    expanded: list[ModelConfig] = []
    for model in pool:
        expanded.extend(model.on_credential(i) for i in range(1, max(1, count_of(model)) + 1))
    return expanded
