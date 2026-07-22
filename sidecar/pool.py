"""Provider-pool resolution and failover ordering (pure, no I/O).

The pool is the ordered list of providers a review may use. Order in the config
is priority: the first eligible provider is pinned for the whole job, and only a
throttle fails the job over to the next one. These helpers are pure so the
selection policy is unit-testable without a backend, a network, or a database.
"""

from __future__ import annotations

from collections.abc import Iterable

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
) -> tuple[list[ReviewModel], list[ReviewModel]]:
    """Split ``models`` into ``(actives, backups)``, preserving config order."""
    actives = [m for m in models if m.role == "active"]
    backups = [m for m in models if m.role == "backup"]
    return actives, backups


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
    provider id because the cooldown is global per provider (a shared API key).
    """
    pool = list(pool)

    def fits(model: ModelConfig) -> bool:
        return (
            required_tokens is None
            or model.max_context is None
            or model.max_context >= required_tokens
        )

    def rank(model: ModelConfig) -> tuple[int, int]:
        return (0 if fits(model) else 1, 0 if model.provider not in cooled else 1)

    return sorted(pool, key=rank)
