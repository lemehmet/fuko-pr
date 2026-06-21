"""Provider-pool resolution and failover ordering (pure, no I/O).

The pool is the ordered list of providers a review may use. Order in the config
is priority: the first eligible provider is pinned for the whole job, and only a
throttle fails the job over to the next one. These helpers are pure so the
selection policy is unit-testable without a backend, a network, or a database.
"""

from __future__ import annotations

from collections.abc import Iterable

from .fukoconfig import ModelConfig, ReviewConfig


def resolve_pool(review: ReviewConfig) -> list[ModelConfig]:
    """Return the effective provider pool.

    Uses ``[[review.providers]]`` when present, otherwise the single
    ``[review.model]`` as a one-entry pool -- so the legacy single-model config
    keeps working unchanged.
    """
    return list(review.providers) if review.providers else [review.model]


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
