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


def order_pool(pool: Iterable[ModelConfig], cooled: set[str]) -> list[ModelConfig]:
    """Order ``pool`` for a failover attempt.

    Eligible providers (whose ``provider`` is not in ``cooled``) come first in
    priority order; providers currently in cooldown are appended as a last
    resort, so a fully-cooled pool is still attempted rather than failing
    outright. ``cooled`` is keyed by provider id because the cooldown is global
    per provider (a shared API key), so two pool entries on the same provider are
    both treated as cooling.
    """
    pool = list(pool)
    eligible = [m for m in pool if m.provider not in cooled]
    cooling = [m for m in pool if m.provider in cooled]
    return eligible + cooling
