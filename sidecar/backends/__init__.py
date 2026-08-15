"""Review backend registry.

Maps a backend name (as used in ``.fuko.toml``) to its implementation. Adding a
backend is registering it here; the rest of the system depends only on the
``ReviewBackend`` protocol in :mod:`sidecar.backends.base`.
"""

from ..fukoconfig import ReviewConfig
from .agentic import AgenticBackend
from .base import ReviewBackend
from .pragent import PrAgentBackend

_BACKENDS: dict[str, type] = {
    PrAgentBackend.name: PrAgentBackend,
    AgenticBackend.name: AgenticBackend,
}


class UnknownBackendError(KeyError):
    """Raised when a ``.fuko.toml`` names a review backend that is not registered."""


def known_backends() -> frozenset[str]:
    """Return the registered backend names, for config-time validation.

    A names-only view so :mod:`sidecar.fukoconfig` can validate a config's
    ``backend`` fields without importing backend classes -- ``backends`` imports
    ``fukoconfig``, so the reverse edge must stay lazy (called from a validator at
    runtime, never at module import).
    """
    return frozenset(_BACKENDS)


def get_backend(name: str, config: ReviewConfig | None = None) -> ReviewBackend:
    """Return an instance of the registered backend ``name``, configured, or raise."""
    try:
        cls = _BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(_BACKENDS))
        raise UnknownBackendError(
            f"unknown review backend '{name}'; known backends: {known}"
        ) from None
    return cls(config)
