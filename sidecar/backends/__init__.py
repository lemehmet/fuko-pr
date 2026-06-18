"""Review backend registry.

Maps a backend name (as used in ``.fuko.toml``) to its implementation. Adding a
backend is registering it here; the rest of the system depends only on the
``ReviewBackend`` protocol in :mod:`sidecar.backends.base`.
"""

from ..fukoconfig import ReviewConfig
from .base import ReviewBackend
from .pragent import PrAgentBackend

_BACKENDS: dict[str, type] = {
    PrAgentBackend.name: PrAgentBackend,
}


class UnknownBackendError(KeyError):
    """Raised when a ``.fuko.toml`` names a review backend that is not registered."""


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
