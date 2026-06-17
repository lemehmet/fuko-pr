"""OpenAI-compatible embeddings client (default: a local Ollama model)."""

import httpx

from .config import settings


class EmbedError(RuntimeError):
    """Raised when an embeddings request fails or returns no data."""


class Embedder:
    """Embed text via an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self) -> None:
        """Configure the endpoint URL, auth header, and HTTP client."""
        self._client = httpx.Client(timeout=120.0)
        self._dim: int | None = None
        self._url = f"{settings.embed_base_url.rstrip('/')}/embeddings"
        self._headers = {"Content-Type": "application/json"}
        if settings.embed_api_key:
            self._headers["Authorization"] = f"Bearer {settings.embed_api_key}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input."""
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), settings.embed_batch_size):
            batch = texts[i : i + settings.embed_batch_size]
            try:
                resp = self._client.post(
                    self._url,
                    headers=self._headers,
                    json={"model": settings.embed_model, "input": batch},
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                body = ""
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                    body = e.response.text[:1000]
                raise EmbedError(
                    f"embedding request to {self._url} failed: {e}\nbody: {body}"
                ) from e
            data = resp.json().get("data")
            if not data:
                raise EmbedError(f"empty 'data' in embeddings response from {self._url}")
            out.extend(d["embedding"] for d in data)
        return out

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string."""
        return self.embed([text])[0]

    def probe_dim(self) -> int:
        """Return the embedding dimension reported by the configured model.

        Embeds a tiny probe string and caches the resulting vector length, so
        the pgvector column can be sized to whatever the model actually returns
        instead of a hard-coded guess.
        """
        if self._dim is None:
            self._dim = len(self.embed_one("dimension probe"))
        return self._dim


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the process-wide singleton ``Embedder``."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
