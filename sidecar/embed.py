"""OpenAI-compatible embeddings client (default: a local Ollama model)."""

import httpx

from .config import settings


class EmbedError(RuntimeError):
    """Raised when an embeddings request fails or returns no data."""


class EmbedModelMismatch(EmbedError):
    """Raised when the endpoint does not serve the configured embedding model."""


def _fit(text: str) -> str:
    # Last line of defence, deliberately at the transport edge rather than at
    # each call site: every path that reaches an embedding endpoint goes
    # through here, so no future caller can reintroduce the 500 that a 12k-token
    # PR body used to cause. Truncating only what is *embedded* is safe -- the
    # full text is what gets stored and what a review reads back.
    return text[: settings.embed_max_chars]


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
        """Embed a batch of texts, returning one vector per input.

        Each input is truncated to ``embed_max_chars`` before it is sent. An
        embedding endpoint rejects an over-long input outright rather than
        truncating it for you, and the model behind it has a fixed context, so
        the choice is between a shorter vector and no vector at all.
        """
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), settings.embed_batch_size):
            batch = [_fit(t) for t in texts[i : i + settings.embed_batch_size]]
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
        """Embed a single string as a *document*."""
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single string as a *query*, with the model's task instruction.

        Kept separate from :meth:`embed_one` rather than handled by a flag
        because the asymmetry is easy to get half-right: prefixing both sides,
        or neither, still returns 1024 well-formed dimensions and degrades
        retrieval quietly. One method per side makes each call site say which
        it means.
        """
        return self.embed([settings.embed_query_prefix + text])[0]

    def served_models(self) -> list[str] | None:
        """Return the model names the endpoint reports, or ``None`` if it will not say.

        ``None`` means *cannot verify*, which is deliberately different from
        *mismatch*: an endpoint that is down, does not implement ``/models``, or
        answers with something unparseable must not be able to stop the sidecar
        from starting.

        Two response shapes are accepted because both are in use here. OpenAI's
        is ``{"data": [{"id": ...}]}``; Ollama's is ``{"models": [{"name": ...}]}``.
        llama.cpp answers with both keys at once, but a stricter OpenAI endpoint
        sends only the first.
        """
        try:
            resp = self._client.get(
                f"{settings.embed_base_url.rstrip('/')}/models",
                headers=self._headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(body, dict):
            return None
        names: list[str] = []
        for key, fields in (("data", ("id",)), ("models", ("name", "model"))):
            entries = body.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for field in fields:
                    value = entry.get(field)
                    if isinstance(value, str) and value:
                        names.append(value)
                        break
        return names or None

    def verify_model(self) -> None:
        """Raise if the endpoint definitively does not serve the configured model.

        The request's ``model`` field cannot be trusted to mean anything:
        llama-server ignores it and answers with whatever it has loaded, echoing
        the requested name back. That is not a hypothetical -- a server-side
        model swap under a constant alias silently mixed two embedding spaces
        for a day (see #220), invisible to both the dimension check and the
        ``meta.embed_model`` marker, because the marker records the name that
        was *asked for*.

        So ask the endpoint what it actually has. Silent when it will not say,
        loud only when it answers and the configured model is not in the list.

        Raises:
            EmbedModelMismatch: The endpoint serves models, none of them the
                configured one.
        """
        names = self.served_models()
        if names is None or settings.embed_model in names:
            return
        raise EmbedModelMismatch(
            f"{self._url.removesuffix('/embeddings')} does not serve "
            f"'{settings.embed_model}' -- it serves {', '.join(sorted(set(names)))}. "
            "Embedding requests would still succeed and return vectors from the "
            "wrong model, so this refuses to start instead. Point "
            "FUKO_EMBED_MODEL at what is actually served (and move "
            "FUKO_EMBED_QUERY_PREFIX with it)."
        )

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
