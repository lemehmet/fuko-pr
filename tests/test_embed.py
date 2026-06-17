"""Unit tests for the embeddings client error handling (no network)."""

import httpx
import pytest

from sidecar.embed import EmbedError, Embedder


def test_embed_empty_returns_empty():
    assert Embedder().embed([]) == []


def test_embed_surfaces_error_body(monkeypatch):
    def fake_post(self, url, headers=None, json=None):
        request = httpx.Request("POST", url, headers=headers or {})
        response = httpx.Response(400, request=request, text='{"error":{"message":"bad model"}}')
        raise httpx.HTTPStatusError("400 Bad Request", request=request, response=response)

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(EmbedError) as exc:
        Embedder().embed(["hi"])
    assert "bad model" in str(exc.value)
    assert "400" in str(exc.value)
