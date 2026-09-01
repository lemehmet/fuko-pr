"""Unit tests for the embeddings client error handling (no network)."""

import httpx
import pytest

from sidecar.config import settings
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


def test_embed_truncates_oversized_input(monkeypatch):
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(settings, "embed_max_chars", 100)
    Embedder().embed(["x" * 5000])
    assert sent == [["x" * 100]]


def test_embed_leaves_a_short_input_alone(monkeypatch):
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(settings, "embed_max_chars", 100)
    Embedder().embed(["short enough"])
    assert sent == [["short enough"]]


def test_embed_query_applies_the_instruction_prefix(monkeypatch):
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(settings, "embed_query_prefix", "Instruct: task\nQuery: ")
    Embedder().embed_query("why does retrieval fail")
    assert sent == [["Instruct: task\nQuery: why does retrieval fail"]]


def test_embed_one_stays_unprefixed(monkeypatch):
    # Documents must not carry the query instruction: prefixing both sides is
    # the failure this asymmetry exists to prevent.
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(settings, "embed_query_prefix", "Instruct: task\nQuery: ")
    Embedder().embed_one("a stored learning")
    assert sent == [["a stored learning"]]
