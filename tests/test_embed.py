"""Unit tests for the embeddings client error handling (no network)."""

import httpx
import pytest

from sidecar.config import settings
from sidecar.embed import EmbedError, Embedder
from sidecar.retrieve import _build_query


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


def test_assembled_query_reaches_the_endpoint_with_its_files_block(monkeypatch):
    # The end-to-end property the three caps have to compose into: what _fit
    # actually posts still carries the "Changed files:" block, so the transport
    # backstop never fires on a query _build_query assembled.
    #
    # Driven through embed_query, not embed: the query instruction is prepended
    # inside embed_query and _fit cuts the COMBINED string, so a test that
    # posted the bare assembled query would assert a composition the production
    # path never performs -- and would keep passing while the real one lost the
    # tail of the files block.
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    files = [f"packages/shared/src/module_{i:04d}/index.ts" for i in range(300)]
    q = _build_query(files, "log line\n" * 5000, None)
    Embedder().embed_query(q)
    assert sent == [[settings.embed_query_prefix + q]]
    assert "Changed files:\npackages/shared/src/module_0000/index.ts" in sent[0][0]
    # Nothing was cut: the last path posted is the last path assembled, whole.
    assert sent[0][0].endswith(q[-len("index.ts") - 60 :])
