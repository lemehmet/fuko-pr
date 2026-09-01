"""The startup guard against an endpoint serving a different model than configured.

The failure this exists for is silent: llama-server ignores the request's
``model`` field, so asking for one model and being served another returns
well-formed vectors from the wrong space (#220). These tests pin the *direction*
of the guard as much as its behaviour -- "cannot verify" must never become
"refuse to start", or a slow embedder turns into a sidecar outage.
"""

import httpx
import pytest

from sidecar.config import settings
from sidecar.embed import EmbedModelMismatch, Embedder


def _responder(monkeypatch, *, status=200, body=None, raise_exc=None):
    def fake_get(self, url, headers=None, timeout=None):
        if raise_exc is not None:
            raise raise_exc
        request = httpx.Request("GET", url)
        return httpx.Response(status, request=request, json=body)

    monkeypatch.setattr(httpx.Client, "get", fake_get)


def test_openai_shape_is_parsed(monkeypatch):
    _responder(monkeypatch, body={"object": "list", "data": [{"id": "bge-m3"}]})
    assert Embedder().served_models() == ["bge-m3"]


def test_ollama_shape_is_parsed(monkeypatch):
    _responder(monkeypatch, body={"models": [{"name": "qwen3-embedding-0.6b"}]})
    assert Embedder().served_models() == ["qwen3-embedding-0.6b"]


def test_both_shapes_at_once(monkeypatch):
    # llama.cpp answers with both keys; the same name appearing twice is fine.
    _responder(
        monkeypatch,
        body={"data": [{"id": "qwen3"}], "models": [{"name": "qwen3", "model": "qwen3"}]},
    )
    assert Embedder().served_models() == ["qwen3", "qwen3"]


def test_matching_model_passes(monkeypatch):
    _responder(monkeypatch, body={"data": [{"id": "qwen3-embedding-0.6b"}]})
    monkeypatch.setattr(settings, "embed_model", "qwen3-embedding-0.6b")
    Embedder().verify_model()  # does not raise


def test_mismatch_refuses(monkeypatch):
    _responder(monkeypatch, body={"data": [{"id": "qwen3-embedding-0.6b"}]})
    monkeypatch.setattr(settings, "embed_model", "bge-m3")
    with pytest.raises(EmbedModelMismatch) as exc:
        Embedder().verify_model()
    # The message has to name both sides, or it cannot be acted on.
    assert "bge-m3" in str(exc.value) and "qwen3-embedding-0.6b" in str(exc.value)


def test_ollama_latest_tag_matches_untagged_config(monkeypatch):
    # Ollama lists `bge-m3:latest` and resolves an untagged request to it, so
    # every Ollama deployment in this repo (compose, .env.example, CI) pins the
    # untagged name against a list that never contains it verbatim.
    _responder(monkeypatch, body={"models": [{"name": "bge-m3:latest"}]})
    monkeypatch.setattr(settings, "embed_model", "bge-m3")
    Embedder().verify_model()  # does not raise


def test_latest_tag_does_not_mask_a_different_model(monkeypatch):
    _responder(monkeypatch, body={"models": [{"name": "qwen3-embedding-0.6b:latest"}]})
    monkeypatch.setattr(settings, "embed_model", "bge-m3")
    with pytest.raises(EmbedModelMismatch):
        Embedder().verify_model()


def test_only_the_latest_tag_is_accepted(monkeypatch):
    # A different tag is a different artefact that an untagged request would
    # not resolve to, so it must stay a refusal rather than become a match.
    _responder(monkeypatch, body={"models": [{"name": "bge-m3:q4_K_M"}]})
    monkeypatch.setattr(settings, "embed_model", "bge-m3")
    with pytest.raises(EmbedModelMismatch):
        Embedder().verify_model()


def test_tagged_config_is_not_loosened(monkeypatch):
    # The allowance runs one way only: a config that names a tag is taken at
    # its word, so nothing silently widens a deliberately pinned tag.
    _responder(monkeypatch, body={"data": [{"id": "bge-m3"}]})
    monkeypatch.setattr(settings, "embed_model", "bge-m3:latest")
    with pytest.raises(EmbedModelMismatch):
        Embedder().verify_model()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raise_exc": httpx.ConnectError("down")},  # endpoint unreachable
        {"status": 404, "body": {}},  # no /models route
        {"body": {"object": "list"}},  # answered, but says nothing about models
        {"body": {"data": "not-a-list"}},  # unparseable
        {"body": {"data": [{"no_id_field": 1}]}},  # entries without a usable name
    ],
)
def test_cannot_verify_never_blocks_startup(monkeypatch, kwargs):
    _responder(monkeypatch, **kwargs)
    monkeypatch.setattr(settings, "embed_model", "anything")
    assert Embedder().served_models() is None
    Embedder().verify_model()  # silence, not refusal
