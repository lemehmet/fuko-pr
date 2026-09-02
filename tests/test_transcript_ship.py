"""The path a captured transcript takes from the runner into shared storage (#238)."""

from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from sidecar import main
from sidecar.config import settings
from sidecar.objectstore import FileBlobStore, transcript_store
from sidecar.reviewer import transcript_client

_TOKEN = "test-token"
KEY = "20260901T101500Z-abcdef012345"
BODY = b'{"type":"assistant"}\n{"type":"user"}\n{"type":"result"}\n'


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    """Point the sidecar's transcript store at a local directory."""
    root = tmp_path / "blobs"
    monkeypatch.setattr(settings, "transcript_store_backend", "file")
    monkeypatch.setattr(settings, "transcript_store_root", str(root))
    return root


def _client(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    return TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})


# --- the endpoint ----------------------------------------------------------


def test_a_transcript_is_retrievable_byte_identical_after_upload(monkeypatch, store_dir):
    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=BODY)
    assert resp.status_code == 200
    assert resp.json() == {"stored": True, "key": KEY, "bytes": len(BODY)}
    assert FileBlobStore(str(store_dir)).get(KEY) == BODY


def test_two_runners_land_side_by_side_without_either_overwriting(monkeypatch, store_dir):
    client = _client(monkeypatch)
    client.post("/transcripts/20260901T101500Z-aaaaaaaaaaaa", content=b"runner-a")
    client.post("/transcripts/20260901T101501Z-bbbbbbbbbbbb", content=b"runner-b")
    store = FileBlobStore(str(store_dir))
    assert store.get("20260901T101500Z-aaaaaaaaaaaa") == b"runner-a"
    assert store.get("20260901T101501Z-bbbbbbbbbbbb") == b"runner-b"


def test_a_re_delivered_key_is_a_conflict_rather_than_an_overwrite(monkeypatch, store_dir):
    client = _client(monkeypatch)
    client.post(f"/transcripts/{KEY}", content=b"first")
    resp = client.post(f"/transcripts/{KEY}", content=b"second")
    assert resp.status_code == 409
    assert FileBlobStore(str(store_dir)).get(KEY) == b"first"


def test_a_key_that_is_not_a_blob_key_is_refused(monkeypatch, store_dir):
    resp = _client(monkeypatch).post("/transcripts/.hidden", content=BODY)
    assert resp.status_code == 400
    assert not store_dir.exists()


def test_an_unconfigured_store_answers_503_and_the_sidecar_still_serves(monkeypatch):
    monkeypatch.setattr(settings, "transcript_store_backend", "")
    client = _client(monkeypatch)
    resp = client.post(f"/transcripts/{KEY}", content=BODY)
    assert resp.status_code == 503
    assert "FUKO_TRANSCRIPT_STORE_BACKEND" in resp.json()["detail"]
    assert client.get("/healthz").json() == {"ok": True}


def test_the_upload_endpoint_needs_the_sidecar_token(monkeypatch, store_dir):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    resp = TestClient(main.app).post(f"/transcripts/{KEY}", content=BODY)
    assert resp.status_code == 401
    assert not store_dir.exists()


def test_an_8_mb_transcript_uploads_over_the_dedicated_path(monkeypatch, store_dir):
    """Substantially larger than a metrics row, which is the point of not
    widening `/metrics/run` (a few hundred bytes under a 10s timeout)."""
    event = b'{"type":"user","content":"' + b"x" * 1000 + b'"}\n'
    big = event * (8 * 1024 * 1024 // len(event) + 1)
    assert len(big) > 8 * 1024 * 1024
    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=big)
    assert resp.status_code == 200
    assert FileBlobStore(str(store_dir)).get(KEY) == big


# --- the runner's side of it -----------------------------------------------


def test_upload_target_names_the_sidecar_over_a_locally_configured_store(monkeypatch, store_dir):
    assert transcript_client.upload_target() == "store"
    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    assert transcript_client.upload_target() == "sidecar"


def test_no_sidecar_and_no_store_is_no_destination(monkeypatch):
    monkeypatch.setattr(settings, "transcript_store_backend", "")
    assert transcript_client.upload_target() == ""


def test_a_runner_with_no_sidecar_writes_straight_to_its_own_store(tmp_path, store_dir):
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    transcript_client.ship(KEY, path)
    assert transcript_store().get(KEY) == BODY


def test_a_runner_with_neither_raises_so_the_capture_reports_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transcript_store_backend", "")
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    with pytest.raises(RuntimeError, match="no transcript store configured"):
        transcript_client.ship(KEY, path)


def test_the_runner_streams_the_file_to_the_sidecar_under_its_token(tmp_path, monkeypatch):
    seen = {}

    def fake_post(url, content=None, headers=None, timeout=None):
        seen.update(url=url, body=content.read(), headers=headers, timeout=timeout)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000/")
    monkeypatch.setenv("FUKO_TOKEN", "runner-token")
    monkeypatch.setattr(httpx, "post", fake_post)
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    transcript_client.ship(KEY, path)

    assert seen["url"] == f"http://sidecar:8000/transcripts/{KEY}"
    assert seen["body"] == BODY
    assert seen["headers"]["Authorization"] == "Bearer runner-token"
    assert seen["headers"]["Content-Type"] == "application/x-ndjson"
    # An order of magnitude above `/metrics/run`'s 10s, which is the whole
    # reason this is not that endpoint.
    assert seen["timeout"] == transcript_client.UPLOAD_TIMEOUT_S > 10.0


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.ConnectError("sidecar unreachable"),
        httpx.ReadTimeout("too slow"),
        httpx.Response(503, request=httpx.Request("POST", "http://sidecar:8000/")),
        httpx.Response(409, request=httpx.Request("POST", "http://sidecar:8000/")),
    ],
)
def test_every_sidecar_failure_reaches_the_caller_to_be_reported(tmp_path, monkeypatch, outcome):
    def fake_post(url, content=None, headers=None, timeout=None):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    monkeypatch.setattr(httpx, "post", fake_post)
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    with pytest.raises(httpx.HTTPError):
        transcript_client.ship(KEY, path)


def test_the_whole_path_end_to_end_from_ship_to_stored_blob(tmp_path, monkeypatch, store_dir):
    """The runner's `ship` against the real endpoint and a real store: the two
    halves of #238 meeting, with only the socket faked."""
    client = _client(monkeypatch)

    def fake_post(url, content=None, headers=None, timeout=None):
        return client.post(urlsplit(url).path, content=content.read(), headers=headers)

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    monkeypatch.setenv("FUKO_TOKEN", _TOKEN)
    monkeypatch.setattr(httpx, "post", fake_post)
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    transcript_client.ship(KEY, path)
    assert FileBlobStore(str(store_dir)).get(KEY) == BODY


def test_an_oversized_body_is_refused_whole_rather_than_stored_truncated(monkeypatch, store_dir):
    """A partial blob under a write-once key could never be corrected, so the
    cap refuses the upload instead of keeping what fitted."""
    monkeypatch.setattr(settings, "transcript_max_bytes", 64)
    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=b"x" * 65)
    assert resp.status_code == 413
    assert FileBlobStore(str(store_dir)).get(KEY) is None
