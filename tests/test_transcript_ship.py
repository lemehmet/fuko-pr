"""The path a captured transcript takes from the runner into shared storage (#238).

Every branch below that asserts the NO-SIDECAR path depends on ``FUKO_URL``
being unset, which is arranged repo-wide by ``conftest``'s autouse
``_no_ambient_sidecar`` (it clears ``FUKO_URL`` and ``FUKO_TOKEN`` for every
test in the suite). The tests that want the HTTP branch set them back
themselves. Do not add a second fixture here: a module-level one of the same
name SHADOWS the shared guard, which is worse than nothing the day the shared
one grows another variable.
"""

import asyncio
import contextlib
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect, Request

from sidecar import main
from sidecar.config import settings
from sidecar import objectstore
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


def _as_stream(fake_post):
    """Adapt a ``(url, content, headers, timeout) -> Response`` fake to the
    shape `ship` now calls: :func:`httpx.stream`, a context manager over a
    response whose BODY has not been read.

    `ship` switched off `httpx.post` because that one reads the response to
    completion, under a `read` timeout that bounds the gap between chunks
    rather than the whole read -- which made `UPLOAD_CEILING_S` a claim rather
    than a ceiling. The fakes stay written against the simpler shape; this
    adapts them, and asserts the method so a regression to a GET is visible.
    """

    @contextlib.contextmanager
    def fake_stream(method, url, content=None, headers=None, timeout=None):
        assert method == "POST"
        yield fake_post(url, content=content, headers=headers, timeout=timeout)

    return fake_stream


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


def test_a_filesystem_root_store_is_refused_rather_than_left_undenied(monkeypatch, capsys):
    """`_permission_settings` rstrips a candidate to the empty string and drops
    it silently, so a root store would keep a transcript corpus that no read
    deny rule covers and nothing reports -- the same third spelling of
    "written but undenied" that `transcript_dir()` refuses."""
    monkeypatch.setattr(settings, "transcript_store_backend", "file")
    monkeypatch.setattr(settings, "transcript_store_root", "/")
    with pytest.raises(ValueError, match="filesystem root"):
        transcript_store()
    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=BODY)
    assert resp.status_code == 503
    assert "filesystem root" in resp.json()["detail"]
    assert "transcript store unusable" in capsys.readouterr().err


def test_an_unresolvable_store_root_is_the_same_refusal_as_a_root_one(monkeypatch):
    """`expanduser` raises RuntimeError for a `~user` with no home, and
    `resolve` can raise OSError. Normalized inside `local_blob_root`, so the
    endpoint answers 503 rather than 500 and the driver's guard sees the class
    it already handles."""
    monkeypatch.setattr(settings, "transcript_store_backend", "file")
    monkeypatch.setattr(settings, "transcript_store_root", "~nosuchuser12345/blobs")
    with pytest.raises(ValueError, match="cannot be resolved"):
        transcript_store()
    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=BODY)
    assert resp.status_code == 503
    assert "cannot be resolved" in resp.json()["detail"]


def test_a_relative_store_root_is_refused(monkeypatch):
    """`resolve()` anchors a relative root at the CALLING process's cwd, and
    this runs in two processes -- the sidecar deciding where blobs are written,
    the runner driver deciding what the deny rule covers. They would disagree
    silently, in the direction that leaves the corpus undenied."""
    monkeypatch.setattr(settings, "transcript_store_backend", "file")
    monkeypatch.setattr(settings, "transcript_store_root", "blobs")
    with pytest.raises(ValueError, match="relative"):
        transcript_store()


def test_a_client_that_vanishes_mid_body_stays_inside_the_taxonomy(monkeypatch, store_dir):
    """`request.stream()` raises starlette's `ClientDisconnect`, which is a
    plain Exception -- the one shape that would otherwise reach
    ServerErrorMiddleware as an attempted 500 and a traceback per occurrence."""

    async def _disconnect(self):
        raise ClientDisconnect()
        yield b""  # pragma: no cover - never reached, makes this a generator

    monkeypatch.setattr(Request, "stream", _disconnect)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(main.transcripts_put_endpoint(KEY, Request({"type": "http"})))
    assert caught.value.status_code == 400
    assert "disconnected" in caught.value.detail


def test_a_configured_but_unusable_store_is_a_503_with_the_reason(monkeypatch):
    """The store is built per request, so a bad configuration would otherwise
    reach the caller as a 500 and a traceback on every single upload."""
    monkeypatch.setattr(settings, "transcript_store_backend", "s3")
    monkeypatch.setattr(settings, "transcript_store_bucket", "")
    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=BODY)
    assert resp.status_code == 503
    assert "FUKO_TRANSCRIPT_STORE_BUCKET" in resp.json()["detail"]


def test_a_bucket_backend_without_boto3_is_a_503_naming_the_extra(monkeypatch):
    """`docker/Dockerfile.sidecar` installs `.[s3]`; a sidecar installed without
    it must say so rather than raise ModuleNotFoundError per upload."""

    def _no_boto3():
        raise ImportError("No module named 'boto3'")

    monkeypatch.setattr(main, "transcript_store", _no_boto3)
    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=BODY)
    assert resp.status_code == 503
    assert "boto3" in resp.json()["detail"]


def test_the_unconfigured_503_is_answered_only_after_the_body_is_drained(monkeypatch):
    """Answering before the body is consumed closes the connection under a
    client that is still sending, so the runner gets a write error instead of
    the marked 503 it reads as the off state -- and the "no failure line per
    run" promise becomes a failure line per run. A few-byte body never shows
    this; a real transcript would show nothing else."""
    monkeypatch.setattr(settings, "transcript_store_backend", "")
    sent = []

    def chunks():
        for chunk in (b"a" * 16, b"b" * 16, b"c" * 16):
            sent.append(chunk)
            yield chunk

    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=chunks())
    assert resp.status_code == 503
    assert resp.headers.get(objectstore.STORE_HEADER) == objectstore.STORE_UNCONFIGURED
    # Every chunk was pulled before the refusal was written.
    assert sent == [b"a" * 16, b"b" * 16, b"c" * 16]


def test_a_refused_upload_drains_without_buffering_the_body(monkeypatch):
    """The drain is required so the marked 503 reaches a still-sending client,
    but nothing in the classification needs the bytes -- so the recommended
    rollout order (capture on, storage not yet) must not push every transcript
    into sidecar memory purely to discard it."""
    monkeypatch.setattr(settings, "transcript_store_backend", "")
    monkeypatch.setattr(settings, "transcript_max_bytes", 1024)
    held = []

    class _Watched(bytearray):
        def __iadd__(self, other):
            held.append(len(other))
            return super().__iadd__(other)

    monkeypatch.setattr(main, "bytearray", _Watched, raising=False)
    sent = []

    def chunks():
        for chunk in (b"a" * 16, b"b" * 16, b"c" * 16):
            sent.append(chunk)
            yield chunk

    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=chunks())
    assert resp.status_code == 503
    assert sent == [b"a" * 16, b"b" * 16, b"c" * 16]  # drained
    assert held == []  # ...and never accumulated


def test_a_store_that_fails_when_used_is_a_503_and_a_named_log_line(monkeypatch, capsys):
    """A store can construct and then fail at request time -- credentials boto3
    resolves lazily, an unreachable endpoint, a full disk. Unmapped, those
    escaped the endpoint's own taxonomy as a 500 and a traceback per upload."""

    class _Failing:
        def put(self, key, data):
            raise RuntimeError("Unable to locate credentials")

    monkeypatch.setattr(main, "transcript_store", lambda: _Failing())
    resp = _client(monkeypatch).post(f"/transcripts/{KEY}", content=BODY)
    assert resp.status_code == 503
    assert "Unable to locate credentials" in resp.json()["detail"]
    # Not the off state: the runner must report this one.
    assert objectstore.STORE_HEADER not in resp.headers
    assert "transcript store failed" in capsys.readouterr().err


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
    assert transcript_client.ship(KEY, path) is True
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
        seen.update(url=url, body=b"".join(content), headers=headers, timeout=timeout)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000/")
    monkeypatch.setenv("FUKO_TOKEN", "runner-token")
    monkeypatch.setattr(httpx, "stream", _as_stream(fake_post))
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    assert transcript_client.ship(KEY, path) is True

    assert seen["url"] == f"http://sidecar:8000/transcripts/{KEY}"
    assert seen["body"] == BODY
    assert seen["headers"]["Authorization"] == "Bearer runner-token"
    assert seen["headers"]["Content-Type"] == "application/x-ndjson"
    # An order of magnitude above `/metrics/run`'s 10s, which is the whole
    # reason this is not that endpoint. Per-phase, because httpx has no request
    # lifetime -- the absolute bound is `_deadlined`, asserted below.
    assert transcript_client.UPLOAD_TIMEOUT_S > 10.0
    # Per-phase bounds are deliberately FAR below the body deadline: the
    # deadline is only tested between chunks, so a write phase given the whole
    # ceiling would make the true worst case several multiples of it.
    assert seen["timeout"].write == transcript_client.PHASE_TIMEOUT_S
    assert seen["timeout"].read == transcript_client.PHASE_TIMEOUT_S
    assert seen["timeout"].connect == transcript_client.CONNECT_TIMEOUT_S
    assert transcript_client.PHASE_TIMEOUT_S < transcript_client.UPLOAD_TIMEOUT_S
    # Conservative: the deadline path and the response path are exclusive, so
    # the last phase is counted once for whichever occurs.
    assert transcript_client.UPLOAD_CEILING_S == (
        transcript_client.CONNECT_TIMEOUT_S
        + transcript_client.UPLOAD_TIMEOUT_S
        + transcript_client.PHASE_TIMEOUT_S
    )


def test_an_upload_that_outlives_its_ceiling_is_abandoned_rather_than_prolonged(
    tmp_path, monkeypatch
):
    """httpx's `timeout=` is per phase, so a peer that keeps accepting chunks
    slowly resets it forever. The deadline rides on the body stream instead."""

    def fake_post(url, content=None, headers=None, timeout=None):
        b"".join(content)  # drain the generator, which is what raises
        raise AssertionError("the upload should not have completed")

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    monkeypatch.setattr(httpx, "stream", _as_stream(fake_post))
    monkeypatch.setattr(transcript_client, "UPLOAD_TIMEOUT_S", -1.0)
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    with pytest.raises(TimeoutError, match="UPLOAD_TIMEOUT_S"):
        transcript_client.ship(KEY, path)


def test_the_body_is_chunked_by_size_rather_than_by_event(tmp_path, monkeypatch):
    """Iterating the handle itself would yield one write per NDJSON line, on a
    file whose whole point is that it has many."""
    chunks = []

    def fake_post(url, content=None, headers=None, timeout=None):
        chunks.extend(content)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    monkeypatch.setattr(transcript_client, "UPLOAD_CHUNK_BYTES", 16)
    monkeypatch.setattr(httpx, "stream", _as_stream(fake_post))
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    transcript_client.ship(KEY, path)

    assert b"".join(chunks) == BODY
    assert all(len(c) <= 16 for c in chunks)
    assert len(chunks) == -(-len(BODY) // 16)


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.ConnectError("sidecar unreachable"),
        httpx.ReadTimeout("too slow"),
        httpx.Response(500, request=httpx.Request("POST", "http://sidecar:8000/")),
        httpx.Response(409, request=httpx.Request("POST", "http://sidecar:8000/")),
    ],
)
def test_every_sidecar_failure_reaches_the_caller_to_be_reported(tmp_path, monkeypatch, outcome):
    def fake_post(url, content=None, headers=None, timeout=None):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    monkeypatch.setattr(httpx, "stream", _as_stream(fake_post))
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    with pytest.raises(httpx.HTTPError):
        transcript_client.ship(KEY, path)


def test_a_sidecar_with_storage_turned_off_is_the_off_state_not_a_failure(tmp_path, monkeypatch):
    """Storage not configured is the OFF state. Reporting it would print one
    "capture failed" line per agentic run on every fleet that stages capture
    ahead of storage, which is the rollout order the deployment docs
    recommend."""

    def fake_post(url, content=None, headers=None, timeout=None):
        b"".join(content)
        return httpx.Response(
            503,
            headers={objectstore.STORE_HEADER: objectstore.STORE_UNCONFIGURED},
            json={"detail": "no transcript store configured (set FUKO_TRANSCRIPT_STORE_BACKEND)"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    monkeypatch.setattr(httpx, "stream", _as_stream(fake_post))
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    # Returns, raises nothing -- but reports that nothing was STORED, which is
    # what keeps #239 from writing a reference to a blob that only ever existed
    # on this runner's disk.
    assert transcript_client.ship(KEY, path) is False


def test_a_503_from_a_store_that_was_meant_to_work_still_reports(tmp_path, monkeypatch):
    """The other 503 -- an unknown backend, a missing bucket, an absent boto3.
    Swallowing it too would make the feature store nothing in silence, which is
    exactly what the endpoint's distinguished statuses exist to prevent."""

    def fake_post(url, content=None, headers=None, timeout=None):
        b"".join(content)
        return httpx.Response(
            503,
            json={"detail": "transcript store unusable: No module named 'boto3'"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    monkeypatch.setattr(httpx, "stream", _as_stream(fake_post))
    path = tmp_path / "t.ndjson"
    path.write_bytes(BODY)
    with pytest.raises(httpx.HTTPError):
        transcript_client.ship(KEY, path)


def test_the_endpoint_marks_only_the_unconfigured_503(monkeypatch, capsys):
    """The header is the contract between the two halves, so assert it on the
    endpoint rather than only on the runner's reading of it."""
    monkeypatch.setattr(settings, "transcript_store_backend", "")
    off = _client(monkeypatch).post(f"/transcripts/{KEY}", content=BODY)
    assert off.status_code == 503
    assert off.headers.get(objectstore.STORE_HEADER) == objectstore.STORE_UNCONFIGURED

    monkeypatch.setattr(settings, "transcript_store_backend", "s3")
    monkeypatch.setattr(settings, "transcript_store_bucket", "")
    broken = _client(monkeypatch).post(f"/transcripts/{KEY}", content=BODY)
    assert broken.status_code == 503
    assert objectstore.STORE_HEADER not in broken.headers
    # `HTTPException` logs nothing, and the runner deliberately says nothing
    # about a 503 -- so the deployment fault has to be named here or nowhere.
    assert "transcript store unusable" in capsys.readouterr().err


def test_the_whole_path_end_to_end_from_ship_to_stored_blob(tmp_path, monkeypatch, store_dir):
    """The runner's `ship` against the real endpoint and a real store: the two
    halves of #238 meeting, with only the socket faked."""
    client = _client(monkeypatch)

    def fake_post(url, content=None, headers=None, timeout=None):
        return client.post(urlsplit(url).path, content=b"".join(content), headers=headers)

    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")
    monkeypatch.setenv("FUKO_TOKEN", _TOKEN)
    monkeypatch.setattr(httpx, "stream", _as_stream(fake_post))
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
