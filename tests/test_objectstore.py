"""Unit tests for the object-storage sync layer (file + S3/R2 conditional writes)."""

import io

import pytest

from sidecar import objectstore
from sidecar.config import settings
from sidecar.fukoconfig import ObjectStoreConfig
from sidecar.objectstore import (
    BlobExists,
    BlobStoreConfig,
    FileBlobStore,
    FileObjectStore,
    PreconditionFailed,
    S3BlobStore,
    S3ObjectStore,
    make_blob_store,
    make_object_store,
    transcript_store,
    validate_blob_key,
)


def test_file_store_load_missing(tmp_path):
    store = FileObjectStore(str(tmp_path / "kb.db"))
    assert store.load() == (None, None)


def test_file_store_create_then_update(tmp_path):
    store = FileObjectStore(str(tmp_path / "kb.db"))
    token = store.save(b"v1", None)
    data, t = store.load()
    assert data == b"v1" and t == token
    token2 = store.save(b"v2", t)
    assert store.load()[0] == b"v2" and token2 != token


def test_file_store_create_conflict_when_exists(tmp_path):
    store = FileObjectStore(str(tmp_path / "kb.db"))
    store.save(b"v1", None)
    with pytest.raises(PreconditionFailed):
        store.save(b"v2", None)


def test_file_store_stale_token_conflict(tmp_path):
    store = FileObjectStore(str(tmp_path / "kb.db"))
    store.save(b"v1", None)
    with pytest.raises(PreconditionFailed):
        store.save(b"v2", "999")


def test_file_store_delete_is_conflict(tmp_path):
    path = tmp_path / "kb.db"
    store = FileObjectStore(str(path))
    token = store.save(b"v1", None)
    path.unlink()  # an intervening delete is a change
    with pytest.raises(PreconditionFailed):
        store.save(b"v2", token)


class _FakeClientError(Exception):
    def __init__(self, code, status):
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class _FakeS3:
    def __init__(self):
        self.objs = {}
        self._n = 0

    def get_object(self, Bucket, Key):
        if Key not in self.objs:
            raise _FakeClientError("NoSuchKey", 404)
        data, etag = self.objs[Key]
        return {"Body": io.BytesIO(data), "ETag": etag}

    def put_object(self, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None):
        cur = self.objs.get(Key)
        if IfNoneMatch == "*" and cur is not None:
            raise _FakeClientError("PreconditionFailed", 412)
        if IfMatch is not None and (cur is None or cur[1] != IfMatch):
            raise _FakeClientError("PreconditionFailed", 412)
        self._n += 1
        etag = f'"etag{self._n}"'
        self.objs[Key] = (Body, etag)
        return {"ETag": etag}


def test_s3_store_create_load_update():
    c = _FakeS3()
    store = S3ObjectStore(c, "bucket", "k.db")
    assert store.load() == (None, None)
    e1 = store.save(b"v1", None)
    assert store.load() == (b"v1", e1)
    e2 = store.save(b"v2", e1)
    assert store.load() == (b"v2", e2)


def test_s3_store_create_conflict():
    c = _FakeS3()
    store = S3ObjectStore(c, "bucket", "k.db")
    store.save(b"v1", None)
    with pytest.raises(PreconditionFailed):
        store.save(b"v2", None)  # IfNoneMatch=* but exists


def test_s3_store_stale_token_conflict():
    c = _FakeS3()
    store = S3ObjectStore(c, "bucket", "k.db")
    e1 = store.save(b"v1", None)
    store.save(b"v2", e1)  # advances etag
    with pytest.raises(PreconditionFailed):
        store.save(b"v3", e1)  # stale


def test_s3_store_reraises_unexpected_error():
    class _C:
        def get_object(self, **k):
            raise _FakeClientError("AccessDenied", 403)

    with pytest.raises(_FakeClientError):
        S3ObjectStore(_C(), "b", "k").load()


def test_s3_store_save_reraises_unexpected_error():
    class _C:
        def put_object(self, **k):
            raise _FakeClientError("AccessDenied", 403)

    with pytest.raises(_FakeClientError):
        S3ObjectStore(_C(), "b", "k").save(b"x", None)


def test_s3_store_load_closes_body():
    class _Body:
        def __init__(self, data):
            self._b = io.BytesIO(data)
            self.closed = False

        def read(self):
            return self._b.read()

        def close(self):
            self.closed = True

    body = _Body(b"data")

    class _C:
        def get_object(self, **k):
            return {"Body": body, "ETag": '"e"'}

    data, etag = S3ObjectStore(_C(), "b", "k").load()
    assert (data, etag) == (b"data", '"e"')
    assert body.closed  # StreamingBody must be closed to avoid connection leaks


def test_make_object_store_file(tmp_path):
    cfg = ObjectStoreConfig(backend="file", key=str(tmp_path / "kb.db"))
    assert isinstance(make_object_store(cfg), FileObjectStore)


def test_make_object_store_file_requires_key():
    with pytest.raises(ValueError):
        make_object_store(ObjectStoreConfig(backend="file", key=None))


def test_make_object_store_s3_requires_bucket_and_key():
    with pytest.raises(ValueError):
        make_object_store(ObjectStoreConfig(backend="s3", bucket="b", key=None))


def test_make_object_store_s3_builds_client():
    cfg = ObjectStoreConfig(backend="s3", bucket="b", key="k.db", endpoint_url="https://x")
    assert isinstance(make_object_store(cfg), S3ObjectStore)


# --- Keyed, write-once blobs (#238).


KEY = "20260901T101500Z-abcdef012345"


def test_a_blob_round_trips_byte_identical_through_the_file_backend(tmp_path):
    store = FileBlobStore(str(tmp_path / "transcripts"))
    payload = b'{"type":"user"}\n{"type":"result"}\n\xf0\x9f\x99\x82\n'
    store.put(KEY, payload)
    assert store.get(KEY) == payload


def test_a_missing_blob_reads_as_none(tmp_path):
    assert FileBlobStore(str(tmp_path)).get(KEY) is None


def test_a_file_blob_is_write_once_and_the_first_bytes_survive(tmp_path):
    store = FileBlobStore(str(tmp_path))
    store.put(KEY, b"first")
    with pytest.raises(BlobExists):
        store.put(KEY, b"second")
    assert store.get(KEY) == b"first"


def test_two_runs_land_under_distinct_keys_without_clobbering(tmp_path):
    store = FileBlobStore(str(tmp_path))
    store.put("20260901T101500Z-aaaaaaaaaaaa", b"runner-a")
    store.put("20260901T101501Z-bbbbbbbbbbbb", b"runner-b")
    assert store.get("20260901T101500Z-aaaaaaaaaaaa") == b"runner-a"
    assert store.get("20260901T101501Z-bbbbbbbbbbbb") == b"runner-b"


def test_a_file_blob_and_its_directory_are_owner_only(tmp_path):
    root = tmp_path / "transcripts"
    store = FileBlobStore(str(root))
    store.put(KEY, b"x")
    assert (root / KEY).stat().st_mode & 0o777 == 0o600
    assert root.stat().st_mode & 0o777 == 0o700


def test_a_put_leaves_no_staging_file_behind(tmp_path):
    root = tmp_path / "transcripts"
    store = FileBlobStore(str(root))
    store.put(KEY, b"x")
    with pytest.raises(BlobExists):
        store.put(KEY, b"y")
    assert sorted(p.name for p in root.iterdir()) == [KEY]


@pytest.mark.parametrize(
    "key",
    [
        "../../etc/passwd",
        "a/b",
        "a\\b",
        ".hidden",
        "..",
        "",
        "key with spaces",
        "k" * 129,
        "\x00",
    ],
)
def test_a_key_that_is_not_one_path_segment_is_refused(tmp_path, key):
    with pytest.raises(ValueError):
        FileBlobStore(str(tmp_path)).put(key, b"x")
    with pytest.raises(ValueError):
        FileBlobStore(str(tmp_path)).get(key)
    with pytest.raises(ValueError):
        S3BlobStore(_FakeS3(), "b").put(key, b"x")


def test_mint_key_is_a_valid_blob_key():
    from sidecar.reviewer.transcript import mint_key

    assert validate_blob_key(mint_key())


def test_an_s3_blob_round_trips_under_its_prefix():
    client = _FakeS3()
    store = S3BlobStore(client, "bucket", "transcripts/")
    store.put(KEY, b"payload")
    assert store.get(KEY) == b"payload"
    assert list(client.objs) == [f"transcripts/{KEY}"]


def test_an_s3_blob_with_no_prefix_is_named_by_its_key_alone():
    client = _FakeS3()
    S3BlobStore(client, "bucket", "  ").put(KEY, b"payload")
    assert list(client.objs) == [KEY]


def test_an_s3_blob_is_write_once_by_the_servers_condition():
    client = _FakeS3()
    store = S3BlobStore(client, "bucket")
    store.put(KEY, b"first")
    with pytest.raises(BlobExists):
        store.put(KEY, b"second")
    assert store.get(KEY) == b"first"


def test_a_missing_s3_blob_reads_as_none():
    assert S3BlobStore(_FakeS3(), "bucket").get(KEY) is None


def test_an_s3_blob_store_reraises_an_error_that_is_not_a_lost_race():
    class _C:
        def put_object(self, **kw):
            raise _FakeClientError("AccessDenied", 403)

        def get_object(self, **kw):
            raise _FakeClientError("AccessDenied", 403)

    with pytest.raises(_FakeClientError):
        S3BlobStore(_C(), "b").put(KEY, b"x")
    with pytest.raises(_FakeClientError):
        S3BlobStore(_C(), "b").get(KEY)


def test_an_s3_blob_get_closes_the_streaming_body():
    body = io.BytesIO(b"data")

    class _C:
        def get_object(self, **kw):
            return {"Body": body}

    assert S3BlobStore(_C(), "b").get(KEY) == b"data"
    assert body.closed


def test_no_backend_configured_is_no_store_rather_than_an_error():
    assert make_blob_store(BlobStoreConfig()) is None


def test_the_file_blob_backend_needs_a_root(tmp_path):
    assert isinstance(
        make_blob_store(BlobStoreConfig(backend="file", root=str(tmp_path))), FileBlobStore
    )
    with pytest.raises(ValueError):
        make_blob_store(BlobStoreConfig(backend="file"))


def test_an_unknown_blob_backend_is_refused():
    with pytest.raises(ValueError):
        make_blob_store(BlobStoreConfig(backend="gcs", bucket="b"))


def test_the_s3_blob_backend_needs_a_bucket():
    with pytest.raises(ValueError):
        make_blob_store(BlobStoreConfig(backend="s3"))


def test_the_s3_blob_client_is_built_with_bounded_timeouts(monkeypatch):
    seen = {}

    def fake_client(endpoint_url, creds_env_prefix, config=None):
        seen.update(endpoint_url=endpoint_url, prefix=creds_env_prefix, config=config)
        return _FakeS3()

    monkeypatch.setattr(objectstore, "_s3_client", fake_client)
    store = make_blob_store(
        BlobStoreConfig(backend="r2", bucket="b", prefix="t", endpoint_url="https://x")
    )
    assert isinstance(store, S3BlobStore)
    assert seen["endpoint_url"] == "https://x"
    # Bounded, unlike the knowledge-file client: this one stands between a
    # review and its completion (#238).
    assert seen["config"].connect_timeout == 5
    assert seen["config"].read_timeout == 60
    assert seen["config"].retries == {"max_attempts": 2}


def test_the_transcript_store_is_read_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "transcript_store_backend", "")
    assert transcript_store() is None
    monkeypatch.setattr(settings, "transcript_store_backend", "FILE")
    monkeypatch.setattr(settings, "transcript_store_root", str(tmp_path))
    assert isinstance(transcript_store(), FileBlobStore)
