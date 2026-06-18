"""Unit tests for the object-storage sync layer (file + S3/R2 conditional writes)."""

import io

import pytest

from sidecar.fukoconfig import ObjectStoreConfig
from sidecar.objectstore import (
    FileObjectStore,
    PreconditionFailed,
    S3ObjectStore,
    make_object_store,
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
