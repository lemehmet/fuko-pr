"""Object-storage sync for the single sqlite-vec knowledge file.

Narrow interface: ``load()`` returns the file's bytes plus a concurrency token,
and ``save(data, token)`` writes them back only if the object is unchanged since
that token -- optimistic concurrency for the download -> mutate -> upload loop.
S3/R2 use the object ETag with conditional ``PutObject``; the local ``file``
backend uses the file's mtime. ``boto3`` is a lazy, optional dependency
(``pip install fuko-pr[sqlite]``).
"""

from __future__ import annotations

import os
from pathlib import Path

from .fukoconfig import ObjectStoreConfig


class PreconditionFailed(RuntimeError):
    """Raised when a conditional save loses a race (the object changed meanwhile)."""


class FileObjectStore:
    """Local-file backend (no server); the token is the file's mtime in ns."""

    def __init__(self, path: str) -> None:
        """Store the local file path that holds the knowledge db."""
        self._path = Path(path)

    def load(self) -> tuple[bytes | None, str | None]:
        """Return the file's bytes and an mtime token, or ``(None, None)`` if absent."""
        if not self._path.exists():
            return None, None
        return self._path.read_bytes(), str(self._path.stat().st_mtime_ns)

    def save(self, data: bytes, token: str | None) -> str:
        """Write ``data`` if the file is unchanged since ``token`` (else raise)."""
        exists = self._path.exists()
        if token is None and exists:
            raise PreconditionFailed("object already exists")
        if token is not None and not exists:
            raise PreconditionFailed("object deleted since load")
        if token is not None and str(self._path.stat().st_mtime_ns) != token:
            raise PreconditionFailed("object changed since load")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(data)
        return str(self._path.stat().st_mtime_ns)


def _error_code(exc: Exception) -> str:
    """Extract a botocore-style error code/status from a ClientError, if present."""
    resp = getattr(exc, "response", None) or {}
    err = resp.get("Error", {}) if isinstance(resp, dict) else {}
    status = (
        resp.get("ResponseMetadata", {}).get("HTTPStatusCode") if isinstance(resp, dict) else None
    )
    return f"{err.get('Code', '')}:{status}"


class S3ObjectStore:
    """S3/R2 backend using conditional ``PutObject`` (ETag) for safe write-back."""

    def __init__(self, client, bucket: str, key: str) -> None:
        """Wrap a boto3 S3 client bound to ``bucket``/``key``."""
        self._client = client
        self._bucket = bucket
        self._key = key

    def load(self) -> tuple[bytes | None, str | None]:
        """Get the object's bytes and ETag, or ``(None, None)`` if it doesn't exist."""
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=self._key)
        except Exception as exc:
            code = _error_code(exc)
            if "NoSuchKey" in code or "404" in code:
                return None, None
            raise
        body = resp["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        return data, resp["ETag"]

    def save(self, data: bytes, token: str | None) -> str:
        """Conditionally put the object: create-only when new, If-Match otherwise."""
        kwargs = {"Bucket": self._bucket, "Key": self._key, "Body": data}
        if token is None:
            kwargs["IfNoneMatch"] = "*"
        else:
            kwargs["IfMatch"] = token
        try:
            resp = self._client.put_object(**kwargs)
        except Exception as exc:
            code = _error_code(exc)
            if "PreconditionFailed" in code or "412" in code or "PreconditionRequired" in code:
                raise PreconditionFailed("conditional put failed (object changed)") from exc
            raise
        return resp["ETag"]


def make_object_store(cfg: ObjectStoreConfig):
    """Build the object store selected by ``cfg.backend`` (``file`` | ``s3`` | ``r2``)."""
    if cfg.backend == "file":
        if not cfg.key:
            raise ValueError("object_store.key must be the local path for the 'file' backend")
        return FileObjectStore(cfg.key)

    if not (cfg.bucket and cfg.key):
        raise ValueError("object_store.bucket and .key are required for s3/r2")
    import boto3

    prefix = cfg.creds_env_prefix
    client = boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=os.environ.get(f"{prefix}_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get(f"{prefix}_SECRET_ACCESS_KEY"),
        region_name=os.environ.get(f"{prefix}_REGION", "auto"),
    )
    return S3ObjectStore(client, cfg.bucket, cfg.key)
