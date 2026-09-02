"""Object storage: the sqlite-vec knowledge file, and keyed write-once blobs.

TWO ACCESS PATTERNS, TWO INTERFACES, one module -- deliberately, because they
share nothing but the boto3 client :func:`_s3_client` builds and everything
about how a bucket is reached.

* **One mutable object** (:class:`FileObjectStore` / :class:`S3ObjectStore`).
  ``load()`` returns the file's bytes plus a concurrency token, and
  ``save(data, token)`` writes them back only if the object is unchanged since
  that token -- optimistic concurrency for the download -> mutate -> upload
  loop. S3/R2 use the object ETag with conditional ``PutObject``; the local
  ``file`` backend uses a content hash (mtime collides for sub-tick writes on
  fast disks, breaking the optimistic-concurrency check).
* **Many keyed, write-once blobs** (:class:`FileBlobStore` /
  :class:`S3BlobStore`, #238). A session transcript is written once, under a
  key minted before its first byte exists, and never modified. ``put`` takes
  any bytes-like body so the sidecar can hand over the ``bytearray`` it
  accumulated the request into without copying it whole a second time. A concurrency
  token means nothing for that: there is no prior version to be unchanged
  since, and the only conflict worth naming is "this key is already taken",
  which is :class:`BlobExists`. Hence a sibling interface rather than a
  ``save(data, token)`` whose second argument every caller would have to invent
  a value for (#238's first non-preference).

``boto3`` is a lazy, optional dependency (``pip install fuko-pr[sqlite]``) for
both, so a deployment using neither -- or only the local ``file`` backend --
never needs it installed.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .fukoconfig import ObjectStoreConfig


class PreconditionFailed(RuntimeError):
    """Raised when a conditional save loses a race (the object changed meanwhile)."""


def _content_token(data: bytes) -> str:
    """Return a content-derived concurrency token (an ETag-like digest)."""
    return hashlib.sha256(data).hexdigest()


class FileObjectStore:
    """Local-file backend (no server); the token is a content hash of the file."""

    def __init__(self, path: str) -> None:
        """Store the local file path that holds the knowledge db."""
        self._path = Path(path)

    def load(self) -> tuple[bytes | None, str | None]:
        """Return the file's bytes and a content token, or ``(None, None)`` if absent."""
        if not self._path.exists():
            return None, None
        data = self._path.read_bytes()
        return data, _content_token(data)

    def save(self, data: bytes, token: str | None) -> str:
        """Write ``data`` if the file is unchanged since ``token`` (else raise)."""
        exists = self._path.exists()
        if token is None and exists:
            raise PreconditionFailed("object already exists")
        if token is not None and not exists:
            raise PreconditionFailed("object deleted since load")
        if token is not None:
            try:
                current = self._path.read_bytes()
            except FileNotFoundError as exc:
                raise PreconditionFailed("object deleted since load") from exc
            if _content_token(current) != token:
                raise PreconditionFailed("object changed since load")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(data)
        return _content_token(data)


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


def _s3_client(endpoint_url: str | None, creds_env_prefix: str, config=None):
    """Build a boto3 S3 client from ``<prefix>_ACCESS_KEY_ID`` and friends.

    ``config`` is an optional ``botocore.config.Config``; the knowledge-file
    store passes none and keeps botocore's defaults, while the blob store
    passes bounded timeouts (see :func:`make_blob_store`).
    """
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get(f"{creds_env_prefix}_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get(f"{creds_env_prefix}_SECRET_ACCESS_KEY"),
        region_name=os.environ.get(f"{creds_env_prefix}_REGION", "auto"),
        config=config,
    )


def make_object_store(cfg: ObjectStoreConfig):
    """Build the object store selected by ``cfg.backend`` (``file`` | ``s3`` | ``r2``)."""
    if cfg.backend == "file":
        if not cfg.key:
            raise ValueError("object_store.key must be the local path for the 'file' backend")
        return FileObjectStore(cfg.key)

    if not (cfg.bucket and cfg.key):
        raise ValueError("object_store.bucket and .key are required for s3/r2")
    return S3ObjectStore(_s3_client(cfg.endpoint_url, cfg.creds_env_prefix), cfg.bucket, cfg.key)


# --- Keyed, write-once blobs (#238).


#: Marks the ``503`` from ``POST /transcripts/{key}`` that means "no transcript
#: store is CONFIGURED", so a runner can tell the off state from a deployment
#: fault (an unknown backend, a missing bucket, an absent ``boto3``) without
#: parsing prose. Both are 503 because both really are "this service cannot
#: store"; a caller that ignores the header degrades safely, to reporting.
#:
#: The two constants live HERE, in the module that defines what a transcript
#: blob store is, because that is the only thing the endpoint
#: (:func:`sidecar.main.transcripts_put_endpoint`) and the runner-side shipper
#: (:func:`sidecar.reviewer.transcript_client.ship`) already share -- importing
#: one from the other is a cycle.
STORE_HEADER = "X-Fuko-Transcript-Store"
STORE_UNCONFIGURED = "unconfigured"


class BlobExists(RuntimeError):
    """Raised when a write-once ``put`` names a key the store already holds."""


#: What a blob key may be. ONE path segment of conservative characters, so the
#: key travels through a URL path, an S3 object name and a local filename
#: without meaning anything different in any of them.
#:
#: This is an ALLOWLIST because the key is attacker-shaped input at the sidecar
#: boundary: ``POST /transcripts/{key}`` puts it in a URL path, and the ``file``
#: backend turns it into a filename. Excluding ``/`` and ``\`` leaves no way to
#: name a directory, and requiring an alphanumeric first character rules out
#: ``..`` and dotfiles -- so a traversal is not merely normalized away
#: somewhere upstream, it is unrepresentable.
#:
#: :func:`sidecar.reviewer.transcript.mint_key` emits a strict subset
#: (``<UTC stamp>-<12 hex>``); the pattern is wider so a later minting rule does
#: not have to change the storage layer to be storable.
BLOB_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_blob_key(key: str) -> str:
    """Return ``key`` if it satisfies :data:`BLOB_KEY_RE`, else raise ``ValueError``."""
    if not isinstance(key, str) or not BLOB_KEY_RE.fullmatch(key):
        raise ValueError(f"invalid blob key {key!r}")
    return key


class FileBlobStore:
    """Keyed write-once blobs as owner-only files under one local directory.

    The create is ATOMIC and exclusive in one step: the bytes land in a
    temporary file and are then ``os.link``-ed into place, which fails outright
    if the key is taken. Writing straight into the final name with ``O_EXCL``
    would be exclusive but not atomic -- a sidecar killed mid-write would leave
    a truncated blob under a key that write-once semantics then forbid anyone
    from replacing, and #238 asks for bytes identical to what capture produced
    or nothing at all.

    Modes match :class:`sidecar.reviewer.transcript.FileTranscriptSink`, for the
    same reason: scrubbing removes the credential VALUES the driver holds, so
    what remains is the whole reviewed repository as the agent read it.
    """

    DIR_MODE = 0o700
    FILE_MODE = 0o600

    def __init__(self, root: str) -> None:
        """Store the directory that holds the blobs; it is created on first put."""
        self._root = Path(root)

    def put(self, key: str, data: bytes | bytearray) -> None:
        """Write ``data`` under ``key``, or raise :class:`BlobExists` if taken."""
        target = self._root / validate_blob_key(key)
        self._root.mkdir(mode=self.DIR_MODE, parents=True, exist_ok=True)
        # `mkstemp` rather than a name derived from the key: it creates the file
        # 0600 as it opens it (the mode has to be set AS the file is created,
        # same argument as FileTranscriptSink) and its name is unique across
        # concurrent puts, which a pid-derived one would not be for two threads
        # of one process -- and the sidecar puts from a threadpool.
        fd, staging_name = tempfile.mkstemp(dir=self._root, prefix=f".{key}.", suffix=".part")
        staging = Path(staging_name)
        try:
            with open(fd, "wb") as handle:
                handle.write(data)
            try:
                os.link(staging, target)
            except FileExistsError as exc:
                raise BlobExists(f"blob {key} already exists") from exc
        finally:
            staging.unlink(missing_ok=True)

    def get(self, key: str) -> bytes | None:
        """Return the blob's bytes, or ``None`` when the key holds nothing."""
        try:
            return (self._root / validate_blob_key(key)).read_bytes()
        except FileNotFoundError:
            return None


class S3BlobStore:
    """Keyed write-once blobs in an S3/R2 bucket under one key prefix.

    Write-once is enforced by the SERVER, not by a read-then-write check here:
    ``IfNoneMatch="*"`` makes the create conditional, so two runners racing on
    one key produce one stored object and one :class:`BlobExists` rather than a
    last-writer-wins overwrite.
    """

    def __init__(self, client, bucket: str, prefix: str = "") -> None:
        """Wrap a boto3 S3 client bound to ``bucket``, naming blobs under ``prefix``."""
        self._client = client
        self._bucket = bucket
        cleaned = prefix.strip().strip("/")
        self._prefix = f"{cleaned}/" if cleaned else ""

    def put(self, key: str, data: bytes | bytearray) -> None:
        """Conditionally create the object, mapping a lost race to ``BlobExists``."""
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._prefix + validate_blob_key(key),
                Body=data,
                IfNoneMatch="*",
            )
        except Exception as exc:
            code = _error_code(exc)
            if "PreconditionFailed" in code or "412" in code:
                raise BlobExists(f"blob {key} already exists") from exc
            raise

    def get(self, key: str) -> bytes | None:
        """Return the object's bytes, or ``None`` when the key holds nothing."""
        try:
            resp = self._client.get_object(
                Bucket=self._bucket, Key=self._prefix + validate_blob_key(key)
            )
        except Exception as exc:
            code = _error_code(exc)
            if "NoSuchKey" in code or "404" in code:
                return None
            raise
        body = resp["Body"]
        try:
            return body.read()
        finally:
            body.close()


@dataclass(frozen=True)
class BlobStoreConfig:
    """Where keyed write-once blobs live.

    Environment-shaped rather than ``.fuko.toml``-shaped, and that is the whole
    reason this is a dataclass here instead of a pydantic model beside
    :class:`sidecar.fukoconfig.ObjectStoreConfig`: the deployed sidecar has no
    repo checkout to read a toml FROM. ``docker/Dockerfile.sidecar`` copies
    ``sidecar/`` and ``migrations/`` into ``/app`` and nothing else, so
    :func:`sidecar.fukoconfig.load_config` finds no file and returns defaults;
    ``docker/runner-compose.yml`` configures the service purely through
    ``FUKO_*``. That is the same argument #216 made for the embedding endpoint
    ("the sidecar runs as its own process and may hold no repo checkout at
    all"), and transcripts land on the same side of it.
    """

    backend: str = ""
    root: str = ""
    bucket: str = ""
    prefix: str = ""
    endpoint_url: str | None = None
    creds_env_prefix: str = "FUKO_S3"

    @classmethod
    def from_settings(cls) -> BlobStoreConfig:
        """Read the ``FUKO_TRANSCRIPT_STORE_*`` environment into this shape."""
        return cls(
            backend=settings.transcript_store_backend.strip().lower(),
            root=settings.transcript_store_root,
            bucket=settings.transcript_store_bucket,
            prefix=settings.transcript_store_prefix,
            endpoint_url=settings.transcript_store_endpoint_url or None,
            creds_env_prefix=settings.transcript_store_creds_env_prefix,
        )


def local_blob_root(cfg: BlobStoreConfig) -> Path | None:
    """The resolved directory the ``file`` backend writes into, or ``None``.

    ``None`` for every other backend: an ``s3``/``r2`` root is a bucket prefix
    and names nothing on this host.

    ONE resolver for the two callers that must agree -- :func:`make_blob_store`,
    which opens the store, and :mod:`sidecar.backends.agentic`, which hands the
    directory to the reviewer's read denylist. They have to see the same path or
    the blobs land somewhere no rule covers, which is the whole hazard
    :func:`sidecar.reviewer.transcript.transcript_dir` was written against.

    Resolved, and the filesystem ROOT is refused, for exactly the reasons stated
    there: ``_permission_settings`` normalizes a candidate with ``rstrip("/")``,
    which turns ``"/"`` into the empty string, and an empty candidate is dropped
    WITHOUT reaching the non-POSIX announcement -- so a root store would keep a
    transcript corpus that no rule covers and nothing reports. A symlinked root
    is the same failure in a different spelling: the rule would name the alias
    while the store writes through to the target.
    """
    if cfg.backend != "file" or not cfg.root:
        return None
    resolved = Path(cfg.root).expanduser().resolve()
    if resolved.parent == resolved:
        raise ValueError(
            f"transcript store root {resolved} is the filesystem root; "
            "set FUKO_TRANSCRIPT_STORE_ROOT to a dedicated directory"
        )
    return resolved


def make_blob_store(cfg: BlobStoreConfig):
    """Build the blob store ``cfg`` selects, or ``None`` when no backend is set.

    An empty backend is the OFF state rather than an error: object storage
    becoming newly relevant to Postgres deployments (#238's stated risk) must
    not mean an unconfigured one fails to start. Configured-but-incomplete IS
    an error, because that is a deployment that meant to store something.

    The S3 client is given explicit, bounded timeouts, unlike the
    knowledge-file client. That store is reached from a CLI command which can
    afford botocore's minutes-long defaults against a dead endpoint; this one is
    reached from a request handler standing between a review and its
    completion, and a hung PUT there would hold a worker for the whole default
    retry ladder.
    """
    if not cfg.backend:
        return None
    if cfg.backend == "file":
        if not cfg.root:
            raise ValueError("the 'file' transcript store needs FUKO_TRANSCRIPT_STORE_ROOT")
        return FileBlobStore(str(local_blob_root(cfg)))
    if cfg.backend not in ("s3", "r2"):
        raise ValueError(f"unknown transcript store backend {cfg.backend!r} (file | s3 | r2)")
    if not cfg.bucket:
        raise ValueError("the s3/r2 transcript store needs FUKO_TRANSCRIPT_STORE_BUCKET")
    # `boto3` before `botocore`, deliberately: on an install without the `s3`
    # extra BOTH are missing, and whichever is imported first names the module
    # in the `ModuleNotFoundError` the endpoint turns into a 503. Naming the
    # package an operator installs (`fuko-pr[s3]` -> boto3) rather than its
    # transitive dependency is the difference between a log line that answers
    # the question and one that starts another.
    import boto3  # noqa: F401
    from botocore.config import Config

    client = _s3_client(
        cfg.endpoint_url,
        cfg.creds_env_prefix,
        Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 2}),
    )
    return S3BlobStore(client, cfg.bucket, cfg.prefix)


def transcript_store():
    """The blob store session transcripts go to, or ``None`` when unconfigured.

    Built per call rather than cached. A client costs milliseconds next to the
    multi-megabyte upload it is built for, and the alternative -- a module-level
    singleton -- would need a reset hook that exists only for tests and would
    let a process keep serving a store its configuration no longer describes.
    """
    return make_blob_store(BlobStoreConfig.from_settings())
