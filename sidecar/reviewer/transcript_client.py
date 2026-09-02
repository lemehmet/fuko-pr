"""How a finished transcript leaves the runner (#238).

**The upload path, and why this one.** #236 left the fork open and #238 asks for
it to be decided in the open: either the runner gains its own blob-store
credentials, or the bytes travel through the sidecar. This module is the second.

The runner is configured with ``FUKO_URL`` + ``FUKO_TOKEN`` and nothing else --
no connection string, no bucket, no keys -- and every other piece of shared
state it produces already crosses that same boundary: ``/cb/trip``,
``/cb/cooldowns``, ``/metrics/run``, ``/rh/observe``, ``/rh/state``, and the
whole review ledger (:mod:`sidecar.review_state_client`). The alternative costs
a new pair of secrets in every consuming repo's workflows, at four sites each
(mepro and shuanda pin fuko-pr in four places apiece), on machines that run an
agent over contributor-controlled code. Storage credentials that never reach the
runner cannot leak from it, and this PR adds none to any workflow.

What the sidecar-mediated path costs is a large body over a boundary otherwise
budgeted for small JSON rows, which is the shape that produced the sweep-ingest
timeout. It is paid for here rather than papered over:

* a DEDICATED endpoint (``POST /transcripts/{key}``) with its own
  :data:`UPLOAD_TIMEOUT_S`, an order of magnitude above ``/metrics/run``'s 10
  seconds, instead of widening the metrics row into a body it was never sized
  for;
* the body is STREAMED off disk rather than read into memory, so the runner's
  peak stays what #237 made it -- one chunk, not one session -- and the stream
  carries an ABSOLUTE deadline, because ``httpx``'s ``timeout=`` is per phase
  and would otherwise let a slow peer outlast the ceiling this promises;
* exactly ONE attempt. The blob is write-once, so a retry after a client-side
  timeout races the upload that may already have landed and answers ``409``;
  and a transcript is an observability artifact, so a second chance at it is
  worth less than the wall-clock it charges the review's completion path.

The ``FUKO_URL``-unset path (a laptop ``fuko review``, or a host that IS the
sidecar) writes straight to the configured store, mirroring
:func:`sidecar.runner._record_run`'s own two transports.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

#: How long the runner will wait for a transcript upload, in seconds.
#:
#: ``/metrics/run``'s 10s prices a small JSON row on a LAN; this prices a
#: multi-megabyte NDJSON body, and the acceptance criterion it answers is that
#: an upload substantially larger than a metrics row completes without tripping
#: the timeout governing its path. It is also the entire latency this adds to a
#: review's completion in the worst case, which is why it is a ceiling rather
#: than a generous ceiling.
UPLOAD_TIMEOUT_S = 120.0

#: How much of the file is handed to the transport at a time.
#:
#: Iterating the open handle directly would yield one chunk per NDJSON LINE --
#: a chunked write per event, on a file whose whole point is that it has many.
#: A fixed read keeps the peak at one chunk while making the number of writes a
#: function of size rather than of event count.
UPLOAD_CHUNK_BYTES = 64 * 1024

#: Per-phase bound on the parts of the request a deadline cannot reach.
#:
#: :data:`UPLOAD_TIMEOUT_S` bounds the body transfer ABSOLUTELY (see
#: :func:`_deadlined`); connecting and waiting for the response are still
#: phase-scoped, because there is nothing to interleave a clock check with.
CONNECT_TIMEOUT_S = 10.0


def _deadlined(handle, deadline: float) -> Iterator[bytes]:
    """Yield ``handle`` in chunks, refusing to continue past ``deadline``.

    ``httpx``'s ``timeout=`` is PER PHASE -- connect, write, read, pool -- and
    it has no request lifetime, so a peer that accepts a chunk just often enough
    to reset the write timeout can hold the upload open indefinitely. The
    docstring above promises :data:`UPLOAD_TIMEOUT_S` is the entire latency this
    adds to a review's completion, and a per-phase timeout does not deliver
    that; checking the clock between chunks does, for the phase that carries the
    bytes.
    """
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"transcript upload exceeded UPLOAD_TIMEOUT_S ({UPLOAD_TIMEOUT_S}s)")
        chunk = handle.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            return
        yield chunk


def upload_target() -> str:
    """Name where a transcript would be shipped, or ``""`` when nowhere.

    Consulted BEFORE capture opens, so a deployment with capture on and no
    destination keeps #237's behaviour exactly -- a local file and no shipping
    attempt -- rather than reporting a failure per run for something it never
    asked for.
    """
    from ..objectstore import BlobStoreConfig

    if os.environ.get("FUKO_URL", "").strip():
        return "sidecar"
    return "store" if BlobStoreConfig.from_settings().backend else ""


def ship(key: str, path: Path) -> None:
    """Send the finished transcript at ``path`` to shared storage under ``key``.

    Raises on any failure; the caller (:class:`sidecar.reviewer.transcript.
    Transcript`) is what turns that into one stderr line and an inert capture.
    """
    fuko_url = os.environ.get("FUKO_URL", "").strip()
    if not fuko_url:
        from ..objectstore import transcript_store

        store = transcript_store()
        if store is None:
            raise RuntimeError("no transcript store configured")
        store.put(key, path.read_bytes())
        return

    headers = {"Content-Type": "application/x-ndjson"}
    token = os.environ.get("FUKO_TOKEN", "")
    if token:
        headers["Authorization"] = "Bearer " + token
    deadline = time.monotonic() + UPLOAD_TIMEOUT_S
    with path.open("rb") as handle:
        resp = httpx.post(
            f"{fuko_url.rstrip('/')}/transcripts/{key}",
            content=_deadlined(handle, deadline),
            headers=headers,
            timeout=httpx.Timeout(
                UPLOAD_TIMEOUT_S, connect=CONNECT_TIMEOUT_S, pool=CONNECT_TIMEOUT_S
            ),
        )
    resp.raise_for_status()
