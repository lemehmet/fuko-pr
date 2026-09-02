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
  worth less than the wall-clock it charges the review's completion path. The
  worst case that costs is :data:`UPLOAD_CEILING_S`, stated in full there;
* a sidecar that has no store CONFIGURED answers ``503`` with
  ``X-Fuko-Transcript-Store: unconfigured``, which is the off state rather than
  a failure and is silent here -- so turning capture on before storage, the
  staged rollout the deployment docs recommend, does not print a line per run.
  A 503 without that header is a store that was meant to work and does not, and
  reports like anything else.

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

from ..objectstore import STORE_HEADER, STORE_UNCONFIGURED

#: How long the runner will wait for a transcript upload, in seconds.
#:
#: ``/metrics/run``'s 10s prices a small JSON row on a LAN; this prices a
#: multi-megabyte NDJSON body, and the acceptance criterion it answers is that
#: an upload substantially larger than a metrics row completes without tripping
#: the timeout governing its path.
#:
#: It bounds the BODY TRANSFER absolutely (:func:`_deadlined`). The whole
#: request's worst case is :data:`UPLOAD_CEILING_S`, which is this plus the
#: phases a clock check cannot interleave with.
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

#: Bound on one socket write and on waiting for the response.
#:
#: Deliberately far below :data:`UPLOAD_TIMEOUT_S`. The deadline is only tested
#: BETWEEN chunks, so the chunk in flight when it passes still runs out its own
#: write timeout; giving that phase the full ceiling would make the true worst
#: case several multiples of the number this module promises. Thirty seconds for
#: :data:`UPLOAD_CHUNK_BYTES` is 2 KB/s, which no working peer approaches, and
#: for the response it is a few hundred bytes of JSON.
PHASE_TIMEOUT_S = 30.0

#: A CONSERVATIVE bound on what this module can cost a review's completion.
#:
#: Stated rather than implied, because the arithmetic is the whole point: a
#: connect, then the absolutely-bounded body, then one more phase. Conservative
#: because the last term cannot be paid twice -- either the deadline passes and
#: the run ends inside the write already in flight (no response is ever
#: awaited), or the body finishes in time and the response's STATUS LINE is
#: awaited under the same bound. It is summed once for whichever occurs.
#:
#: This holds only because :func:`ship` never reads the response BODY (see the
#: comment there): a read to completion is bounded per chunk, not in total, and
#: would make this number a claim rather than a ceiling. Nothing else is
#: attempted -- there is no retry -- so this is the ceiling, not a typical cost
#: (a real upload finishes in well under a second on a LAN).
#:
#: It is a CLIENT-side bound, and the sidecar's own store call is not inside it:
#: a pathological bucket endpoint can hold ``put_object`` for longer than this
#: waits for the response, in which case the runner reports a failure for a
#: transcript that then lands. Nothing is lost or duplicated -- keys are freshly
#: minted and there is no retry -- but a reader of that stderr line should not
#: assume the blob is missing.
UPLOAD_CEILING_S = CONNECT_TIMEOUT_S + UPLOAD_TIMEOUT_S + PHASE_TIMEOUT_S


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


def ship(key: str, path: Path) -> bool:
    """Send the finished transcript at ``path`` to shared storage under ``key``.

    Raises on any failure; the caller (:class:`sidecar.reviewer.transcript.
    Transcript`) is what turns that into one stderr line and an inert capture.

    Returns whether the bytes were STORED. ``False`` has exactly one source --
    the sidecar's marked off state below, which is a silent success as far as
    the review is concerned but is not a stored blob -- and it exists because
    #239's index row names a blob a reader is meant to be able to fetch. Without
    it the recommended staged rollout (capture on before storage) would write a
    reference per run to something that is only ever on the runner's disk.
    """
    fuko_url = os.environ.get("FUKO_URL", "").strip()
    if not fuko_url:
        from ..objectstore import transcript_store

        store = transcript_store()
        if store is None:
            raise RuntimeError("no transcript store configured")
        store.put(key, path.read_bytes())
        return True

    headers = {"Content-Type": "application/x-ndjson"}
    token = os.environ.get("FUKO_TOKEN", "")
    if token:
        headers["Authorization"] = "Bearer " + token
    deadline = time.monotonic() + UPLOAD_TIMEOUT_S
    # STREAMED, and the response body is never read. `httpx.post` reads the
    # response to completion before returning, and its `read` timeout bounds the
    # gap between chunks rather than the whole read -- so a peer trickling a
    # body would keep this blocked past any ceiling this module could state.
    # Everything the caller needs is the status line and one header, both
    # available before the body, so not reading it is both the cheaper and the
    # only bounded option. `raise_for_status` builds its message from the
    # status and the URL, never from content, so it is safe on an unread
    # response.
    with path.open("rb") as handle:
        with httpx.stream(
            "POST",
            f"{fuko_url.rstrip('/')}/transcripts/{key}",
            content=_deadlined(handle, deadline),
            headers=headers,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_S,
                read=PHASE_TIMEOUT_S,
                write=PHASE_TIMEOUT_S,
                pool=CONNECT_TIMEOUT_S,
            ),
        ) as resp:
            status_code, resp_headers = resp.status_code, resp.headers
            if status_code == 503 and resp_headers.get(STORE_HEADER) == STORE_UNCONFIGURED:
                # Storage is not turned on. That is the OFF STATE, not this
                # run's failure: reporting it would print a line per run, per
                # seat, on every fleet that turned capture on before storage --
                # which is the staged rollout the deployment docs recommend.
                #
                # ONLY this shape. A 503 without the header means a store that
                # was meant to work does not (an unknown backend, a missing
                # bucket, a bucket backend without boto3), and swallowing that
                # would make the feature store nothing in silence -- the exact
                # failure the endpoint's distinguished statuses exist to
                # prevent. It raises like any other.
                #
                # `False` rather than `None`: silent for the operator, but the
                # bytes are NOT in the store, and #239's caller has to know
                # that before it writes a reference to them.
                return False
            resp.raise_for_status()
            return True
