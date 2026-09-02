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
  peak stays what #237 made it -- one event, not one session;
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
    with path.open("rb") as handle:
        resp = httpx.post(
            f"{fuko_url.rstrip('/')}/transcripts/{key}",
            content=handle,
            headers=headers,
            timeout=UPLOAD_TIMEOUT_S,
        )
    resp.raise_for_status()
