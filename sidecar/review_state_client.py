"""The review ledger's transport: the sidecar over HTTP, or the local store.

:mod:`sidecar.review_state` reaches Postgres directly, and on the deployed fleet
nothing that calls it can. The agentic backend runs on the REVIEW RUNNER --
it needs a checkout and spawns the harness there -- and a runner is configured
with ``FUKO_URL`` + ``FUKO_TOKEN`` only; the LAN sidecar holds the database and
the runner never gets a connection string. Every ledger call therefore took
:func:`sidecar.review_state._best_effort`'s no-op path and each round was
byte-for-byte a pre-ledger round: a correct degradation (#156's acceptance names
it) but one that made nine merged PRs of epic #160 inert wherever it matters
(#171).

This module is the seam that closes that, in the shape the runner already uses
for every other piece of shared state -- ``/cb/trip``, ``/cb/cooldowns``,
``/metrics/run``, ``/rh/observe``, ``/rh/state``: **sidecar over HTTP when
``FUKO_URL`` is set, else the local module.** ``FUKO_URL`` wins over a locally
configured ``FUKO_DATABASE_URL`` for the same reason it does in
:mod:`sidecar.runner` -- a host with both is a sidecar-hosted runner, and the
sidecar is the copy of the state the rest of the fleet shares.

Three properties, each of which is why this is a module and not a few lines in
:mod:`sidecar.reviewer.ledger`:

* **One semantics, not two.** The endpoints are 1:1 with the primitives, and
  each handler IS the matching :mod:`sidecar.review_state` function. The
  alternative -- the "one read, one write" pair #171 sketches -- founders on
  :func:`sidecar.reviewer.ledger.settle`, whose reopen candidates are read AFTER
  its verdicts are applied, on purpose: collapsing that into a single write would
  move the reopen and dedup policy to the server and split a policy that
  ``ledger.py``'s charter keeps in one place. Chattier, and worth it.
* **Best-effort stays best-effort.** Every function here returns the same
  neutral value ``_best_effort`` returns for it, for a sidecar that is
  unreachable, slow, erroring or answering nonsense. A review must never fail
  because its ledger did.
* **A dead sidecar costs one timeout, not ten.** #170 records a
  configured-but-unreachable Postgres costing 30s per best-effort call; ten
  primitives behind a 5s HTTP timeout would reproduce exactly that shape at
  50s per round. So the first TRANSPORT failure latches this process offline
  (:func:`_offline`) and the rest of the round's calls no-op immediately.

The request and response bodies live here rather than in :mod:`sidecar.models`
because that module is charter-bound to import nothing from the rest of the
package, while these bodies carry the reviewer's own vocabulary types
(:class:`sidecar.reviewer.prompt.AgenticFinding` and friends). Reusing those
types rather than restating their fields is deliberate: a wire schema that
merely resembles the stored one is a schema that drifts from it.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Sequence
from functools import wraps

import httpx
from pydantic import BaseModel, Field

from . import review_state
from .review_state import OpenLedger, SettledFinding, StoredFinding
from .reviewer.prompt import AgenticFinding, ExaminedRegion, PriorCoverage

TIMEOUT_S = 5.0
"""Per-request budget for the ledger endpoints, in seconds.

Half what the observability seams in :mod:`sidecar.runner` allow themselves
(``/metrics/run``, ``/rh/observe``), and for a reason those do not have: a
metrics POST happens after the review, where a stalled call costs only its own
latency, while :func:`sidecar.reviewer.ledger.carry_in` blocks prompt
construction -- the review cannot start until it returns. On the LAN these calls
answer in single-digit milliseconds, so this bound is only ever reached by a
sidecar that is wedged or black-holed, and for that case the latch below matters
more than the number.
"""


class LedgerScope(BaseModel):
    """The ``(repo, pr, seat)`` lane every ledger request names.

    Present on the id-addressed writes as well as the reads, which is the half
    that is new over the wire: in-process a finding id can only have come out of
    :func:`sidecar.review_state.open_findings` for the same lane, whereas in a
    request body its provenance is a claim. The store re-checks it in SQL
    (:func:`sidecar.review_state.transition`), so the endpoints cannot be used to
    close, re-raise or touch another seat's row -- the cross-seat coupling #160
    forbids.
    """

    repo: str
    pr: int
    seat: str


class RecordFindingsRequest(LedgerScope):
    """One round's newly-opened findings."""

    round: int = 1
    head_sha: str = ""
    findings: list[AgenticFinding] = Field(default_factory=list)


class RecordCoverageRequest(LedgerScope):
    """One round's examined regions."""

    round: int = 1
    head_sha: str = ""
    regions: list[ExaminedRegion] = Field(default_factory=list)


class ExpireCoverageRequest(LedgerScope):
    """The files whose coverage this round's delta invalidated.

    ``files`` is required and a list. :func:`sidecar.review_state.expire_coverage`
    reads ``None`` as "expire this seat's coverage WHOLESALE", which is the
    rebase/force-push case -- and no ledger caller reaches it, so the wire does
    not carry it. A field that defaulted to ``None`` would turn an omitted key,
    a typo'd key or a truncated body into the most destructive call the ledger
    has; the same ``None``-vs-``[]`` hazard
    :func:`sidecar.review_state.expire_coverage` already warns about, arriving
    through serialization. Wholesale expiry stays a local-module call until
    something needs it over the wire.
    """

    files: list[str]


class TransitionRequest(LedgerScope):
    """A verdict closing one carried finding."""

    finding_id: str
    status: str
    reason: str = ""


class ReopenRequest(LedgerScope):
    """A re-raise of one finding an earlier round closed by verdict."""

    finding_id: str
    reason: str = ""


class TouchRequest(LedgerScope):
    """The findings a round re-asserted rather than settled."""

    finding_ids: list[str] = Field(default_factory=list)


class OpenFindingsResponse(BaseModel):
    """This seat's open ledger as one read saw it, cut included."""

    rows: list[StoredFinding] = Field(default_factory=list)
    truncated: int = 0


class SettledFindingsResponse(BaseModel):
    """This seat's model-closed findings, the projection a re-raise needs."""

    rows: list[SettledFinding] = Field(default_factory=list)


class LiveCoverageResponse(BaseModel):
    """This seat's unexpired coverage entries."""

    rows: list[PriorCoverage] = Field(default_factory=list)


class NextRoundResponse(BaseModel):
    """The round number this seat's next round records under."""

    round: int = 1


class LedgerCountResponse(BaseModel):
    """How many rows a ledger write affected."""

    count: int = 0


class LedgerChangedResponse(BaseModel):
    """Whether a single-row ledger write changed anything."""

    changed: bool = False


_LOCK = threading.Lock()
_transport_down = False


def _endpoint() -> tuple[str, str]:
    """Return ``(fuko_url, fuko_token)`` for the sidecar's ledger API.

    Read per call rather than captured at import, matching
    :func:`sidecar.runner._cb_endpoint`: the runner's tests and the CLI both set
    these after the module is loaded.
    """
    return os.environ.get("FUKO_URL", "").strip(), os.environ.get("FUKO_TOKEN", "")


def _mark_down(url: str, exc: Exception) -> None:
    """Latch this process offline after the first TRANSPORT failure, once.

    Bounds the cost of a wedged or black-holed sidecar at one
    :data:`TIMEOUT_S` for the whole round instead of one per primitive (#170's
    shape). Deliberately NOT tripped by an HTTP status: a 500 answers as fast as
    a 200, so latching on it would turn one broken handler into a lost ledger for
    every remaining call, and the latch exists to bound time, not errors.

    Process-wide and never reset, which is the right lifetime: a runner process
    is one review run, and an A/B run's branches are threads of it that share the
    one sidecar. The worst case is the degradation the epic already accepts --
    the rest of this round reviews exactly as it did before the ledger existed.
    """
    global _transport_down
    with _LOCK:
        if _transport_down:
            return
        _transport_down = True
    print(
        f"fuko: review-state sidecar at {url} did not answer ({exc}); "
        "this run continues without the ledger",
        file=sys.stderr,
    )


def _guarded(default_factory):
    """Make a ledger call no-op instead of raising into the review.

    The transport half of :func:`sidecar.review_state._best_effort`, and it
    returns the same neutral values, because a caller must not be able to tell
    which transport degraded. Everything is caught: a refused connection, a
    timeout, a 4xx/5xx, a body that does not parse. The default is a FACTORY for
    the same reason it is there -- a caller must never be handed a shared mutable
    default to append to.

    It wraps the local branch too, where :func:`sidecar.review_state._best_effort`
    already guarantees no exception escapes. That is not redundant so much as
    unconditional: the guarantee belongs to the seam rather than to whichever
    branch it happens to take, so a later transport (or a stubbed primitive)
    cannot reach a caller through an unguarded path. The line names the endpoint
    so the two branches stay distinguishable in a log.
    """

    def _decorate(fn):
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except httpx.TransportError as e:
                _mark_down(_endpoint()[0] or "?", e)
                return default_factory()
            except Exception as e:
                where = _endpoint()[0] or "local store"
                print(f"fuko: review-state {fn.__name__} failed ({where}): {e}", file=sys.stderr)
                return default_factory()

        return _wrapped

    return _decorate


def _headers(token: str, *, body: bool) -> dict[str, str]:
    """Auth (and content-type for a body), matching the other runner seams."""
    headers = {"Content-Type": "application/json"} if body else {}
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _get(path: str, scope: dict) -> dict:
    """GET one ledger read, raising on anything but a parsed 2xx JSON object."""
    url, token = _endpoint()
    resp = httpx.get(
        url.rstrip("/") + path,
        params=scope,
        headers=_headers(token, body=False),
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: BaseModel) -> dict:
    """POST one ledger write, raising on anything but a parsed 2xx JSON object."""
    url, token = _endpoint()
    resp = httpx.post(
        url.rstrip("/") + path,
        content=payload.model_dump_json(),
        headers=_headers(token, body=True),
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def _remote() -> str:
    """The sidecar base URL to use, or ``""`` for the local module.

    Empty once the latch has tripped, so the rest of the run takes the local
    branch. That is a fall-back to the same state by another route, never to a
    second copy of it: on the runners this issue is about there is no
    ``FUKO_DATABASE_URL`` at all, so the local branch is
    :func:`sidecar.review_state._best_effort`'s no-op and the round degrades
    exactly as #156 permits -- while on a sidecar-hosted runner, where both are
    configured, it is the same Postgres the sidecar would have written. A round
    can therefore split across the two branches without the halves disagreeing.
    """
    if _transport_down:
        return ""
    return _endpoint()[0]


@_guarded(OpenLedger)
def open_findings(repo: str, pr: int, seat: str) -> OpenLedger:
    """This seat's still-open findings; see :func:`sidecar.review_state.open_findings`."""
    if not _remote():
        return review_state.open_findings(repo, pr, seat)
    parsed = OpenFindingsResponse.model_validate(
        _get("/rs/findings", {"repo": repo, "pr": pr, "seat": seat})
    )
    return OpenLedger(rows=tuple(parsed.rows), truncated=parsed.truncated)


@_guarded(lambda: 1)
def next_round(repo: str, pr: int, seat: str) -> int:
    """This seat's next round number; see :func:`sidecar.review_state.next_round`."""
    if not _remote():
        return review_state.next_round(repo, pr, seat)
    return NextRoundResponse.model_validate(
        _get("/rs/round", {"repo": repo, "pr": pr, "seat": seat})
    ).round


@_guarded(tuple)
def settled_findings(repo: str, pr: int, seat: str) -> tuple[SettledFinding, ...]:
    """This seat's model-closed findings; see :func:`sidecar.review_state.settled_findings`."""
    if not _remote():
        return review_state.settled_findings(repo, pr, seat)
    return tuple(
        SettledFindingsResponse.model_validate(
            _get("/rs/settled", {"repo": repo, "pr": pr, "seat": seat})
        ).rows
    )


@_guarded(list)
def live_coverage(repo: str, pr: int, seat: str) -> list[PriorCoverage]:
    """This seat's unexpired coverage; see :func:`sidecar.review_state.live_coverage`."""
    if not _remote():
        return review_state.live_coverage(repo, pr, seat)
    return list(
        LiveCoverageResponse.model_validate(
            _get("/rs/coverage", {"repo": repo, "pr": pr, "seat": seat})
        ).rows
    )


@_guarded(lambda: 0)
def expire_coverage(repo: str, pr: int, seat: str, files: Sequence[str]) -> int:
    """Expire this seat's coverage for ``files``; see :func:`sidecar.review_state.expire_coverage`.

    Narrower than the primitive on purpose: ``files`` is a sequence and never
    ``None``, so the wholesale case is not reachable through this seam (see
    :class:`ExpireCoverageRequest`). A bare ``str`` is still rejected rather than
    iterated into characters, here as there -- the local path delegates to the
    guard that owns that rule, and the HTTP path cannot serialize one as a list.
    """
    if not _remote():
        return review_state.expire_coverage(repo, pr, seat, files)
    if isinstance(files, str):
        raise TypeError(f"files must be a sequence of strings, not a single str: {files!r}")
    return LedgerCountResponse.model_validate(
        _post(
            "/rs/coverage/expire",
            ExpireCoverageRequest(repo=repo, pr=pr, seat=seat, files=list(files)),
        )
    ).count


@_guarded(lambda: 0)
def record_findings(
    repo: str,
    pr: int,
    seat: str,
    round: int,
    head_sha: str,
    findings: Sequence[AgenticFinding],
) -> int:
    """Record this round's findings; see :func:`sidecar.review_state.record_findings`."""
    if not _remote():
        return review_state.record_findings(repo, pr, seat, round, head_sha, findings)
    if not findings:
        return 0
    return LedgerCountResponse.model_validate(
        _post(
            "/rs/findings",
            RecordFindingsRequest(
                repo=repo,
                pr=pr,
                seat=seat,
                round=round,
                head_sha=head_sha,
                findings=list(findings),
            ),
        )
    ).count


@_guarded(lambda: 0)
def record_coverage(
    repo: str,
    pr: int,
    seat: str,
    round: int,
    head_sha: str,
    regions: Sequence[ExaminedRegion],
) -> int:
    """Record this round's examined regions; see :func:`sidecar.review_state.record_coverage`."""
    if not _remote():
        return review_state.record_coverage(repo, pr, seat, round, head_sha, regions)
    if not regions:
        return 0
    return LedgerCountResponse.model_validate(
        _post(
            "/rs/coverage",
            RecordCoverageRequest(
                repo=repo,
                pr=pr,
                seat=seat,
                round=round,
                head_sha=head_sha,
                regions=list(regions),
            ),
        )
    ).count


@_guarded(lambda: False)
def transition(
    repo: str, pr: int, seat: str, finding_id: str, status: str, reason: str = ""
) -> bool:
    """Close one of this seat's findings; see :func:`sidecar.review_state.transition`."""
    if not _remote():
        return review_state.transition(repo, pr, seat, finding_id, status, reason)
    return LedgerChangedResponse.model_validate(
        _post(
            "/rs/findings/transition",
            TransitionRequest(
                repo=repo, pr=pr, seat=seat, finding_id=finding_id, status=status, reason=reason
            ),
        )
    ).changed


@_guarded(lambda: False)
def reopen(repo: str, pr: int, seat: str, finding_id: str, reason: str) -> bool:
    """Re-raise one of this seat's closed findings; see :func:`sidecar.review_state.reopen`."""
    if not _remote():
        return review_state.reopen(repo, pr, seat, finding_id, reason)
    return LedgerChangedResponse.model_validate(
        _post(
            "/rs/findings/reopen",
            ReopenRequest(repo=repo, pr=pr, seat=seat, finding_id=finding_id, reason=reason),
        )
    ).changed


@_guarded(lambda: 0)
def touch_findings(repo: str, pr: int, seat: str, finding_ids: Sequence[str]) -> int:
    """Re-assert this seat's findings; see :func:`sidecar.review_state.touch_findings`."""
    if not _remote():
        return review_state.touch_findings(repo, pr, seat, finding_ids)
    if isinstance(finding_ids, str):
        raise TypeError(
            f"finding_ids must be a sequence of strings, not a single str: {finding_ids!r}"
        )
    if not finding_ids:
        return 0
    return LedgerCountResponse.model_validate(
        _post(
            "/rs/findings/touch",
            TouchRequest(repo=repo, pr=pr, seat=seat, finding_ids=list(finding_ids)),
        )
    ).count
