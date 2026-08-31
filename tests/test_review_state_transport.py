"""Tests for the review-ledger HTTP seam (#171): client, endpoints, round trip.

The HTTP cases run the CLIENT against the REAL endpoints through a FastAPI
``TestClient``, rather than asserting on a mocked request. The thing #171 is
about is a wire, and a wire's failure mode is that the two ends stop agreeing --
which a test that stubs one end cannot see. So ``httpx.get``/``httpx.post`` are
redirected into the app and only ``sidecar.review_state`` (the store both ends
sit behind) is faked.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from sidecar import main, review_state
from sidecar import review_state_client as rsc
from sidecar.reviewer.prompt import AgenticFinding, ExaminedRegion, PriorCoverage, PriorFinding

_TOKEN = "test-token"
_URL = "http://sidecar.lan:8000"
_UUID = "0f9d1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b"
REPO, PR, SEAT = "o/r", 9, "henry"


def _finding(**kw) -> AgenticFinding:
    base = dict(
        file="src/app.py",
        line=42,
        severity="high",
        category="bug",
        title="unchecked None device",
        body="open_source() may return None",
        evidence="src/app.py:118-166",
    )
    base.update(kw)
    return AgenticFinding(**base)


def _region(**kw) -> ExaminedRegion:
    base = dict(
        file="src/util.py",
        region="helper",
        checked="whether every caller handles a None device",
        conclusion="all three call sites guard it",
        evidence="src/util.py:10-40",
    )
    base.update(kw)
    return ExaminedRegion(**base)


@pytest.fixture(autouse=True)
def _fresh_latch(monkeypatch):
    """A latched-offline process would silently no-op every later test."""
    monkeypatch.setattr(rsc, "_transport_down", False)
    monkeypatch.delenv("FUKO_URL", raising=False)
    monkeypatch.delenv("FUKO_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_database(monkeypatch):
    """Unset the connection string for every test here, the other half of #171's
    ambient-environment hazard.

    ``conftest.py``'s ``_no_ambient_sidecar`` keeps a real ``FUKO_URL`` out of the
    suite; this keeps a real ``FUKO_DATABASE_URL`` out of THIS one. Both ends of
    the seam pick their target at call time, so whichever one is configured in the
    ambient environment becomes part of the fixture -- and the local branch is the
    end with no fake in front of it. Concretely, before this fixture existed
    ``test_the_first_transport_failure_latches_the_run_offline`` latched and then
    ran its remaining nine calls against whatever store was configured: on CI that
    inserted a real ``o/r#9`` finding row into Postgres and returned ``1`` where
    the test asserts the neutral ``0``, and on a developer box it would write that
    junk into a live ledger.

    The tests that fake ``sidecar.review_state`` (``local``, ``store``) are
    unaffected -- they replace the module's functions, which never read
    ``settings`` -- so this only ever decides what the UNFAKED local branch does,
    which for every test here is "nothing". A test whose subject IS the empty
    connection string still says so itself.
    """
    monkeypatch.setattr(review_state.settings, "database_url", "")


@pytest.fixture
def local(monkeypatch):
    """Record what the ledger asked of the LOCAL store, with no sidecar configured."""
    calls: list[tuple] = []

    def _spy(name, result):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        monkeypatch.setattr(review_state, name, _fn)

    _spy("open_findings", review_state.OpenLedger(rows=(), truncated=3))
    _spy("next_round", 7)
    _spy("settled_findings", ())
    _spy("live_coverage", [])
    _spy("expire_coverage", 2)
    _spy("record_findings", 1)
    _spy("record_coverage", 1)
    _spy("transition", True)
    _spy("reopen", True)
    _spy("touch_findings", 4)
    return calls


@pytest.fixture
def wire(monkeypatch):
    """Point the client at the real app, and hand the test the store both ends use."""
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    monkeypatch.setenv("FUKO_URL", _URL)
    monkeypatch.setenv("FUKO_TOKEN", _TOKEN)
    client = TestClient(main.app)
    seen: dict = {"timeouts": []}

    def _route(method):
        def _call(url, *, headers=None, timeout=None, **kw):
            seen["timeouts"].append(timeout)
            seen["headers"] = headers
            assert url.startswith(_URL)
            return getattr(client, method)(url[len(_URL) :], headers=headers, **kw)

        return _call

    monkeypatch.setattr(rsc.httpx, "get", _route("get"))
    monkeypatch.setattr(rsc.httpx, "post", _route("post"))
    return seen


@pytest.fixture
def store(monkeypatch):
    """Fake the primitives the ENDPOINTS call, recording the arguments they got."""
    calls: list[tuple] = []

    def _spy(name, result):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        monkeypatch.setattr(review_state, name, _fn)

    return _spy, calls


# --- The local branch: unchanged behaviour when no sidecar is configured. ---


def test_without_fuko_url_every_call_goes_to_the_local_store(local):
    """The pre-#171 path, byte for byte: a runner with a connection string and no
    sidecar keeps talking to Postgres directly."""
    assert rsc.open_findings(REPO, PR, SEAT).truncated == 3
    assert rsc.next_round(REPO, PR, SEAT) == 7
    assert rsc.settled_findings(REPO, PR, SEAT) == ()
    assert rsc.live_coverage(REPO, PR, SEAT) == []
    assert rsc.expire_coverage(REPO, PR, SEAT, ["src/app.py"]) == 2
    assert rsc.record_findings(REPO, PR, SEAT, 3, "deadbee", [_finding()]) == 1
    assert rsc.record_coverage(REPO, PR, SEAT, 3, "deadbee", [_region()]) == 1
    assert rsc.transition(REPO, PR, SEAT, _UUID, "fixed", "rewritten") is True
    assert rsc.reopen(REPO, PR, SEAT, _UUID, "re-found") is True
    assert rsc.touch_findings(REPO, PR, SEAT, [_UUID]) == 4

    assert [name for name, _, _ in local] == [
        "open_findings",
        "next_round",
        "settled_findings",
        "live_coverage",
        "expire_coverage",
        "record_findings",
        "record_coverage",
        "transition",
        "reopen",
        "touch_findings",
    ]
    # The lane leads every call on both transports -- one signature, one semantics.
    assert all(args[:3] == (REPO, PR, SEAT) for _, args, _ in local)


def test_a_local_store_that_raises_still_cannot_reach_the_review(monkeypatch, capsys):
    """The seam guards its local branch too, so a stubbed or future primitive
    cannot reach a caller through an unguarded path."""

    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(review_state, "open_findings", _boom)

    assert rsc.open_findings(REPO, PR, SEAT) == review_state.OpenLedger()
    err = capsys.readouterr().err
    assert "open_findings failed (local store)" in err and "connection refused" in err


# --- The HTTP branch: a real round trip through the real endpoints. ---


def test_open_findings_round_trips_the_rows_and_the_cut(wire, store):
    """#173's ``truncated`` has to survive the wire: it is the one ledger fact a
    later reader cannot recover from the table."""
    _spy, calls = store
    row = review_state.StoredFinding(
        id=_UUID,
        prior=PriorFinding(
            file="src/app.py",
            title="unchecked None",
            body="may return None",
            line=42,
            severity="high",
            category="bug",
            round=2,
            evidence="src/app.py:118-166",
        ),
    )
    _spy("open_findings", review_state.OpenLedger(rows=(row,), truncated=5))

    ledger = rsc.open_findings(REPO, PR, SEAT)

    assert ledger == review_state.OpenLedger(rows=(row,), truncated=5)
    assert calls[0][1] == (REPO, PR, SEAT)


def test_the_reads_round_trip_their_projections(wire, store):
    _spy, calls = store
    settled = review_state.SettledFinding(
        id=_UUID,
        file="src/app.py",
        title="leaks the handle",
        status="fixed",
        round=2,
        reason="gone",
    )
    coverage = PriorCoverage(
        file="src/util.py",
        checked="every caller",
        conclusion="all guarded",
        evidence="src/util.py:10-40",
        region="helper",
        round=3,
    )
    _spy("settled_findings", (settled,))
    _spy("live_coverage", [coverage])
    _spy("next_round", 12)

    assert rsc.settled_findings(REPO, PR, SEAT) == (settled,)
    assert rsc.live_coverage(REPO, PR, SEAT) == [coverage]
    assert rsc.next_round(REPO, PR, SEAT) == 12
    assert [c[1] for c in calls] == [(REPO, PR, SEAT)] * 3


def test_the_writes_round_trip_their_payloads_and_their_lane(wire, store):
    """Every id-addressed write names ``(repo, pr, seat)`` on the wire, and the
    store matches it in SQL -- one seat may not settle another's row (#160)."""
    _spy, calls = store
    _spy("record_findings", 1)
    _spy("record_coverage", 1)
    _spy("transition", True)
    _spy("reopen", True)
    _spy("touch_findings", 2)
    _spy("expire_coverage", 3)

    finding = _finding()
    region = _region()
    assert rsc.record_findings(REPO, PR, SEAT, 4, "deadbee", [finding]) == 1
    assert rsc.record_coverage(REPO, PR, SEAT, 4, "deadbee", [region]) == 1
    assert rsc.transition(REPO, PR, SEAT, _UUID, "fixed", "rewritten") is True
    assert rsc.reopen(REPO, PR, SEAT, _UUID, "re-found") is True
    assert rsc.touch_findings(REPO, PR, SEAT, [_UUID, "row-2"]) == 2
    assert rsc.expire_coverage(REPO, PR, SEAT, ["src/app.py"]) == 3

    assert [name for name, _, _ in calls] == [
        "record_findings",
        "record_coverage",
        "transition",
        "reopen",
        "touch_findings",
        "expire_coverage",
    ]
    assert all(args[:3] == (REPO, PR, SEAT) for _, args, _ in calls)
    # The reviewer's own vocabulary types survive serialization unchanged, so the
    # row the sidecar writes is the row the round produced.
    assert calls[0][1][3:] == (4, "deadbee", [finding])
    assert calls[1][1][3:] == (4, "deadbee", [region])
    assert calls[2][1][3:] == (_UUID, "fixed", "rewritten")
    assert calls[3][1][3:] == (_UUID, "re-found")
    assert calls[4][1][3:] == ([_UUID, "row-2"],)
    assert calls[5][1][3:] == (["src/app.py"],)


def test_an_empty_write_never_leaves_the_runner(wire, store):
    """`record_findings([])` and `record_coverage([])` are already no-ops in the
    store; spending a round-trip to be told so is pure latency on the review's
    critical path."""
    _spy, calls = store
    _spy("record_findings", 99)
    _spy("record_coverage", 99)
    _spy("touch_findings", 99)
    _spy("expire_coverage", 99)

    assert rsc.record_findings(REPO, PR, SEAT, 4, "sha", []) == 0
    assert rsc.record_coverage(REPO, PR, SEAT, 4, "sha", []) == 0
    assert rsc.touch_findings(REPO, PR, SEAT, []) == 0
    # `expire_coverage` most of all: it is `carry_in`'s first store call, so its
    # round-trip is paid before the prompt is even built.
    assert rsc.expire_coverage(REPO, PR, SEAT, []) == 0
    assert calls == []


def test_the_requests_carry_the_bearer_token_and_the_short_timeout(wire, store):
    """Same auth treatment `_auth` gives every other runner seam, and half their
    timeout: `carry_in` blocks prompt construction, a metrics POST blocks nothing."""
    _spy, _ = store
    _spy("next_round", 1)
    _spy("touch_findings", 0)

    rsc.next_round(REPO, PR, SEAT)
    rsc.touch_findings(REPO, PR, SEAT, [_UUID])

    assert wire["headers"]["Authorization"] == f"Bearer {_TOKEN}"
    assert wire["timeouts"] == [rsc.TIMEOUT_S, rsc.TIMEOUT_S]
    assert rsc.TIMEOUT_S <= 5.0


# --- Degradation: the same neutral values, whatever the wire does. ---


def _neutral_calls():
    return [
        (rsc.open_findings, (REPO, PR, SEAT), review_state.OpenLedger()),
        (rsc.next_round, (REPO, PR, SEAT), 1),
        (rsc.settled_findings, (REPO, PR, SEAT), ()),
        (rsc.live_coverage, (REPO, PR, SEAT), []),
        (rsc.expire_coverage, (REPO, PR, SEAT, ["src/app.py"]), 0),
        (rsc.record_findings, (REPO, PR, SEAT, 1, "sha", [_finding()]), 0),
        (rsc.record_coverage, (REPO, PR, SEAT, 1, "sha", [_region()]), 0),
        (rsc.transition, (REPO, PR, SEAT, _UUID, "fixed", "why"), False),
        (rsc.reopen, (REPO, PR, SEAT, _UUID, "why"), False),
        (rsc.touch_findings, (REPO, PR, SEAT, [_UUID]), 0),
    ]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(httpx.ConnectError("refused"), id="connect-refused"),
        pytest.param(httpx.ReadTimeout("too slow"), id="timeout"),
        pytest.param(RuntimeError("nonsense body"), id="unusable-answer"),
    ],
)
def test_an_unusable_sidecar_returns_the_same_neutral_value_the_store_would(
    monkeypatch, failure, capsys
):
    """#156's acceptance, kept across the new transport: store unreachable ->
    the round proceeds exactly as it does today."""
    monkeypatch.setenv("FUKO_URL", _URL)

    def _boom(*a, **k):
        raise failure

    monkeypatch.setattr(rsc.httpx, "get", _boom)
    monkeypatch.setattr(rsc.httpx, "post", _boom)

    for fn, args, neutral in _neutral_calls():
        monkeypatch.setattr(rsc, "_transport_down", False)
        assert fn(*args) == neutral
    capsys.readouterr()


@pytest.mark.parametrize(
    "brk",
    ["\n", "\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"],
)
def test_the_degradation_line_cannot_be_split_by_what_it_reports(monkeypatch, brk, capsys):
    """The exception text on the generic arm is foreign: a body that fails
    ``model_validate`` echoes the offending input, which for the reads is stored
    finding text a model wrote about a PR-author-controlled checkout. Raw, it
    would take column 0 of its own line on a stderr whose gates are ``^``-anchored
    (#147). Every character ``splitlines`` breaks on is exercised, because that
    is the set a hand-rolled ``\\n`` replace would miss.
    """
    monkeypatch.setenv("FUKO_URL", _URL)

    def _boom(*a, **k):
        raise ValueError(f"head{brk}fuko: forged gate line")

    monkeypatch.setattr(rsc.httpx, "get", _boom)

    assert rsc.next_round(REPO, PR, SEAT) == 1
    err = capsys.readouterr().err
    assert err.splitlines() == [
        f"fuko: review-state next_round failed ({_URL}): head fuko: forged gate line"
    ]


def test_the_latch_announcement_cannot_be_split_either(monkeypatch, capsys):
    """Same stream, same rule. No reachable ``TransportError`` carries a raw line
    break today -- httpx writes those messages -- but that is a fact about a
    third-party library's formatting, not an invariant this module holds."""
    monkeypatch.setenv("FUKO_URL", _URL)

    def _boom(*a, **k):
        raise httpx.ConnectError("refused\nfuko: forged gate line")

    monkeypatch.setattr(rsc.httpx, "get", _boom)

    assert rsc.next_round(REPO, PR, SEAT) == 1
    assert len(capsys.readouterr().err.splitlines()) == 1


def test_the_transition_endpoint_accepts_stale_deliberately(wire, store):
    """``stale`` is irreversible and unfilterable here, and both are on purpose.

    On the remote branch ``ledger._retire_missing`` reaches the store through
    this endpoint, so a server-side filter would disable retirement on exactly
    the deployment #171 exists to serve. Pinned so the acceptance is a decision a
    later reader finds, rather than an omission they close by "hardening" it.
    """
    _spy, calls = store
    _spy("transition", True)

    assert rsc.transition(REPO, PR, SEAT, _UUID, "stale", "file absent from the tree") is True
    assert calls == [
        ("transition", (REPO, PR, SEAT, _UUID, "stale", "file absent from the tree"), {})
    ]


def test_an_erroring_endpoint_degrades_without_latching(monkeypatch, store, wire, capsys):
    """A 500 answers as fast as a 200, so latching on it would turn one broken
    handler into a lost ledger for every remaining call. The latch bounds TIME."""
    _spy, calls = store

    def _boom(*a, **k):
        raise ValueError("row is not what you think")

    monkeypatch.setattr(review_state, "next_round", _boom)
    _spy("touch_findings", 5)

    assert rsc.next_round(REPO, PR, SEAT) == 1
    assert rsc._transport_down is False
    # Not latched, so the next call still goes over the wire and still works.
    assert rsc.touch_findings(REPO, PR, SEAT, [_UUID]) == 5
    assert "next_round failed" in capsys.readouterr().err


def test_a_status_error_degrades_without_latching(monkeypatch, capsys):
    """The same rule reached through a real response rather than a raised handler."""
    monkeypatch.setenv("FUKO_URL", _URL)
    request = httpx.Request("GET", _URL + "/rs/round")

    monkeypatch.setattr(
        rsc.httpx, "get", lambda *a, **k: httpx.Response(500, text="boom", request=request)
    )

    assert rsc.next_round(REPO, PR, SEAT) == 1
    assert rsc._transport_down is False
    assert "next_round failed" in capsys.readouterr().err


def test_the_first_transport_failure_latches_the_run_offline(monkeypatch, capsys):
    """#170's shape must not reappear over HTTP: ten primitives behind a 5s
    timeout would cost 50s a round on a black-holed sidecar."""
    monkeypatch.setenv("FUKO_URL", _URL)
    attempts = []

    def _boom(*a, **k):
        attempts.append(1)
        raise httpx.ConnectTimeout("black hole")

    monkeypatch.setattr(rsc.httpx, "get", _boom)
    monkeypatch.setattr(rsc.httpx, "post", _boom)

    for fn, args, neutral in _neutral_calls():
        assert fn(*args) == neutral

    assert len(attempts) == 1
    err = capsys.readouterr().err
    # One line, not ten: the latch is announced once.
    assert err.count("did not answer") == 1
    assert _URL in err


def test_a_latched_run_falls_back_to_the_same_state_by_the_other_route(monkeypatch, local):
    """The fall-back is to the SAME Postgres, so a round split across the two
    branches cannot end up with two disagreeing halves."""
    monkeypatch.setenv("FUKO_URL", _URL)
    monkeypatch.setattr(rsc.httpx, "get", lambda *a, **k: pytest.fail("latched: no request"))
    monkeypatch.setattr(rsc.httpx, "post", lambda *a, **k: pytest.fail("latched: no request"))
    monkeypatch.setattr(rsc, "_transport_down", True)

    assert rsc.next_round(REPO, PR, SEAT) == 7
    assert rsc.record_findings(REPO, PR, SEAT, 1, "sha", [_finding()]) == 1
    assert [name for name, _, _ in local] == ["next_round", "record_findings"]


def test_on_a_runner_the_fall_back_is_the_no_op_156_already_permits(monkeypatch, capsys):
    """The deployment #171 is about: `FUKO_URL` only, no connection string. Once
    the sidecar is unreachable there is no store at all, and the round proceeds
    exactly as a pre-ledger round did."""
    monkeypatch.setenv("FUKO_URL", _URL)
    monkeypatch.setattr(review_state.settings, "database_url", "")
    monkeypatch.setattr(rsc, "_transport_down", True)

    for fn, args, neutral in _neutral_calls():
        assert fn(*args) == neutral
    assert capsys.readouterr().err == ""


# --- The wire cannot widen what the primitives allow. ---


def test_a_bare_string_is_refused_on_the_wire_as_it_is_in_the_store(monkeypatch, capsys):
    """``str`` satisfies ``Sequence[str]``: serialized as a list it would expire
    coverage for single CHARACTERS, and be reported as a plain 0."""
    monkeypatch.setenv("FUKO_URL", _URL)
    monkeypatch.setattr(rsc.httpx, "post", lambda *a, **k: pytest.fail("must not be sent"))

    assert rsc.expire_coverage(REPO, PR, SEAT, "src/app.py") == 0
    assert rsc.touch_findings(REPO, PR, SEAT, _UUID) == 0

    err = capsys.readouterr().err
    assert "files must be a sequence" in err and "finding_ids must be a sequence" in err


def test_wholesale_expiry_is_refused_on_BOTH_branches_not_just_the_wire(monkeypatch, local, capsys):
    """The guard has to sit above the branch point, because the two branches
    disagree about what ``None`` MEANS.

    Over HTTP it is unserializable and degrades to 0; in the store it is
    "expire this seat's coverage wholesale". A guard placed after the branch
    would therefore make the same call a logged no-op on a runner and a silent
    discard of the whole ledger on the sidecar host -- the two-transports-two-
    semantics outcome this module exists to prevent, on its most destructive
    call. Asserted with NO sidecar configured, which is the branch that used to
    let it through.
    """
    assert rsc.expire_coverage(REPO, PR, SEAT, None) == 0
    assert rsc.expire_coverage(REPO, PR, SEAT, "src/app.py") == 0
    # Never delegated: the store never saw a call it would have read as wholesale.
    assert local == []
    # And the legitimate call on the same branch still reaches it.
    assert rsc.expire_coverage(REPO, PR, SEAT, ["src/app.py"]) == 2
    assert [name for name, _, _ in local] == ["expire_coverage"]
    assert capsys.readouterr().err.count("files must be a sequence") == 2


def test_an_expire_request_without_files_is_rejected_not_read_as_wholesale(wire, store):
    """``expire_coverage(files=None)`` discards a seat's whole coverage ledger.
    An omitted key, a typo'd key or a truncated body must not reach it."""
    _spy, calls = store
    _spy("expire_coverage", 99)
    client = TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})

    resp = client.post("/rs/coverage/expire", json={"repo": REPO, "pr": PR, "seat": SEAT})

    assert resp.status_code == 422
    assert calls == []
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        rsc.ExpireCoverageRequest(repo=REPO, pr=PR, seat=SEAT)


def test_every_ledger_route_is_behind_the_bearer_dependency(monkeypatch):
    """Enumerated off the app rather than listed, so an endpoint added later
    without ``Depends(_auth)`` fails this test instead of shipping open. These
    routes accept ledger WRITES; an unauthenticated one is a write path into the
    reviewer's own next prompt."""
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    anon = TestClient(main.app)
    routes = [r for r in main.app.routes if getattr(r, "path", "").startswith("/rs/")]

    assert len(routes) == 10
    for route in routes:
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            resp = anon.request(method, route.path, json={})
            assert resp.status_code == 401, (method, route.path, resp.status_code)

    monkeypatch.setattr(main.settings, "auth_token", None)
    assert TestClient(main.app).get("/rs/round", params={"repo": REPO}).status_code == 503


def test_the_transition_endpoint_cannot_invent_a_verdict_the_store_refuses(monkeypatch):
    """The vocabulary guard travels with the primitive, so it holds for the wire
    too: a request naming an unrecognised status closes nothing.

    The store is ENABLED here, which is the whole difficulty. With an empty
    ``database_url`` -- the module default under this file's autouse fixture --
    ``_best_effort`` short-circuits before the wrapped body ever runs, so
    ``{"changed": false}`` would be produced by the disabled store rather than by
    the vocabulary check and deleting that check would leave this green. So the
    connection string is set and :func:`sidecar.db.db` is replaced by something
    that fails the test if it is reached: ``transition`` returns ``False`` at the
    ``status not in FINDING_STATUSES`` line, BEFORE it imports ``db``, and that
    ordering is exactly what the assertion is now about. Raised by
    ``qwen-anthropic/qwen3.8-max`` on #171.
    """
    import sidecar.db

    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    monkeypatch.setattr(review_state.settings, "database_url", "postgresql://unused/never")
    monkeypatch.setattr(
        sidecar.db, "db", lambda *a, **k: pytest.fail("refused before any connection")
    )
    client = TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})

    resp = client.post(
        "/rs/findings/transition",
        json={
            "repo": REPO,
            "pr": PR,
            "seat": SEAT,
            "finding_id": _UUID,
            "status": "resolved",
            "reason": "trust me",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"changed": False}


def test_the_latch_is_announced_once_even_if_two_branches_trip_it(capsys):
    """An A/B run's branches are threads of one process, so both can fail before
    either has latched. The announcement is the loud one -- it says the run lost
    its ledger -- and repeating it per branch would misreport how often."""
    rsc._mark_down(_URL, httpx.ConnectError("refused"))
    rsc._mark_down(_URL, httpx.ConnectError("refused again"))

    assert rsc._transport_down is True
    assert capsys.readouterr().err.count("did not answer") == 1
