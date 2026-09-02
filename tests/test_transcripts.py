"""Tests for the transcript reader: query shapes, endpoints, and the blob fetch (#240).

Like ``test_review_state_reads.py`` and unlike the write path in
``test_run_metrics.py``, these reads must RAISE when the store is unreachable --
the whole sub-issue turns on an operator being able to tell an outage from an
empty corpus -- so the degradation tests here assert the opposite of a
best-effort suite's.
"""

import contextlib
import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import sidecar.db
from sidecar import main, transcripts
from sidecar.objectstore import STORE_HEADER, STORE_UNCONFIGURED, FileBlobStore
from sidecar.reviewer.transcript import (
    FileTranscriptSink,
    Scrubber,
    ShippingTranscriptSink,
    Transcript,
    _Meter,
)

_TOKEN = "test-token"
_CREATED = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
_STARTED = datetime(2026, 9, 1, 11, 58, tzinfo=UTC)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self.rows = list(rows)
        self.statements: list[tuple[str, tuple]] = []
        self.opened_with: list[dict] = []

    def execute(self, sql, params=()):
        self.statements.append((" ".join(sql.split()), tuple(params)))
        return _FakeCursor(self.rows)


@pytest.fixture
def pg(monkeypatch):
    """Patch ``sidecar.db.db`` -- the RAISING connection helper these reads use."""

    def install(rows=()):
        conn = _FakeConn(rows)

        @contextlib.contextmanager
        def fake_db(*_a, **kwargs):
            conn.opened_with.append(kwargs)
            yield conn

        monkeypatch.setattr(sidecar.db, "db", fake_db)
        return conn

    return install


@pytest.fixture
def dead_pg(monkeypatch):
    """Patch ``sidecar.db.db`` with a helper that fails the way an outage does."""

    @contextlib.contextmanager
    def fake_db(*_a, **_k):
        raise RuntimeError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(sidecar.db, "db", fake_db)


def _row(**overrides):
    row = {
        "key": "20260901T120000Z-a1b2c3d4e5f6",
        "created_at": _CREATED,
        "complete": True,
        "tool_calls": {"Read": 182, "Grep": 9},
        "tool_result_bytes": 41_000_000,
        "repeated_read_files": 37,
        "repo": "lemehmet/mepro",
        "pr": 42,
        "slot": "dorian",
        "provider": "openrouter",
        "model": "qwen3.8-max",
        "backend": "agentic",
        "outcome": "ok",
        "started_at": _STARTED,
        "duration_s": 612.5,
        "total": 1,
    }
    row.update(overrides)
    return tuple(row.values())


def _client(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    return TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})


# --- the query shape #241 shares.


def test_list_maps_every_column_onto_the_row(pg):
    pg([_row()])
    page = transcripts.list_transcripts()
    assert page.total == 1
    run = page.rows[0]
    assert run.key == "20260901T120000Z-a1b2c3d4e5f6"
    assert run.complete is True
    assert run.tool_calls == {"Read": 182, "Grep": 9}
    assert run.tool_calls_total == 191
    assert run.tool_result_bytes == 41_000_000
    assert run.repeated_read_files == 37
    assert (run.repo, run.pr, run.seat) == ("lemehmet/mepro", 42, "dorian")
    assert (run.provider, run.model, run.backend) == ("openrouter", "qwen3.8-max", "agentic")
    assert run.created_at == _CREATED.isoformat()
    assert run.started_at == _STARTED.isoformat()
    assert run.duration_s == 612.5


def test_a_transcript_with_no_run_row_still_lists(pg):
    """The reference lands in a later transaction, so it may never have followed."""
    pg(
        [
            _row(
                repo=None,
                pr=None,
                slot=None,
                provider=None,
                model=None,
                backend=None,
                outcome=None,
                started_at=None,
                duration_s=None,
            )
        ]
    )
    run = transcripts.list_transcripts().rows[0]
    assert run.repo is None and run.pr is None and run.seat is None
    assert run.tool_result_bytes == 41_000_000


def test_incomplete_is_carried_through_as_false(pg):
    pg([_row(complete=False)])
    assert transcripts.list_transcripts().rows[0].complete is False


def test_empty_result_is_an_empty_page_with_no_total(pg):
    pg([])
    page = transcripts.list_transcripts()
    assert page.rows == () and page.total == 0


def test_filters_are_all_anded_and_bound_as_parameters(pg):
    conn = pg([])
    transcripts.list_transcripts(
        repo="lemehmet/mepro", pr=42, seat="dorian", since="2026-08-30", until="2026-09-01"
    )
    sql, params = conn.statements[0]
    # Each filter is a `%s IS NULL OR col = %s` pair, so combining them narrows.
    assert sql.count("IS NULL OR") == 5
    assert params[:6] == ("lemehmet/mepro", "lemehmet/mepro", 42, 42, "dorian", "dorian")
    assert params[6] == params[7] == datetime(2026, 8, 30, tzinfo=UTC)
    assert params[8] == params[9] == datetime(2026, 9, 1, tzinfo=UTC)


def test_absent_filters_bind_null(pg):
    conn = pg([])
    transcripts.list_transcripts()
    _, params = conn.statements[0]
    assert params[:10] == (None,) * 10


def test_limit_is_clamped_and_offset_floored(pg):
    conn = pg([])
    transcripts.list_transcripts(limit=10_000, offset=-5)
    assert conn.statements[0][1][-2:] == (transcripts.MAX_ROWS, 0)
    transcripts.list_transcripts(limit=0)
    assert conn.statements[1][1][-2] == 1


def test_the_join_is_lateral_so_a_redelivered_metrics_post_lists_once(pg):
    """`record()` has no ON CONFLICT on review_runs, so one key can have two rows."""
    conn = pg([])
    transcripts.list_transcripts()
    sql = conn.statements[0][0]
    assert "LEFT JOIN LATERAL" in sql and "ORDER BY started_at DESC LIMIT 1" in sql


def test_ordering_breaks_ties_on_the_key(pg):
    conn = pg([])
    transcripts.list_transcripts()
    assert "ORDER BY t.created_at DESC, t.key DESC" in conn.statements[0][0]


def test_the_read_is_bounded_and_skips_the_embedding_space(pg):
    conn = pg([])
    transcripts.list_transcripts()
    assert conn.opened_with[0] == {
        "timeout": transcripts.READ_TIMEOUT_S,
        "embed_space": False,
    }


def test_the_read_raises_rather_than_returning_an_empty_page(dead_pg):
    with pytest.raises(RuntimeError):
        transcripts.list_transcripts()


def test_a_naive_since_is_read_as_utc(pg):
    """TIMESTAMPTZ + a naive bind would resolve through the server's session TZ."""
    conn = pg([])
    transcripts.list_transcripts(since=datetime(2026, 8, 30, 6, 0))
    assert conn.statements[0][1][6] == datetime(2026, 8, 30, 6, 0, tzinfo=UTC)


def test_an_aware_since_keeps_its_own_offset(pg):
    conn = pg([])
    transcripts.list_transcripts(since="2026-08-30T06:00:00+02:00")
    assert conn.statements[0][1][6].utcoffset().total_seconds() == 7200


def test_a_malformed_date_is_a_value_error(pg):
    pg([])
    with pytest.raises(ValueError):
        transcripts.list_transcripts(since="last tuesday")


def test_a_blank_date_is_no_filter(pg):
    conn = pg([])
    transcripts.list_transcripts(since="", until=None)
    assert conn.statements[0][1][6] is None


def test_tool_calls_of_an_unexpected_shape_cost_only_that_figure(pg):
    pg([_row(tool_calls=["Read"])])
    assert transcripts.list_transcripts().rows[0].tool_calls == {}


def test_as_dict_carries_every_declared_field(pg):
    pg([_row()])
    payload = transcripts.list_transcripts().rows[0].as_dict()
    assert set(payload) == set(main.models.TranscriptRunRow.model_fields)


# --- the blob fetch.


def _file_store(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "transcript_store_backend", "file")
    monkeypatch.setattr(main.settings, "transcript_store_root", str(tmp_path / "blobs"))
    return FileBlobStore(str(tmp_path / "blobs"))


def test_fetch_round_trips_the_stored_bytes(monkeypatch, tmp_path):
    store = _file_store(monkeypatch, tmp_path)
    body = b'{"type":"assistant"}\n{"type":"result"}\n'
    store.put("abc123", body)
    assert transcripts.fetch("abc123") == body


def test_fetch_of_an_absent_key_is_none_not_an_error(monkeypatch, tmp_path):
    _file_store(monkeypatch, tmp_path)
    assert transcripts.fetch("nothinghere") is None


def test_fetch_without_a_store_raises_rather_than_returning_none(monkeypatch):
    monkeypatch.setattr(main.settings, "transcript_store_backend", "")
    with pytest.raises(transcripts.StoreUnconfigured):
        transcripts.fetch("abc123")


# --- the endpoints.


def test_list_endpoint_returns_rows_and_the_window_total(monkeypatch, pg):
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    pg([_row(total=17)])
    resp = _client(monkeypatch).get("/transcripts", params={"repo": "lemehmet/mepro"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 17
    assert body["transcripts"][0]["tool_calls"] == {"Read": 182, "Grep": 9}
    assert body["transcripts"][0]["seat"] == "dorian"


def test_list_endpoint_passes_every_filter_through(monkeypatch, pg):
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    conn = pg([])
    _client(monkeypatch).get(
        "/transcripts",
        params={
            "repo": "o/r",
            "pr": 7,
            "seat": "gray",
            "since": "2026-08-01",
            "until": "2026-09-01",
            "limit": 5,
            "offset": 10,
        },
    )
    _, params = conn.statements[0]
    assert params[0] == "o/r" and params[2] == 7 and params[4] == "gray"
    assert params[-2:] == (5, 10)


def test_list_endpoint_treats_an_empty_filter_box_as_no_filter(monkeypatch, pg):
    # An HTML form submits an unfilled box as "", and the date filters already
    # read that as "not filtering". repo/seat must agree, or the same blank form
    # narrows the listing to nothing on one half of its fields and not the other.
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    conn = pg([])
    _client(monkeypatch).get(
        "/transcripts", params={"repo": "", "seat": "", "since": "", "until": "", "pr": ""}
    )
    _, params = conn.statements[0]
    assert params[0] is None and params[2] is None and params[4] is None


def test_list_endpoint_rejects_a_non_numeric_pr_with_400(monkeypatch, pg):
    # Named here rather than left to FastAPI's 422, matching what a malformed
    # date gets: one taxonomy for every filter this endpoint parses itself.
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    pg([])
    resp = _client(monkeypatch).get("/transcripts", params={"pr": "twelve"})
    assert resp.status_code == 400
    assert "twelve" in resp.json()["detail"]


def test_list_endpoint_without_a_database_is_503_not_an_empty_list(monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", "")
    resp = _client(monkeypatch).get("/transcripts")
    assert resp.status_code == 503
    assert "FUKO_DATABASE_URL" in resp.json()["detail"]


def test_list_endpoint_with_an_unreachable_store_is_503(monkeypatch, dead_pg, capsys):
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    resp = _client(monkeypatch).get("/transcripts")
    assert resp.status_code == 503
    assert "not an empty corpus" in resp.json()["detail"]
    # The status says nothing durable; the underlying fault has to reach stderr.
    assert "connection refused" in capsys.readouterr().err


def test_list_endpoint_rejects_a_malformed_date_with_400(monkeypatch, pg):
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    pg([])
    resp = _client(monkeypatch).get("/transcripts", params={"since": "yesterday"})
    assert resp.status_code == 400


def test_get_endpoint_returns_the_bytes_verbatim(monkeypatch, tmp_path):
    store = _file_store(monkeypatch, tmp_path)
    body = b'{"type":"assistant","message":{"content":[]}}\n{"type":"result"}\n'
    store.put("20260901T120000Z-a1b2c3d4e5f6", body)
    resp = _client(monkeypatch).get("/transcripts/20260901T120000Z-a1b2c3d4e5f6")
    assert resp.status_code == 200
    assert resp.content == body
    assert resp.headers["content-type"].startswith("application/x-ndjson")


def test_get_endpoint_404s_a_key_the_store_does_not_hold(monkeypatch, tmp_path):
    _file_store(monkeypatch, tmp_path)
    resp = _client(monkeypatch).get("/transcripts/neverstored")
    assert resp.status_code == 404
    assert "neverstored" in resp.json()["detail"]


def test_get_endpoint_400s_a_malformed_key(monkeypatch, tmp_path):
    _file_store(monkeypatch, tmp_path)
    # Starlette percent-decodes before routing, so a traversal never reaches the
    # validator at all -- the path simply matches no route. The allowlist is
    # what catches everything that DOES route here, e.g. a dotfile.
    assert _client(monkeypatch).get("/transcripts/..%2Fetc%2Fpasswd").status_code == 404
    assert _client(monkeypatch).get("/transcripts/.hidden").status_code == 400


def test_get_endpoint_marks_the_unconfigured_store_the_way_the_upload_does(monkeypatch):
    monkeypatch.setattr(main.settings, "transcript_store_backend", "")
    resp = _client(monkeypatch).get("/transcripts/abc123")
    assert resp.status_code == 503
    assert resp.headers.get(STORE_HEADER) == STORE_UNCONFIGURED


def test_get_endpoint_503s_a_store_that_is_configured_and_broken(monkeypatch, capsys):
    monkeypatch.setattr(main.settings, "transcript_store_backend", "nonsense")
    resp = _client(monkeypatch).get("/transcripts/abc123")
    assert resp.status_code == 503
    assert resp.headers.get(STORE_HEADER) is None
    assert "unusable" in resp.json()["detail"]
    assert "nonsense" in capsys.readouterr().err


def test_get_endpoint_serves_a_blob_the_index_never_saw(monkeypatch, tmp_path):
    """#258: a throttled failover leg ships a blob and never indexes it.

    Fetch must not consult the index, or a known gap there becomes missing data.
    """
    store = _file_store(monkeypatch, tmp_path)
    monkeypatch.setattr(main.settings, "database_url", "")
    store.put("unindexed01", b"{}\n")
    assert _client(monkeypatch).get("/transcripts/unindexed01").content == b"{}\n"


# --- the epic's own claim, checked.


_KEY = "20260901T120000Z-a1b2c3d4e5f6"


def _capture(tmp_path, events, *, terminal=True):
    """Run ``events`` through a real capture into a real blob store.

    A shipping sink, not a bare file one, because only a sink that AFFIRMS
    storage produces an index row -- which is the row this test is checking the
    stored bytes against.
    """
    store = FileBlobStore(str(tmp_path / "blobs"))
    path = tmp_path / "session.ndjson"
    inner = FileTranscriptSink(path)

    def ship(key, source):
        store.put(key, source.read_bytes())
        return True

    tr = Transcript(_KEY, ShippingTranscriptSink(inner, _KEY, ship), Scrubber.for_secrets([]))
    for event in events:
        tr.write(json.dumps(event) + "\n")
    if terminal:
        tr.write(json.dumps({"type": "result", "subtype": "success"}) + "\n")
    tr.close()
    return tr, store.get(_KEY)


_EVENTS = [
    {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
                {"type": "tool_use", "name": "Grep", "input": {"pattern": "x"}},
            ]
        },
    },
    {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "content": "some file text"}]},
    },
    {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}}]
        },
    },
    {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "content": [{"type": "text", "text": "again"}]}]
        },
    },
]


@pytest.mark.parametrize("terminal", [True, False])
def test_recomputing_from_the_stored_blob_reproduces_the_indexed_figures(tmp_path, terminal):
    """#239's stated premise, and #240's licence to trust the row it reads.

    The meter is fed the SCRUBBED line after the sink accepted it, so a reader
    that folds the stored bytes back through a fresh meter must land on the same
    numbers. Checked for a cut-short feed too, since that is the case where the
    figures describe a prefix rather than a run.
    """
    tr, stored = _capture(tmp_path, _EVENTS, terminal=terminal)
    index = tr.index()
    assert index is not None

    meter = _Meter()
    for line in stored.decode("utf-8").splitlines():
        meter.feed(line + "\n")
    assert not meter.broken
    # The figures the reader would derive from the blob, against the row the
    # capture published for it.
    assert meter.tool_calls == index.tool_calls == {"Read": 2, "Grep": 1}
    assert meter.tool_result_bytes == index.tool_result_bytes
    assert meter.tool_result_bytes == len(b"some file text") + len(b"again")
    assert meter.repeated_read_files == index.repeated_read_files == 1
    assert meter.complete is index.complete is terminal
