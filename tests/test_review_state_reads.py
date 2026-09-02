"""Tests for the operator reads over the review-state ledgers (#235).

Unlike the prompt-path reads in ``test_review_state.py``, these must RAISE when
the store is unreachable -- the page needs to tell an outage from an empty
ledger -- so the degradation tests here assert the opposite of that suite's.
"""

import contextlib
from datetime import UTC, datetime

import pytest

import sidecar.db
from sidecar import review_state

_ACTIVITY = datetime(2026, 7, 22, 20, 15, tzinfo=UTC)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self.rows = list(rows)
        self.statements: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.statements.append((" ".join(sql.split()), tuple(params)))
        return _FakeCursor(self.rows)


@pytest.fixture
def pg(monkeypatch):
    """Patch ``sidecar.db.db`` -- the RAISING connection helper these reads use."""

    def install(rows=()):
        conn = _FakeConn(rows)

        @contextlib.contextmanager
        def fake_db(*_a, **_k):
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


def _lane_row(**overrides):
    row = {
        "repo": "lemehmet/mepro",
        "pr": 1343,
        "seat": "henry",
        "latest_round": 3,
        "last_activity": _ACTIVITY,
        "open": 2,
        "fixed": 1,
        "rejected": 1,
        "stale": 1,
        "reopened": 0,
        "offerable": 2,
        "eligible": 4,
        "carried": 1,
        "settled": 2,
        "coverage_total": 9,
        "coverage_live": 7,
        "lanes_total": 1,
    }
    row.update(overrides)
    return tuple(row.values())


def _finding_row(**overrides):
    row = {
        "id": "0f9d1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b",
        "seat": "henry",
        "round": 2,
        "head_sha": "abc123",
        "file": "src/app.py",
        "line": 42,
        "severity": "high",
        "category": "bug",
        "title": "unchecked None device",
        "body": "open_source() may return None",
        "evidence": "src/app.py:118-166",
        "status": "open",
        "status_reason": "",
        "reopened": 0,
        "created_at": _ACTIVITY,
        "updated_at": _ACTIVITY,
        "total": 1,
    }
    row.update(overrides)
    return tuple(row.values())


def _coverage_row(**overrides):
    row = {
        "id": "1f9d1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b",
        "seat": "henry",
        "round": 1,
        "head_sha": "abc123",
        "file": "src/app.py",
        "region": "open_source",
        "checked": "null return path",
        "conclusion": "guarded",
        "evidence": "src/app.py:118",
        "expired_at": None,
        "created_at": _ACTIVITY,
        "total": 1,
    }
    row.update(overrides)
    return tuple(row.values())


def test_status_order_covers_every_stored_status():
    assert set(review_state.STATUS_ORDER) == review_state.FINDING_STATUSES
    assert review_state.STATUS_ORDER[0] == "open"


def test_lanes_maps_a_row_into_counts_rates_and_totals(pg):
    conn = pg([_lane_row()])
    index = review_state.lanes()
    assert index.total == 1
    lane = index.lanes[0]
    assert (lane.repo, lane.pr, lane.seat, lane.latest_round) == (
        "lemehmet/mepro",
        1343,
        "henry",
        3,
    )
    assert lane.last_activity == _ACTIVITY.isoformat()
    assert lane.counts == {"open": 2, "fixed": 1, "rejected": 1, "stale": 1}
    assert lane.findings_total == 5
    assert (lane.coverage_total, lane.coverage_live) == (9, 7)
    assert lane.carry_forward_rate == 0.25
    assert lane.settle_rate == 0.5
    assert conn.statements[0][1] == (None, None, None, None, None, None, 90, 200, 0)


def test_lanes_reads_both_ledgers_so_a_coverage_only_lane_appears(pg):
    conn = pg(
        [
            _lane_row(
                open=0, fixed=0, rejected=0, stale=0, offerable=0, eligible=0, carried=0, settled=0
            )
        ]
    )
    lane = review_state.lanes().lanes[0]
    assert lane.findings_total == 0
    assert lane.coverage_total == 9
    assert lane.carry_forward_rate is None and lane.settle_rate is None
    sql = conn.statements[0][0]
    assert "FROM review_findings UNION ALL" in sql and "FROM review_coverage" in sql


def test_lanes_reports_open_rows_no_round_will_ever_be_offered(pg):
    over = review_state.MAX_OPEN_FINDINGS + 7
    pg([_lane_row(open=over, offerable=over)])
    lane = review_state.lanes().lanes[0]
    assert lane.offerable == over
    assert lane.never_offered == 7


def test_lanes_never_reports_a_negative_truncation(pg):
    pg([_lane_row(open=3, offerable=3)])
    assert review_state.lanes().lanes[0].never_offered == 0


def test_lanes_shows_rows_outside_the_prompt_retention_window(pg):
    """The window bounds only ``offerable`` -- never which rows the operator sees."""
    conn = pg([_lane_row()])
    review_state.lanes()
    sql = conn.statements[0][0]
    assert sql.count("make_interval") == 1
    assert "n_offerable" in sql.split("make_interval")[1][:80]


def test_lanes_passes_and_clamps_its_filters(pg):
    conn = pg([])
    index = review_state.lanes(repo="a/b", pr=7, seat="henry", limit=9999, offset=-5)
    assert index == review_state.LaneIndex()
    assert conn.statements[0][1] == ("a/b", "a/b", 7, 7, "henry", "henry", 90, 200, 0)


def test_lanes_orders_by_last_activity_so_a_closed_pr_still_appears(pg):
    conn = pg([_lane_row()])
    review_state.lanes()
    assert "ORDER BY l.last_activity DESC" in conn.statements[0][0]


def test_lanes_raises_instead_of_degrading_to_an_empty_page(dead_pg):
    with pytest.raises(RuntimeError):
        review_state.lanes()


def test_pr_findings_returns_every_status_whole(pg):
    conn = pg(
        [
            _finding_row(status="open"),
            _finding_row(status="stale", reopened=0),
            _finding_row(status="fixed", status_reason="patched in round 3", reopened=2, total=3),
        ]
    )
    page = review_state.pr_findings("lemehmet/mepro", 1343)
    assert page.total == 1  # the window count travels on the FIRST row
    assert [row.status for row in page.rows] == ["open", "stale", "fixed"]
    assert page.rows[2].status_reason == "patched in round 3"
    assert page.rows[2].anomalous and not page.rows[0].anomalous
    assert page.rows[0].evidence == "src/app.py:118-166"
    assert page.rows[0].head_sha == "abc123"
    assert page.rows[0].created_at == _ACTIVITY.isoformat()
    assert "status =" not in conn.statements[0][0]


def test_pr_findings_passes_and_clamps_its_arguments(pg):
    conn = pg([])
    assert review_state.pr_findings("a/b", 7, seat="henry", limit=0, offset=-1) == (
        review_state.FindingPage()
    )
    assert conn.statements[0][1] == ("a/b", 7, "henry", "henry", 1, 0)


def test_pr_findings_keeps_a_missing_line_missing(pg):
    pg([_finding_row(line=None)])
    assert review_state.pr_findings("a/b", 7).rows[0].line is None


def test_pr_findings_raises_instead_of_degrading(dead_pg):
    with pytest.raises(RuntimeError):
        review_state.pr_findings("a/b", 7)


def test_pr_coverage_returns_expired_entries_beside_live_ones(pg):
    expired = datetime(2026, 7, 23, 9, 0, tzinfo=UTC)
    conn = pg([_coverage_row(total=2), _coverage_row(expired_at=expired, total=2)])
    page = review_state.pr_coverage("lemehmet/mepro", 1343)
    assert page.total == 2
    assert page.rows[0].live and page.rows[0].expired_at is None
    assert not page.rows[1].live and page.rows[1].expired_at == expired.isoformat()
    assert page.rows[0].region == "open_source"
    assert "expired_at IS NULL" not in conn.statements[0][0]


def test_pr_coverage_passes_and_clamps_its_arguments(pg):
    conn = pg([])
    assert review_state.pr_coverage("a/b", 7, limit=10_000, offset=25) == (
        review_state.CoveragePage()
    )
    assert conn.statements[0][1] == ("a/b", 7, None, None, review_state.MAX_LEDGER_ROWS, 25)


def test_pr_coverage_raises_instead_of_degrading(dead_pg):
    with pytest.raises(RuntimeError):
        review_state.pr_coverage("a/b", 7)


def _stat(**overrides) -> review_state.LaneStat:
    base = dict(
        repo="a/b",
        pr=1,
        seat="henry",
        latest_round=2,
        last_activity=None,
        counts={"open": 0, "fixed": 0, "rejected": 0, "stale": 0},
        reopened=0,
        offerable=0,
        never_offered=0,
        coverage_total=0,
        coverage_live=0,
        eligible=0,
        carried=0,
        settled=0,
    )
    base.update(overrides)
    return review_state.LaneStat(**base)


@pytest.mark.parametrize(
    ("eligible", "carried", "settled", "carry", "settle"),
    [
        (0, 0, 0, None, None),  # a single-round lane: nothing a later round could act on
        (4, 1, 2, 0.25, 0.5),
        (3, 3, 0, 1.0, 0.0),
        (2, 0, 1, 0.0, 0.5),  # the remainder is `stale`: the rates need not sum to one
    ],
)
def test_rate_arithmetic(eligible, carried, settled, carry, settle):
    lane = _stat(eligible=eligible, carried=carried, settled=settled)
    assert lane.carry_forward_rate == carry
    assert lane.settle_rate == settle


def test_iso_keeps_none_distinguishable_and_accepts_a_string():
    assert review_state._iso(None) is None
    assert review_state._iso(_ACTIVITY) == _ACTIVITY.isoformat()
    assert review_state._iso("2026-07-22T20:15:00+00:00") == "2026-07-22T20:15:00+00:00"
