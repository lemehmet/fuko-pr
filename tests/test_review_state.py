"""Tests for the per-PR review-state ledgers (#155): migration, store, degradation."""

import contextlib
import re
from pathlib import Path

import pytest

import sidecar.db
from sidecar import review_state
from sidecar.reviewer.prompt import AgenticFinding, ExaminedRegion, PriorCoverage, PriorFinding

_MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "009_review_state.sql"

_UUID = "0f9d1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b"


def _statements(path: Path) -> list[str]:
    """Split a migration exactly as ``db._migration_sql`` does."""
    sql = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
    return [" ".join(s.split()) for s in sql.split(";") if s.strip()]


class _FakeCursor:
    def __init__(self, rows, rowcount):
        self._rows = rows
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Records every statement; replays one canned result set for all of them."""

    def __init__(self, rows=(), rowcount=0):
        self.rows = list(rows)
        self.rowcount = rowcount
        self.statements: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.statements.append((" ".join(sql.split()), tuple(params)))
        return _FakeCursor(self.rows, self.rowcount)


@pytest.fixture
def pg(monkeypatch):
    """Enable persistence and patch ``sidecar.db.db`` with a recording connection."""
    monkeypatch.setattr(review_state.settings, "database_url", "postgres://x")

    def install(rows=(), rowcount=0):
        conn = _FakeConn(rows, rowcount)

        @contextlib.contextmanager
        def fake_db():
            yield conn

        monkeypatch.setattr(sidecar.db, "db", fake_db)
        return conn

    return install


def _finding(**overrides) -> AgenticFinding:
    base = dict(
        file="src/app.py",
        line=42,
        severity="high",
        category="bug",
        title="unchecked None device",
        body="open_source() may return None",
        evidence="src/app.py:118-166",
    )
    base.update(overrides)
    return AgenticFinding(**base)


def _region(**overrides) -> ExaminedRegion:
    base = dict(
        file="src/util.py",
        region="helper",
        checked="whether every caller handles a None device",
        conclusion="all four callers branch on None before use",
        evidence="src/util.py:118-166",
    )
    base.update(overrides)
    return ExaminedRegion(**base)


def test_migration_009_creates_both_ledgers_idempotently():
    """Migrations re-run on EVERY pool creation, so an unnamed CREATE INDEX would
    mint a fresh duplicate index each startup. Both tables and both indexes must
    be IF NOT EXISTS and explicitly named."""
    stmts = _statements(_MIGRATION)

    assert len(stmts) == 4
    assert stmts[0].startswith("CREATE TABLE IF NOT EXISTS review_findings")
    assert stmts[2].startswith("CREATE TABLE IF NOT EXISTS review_coverage")
    for index in (stmts[1], stmts[3]):
        assert index.startswith("CREATE INDEX IF NOT EXISTS review_")
    # The coverage index only covers rows a prompt can still read.
    assert stmts[3].endswith("WHERE expired_at IS NULL")


def test_migration_009_pins_the_status_vocabulary_but_not_the_models_words():
    """`status` is fuko's own state machine and is constrained; `severity` and
    `category` are the model's words and must never fail a write."""
    findings = _statements(_MIGRATION)[0]

    assert "CHECK (status IN ('open', 'fixed', 'rejected', 'stale'))" in findings
    assert set(review_state.FINDING_STATUSES) == {"open", "fixed", "rejected", "stale"}
    assert "CHECK (severity" not in findings and "CHECK (category" not in findings


def test_review_state_no_ops_without_a_database(monkeypatch):
    """The whole acceptance criterion: no store configured, review unaffected."""
    monkeypatch.setattr(review_state.settings, "database_url", "")

    assert review_state.record_findings("o/r", 7, "henry", 1, "sha", [_finding()]) == 0
    assert review_state.open_findings("o/r", 7, "henry") == []
    assert review_state.transition(_UUID, "fixed", "rewritten") is False
    assert review_state.touch_findings([_UUID]) == 0
    assert review_state.record_coverage("o/r", 7, "henry", 1, "sha", [_region()]) == 0
    assert review_state.live_coverage("o/r", 7, "henry") == []
    assert review_state.expire_coverage("o/r", 7, "henry", ["src/app.py"]) == 0


def test_a_store_failure_never_reaches_the_review(pg, monkeypatch, capsys):
    """State must never fail a review: a raising store degrades to the no-op value."""

    @contextlib.contextmanager
    def exploding_db():
        raise RuntimeError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(sidecar.db, "db", exploding_db)

    assert review_state.record_findings("o/r", 7, "henry", 1, "sha", [_finding()]) == 0
    assert review_state.open_findings("o/r", 7, "henry") == []
    assert review_state.transition(_UUID, "fixed") is False
    assert review_state.live_coverage("o/r", 7, "henry") == []
    assert "connection refused" in capsys.readouterr().err


def test_record_findings_writes_open_rows_and_skips_an_empty_round(pg):
    conn = pg()

    assert review_state.record_findings("o/r", 7, "henry", 3, "deadbee", [_finding()]) == 1
    sql, params = conn.statements[0]
    assert sql.startswith("INSERT INTO review_findings")
    assert params[:6] == ("o/r", 7, "henry", 3, "deadbee", "src/app.py")
    # `status` is left to the column default so 'open' is defined in one place.
    assert "status" not in sql

    assert review_state.record_findings("o/r", 7, "henry", 3, "deadbee", []) == 0
    assert len(conn.statements) == 1


def test_stored_text_is_bounded_because_the_ledger_is_replayed_every_round(pg):
    conn = pg()
    review_state.record_findings(
        "o/r", 7, "henry", 1, "sha", [_finding(body="x" * (review_state.MAX_TEXT + 500))]
    )

    body = conn.statements[0][1][10]
    assert len(body) == review_state.MAX_TEXT


def test_open_findings_returns_rows_paired_with_their_prior_finding(pg):
    conn = pg(
        rows=[
            (_UUID, "src/app.py", 42, "high", "bug", "unchecked None", "body text", 2),
        ]
    )
    stored = review_state.open_findings("o/r", 7, "henry")

    assert len(stored) == 1
    assert stored[0].id == _UUID
    assert stored[0].prior == PriorFinding(
        file="src/app.py",
        title="unchecked None",
        body="body text",
        line=42,
        severity="high",
        category="bug",
        round=2,
    )
    sql, params = conn.statements[0]
    assert "status = 'open'" in sql and "ORDER BY round, created_at, id" in sql
    assert params[3] == review_state.RETENTION_DAYS
    assert params[4] == review_state.MAX_OPEN_FINDINGS


def test_open_findings_measures_retention_from_the_last_reassertion(pg):
    """``touch_findings`` is what keeps an unsettled finding live, so the window
    has to be measured from ``updated_at``: keyed on ``created_at`` a finding a
    seat is still re-asserting would age out of its own ledger."""
    conn = pg()
    review_state.open_findings("o/r", 7, "henry")

    sql = conn.statements[0][0]
    assert "updated_at > now() - make_interval(days => %s)" in sql
    assert "created_at > now()" not in sql


def test_open_findings_orders_same_round_findings_by_a_total_tiebreaker(pg):
    """One round's rows share a transaction timestamp, and equal sort keys have
    no stable order in Postgres -- so the minted ``pN`` ids would permute between
    reads without an immutable tiebreaker."""
    conn = pg()
    review_state.open_findings("o/r", 7, "henry")

    assert conn.statements[0][0].endswith("ORDER BY round, created_at, id LIMIT %s")


def test_open_findings_clamps_a_caller_supplied_limit(pg):
    conn = pg()
    review_state.open_findings("o/r", 7, "henry", limit=10_000)

    assert conn.statements[0][1][4] == review_state.MAX_OPEN_FINDINGS


def test_transition_settles_only_open_rows(pg):
    conn = pg(rowcount=1)

    assert review_state.transition(_UUID, "fixed", "rewritten in this head") is True
    sql, params = conn.statements[0]
    assert "updated_at = now()" in sql
    assert sql.endswith("WHERE id = %s AND status = 'open'")
    assert params == ("fixed", "rewritten in this head", _UUID)


def test_transition_refuses_an_unknown_verdict_or_a_malformed_id(pg):
    """Fail-safe direction: a finding that does not transition stays open."""
    conn = pg(rowcount=1)

    assert review_state.transition(_UUID, "still_open") is False
    assert review_state.transition(_UUID, "resolved") is False
    assert review_state.transition("p1", "fixed") is False
    assert review_state.transition("../../etc/passwd", "fixed") is False
    assert conn.statements == []


def test_transition_reports_no_change_when_the_row_was_already_settled(pg):
    conn = pg(rowcount=0)

    assert review_state.transition(_UUID, "stale", "file deleted") is False
    assert len(conn.statements) == 1


def test_touch_findings_refreshes_reasserted_rows_only(pg):
    conn = pg(rowcount=2)

    assert review_state.touch_findings([_UUID, "p1"]) == 2
    sql, params = conn.statements[0]
    assert "SET updated_at = now()" in sql and "status = 'open'" in sql
    assert params == ([_UUID],)

    assert review_state.touch_findings(["p1", "p2"]) == 0
    assert len(conn.statements) == 1


def test_record_coverage_writes_what_the_round_said(pg):
    conn = pg()

    assert review_state.record_coverage("o/r", 7, "henry", 2, "deadbee", [_region()]) == 1
    sql, params = conn.statements[0]
    assert sql.startswith("INSERT INTO review_coverage")
    assert params[5:8] == ("src/util.py", "helper", "whether every caller handles a None device")

    assert review_state.record_coverage("o/r", 7, "henry", 2, "deadbee", []) == 0
    assert len(conn.statements) == 1


def test_live_coverage_reads_unexpired_rows_newest_first(pg):
    conn = pg(rows=[("src/util.py", "checked", "established", "evidence", "helper", 4)])
    coverage = review_state.live_coverage("o/r", 7, "henry")

    assert coverage == [
        PriorCoverage(
            file="src/util.py",
            checked="checked",
            conclusion="established",
            evidence="evidence",
            region="helper",
            round=4,
        )
    ]
    sql, params = conn.statements[0]
    assert "expired_at IS NULL" in sql and "ORDER BY round DESC" in sql
    assert params[4] == review_state.MAX_LIVE_COVERAGE


def test_expire_coverage_scopes_to_the_files_the_delta_touched(pg):
    conn = pg(rowcount=3)

    assert review_state.expire_coverage("o/r", 7, "henry", ["src/app.py", "src/util.py"]) == 3
    sql, params = conn.statements[0]
    assert "SET expired_at = now()" in sql and "file = ANY(%s)" in sql
    assert params == ("o/r", 7, "henry", ["src/app.py", "src/util.py"])


def test_expire_coverage_wholesale_needs_none_not_an_empty_list(pg):
    """A rebase expires everything; a round whose delta touched nothing expires
    nothing. Conflating the two would silently discard the ledger."""
    conn = pg(rowcount=9)

    assert review_state.expire_coverage("o/r", 7, "henry") == 9
    assert "file = ANY" not in conn.statements[0][0]

    assert review_state.expire_coverage("o/r", 7, "henry", []) == 0
    assert len(conn.statements) == 1


def test_review_state_is_postgres_only_and_says_so():
    """The sqlite-vec gap is a decision, not an omission -- both modules state it."""
    from sidecar import sqlite_store

    assert "review_state" in sqlite_store.__doc__
    assert "Postgres only" in review_state.__doc__
