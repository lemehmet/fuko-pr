"""Tests for the per-PR review-state ledgers (#155): migration, store, degradation."""

import contextlib
import re
from pathlib import Path

import pytest

import sidecar.db
from sidecar import review_state
from sidecar.reviewer.prompt import AgenticFinding, ExaminedRegion, PriorCoverage, PriorFinding

_MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"
_MIGRATION = _MIGRATIONS / "009_review_state.sql"
_MIGRATION_REOPEN = _MIGRATIONS / "010_review_finding_reopen.sql"

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


def test_migration_010_adds_the_reopen_counter_idempotently():
    """A re-applied ALTER must not fail the pool: every migration re-runs on each
    pool creation, so the column add is IF NOT EXISTS like everything else."""
    stmts = _statements(_MIGRATION_REOPEN)

    assert len(stmts) == 1
    assert stmts[0].startswith("ALTER TABLE review_findings")
    assert "ADD COLUMN IF NOT EXISTS reopened INTEGER NOT NULL DEFAULT 0" in stmts[0]
    # A reversal is only defined for the two statuses a model VERDICT produces.
    assert set(review_state.REOPENABLE_STATUSES) == {"fixed", "rejected"}
    assert review_state.REOPENABLE_STATUSES < review_state.FINDING_STATUSES


def test_review_state_no_ops_without_a_database(monkeypatch):
    """The whole acceptance criterion: no store configured, review unaffected."""
    monkeypatch.setattr(review_state.settings, "database_url", "")

    assert review_state.record_findings("o/r", 7, "henry", 1, "sha", [_finding()]) == 0
    assert review_state.open_findings("o/r", 7, "henry") == review_state.OpenLedger()
    assert review_state.transition("o/r", 7, "henry", _UUID, "fixed", "rewritten") is False
    assert review_state.settled_findings("o/r", 7, "henry") == ()
    assert review_state.reopen("o/r", 7, "henry", _UUID, "re-found") is False
    assert review_state.touch_findings("o/r", 7, "henry", [_UUID]) == 0
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
    assert review_state.open_findings("o/r", 7, "henry") == review_state.OpenLedger()
    assert review_state.transition("o/r", 7, "henry", _UUID, "fixed") is False
    assert review_state.settled_findings("o/r", 7, "henry") == ()
    assert review_state.reopen("o/r", 7, "henry", _UUID, "re-found") is False
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
            (
                _UUID,
                "src/app.py",
                42,
                "high",
                "bug",
                "unchecked None",
                "body text",
                "read src/app.py:118-166",
                2,
                1,
            ),
        ]
    )
    ledger = review_state.open_findings("o/r", 7, "henry")
    stored = ledger.rows

    assert len(stored) == 1
    assert ledger.truncated == 0
    assert stored[0].id == _UUID
    assert stored[0].prior == PriorFinding(
        file="src/app.py",
        title="unchecked None",
        body="body text",
        line=42,
        severity="high",
        category="bug",
        round=2,
        evidence="read src/app.py:118-166",
    )
    sql, params = conn.statements[0]
    assert "status = 'open'" in sql and "ORDER BY round, created_at, id" in sql
    assert params[3] == review_state.RETENTION_DAYS
    assert params[4] == review_state.MAX_OPEN_FINDINGS


def test_findings_evidence_is_both_written_and_read_back(pg):
    """#174: the column was written on every recorded finding and named by no
    read, so a carried finding reached the next round with the citation its
    publication carried stripped off -- the half that supports re-verifying it.
    Pinning BOTH ends here is what stops it drifting back to write-only."""
    written = pg()
    review_state.record_findings("o/r", 7, "henry", 1, "sha", [_finding()])
    write_sql, write_params = written.statements[0]

    read = pg()
    review_state.open_findings("o/r", 7, "henry")
    projection = read.statements[0][0].split(" FROM ")[0]

    assert "evidence" in write_sql
    assert "src/app.py:118-166" in write_params
    assert "evidence" in projection


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


def _open_row(n: int, total: int) -> tuple:
    return (_UUID, "src/app.py", n, "high", "bug", f"finding {n}", "body", "evidence", 1, total)


def test_open_findings_reports_how_many_rows_the_cap_cut(pg):
    """The cut rows are the NEWEST, and a row no round is offered is a row no
    round can settle (#173) -- so the read has to say it happened."""
    conn = pg(rows=[_open_row(n, 512) for n in range(3)])
    ledger = review_state.open_findings("o/r", 7, "henry")

    assert len(ledger.rows) == 3
    assert ledger.truncated == 509
    # From the same read that did the cutting: a second COUNT query could see a
    # different window than the one these rows came from.
    assert "count(*) OVER ()" in conn.statements[0][0]
    assert len(conn.statements) == 1


def test_open_findings_reports_no_truncation_when_the_window_fits(pg):
    conn = pg(rows=[_open_row(n, 2) for n in range(2)])

    assert review_state.open_findings("o/r", 7, "henry").truncated == 0
    assert len(conn.statements) == 1


def test_open_findings_reports_no_truncation_on_an_empty_ledger(pg):
    """No rows means no window column to read: the count must not be guessed."""
    pg()

    assert review_state.open_findings("o/r", 7, "henry") == review_state.OpenLedger()


def test_open_findings_never_reports_a_negative_truncation(pg):
    """A total below the rows in hand is a contradiction; report none, not less
    than none -- a negative would read as a cut in the caller's log line."""
    pg(rows=[_open_row(n, 1) for n in range(3)])

    assert review_state.open_findings("o/r", 7, "henry").truncated == 0


def test_next_round_counts_settled_rounds_too(pg):
    """A round whose findings were all fixed still happened: re-issuing its
    number would put two different rounds behind one label in the prompt."""
    conn = pg(rows=[(4,)])

    assert review_state.next_round("o/r", 7, "henry") == 4
    sql, params = conn.statements[0]
    assert "coalesce(max(round), 0) + 1" in sql and "status" not in sql
    assert params == ("o/r", 7, "henry", "o/r", 7, "henry")


def test_next_round_counts_coverage_rounds_as_well_as_finding_rounds(pg):
    """A round that recorded only coverage still happened (#157).

    Reading `review_findings` alone would re-issue `1` for every round in a
    found-nothing streak, and the prompt would then show N rounds of coverage
    under one label."""
    conn = pg(rows=[(4,)])

    assert review_state.next_round("o/r", 7, "henry") == 4
    sql, _ = conn.statements[0]
    assert "review_findings" in sql and "review_coverage" in sql and "UNION ALL" in sql


def test_next_round_is_one_for_a_seats_first_round(pg):
    """An empty read is "nothing recorded before this", the same as no store."""
    pg()

    assert review_state.next_round("o/r", 7, "henry") == 1


def test_next_round_is_one_without_a_store(monkeypatch):
    monkeypatch.setattr(review_state.settings, "database_url", "")

    assert review_state.next_round("o/r", 7, "henry") == 1


def test_transition_settles_only_open_rows(pg):
    conn = pg(rowcount=1)

    assert (
        review_state.transition("o/r", 7, "henry", _UUID, "fixed", "rewritten in this head") is True
    )
    sql, params = conn.statements[0]
    assert "updated_at = now()" in sql
    # The lane is matched in SQL, not merely supplied by the caller: over the
    # HTTP seam (#171) a row id arrives as a claim in a request body, and one
    # seat closing another's finding is what #160 forbids.
    assert sql.endswith("WHERE id = %s AND status = 'open' AND repo = %s AND pr = %s AND seat = %s")
    assert params == ("fixed", "rewritten in this head", _UUID, "o/r", 7, "henry")


def test_transition_refuses_an_unknown_verdict_or_a_malformed_id(pg):
    """Fail-safe direction: a finding that does not transition stays open."""
    conn = pg(rowcount=1)

    assert review_state.transition("o/r", 7, "henry", _UUID, "still_open") is False
    assert review_state.transition("o/r", 7, "henry", _UUID, "resolved") is False
    assert review_state.transition("o/r", 7, "henry", "p1", "fixed") is False
    assert review_state.transition("o/r", 7, "henry", "../../etc/passwd", "fixed") is False
    assert conn.statements == []


def test_transition_reports_no_change_when_the_row_was_already_settled(pg):
    conn = pg(rowcount=0)

    assert review_state.transition("o/r", 7, "henry", _UUID, "stale", "file deleted") is False
    assert len(conn.statements) == 1


def test_settled_findings_reads_this_seats_model_closed_rows_only(pg):
    """Scope is the control (#177): a wider read would let one seat re-raise -- and
    so overrule -- a closure another seat made, which no verdict of its own could."""
    conn = pg(rows=[("id-1", "src/app.py", "leaks the handle", "fixed", 2, "rewritten")])

    rows = review_state.settled_findings("o/r", 7, "henry")

    assert rows == (
        review_state.SettledFinding(
            id="id-1",
            file="src/app.py",
            title="leaks the handle",
            status="fixed",
            round=2,
            reason="rewritten",
        ),
    )
    sql, params = conn.statements[0]
    assert "WHERE repo = %s AND pr = %s AND seat = %s AND status = ANY(%s)" in sql
    # Ordered by closure time so a cut keeps the rows a round is likeliest to
    # contradict, and windowed on the same column `transition` refreshes.
    assert "updated_at > now() - make_interval(days => %s)" in sql
    assert sql.endswith("ORDER BY updated_at DESC, id DESC LIMIT %s")
    assert params == ("o/r", 7, "henry", ["fixed", "rejected"], review_state.RETENTION_DAYS, 200)


def test_settled_findings_clamps_a_caller_supplied_limit(pg):
    conn = pg()

    review_state.settled_findings("o/r", 7, "henry", limit=10_000)
    review_state.settled_findings("o/r", 7, "henry", limit=0)

    assert [s[1][-1] for s in conn.statements] == [review_state.MAX_SETTLED_FINDINGS, 1]


def test_reopen_returns_a_model_closed_row_to_open(pg):
    conn = pg(rowcount=1)

    reason = "re-raised in round 4: contradicts fixed in round 2"
    assert review_state.reopen("o/r", 7, "henry", _UUID, reason) is True
    sql, params = conn.statements[0]
    assert "SET status = 'open'" in sql and "reopened = reopened + 1" in sql
    assert "updated_at = now()" in sql
    # Scoped like `transition`: re-raising another seat's row is the same
    # cross-seat coupling as closing one, reached from the other direction.
    assert sql.endswith(
        "WHERE id = %s AND status = ANY(%s) AND repo = %s AND pr = %s AND seat = %s"
    )
    assert params == (reason, _UUID, ["fixed", "rejected"], "o/r", 7, "henry")


def test_reopen_refuses_a_malformed_id_and_reports_an_unmatched_row(pg):
    """`stale` and `open` rows are outside the matched statuses, so the UPDATE
    matches nothing and the caller records the claim as a fresh row instead."""
    conn = pg(rowcount=0)

    assert review_state.reopen("o/r", 7, "henry", "p1", "re-found") is False
    assert conn.statements == []
    assert review_state.reopen("o/r", 7, "henry", _UUID, "re-found") is False
    assert len(conn.statements) == 1


def test_touch_findings_refreshes_reasserted_rows_only(pg):
    conn = pg(rowcount=2)

    assert review_state.touch_findings("o/r", 7, "henry", [_UUID, "p1"]) == 2
    sql, params = conn.statements[0]
    assert "SET updated_at = now()" in sql and "status = 'open'" in sql
    assert "AND repo = %s AND pr = %s AND seat = %s" in sql
    assert params == ([_UUID], "o/r", 7, "henry")

    assert review_state.touch_findings("o/r", 7, "henry", ["p1", "p2"]) == 0
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


def test_live_coverage_orders_same_round_entries_by_a_total_tiebreaker(pg):
    """Coverage mints no ids, but both caps that can fall inside a round -- this
    LIMIT and the renderer's max_coverage -- count entries, so which same-round
    siblings survive a cut must not move between two reads that changed nothing."""
    conn = pg()
    review_state.live_coverage("o/r", 7, "henry")

    assert conn.statements[0][0].endswith("ORDER BY round DESC, created_at DESC, id DESC LIMIT %s")


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


def test_expire_coverage_clips_paths_the_way_record_coverage_stored_them(pg):
    """``file`` is a matching key, so both sides must apply the same transform:
    a path over MAX_TEXT was stored truncated, and matching it raw would expire
    nothing while reporting the same 0 as 'no coverage for that file'."""
    conn = pg(rowcount=1)
    long_path = "src/" + "a" * (review_state.MAX_TEXT + 90) + ".py"

    assert review_state.expire_coverage("o/r", 7, "henry", [long_path]) == 1
    assert conn.statements[0][1][3] == [long_path[: review_state.MAX_TEXT]]


def test_a_bare_string_is_refused_where_a_sequence_of_them_is_meant(pg, capsys):
    """``str`` satisfies ``Sequence[str]`` and no type checker objects, so the
    argument would otherwise be iterated CHARACTER by character: coverage that
    expires nothing, ids that match nothing, both reported as a plain 0."""
    conn = pg(rowcount=5)

    assert review_state.expire_coverage("o/r", 7, "henry", "src/app.py") == 0
    assert review_state.touch_findings("o/r", 7, "henry", _UUID) == 0
    assert conn.statements == []

    err = capsys.readouterr().err
    assert "expire_coverage failed" in err and "files must be a sequence" in err
    assert "touch_findings failed" in err and "finding_ids must be a sequence" in err


def test_review_state_is_postgres_only_and_says_so():
    """The sqlite-vec gap is a decision, not an omission -- both modules state it."""
    from sidecar import sqlite_store

    assert "review_state" in sqlite_store.__doc__
    assert "Postgres only" in review_state.__doc__
