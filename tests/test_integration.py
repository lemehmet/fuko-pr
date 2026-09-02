"""Integration tests: require a live pgvector + embeddings backend.

Enable by exporting FUKO_DATABASE_URL (and running the embeddings model, e.g. Ollama).
Skipped otherwise.
"""

import io
import os
import sys

import pytest

from sidecar.config import settings

pytestmark = pytest.mark.skipif(
    not (settings.database_url or os.environ.get("FUKO_DATABASE_URL")),
    reason="set FUKO_DATABASE_URL (and run the embeddings backend) to enable",
)

TEST_REPO = "fuko-ci/test"
_TOKEN = "ci-test-token"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


@pytest.fixture(autouse=True)
def _auth_token(monkeypatch):
    from sidecar.config import settings

    monkeypatch.setattr(settings, "auth_token", _TOKEN)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    from sidecar.db import db

    with db() as conn:
        conn.execute("DELETE FROM learnings WHERE repo = %s", (TEST_REPO,))
        conn.execute("DELETE FROM review_findings WHERE repo = %s", (TEST_REPO,))
        conn.execute("DELETE FROM review_coverage WHERE repo = %s", (TEST_REPO,))


def test_ingest_query_roundtrip():
    from sidecar import ingest as I
    from sidecar import retrieve
    from sidecar.models import IngestItem

    inserted, skipped = I.ingest(
        TEST_REPO,
        [
            IngestItem(
                text="Always use absolute imports in this codebase.",
                source="remember",
                file_globs=["src/**/*.py"],
            )
        ],
    )
    assert inserted == 1
    assert skipped == 0

    results = retrieve.query(TEST_REPO, ["src/foo/bar.py", "README.md"])
    assert any("absolute imports" in r["text"] for r in results)


def test_api_endpoints():
    from fastapi.testclient import TestClient

    from sidecar.main import app

    client = TestClient(app, headers=_AUTH)
    assert client.get("/healthz").json() == {"ok": True}

    r = client.post("/query", json={"repo": TEST_REPO, "files": ["src/x.py"]})
    assert r.status_code == 200
    assert "results" in r.json()

    f = client.post("/forget", json={"repo": TEST_REPO, "all": True})
    assert f.status_code == 200
    assert f.json()["deleted"] >= 0


def test_list_learnings_endpoint():
    from fastapi.testclient import TestClient

    from sidecar import ingest as I
    from sidecar.main import app
    from sidecar.models import IngestItem

    I.ingest(
        TEST_REPO,
        [
            IngestItem(
                text="Declining — this synchronous path is intentional for ordering here.",
                source="review_thread",
                file_globs=["src/q.py"],
            ),
            IngestItem(text="Use absolute imports across the codebase always.", source="remember"),
        ],
    )
    client = TestClient(app, headers=_AUTH)

    r = client.get("/learnings", params={"repo": TEST_REPO})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 2
    assert any("intentional for ordering" in x["text"] for x in body["learnings"])

    scoped = client.get("/learnings", params={"repo": TEST_REPO, "source": "remember"})
    assert [x["source"] for x in scoped.json()["learnings"]] == ["remember"]


def test_ingest_threads_mines_declines():
    from fastapi.testclient import TestClient

    from sidecar import retrieve
    from sidecar.main import app

    threads = [
        {
            "isResolved": True,
            "path": "src/a.py",
            "comments": {
                "nodes": [
                    {"author": {"login": "github-actions[bot]"}, "body": "consider X", "url": "u1"},
                    {
                        "author": {"login": "bob"},
                        "body": "We keep pattern Z intentionally; async reorders otherwise.",
                        "url": "u2",
                    },
                ]
            },
        },
        {
            "isResolved": True,
            "path": "src/b.py",
            "comments": {
                "nodes": [
                    {"author": {"login": "github-actions[bot]"}, "body": "bot only", "url": "u3"}
                ]
            },
        },
    ]
    client = TestClient(app, headers=_AUTH)
    r = client.post("/ingest-threads", json={"repo": TEST_REPO, "threads": threads})
    assert r.status_code == 200
    assert r.json()["considered"] == 2
    assert r.json()["inserted"] == 1
    assert any("pattern Z" in x["text"] for x in retrieve.query(TEST_REPO, ["src/a.py"]))


def test_comment_remember_and_forget():
    from fastapi.testclient import TestClient

    from sidecar import retrieve
    from sidecar.main import app

    client = TestClient(app, headers=_AUTH)
    r = client.post(
        "/comment",
        json={
            "repo": TEST_REPO,
            "body": "/remember prefer keyword-only arguments",
            "source_url": "http://x/1",
            "origin_user": "alice",
        },
    )
    assert r.json() == {"action": "remember", "inserted": 1, "skipped": 0}
    assert any("keyword" in x["text"] for x in retrieve.query(TEST_REPO, ["a.py"]))

    f = client.post("/comment", json={"repo": TEST_REPO, "body": "/forget source=remember"})
    assert f.json()["action"] == "forget"
    assert f.json()["deleted"] >= 1

    ignored = client.post("/comment", json={"repo": TEST_REPO, "body": "nice PR"})
    assert ignored.json() == {"action": "ignored"}


def test_pg_migrate_embed_dim_rebuilds_column():
    from sidecar import ingest as I
    from sidecar import retrieve
    from sidecar.db import _existing_embed_dim, _migrate_embed_dim, db
    from sidecar.embed import get_embedder
    from sidecar.models import IngestItem

    dim = get_embedder().probe_dim()
    I.ingest(TEST_REPO, [IngestItem(text="a rule that must survive re-embedding", source="docs")])

    # exercise the full migration SQL (drop/add column, re-embed every row, rebuild
    # the HNSW index); migrating to the same dim keeps the test self-contained.
    with db() as conn:
        _migrate_embed_dim(conn, dim)
        assert _existing_embed_dim(conn) == dim

    results = retrieve.query(TEST_REPO, ["x.py"], query_text="rule survive re-embedding")
    assert any("survive re-embedding" in r["text"] for r in results)


def test_cli_query_runs(capsys, monkeypatch):
    from sidecar.cli import main

    monkeypatch.setattr(sys, "argv", ["fuko", "query", "--repo", TEST_REPO, "--file", "src/x.py"])
    main()
    assert isinstance(capsys.readouterr().out, str)


def test_cli_ingest_docs_and_forget(tmp_path, monkeypatch, capsys):
    from sidecar.cli import main

    doc = tmp_path / "note.md"
    doc.write_text("# Title\n\nimportant rule for the service\n")

    monkeypatch.setattr(sys, "argv", ["fuko", "ingest-docs", str(doc), "--repo", TEST_REPO])
    main()
    assert "ingested" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["fuko", "forget", "--repo", TEST_REPO, "--all"])
    main()
    assert "deleted" in capsys.readouterr().out


def test_cli_retrieve(tmp_path, monkeypatch):
    from sidecar import ingest as I
    from sidecar.cli import main
    from sidecar.models import IngestItem

    I.ingest(TEST_REPO, [IngestItem(text="a rule to recall", source="docs")])
    out = tmp_path / "extra.md"
    monkeypatch.setattr(sys, "stdin", io.StringIO("src/x.py\n"))
    monkeypatch.setattr(sys, "argv", ["fuko", "retrieve", "--repo", TEST_REPO, "--out", str(out)])
    main()
    assert out.exists()


def test_review_state_ledgers_roundtrip_on_a_live_server():
    """Exercise the review-state store against real SQL, not a fake connection.

    Everything the unit tests cannot see lives here: ``make_interval``,
    ``rowcount``, ``= ANY(%s)``, and above all the ledger's ids, which are
    Python ``str`` bound against ``uuid`` columns. psycopg 3 sends a ``str``
    parameter with an unspecified type, so the server resolves it from the
    column it is compared against -- but that is a driver property a recording
    connection can never demonstrate. Without this test a binding failure would
    be swallowed by ``_best_effort`` and read as "nothing to settle": findings
    would insert and read back while never transitioning, and the open ledger
    would grow without bound.

    ``count(*) OVER ()`` is here for the same reason: a recording connection
    replays whatever the fake decided the window column holds, so only a real
    server can show that the count is evaluated after the ``WHERE`` and before
    the ``LIMIT`` -- which is the whole basis for ``truncated`` being unable to
    disagree with the rows it accompanies.
    """
    from sidecar import review_state as R
    from sidecar.db import db
    from sidecar.reviewer.prompt import AgenticFinding, ExaminedRegion

    head = "0" * 40

    def _finding(title: str) -> AgenticFinding:
        return AgenticFinding(
            file="src/app.py",
            line=42,
            severity="high",
            category="bug",
            title=title,
            body="body text",
            evidence="src/app.py:40-44",
        )

    assert R.record_findings(TEST_REPO, 1, "henry", 0, head, [_finding("a"), _finding("b")]) == 2

    ledger = R.open_findings(TEST_REPO, 1, "henry")
    stored = ledger.rows
    assert sorted(s.prior.title for s in stored) == ["a", "b"]
    # #174: the evidence a finding was published with survives the round trip,
    # so the round asked to re-verify it is not handed the claim ungrounded.
    assert {s.prior.evidence for s in stored} == {"src/app.py:40-44"}
    assert ledger.truncated == 0
    # A window the cap cuts: two open rows, one asked for, so the count the
    # server computes over the pre-LIMIT window must report the other one.
    cut = R.open_findings(TEST_REPO, 1, "henry", limit=1)
    assert len(cut.rows) == 1
    assert cut.truncated == 1
    # WHICH row survives is the half the count cannot show. The cap has to keep
    # the OLDEST, because that is what makes a row's minted ``pN`` id the same
    # id next round; a newest-first or unordered LIMIT would satisfy both
    # assertions above while permuting every id the prompt carries.
    assert [s.id for s in cut.rows] == [s.id for s in stored[:1]]
    # Same-round rows share a transaction timestamp, so their relative order is
    # decided by the ``id`` tiebreaker: arbitrary, but the same on every read --
    # which is the stability the minted ``pN`` ids depend on.
    assert [s.id for s in R.open_findings(TEST_REPO, 1, "henry").rows] == [s.id for s in stored]

    # str id against a uuid column, both shapes the store uses.
    ids = [s.id for s in stored]
    assert R.touch_findings(TEST_REPO, 1, "henry", ids) == 2

    # The lane the id-addressed writes carry (#171) is matched in SQL, and the
    # MISMATCH is the half that proves it. Over the wire a row id is a claim in
    # a request body, so what has to hold is that offering these same real ids
    # under a lane that is not theirs changes nothing -- #160's cross-seat
    # coupling, and the entire reason the lane travels with the id.
    #
    # A positive result cannot show this. It proves binding and parameter ORDER
    # (a mis-ordered bind could not match the row), but `repo` and `seat` are
    # both TEXT, so a WHERE that compared the wrong column to the wrong
    # parameter would satisfy every positive assertion in the suite. Nor can the
    # unit tier: it pins SQL text and parameter tuples against a replay fake
    # returning a canned rowcount, which is true of any WHERE clause, and the
    # ledger's in-memory fake enforces the lane in Python rather than in SQL.
    # Only a real server evaluates the predicate. (`qwen-anthropic/qwen3.8-max`.)
    assert R.touch_findings(TEST_REPO, 1, "gray", ids) == 0
    assert R.touch_findings("other/repo", 1, "henry", ids) == 0
    assert R.touch_findings(TEST_REPO, 2, "henry", ids) == 0

    settled = next(s for s in stored if s.prior.title == "a")
    # Refused for the wrong seat, and the row it refused is still open -- which
    # the transition immediately below proves by succeeding on it.
    assert R.transition(TEST_REPO, 1, "gray", settled.id, "fixed", "another seat's row") is False
    assert (
        R.transition(TEST_REPO, 1, "henry", settled.id, "fixed", "rewritten in this head") is True
    )
    assert R.transition(TEST_REPO, 1, "henry", settled.id, "fixed", "replayed stale id") is False

    assert [s.prior.title for s in R.open_findings(TEST_REPO, 1, "henry").rows] == ["b"]

    # #177's reversal, on the row the transition above actually closed. Both
    # halves are SQL a recording connection cannot demonstrate: a status list
    # bound with `= ANY(%s)`, and an increment of the column migration 010 adds
    # -- the one genuinely new write here. A binding failure in either is
    # swallowed by `_best_effort` and reads as its neutral value (`()`, `False`),
    # i.e. "no closure to re-raise", so the feature would be silently off while
    # every unit test stayed green (qwen3.8-max on #189, the #169 shape).
    closed = R.settled_findings(TEST_REPO, 1, "henry")
    assert [(c.id, c.status, c.title, c.reason) for c in closed] == [
        (settled.id, "fixed", "a", "rewritten in this head")
    ]
    # A closure is not another seat's to undo either: same id, wrong lane, and
    # the row stays closed -- which the reopen below proves by succeeding on it.
    assert R.settled_findings(TEST_REPO, 1, "gray") == ()
    assert R.reopen(TEST_REPO, 1, "gray", settled.id, "another seat's closure") is False
    assert (
        R.reopen(
            TEST_REPO, 1, "henry", settled.id, "re-raised: an independent finding contradicts fixed"
        )
        is True
    )
    assert sorted(s.prior.title for s in R.open_findings(TEST_REPO, 1, "henry").rows) == ["a", "b"]
    # Open again, so the row is no longer the settled read's to offer and a
    # replayed reopen changes nothing -- the count cannot inflate on an open row.
    assert R.settled_findings(TEST_REPO, 1, "henry") == ()
    assert R.reopen(TEST_REPO, 1, "henry", settled.id, "replayed") is False
    with db() as conn:
        row = conn.execute(
            "SELECT reopened, status_reason FROM review_findings WHERE id = %s",
            (settled.id,),
        ).fetchone()
    assert row == (1, "re-raised: an independent finding contradicts fixed")

    region = ExaminedRegion(
        file="src/app.py",
        region="L40-L44",
        checked="does the caller handle None?",
        conclusion="it does not; the guard is in the caller above",
        evidence="src/app.py:40-44",
    )
    other = region.model_copy(update={"file": "src/util.py"})
    assert R.record_coverage(TEST_REPO, 1, "henry", 0, head, [region, other]) == 2
    assert len(R.live_coverage(TEST_REPO, 1, "henry")) == 2

    assert R.expire_coverage(TEST_REPO, 1, "henry", []) == 0
    assert R.expire_coverage(TEST_REPO, 1, "henry", ["src/app.py"]) == 1
    assert [c.file for c in R.live_coverage(TEST_REPO, 1, "henry")] == ["src/util.py"]
    assert R.expire_coverage(TEST_REPO, 1, "henry") == 1
    assert R.live_coverage(TEST_REPO, 1, "henry") == []

    # `next_round` spans BOTH ledgers (#157), and its UNION ALL is the only new
    # SQL this tier adds. A recording connection can show the statement contains
    # the union but never that the server accepts it, and `@_best_effort` turns a
    # server-side failure into `1` -- which is precisely the "N rounds under one
    # label" defect the union exists to prevent, silently, behind a default-off
    # flag. Same standard as #169 (`qwen-anthropic/qwen3.8-max`).
    assert R.next_round(TEST_REPO, 1, "henry") == 1
    assert R.record_coverage(TEST_REPO, 1, "henry", 5, head, [region]) == 1
    assert R.next_round(TEST_REPO, 1, "henry") == 6
    # An EXPIRED coverage row is still a round that happened: the query has no
    # `expired_at` filter, and re-issuing the number of a round whose coverage
    # has since been invalidated is the same collision.
    assert R.expire_coverage(TEST_REPO, 1, "henry") == 1
    assert R.next_round(TEST_REPO, 1, "henry") == 6


def test_operator_ledger_reads_on_a_live_server():
    """Run the three operator statements against real SQL, for the reasons above.

    ``lanes``/``pr_findings``/``pr_coverage`` (#235) are the most complex SQL in
    the module -- a ``UNION ALL`` over both ledgers feeding three CTEs, four
    ``count(...) FILTER`` tallies, ``make_interval(days => %s)``, the
    ``%s::text IS NULL OR col = %s::text`` optional-filter pattern, and
    ``count(*) OVER ()`` -- and a recording connection can demonstrate none of
    it: it replays whatever the fake decided, so a wrong FILTER predicate or an
    unbound cast reads as a pass. The consequence is specific to these reads.
    They raise instead of degrading, and the page turns any exception into
    "store unreachable", so a broken statement would report a HEALTHY store as
    an outage on every load while the whole unit suite stayed green.

    Three lanes, because the shapes that can vanish are the ones only a server
    can show: a lane with coverage and no findings must still appear (the round
    happened), a lane with findings and no coverage must report ``0/0`` rather
    than dropping out of the join, and a row older than the retention window
    must be SHOWN while being excluded from ``offerable`` -- retention bounds
    what a round would be OFFERED, never what an operator may read.
    """
    from sidecar import review_state as R
    from sidecar.db import db
    from sidecar.reviewer.prompt import AgenticFinding, ExaminedRegion

    head = "2" * 40

    def _finding(title: str) -> AgenticFinding:
        return AgenticFinding(
            file="src/app.py",
            line=7,
            severity="high",
            category="bug",
            title=title,
            body="body text",
            evidence="src/app.py:5-9",
        )

    def _region(file: str) -> ExaminedRegion:
        return ExaminedRegion(
            file=file,
            region="L1-L20",
            checked="the null path",
            conclusion="guarded",
            evidence=f"{file}:1-20",
        )

    # Lane A: two rounds of findings plus coverage -- the lane every rate is
    # measured on. Round 0 files three claims; two are settled; round 1 adds one
    # more, which makes round 1 the "latest" the rates measure BEFORE.
    assert R.record_findings(TEST_REPO, 10, "henry", 0, head, [_finding(t) for t in "abc"]) == 3
    first = {s.prior.title: s.id for s in R.open_findings(TEST_REPO, 10, "henry").rows}
    assert R.transition(TEST_REPO, 10, "henry", first["a"], "fixed", "patched")
    assert R.transition(TEST_REPO, 10, "henry", first["b"], "rejected", "premise is wrong")
    assert R.record_findings(TEST_REPO, 10, "henry", 1, head, [_finding("d")]) == 1
    assert R.record_coverage(TEST_REPO, 10, "henry", 1, head, [_region("a.py"), _region("b.py")])
    assert R.expire_coverage(TEST_REPO, 10, "henry", ["b.py"]) == 1
    # Age one still-open row past the retention window. It is the assertion the
    # whole "not a sweep" contract rests on: the row stays visible and keeps
    # counting as open, and only `offerable` drops it.
    with db() as conn:
        conn.execute(
            "UPDATE review_findings SET updated_at = now() - make_interval(days => 200) "
            "WHERE id = %s",
            (first["c"],),
        )

    # Lane B: coverage and no findings at all. `next_round` counts both ledgers,
    # so this is a real round and the lane must not vanish.
    assert R.record_coverage(TEST_REPO, 11, "gray", 0, head, [_region("c.py")]) == 1
    # Lane C: findings and no coverage, the mirror case.
    assert R.record_findings(TEST_REPO, 12, "sybil", 0, head, [_finding("e")]) == 1

    index = R.lanes(repo=TEST_REPO)
    assert index.total == 3
    by_lane = {(lane.pr, lane.seat): lane for lane in index.lanes}
    assert set(by_lane) == {(10, "henry"), (11, "gray"), (12, "sybil")}

    a = by_lane[(10, "henry")]
    assert a.latest_round == 1
    assert a.counts == {"open": 2, "fixed": 1, "rejected": 1, "stale": 0}
    assert (a.eligible, a.carried, a.settled) == (3, 1, 2)
    assert a.carry_forward_rate == 1 / 3 and a.settle_rate == 2 / 3
    assert a.offerable == 1  # the 200-day-old open row is not offerable...
    assert a.counts["open"] == 2  # ...and is still counted and still shown
    assert a.never_offered == 0
    assert (a.coverage_total, a.coverage_live) == (2, 1)
    assert a.last_activity is not None

    b = by_lane[(11, "gray")]
    assert b.counts == dict.fromkeys(R.STATUS_ORDER, 0)
    assert (b.coverage_total, b.coverage_live) == (1, 1)
    # One round, so nothing a later round could have acted on: a rate here would
    # be a seat that settles nothing rather than a seat with nothing to settle.
    assert b.carry_forward_rate is None and b.settle_rate is None

    assert by_lane[(12, "sybil")].coverage_total == 0

    # The optional filters are evaluated by the server, one cast per parameter.
    assert R.lanes(repo=TEST_REPO, pr=10).total == 1
    assert R.lanes(repo=TEST_REPO, seat="gray").total == 1
    assert R.lanes(repo=TEST_REPO, pr=10, seat="gray").total == 0
    # `count(*) OVER ()` is computed after WHERE and before LIMIT, which is what
    # lets one page report how many lanes it is a page OF.
    page = R.lanes(repo=TEST_REPO, limit=1)
    assert len(page.lanes) == 1 and page.total == 3

    history = R.pr_findings(TEST_REPO, 10)
    assert history.total == 4
    assert history.rows[0].round == 1  # newest round first
    assert {row.status for row in history.rows} == {"open", "fixed", "rejected"}
    assert {row.status_reason for row in history.rows} >= {"patched", "premise is wrong"}
    assert R.pr_findings(TEST_REPO, 10, seat="gray").total == 0
    cut = R.pr_findings(TEST_REPO, 10, limit=1)
    assert len(cut.rows) == 1 and cut.total == 4

    examined = R.pr_coverage(TEST_REPO, 10)
    assert examined.total == 2
    # The expired row is projected rather than filtered out -- the opposite of
    # `live_coverage`, which must never hand a round a dead assurance.
    assert sorted(row.live for row in examined.rows) == [False, True]
    assert [row.expired_at for row in examined.rows if not row.live] != [None]
    assert R.pr_coverage(TEST_REPO, 10, seat="gray").total == 0

    # An offset past the end is an empty page with no count to read, not a
    # count with no rows.
    beyond = R.pr_findings(TEST_REPO, 10, offset=99)
    assert beyond.rows == () and beyond.total == 0


def test_digest_source_is_accepted_and_gated(monkeypatch):
    """The CHECK constraint admits 'digest' and retrieval hides it while dark (#158)."""
    from sidecar import ingest as I
    from sidecar import retrieve
    from sidecar.digest import DIGEST_SOURCE, build_item
    from sidecar.models import IngestItem

    path = "src/big/module.py"
    body = "".join(f"def helper_{i}(x):\n    return x\n" for i in range(40))
    inserted, _ = I.ingest(
        TEST_REPO,
        [
            IngestItem(
                text="Prefer absolute imports in this module.",
                source="remember",
                file_globs=[path],
            ),
            build_item(path, body),
        ],
    )
    assert inserted == 2

    monkeypatch.setattr(settings, "digest_retrieval", False)
    dark = retrieve.query(TEST_REPO, [path], query_text="imports")
    assert dark and all(r["source"] != DIGEST_SOURCE for r in dark)

    monkeypatch.setattr(settings, "digest_retrieval", True)
    lit = retrieve.query(TEST_REPO, [path], query_text="imports")
    assert any(r["source"] == DIGEST_SOURCE for r in lit)


def test_digest_is_not_surfaced_for_a_pr_that_does_not_touch_the_file(monkeypatch):
    """file_globs scoping is what keeps an index off unrelated pull requests (#158)."""
    from sidecar import ingest as I
    from sidecar import retrieve
    from sidecar.digest import DIGEST_SOURCE, build_item

    path = "src/big/module.py"
    body = "".join(f"def helper_{i}(x):\n    return x\n" for i in range(40))
    I.ingest(TEST_REPO, [build_item(path, body)])

    monkeypatch.setattr(settings, "digest_retrieval", True)
    other = retrieve.query(TEST_REPO, ["docs/readme.md"], query_text="helper")
    assert all(r["source"] != DIGEST_SOURCE for r in other)


def test_migrations_replay_cleanly_with_a_digest_row_present():
    """A restart re-runs every migration; a stored digest must not break that (#158)."""
    from sidecar import ingest as I
    from sidecar.db import _migration_sql, _resolve_embed_dim, db
    from sidecar.digest import build_item

    body = "".join(f"def helper_{i}(x):\n    return x\n" for i in range(40))
    I.ingest(TEST_REPO, [build_item("src/big/module.py", body)])

    # Exactly what get_pool() does on the next process start.
    with db() as conn:
        for stmt in _migration_sql(_resolve_embed_dim()[0]):
            conn.execute(stmt)
