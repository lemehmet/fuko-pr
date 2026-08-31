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
    assert R.touch_findings([s.id for s in stored]) == 2
    settled = next(s for s in stored if s.prior.title == "a")
    assert R.transition(settled.id, "fixed", "rewritten in this head") is True
    assert R.transition(settled.id, "fixed", "replayed stale id") is False

    assert [s.prior.title for s in R.open_findings(TEST_REPO, 1, "henry").rows] == ["b"]

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
