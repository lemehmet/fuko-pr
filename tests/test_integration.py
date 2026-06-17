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


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    from sidecar.db import db

    with db() as conn:
        conn.execute("DELETE FROM learnings WHERE repo = %s", (TEST_REPO,))


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

    client = TestClient(app)
    assert client.get("/healthz").json() == {"ok": True}

    r = client.post("/query", json={"repo": TEST_REPO, "files": ["src/x.py"]})
    assert r.status_code == 200
    assert "results" in r.json()

    f = client.post("/forget", json={"repo": TEST_REPO, "all": True})
    assert f.status_code == 200
    assert f.json()["deleted"] >= 0


def test_ingest_threads_mines_resolved():
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
                    {"author": {"login": "bob"}, "body": "we use pattern Z", "url": "u2"},
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
    client = TestClient(app)
    r = client.post("/ingest-threads", json={"repo": TEST_REPO, "threads": threads})
    assert r.status_code == 200
    assert r.json()["considered"] == 2
    assert r.json()["inserted"] == 1
    assert any("pattern Z" in x["text"] for x in retrieve.query(TEST_REPO, ["src/a.py"]))


def test_comment_remember_and_forget():
    from fastapi.testclient import TestClient

    from sidecar import retrieve
    from sidecar.main import app

    client = TestClient(app)
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
