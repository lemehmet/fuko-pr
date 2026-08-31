"""Tests for the `fuko digest` command and its supersede-on-write step."""

from argparse import Namespace

from sidecar import cli, digest
from tests.fakes import FakeStore

REPO = "owner/name"


def _args(tmp_path, **over):
    base = {
        "paths": [str(tmp_path)],
        "repo": REPO,
        "min_bytes": 100,
        "max_chars": digest.MAX_CHARS,
        "dry_run": False,
        "config": ".fuko.toml",
    }
    base.update(over)
    return Namespace(**base)


def _big(tmp_path, name, decls=60):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"pub fn f{i}() {{ /* padding padding */ }}\n" for i in range(decls)))
    return p


class _DedupingStore(FakeStore):
    """FakeStore plus the real stores' (repo, text, source) uniqueness.

    Supersession is only meaningful against a store that actually skips an
    unchanged re-ingest, which is what both real backends do.
    """

    def ingest(self, repo, items, *, max_new=None):
        fresh = [
            i
            for i in items
            if not any(
                r["repo"] == repo and r["text"] == i.text and r["source"] == i.source
                for r in self.items
            )
        ]
        if fresh:
            super().ingest(repo, fresh)
        return len(fresh), len(items) - len(fresh)


def _use(monkeypatch, store=None):
    store = store or _DedupingStore()
    monkeypatch.setattr(cli, "_store", lambda _config: store)
    return store


def test_candidates_skip_files_below_the_floor(tmp_path):
    _big(tmp_path, "big.rs")
    (tmp_path / "tiny.rs").write_text("fn a() {}\n")
    found = cli._digest_candidates([str(tmp_path)], 100)
    assert [p for p in found if p.endswith("big.rs")]
    assert not [p for p in found if p.endswith("tiny.rs")]


def test_candidates_skip_vendored_and_generated_trees(tmp_path):
    _big(tmp_path, "src/real.rs")
    _big(tmp_path, "node_modules/dep/index.js")
    _big(tmp_path, "target/debug/gen.rs")
    found = cli._digest_candidates([str(tmp_path)], 100)
    assert len(found) == 1 and found[0].endswith("real.rs")


def test_candidates_survive_a_path_that_vanished_after_collection(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_collect_files", lambda _patterns: ["no/such/file.rs"])
    assert cli._digest_candidates(["."], 100) == []
    assert "could not stat" in capsys.readouterr().err


def test_dry_run_prints_the_index_and_stores_nothing(tmp_path, monkeypatch, capsys):
    _big(tmp_path, "big.rs")
    store = _use(monkeypatch)
    cli._cmd_digest(_args(tmp_path, dry_run=True))
    out = capsys.readouterr()
    assert "Structural index of" in out.out
    assert "nothing stored" in out.err
    assert store.items == []


def test_no_candidates_reports_and_stores_nothing(tmp_path, monkeypatch, capsys):
    (tmp_path / "tiny.rs").write_text("fn a() {}\n")
    store = _use(monkeypatch)
    cli._cmd_digest(_args(tmp_path))
    assert "nothing to index" in capsys.readouterr().err
    assert store.items == []


def test_binary_files_are_skipped_with_a_warning(tmp_path, monkeypatch, capsys):
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe" + b"\x00" * 400)
    store = _use(monkeypatch)
    cli._cmd_digest(_args(tmp_path))
    err = capsys.readouterr().err
    assert "could not read" in err and "nothing to index" in err
    assert store.items == []


def test_digest_is_stored_scoped_to_its_own_path(tmp_path, monkeypatch, capsys):
    path = _big(tmp_path, "big.rs")
    store = _use(monkeypatch)
    cli._cmd_digest(_args(tmp_path))
    assert "indexed 1 file(s): 1 new" in capsys.readouterr().out
    (row,) = store.items
    assert row["source"] == digest.DIGEST_SOURCE
    assert row["file_globs"] == [str(path)]
    assert digest.topic_path(row["topic"]) == str(path)


def test_a_changed_file_supersedes_its_previous_index(tmp_path, monkeypatch, capsys):
    path = _big(tmp_path, "big.rs")
    store = _use(monkeypatch)
    cli._cmd_digest(_args(tmp_path))
    stale_topic = store.items[0]["topic"]

    path.write_text(path.read_text() + "pub fn extra() { /* padding padding */ }\n")
    cli._cmd_digest(_args(tmp_path))

    assert "1 superseded" in capsys.readouterr().out
    topics = [i["topic"] for i in store.items]
    assert stale_topic not in topics
    assert len(topics) == 1


def test_an_unchanged_file_supersedes_nothing(tmp_path, monkeypatch, capsys):
    _big(tmp_path, "big.rs")
    store = _use(monkeypatch)
    cli._cmd_digest(_args(tmp_path))
    capsys.readouterr()
    cli._cmd_digest(_args(tmp_path))
    out = capsys.readouterr().out
    assert "0 superseded" in out
    assert len(store.items) == 1


def test_supersede_leaves_other_files_and_other_sources_alone(tmp_path, monkeypatch):
    path = _big(tmp_path, "big.rs")
    other = _big(tmp_path, "other.rs")
    store = _use(monkeypatch)
    cli._cmd_digest(_args(tmp_path))
    store.items.append(
        {
            "repo": REPO,
            "id": "keepme",
            "text": "a convention",
            "source": "review_thread",
            "topic": "review decision",
            "file_globs": [str(path)],
        }
    )

    path.write_text(path.read_text() + "pub fn extra() { /* padding padding */ }\n")
    cli._cmd_digest(_args(tmp_path))

    kept = {i["id"] for i in store.items}
    assert "keepme" in kept
    assert len([i for i in store.items if i.get("file_globs") == [str(other)]]) == 1


def test_supersede_pages_through_a_long_digest_backlog(tmp_path, monkeypatch):
    path = _big(tmp_path, "big.rs")
    store = _use(monkeypatch)
    for n in range(250):
        store.items.append(
            {
                "repo": REPO,
                "id": f"filler-{n}",
                "text": "x",
                "source": digest.DIGEST_SOURCE,
                "topic": digest.topic_for(f"other/{n}.rs", "0" * 12),
                "file_globs": [f"other/{n}.rs"],
            }
        )
    store.items.append(
        {
            "repo": REPO,
            "id": "old",
            "text": "x",
            "source": digest.DIGEST_SOURCE,
            "topic": digest.topic_for(str(path), "f" * 12),
            "file_globs": [str(path)],
        }
    )
    cli._cmd_digest(_args(tmp_path))
    assert "old" not in {i["id"] for i in store.items}
    assert len([i for i in store.items if i["id"].startswith("filler-")]) == 250


def test_absolute_paths_are_warned_about_as_unreachable(tmp_path, monkeypatch, capsys):
    _big(tmp_path, "big.rs")
    _use(monkeypatch)
    cli._cmd_digest(_args(tmp_path))
    assert "can never match a pull request" in capsys.readouterr().err


def test_relative_paths_draw_no_warning(tmp_path, monkeypatch, capsys):
    _big(tmp_path, "big.rs")
    _use(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cli._cmd_digest(_args(tmp_path, paths=["."]))
    assert "can never match" not in capsys.readouterr().err
