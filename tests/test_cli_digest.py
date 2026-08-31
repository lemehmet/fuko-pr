"""Tests for the `fuko digest` command and its supersede-on-write step."""

import fnmatch
import hashlib
from argparse import Namespace
from pathlib import Path

from sidecar import cli, digest
from tests.fakes import FakeStore

REPO = "owner/name"


def _args(tmp_path, **over):
    base = {
        "paths": ["."],
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


def _use(monkeypatch, tmp_path, store=None):
    """Point the CLI at a fake store and run the command from inside the checkout.

    `fuko digest` indexes paths relative to the working directory and skips
    anything outside it, so a test that exercises the command has to run from a
    checkout root the way an operator does.
    """
    monkeypatch.chdir(tmp_path)
    store = store or _DedupingStore()
    monkeypatch.setattr(cli, "_store", lambda _config: store)
    return store


def test_candidates_skip_files_below_the_floor(tmp_path):
    _big(tmp_path, "big.rs")
    (tmp_path / "tiny.rs").write_text("fn a() {}\n")
    found = cli._digest_candidates([str(tmp_path)], 100, tmp_path.resolve())
    assert [p for p in found if p.endswith("big.rs")]
    assert not [p for p in found if p.endswith("tiny.rs")]


def test_candidates_skip_vendored_and_generated_trees(tmp_path):
    _big(tmp_path, "src/real.rs")
    _big(tmp_path, "node_modules/dep/index.js")
    _big(tmp_path, "target/debug/gen.rs")
    found = cli._digest_candidates([str(tmp_path)], 100, tmp_path.resolve())
    assert len(found) == 1 and found[0].endswith("real.rs")


def test_candidates_survive_a_path_that_vanished_after_collection(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_collect_files", lambda _patterns: ["no/such/file.rs"])
    assert cli._digest_candidates(["."], 100, Path.cwd().resolve()) == []
    assert "could not stat" in capsys.readouterr().err


def test_dry_run_prints_the_index_and_stores_nothing(tmp_path, monkeypatch, capsys):
    _big(tmp_path, "big.rs")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path, dry_run=True))
    out = capsys.readouterr()
    assert "Structural index of" in out.out
    assert "nothing stored" in out.err
    assert store.items == []


def test_no_candidates_reports_and_stores_nothing(tmp_path, monkeypatch, capsys):
    (tmp_path / "tiny.rs").write_text("fn a() {}\n")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    assert "nothing to index" in capsys.readouterr().err
    assert store.items == []


def test_binary_files_are_skipped_with_a_warning(tmp_path, monkeypatch, capsys):
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe" + b"\x00" * 400)
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    err = capsys.readouterr().err
    assert "could not read" in err and "nothing to index" in err
    assert store.items == []


def test_digest_is_stored_scoped_to_its_own_path(tmp_path, monkeypatch, capsys):
    _big(tmp_path, "big.rs")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    assert "indexed 1 file(s): 1 new" in capsys.readouterr().out
    (row,) = store.items
    assert row["source"] == digest.DIGEST_SOURCE
    assert row["file_globs"] == ["big.rs"]
    assert digest.topic_path(row["topic"]) == "big.rs"


def test_a_changed_file_supersedes_its_previous_index(tmp_path, monkeypatch, capsys):
    path = _big(tmp_path, "big.rs")
    store = _use(monkeypatch, tmp_path)
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
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    capsys.readouterr()
    cli._cmd_digest(_args(tmp_path))
    out = capsys.readouterr().out
    assert "0 superseded" in out
    assert len(store.items) == 1


def test_supersede_leaves_other_files_and_other_sources_alone(tmp_path, monkeypatch):
    path = _big(tmp_path, "big.rs")
    _big(tmp_path, "other.rs")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    store.items.append(
        {
            "repo": REPO,
            "id": "keepme",
            "text": "a convention",
            "source": "review_thread",
            "topic": "review decision",
            "file_globs": ["big.rs"],
        }
    )

    path.write_text(path.read_text() + "pub fn extra() { /* padding padding */ }\n")
    cli._cmd_digest(_args(tmp_path))

    kept = {i["id"] for i in store.items}
    assert "keepme" in kept
    assert len([i for i in store.items if i.get("file_globs") == ["other.rs"]]) == 1


def test_supersede_pages_through_a_long_digest_backlog(tmp_path, monkeypatch):
    _big(tmp_path, "big.rs")
    store = _use(monkeypatch, tmp_path)
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
            "topic": digest.topic_for("big.rs", "f" * 12),
            "file_globs": ["big.rs"],
        }
    )
    cli._cmd_digest(_args(tmp_path))
    assert "old" not in {i["id"] for i in store.items}
    assert len([i for i in store.items if i["id"].startswith("filler-")]) == 250


def test_files_outside_the_checkout_are_skipped_with_a_warning(tmp_path, monkeypatch, capsys):
    """An index of a file the working directory does not contain is unreachable.

    Retrieval matches a stored glob against the repository-relative paths a pull
    request reports, so such a row would embed, cost money, and never surface.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outside = tmp_path / "outside"
    _big(outside, "big.rs")
    store = _use(monkeypatch, checkout)
    cli._cmd_digest(_args(tmp_path, paths=[str(outside)]))
    err = capsys.readouterr().err
    assert "outside the checkout" in err and "nothing to index" in err
    assert store.items == []


def test_relative_paths_draw_no_warning(tmp_path, monkeypatch, capsys):
    _big(tmp_path, "big.rs")
    _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    assert "outside the checkout" not in capsys.readouterr().err


def test_an_absolute_path_inside_the_checkout_is_stored_relative(tmp_path, monkeypatch):
    """Where the operator points at the file is not how the index is keyed."""
    _big(tmp_path, "src/big.rs")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path, paths=[str(tmp_path / "src")]))
    (row,) = store.items
    assert row["file_globs"] == ["src/big.rs"]
    assert digest.topic_path(row["topic"]) == "src/big.rs"


def test_a_path_with_glob_metacharacters_still_matches_itself(tmp_path, monkeypatch):
    """`app/[slug]/page.tsx` is an ordinary route file, not a character class."""
    _big(tmp_path, "app/[slug]/page.tsx")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    (row,) = store.items
    assert fnmatch.fnmatch("app/[slug]/page.tsx", row["file_globs"][0])
    assert digest.topic_path(row["topic"]) == "app/[slug]/page.tsx"


def test_a_metacharacter_path_supersedes_its_own_previous_index(tmp_path, monkeypatch, capsys):
    """Supersession keys on the literal path, so escaping the glob cannot break it."""
    path = _big(tmp_path, "app/[slug]/page.tsx")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    path.write_text(path.read_text() + "pub fn extra() { /* padding padding */ }\n")
    capsys.readouterr()
    cli._cmd_digest(_args(tmp_path))
    assert "1 superseded" in capsys.readouterr().out
    assert len(store.items) == 1


def test_a_cap_too_small_for_the_header_skips_the_file(tmp_path, monkeypatch, capsys):
    """A cap the index cannot honour is refused, not silently overshot."""
    _big(tmp_path, "big.rs")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path, max_chars=10))
    err = capsys.readouterr().err
    assert "cannot index big.rs" in err and "nothing to index" in err
    assert store.items == []


def test_a_checkout_under_a_skip_named_directory_is_still_indexed(tmp_path, monkeypatch):
    """The skip set names directories inside a repository, not above it.

    A checkout at `/srv/build/myrepo` is plausible on a CI host, and matching the
    skip set against a candidate's absolute ancestors dropped every file in it --
    before the outside-checkout warning could say anything, so the command
    reported "nothing to index", which was false and pointed at the wrong cause.
    """
    checkout = tmp_path / "build" / "myrepo"
    _big(checkout, "src/big.rs")
    _big(checkout, "node_modules/dep/index.js")
    store = _use(monkeypatch, checkout)
    cli._cmd_digest(_args(tmp_path, paths=[str(checkout)]))
    assert [row["file_globs"] for row in store.items] == [["src/big.rs"]]


def test_the_rendered_blob_hash_is_reproducible_from_the_file_on_disk(tmp_path, monkeypatch):
    """The index invites the reader to check it, so `sha256sum` has to agree.

    Reading in text mode normalised newlines, so a CRLF file's hash and size
    described content that exists nowhere on disk.
    """
    path = tmp_path / "big.rs"
    path.write_bytes(b"".join(b"pub fn f%d() { /* padding padding */ }\r\n" % i for i in range(60)))
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    (row,) = store.items
    on_disk = path.read_bytes()
    assert f"blob sha256:{hashlib.sha256(on_disk).hexdigest()[:12]}" in row["text"]
    assert f"{len(on_disk) / 1024:.1f} KB" in row["text"]


def test_a_rerender_of_an_unchanged_blob_replaces_its_index(tmp_path, monkeypatch, capsys):
    """One index row per path, even when the blob is identical.

    The topic is `<path>@<blob hash>` and carries nothing about rendering, so a
    re-run with a different `--max-chars` produced text that ingest saw as new
    beside a row supersession saw as current — two rows for one file, which no
    later run could collapse.
    """
    _big(tmp_path, "big.rs")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    capsys.readouterr()

    cli._cmd_digest(_args(tmp_path, max_chars=1200))

    assert "1 superseded" in capsys.readouterr().out
    assert len(store.items) == 1
    assert len(store.items[0]["text"]) <= 1200


def test_a_working_directory_without_a_git_dir_is_called_out(tmp_path, monkeypatch, capsys):
    """Running from a subdirectory is the silent half of the unreachable-path bug."""
    _big(tmp_path, "big.rs")
    _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    assert "may not be the checkout root" in capsys.readouterr().err


def test_a_checkout_root_draws_no_layout_warning(tmp_path, monkeypatch, capsys):
    _big(tmp_path, "big.rs")
    (tmp_path / ".git").mkdir()
    _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    assert "may not be the checkout root" not in capsys.readouterr().err


def test_a_file_with_no_declarations_is_reported_and_not_stored(tmp_path, monkeypatch, capsys):
    """A lockfile-shaped index has nothing to navigate to, so it is not worth a slot."""
    (tmp_path / "pnpm-lock.yaml").write_text("a: 1\nb: 2\n" * 200)
    _big(tmp_path, "big.rs")
    store = _use(monkeypatch, tmp_path)
    cli._cmd_digest(_args(tmp_path))
    assert "no recognised declarations" in capsys.readouterr().err
    assert [row["file_globs"] for row in store.items] == [["big.rs"]]
