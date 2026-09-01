"""Unit tests for pure logic (no database or embeddings backend required)."""

from sidecar.cli import (
    _collect_files,
    _configured_seat_labels,
    chunk_markdown,
    format_extra_instructions,
)
from sidecar.db import vector_literal
from sidecar.ingest import _parse_dt
from sidecar.models import ForgetRequest, IngestItem, IngestRequest, QueryRequest
from sidecar.retrieve import _build_query


def test_vector_literal_format():
    s = vector_literal([1.0, 2.5, 3.0])
    assert s.startswith("[") and s.endswith("]")
    assert "1.0" in s and "2.5" in s and "3.0" in s


def test_parse_dt_variants():
    assert _parse_dt(None) is None
    assert _parse_dt("nope") is None
    assert _parse_dt("2024-01-02T03:04:05Z").year == 2024


def test_build_query_combines_parts():
    assert _build_query([], None, None) == ""
    q = _build_query(["a.py", "b.py"], "fix login", "remember X")
    assert "fix login" in q
    assert "remember X" in q
    assert "a.py" in q and "b.py" in q


def test_build_query_bounds_the_pr_body_and_keeps_the_files_block():
    body = "log line\n" * 5000
    q = _build_query(["a.py"], body, "remember X")
    assert len(q) < len(body)
    # The files block is what a tail-truncation of the assembled query would
    # have eaten; it has to survive a body that overflows the budget.
    assert q.startswith("remember X")
    assert q.endswith("Changed files:\na.py")


def test_models_defaults():
    it = IngestItem(text="t", source="docs")
    assert it.file_globs == [] and it.source_url is None and it.origin_user is None
    qr = QueryRequest(repo="r")
    assert qr.files == [] and qr.top_k is None
    ir = IngestRequest(repo="r", items=[it])
    assert len(ir.items) == 1
    fg = ForgetRequest(repo="r", all=True)
    assert fg.all is True


def test_collect_files_skips_missing(tmp_path, capsys):
    f = tmp_path / "a.md"
    f.write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("y")

    res = _collect_files([str(f), str(sub), "nope/*.md"])
    assert str(f) in res
    assert any(p.endswith("b.txt") for p in res)
    assert not any("nope" in p for p in res)
    assert "nope" in capsys.readouterr().err


def test_format_extra_instructions_with_globs():
    md = format_extra_instructions(
        [
            {
                "text": "rule",
                "source": "remember",
                "source_url": None,
                "file_globs": ["src/**"],
                "topic": None,
                "score": 0.5,
            }
        ]
    )
    assert "rule" in md and "src/**" in md


def test_chunk_markdown_single_when_no_heading():
    assert len(chunk_markdown("plain text only")) == 1


def test_forget_invalid_uuid_is_noop():
    from sidecar.ingest import forget

    assert forget("owner/repo", id="not-a-uuid") == 0
    assert forget("owner/repo", id="/forget all") == 0


def test_configured_seat_labels_from_a_models_config(tmp_path):
    """#116: labels are the `provider/name` of every configured entry, all roles."""
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "openrouter"\n'
        'name = "x-ai/grok-4.5"\n'
        'role = "active"\n'
        "[[review.models]]\n"
        'provider = "openrouter"\n'
        'name = "qwen/qwen3.8-max"\n'
        'role = "backup"\n'
    )
    labels = _configured_seat_labels(str(cfg))
    assert labels == ["openrouter/x-ai/grok-4.5", "openrouter/qwen/qwen3.8-max"]


def test_configured_seat_labels_fails_safe_to_none_when_missing(capsys):
    """A missing config yields None so `fuko_states` keeps every pending row.

    Crucially it must NOT fall back to `load_config`'s built-in defaults, whose
    single entry would wrongly supersede every real seat.
    """
    assert _configured_seat_labels("/no/such/.fuko.toml") is None
    assert "not found" in capsys.readouterr().err


def test_configured_seat_labels_fails_safe_to_none_on_a_malformed_file(tmp_path, capsys):
    """A present-but-unparseable config also fails safe to None."""
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text("this = is = not = valid = toml\n")
    assert _configured_seat_labels(str(cfg)) is None
    assert "could not load" in capsys.readouterr().err


def test_configured_seat_labels_fails_safe_on_a_blank_provider(tmp_path, capsys):
    """A blank provider/name must fail safe to None, not emit a junk `/name` label.

    pydantic types provider/name as `str` but permits `""`, and a junk label that
    matches no real receipt would supersede every genuine seat (a receipt is
    superseded when its label is absent from the configured set) — the exact
    merge-past-unreviewed-seat direction the function fails safe against.
    """
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text('[[review.models]]\nprovider = ""\nname = "x"\nrole = "active"\n')
    assert _configured_seat_labels(str(cfg)) is None
    assert "could not load" in capsys.readouterr().err


def _result(**over):
    base = {
        "text": "rule",
        "source": "remember",
        "source_url": None,
        "file_globs": [],
        "topic": None,
        "score": 0.5,
    }
    base.update(over)
    return base


def test_format_extra_instructions_gives_digests_their_own_section():
    md = format_extra_instructions(
        [
            _result(text="a convention"),
            _result(
                text="Structural index of src/big.rs\nL1-L9  fn a",
                source="digest",
                file_globs=["src/big.rs"],
                topic="file-index:src/big.rs@0011aabbccdd",
            ),
        ]
    )
    assert "## Repository knowledge (from fuko-pr)" in md
    assert "## File structure index (from fuko-pr)" in md
    # The index keeps its own lines rather than being flattened into a bullet.
    assert "\nL1-L9  fn a" in md
    assert "- Structural index" not in md


def test_format_extra_instructions_does_not_present_a_digest_as_a_conclusion():
    md = format_extra_instructions([_result(text="index", source="digest")])
    assert "NOT review conclusions" in md
    assert "Repository knowledge" not in md


def test_format_extra_instructions_unchanged_without_digests():
    md = format_extra_instructions([_result(text="rule", file_globs=["src/**"])])
    assert md.startswith("## Repository knowledge (from fuko-pr)")
    assert "File structure index" not in md
    assert md.endswith("\n")
