"""Unit tests for pure logic (no database or embeddings backend required)."""

import pytest
from pydantic import ValidationError

from sidecar.cli import (
    _collect_files,
    _configured_seat_labels,
    chunk_markdown,
    format_extra_instructions,
)
from sidecar.config import Settings, settings
from sidecar.db import vector_literal
from sidecar.embed import _fit
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


def test_build_query_keeps_the_files_block_within_the_transport_cap():
    # A body at its own cap plus a long file list used to overrun
    # embed_max_chars, so the transport cut in embed._fit ate the tail of the
    # files block -- the part the ordering exists to protect. Asserted on the
    # PREFIXED string, because that is what _fit actually sees: the query
    # instruction is prepended after this budget is spent, so a budget that
    # ignored it would push the assembled query past the cap by exactly the
    # prefix's length and put the cut back in the files block.
    files = [f"packages/shared/src/module_{i:04d}/index.ts" for i in range(300)]
    q = _build_query(files, "log line\n" * 5000, None)
    sent = settings.embed_query_prefix + q
    assert len(sent) <= settings.embed_max_chars
    assert sent == _fit(sent)
    assert "Changed files:" in q
    # Whatever paths survive are whole paths, never a half-written one.
    kept = q.split("Changed files:\n", 1)[1].split("\n")
    assert kept and all(f in files for f in kept)


def test_build_query_never_cuts_a_path_in_half(monkeypatch):
    files = ["aaaaaaaaaaaaaaaaaaaa.py", "bbbbbbbbbbbbbbbbbbbb.py"]
    monkeypatch.setattr(settings, "embed_query_prefix", "")
    monkeypatch.setattr(settings, "embed_max_chars", 40)
    assert _build_query(files, None, None) == "Changed files:\naaaaaaaaaaaaaaaaaaaa.py"
    # A budget that ends exactly on a path boundary keeps that path: rewinding
    # to the previous newline here would drop a whole path for nothing, and
    # with no body to take the freed room the query would come out empty.
    monkeypatch.setattr(settings, "embed_max_chars", 38)
    assert _build_query(files, None, None) == "Changed files:\naaaaaaaaaaaaaaaaaaaa.py"


def test_build_query_cuts_the_body_not_the_files_block(monkeypatch):
    # The body absorbs the whole shortfall; the files block is kept intact and
    # is dropped only when there is no room for even one whole path.
    monkeypatch.setattr(settings, "embed_query_prefix", "")
    monkeypatch.setattr(settings, "embed_max_chars", 30)
    q = _build_query(["a.py"], "b" * 500, None)
    assert q.endswith("Changed files:\na.py")
    assert len(q) <= 30
    monkeypatch.setattr(settings, "embed_max_chars", 19)
    assert _build_query(["a.py"], "b" * 500, None) == "Changed files:\na.py"


def test_the_query_prefix_is_charged_to_the_budget(monkeypatch):
    # The prefix is paid for here or it is paid for by _fit, and _fit takes it
    # out of the tail -- the files block. So a longer instruction has to cost
    # the files block its place in the budget, not cost it its last paths after
    # the fact.
    monkeypatch.setattr(settings, "embed_max_chars", 29)
    monkeypatch.setattr(settings, "embed_query_prefix", "")
    assert _build_query(["a.py"], "b" * 500, None).endswith("Changed files:\na.py")

    # 29 - 10 leaves exactly the 19-char files block and no room for the body.
    monkeypatch.setattr(settings, "embed_query_prefix", "P" * 10)
    q = _build_query(["a.py"], "b" * 500, None)
    assert q == "Changed files:\na.py"
    assert len(settings.embed_query_prefix + q) <= settings.embed_max_chars

    # One more character of instruction and the block no longer fits whole, so
    # it is dropped rather than half-written.
    monkeypatch.setattr(settings, "embed_query_prefix", "P" * 11)
    assert "Changed files:" not in _build_query(["a.py"], "b" * 500, None)


def test_embed_max_chars_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(embed_max_chars=0)
    with pytest.raises(ValidationError):
        Settings(embed_max_chars=-1)


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
