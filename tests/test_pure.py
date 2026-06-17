"""Unit tests for pure logic (no database or embeddings backend required)."""

from sidecar.cli import _collect_files, chunk_markdown, format_extra_instructions
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
