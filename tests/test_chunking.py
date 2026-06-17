from sidecar.cli import _split_paragraphs, chunk_markdown, format_extra_instructions


def test_chunk_markdown_splits_on_headings():
    md = "# Title\n\nintro\n\n## Section A\nbody A\n## Section B\nbody B"
    topics = [t for _, t in chunk_markdown(md)]
    assert "Title" in topics
    assert "Section A" in topics
    assert "Section B" in topics


def test_chunk_markdown_no_headings():
    chunks = chunk_markdown("just a paragraph of text")
    assert len(chunks) == 1


def test_split_paragraphs_caps_length():
    para = "x" * 600
    body = "\n\n".join([para, para, para])
    parts = _split_paragraphs(body, 1000)
    assert len(parts) >= 2


def test_format_extra_instructions_empty():
    assert format_extra_instructions([]) == ""


def test_format_extra_instructions_with_items():
    results = [
        {
            "text": "do the thing",
            "source": "remember",
            "source_url": "http://x/1",
            "file_globs": [],
            "topic": None,
            "score": 0.9,
        }
    ]
    md = format_extra_instructions(results)
    assert "do the thing" in md
    assert "http://x/1" in md
