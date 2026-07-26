from sidecar.chunking import _split_paragraphs, chunk_markdown
from sidecar.cli import format_extra_instructions


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


def test_split_paragraphs_keeps_every_character_of_an_oversized_paragraph():
    body = "".join(str(i % 10) for i in range(4000))
    parts = _split_paragraphs(body, 1500)
    assert "".join(parts) == body
    assert all(len(p) <= 1500 for p in parts)


def test_split_paragraphs_carries_the_tail_into_the_next_paragraph():
    long_para = "a" * 1600
    body = "\n\n".join([long_para, "short tail paragraph"])
    parts = _split_paragraphs(body, 1500)
    assert "".join(parts).replace("\n\n", "") == (long_para + "short tail paragraph")
    assert "short tail paragraph" in parts[-1]


def test_chunk_markdown_does_not_drop_an_oversized_section():
    body = "b" * 3000
    chunks = chunk_markdown(f"# Title\n\n{body}", max_len=1000)
    assert body in "".join(text for text, _ in chunks)


def test_chunk_markdown_emits_nothing_for_empty_input():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n \t ") == []
    assert chunk_markdown("real content") != []
