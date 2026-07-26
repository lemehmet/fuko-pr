"""Split markdown documents into learning-sized chunks carrying their heading.

Used by every doc-ingestion path -- ``fuko ingest-docs`` and the knowledge-base
console's upload -- so both produce identical chunks from the same file. Pure
string handling, no I/O.

Nothing is discarded: a paragraph longer than ``max_len`` is emitted as
successive full-length chunks rather than truncated, so ingesting a document
cannot silently lose part of it.
"""

import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*)")


def _split_paragraphs(body: str, max_len: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", body):
        if not cur or len(cur) + len(para) + 2 <= max_len:
            cur = (cur + "\n\n" + para) if cur else para
        else:
            out.append(cur)
            cur = para
        while len(cur) > max_len:
            out.append(cur[:max_len])
            cur = cur[max_len:]
    if cur:
        out.append(cur)
    return out or [body[:max_len]]


def chunk_markdown(text: str, max_len: int = 1500) -> list[tuple[str, str]]:
    """Split ``text`` into ``(chunk, heading)`` pairs, capping each chunk near ``max_len``."""
    chunks: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        for part in _split_paragraphs(body, max_len):
            chunks.append((part, heading))

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            heading = m.group(2).strip()
            buf = [line]
        else:
            buf.append(line)
    flush()
    return chunks or [(text.strip()[:max_len], "")]
