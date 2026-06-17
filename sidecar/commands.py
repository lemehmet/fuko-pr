"""Parsing of ``/remember`` and ``/forget`` PR comments."""


def _strip_prefix(text: str, prefix: str) -> str | None:
    lowered = text.strip().lower()
    if not lowered.startswith(prefix):
        return None
    rest = text.strip()[len(prefix) :].strip()
    if rest.startswith(":") or rest.startswith("-"):
        rest = rest[1:].strip()
    return rest


def parse_remember(body: str) -> tuple[str, list[str]] | None:
    """Parse a ``/remember ...`` comment into ``(text, file_globs)``.

    An optional trailing ``paths: a/**, b.py`` line sets the file globs and is
    removed from the stored text. Returns ``None`` if the comment is not a
    ``/remember`` command or contains no text.
    """
    rest = _strip_prefix(body, "/remember")
    if rest is None:
        return None
    lines = rest.splitlines()
    globs: list[str] = []
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("paths:"):
            raw = line.split(":", 1)[1]
            globs = [g.strip() for g in raw.split(",") if g.strip()]
            del lines[i]
            break
    text = "\n".join(lines).strip()
    if not text:
        return None
    return text, globs


def parse_forget(body: str) -> dict | None:
    """Parse a ``/forget ...`` comment into a criteria dict.

    Supported forms: ``/forget all``, ``/forget source=<src>``, ``/forget <id>``.
    Returns ``None`` if the comment is not a ``/forget`` command or has no argument.
    """
    rest = _strip_prefix(body, "/forget")
    if not rest:
        return None
    lowered = rest.lower()
    if lowered == "all":
        return {"all": True}
    if lowered.startswith("source="):
        return {"source": rest.split("=", 1)[1].strip()}
    return {"id": rest.strip()}
