"""Mechanically derived structural indexes of large repository files (#158).

A *digest* is a map of one file: which named things it declares and at which
line ranges, plus the file's size and the hash of the exact blob the map was
built from. Nothing else. It exists so a reviewer facing a 400 KB source file
can read the two hundred lines it actually needs instead of the whole file --
the measurement behind #158 found one 428 KB file read 182 times across 24
seat-runs, at ~122k tokens a read.

Two properties are load-bearing and are why this module is deterministic rather
than a model call:

* **It cannot carry a clean verdict.** The epic (#160) prohibits storing "module
  X is sound", because a round-1 mistake would become a permanent blind spot. A
  digest built by extracting identifiers and line numbers has no channel through
  which an assessment could enter. That is a structural guarantee, not a filter
  that has to hold. Doc comments are deliberately *not* extracted for the same
  reason: they are prose, and prose is where a verdict would hide.
* **It does not couple two seats' judgement.** #160 rejects a shared cross-seat
  ledger because it would share conclusions, and the second seat exists to be an
  independent opinion. A digest is repo-level, so that argument has to be met
  head-on: what a digest shares is a lossy compression of the source file, which
  both seats already read independently. A model-generated summary would share
  one model's *reading* of the file and would sit on the wrong side of that line;
  a symbol table does not.

Scanning is best-effort by design. Python goes through :mod:`ast`; everything
else goes through a language-agnostic declaration regex whose line ranges are
approximations (a declaration runs until the next one at the same or shallower
indentation). The rendered header names which scanner ran, and a truncated index
says so in the text, because an index that silently omits a symbol would be read
as evidence the symbol does not exist.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import re
from dataclasses import dataclass

from .models import IngestItem

DIGEST_SOURCE = "digest"
"""The ``learnings.source`` value digests are stored under."""

MIN_BYTES = 65_536
"""Default size floor for digesting a file.

A digest earns its keep only when reading the file is expensive. Below this the
agent can just read the file, and an index would be pure added cost.
"""

MAX_CHARS = 6_000
"""Default cap on a rendered digest.

Bounds three things at once: the embedding call (a digest is one learning, one
embed, and the default ``bge-m3`` has a finite context), the share of the
``top_k`` knowledge budget one digest can take, and the prompt itself.
"""

_TAIL_RESERVE = 200
"""Characters held back from ``max_chars`` for whichever trailing note applies.

Every rendered index ends in at most one note -- either "no declarations
recognised" or "N shorter declarations omitted" -- and both of those say
something a reader must not miss, so they are reserved for rather than
truncated. This reserve is also what makes the cap in :func:`render` a cap on
the *whole* result: a limit too small to hold the header plus this reserve is
rejected instead of quietly overshot.
"""

_NO_DECLARATIONS = "(no declarations recognised in this file)"

_TOPIC_PREFIX = "file-index:"

_DECL_KEYWORDS = (
    "fn",
    "func",
    "function",
    "def",
    "class",
    "struct",
    "enum",
    "trait",
    "impl",
    "interface",
    "type",
    "mod",
    "module",
    "namespace",
    "package",
    "record",
    "union",
    "protocol",
    "extension",
    "const",
    "static",
)

_DECL_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?:(?:pub|public|private|protected|internal|export|default|open|final|abstract|"
    r"async|unsafe|extern|inline|virtual|override|sealed|partial|declare)\b"
    r"(?:\([^)\n]*\))?[ \t]+)*"
    r"(?P<kind>" + "|".join(_DECL_KEYWORDS) + r")\b[ \t]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)

_MAX_DECL_INDENT = 4
"""Deepest indentation a regex-scanned declaration may sit at to be indexed.

Deeply nested declarations are local detail; the index is for finding the region
to read, and listing every closure would bury the regions under them.
"""


@dataclass(frozen=True)
class Symbol:
    """One named declaration and the line range it spans.

    Attributes:
        kind: The declaring keyword (``fn``, ``class``, ``struct``, ...).
        name: The declared identifier.
        start: 1-based first line of the declaration.
        end: 1-based last line, inclusive. Approximate outside Python.
    """

    kind: str
    name: str
    start: int
    end: int

    @property
    def span(self) -> int:
        """Return how many lines the declaration covers."""
        return max(1, self.end - self.start + 1)


def blob_hash(text: str) -> str:
    """Return the short content hash identifying the exact blob a digest describes.

    Digests are keyed on this: a file edit produces a different hash, which is
    what makes a stale digest detectable rather than silently wrong. It is
    rendered into the digest body too, so a reader can tell whether the index
    matches the checkout in front of them.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def topic_for(path: str, digest_hash: str) -> str:
    """Return the ``topic`` a digest row carries: its path and the blob it describes."""
    return f"{_TOPIC_PREFIX}{path}@{digest_hash}"


def topic_path(topic: str | None) -> str | None:
    """Return the path encoded in a digest ``topic``, or ``None`` if it is not one.

    Supersession needs to recognise *this file's* previous digest without
    trusting ``file_globs``, which an operator can edit through the console.
    """
    if not topic or not topic.startswith(_TOPIC_PREFIX):
        return None
    rest = topic[len(_TOPIC_PREFIX) :]
    path, _, _hash = rest.rpartition("@")
    return path or None


def _scan_python(text: str) -> list[Symbol] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    out: list[Symbol] = []

    def visit(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                kind = "class" if isinstance(child, ast.ClassDef) else "def"
                end = getattr(child, "end_lineno", None) or child.lineno
                out.append(Symbol(kind, child.name, child.lineno, end))
                if depth < 1:
                    visit(child, depth + 1)

    visit(tree, 0)
    out.sort(key=lambda s: s.start)
    return out


def _indent_width(indent: str) -> int:
    return len(indent.replace("\t", "    "))


def _scan_declarations(text: str) -> list[Symbol]:
    hits: list[tuple[int, int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _DECL_RE.match(line)
        if not m:
            continue
        width = _indent_width(m.group("indent"))
        if width > _MAX_DECL_INDENT:
            continue
        hits.append((lineno, width, m.group("kind"), m.group("name")))

    total = len(text.splitlines())
    out: list[Symbol] = []
    for i, (lineno, width, kind, name) in enumerate(hits):
        # A regex cannot see block structure, so a declaration is taken to run
        # until the next one that is not nested inside it. Wrong in the details,
        # right about which region of the file to open -- which is all the index
        # is for.
        end = total
        for later_line, later_width, _k, _n in hits[i + 1 :]:
            if later_width <= width:
                end = max(lineno, later_line - 1)
                break
        out.append(Symbol(kind, name, lineno, end))
    return out


def scan(path: str, text: str) -> tuple[list[Symbol], str]:
    """Return the declarations found in ``text`` and the name of the scanner used.

    Args:
        path: Repository-relative path, used only to pick a scanner.
        text: Full file contents.

    Returns:
        A ``(symbols, scanner)`` tuple. ``scanner`` is reported in the rendered
        digest so a reader knows whether the line ranges are exact (``ast``) or
        approximate (``declarations``).
    """
    if path.endswith((".py", ".pyi")):
        parsed = _scan_python(text)
        if parsed is not None:
            return parsed, "ast"
    return _scan_declarations(text), "declarations"


def _header(path: str, text: str, scanner: str) -> list[str]:
    size = len(text.encode("utf-8", errors="replace"))
    return [
        f"Structural index of {path}",
        f"blob {blob_hash(text)} | {size / 1024:.1f} KB | "
        f"{len(text.splitlines())} lines | scanner: {scanner}",
        "This index lists only WHERE declarations are, so a targeted read can "
        "replace a whole-file read. It is not a review, states nothing about "
        "whether any of this code is correct, and the absence of an entry is "
        "not evidence that something is absent.",
        "",
    ]


def _entry(symbol: Symbol) -> str:
    return f"L{symbol.start}-L{symbol.end}  {symbol.kind} {symbol.name}"


def _fit(symbols: list[Symbol], budget: int) -> tuple[list[Symbol], int]:
    """Return the symbols that fit in ``budget`` characters, plus how many were dropped.

    The smallest spans go first: the index exists to point at the large regions
    that make a whole-file read expensive, so a five-line helper is the least
    costly thing to lose.
    """
    kept = sorted(symbols, key=lambda s: (-s.span, s.start))
    used = 0
    fitting: list[Symbol] = []
    for symbol in kept:
        cost = len(_entry(symbol)) + 1
        if used + cost > budget:
            break
        used += cost
        fitting.append(symbol)
    fitting.sort(key=lambda s: s.start)
    return fitting, len(symbols) - len(fitting)


def render(path: str, text: str, max_chars: int = MAX_CHARS) -> str:
    """Return the digest body stored as a learning's ``text``.

    Args:
        path: Repository-relative path of the indexed file.
        text: Full file contents.
        max_chars: Cap on the rendered result.

    Returns:
        The rendered index, never longer than ``max_chars``. A file whose
        declarations do not all fit says so explicitly rather than presenting a
        partial list as complete.

    Raises:
        ValueError: ``max_chars`` is too small to hold the header plus a
            trailing note. The header states what the index is and is not, so
            it is never truncated to fit a cap -- an impossible cap is refused
            instead, which is the only outcome that keeps the cap honest.
    """
    symbols, scanner = scan(path, text)
    head = _header(path, text, scanner)
    head_text = "\n".join(head)
    floor = len(head_text) + 1 + _TAIL_RESERVE
    if max_chars < floor:
        raise ValueError(
            f"max_chars={max_chars} is below the {floor} characters the index of {path} "
            "needs for its header and trailing note"
        )
    if not symbols:
        return "\n".join([*head, _NO_DECLARATIONS])
    budget = max(0, max_chars - len(head_text) - _TAIL_RESERVE)
    fitting, dropped = _fit(symbols, budget)
    lines = [*head, *(_entry(s) for s in fitting)]
    if dropped:
        lines.append(
            f"({dropped} shorter declaration(s) omitted to fit -- this index is "
            "INCOMPLETE; read the file for anything not listed above)"
        )
    return "\n".join(lines)


def build_item(path: str, text: str, max_chars: int = MAX_CHARS) -> IngestItem:
    """Return the learning that stores the digest of ``path``.

    ``file_globs`` is the file's own path, which is what makes the existing
    glob filter in :func:`sidecar.retrieve.query` surface the digest for exactly
    the pull requests that touch that file and no others. It is escaped first:
    both retrieval backends match with :func:`fnmatch.fnmatch`, which reads
    ``*``, ``?`` and ``[...]`` in the *stored* path as pattern syntax, so an
    unescaped ``app/[slug]/page.tsx`` would not even match itself and its index
    would be stored, embedded, and permanently unreachable. ``topic`` keeps the
    literal path, so supersession still recognises the file by name.
    """
    return IngestItem(
        text=render(path, text, max_chars),
        source=DIGEST_SOURCE,
        file_globs=[glob.escape(path)],
        topic=topic_for(path, blob_hash(text)),
    )
