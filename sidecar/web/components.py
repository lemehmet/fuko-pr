"""Escape-safe HTML primitives shared by every page under :mod:`sidecar.web`.

Pure string building: no I/O, no framework types, and nothing here raises on
empty or missing input, so a page can always be rendered into a 200. Every
caller-supplied value passes through :func:`esc` on the way in -- the store's
writers are trusted, but a learning body or a reviewer's detail string is still
arbitrary text that must not become markup.
"""

from __future__ import annotations

from html import escape
from urllib.parse import urlencode

Column = tuple[str, bool]
"""A table heading: ``(label, numeric)``.

Numericness is declared per column rather than inferred from the label, so
renaming a heading can never silently change its alignment.
"""


def esc(value: object) -> str:
    """Render ``value`` as escaped text, with ``None`` becoming an em dash."""
    return escape(str(value)) if value is not None else "&mdash;"


def attrs(**kwargs: object) -> str:
    """Build an escaped attribute string, dropping ``None`` and ``False`` values.

    Trailing underscores are stripped and remaining underscores become dashes,
    so ``class_="x"`` and ``data_id=1`` render as ``class`` and ``data-id``. A
    ``True`` value renders as a bare boolean attribute.
    """
    out: list[str] = []
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        name = key.rstrip("_").replace("_", "-")
        if value is True:
            out.append(f" {name}")
        else:
            out.append(f' {name}="{escape(str(value), quote=True)}"')
    return "".join(out)


def cell(value: object, *, numeric: bool = False, css: str = "") -> str:
    """Render one escaped ``<td>``; ``numeric`` right-aligns it with tabular figures."""
    classes = " ".join(c for c in (("num" if numeric else ""), css) if c)
    return f"<td{attrs(class_=classes or None)}>{esc(value)}</td>"


def raw_cell(html: str, *, numeric: bool = False, css: str = "") -> str:
    """Render a ``<td>`` around already-escaped ``html`` (links, badges, nested markup)."""
    classes = " ".join(c for c in (("num" if numeric else ""), css) if c)
    return f"<td{attrs(class_=classes or None)}>{html}</td>"


_SAFE_SCHEMES = frozenset({"http", "https", "mailto"})


def safe_href(href: object) -> str | None:
    """Return ``href`` if it is safe to place in an anchor, else ``None``.

    Escaping stops a URL breaking out of the attribute, but it does not stop
    ``javascript:`` -- an escaped ``javascript:alert(1)`` is still a working XSS
    gadget once clicked. Stored URLs are attacker-influenced (``source_url``
    arrives from ``/ingest`` and from the console form), so schemes are
    allow-listed rather than deny-listed.

    Non-printing characters are stripped before parsing, because browsers strip
    tabs and newlines from URLs and would resolve ``java&#9;script:`` back to a
    scheme this function must already have rejected. A URL with no scheme (a
    relative path) is always allowed -- that is every internal link.
    """
    cleaned = "".join(ch for ch in str(href) if ch.isprintable()).strip()
    scheme, _, _ = cleaned.partition(":")
    if "/" in scheme or "?" in scheme or "#" in scheme:
        return cleaned
    if ":" in cleaned and scheme.lower() not in _SAFE_SCHEMES:
        return None
    return cleaned


def link(href: object, text: object, **kwargs: object) -> str:
    """Render an escaped anchor, degrading to inert text for an unsafe scheme."""
    target = safe_href(href)
    if target is None:
        return f'<span class="muted">{esc(text)}</span>'
    return f"<a{attrs(href=target, **kwargs)}>{esc(text)}</a>"


def badge(text: object, *, css: str = "") -> str:
    """Render a small inline chip, used for sources and states."""
    return f'<span class="badge {escape(css, quote=True)}">{esc(text)}</span>'


def table(headers: list[Column], rows: list[str], empty: str) -> str:
    """Render a table, or a muted notice in its place when there are no rows.

    ``rows`` are pre-rendered ``<tr>`` strings -- assembled by the caller from
    :func:`cell` / :func:`raw_cell`, because only the page knows which of its
    columns are plain text and which are markup.
    """
    if not rows:
        return f'<p class="muted">{esc(empty)}</p>'
    head = "".join(
        f'<th class="num">{esc(label)}</th>' if numeric else f"<th>{esc(label)}</th>"
        for label, numeric in headers
    )
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def section(title: str, body: str) -> str:
    """Wrap ``body`` under an escaped ``<h2>``."""
    return f"<h2>{esc(title)}</h2>{body}"


def notice(text: str, *, kind: str = "info") -> str:
    """Render a boxed message; ``kind`` is one of ``info``, ``warn``, ``danger``, ``ok``."""
    return f'<p class="notice {escape(kind, quote=True)}">{esc(text)}</p>'


def form_value(value: object) -> str:
    """Render ``value`` for a form control: only ``None`` becomes empty.

    Every form helper routes through this rather than testing ``value or ""``,
    which would blank out a legitimate ``0`` or ``False`` and silently drop it
    from the round-trip.
    """
    return "" if value is None else str(value)


def form_int(value: object) -> int | None:
    """Read a number a form submitted, treating anything unusable as "no filter".

    The inverse of :func:`form_value`, and it exists for the same round-trip:
    a browser submits every text input in the form, so an untouched numeric
    field arrives as ``name=`` rather than not arriving at all. Declaring such a
    parameter ``int | None`` does NOT cover that -- ``Optional`` admits an
    ABSENT parameter, not an empty one -- so the operator who clicks "filter"
    having typed only a repository gets FastAPI's 422 instead of a page. A route
    whose integer is bound to a form field therefore takes it as text and parses
    it here.

    ASCII digits only, positive, and inside a Postgres ``integer``: these are
    identifiers (a pull request number), so there is no reading of ``-1`` or
    ``1.5`` worth guessing at, ``²`` is a digit to :meth:`str.isdigit` that
    :func:`int` then refuses, and a number too large for the column would reach
    the page as "store unreachable" -- a fault report for a typo. Every one of
    those is answered the way an empty field is, with the unfiltered page,
    because a filter nobody can read should drop itself rather than the
    response.
    """
    text = form_value(value).strip()
    if not (text.isascii() and text.isdigit()):
        return None
    number = int(text)
    return number if 0 < number <= 2**31 - 1 else None


def field(label: str, name: str, value: object = "", **kwargs: object) -> str:
    """Render a labelled ``<input>``."""
    tag = attrs(name=name, value=form_value(value), **kwargs)
    return f"<label>{esc(label)}<input{tag}></label>"


def textarea(label: str, name: str, value: object = "", **kwargs: object) -> str:
    """Render a labelled ``<textarea>``."""
    tag = attrs(name=name, **kwargs)
    return f"<label>{esc(label)}<textarea{tag}>{esc(form_value(value))}</textarea></label>"


def select(label: str, name: str, options: list[tuple[str, str]], value: object = "") -> str:
    """Render a labelled ``<select>`` from ``(value, label)`` pairs, marking the match."""
    selected = form_value(value)
    opts = "".join(
        f"<option{attrs(value=opt_value, selected=(selected == opt_value))}>"
        f"{esc(opt_label)}</option>"
        for opt_value, opt_label in options
    )
    return f"<label>{esc(label)}<select{attrs(name=name)}>{opts}</select></label>"


def hidden(**kwargs: object) -> str:
    """Render one hidden input per keyword, skipping ``None`` values."""
    return "".join(
        f'<input type="hidden"{attrs(name=name, value=value)}>'
        for name, value in kwargs.items()
        if value is not None
    )


def query_string(params: dict) -> str:
    """Encode ``params`` as a leading-``?`` query string, dropping absent values.

    ``None``, ``False`` and ``""`` are dropped; every other value is kept. The
    ``False`` test is by identity because ``0 == False`` in Python, and a
    dropped ``offset=0`` would silently break the first page of a pager.
    """
    kept = {k: v for k, v in params.items() if v is not None and v is not False and v != ""}
    return f"?{urlencode(kept)}" if kept else ""


def pager(path: str, params: dict, *, offset: int, limit: int, total: int) -> str:
    """Render prev/next links and a position line, or nothing when it all fits on one page.

    ``params`` are the page's other filters, carried through both links so
    paging never silently drops the active filter set.
    """
    if total <= limit:
        return ""
    first = offset + 1 if total else 0
    last = min(offset + limit, total)
    parts = [f'<span class="muted">{first}&ndash;{last} of {total}</span>']
    if offset > 0:
        prev = query_string({**params, "offset": max(0, offset - limit), "limit": limit})
        parts.insert(0, link(f"{path}{prev}", "← prev"))
    if last < total:
        nxt = query_string({**params, "offset": offset + limit, "limit": limit})
        parts.append(link(f"{path}{nxt}", "next →"))
    return f'<p class="pager">{" ".join(parts)}</p>'
