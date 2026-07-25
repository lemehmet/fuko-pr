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


def link(href: str, text: object, **kwargs: object) -> str:
    """Render an escaped anchor."""
    return f"<a{attrs(href=href, **kwargs)}>{esc(text)}</a>"


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


def field(label: str, name: str, value: object = "", **kwargs: object) -> str:
    """Render a labelled ``<input>``."""
    tag = attrs(name=name, value="" if value is None else value, **kwargs)
    return f"<label>{esc(label)}<input{tag}></label>"


def textarea(label: str, name: str, value: object = "", **kwargs: object) -> str:
    """Render a labelled ``<textarea>``."""
    tag = attrs(name=name, **kwargs)
    return f"<label>{esc(label)}<textarea{tag}>{esc(value or '')}</textarea></label>"


def select(label: str, name: str, options: list[tuple[str, str]], value: object = "") -> str:
    """Render a labelled ``<select>`` from ``(value, label)`` pairs, marking the match."""
    opts = "".join(
        f"<option{attrs(value=opt_value, selected=(str(value or '') == opt_value))}>"
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
