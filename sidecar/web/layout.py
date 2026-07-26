"""Shared chrome for the browser-facing pages: the page registry and the document shell.

:data:`PAGES` is the registry, and it is plain data rather than an import-time
side effect -- a new page is one ``Page`` entry here plus a module listed in
:mod:`sidecar.web`. Keeping the registry in this module (which page modules
import) rather than in the package ``__init__`` (which imports page modules)
is what keeps the wiring free of import cycles.

Styling is one inline stylesheet: no external assets, no build step, and no
JavaScript required to use a page.
"""

from __future__ import annotations

from dataclasses import dataclass

from .components import attrs, esc


@dataclass(frozen=True)
class Page:
    """One registered page, as it appears in the shared navigation.

    Attributes:
        slug: Stable identifier, used to mark the active nav entry.
        title: Human label for the nav and the document title.
        path: URL the nav links to.
        order: Sort key for the nav; ties break on title.
    """

    slug: str
    title: str
    path: str
    order: int = 100


PREFIX = "/ui"

PAGES: tuple[Page, ...] = (
    Page(slug="metrics", title="Metrics", path=f"{PREFIX}/metrics", order=10),
    Page(slug="kb", title="Knowledge base", path=f"{PREFIX}/kb", order=20),
)


class UnregisteredPageError(LookupError):
    """Raised when a page module asks for a slug that :data:`PAGES` does not declare."""


def page(slug: str) -> Page:
    """Return the registered page for ``slug``.

    Page modules call this instead of constructing their own ``Page`` so the
    registry stays the one declaration of title, path, and nav order. An
    unregistered slug fails at import, where a wiring mistake is cheap to see.
    """
    for entry in PAGES:
        if entry.slug == slug:
            return entry
    raise UnregisteredPageError(f"page '{slug}' is not declared in layout.PAGES")


_STYLE = """
:root { color-scheme: light dark; --line: color-mix(in srgb, currentColor 25%, transparent);
        --fill: color-mix(in srgb, currentColor 8%, transparent); }
body { font-family: -apple-system, system-ui, sans-serif; margin: 0 auto 3rem;
       max-width: 72rem; padding: 0 1rem; line-height: 1.4; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
nav { display: flex; gap: 1rem; align-items: baseline; flex-wrap: wrap;
      border-bottom: 1px solid var(--line); margin-bottom: 1.5rem; padding: 1rem 0; }
nav .brand { font-weight: 700; letter-spacing: 0.02em; }
nav a { text-decoration: none; opacity: 0.7; }
nav a:hover { opacity: 1; }
nav a.active { opacity: 1; font-weight: 600; text-decoration: underline; }
nav .spacer { flex: 1; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: #2da44e; } .bad { color: #cf222e; font-weight: 600; }
.muted { opacity: 0.65; }
.notice { padding: 0.8rem 1rem; border-radius: 6px; background: var(--fill); margin: 1rem 0; }
.notice.warn { border-left: 3px solid #bf8700; }
.notice.danger { border-left: 3px solid #cf222e; }
.notice.ok { border-left: 3px solid #2da44e; }
.badge { display: inline-block; padding: 0.05rem 0.4rem; border-radius: 999px;
         background: var(--fill); font-size: 0.8rem; }
form { margin: 1rem 0; display: flex; gap: 0.6rem; align-items: flex-end; flex-wrap: wrap; }
form.stacked { display: block; }
form.inline { display: inline; margin: 0; }
.fields { display: flex; gap: 0.6rem; align-items: flex-end; flex-wrap: wrap; margin: 0.6rem 0; }
label { display: inline-flex; flex-direction: column; gap: 0.2rem; font-size: 0.85rem; }
label.row { flex-direction: row; gap: 0.3rem; align-items: center; }
form.stacked > label { display: flex; }
.nowrap { white-space: nowrap; }
input, select, textarea { font: inherit; font-size: 0.9rem; padding: 0.2rem 0.4rem; }
textarea { width: 100%; min-height: 9rem; font-family: ui-monospace, monospace; }
button { font: inherit; padding: 0.25rem 0.8rem; }
details summary { cursor: pointer; }
pre, code { font-family: ui-monospace, monospace; font-size: 0.85rem; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--fill);
      padding: 0.6rem 0.8rem; border-radius: 6px; margin: 0.4rem 0; }
.pager { display: flex; gap: 1rem; align-items: baseline; font-size: 0.9rem; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
         font-size: 0.8rem; opacity: 0.6; }
"""


def nav(active: str, extra: str = "") -> str:
    """Render the shared navigation, marking the entry whose slug is ``active``.

    ``extra`` is already-escaped markup pushed to the right of the links, for
    page-specific chrome such as a session indicator.
    """
    links = "".join(
        f"<a{attrs(href=entry.path, class_='active' if entry.slug == active else None)}>"
        f"{esc(entry.title)}</a>"
        for entry in sorted(PAGES, key=lambda p: (p.order, p.title))
    )
    return f'<nav><span class="brand">fuko</span>{links}<span class="spacer"></span>{extra}</nav>'


def document(*, title: str, body: str, active: str = "", extra_nav: str = "") -> str:
    """Wrap a page ``body`` in the shared document shell.

    Pure string assembly -- the caller has already fetched and rendered its
    data, so this never touches I/O and never raises.
    """
    chrome = nav(active, extra_nav)
    return (
        f"<!doctype html><meta charset=utf-8>"
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_STYLE}</style>"
        f"{chrome}<main>{body}</main>"
        f"<footer>fuko-pr sidecar</footer>"
    )
