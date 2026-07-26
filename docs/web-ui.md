# The sidecar's web UI

The sidecar serves a small set of browser-facing utility pages under `/ui`. They
live in `sidecar/web/` and are mounted once, in `sidecar/main.py`:

```python
app.include_router(web.router)
```

```
sidecar/web/
  __init__.py    # imports the page modules, assembles one APIRouter
  layout.py      # Page, PAGES (the registry), the shared stylesheet, document()
  components.py  # escape-safe table/section/notice/pager/field primitives
  metrics.py     # the review-metrics page
```

## Adding a page

Two edits and a module.

**1. Declare it in the registry.** `layout.PAGES` is a literal tuple — the one
place a page's title, path, and nav order are written down:

```python
PAGES: tuple[Page, ...] = (
    Page(slug="metrics", title="Metrics", path=f"{PREFIX}/metrics", order=10),
    Page(slug="kb", title="Knowledge base", path=f"{PREFIX}/kb", order=20),
)
```

**2. Write the module.** It looks its own entry up by slug (an unregistered slug
raises at import, where the mistake is cheap to see), exposes a `router`, and
splits fetching from rendering:

```python
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from . import components as c
from .layout import document, page

PAGE = page("kb")
router = APIRouter()


def render(*, rows: list[dict], db_enabled: bool) -> str:
    """Pure: takes already-queried data, never does I/O, never raises."""
    body = c.section("Learnings", c.table([("repo", False)], [...], "nothing stored"))
    return document(title="fuko knowledge base", body=body, active=PAGE.slug)


@router.get(PAGE.path, response_class=HTMLResponse)
def view() -> str:
    """Fetches, degrades, delegates."""
    try:
        rows = store.list_learnings()
    except Exception:
        rows = []
    return render(rows=rows, db_enabled=bool(settings.database_url))
```

**3. List the module** in `sidecar/web/__init__.py`'s `_MODULES` tuple. Its
router is included automatically and it appears in the shared nav.

The registry lives in `layout.py` rather than the package `__init__` on purpose:
page modules import from `layout`, and `__init__` imports page modules, so there
is never a cycle to work around.

## Conventions

**The route fetches and degrades; the render function is pure.** A render
function takes already-queried plain data, does no I/O, and never raises on
empty input — so a page is always a 200 and never depends on the store being
reachable to draw its own chrome. The route catches, logs to stderr, substitutes
empty data, and passes a flag the render function turns into a notice. See
`metrics.view` / `metrics.render`.

**Everything caller-supplied goes through `components`.** `esc`, `cell`, `link`,
`field`, `attrs` and friends escape on the way in. Writers to the knowledge base
and the metrics tables are trusted, but a learning body and a reviewer's detail
string are still arbitrary text that must not become markup. `raw_cell` is the
one deliberate exception, for markup the page already assembled and escaped.

**No external assets, no build step, no required JavaScript.** One inline
stylesheet in `layout._STYLE`; pages work with plain forms and links. A page that
wants progressive enhancement can add it, but it must not be load-bearing.

**Read is open, mutation is not.** `/ui/metrics` is deliberately unauthenticated
— read-only aggregates on a LAN-only deployment, following the precedent
`/healthz` set. Every API endpoint keeps its bearer auth. A page that mutates
anything must gate those routes behind a session (see `security.py` once it
lands with the knowledge-base console).

## URLs

Pages live under `/ui`. `GET /ui` redirects to the first page in nav order.
`GET /metrics/view` — the pre-`/ui` metrics URL — redirects to `/ui/metrics`
with its query string preserved, because it is a bookmark on deployed sidecars.
