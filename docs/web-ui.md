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
  security.py    # session cookie + CSRF for the routes that write
  metrics.py     # the review-metrics page
  kb.py          # the knowledge-base console
  ledger.py      # the review-state ledgers (read-only)
```

## Adding a page

Two edits and a module.

**1. Declare it in the registry.** `layout.PAGES` is a literal tuple — the one
place a page's title, path, and nav order are written down:

```python
PAGES: tuple[Page, ...] = (
    Page(slug="metrics", title="Metrics", path=f"{PREFIX}/metrics", order=10),
    Page(slug="kb", title="Knowledge base", path=f"{PREFIX}/kb", order=20),
    Page(slug="ledger", title="Ledger", path=f"{PREFIX}/ledger", order=30),
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

**Read is open, mutation is not.** `/ui/metrics`, `/ui/ledger` and the
knowledge-base browsing and preview views are deliberately unauthenticated —
read-only on a LAN-only deployment, following the precedent `/healthz` set.
Every API endpoint keeps its bearer auth. Anything that writes goes through
`security.py`.

**A degraded store has more than two states.** `metrics.render` distinguishes
"no database configured" from "configured but unreachable"; `ledger` keeps the
same split and it is load-bearing there, since a sqlite-vec deployment holds no
review-state tables at all. The configuration test belongs in the route, before
the read: a read that raises on an unset `database_url` would report an
unconfigured deployment as a broken one. Note also that `ledger`'s reads
deliberately do *not* go through `review_state._best_effort` — a swallowed
exception renders exactly the empty table a healthy-but-idle store does, which
is the fail-unsafe direction for a page a human is reading.

## Writing routes that mutate

`security.py` exchanges the existing `FUKO_AUTH_TOKEN` for a signed cookie at
`/ui/login`, so there is no second secret and no new configuration. Three calls
are all a page needs:

```python
@router.post(MY_PATH)
def submit(request: Request, csrf: str = Form(default=""), ...):
    session = security.require(request)      # 303 to login, or 503 with no token set
    security.check_csrf(session, csrf)       # 400 on a missing or forged token
    ...
    return RedirectResponse(..., status_code=303)   # post/redirect/get
```

and every form that posts must embed `security.csrf_field(session)`.

Details worth knowing before you touch it:

- The cookie is `HttpOnly`, `SameSite=Strict`, scoped to `Path=/ui`, and expires
  after `TTL_SECONDS`. Its value is an HMAC over the auth token, so rotating the
  token invalidates every outstanding session for free.
- `Secure` is set when the request arrived over https — including via
  `X-Forwarded-Proto` from a TLS-terminating proxy, which is how the sidecar is
  usually deployed. It is not set unconditionally because a plain-http LAN
  deployment would then never store the cookie at all.
- The CSRF token is derived from the *session*, not global — one lifted from
  another user's page does not verify. `SameSite=Strict` already blocks the
  cross-site POST; this is the second lock.
- `require()` redirects rather than returning 401, because the caller is a
  browser and a bare 401 is a dead end. With `FUKO_AUTH_TOKEN` unset it raises
  503 instead: no login could ever succeed, so a redirect would loop.
- Pass `extra_nav=security.nav_extra(request)` to `document()` so the page shows
  a sign-in link or a sign-out button.
- Rejected writes should re-render the form with the submitted values, not
  redirect. Losing a long edit to a unique collision is its own bug; see
  `kb.edit_submit`.

## URLs

Pages live under `/ui`. `GET /ui` redirects to the first page in nav order.
`GET /metrics/view` — the pre-`/ui` metrics URL — redirects to `/ui/metrics`
with its query string preserved, because it is a bookmark on deployed sidecars.
