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
  transcripts.py # captured agentic sessions: the listing, and one session
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
    Page(slug="transcripts", title="Transcripts", path=f"{PREFIX}/transcripts", order=40),
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

**Read is open, mutation is not — with one read that is not.** `/ui/metrics`,
`/ui/ledger`, the knowledge-base browsing and preview views, and the
`/ui/transcripts` **listing** are deliberately unauthenticated — read-only on a
LAN-only deployment, following the precedent `/healthz` set. Every API endpoint
keeps its bearer auth. Anything that writes goes through `security.py`.

The exception is `/ui/transcripts?key=…`, the single-session view, which calls
`security.require` even though it mutates nothing. What it renders is not an
aggregate over a review: it is the reviewed repository's own file contents,
verbatim, as the agent read them out of a contributor-controlled checkout (#236's
risk section, #240's note that this corpus is the first read path onto full
reviewed-repo content). Publishing counts to anyone who can reach the port is a
decision the LAN argument covers; publishing a repository's source is not the
same decision, so the session view takes the browser session `security.py`
already mints from `FUKO_AUTH_TOKEN` and the listing beside it stays open. A
future page that renders stored *content* rather than figures about it should
land on the same side of that line.

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
`/ui/login`, so there is no second secret and no new configuration. Four calls
are all a page needs:

```python
@router.get(MY_PATH, response_class=HTMLResponse)
def form(request: Request, response: Response, ...) -> str:
    session = security.require(request)      # 303 to login, or 503 with no token set
    security.no_store(response)              # nothing behind require may be cached
    ...

@router.post(MY_PATH)
def submit(request: Request, csrf: str = Form(default=""), ...):
    session = security.require(request)
    security.check_csrf(session, csrf)       # 400 on a missing or forged token
    ...
    # a handler returning its own Response stamps it directly
    return security.no_store(RedirectResponse(..., status_code=303))
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
  a sign-in link or a sign-out button — and then take `response: Response` and
  call `security.vary_by_cookie(response)`, because that nav makes the body
  differ by cookie even on an open page. `security.no_store` sets `Vary` too, so
  a gated page needs only the one call. `/ui/metrics` and `/ui/ledger` pass no
  `extra_nav` today and so need neither.
- **Anything behind `require` is `no-store`** (#266). A 200 carrying no
  freshness information is heuristically cacheable by a shared cache (RFC 9111
  §4.2.2), and a LAN forward proxy in front of the sidecar can hand a stored
  authenticated page to a request with no cookie at all — `require` is
  per-request and never sees the hit. FastAPI merges the injected `response`'s
  headers only when the handler returns data rather than a `Response`, so a
  handler that builds its own stamps it directly: `return
  security.no_store(HTMLResponse(...))` — the write redirects do this in one
  place, `kb._redirect`. The route sweep in `tests/test_web_security.py` holds
  the rule for the pages listed in its `_GATED` / `_OPEN_WITH_NAV` tables, which
  are hand-maintained: a new page is only swept once its author adds it there,
  so add it in the same commit.
- Comparisons against the token or a CSRF value go through `security._same`, not
  `hmac.compare_digest` directly: form fields are not ASCII, and `compare_digest`
  raises `TypeError` on a non-ASCII `str` (#267). It is still constant time — the
  fix is comparing encoded bytes, not screening the input first.
- Rejected writes should re-render the form with the submitted values, not
  redirect. Losing a long edit to a unique collision is its own bug; see
  `kb.edit_submit` — and stamp that re-render too, since it is the one write
  response carrying both a CSRF token and an unsaved draft.

## URLs

Pages live under `/ui`. `GET /ui` redirects to the first page in nav order.
`GET /metrics/view` — the pre-`/ui` metrics URL — redirects to `/ui/metrics`
with its query string preserved, because it is a bookmark on deployed sidecars.
