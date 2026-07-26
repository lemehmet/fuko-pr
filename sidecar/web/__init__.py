"""The sidecar's browser-facing UI: one router assembled from the page modules.

Adding a utility page is two edits and no framework work: declare it in
:data:`sidecar.web.layout.PAGES`, then write a module here that exposes
``router`` and list it in :data:`_MODULES`. Shared chrome, navigation, escaping,
and the degrade-on-database-error convention come from :mod:`.layout` and
:mod:`.components`; a page that mutates anything gates those routes with
:mod:`.security`. See ``docs/web-ui.md``.

:mod:`.security` carries no ``Page`` entry -- its login and logout routes are
chrome, not a destination, so they are mounted without appearing in the nav.

:mod:`sidecar.main` mounts the result once with ``app.include_router(web.router)``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from . import kb, metrics, security
from .layout import PAGES, PREFIX

_MODULES = (metrics, kb, security)

router = APIRouter()
for _module in _MODULES:
    router.include_router(_module.router)


@router.get(PREFIX, include_in_schema=False)
def ui_index() -> RedirectResponse:
    """Send the bare UI prefix to the first page in nav order."""
    first = sorted(PAGES, key=lambda p: (p.order, p.title))[0]
    return RedirectResponse(first.path, status_code=307)


@router.get("/metrics/view", include_in_schema=False)
def metrics_view_legacy(request: Request) -> RedirectResponse:
    """Redirect the pre-``/ui`` metrics URL, preserving its query string.

    Nothing in the repo links to it, but it is a bookmark on every deployed
    sidecar, so it stays a redirect rather than a 404. 307 keeps the method and
    body intact for anything that is not a plain browser navigation.
    """
    query = request.url.query
    target = f"{metrics.PAGE.path}?{query}" if query else metrics.PAGE.path
    return RedirectResponse(target, status_code=307)
