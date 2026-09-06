"""Browser sessions for the gated UI routes: signed cookie, CSRF, login page.

The read-only pages stay open -- that was the deliberate call for the metrics
view on a LAN-only deployment. Anything that *writes* needs proof the caller
holds ``FUKO_AUTH_TOKEN``, and a browser cannot send a bearer header on a plain
navigation, so the token is exchanged once at a login form for a signed cookie.

One READ takes the same session (#241): the single-session transcript view
renders a reviewed repository's own file contents rather than figures about a
review, which is a different exposure from the aggregates the LAN argument
covers. See ``docs/web-ui.md``.

No new secret and no new configuration: the cookie is an HMAC over the same
``FUKO_AUTH_TOKEN`` the API already requires, so a deployment that can serve the
API can serve the console. With no token configured, login is impossible and
every mutating route refuses -- the same fail-closed stance as
:func:`sidecar.main._auth`, rather than serving writes unauthenticated.

Two rules this module owns for every page, not just its own routes (#266, #267):
the caching headers a gated response needs (:func:`no_store`,
:func:`vary_by_cookie`), and the one constant-time comparison every token check
goes through (:func:`_same`). Both exist because the per-call-site version of
them was got right in one place and missed in the next.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import TypeVar
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse

from ..config import settings
from . import components as c
from .layout import PREFIX, document

#: Any response class, so the header helpers can stamp an injected ``Response``
#: and a returned ``RedirectResponse`` alike without widening either's type.
ResponseT = TypeVar("ResponseT", bound=Response)

COOKIE = "fuko_session"

TTL_SECONDS = 12 * 3600

LOGIN_PATH = f"{PREFIX}/login"

LOGOUT_PATH = f"{PREFIX}/logout"

router = APIRouter()


def no_store(response: ResponseT) -> ResponseT:
    """Forbid every cache from keeping a response that sits behind :func:`require`.

    A 200 carrying no freshness information is heuristically cacheable by a
    shared cache (RFC 9111 4.2.2), so a LAN forward proxy in front of the sidecar
    may store an authenticated page and later hand it to a request with no
    session -- serving the very bytes ``require`` stands in front of, which is
    per-request and never sees a cache hit. ``Vary`` rides along because a gated
    body is session-derived by definition.

    Returns the response so a route can stamp one it built
    (``return no_store(RedirectResponse(...))``) as readily as the one FastAPI
    injected.
    """
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Cookie"
    return response


def vary_by_cookie(response: ResponseT) -> ResponseT:
    """Key a cacheable response on the session cookie.

    Wider than :func:`no_store` and it reaches the OPEN pages: :func:`nav_extra`
    renders a sign-out form carrying the viewer's CSRF token to a signed-in
    operator and a sign-in link to everyone else, so every page passing
    ``extra_nav`` has a body that differs by cookie whether or not the page is
    gated. Without this a shared cache may serve one viewer's copy -- token
    included -- to the next. The page itself stays cacheable, which is the point:
    its content is published by design.
    """
    response.headers["Vary"] = "Cookie"
    return response


def _same(expected: str, submitted: str) -> bool:
    """Compare two strings in constant time, whatever characters they carry.

    ``hmac.compare_digest`` raises TypeError on non-ASCII ``str``s, and two of
    this module's three comparisons are handed a raw form field. Comparing the
    encoded bytes keeps the constant-time property -- it is the same routine --
    where pre-screening for ASCII would answer a non-ASCII token faster than a
    wrong ASCII one, which is the timing signal these call sites exist to avoid.

    ``surrogatepass`` so that this cannot raise either: a lone surrogate has no
    UTF-8 encoding and would trade one uncaught exception for another, while
    ``replace`` would map distinct surrogates onto one byte string and call two
    different tokens equal.
    """
    return hmac.compare_digest(
        expected.encode("utf-8", "surrogatepass"), submitted.encode("utf-8", "surrogatepass")
    )


def _sign(payload: str) -> str | None:
    """HMAC ``payload`` with the configured token, or ``None`` when there is none.

    The key is encoded the same way :func:`_same` compares, and for the same
    reason: a ``FUKO_AUTH_TOKEN`` whose bytes are not valid UTF-8 reaches
    ``settings`` as a str carrying lone surrogates (``os.environ`` decodes with
    ``surrogateescape``), which a strict ``.encode()`` refuses. That raise is
    reached on every page render -- ``nav_extra`` reads the cookie, ``is_valid``
    signs to check it -- and nothing above it catches, so the whole console
    answers 500 rather than the operator merely being unable to sign in. The
    payload is this module's own ``str(int)`` and needs no such handling.
    """
    if not settings.auth_token:
        return None
    key = settings.auth_token.encode("utf-8", "surrogatepass")
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


def issue(now: float | None = None) -> str | None:
    """Mint a session value that expires ``TTL_SECONDS`` from now."""
    expires = int((now if now is not None else time.time()) + TTL_SECONDS)
    signature = _sign(str(expires))
    return f"v1.{expires}.{signature}" if signature else None


def is_valid(value: str | None, now: float | None = None) -> bool:
    """Return whether ``value`` is an unexpired session this server signed."""
    # ASCII is tested ONCE, on the whole value, rather than per field: a session
    # this server minted is `v1.<digits>.<hex>` and so ASCII by construction, so
    # a value carrying anything else cannot be ours and is rejected before it
    # can reach `int()`, which refuses `²` -- a character `str.isdigit` calls a
    # digit -- and which nothing catches above `nav_extra`, the cookie reader on
    # every page render, the open ones included. The signature comparison below
    # no longer needs this guard (`_same` carries its own), but `int()` does.
    if not value or not value.isascii():
        return False
    parts = value.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        return False
    _, expires, signature = parts
    # The length test runs before ``int()``, for the reason
    # :func:`sidecar.web.components.form_int` states: CPython refuses a
    # conversion past ``sys.get_int_max_str_digits()`` with a ValueError, and
    # nothing here would catch it. Twenty digits is far past any epoch second a
    # signature of ours will ever carry.
    if not expires.isdigit() or len(expires) > 20:
        return False
    if int(expires) <= (now if now is not None else time.time()):
        return False
    expected = _sign(expires)
    return bool(expected) and _same(expected, signature)


def csrf_token(session: str) -> str:
    """Derive the CSRF token bound to one session.

    Bound to the session rather than global, so a token lifted from one user's
    page is inert in another's. ``SameSite=Strict`` on the cookie already blocks
    the cross-site POST; this is the second lock.
    """
    return _sign(f"csrf:{session}") or ""


def is_secure(request: Request) -> bool:
    """Return whether this request reached the sidecar over https.

    Honours ``X-Forwarded-Proto`` because the sidecar habitually sits behind a
    reverse proxy that terminates TLS -- reading only ``request.url.scheme``
    there would see http and drop the ``Secure`` flag on a connection that was
    in fact encrypted.
    """
    forwarded = request.headers.get("x-forwarded-proto", "")
    if forwarded:
        return forwarded.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


def session_of(request: Request) -> str | None:
    """Return the request's valid session value, or ``None``."""
    value = request.cookies.get(COOKIE)
    return value if is_valid(value) else None


def require(request: Request) -> str:
    """Return the caller's session, or refuse the request.

    A missing session redirects to the login page (the caller is a browser, so a
    bare 401 would be a dead end). An unconfigured ``FUKO_AUTH_TOKEN`` is a
    different failure -- no login could ever succeed -- and says so with a 503.
    """
    session = session_of(request)
    if session:
        return session
    if not settings.auth_token:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "server auth not configured (set FUKO_AUTH_TOKEN)",
        )
    target = request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"
    raise HTTPException(
        status.HTTP_303_SEE_OTHER, "sign in required", headers={"Location": _login_url(target)}
    )


def check_csrf(session: str, submitted: str | None) -> None:
    """Reject a form post whose CSRF token is missing or not bound to ``session``."""
    expected = csrf_token(session)
    if not submitted or not _same(expected, submitted):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid or missing CSRF token")


def csrf_field(session: str) -> str:
    """Render the hidden CSRF input every mutating form must carry."""
    return c.hidden(csrf=csrf_token(session))


def _login_url(next_path: str) -> str:
    return f"{LOGIN_PATH}?next={quote(next_path, safe='')}"


def _safe_next(next_path: str | None) -> str:
    """Confine post-login redirects to this app's UI prefix.

    An attacker-supplied ``next`` is the classic open-redirect; anything that is
    not a plain path under ``/ui`` falls back to the UI index.
    """
    if next_path and next_path.startswith(f"{PREFIX}/") and "//" not in next_path:
        return next_path
    return PREFIX


def nav_extra(request: Request) -> str:
    """Render the nav's session indicator: a sign-out form, or a sign-in link."""
    if session_of(request) is None:
        return c.link(_login_url(str(request.url.path)), "sign in", class_="muted")
    return (
        f'<form method="post" action="{LOGOUT_PATH}" class="inline">'
        f"{csrf_field(session_of(request) or '')}<button>sign out</button></form>"
    )


def render_login(*, next_path: str, error: str = "") -> str:
    """Render the login form (pure)."""
    body = ["<h1>Sign in</h1>"]
    if not settings.auth_token:
        body.append(
            c.notice(
                "This sidecar has no FUKO_AUTH_TOKEN configured, so no sign-in can "
                "succeed and every editing action is refused. Set it and restart.",
                kind="danger",
            )
        )
    elif error:
        body.append(c.notice(error, kind="danger"))
    else:
        body.append(
            c.notice(
                "Editing the knowledge base, and reading a captured session "
                "transcript, need the sidecar's FUKO_AUTH_TOKEN. Browsing the "
                "other pages does not.",
            )
        )
    body.append(
        f'<form method="post" action="{LOGIN_PATH}">'
        + c.hidden(next=next_path)
        + c.field("token", "token", "", type="password", size=48, autofocus=True)
        + "<button>sign in</button></form>"
    )
    return document(title="fuko · sign in", body="".join(body))


@router.get(LOGIN_PATH, response_class=HTMLResponse, include_in_schema=False)
def login_form(request: Request, next: str | None = None) -> str:
    """Serve the sign-in form."""
    return render_login(next_path=_safe_next(next))


@router.post(LOGIN_PATH, include_in_schema=False)
def login_submit(
    request: Request, token: str = Form(default=""), next: str | None = Form(default=None)
) -> Response:
    """Exchange ``FUKO_AUTH_TOKEN`` for a session cookie.

    The token is compared in constant time, and a failure re-renders the form
    rather than redirecting, so a wrong paste does not lose the destination.

    ``Secure`` is set whenever the request arrived over https, which keeps the
    cookie off any plaintext path to the same host. It is not set unconditionally
    because the LAN deployments this console targets are commonly plain http, and
    a ``Secure`` cookie there would simply never be stored.
    """
    destination = _safe_next(next)
    if not settings.auth_token or not _same(token, settings.auth_token):
        return HTMLResponse(
            render_login(next_path=destination, error="That token was not accepted."),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    value = issue()
    response = RedirectResponse(destination, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        COOKIE,
        value or "",
        max_age=TTL_SECONDS,
        httponly=True,
        samesite="strict",
        secure=is_secure(request),
        path=PREFIX,
    )
    return response


@router.post(LOGOUT_PATH, include_in_schema=False)
def logout(request: Request, csrf: str = Form(default="")) -> RedirectResponse:
    """Drop the session cookie."""
    session = session_of(request)
    if session:
        check_csrf(session, csrf)
    response = RedirectResponse(PREFIX, status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(COOKIE, path=PREFIX)
    return response
