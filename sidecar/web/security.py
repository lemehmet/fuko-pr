"""Browser sessions for the mutating UI routes: signed cookie, CSRF, login page.

The read-only pages stay open -- that was the deliberate call for the metrics
view on a LAN-only deployment. Anything that *writes* needs proof the caller
holds ``FUKO_AUTH_TOKEN``, and a browser cannot send a bearer header on a plain
navigation, so the token is exchanged once at a login form for a signed cookie.

No new secret and no new configuration: the cookie is an HMAC over the same
``FUKO_AUTH_TOKEN`` the API already requires, so a deployment that can serve the
API can serve the console. With no token configured, login is impossible and
every mutating route refuses -- the same fail-closed stance as
:func:`sidecar.main._auth`, rather than serving writes unauthenticated.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse

from ..config import settings
from . import components as c
from .layout import PREFIX, document

COOKIE = "fuko_session"

TTL_SECONDS = 12 * 3600

LOGIN_PATH = f"{PREFIX}/login"

LOGOUT_PATH = f"{PREFIX}/logout"

router = APIRouter()


def _sign(payload: str) -> str | None:
    """HMAC ``payload`` with the configured token, or ``None`` when there is none."""
    if not settings.auth_token:
        return None
    return hmac.new(settings.auth_token.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue(now: float | None = None) -> str | None:
    """Mint a session value that expires ``TTL_SECONDS`` from now."""
    expires = int((now if now is not None else time.time()) + TTL_SECONDS)
    signature = _sign(str(expires))
    return f"v1.{expires}.{signature}" if signature else None


def is_valid(value: str | None, now: float | None = None) -> bool:
    """Return whether ``value`` is an unexpired session this server signed."""
    if not value:
        return False
    parts = value.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        return False
    _, expires, signature = parts
    if not expires.isdigit() or int(expires) <= (now if now is not None else time.time()):
        return False
    expected = _sign(expires)
    return bool(expected) and hmac.compare_digest(expected, signature)


def csrf_token(session: str) -> str:
    """Derive the CSRF token bound to one session.

    Bound to the session rather than global, so a token lifted from one user's
    page is inert in another's. ``SameSite=Strict`` on the cookie already blocks
    the cross-site POST; this is the second lock.
    """
    return _sign(f"csrf:{session}") or ""


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
    if not submitted or not hmac.compare_digest(expected, submitted):
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
                "Editing the knowledge base needs the sidecar's FUKO_AUTH_TOKEN. "
                "Browsing does not.",
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
def login_submit(token: str = Form(default=""), next: str | None = Form(default=None)) -> Response:
    """Exchange ``FUKO_AUTH_TOKEN`` for a session cookie.

    The token is compared in constant time, and a failure re-renders the form
    rather than redirecting, so a wrong paste does not lose the destination.
    """
    destination = _safe_next(next)
    if not settings.auth_token or not hmac.compare_digest(token, settings.auth_token):
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
