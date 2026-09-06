"""Tests for the UI session: signed cookie, CSRF, login/logout, fail-closed (#89)."""

import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sidecar import main
from sidecar.web import kb, security, transcripts

from .fakes import FakeStore

_TOKEN = "s3cret-token"


@pytest.fixture(autouse=True)
def _store(monkeypatch):
    """Keep the console's pages off a real database for every test in this module."""
    monkeypatch.setattr(kb, "current_store", lambda: FakeStore())


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    return TestClient(main.app)


@pytest.fixture
def no_token(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", None)
    return TestClient(main.app)


def _sign_in(client) -> str:
    resp = client.post(
        security.LOGIN_PATH, data={"token": _TOKEN, "next": "/ui/kb"}, follow_redirects=False
    )
    assert resp.status_code == 303
    return client.cookies[security.COOKIE]


def test_issued_session_validates(client):
    value = security.issue()
    assert security.is_valid(value)


def test_session_expires(client):
    value = security.issue(now=time.time() - security.TTL_SECONDS - 1)
    assert security.is_valid(value) is False


def test_forged_and_malformed_sessions_are_rejected(client):
    value = security.issue()
    expires, signature = value.split(".")[1:]
    assert security.is_valid(f"v1.{expires}.{'0' * len(signature)}") is False
    assert security.is_valid(f"v2.{expires}.{signature}") is False
    assert security.is_valid("garbage") is False
    assert security.is_valid(f"v1.notanumber.{signature}") is False
    assert security.is_valid(None) is False
    assert security.is_valid("") is False


@pytest.mark.parametrize("expires", ["9" * 5000, "²", "٧"])
def test_an_expiry_int_would_refuse_is_rejected_not_raised(client, expires):
    """Past CPython's digit limit, and the non-ASCII digits `str` calls digits."""
    value = security.issue()
    signature = value.split(".")[2]
    assert security.is_valid(f"v1.{expires}.{signature}") is False


def test_the_login_page_names_the_transcript_read_it_now_gates(client):
    """The one READ behind this sign-in, so the copy must say so (#241)."""
    text = client.get(security.LOGIN_PATH).text
    assert "reading a captured session" in text


def test_a_crafted_cookie_cannot_500_an_open_page(client):
    """`nav_extra` reads the cookie on every render, so this reaches the open pages too.

    Asserted against the transcripts LISTING, not the login page: `render_login`
    passes no ``extra_nav``, so ``/ui/login`` never calls ``nav_extra`` and this
    same assertion pointed there would pass whether or not the guard exists.

    Only the long-digit value goes through the cookie jar, which encodes as
    ASCII. A non-ASCII field still reaches the server over a pure-ASCII wire
    header -- see the octal-escape test below.
    """
    client.cookies.set(security.COOKIE, f"v1.{'9' * 5000}.x")
    assert client.get(transcripts.PAGE.path).status_code == 200


def test_an_octal_escaped_cookie_cannot_500_an_open_page(client):
    """A pure-ASCII wire header still delivers a non-ASCII cookie value.

    ``http.cookies`` unquotes ``\\351`` to ``é`` before the app sees it, so
    ``hmac.compare_digest`` would be handed a non-ASCII ``str`` and raise
    ``TypeError`` -- reachable with any in-range expiry, independent of the
    expiry's own contents.
    """
    resp = client.get(
        transcripts.PAGE.path, headers={"Cookie": f'{security.COOKIE}="v1.9999999999.\\351"'}
    )
    assert resp.status_code == 200


def test_a_session_signed_by_another_token_is_rejected(client, monkeypatch):
    value = security.issue()
    monkeypatch.setattr(main.settings, "auth_token", "a-different-token")
    assert security.is_valid(value) is False


def test_nothing_validates_without_a_configured_token(no_token):
    assert security.issue() is None
    assert security.is_valid("v1.99999999999.abc") is False
    assert security.csrf_token("anything") == ""


def test_csrf_token_is_bound_to_its_session(client):
    one, two = security.issue(), security.issue(now=time.time() + 5)
    assert security.csrf_token(one) != security.csrf_token(two)


def test_login_rejects_a_wrong_token_without_issuing_a_session(client):
    resp = client.post(security.LOGIN_PATH, data={"token": "nope"}, follow_redirects=False)
    assert resp.status_code == 401
    assert "not accepted" in resp.text
    assert security.COOKIE not in resp.cookies


def test_login_issues_a_session_and_returns_to_the_destination(client):
    resp = client.post(
        security.LOGIN_PATH, data={"token": _TOKEN, "next": "/ui/kb"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/ui/kb"
    cookie = resp.headers["set-cookie"]
    assert "HttpOnly" in cookie and "samesite=strict" in cookie.lower()
    assert "Path=/ui" in cookie
    assert security.is_valid(client.cookies[security.COOKIE])


def test_login_refuses_an_offsite_next(client):
    for hostile in ("https://evil.example/x", "//evil.example", "/etc/passwd", "/ui//evil.example"):
        resp = client.post(
            security.LOGIN_PATH, data={"token": _TOKEN, "next": hostile}, follow_redirects=False
        )
        assert resp.headers["location"] == "/ui"


def test_login_page_warns_when_no_token_is_configured(no_token):
    resp = no_token.get(security.LOGIN_PATH)
    assert resp.status_code == 200
    assert "no FUKO_AUTH_TOKEN configured" in resp.text


def test_login_cannot_succeed_without_a_configured_token(no_token):
    resp = no_token.post(security.LOGIN_PATH, data={"token": ""}, follow_redirects=False)
    assert resp.status_code == 401
    assert security.COOKIE not in resp.cookies


def test_logout_clears_the_session(client):
    session = _sign_in(client)
    resp = client.post(
        security.LOGOUT_PATH,
        data={"csrf": security.csrf_token(session)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert client.cookies.get(security.COOKIE) in (None, "")


def test_logout_rejects_a_forged_csrf_token(client):
    _sign_in(client)
    resp = client.post(security.LOGOUT_PATH, data={"csrf": "wrong"}, follow_redirects=False)
    assert resp.status_code == 400


def test_nav_shows_sign_in_when_signed_out_and_sign_out_when_in(client):
    assert "sign in" in client.get("/ui/kb").text
    _sign_in(client)
    assert "sign out" in client.get("/ui/kb").text


def test_metrics_page_stays_open_without_a_session(client):
    assert client.get("/ui/metrics").status_code == 200


def test_cookie_is_not_secure_over_plain_http(client):
    resp = client.post(security.LOGIN_PATH, data={"token": _TOKEN}, follow_redirects=False)
    assert "Secure" not in resp.headers["set-cookie"]


def test_cookie_is_secure_over_https(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    https = TestClient(main.app, base_url="https://fuko.example")
    resp = https.post(security.LOGIN_PATH, data={"token": _TOKEN}, follow_redirects=False)
    assert "Secure" in resp.headers["set-cookie"]


def test_cookie_is_secure_behind_a_tls_terminating_proxy(client):
    resp = client.post(
        security.LOGIN_PATH,
        data={"token": _TOKEN},
        headers={"X-Forwarded-Proto": "https"},
        follow_redirects=False,
    )
    assert "Secure" in resp.headers["set-cookie"]


def test_forwarded_proto_reads_only_the_first_hop(client):
    resp = client.post(
        security.LOGIN_PATH,
        data={"token": _TOKEN},
        headers={"X-Forwarded-Proto": "https, http"},
        follow_redirects=False,
    )
    assert "Secure" in resp.headers["set-cookie"]
    plain = client.post(
        security.LOGIN_PATH,
        data={"token": _TOKEN},
        headers={"X-Forwarded-Proto": "http, https"},
        follow_redirects=False,
    )
    assert "Secure" not in plain.headers["set-cookie"]


def test_a_non_ascii_token_is_refused_not_500(client):
    """The open form's field reaches `compare_digest`, which raises on non-ASCII strs.

    An unauthenticated POST that answers 500 instead of the 401 every other
    rejected token gets (#267). Asserted through the route, not the helper: the
    bug was that nothing between the form and the comparison catches it.
    """
    resp = client.post(
        security.LOGIN_PATH, data={"token": "é", "next": "/ui"}, follow_redirects=False
    )
    assert resp.status_code == 401
    assert security.COOKIE not in resp.cookies


def test_a_non_ascii_csrf_field_is_refused_not_500(client):
    """Same hazard behind a valid session: `check_csrf` compares a raw form field."""
    _sign_in(client)
    resp = client.post(security.LOGOUT_PATH, data={"csrf": "é"}, follow_redirects=False)
    assert resp.status_code == 400


def test_a_non_ascii_auth_token_can_now_sign_in(monkeypatch):
    """The other half of the same bug: no operator could hold a non-ASCII token."""
    monkeypatch.setattr(main.settings, "auth_token", "tökén-π")
    signed_in = TestClient(main.app)
    resp = signed_in.post(security.LOGIN_PATH, data={"token": "tökén-π"}, follow_redirects=False)
    assert resp.status_code == 303
    assert security.COOKIE in signed_in.cookies


def test_a_surrogate_bearing_auth_token_signs_rather_than_500s(monkeypatch):
    """A ``FUKO_AUTH_TOKEN`` whose bytes are not UTF-8 must not break every page.

    ``os.environ`` decodes undecodable bytes with ``surrogateescape``, so the
    token can reach ``settings`` carrying a lone surrogate. ``_same`` was taught
    to survive that; ``_sign`` is the call right behind it, and its key encoding
    is reached on every render through ``nav_extra`` -> ``is_valid``, not only
    at login. Asserted through the routes, since nothing under ``sidecar/``
    catches a ``UnicodeEncodeError``.
    """
    monkeypatch.setattr(main.settings, "auth_token", "t\udcffken")
    odd = TestClient(main.app)
    # An open page, whose nav reads the cookie and so signs on every render.
    assert odd.get("/ui/kb").status_code == 200
    # A syntactically well-formed cookie from any visitor reaches `_sign` too.
    odd.cookies.set(security.COOKIE, "v1.99999999999.deadbeef", path="/ui")
    assert odd.get("/ui/kb").status_code == 200
    # And a session this server minted still verifies, so the token is usable
    # rather than merely non-fatal.
    session = security.issue()
    assert session is not None and security.is_valid(session)


def test_the_comparison_survives_a_lone_surrogate(client):
    """A character with no UTF-8 encoding must answer False, not raise.

    `surrogatepass` rather than plain `.encode()` for the raise, and rather than
    `errors="replace"`, which would map two different surrogates onto one byte
    string and call two different tokens equal.
    """
    assert security._same("a", "\ud800") is False
    assert security._same("\ud800", "\ud800") is True
    assert security._same("\ud800", "\udc00") is False
    with pytest.raises(HTTPException) as raised:
        security.check_csrf(security.issue() or "", "\ud800")
    assert raised.value.status_code == 400


#: Gated pages, as ``(path, params)``. The sweep below is the enforcement of
#: "anything behind `require` is `no-store`" (#266) -- a new gated page that
#: forgets the header fails here rather than at the next audit.
_GATED = [
    (kb.PAGE.path + "/edit", {"repo": "o/r"}),
    (kb.PAGE.path + "/edit", {"repo": "o/r", "id": "id-1"}),
    (kb.PAGE.path + "/delete", {"repo": "o/r", "id": "id-1"}),
    (kb.PAGE.path + "/tools", {"repo": "o/r"}),
    (transcripts.PAGE.path, {"key": "not a key"}),
]

#: Open pages that still render `nav_extra`, so their bodies differ by cookie.
_OPEN_WITH_NAV = [
    (kb.PAGE.path, {}),
    (kb.PAGE.path, {"repo": "o/r"}),
    (kb.PAGE.path + "/preview", {"repo": "o/r"}),
    (transcripts.PAGE.path, {}),
]


@pytest.fixture
def stocked(monkeypatch):
    """A store holding the one learning the edit and delete pages need."""
    monkeypatch.setattr(
        kb,
        "current_store",
        lambda: FakeStore(
            [
                {
                    "id": "id-1",
                    "repo": "o/r",
                    "text": "Keep migrations idempotent.",
                    "source": "docs",
                    "source_url": None,
                    "file_globs": [],
                    "topic": None,
                    "created_at": None,
                    "expires_at": None,
                }
            ]
        ),
    )


@pytest.mark.parametrize(("path", "params"), _GATED)
def test_every_gated_page_refuses_to_be_stored(client, stocked, path, params):
    # Signed out first, so the table cannot quietly list a page that stopped
    # being gated -- the headers would then be asserted on an open page.
    assert client.get(path, params=params, follow_redirects=False).status_code == 303
    _sign_in(client)
    resp = client.get(path, params=params)
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["vary"] == "Cookie"


@pytest.mark.parametrize(("path", "params"), _OPEN_WITH_NAV)
def test_every_open_page_with_a_nav_varies_on_the_cookie(client, stocked, path, params):
    """Cacheable as before, but keyed on the cookie: the nav carries a CSRF token."""
    resp = client.get(path, params=params)
    assert resp.status_code == 200
    assert "cache-control" not in resp.headers
    assert resp.headers["vary"] == "Cookie"


def test_a_write_that_redirects_is_not_stored(client, stocked):
    session = _sign_in(client)
    resp = client.post(
        kb.PAGE.path + "/delete",
        data={"repo": "o/r", "id": "id-1", "csrf": security.csrf_token(session)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["cache-control"] == "no-store"
