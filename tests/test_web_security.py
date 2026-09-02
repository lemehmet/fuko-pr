"""Tests for the UI session: signed cookie, CSRF, login/logout, fail-closed (#89)."""

import time

import pytest
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

    Only the long-digit value is exercised end to end. The non-ASCII digits are
    unit-tested above instead, because the test client encodes headers as ASCII
    and cannot send them at all -- a raw HTTP client can, since Starlette
    decodes the header bytes as latin-1.
    """
    client.cookies.set(security.COOKIE, f"v1.{'9' * 5000}.x")
    assert client.get(transcripts.PAGE.path).status_code == 200


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
