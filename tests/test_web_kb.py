"""Tests for the knowledge-base console at /ui/kb (#89)."""

import pytest
from fastapi.testclient import TestClient

from sidecar import main
from sidecar.models import DuplicateLearningError, UnknownSourceError
from sidecar.web import kb, security

from .fakes import FakeStore

_TOKEN = "s3cret-token"

_ITEMS = [
    {
        "id": "id-1",
        "repo": "o/r",
        "text": "Keep migrations idempotent — the runner replays every file.",
        "source": "docs",
        "source_url": "https://example/docs",
        "file_globs": ["migrations/*.sql"],
        "topic": "Migrations",
        "created_at": "2026-06-01T00:00:00+00:00",
        "expires_at": None,
    },
    {
        "id": "id-2",
        "repo": "o/r",
        "text": "Declining: this synchronous path is intentional for ordering.",
        "source": "review_thread",
        "source_url": None,
        "file_globs": [],
        "topic": None,
        "created_at": "2026-06-02T00:00:00+00:00",
        "expires_at": None,
    },
    {
        "id": "id-3",
        "repo": "o/s",
        "text": "UI spacing uses the 4px scale.",
        "source": "remember",
        "source_url": None,
        "file_globs": [],
        "topic": None,
        "created_at": None,
        "expires_at": None,
    },
]


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore([dict(i) for i in _ITEMS])
    monkeypatch.setattr(kb, "current_store", lambda: fake)
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    return fake


@pytest.fixture
def anon(store):
    return TestClient(main.app)


@pytest.fixture
def user(store):
    client = TestClient(main.app)
    client.post(security.LOGIN_PATH, data={"token": _TOKEN}, follow_redirects=False)
    return client


def _csrf(client) -> str:
    return security.csrf_token(client.cookies[security.COOKIE])


def test_picker_lists_every_repo_with_counts(anon):
    page = anon.get("/ui/kb").text
    assert "o/r" in page and "o/s" in page
    assert "docs 1" in page and "review_thread 1" in page


def test_picker_degrades_when_the_store_is_unreachable(anon, store):
    store.read_raises = RuntimeError("connection refused")
    resp = anon.get("/ui/kb")
    assert resp.status_code == 200
    assert "Knowledge store unreachable" in resp.text


def test_browse_lists_one_repo(anon):
    page = anon.get("/ui/kb?repo=o/r").text
    assert "Keep migrations idempotent" in page
    assert "UI spacing" not in page
    assert "migrations/*.sql" in page


def test_browse_passes_filters_to_the_store(anon, store):
    anon.get("/ui/kb?repo=o/r&source=docs&q=migrations&include_expired=true&offset=25")
    assert store.seen["list"] == {
        "repo": "o/r",
        "source": "docs",
        "limit": kb.PAGE_SIZE,
        "offset": 25,
        "q": "migrations",
        "include_expired": True,
    }


def test_browse_floors_a_negative_offset(anon, store):
    anon.get("/ui/kb?repo=o/r&offset=-10")
    assert store.seen["list"]["offset"] == 0


def test_browse_hides_editing_links_from_a_signed_out_visitor(anon):
    page = anon.get("/ui/kb?repo=o/r").text
    assert "/ui/kb/edit?" not in page
    assert "/ui/kb/delete?" not in page


def test_browse_shows_editing_links_once_signed_in(user):
    page = user.get("/ui/kb?repo=o/r").text
    assert "id=id-1" in page
    assert "add a learning" in page and "upload docs / purge" in page


def test_browse_escapes_stored_text(anon, store):
    store.items[0]["text"] = '<script>alert("x")</script>'
    page = anon.get("/ui/kb?repo=o/r").text
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_every_mutating_route_redirects_a_signed_out_caller(anon, store):
    gets = ["/ui/kb/edit?repo=o/r", "/ui/kb/delete?repo=o/r&id=id-1", "/ui/kb/tools?repo=o/r"]
    for path in gets:
        resp = anon.get(path, follow_redirects=False)
        assert resp.status_code == 303, path
        assert resp.headers["location"].startswith("/ui/login?next="), path

    posts = ["/ui/kb/edit", "/ui/kb/delete", "/ui/kb/upload", "/ui/kb/purge"]
    for path in posts:
        resp = anon.post(
            path, data={"repo": "o/r", "id": "id-1", "text": "x"}, follow_redirects=False
        )
        assert resp.status_code == 303, path
        assert resp.headers["location"].startswith("/ui/login?next="), path
    assert store.seen.get("update") is None
    assert store.seen.get("forget") is None
    assert store.seen.get("ingest") is None


def test_mutating_routes_refuse_when_no_token_is_configured(anon, monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", None)
    assert anon.get("/ui/kb/edit?repo=o/r", follow_redirects=False).status_code == 503
    assert anon.post("/ui/kb/delete", data={"repo": "o/r", "id": "id-1"}).status_code == 503


def test_a_post_without_a_csrf_token_is_rejected(user, store):
    resp = user.post("/ui/kb/delete", data={"repo": "o/r", "id": "id-1"})
    assert resp.status_code == 400
    assert store.seen.get("forget") is None


def test_a_post_with_a_forged_csrf_token_is_rejected(user, store):
    resp = user.post("/ui/kb/delete", data={"repo": "o/r", "id": "id-1", "csrf": "nope"})
    assert resp.status_code == 400
    assert store.seen.get("forget") is None


def test_edit_form_shows_the_stored_values(user):
    page = user.get("/ui/kb/edit?repo=o/r&id=id-1").text
    assert "Keep migrations idempotent" in page
    assert 'value="Migrations"' in page
    assert 'value="migrations/*.sql"' in page
    assert '<option value="docs" selected>' in page


def test_edit_form_404s_for_an_unknown_learning(user):
    assert user.get("/ui/kb/edit?repo=o/r&id=nope").status_code == 404


def test_add_form_defaults_to_remember(user):
    page = user.get("/ui/kb/edit?repo=o/r").text
    assert "Add a learning" in page
    assert '<option value="remember" selected>' in page


def test_saving_an_edit_persists_and_redirects(user, store):
    resp = user.post(
        "/ui/kb/edit",
        data={
            "repo": "o/r",
            "id": "id-1",
            "text": "Keep migrations idempotent — the runner replays every file.",
            "source": "docs",
            "topic": "Schema",
            "source_url": "",
            "file_globs": "migrations/*.sql, docs/*.md",
            "expires_at": "",
            "csrf": _csrf(user),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/ui/kb?repo=o%2Fr&msg=Saved")
    _, _, changes = store.seen["update"]
    assert changes["topic"] == "Schema"
    assert changes["file_globs"] == ["migrations/*.sql", "docs/*.md"]
    assert changes["source_url"] is None


def test_saving_a_new_learning_ingests_it(user, store):
    resp = user.post(
        "/ui/kb/edit",
        data={
            "repo": "o/r",
            "text": "New convention.",
            "source": "remember",
            "csrf": _csrf(user),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    repo, items = store.seen["ingest"]
    assert repo == "o/r"
    assert items[0].text == "New convention." and items[0].source == "remember"


def test_a_collision_re_renders_the_form_with_the_submitted_values(user, store):
    store.raises = DuplicateLearningError("another learning already has that (text, source)")
    resp = user.post(
        "/ui/kb/edit",
        data={
            "repo": "o/r",
            "id": "id-1",
            "text": "a long edit worth keeping",
            "source": "docs",
            "topic": "Kept",
            "csrf": _csrf(user),
        },
    )
    assert resp.status_code == 409
    assert "another learning already has that" in resp.text
    assert "a long edit worth keeping" in resp.text
    assert 'value="Kept"' in resp.text


def test_an_unknown_source_re_renders_the_form(user, store):
    store.raises = UnknownSourceError("unknown source 'nope'")
    resp = user.post(
        "/ui/kb/edit",
        data={"repo": "o/r", "id": "id-1", "text": "x", "source": "nope", "csrf": _csrf(user)},
    )
    assert resp.status_code == 422
    assert "unknown source" in resp.text


def test_editing_a_vanished_learning_404s(user, store):
    store.items.clear()
    resp = user.post(
        "/ui/kb/edit",
        data={"repo": "o/r", "id": "id-1", "text": "x", "csrf": _csrf(user)},
    )
    assert resp.status_code == 404


def test_delete_needs_the_confirmation_step(user, store):
    page = user.get("/ui/kb/delete?repo=o/r&id=id-1").text
    assert "cannot be undone" in page
    assert "Keep migrations idempotent" in page
    assert store.seen.get("forget") is None


def test_delete_confirm_404s_for_an_unknown_learning(user):
    assert user.get("/ui/kb/delete?repo=o/r&id=nope").status_code == 404


def test_confirmed_delete_removes_the_learning(user, store):
    resp = user.post(
        "/ui/kb/delete",
        data={"repo": "o/r", "id": "id-1", "csrf": _csrf(user)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "msg=Deleted" in resp.headers["location"]
    assert store.seen["forget"] == ("o/r", "id-1", None, False)
    assert all(i["id"] != "id-1" for i in store.items)


def test_delete_of_an_already_gone_learning_says_so(user, store):
    store.items.clear()
    resp = user.post(
        "/ui/kb/delete",
        data={"repo": "o/r", "id": "id-1", "csrf": _csrf(user)},
        follow_redirects=False,
    )
    assert "already+gone" in resp.headers["location"].replace("%20", "+")


def test_purge_requires_the_repo_name_typed_exactly(user, store):
    resp = user.post(
        "/ui/kb/purge",
        data={"repo": "o/r", "source": "", "confirm": "o/wrong", "csrf": _csrf(user)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "err=" in resp.headers["location"]
    assert store.seen.get("forget") is None


def test_purge_by_source_deletes_only_that_source(user, store):
    resp = user.post(
        "/ui/kb/purge",
        data={"repo": "o/r", "source": "docs", "confirm": "o/r", "csrf": _csrf(user)},
        follow_redirects=False,
    )
    assert store.seen["forget"] == ("o/r", None, "docs", False)
    assert "msg=Purged+1" in resp.headers["location"].replace("%20", "+")
    assert {i["id"] for i in store.items} == {"id-2", "id-3"}


def test_purge_everything_clears_only_that_repo(user, store):
    user.post(
        "/ui/kb/purge",
        data={"repo": "o/r", "source": "", "confirm": " o/r ", "csrf": _csrf(user)},
        follow_redirects=False,
    )
    assert store.seen["forget"] == ("o/r", None, None, True)
    assert {i["id"] for i in store.items} == {"id-3"}


def test_upload_chunks_markdown_into_docs_learnings(user, store):
    markdown = "# Title\n\nintro text\n\n## Section A\nbody A\n\n## Section B\nbody B\n"
    resp = user.post(
        "/ui/kb/upload",
        data={"repo": "o/r", "file_globs": "docs/*.md", "csrf": _csrf(user)},
        files={"files": ("design.md", markdown, "text/markdown")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    repo, items = store.seen["ingest"]
    assert repo == "o/r"
    assert [i.topic for i in items] == ["Title", "Section A", "Section B"]
    assert all(i.source == "docs" for i in items)
    assert all(i.file_globs == ["docs/*.md"] for i in items)
    assert "Ingested+3+chunk" in resp.headers["location"].replace("%20", "+")


def test_upload_falls_back_to_the_filename_when_a_chunk_has_no_heading(user, store):
    resp = user.post(
        "/ui/kb/upload",
        data={"repo": "o/r", "csrf": _csrf(user)},
        files={"files": ("notes.txt", "just a paragraph", "text/plain")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    _, items = store.seen["ingest"]
    assert items[0].topic == "notes.txt"


def test_upload_of_nothing_reports_an_error_and_ingests_nothing(user, store):
    resp = user.post(
        "/ui/kb/upload",
        data={"repo": "o/r", "csrf": _csrf(user)},
        files={"files": ("empty.md", "", "text/markdown")},
        follow_redirects=False,
    )
    assert "err=" in resp.headers["location"]
    assert store.seen.get("ingest") is None


def test_preview_is_open_and_runs_the_review_time_retrieval(anon, store):
    resp = anon.get("/ui/kb/preview?repo=o/r&text=migrations&files=migrations/001.sql, a.py")
    assert resp.status_code == 200
    assert store.seen["query"] == ("o/r", ["migrations/001.sql", "a.py"], "migrations")
    assert "0.900" in resp.text
    assert "Keep migrations idempotent" in resp.text


def test_preview_without_a_query_does_not_touch_the_store(anon, store):
    resp = anon.get("/ui/kb/preview?repo=o/r")
    assert resp.status_code == 200
    assert store.seen.get("query") is None
    assert "nothing retrieved" in resp.text


def test_preview_degrades_when_retrieval_fails(anon, store):
    store.read_raises = RuntimeError("embedder down")
    resp = anon.get("/ui/kb/preview?repo=o/r&text=x")
    assert resp.status_code == 200
    assert "embedder down" in resp.text


def test_tools_page_carries_a_flash_message(user):
    page = user.get("/ui/kb/tools?repo=o/r&msg=Purged+2").text
    assert "Purged 2" in page
    assert "Upload documents" in page and "Purge" in page


def test_kb_is_registered_in_the_shared_nav(anon):
    assert 'href="/ui/kb" class="active"' in anon.get("/ui/kb").text


def test_a_javascript_source_url_is_not_rendered_as_a_link(anon, store):
    store.items[0]["source_url"] = "javascript:alert(document.cookie)"
    page = anon.get("/ui/kb?repo=o/r").text
    assert "javascript:" not in page
    assert 'href="javascript' not in page


def test_an_https_source_url_stays_clickable(anon, store):
    page = anon.get("/ui/kb?repo=o/r").text
    assert 'href="https://example/docs"' in page
