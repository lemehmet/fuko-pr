"""Tests for the captured-session transcripts page under /ui (#241)."""

import json

import pytest
from fastapi.testclient import TestClient

from sidecar import main
from sidecar import transcripts as corpus
from sidecar.web import security
from sidecar.web import transcripts as page

_TOKEN = "s3cret-token"


def _run(**overrides) -> corpus.TranscriptRun:
    base = dict(
        key="20260901T100000Z-abc123def456",
        created_at="2026-09-01T10:00:00+00:00",
        complete=True,
        tool_calls={"Read": 12, "Grep": 4, "Bash": 2, "Edit": 1, "Glob": 1},
        tool_result_bytes=1536,
        repeated_read_files=3,
        repo="lemehmet/mepro",
        pr=1343,
        seat="dorian",
        provider="zai",
        model="glm-4.6",
        backend="agentic",
        outcome="ok",
        started_at="2026-09-01T09:58:00+00:00",
        duration_s=612.5,
    )
    base.update(overrides)
    return corpus.TranscriptRun(**base)


def _feed(*events) -> bytes:
    return ("\n".join(json.dumps(e) if not isinstance(e, str) else e for e in events)).encode()


def _assistant(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def _tool_use(name: str, tool_input: dict) -> dict:
    return {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": name, "input": tool_input}]},
    }


def _tool_result(content, *, is_error: bool = False) -> dict:
    return {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "content": content, "is_error": is_error}]},
    }


@pytest.fixture
def wire(monkeypatch):
    """Point the page at canned reads, and report what it asked for."""
    seen: dict = {}
    state: dict = {"rows": (), "run": _run(), "blob": None, "index_error": None, "blob_error": None}

    def configure(**kwargs):
        state.update(kwargs)
        monkeypatch.setattr(
            main.settings,
            "database_url",
            "postgresql://x/y" if state.get("db", True) else None,
        )
        return configure

    def fake_list(**kwargs):
        seen["list"] = kwargs
        if state["index_error"] is not None:
            raise state["index_error"]
        rows = tuple(state["rows"])
        return corpus.TranscriptPage(rows=rows, total=state.get("total", len(rows)))

    def fake_describe(key):
        seen["describe"] = key
        if state["index_error"] is not None:
            raise state["index_error"]
        return state["run"]

    def fake_fetch(key):
        seen["fetch"] = seen.get("fetch", 0) + 1
        if state["blob_error"] is not None:
            raise state["blob_error"]
        return state["blob"]

    monkeypatch.setattr(corpus, "list_transcripts", fake_list)
    monkeypatch.setattr(corpus, "describe", fake_describe)
    monkeypatch.setattr(corpus, "fetch", fake_fetch)
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    configure.seen = seen
    configure.state = state
    return configure


@pytest.fixture
def client():
    return TestClient(main.app)


def _sign_in(client) -> None:
    resp = client.post(
        security.LOGIN_PATH,
        data={"token": _TOKEN, "next": page.PAGE.path},
        follow_redirects=False,
    )
    assert resp.status_code == 303


# --- the listing -----------------------------------------------------------


def test_listing_is_open_and_lists_runs(wire, client):
    wire(rows=[_run()])
    resp = client.get(page.PAGE.path)
    assert resp.status_code == 200
    assert "20260901T100000Z-abc123def456" in resp.text
    assert "lemehmet/mepro" in resp.text
    assert "Read=12" in resp.text
    assert "1.5 KiB" in resp.text
    assert "#1343" in resp.text


def test_listing_cross_links_on_repo_and_pr(wire, client):
    wire(rows=[_run()])
    text = client.get(page.PAGE.path).text
    assert "/ui/metrics?repo=lemehmet%2Fmepro" in text
    assert "/ui/ledger?repo=lemehmet%2Fmepro&amp;pr=1343" in text


def test_healthy_but_empty_is_not_an_outage(wire, client):
    wire(rows=[])
    text = client.get(page.PAGE.path).text
    assert "no transcripts captured yet" in text
    assert "unreachable" not in text


def test_unconfigured_store_says_so(wire, client):
    wire(rows=[], db=False)
    text = client.get(page.PAGE.path).text
    assert "FUKO_DATABASE_URL unset" in text
    assert "unreachable" not in text


def test_unreachable_index_is_not_an_empty_corpus(wire, client):
    wire(rows=[], index_error=RuntimeError("connection refused"))
    text = client.get(page.PAGE.path).text
    assert "Transcript index unreachable" in text
    assert "this is a fault, not an empty corpus" in text


def test_bad_date_filter_is_a_typo_not_an_outage(wire, client):
    wire(rows=[], index_error=ValueError("Invalid isoformat string: 'yesterday'"))
    text = client.get(f"{page.PAGE.path}?since=yesterday").text
    assert "Filter not understood" in text
    assert "unreachable" not in text


def test_a_rejected_filter_does_not_claim_an_empty_corpus(wire, client):
    """The bound is rejected before the query opens, so nothing was listed at all."""
    wire(rows=[], index_error=ValueError("Invalid isoformat string: 'yesterday'"))
    text = client.get(f"{page.PAGE.path}?since=yesterday").text
    assert "the listing was not run" in text
    assert "the filter above was not understood" in text
    assert "no transcripts captured yet" not in text


def test_filters_narrow_individually_and_combined(wire, client):
    configure = wire(rows=[])
    client.get(f"{page.PAGE.path}?repo=lemehmet%2Fmepro")
    assert configure.seen["list"]["repo"] == "lemehmet/mepro"
    assert configure.seen["list"]["pr"] is None
    client.get(f"{page.PAGE.path}?repo=lemehmet%2Fmepro&pr=1343&seat=dorian&since=2026-09-01")
    asked = configure.seen["list"]
    assert asked["repo"] == "lemehmet/mepro"
    assert asked["pr"] == 1343
    assert asked["seat"] == "dorian"
    assert asked["since"] == "2026-09-01"


def test_empty_pr_field_is_treated_as_no_filter(wire, client):
    configure = wire(rows=[])
    resp = client.get(f"{page.PAGE.path}?repo=lemehmet%2Fmepro&pr=")
    assert resp.status_code == 200
    assert configure.seen["list"]["pr"] is None


def test_listing_limit_is_clamped_to_the_module_bound(wire, client):
    configure = wire(rows=[])
    client.get(f"{page.PAGE.path}?limit=100000&offset=-5")
    assert configure.seen["list"]["limit"] == corpus.MAX_ROWS
    assert configure.seen["list"]["offset"] == 0


def test_incomplete_transcript_is_flagged_in_the_listing(wire, client):
    wire(rows=[_run(complete=False)])
    assert "INCOMPLETE" in client.get(page.PAGE.path).text


def test_transcript_with_no_run_row_is_listed(wire, client):
    wire(rows=[_run(repo=None, pr=None, seat=None, model=None, outcome=None)])
    text = client.get(page.PAGE.path).text
    assert "(no run row)" in text


def test_pager_carries_the_filters(wire, client):
    wire(rows=[_run()], total=500)
    text = client.get(f"{page.PAGE.path}?repo=lemehmet%2Fmepro&limit=1").text
    assert "offset=1" in text
    assert "repo=lemehmet%2Fmepro" in text


# --- the session view is authenticated -------------------------------------


def test_session_view_needs_a_session(wire, client):
    wire(blob=_feed(_assistant("hi")))
    resp = client.get(f"{page.PAGE.path}?key=k", follow_redirects=False)
    assert resp.status_code == 303
    assert security.LOGIN_PATH in resp.headers["location"]


def test_session_view_refuses_when_no_token_is_configured(wire, client, monkeypatch):
    wire(blob=_feed(_assistant("hi")))
    monkeypatch.setattr(main.settings, "auth_token", None)
    resp = client.get(f"{page.PAGE.path}?key=k", follow_redirects=False)
    assert resp.status_code == 503


def test_listing_stays_open_when_no_token_is_configured(wire, client, monkeypatch):
    wire(rows=[_run()])
    monkeypatch.setattr(main.settings, "auth_token", None)
    assert client.get(page.PAGE.path).status_code == 200


# --- the session view ------------------------------------------------------


def test_session_renders_turns_in_order(wire, client):
    wire(
        blob=_feed(
            _assistant("looking at the diff"),
            _tool_use("Read", {"file_path": "src/app.py"}),
            _tool_result("def main():\n    pass\n"),
            {"type": "result", "subtype": "success"},
        )
    )
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=20260901T100000Z-abc123def456").text
    assert "looking at the diff" in text
    assert "tool call · Read" in text
    assert "src/app.py" in text
    assert "def main():" in text
    assert text.index("looking at the diff") < text.index("def main():")


def test_session_shows_the_runs_derived_figures(wire, client):
    wire(blob=_feed(_assistant("hi")))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "tool-result bytes" in text
    assert "1.5 KiB" in text
    assert "files read more than once" in text


def test_incomplete_transcript_is_flagged_in_the_session_view(wire, client):
    wire(run=_run(complete=False), blob=_feed(_assistant("hi")))
    _sign_in(client)
    assert "INCOMPLETE" in client.get(f"{page.PAGE.path}?key=k").text


# --- escaping is the security boundary -------------------------------------


def test_markup_in_a_tool_result_is_inert(wire, client):
    wire(blob=_feed(_tool_result("<script>alert('pwned')</script>")))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text


def test_markup_in_a_tool_argument_is_inert(wire, client):
    wire(blob=_feed(_tool_use("Bash", {"command": "<img src=x onerror=alert(1)>"})))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "<img src=x" not in text
    assert "onerror" in text  # shown, as text


def test_a_javascript_url_never_becomes_a_link(wire, client):
    wire(blob=_feed(_tool_use("WebFetch", {"url": "javascript:alert(1)"})))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert 'href="javascript' not in text
    assert "javascript:alert(1)" in text


def test_a_quoted_file_path_cannot_break_an_attribute(wire, client):
    wire(blob=_feed(_tool_use("Read", {"file_path": 'src/"onmouseover=alert(1) x".py'})))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "&quot;onmouseover" in text
    assert '"onmouseover=alert(1)' not in text


def test_a_lone_surrogate_does_not_fault_the_page(wire, client):
    # Legal JSON that json.loads returns as an unencodable str; the response
    # encoder would raise UnicodeEncodeError on it.
    wire(
        blob=b'{"type":"user","message":{"content":[{"type":"tool_result",'
        b'"content":"before \\ud800 after"}]}}'
    )
    _sign_in(client)
    resp = client.get(f"{page.PAGE.path}?key=k")
    assert resp.status_code == 200
    assert "before" in resp.text


def test_a_markup_bearing_key_is_escaped_in_the_heading(wire, client):
    wire(run=None, blob=None)
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=%3Cscript%3Ealert(1)%3C/script%3E").text
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text


# --- the session view's three-way degrade ----------------------------------


def test_unconfigured_blob_store_is_not_an_empty_session(wire, client):
    wire(blob_error=corpus.StoreUnconfigured("no transcript store configured"))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "FUKO_TRANSCRIPT_STORE_BACKEND unset" in text
    assert "the off state, not an empty session" in text


def test_unreachable_blob_store_is_a_fault(wire, client):
    wire(blob_error=RuntimeError("connection reset"))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "Transcript store unreachable" in text


def test_a_malformed_key_is_a_typo_not_an_outage(wire, client):
    wire()
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=..%2Fetc%2Fpasswd").text
    assert "not a well-formed transcript key" in text
    assert "unreachable" not in text


def test_a_malformed_key_is_judged_before_either_read(wire, client):
    """A key nothing could match must not reach the index or the store."""
    configure = wire()
    _sign_in(client)
    client.get(f"{page.PAGE.path}?key=..%2Fetc%2Fpasswd")
    assert "describe" not in configure.seen
    assert "fetch" not in configure.seen


def test_a_configured_but_broken_store_is_a_fault_not_a_bad_key(wire, client):
    """`fetch` builds the store before it reads the key, so its ValueError is ambiguous.

    `make_blob_store` raises for a deployment that meant to store something and
    cannot -- no ROOT, no BUCKET, an unknown backend. Reported off the fetch
    that would tell every operator, holding a perfectly good key, that they had
    mistyped it.
    """
    wire(blob_error=ValueError("the 'file' transcript store needs FUKO_TRANSCRIPT_STORE_ROOT"))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "Transcript store unreachable" in text
    assert "not a well-formed transcript key" not in text


def test_a_line_that_decodes_but_will_not_re_encode_still_draws(wire, client):
    """`json.dumps(indent=…)` recurses too, from a deeper stack than `json.loads` did.

    So a line can parse cleanly and then exhaust the encoder mid-render — the
    parse-side guard cannot see this one.
    """
    nested: dict = {}
    cur = nested
    for _ in range(3000):
        cur["a"] = {}
        cur = cur["a"]
    wire(blob=_feed({"type": "custom", "deep": nested}, _assistant("still here")))
    _sign_in(client)
    resp = client.get(f"{page.PAGE.path}?key=k")
    assert resp.status_code == 200
    assert "unrenderable event" in resp.text
    assert "still here" in resp.text


def test_a_deeply_nested_line_is_drawn_raw_not_a_500(wire, client):
    """`json.loads` raises RecursionError, a RuntimeError, on a deeply nested line."""
    wire(blob=_feed("[" * 100000 + "]" * 100000, _assistant("still here")))
    _sign_in(client)
    resp = client.get(f"{page.PAGE.path}?key=k")
    assert resp.status_code == 200
    assert "still here" in resp.text


def test_a_session_response_forbids_shared_caching(wire, client):
    """A cache that kept this could serve stored repo content past `require`."""
    wire(blob=_feed(_assistant("hi")))
    _sign_in(client)
    resp = client.get(f"{page.PAGE.path}?key=k")
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["vary"] == "Cookie"


def test_the_open_listing_stays_cacheable(wire, client):
    """The listing publishes index-row figures and is open, so it is not no-store."""
    wire(rows=[_run()])
    resp = client.get(page.PAGE.path)
    assert "cache-control" not in resp.headers


def test_a_malformed_key_reports_nothing_about_the_index(wire, client):
    """Neither read ran, so the page has nothing it may say about the index."""
    wire()
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=..%2Fetc%2Fpasswd").text
    assert "not a well-formed transcript key" in text
    assert "No index row for this key" not in text
    assert "real stored session" not in text


def test_a_key_that_holds_nothing_is_not_also_a_real_stored_session(wire, client):
    """The #258 claim needs bytes behind it: "holds nothing" and "is stored" cannot both run."""
    wire(blob=None, run=None)
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "holds nothing under this key" in text
    assert "real stored session" not in text


def test_an_unreachable_store_makes_no_claim_about_a_stored_session(wire, client):
    wire(blob_error=RuntimeError("connection reset"), run=None)
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "Transcript store unreachable" in text
    assert "real stored session" not in text


def test_a_key_that_holds_nothing_says_so(wire, client):
    wire(blob=None)
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "holds nothing under this key" in text


def test_an_unindexed_blob_still_renders_its_session(wire, client):
    wire(run=None, blob=_feed(_assistant("captured but never indexed")))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "No index row for this key" in text
    assert "captured but never indexed" in text


def test_an_unreachable_index_still_renders_the_session(wire, client):
    wire(index_error=RuntimeError("connection refused"), blob=_feed(_assistant("body is fine")))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "Transcript index unreachable" in text
    assert "body is fine" in text


def test_no_database_configured_still_renders_the_session(wire, client):
    configure = wire(blob=_feed(_assistant("body is fine")))
    configure(db=False)
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "no run attribution" in text
    assert "body is fine" in text
    assert "describe" not in configure.seen


# --- paging and clipping ---------------------------------------------------


def test_a_large_session_pages_rather_than_rendering_whole(wire, client):
    wire(blob=_feed(*[_assistant(f"turn {n}") for n in range(60)]))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "turn 0" in text
    assert "turn 24" in text
    assert "turn 25" not in text
    assert "of 60" in text


def test_the_session_pager_walks_the_feed(wire, client):
    wire(blob=_feed(*[_assistant(f"turn {n}") for n in range(60)]))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k&offset=25").text
    assert "turn 25" in text
    assert "turn 0" not in text


def test_session_limit_is_clamped(wire, client):
    """Asserted on the rendered window, not the status code: a 200 proves nothing here."""
    wire(blob=_feed(*[_assistant(f"turn {n}") for n in range(page.MAX_EVENTS + 50)]))
    _sign_in(client)
    resp = client.get(f"{page.PAGE.path}?key=k&limit=100000")
    assert resp.status_code == 200
    assert f"turn {page.MAX_EVENTS - 1}" in resp.text
    assert f"turn {page.MAX_EVENTS}" not in resp.text
    assert f"1&ndash;{page.MAX_EVENTS} of {page.MAX_EVENTS + 50}" in resp.text


def test_one_enormous_block_is_clipped_with_a_pointer_to_the_cli(wire, client):
    wire(blob=_feed(_tool_result("x" * (page.MAX_BLOCK_CHARS + 500))))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "clipped at" in text
    assert "fuko transcripts get" in text


def test_an_unreadable_line_is_shown_rather_than_dropped(wire, client):
    wire(
        blob=b'{"type":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}\n'
        b'{"type":"user","messa'
    )
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "unreadable line" in text
    assert "messa" in text


def test_a_non_text_result_block_is_named_not_dropped(wire, client):
    wire(blob=_feed(_tool_result([{"type": "image", "source": {}}])))
    _sign_in(client)
    assert "non-text block: image" in client.get(f"{page.PAGE.path}?key=k").text


def test_an_error_result_is_marked(wire, client):
    wire(blob=_feed(_tool_result("command not found", is_error=True)))
    _sign_in(client)
    assert "tool result · error" in client.get(f"{page.PAGE.path}?key=k").text


# --- the pure halves -------------------------------------------------------


def test_read_session_counts_every_line_and_decodes_one_page():
    data = _feed(*[_assistant(f"turn {n}") for n in range(10)])
    result = page.read_session(data, offset=3, limit=2)
    assert result.total == 10
    assert [line.number for line in result.lines] == [4, 5]
    assert result.lines[0].event["message"]["content"][0]["text"] == "turn 3"


def test_read_session_skips_blank_framing():
    result = page.read_session(b'\n\n{"type":"result"}\n\n', offset=0, limit=10)
    assert result.total == 1


def test_read_session_keeps_a_line_that_is_not_an_object():
    result = page.read_session(b'"just a string"\n', offset=0, limit=10)
    assert result.lines[0].event is None
    assert result.lines[0].raw == '"just a string"'


def test_render_index_on_empty_input_is_page_chrome():
    html = page.render_index(
        page_rows=corpus.TranscriptPage(),
        repo=None,
        pr=None,
        seat=None,
        since=None,
        until=None,
        offset=0,
        limit=page.PAGE_SIZE,
        db_enabled=True,
    )
    assert "Captured session transcripts" in html
    assert "no transcripts captured yet" in html


def test_render_session_draws_chrome_with_no_body_at_all():
    html = page.render_session(
        key="20260901T100000Z-abc",
        run=None,
        session=page.SessionPage(),
        offset=0,
        limit=page.EVENTS_PER_PAGE,
        store_state="unreachable",
        db_enabled=True,
    )
    assert "20260901T100000Z-abc" in html
    assert "Transcript store unreachable" in html
    assert "no events to show" in html


def test_the_page_is_registered_in_the_nav(wire, client):
    wire(rows=[])
    assert 'href="/ui/transcripts"' in client.get(page.PAGE.path).text


# --- a feed shaped in ways the schema never promised ------------------------


def test_a_result_content_of_an_unexpected_shape_is_shown_as_json(wire, client):
    wire(blob=_feed(_tool_result({"stdout": "hi"})))
    _sign_in(client)
    assert "stdout" in client.get(f"{page.PAGE.path}?key=k").text


def test_a_result_block_that_is_not_a_mapping_is_named(wire, client):
    wire(blob=_feed(_tool_result(["a bare string"])))
    _sign_in(client)
    assert "non-text block: unknown" in client.get(f"{page.PAGE.path}?key=k").text


def test_a_content_block_that_is_not_a_mapping_still_renders(wire, client):
    wire(blob=_feed({"type": "assistant", "message": {"content": ["not a block"]}}))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "not a block" in text


def test_an_unrecognised_block_type_is_shown_verbatim(wire, client):
    wire(blob=_feed({"type": "assistant", "message": {"content": [{"type": "thinking"}]}}))
    _sign_in(client)
    assert "thinking" in client.get(f"{page.PAGE.path}?key=k").text


def test_an_event_with_no_message_is_folded_whole(wire, client):
    wire(blob=_feed({"type": "system", "subtype": "init", "cwd": "/w"}))
    _sign_in(client)
    text = client.get(f"{page.PAGE.path}?key=k").text
    assert "system" in text
    assert "/w" in text


def test_an_event_with_no_content_says_so(wire, client):
    wire(blob=_feed({"type": "assistant", "message": {"content": []}}))
    _sign_in(client)
    assert "no content" in client.get(f"{page.PAGE.path}?key=k").text


def test_a_run_that_called_no_tools_says_so(wire, client):
    wire(rows=[_run(tool_calls={})])
    assert "no tool calls" in client.get(page.PAGE.path).text


def test_a_result_spelled_as_text_blocks_renders_its_text(wire, client):
    wire(blob=_feed(_tool_result([{"type": "text", "text": "line one"}])))
    _sign_in(client)
    assert "line one" in client.get(f"{page.PAGE.path}?key=k").text


def test_a_lone_surrogate_in_a_block_type_does_not_fault_the_page(wire, client):
    # The block's own `type` is stored bytes too, and it reaches the page as a
    # LABEL rather than as body text — the one seam where a forgotten sanitize
    # would still reach the response encoder.
    wire(blob=b'{"type":"assistant","message":{"content":[{"type":"\\ud800"}]}}')
    _sign_in(client)
    assert client.get(f"{page.PAGE.path}?key=k").status_code == 200


def test_a_lone_surrogate_in_an_event_type_does_not_fault_the_page(wire, client):
    wire(blob=b'{"type":"\\ud800","detail":"x"}')
    _sign_in(client)
    assert client.get(f"{page.PAGE.path}?key=k").status_code == 200
