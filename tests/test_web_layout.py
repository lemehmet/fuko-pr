"""Tests for the shared web chrome: page registry, nav, components, legacy redirects (#87)."""

import pytest
from fastapi.testclient import TestClient

from sidecar import main
from sidecar.web import components as c
from sidecar.web import layout


def _client():
    return TestClient(main.app)


def test_every_registered_page_appears_in_nav():
    rendered = layout.nav(active="")
    for page in layout.PAGES:
        assert f'href="{page.path}"' in rendered
        assert page.title in rendered


def test_nav_marks_only_the_active_page():
    rendered = layout.nav(active="metrics")
    assert rendered.count('class="active"') == 1
    assert 'href="/ui/metrics" class="active"' in rendered


def test_nav_extra_is_appended_inside_the_bar():
    rendered = layout.nav(active="", extra="<span id=x></span>")
    assert rendered.endswith("<span id=x></span></nav>")


def test_page_lookup_rejects_an_unregistered_slug():
    assert layout.page("metrics").path == "/ui/metrics"
    with pytest.raises(layout.UnregisteredPageError):
        layout.page("nope")


def test_document_wraps_body_in_shared_chrome():
    html = layout.document(title="T", body="<p>hi</p>", active="metrics")
    assert html.startswith("<!doctype html>")
    assert "<title>T</title>" in html
    assert "<main><p>hi</p></main>" in html
    assert "<nav>" in html and "<footer>" in html


def test_document_escapes_its_title():
    assert "<title>&lt;script&gt;</title>" in layout.document(title="<script>", body="")


def test_legacy_metrics_view_redirects_preserving_query():
    resp = _client().get("/metrics/view?repo=a%2Fb&days=7", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/ui/metrics?repo=a%2Fb&days=7"


def test_legacy_metrics_view_redirects_without_query():
    resp = _client().get("/metrics/view", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/ui/metrics"


def test_ui_index_redirects_to_the_first_page():
    resp = _client().get("/ui", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/ui/metrics"


def test_esc_neutralizes_markup_and_renders_none_as_dash():
    assert c.esc("<b>&") == "&lt;b&gt;&amp;"
    assert c.esc(None) == "&mdash;"


def test_attrs_escapes_quotes_and_drops_absent_values():
    assert c.attrs(href='a"b') == ' href="a&quot;b"'
    assert c.attrs(class_=None, hidden=False) == ""
    assert c.attrs(selected=True) == " selected"
    assert c.attrs(data_id=3) == ' data-id="3"'


def test_cells_carry_alignment_and_escape_content():
    assert c.cell("<i>", numeric=True) == '<td class="num">&lt;i&gt;</td>'
    assert c.cell("x", css="muted") == '<td class="muted">x</td>'
    assert c.raw_cell("<b>ok</b>") == "<td><b>ok</b></td>"


def test_link_and_badge_escape_their_text():
    assert c.link("/p", "<x>") == '<a href="/p">&lt;x&gt;</a>'
    assert "&lt;x&gt;" in c.badge("<x>")


@pytest.mark.parametrize(
    "href",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "java\tscript:alert(1)",
        "  javascript:alert(1)  ",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
    ],
)
def test_link_refuses_an_unsafe_scheme(href):
    assert c.safe_href(href) is None
    assert c.link(href, "click me") == '<span class="muted">click me</span>'
    assert "href" not in c.link(href, "click me")


@pytest.mark.parametrize(
    "href",
    [
        "https://github.com/o/r/pull/1",
        "http://fuko.nonni:8000/ui",
        "mailto:a@b.example",
        "/ui/kb?repo=o%2Fr",
        "?offset=25",
        "#section",
        "relative/path.md",
        "a/b:c/d",
    ],
)
def test_link_keeps_relative_and_web_urls(href):
    assert c.safe_href(href) == href.strip()
    assert c.link(href, "x").startswith("<a href=")


def test_form_value_keeps_falsy_non_none_values():
    assert c.form_value(0) == "0"
    assert c.form_value(False) == "False"
    assert c.form_value(None) == ""
    assert c.form_value("") == ""


def test_form_helpers_round_trip_a_zero():
    assert '<input name="n" value="0">' in c.field("N", "n", 0)
    assert '<textarea name="n">0</textarea>' in c.textarea("N", "n", 0)
    assert '<option value="0" selected>' in c.select("N", "n", [("0", "zero")], value=0)


def test_table_renders_headers_and_falls_back_to_the_empty_notice():
    rendered = c.table([("a", False), ("n", True)], ["<tr><td>1</td></tr>"], "nothing")
    assert "<th>a</th>" in rendered and '<th class="num">n</th>' in rendered
    assert c.table([("a", False)], [], "nothing") == '<p class="muted">nothing</p>'


def test_notice_carries_its_kind_and_escapes_text():
    assert c.notice("<x>", kind="danger") == '<p class="notice danger">&lt;x&gt;</p>'


def test_form_helpers_escape_values():
    assert '<input name="q" value="&quot;">' in c.field("Q", "q", '"')
    assert c.field("Q", "q", None).endswith('value=""></label>')
    assert "&lt;script&gt;" in c.textarea("T", "t", "<script>")
    assert "&lt;script&gt;" in c.hidden(t="<script>")
    assert c.hidden(a=None) == ""


def test_select_marks_the_matching_option():
    rendered = c.select("S", "s", [("a", "A"), ("b", "B")], value="b")
    assert '<option value="b" selected>B</option>' in rendered
    assert '<option value="a">A</option>' in rendered


def test_query_string_drops_empties():
    assert c.query_string({"a": 1, "b": "", "c": None}) == "?a=1"
    assert c.query_string({}) == ""


def test_pager_is_absent_when_everything_fits():
    assert c.pager("/p", {}, offset=0, limit=50, total=12) == ""


def test_pager_carries_filters_into_both_links():
    rendered = c.pager("/p", {"repo": "o/r"}, offset=50, limit=50, total=120)
    assert "51&ndash;100 of 120" in rendered
    assert "repo=o%2Fr&amp;offset=0&amp;limit=50" in rendered
    assert "repo=o%2Fr&amp;offset=100&amp;limit=50" in rendered


def test_pager_omits_prev_on_the_first_page_and_next_on_the_last():
    first = c.pager("/p", {}, offset=0, limit=50, total=120)
    last = c.pager("/p", {}, offset=100, limit=50, total=120)
    assert "prev" not in first and "next" in first
    assert "prev" in last and "next" not in last
