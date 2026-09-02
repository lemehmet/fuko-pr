"""Tests for the review-state ledger page under /ui (#235)."""

import pytest
from fastapi.testclient import TestClient

from sidecar import main, review_state
from sidecar.web import ledger


def _lane(**overrides) -> review_state.LaneStat:
    base = dict(
        repo="lemehmet/mepro",
        pr=1343,
        seat="henry",
        latest_round=3,
        last_activity="2026-07-22T20:15:00+00:00",
        counts={"open": 2, "fixed": 1, "rejected": 1, "stale": 0},
        reopened=0,
        offerable=2,
        never_offered=0,
        coverage_total=9,
        coverage_live=7,
        eligible=4,
        carried=1,
        settled=2,
    )
    base.update(overrides)
    return review_state.LaneStat(**base)


def _finding(**overrides) -> review_state.LedgerFinding:
    base = dict(
        id="0f9d1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b",
        seat="henry",
        round=2,
        head_sha="abc123",
        file="src/app.py",
        line=42,
        severity="high",
        category="bug",
        title="unchecked None device",
        body="open_source() may return None",
        evidence="src/app.py:118-166",
        status="open",
        status_reason="",
        reopened=0,
        created_at="2026-07-22T20:15:00+00:00",
        updated_at="2026-07-22T20:15:00+00:00",
    )
    base.update(overrides)
    return review_state.LedgerFinding(**base)


def _coverage(**overrides) -> review_state.LedgerCoverage:
    base = dict(
        id="1f9d1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b",
        seat="henry",
        round=1,
        head_sha="abc123",
        file="src/app.py",
        region="open_source",
        checked="null return path",
        conclusion="guarded by an explicit branch",
        evidence="src/app.py:118",
        expired_at=None,
        created_at="2026-07-22T20:15:00+00:00",
    )
    base.update(overrides)
    return review_state.LedgerCoverage(**base)


def _wire(monkeypatch, *, lanes=(), findings=(), coverage=(), configured=True, boom=False):
    """Point the page at canned reads; ``boom`` makes every read fail like an outage."""
    seen: dict = {}
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y" if configured else None)

    def fake_lanes(repo=None, pr=None, seat=None, limit=200, offset=0):
        if boom:
            raise RuntimeError("connection refused")
        seen["lanes"] = (repo, pr, seat, limit, offset)
        return review_state.LaneIndex(lanes=tuple(lanes), total=len(lanes))

    def fake_findings(repo, pr, seat=None, limit=200, offset=0):
        if boom:
            raise RuntimeError("connection refused")
        seen["findings"] = (repo, pr, seat, limit, offset)
        return review_state.FindingPage(rows=tuple(findings), total=len(findings))

    def fake_coverage(repo, pr, seat=None, limit=200, offset=0):
        if boom:
            raise RuntimeError("connection refused")
        seen["coverage"] = (repo, pr, seat, limit, offset)
        return review_state.CoveragePage(rows=tuple(coverage), total=len(coverage))

    monkeypatch.setattr(review_state, "lanes", fake_lanes)
    monkeypatch.setattr(review_state, "pr_findings", fake_findings)
    monkeypatch.setattr(review_state, "pr_coverage", fake_coverage)
    return seen, TestClient(main.app)


def test_index_lists_one_row_per_lane_with_its_counts_and_rates(monkeypatch):
    _, client = _wire(monkeypatch, lanes=[_lane(), _lane(seat="sybil", pr=1344, reopened=2)])
    resp = client.get("/ui/ledger")
    assert resp.status_code == 200
    page = resp.text
    assert "lemehmet/mepro" in page and "henry" in page and "sybil" in page
    assert "2026-07-22T20:15" in page
    assert "25%" in page and "50%" in page  # carry-forward and settle rate
    assert 'href="/ui/ledger?repo=lemehmet%2Fmepro&amp;pr=1343"' in page


def test_index_shows_a_reopened_lane_as_anomalous(monkeypatch):
    _, client = _wire(monkeypatch, lanes=[_lane(reopened=3)])
    page = client.get("/ui/ledger").text
    assert 'class="num bad">3</td>' in page


def test_index_counts_rows_no_round_was_ever_offered(monkeypatch):
    _, client = _wire(monkeypatch, lanes=[_lane(offerable=207, never_offered=7)])
    page = client.get("/ui/ledger").text
    assert 'class="num bad">7</td>' in page
    assert "not offered" in page


def test_index_rate_is_a_dash_when_a_lane_has_a_single_round(monkeypatch):
    _, client = _wire(monkeypatch, lanes=[_lane(eligible=0, carried=0, settled=0)])
    page = client.get("/ui/ledger").text
    assert '<td class="num">—</td>' in page
    assert ">0%</td>" not in page


def test_index_passes_its_filters_through(monkeypatch):
    seen, client = _wire(monkeypatch, lanes=[_lane()])
    client.get("/ui/ledger", params={"repo": "a/b", "seat": "henry", "offset": "50"})
    assert seen["lanes"] == ("a/b", None, "henry", ledger.PAGE_SIZE, 50)


def test_index_clamps_a_hostile_limit_and_offset(monkeypatch):
    seen, client = _wire(monkeypatch, lanes=[_lane()])
    client.get("/ui/ledger", params={"limit": "100000", "offset": "-3"})
    assert seen["lanes"] == (None, None, None, review_state.MAX_LEDGER_ROWS, 0)


def test_unconfigured_store_is_distinguishable_from_an_outage_and_from_empty(monkeypatch):
    _, client = _wire(monkeypatch, configured=False)
    unconfigured = client.get("/ui/ledger")
    assert unconfigured.status_code == 200
    assert "Postgres store" in unconfigured.text
    assert "sqlite-vec" in unconfigured.text
    assert "unreachable" not in unconfigured.text

    _, client = _wire(monkeypatch, boom=True)
    outage = client.get("/ui/ledger")
    assert outage.status_code == 200
    assert "unreachable" in outage.text
    assert "notice danger" in outage.text
    assert "Postgres store (FUKO_DATABASE_URL unset)" not in outage.text

    _, client = _wire(monkeypatch)
    empty = client.get("/ui/ledger")
    assert empty.status_code == 200
    assert "no review state recorded yet" in empty.text
    assert "notice danger" not in empty.text and "notice warn" not in empty.text


def test_detail_shows_every_finding_status_and_marks_a_reopened_row(monkeypatch):
    _, client = _wire(
        monkeypatch,
        lanes=[_lane()],
        findings=[
            _finding(),
            _finding(status="stale", status_reason="file is gone"),
            _finding(status="rejected", status_reason="not a bug", reopened=2),
        ],
    )
    page = client.get("/ui/ledger", params={"repo": "lemehmet/mepro", "pr": "1343"}).text
    assert "lemehmet/mepro" in page and "#1343" in page
    for status in ("open", "stale", "rejected"):
        assert f">{status}</span>" in page
    assert "file is gone" in page and "not a bug" in page
    assert "reopened ×2" in page
    assert "unchecked None device" in page and "open_source() may return None" in page
    assert "src/app.py:118-166" in page


def test_detail_links_a_finding_to_its_blob_with_the_path_encoded(monkeypatch):
    _, client = _wire(
        monkeypatch,
        lanes=[_lane()],
        findings=[_finding(file='src/we"ird#?.py', line=7, head_sha="ab c/1")],
    )
    page = client.get("/ui/ledger", params={"repo": "lemehmet/mepro", "pr": "1343"}).text
    assert (
        'href="https://github.com/lemehmet/mepro/blob/ab%20c%2F1/src/we%22ird%23%3F.py#L7"' in page
    )
    assert 'we"ird' not in page.replace("&quot;", "")


def test_detail_renders_a_finding_with_no_line_without_an_anchor(monkeypatch):
    _, client = _wire(monkeypatch, lanes=[_lane()], findings=[_finding(line=None)])
    page = client.get("/ui/ledger", params={"repo": "lemehmet/mepro", "pr": "1343"}).text
    assert 'href="https://github.com/lemehmet/mepro/blob/abc123/src/app.py"' in page


def test_detail_leaves_a_finding_without_a_head_commit_as_inert_text(monkeypatch):
    _, client = _wire(monkeypatch, lanes=[_lane()], findings=[_finding(head_sha="")])
    page = client.get("/ui/ledger", params={"repo": "lemehmet/mepro", "pr": "1343"}).text
    assert "/blob/" not in page
    assert "src/app.py:42" in page


def test_detail_escapes_markup_from_a_finding_a_seat_and_a_repo(monkeypatch):
    _, client = _wire(
        monkeypatch,
        lanes=[_lane(repo="<b>evil</b>/x", seat="<img src=x>")],
        findings=[
            _finding(
                body="<script>alert(1)</script>",
                title="<script>t</script>",
                seat="<img src=x onerror=1>",
                status_reason="<script>r</script>",
                evidence="<script>e</script>",
            )
        ],
    )
    page = client.get("/ui/ledger", params={"repo": "<b>evil</b>/x", "pr": "1343"}).text
    assert "<script>" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_detail_body_is_still_reachable_when_it_is_long(monkeypatch):
    body = "x" * 500
    _, client = _wire(monkeypatch, lanes=[_lane()], findings=[_finding(title="t" * 300, body=body)])
    page = client.get("/ui/ledger", params={"repo": "lemehmet/mepro", "pr": "1343"}).text
    assert "<details>" in page and body in page


def test_coverage_shows_expired_entries_beside_live_ones_as_what_was_examined(monkeypatch):
    _, client = _wire(
        monkeypatch,
        lanes=[_lane()],
        coverage=[
            _coverage(),
            _coverage(expired_at="2026-07-23T09:00:00+00:00", conclusion="looked fine then"),
        ],
    )
    page = client.get(
        "/ui/ledger", params={"repo": "lemehmet/mepro", "pr": "1343", "show": "coverage"}
    ).text
    assert "examined" in page
    assert ">live</span>" in page and ">expired</span>" in page
    assert "2026-07-23T09:00" in page
    assert "guarded by an explicit branch" in page and "looked fine then" in page
    assert "open_source" in page
    assert "knowledge" not in page.lower().replace("knowledge base", "")


def test_coverage_escapes_its_own_free_text(monkeypatch):
    _, client = _wire(
        monkeypatch,
        lanes=[_lane()],
        coverage=[
            _coverage(
                checked="<script>c</script>",
                conclusion="<script>k</script>" + "y" * 300,
                region="<img src=x>",
                evidence="<script>v</script>",
            )
        ],
    )
    page = client.get(
        "/ui/ledger", params={"repo": "lemehmet/mepro", "pr": "1343", "show": "coverage"}
    ).text
    assert "<script>" not in page and "<img src=x" not in page


def test_detail_pages_one_ledger_at_a_time(monkeypatch):
    seen, client = _wire(
        monkeypatch,
        lanes=[_lane()],
        findings=[_finding()],
        coverage=[_coverage()],
    )
    client.get("/ui/ledger", params={"repo": "a/b", "pr": "7", "seat": "henry", "offset": "50"})
    assert seen["findings"] == ("a/b", 7, "henry", ledger.PAGE_SIZE, 50)
    assert "coverage" not in seen

    seen, client = _wire(monkeypatch, lanes=[_lane()], coverage=[_coverage()])
    client.get("/ui/ledger", params={"repo": "a/b", "pr": "7", "show": "coverage"})
    assert seen["coverage"] == ("a/b", 7, None, ledger.PAGE_SIZE, 0)
    assert "findings" not in seen


def test_detail_falls_back_to_findings_for_an_unknown_section(monkeypatch):
    seen, client = _wire(monkeypatch, lanes=[_lane()], findings=[_finding()])
    page = client.get("/ui/ledger", params={"repo": "a/b", "pr": "7", "show": "../etc"}).text
    assert seen["findings"][0:2] == ("a/b", 7)
    assert "what a round claimed" in page


def test_detail_cross_links_back_to_the_index_the_pull_request_and_the_metrics_page(monkeypatch):
    _, client = _wire(monkeypatch, lanes=[_lane()], findings=[_finding()])
    page = client.get("/ui/ledger", params={"repo": "lemehmet/mepro", "pr": "1343"}).text
    assert 'href="https://github.com/lemehmet/mepro/pull/1343"' in page
    assert 'href="/ui/metrics?repo=lemehmet%2Fmepro"' in page
    assert 'href="/ui/ledger"' in page


def test_detail_degrades_without_looking_like_an_empty_ledger(monkeypatch):
    _, client = _wire(monkeypatch, boom=True)
    resp = client.get("/ui/ledger", params={"repo": "a/b", "pr": "7"})
    assert resp.status_code == 200
    assert "unreachable" in resp.text


def test_a_pr_without_a_repo_stays_on_the_index(monkeypatch):
    seen, client = _wire(monkeypatch, lanes=[_lane()])
    page = client.get("/ui/ledger", params={"pr": "1343"}).text
    assert seen["lanes"] == (None, 1343, None, ledger.PAGE_SIZE, 0)
    assert "Review state ledgers" in page


@pytest.mark.parametrize(
    "params",
    [
        {"repo": "a/b", "pr": "", "seat": ""},  # the form's own default submission
        {"repo": "", "pr": "", "seat": ""},
        {"repo": "a/b", "pr": "", "seat": "henry"},
    ],
)
def test_the_filter_form_submits_an_empty_pr_field_and_still_gets_a_page(monkeypatch, params):
    """A browser submits every text input, so an untouched PR box arrives as ``pr=``.

    An ``int | None`` parameter rejects that with a 422 -- ``Optional`` admits
    an ABSENT parameter, not an empty one -- which would break the form on the
    exact clicks it exists for: filtering by repository alone, or by seat alone.
    """
    seen, client = _wire(monkeypatch, lanes=[_lane()])
    resp = client.get("/ui/ledger", params=params)
    assert resp.status_code == 200
    assert "Review state ledgers" in resp.text
    assert seen["lanes"][1] is None


@pytest.mark.parametrize("value", ["abc", "0", "-3", "1.5", "9" * 14])
def test_an_unreadable_pr_filter_drops_itself_rather_than_the_page(monkeypatch, value):
    """A typo in one filter must not become a 422 or an "unreachable store" notice."""
    seen, client = _wire(monkeypatch, lanes=[_lane()])
    resp = client.get("/ui/ledger", params={"repo": "a/b", "pr": value})
    assert resp.status_code == 200
    assert "Review state ledgers" in resp.text
    assert seen["lanes"][1] is None


@pytest.mark.parametrize("path", ["/ui/ledger", "/ui/ledger?repo=a%2Fb&pr=7"])
def test_the_page_is_open_and_read_only(monkeypatch, path):
    _, client = _wire(monkeypatch, lanes=[_lane()], findings=[_finding()])
    assert client.get(path).status_code == 200
    assert client.post(path).status_code == 405


def test_render_index_is_pure_on_empty_input():
    html = ledger.render_index(
        index=review_state.LaneIndex(),
        repo=None,
        pr=None,
        seat=None,
        offset=0,
        limit=50,
        db_enabled=True,
    )
    assert "no review state recorded yet" in html
    assert html.startswith("<!doctype html>")


def test_render_detail_is_pure_on_empty_input():
    html = ledger.render_detail(
        repo="a/b",
        pr=7,
        index=review_state.LaneIndex(),
        findings=review_state.FindingPage(),
        coverage=review_state.CoveragePage(),
        show="coverage",
        seat=None,
        offset=0,
        limit=50,
        db_enabled=True,
    )
    assert "no coverage recorded for this pull request" in html
    assert "no review state recorded for this pull request" in html


def test_pager_carries_the_active_filters(monkeypatch):
    _, client = _wire(monkeypatch, lanes=[_lane(seat=f"s{n}") for n in range(3)])
    page = client.get("/ui/ledger", params={"repo": "a/b", "limit": "2"}).text
    assert "repo=a%2Fb" in page and "offset=2" in page and "limit=2" in page
