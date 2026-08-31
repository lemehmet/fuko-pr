"""Tests for the review-ledger policy: open findings (#156) and coverage (#157).

The store is faked in memory rather than mocked call-by-call: the acceptance
criterion is a ROUND TRIP ("a finding reported in round N and unaddressed is
present in round N+1's prompt"; "coverage recorded in round N is gone from round
N+1's prompt once the delta touches its file"), which per-call assertions cannot
express.
"""

import os

import pytest

from sidecar import review_state
from sidecar.reviewer import ledger
from sidecar.reviewer.ledger import CarriedState, carry_in, settle
from sidecar.reviewer.prompt import (
    COVERAGE_ADVISORY,
    EXAMINED_REQUIRED_FIELDS,
    MAX_PRIOR_COVERAGE,
    AgenticFinding,
    ExaminedRegion,
    PriorCoverage,
    PriorFinding,
    PriorFindingStatus,
    PriorState,
)

REPO, PR, SEAT = "o/r", 9, "dorian"


def _examined(**overrides) -> ExaminedRegion:
    base = dict(
        file="src/util.py",
        region="open_source",
        checked="whether every caller handles a None device",
        conclusion="all four callers branch on None before use",
        evidence="src/util.py:118-166, src/shim.py:402",
    )
    base.update(overrides)
    return ExaminedRegion(**base)


def _finding(**overrides) -> AgenticFinding:
    base = dict(
        file="src/app.py",
        line=42,
        severity="high",
        category="bug",
        title="unchecked None device",
        body="open_source() may return None",
        evidence="src/app.py:118-166",
    )
    base.update(overrides)
    return AgenticFinding(**base)


class _Store:
    """An in-memory stand-in for the ``review_state`` primitives both ledgers use."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.coverage: list[dict] = []
        self.touched: list[str] = []
        self._n = 0

    def record_findings(self, repo, pr, seat, round, head_sha, findings):
        for f in findings:
            self._n += 1
            self.rows[f"row-{self._n}"] = {
                "key": (repo, pr, seat),
                "round": round,
                "head_sha": head_sha,
                "finding": f,
                "status": "open",
                "reason": "",
                "reopened": 0,
            }
        return len(findings)

    def open_findings(self, repo, pr, seat, limit=review_state.MAX_OPEN_FINDINGS):
        rows = [
            review_state.StoredFinding(
                id=row_id,
                prior=PriorFinding(
                    file=row["finding"].file,
                    title=row["finding"].title,
                    body=row["finding"].body,
                    line=row["finding"].line,
                    severity=row["finding"].severity,
                    category=row["finding"].category,
                    round=row["round"],
                    evidence=row["finding"].evidence,
                ),
            )
            for row_id, row in self.rows.items()
            if row["key"] == (repo, pr, seat) and row["status"] == "open"
        ]
        # Cut the way the real read cuts: insertion order is oldest-first, so
        # the rows past the cap are the NEWEST -- the case #173 is about.
        return review_state.OpenLedger(
            rows=tuple(rows[:limit]), truncated=max(0, len(rows) - limit)
        )

    def next_round(self, repo, pr, seat):
        # Both ledgers, like the real read: a round that recorded only coverage
        # still happened, and re-issuing its number would put two rounds behind
        # one label.
        rounds = [r["round"] for r in self.rows.values() if r["key"] == (repo, pr, seat)]
        rounds += [c["round"] for c in self.coverage if c["key"] == (repo, pr, seat)]
        return max(rounds, default=0) + 1

    def transition(self, repo, pr, seat, finding_id, status, reason=""):
        row = self.rows.get(finding_id)
        # The lane is matched, not merely accepted: the real UPDATE now has
        # `AND repo = %s AND pr = %s AND seat = %s` in its WHERE (#171), and a
        # fake that ignored it could not fail a cross-seat write.
        if row is None or row["key"] != (repo, pr, seat) or row["status"] != "open":
            return False
        row["status"], row["reason"] = status, reason
        return True

    def touch_findings(self, repo, pr, seat, finding_ids):
        mine = [i for i in finding_ids if self.rows.get(i, {}).get("key") == (repo, pr, seat)]
        self.touched.extend(mine)
        return len(mine)

    def settled_findings(self, repo, pr, seat, limit=review_state.MAX_SETTLED_FINDINGS):
        rows = [
            review_state.SettledFinding(
                id=row_id,
                file=row["finding"].file,
                title=row["finding"].title,
                status=row["status"],
                round=row["round"],
                reason=row["reason"],
            )
            for row_id, row in self.rows.items()
            if row["key"] == (repo, pr, seat) and row["status"] in review_state.REOPENABLE_STATUSES
        ]
        # Most recently closed first, like the real read's `updated_at DESC`:
        # insertion order stands in for closure order in this fake.
        return tuple(reversed(rows))[:limit]

    def reopen(self, repo, pr, seat, finding_id, reason):
        row = self.rows.get(finding_id)
        if (
            row is None
            or row["key"] != (repo, pr, seat)
            or row["status"] not in review_state.REOPENABLE_STATUSES
        ):
            return False
        row["status"], row["reason"] = "open", reason
        row["reopened"] += 1
        return True

    def record_coverage(self, repo, pr, seat, round, head_sha, regions):
        for region in regions:
            self.coverage.append(
                {"key": (repo, pr, seat), "round": round, "region": region, "expired": False}
            )
        return len(regions)

    def live_coverage(self, repo, pr, seat, limit=review_state.MAX_LIVE_COVERAGE):
        return [
            PriorCoverage(
                file=row["region"].file,
                checked=row["region"].checked,
                conclusion=row["region"].conclusion,
                evidence=row["region"].evidence,
                region=row["region"].region,
                round=row["round"],
            )
            for row in self.coverage
            if row["key"] == (repo, pr, seat) and not row["expired"]
        ][:limit]

    def expire_coverage(self, repo, pr, seat, files=None):
        expired = 0
        for row in self.coverage:
            if row["key"] != (repo, pr, seat) or row["expired"]:
                continue
            if files is None or row["region"].file in files:
                row["expired"] = True
                expired += 1
        return expired


@pytest.fixture
def store(monkeypatch):
    """Replace the review-state primitives the ledger calls with a live fake."""
    fake = _Store()
    for name in (
        "record_findings",
        "open_findings",
        "next_round",
        "transition",
        "touch_findings",
        "settled_findings",
        "reopen",
        "record_coverage",
        "live_coverage",
        "expire_coverage",
    ):
        monkeypatch.setattr(review_state, name, getattr(fake, name))
    return fake


def test_an_unaddressed_finding_is_in_the_next_rounds_prompt(store):
    """The acceptance criterion, and the whole point of Tier 1."""
    first = carry_in(REPO, PR, SEAT)
    assert first.text == "" and first.round == 1

    assert settle(
        first,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(title="leaks the handle"), _finding(title="drops the error")],
    ) == ledger.Settlement(recorded=2)

    second = carry_in(REPO, PR, SEAT)
    assert second.round == 2
    assert "leaks the handle" in second.text and "drops the error" in second.text
    assert list(second.rows) == ["p1", "p2"]
    # The claim AND its grounding make the trip (#174). Asserted here, on the
    # store round trip, because the render-unit tests build PriorFinding by hand
    # and so cannot see a read that drops the column again.
    assert "evidence: src/app.py:118-166" in second.text


def test_settling_closes_only_what_the_round_settled(store):
    """`fixed` closes, `still_open` re-asserts, an unmentioned finding survives."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(title="one"), _finding(title="two"), _finding(title="three")],
    )
    carried = carry_in(REPO, PR, SEAT)

    outcome = settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[
            PriorFindingStatus(id="p1", status="fixed", reason="rewritten in 9f2a1c"),
            PriorFindingStatus(id="p2", status="still_open", reason="still there at L44"),
        ],
    )

    assert outcome == ledger.Settlement(closed=1, reasserted=1, recorded=0)
    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT).rows] == ["two", "three"]
    assert store.touched == [carried.rows["p2"]]


def test_re_reporting_a_carried_finding_touches_it_instead_of_duplicating_it(store):
    """A round that re-reports rather than settles must not mint a second row.

    Two open rows for one claim compound per round, and past the read cap the
    rows cut are the NEWEST -- never rendered, never minted an id, never touched,
    so they age out unsettled (reported by `carry_in`, but no more reachable).
    That is this module's own loss arriving by volume.
    """
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )
    carried = carry_in(REPO, PR, SEAT)

    outcome = settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        findings=[
            # Same claim, re-reported instead of settled -- and cased differently,
            # since the match is on the claim, not on the model's capitalisation.
            _finding(file="src/a.py", title="Leaks The Handle"),
            _finding(file="src/b.py", title="drops the error"),
        ],
    )

    # Named, not counted: the surviving row keeps the EARLIER body, so which
    # claim was suppressed is the one thing the store cannot answer afterwards.
    assert outcome == ledger.Settlement(
        reasserted=1, recorded=1, deduped=("src/a.py: Leaks The Handle",)
    )
    assert sorted(f.prior.title for f in store.open_findings(REPO, PR, SEAT).rows) == [
        "drops the error",
        "leaks the handle",
    ]
    # Touched, not left to age out: a re-report is evidence a round looked at it.
    assert store.touched == [carried.rows["p1"]]


def test_a_round_that_closes_and_republishes_a_claim_leaves_it_open(store):
    """A round contradicting its own verdict resolves toward open (#177).

    The injected-verdict shape: the fenced channel says `fixed`, the round's own
    reading of the code publishes the claim anyway. One row, still open, and the
    contradiction named rather than left as two rows or as a false closure."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )
    carried = carry_in(REPO, PR, SEAT)
    row_id = carried.rows["p1"]

    outcome = settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[PriorFindingStatus(id="p1", status="fixed", reason="closed in 9f2a1c")],
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )

    assert outcome == ledger.Settlement(
        closed=1, recorded=0, reopened=("src/a.py: leaks the handle",)
    )
    assert list(store.rows) == [row_id]
    assert store.rows[row_id]["status"] == "open" and store.rows[row_id]["reopened"] == 1
    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT).rows] == ["leaks the handle"]


def test_a_closed_finding_is_re_raised_by_a_later_round_that_finds_it_again(store):
    """#177's acceptance criterion, end to end.

    Round 1 reports, round 2 closes it by verdict, round 3 independently finds it
    again -- and round 4 is offered the SAME row, not a second one, carrying the
    count that says a verdict on it was contradicted."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )
    carried = carry_in(REPO, PR, SEAT)
    row_id = carried.rows["p1"]
    settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[PriorFindingStatus(id="p1", status="fixed", reason="rewritten in 9f2a1c")],
    )
    assert store.rows[row_id]["status"] == "fixed"
    assert carry_in(REPO, PR, SEAT).rows == {}

    outcome = settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head3",
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )

    assert outcome == ledger.Settlement(recorded=0, reopened=("src/a.py: leaks the handle",))
    assert list(store.rows) == [row_id]
    assert store.rows[row_id]["reopened"] == 1
    # The reason a reader lands on carries the closure it reverses, not just the
    # reversal: the column holds one explanation and this is now the whole story.
    reason = store.rows[row_id]["reason"]
    # Round 2 closed a row and recorded none, so `next_round` -- max(round) + 1
    # over the rows this seat has WRITTEN -- labels the re-raising round 2.
    assert "re-raised in round 2" in reason
    # Round 1 RECORDED the finding; round 2's verdict closed it. `transition`
    # writes no round, so the closing round is persisted nowhere -- the line
    # names the round it can actually account for, and says which one that is.
    assert "contradicts fixed" in reason and "recorded in round 1" in reason
    assert "fixed in round 1" not in reason
    assert "rewritten in 9f2a1c" in reason
    assert list(carry_in(REPO, PR, SEAT).rows.values()) == [row_id]


def test_a_closure_reason_at_the_cap_cannot_clip_the_re_raises_own_provenance():
    """The composed line's structured half outlives the store's clip; the prose does not.

    ``reopen`` clips what it writes at ``MAX_TEXT``, and the closure reason it
    carries forward is model text already stored at up to that same cap. With the
    prose in the middle, a closure reason near the cap pushed "recorded in round
    N" -- the provenance this line exists to add -- past the clip, silently. So
    the fixed parts lead, and truncation can only cost the oldest prose
    (qwen3.8-max, #189).
    """
    reason = ledger._reopen_reason(
        review_state.SettledFinding(
            id="row-1",
            file="src/a.py",
            title="leaks the handle",
            status="fixed",
            round=1,
            reason="x" * (review_state.MAX_TEXT * 2),
        ),
        3,
    )
    stored = review_state._clip(reason)

    assert len(stored) == review_state.MAX_TEXT
    assert stored.startswith("re-raised in round 3: an independent finding contradicts fixed,")
    assert "on a finding recorded in round 1" in stored
    # What the clip took is the tail of the model's prose, which is the only part
    # of the line a reader can lose without losing the sequence itself.
    assert stored.endswith("x")


def test_a_stale_row_is_not_re_raised(store):
    """`stale` is fuko's own retirement, not a verdict -- softening it is #175's."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )
    carried = carry_in(REPO, PR, SEAT)
    row_id = carried.rows["p1"]
    review_state.transition(REPO, PR, SEAT, row_id, "stale", "file absent from the tree at head2")

    outcome = settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head3",
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )

    assert outcome == ledger.Settlement(recorded=1)
    assert store.rows[row_id]["status"] == "stale" and store.rows[row_id]["reopened"] == 0


def test_two_published_findings_on_one_claim_re_raise_one_row(store):
    """One claim, one open row -- however many times the round publishes it.

    The re-raise has to close BOTH doors to the compounding the dedup path
    exists to stop: popping the candidate stops a second reopen, and putting the
    re-raised row into ``still_open`` stops the second finding being recorded
    fresh alongside it. Without the second half the round ends with the reopened
    row AND a duplicate of it, which is the two-rows-for-one-claim state the
    dedup was written to prevent (CodeRabbit, #189).
    """
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[
            _finding(file="src/a.py", title="leaks the handle"),
            _finding(file="src/a.py", title="leaks the handle"),
        ],
    )
    carried = carry_in(REPO, PR, SEAT)
    settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[
            PriorFindingStatus(id="p1", status="fixed", reason="one"),
            PriorFindingStatus(id="p2", status="fixed", reason="two"),
        ],
    )

    outcome = settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head3",
        findings=[
            _finding(file="src/a.py", title="leaks the handle"),
            _finding(file="src/a.py", title="Leaks The Handle"),
        ],
    )

    assert outcome.reopened == ("src/a.py: leaks the handle",)
    # The duplicate is deduped against the row its twin just re-raised, not
    # recorded beside it: the claim ends the round on exactly one open row.
    assert outcome.deduped == ("src/a.py: Leaks The Handle",)
    assert outcome.recorded == 0
    assert [row["status"] for row in store.rows.values()].count("open") == 1


def test_a_reopen_the_store_refuses_records_the_claim_instead(store, monkeypatch):
    """Best-effort throughout: a failed re-raise loses the LINK, never the claim."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )
    carried = carry_in(REPO, PR, SEAT)
    row_id = carried.rows["p1"]
    settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[PriorFindingStatus(id="p1", status="fixed", reason="rewritten")],
    )
    monkeypatch.setattr(review_state, "reopen", lambda *a, **kw: False)

    outcome = settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head3",
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )

    assert outcome == ledger.Settlement(recorded=1)
    assert store.rows[row_id]["status"] == "fixed"
    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT).rows] == ["leaks the handle"]


def test_a_round_with_no_findings_never_reads_the_closed_ledger(store, monkeypatch):
    """Nothing can be re-raised without a published claim, so nothing is read."""
    monkeypatch.setattr(review_state, "settled_findings", _forbidden("settled_findings"))

    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        prior_status=[PriorFindingStatus(id="p1", status="fixed", reason="nothing carried")],
    )


def test_rejected_without_a_reason_does_not_close_a_row(store):
    """A seat must not close its predecessor's finding by assertion alone."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(title="one"), _finding(title="two")],
    )
    carried = carry_in(REPO, PR, SEAT)

    outcome = settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[
            PriorFindingStatus(id="p1", status="rejected", reason="   "),
            PriorFindingStatus(id="p2", status="rejected", reason="the caller already guards it"),
        ],
    )

    assert outcome == ledger.Settlement(closed=1, reasserted=1, recorded=0)
    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT).rows] == ["one"]
    assert store.rows[carried.rows["p2"]]["status"] == "rejected"


def test_a_verdict_on_an_unminted_id_transitions_nothing(store):
    """`accepted_status` owns the gate; the ledger must not route around it."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(title="one")],
    )
    carried = carry_in(REPO, PR, SEAT)

    settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[
            PriorFindingStatus(id="p9", status="fixed", reason="not a row this round was handed"),
            PriorFindingStatus(id="p1", status="resolved", reason="off-vocabulary verdict"),
        ],
    )

    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT).rows] == ["one"]
    assert store.touched == []


def test_a_minted_id_without_a_row_is_skipped(store):
    """Defensive: a state whose ids and rows disagree must not raise mid-round."""
    state = PriorState("p1 ...", {"p1": PriorFinding(file="a.py", title="t")})
    outcome = settle(
        CarriedState(state=state, rows={}),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[PriorFindingStatus(id="p1", status="fixed", reason="gone")],
    )

    assert outcome == ledger.Settlement()


def test_a_minted_id_without_a_row_is_skipped_by_the_dedup_pass_too(store):
    """The ids/rows disagreement reaches the dedup pass as well as the verdict pass.

    Both look up `carried.rows` for a key drawn from `carried.state.ids`, and the
    guard has to hold on both. A minted id with no row is a claim fuko cannot map
    back to the store, so it is neither settled nor treated as already-held: the
    re-report is recorded FRESH rather than silently folded onto a row that does
    not exist.
    """
    state = PriorState("p1 ...", {"p1": PriorFinding(file="a.py", title="t")})

    outcome = settle(
        CarriedState(state=state, rows={}),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        prior_status=[PriorFindingStatus(id="p1", status="fixed", reason="gone")],
        findings=[_finding(file="a.py", title="t")],
    )

    assert outcome == ledger.Settlement(recorded=1)
    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT).rows] == ["t"]


def test_ids_map_to_rows_in_the_renderers_own_minting_order(store):
    """The pN -> row mapping is derived from the renderer, not re-enumerated."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(title="first"), _finding(title="second"), _finding(title="third")],
    )
    carried = carry_in(REPO, PR, SEAT)

    for minted, row_id in carried.rows.items():
        assert carried.state.ids[minted].title == store.rows[row_id]["finding"].title


def test_a_finding_whose_file_is_gone_is_retired_not_re_offered(store, tmp_path):
    """The one closure fuko makes on its own authority: an unreachable anchor."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.py").write_text("x = 1\n")
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/kept.py"), _finding(file="src/deleted.py", title="gone")],
    )

    carried = carry_in(REPO, PR, SEAT, str(tmp_path), "head2")

    assert "gone" not in carried.text
    assert [f.prior.file for f in store.open_findings(REPO, PR, SEAT).rows] == ["src/kept.py"]
    retired = [r for r in store.rows.values() if r["status"] == "stale"]
    assert len(retired) == 1 and "head2" in retired[0]["reason"]


@pytest.mark.parametrize("root", ["", "/definitely/not/a/checkout"])
def test_an_unusable_checkout_root_retires_nothing(store, root):
    """Under a root that does not exist every path is 'missing' -- retire none."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/deleted.py")],
    )

    carried = carry_in(REPO, PR, SEAT, root, "head2")

    assert list(carried.rows) == ["p1"]
    assert all(r["status"] == "open" for r in store.rows.values())


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../../etc/passwd",
        "..",
        "",
        "src/\x00app.py",
        # Too long for the filesystem to look up (ENAMETOOLONG, not ENOENT), so
        # `lstat` raises a plain OSError -- unjudgeable, never "deleted". Lives
        # at the root of the checkout on purpose: under a missing parent the
        # kernel reports ENOENT for the parent and never reaches this name.
        "a" * 400 + ".py",
    ],
)
def test_a_path_the_ledger_cannot_judge_keeps_its_finding(store, tmp_path, path):
    """Absolute, escaping and unstattable anchors are unjudgeable, not absent."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file=path)],
    )

    carried = carry_in(REPO, PR, SEAT, str(tmp_path), "head2")

    assert list(carried.rows) == ["p1"]
    assert all(r["status"] == "open" for r in store.rows.values())


def test_a_dangling_symlink_is_still_a_path_the_tree_carries(store, tmp_path):
    """`lexists`, not `exists`: a broken link is not a deleted file."""
    (tmp_path / "link.py").symlink_to(tmp_path / "nowhere.py")
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="link.py")],
    )

    assert list(carry_in(REPO, PR, SEAT, str(tmp_path), "head2").rows) == ["p1"]


def test_a_symlinked_parent_out_of_the_tree_is_unjudgeable(store, tmp_path):
    """`lstat` follows PARENTS, so lexical containment alone answers off-host.

    The checkout carries `link` -> somewhere outside it. Nothing under that link
    is a path this tree can speak for, and the host's answer for
    `outside/host-only.py` -- absent here -- must not retire the finding.
    """
    root = tmp_path / "checkout"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="link/host-only.py")],
    )

    carried = carry_in(REPO, PR, SEAT, str(root), "head2")

    assert list(carried.rows) == ["p1"]
    assert all(r["status"] == "open" for r in store.rows.values())


def test_a_symlinked_parent_inside_the_tree_still_judges(store, tmp_path):
    """The rule is 'lands outside', not 'is a link': an in-tree link still counts."""
    (tmp_path / "src").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "src", target_is_directory=True)
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="alias/deleted.py")],
    )

    carried = carry_in(REPO, PR, SEAT, str(tmp_path), "head2")

    assert carried.rows == {}
    assert [r["status"] for r in store.rows.values()] == ["stale"]


def test_a_root_that_will_not_resolve_retires_nothing(store, tmp_path, monkeypatch):
    """`carry_in` promises to be best-effort: a refusing filesystem is not evidence."""
    (tmp_path / "src").mkdir()
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/deleted.py")],
    )
    monkeypatch.setattr(
        ledger.os.path, "realpath", lambda *a, **kw: (_ for _ in ()).throw(OSError("nope"))
    )

    carried = carry_in(REPO, PR, SEAT, str(tmp_path), "head2")

    assert list(carried.rows) == ["p1"]
    assert all(r["status"] == "open" for r in store.rows.values())


def test_a_parent_that_will_not_resolve_keeps_its_finding(store, tmp_path, monkeypatch):
    """Same promise one level down: the root resolved, this path's parent did not."""
    (tmp_path / "src").mkdir()
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file="src/deleted.py")],
    )
    real = ledger.os.path.realpath
    root = str(tmp_path)

    def _only_the_root(path, *a, **kw):
        if str(path) == root:
            return real(path)
        raise OSError("nope")

    monkeypatch.setattr(ledger.os.path, "realpath", _only_the_root)

    carried = carry_in(REPO, PR, SEAT, root, "head2")

    assert list(carried.rows) == ["p1"]
    assert all(r["status"] == "open" for r in store.rows.values())


def _fill_ledger(store, count):
    """Record ``count`` distinct open findings for this seat, oldest first."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding(file=f"src/f{n}.py", title=f"finding {n}") for n in range(count)],
    )


def test_a_seat_over_the_read_cap_is_reported_not_silently_cut(store, capsys):
    """#173's acceptance, end to end through `carry_in`.

    Past the cap the rows dropped are the NEWEST, and a row no round is offered
    is a row no round can settle -- it can only leave `open` by ageing out of the
    retention window. That must never be silent.
    """
    over = review_state.MAX_OPEN_FINDINGS + 3
    _fill_ledger(store, over)
    capsys.readouterr()

    carried = carry_in(REPO, PR, SEAT)

    assert len(carried.rows) == review_state.MAX_OPEN_FINDINGS
    assert carried.truncated == 3
    # The newest are the ones missing: the prompt carries `finding 0`, not the tail.
    assert "finding 0" in carried.text
    assert f"finding {over - 1}" not in carried.text
    logged = [line for line in capsys.readouterr().err.splitlines() if "review-state" in line]
    assert len(logged) == 1
    # The total leads, because the count IS the finding: a seat holding this many
    # open claims on one PR is a seat whose rounds are settling nothing.
    assert f"{over} open findings" in logged[0]
    assert "3 NEWEST" in logged[0] and f"seat {SEAT}" in logged[0]


def test_a_ledger_inside_the_read_cap_logs_nothing(store, capsys):
    """The warning is an exception report, not a per-round line."""
    _fill_ledger(store, 3)
    capsys.readouterr()

    carried = carry_in(REPO, PR, SEAT)

    assert carried.truncated == 0
    assert "review-state" not in capsys.readouterr().err


def _forbidden(name):
    def _fail(*a, **kw):
        raise AssertionError(f"Tier 1 must not call {name}")

    return _fail


def test_a_seat_with_the_coverage_ledger_off_neither_reads_nor_writes_coverage(monkeypatch, store):
    """Default off means default off: no read, no write, and the same prompt as before.

    Expiry is deliberately NOT forbidden here -- it runs on every seat (see the
    flag-flip test below), because it can only ever remove a stale assurance.
    """
    for name in ("record_coverage", "live_coverage"):
        monkeypatch.setattr(review_state, name, _forbidden(name))

    carried = carry_in(REPO, PR, SEAT, touched_files=["src/app.py"])
    outcome = settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding()],
        examined=[_examined()],
    )

    assert carried.coverage == 0 and outcome.coverage == 0
    assert COVERAGE_ADVISORY not in carried.text


def test_coverage_recorded_in_one_round_is_carried_into_the_next(store):
    """The Tier-2 acceptance criterion: a round is told what it has already covered."""
    first = settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined()],
        coverage_ledger=True,
    )
    assert first.coverage == 1

    second = carry_in(REPO, PR, SEAT, touched_files=["src/other.py"], coverage_ledger=True)

    assert second.coverage == 1 and second.expired == 0
    assert "src/util.py open_source -- round 1" in second.text
    assert "checked: whether every caller handles a None device" in second.text
    assert "evidence: src/util.py:118-166, src/shim.py:402" in second.text


def test_a_delta_that_touches_a_file_expires_its_coverage(store):
    """Mitigation 2: the conclusion described a tree the head no longer has.

    The one use the epic makes of the delta, and it INVALIDATES rather than
    scopes -- the coverage of a file the round did not touch survives alongside.
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[
            _examined(),
            _examined(file="src/untouched.py", evidence="src/untouched.py:1-40"),
        ],
        coverage_ledger=True,
    )

    second = carry_in(REPO, PR, SEAT, touched_files=["src/util.py"], coverage_ledger=True)

    assert second.expired == 1 and second.coverage == 1
    assert "src/util.py open_source -- round 1" not in second.text
    assert "src/untouched.py open_source -- round 1" in second.text


def test_expiry_runs_even_for_a_seat_whose_coverage_ledger_is_off(store):
    """A flag that only ADDS behaviour must not be able to preserve a stale assurance.

    A seat that ran the ledger, was switched off for some rounds and was switched
    back on would otherwise be handed entries describing heads no round in
    between had a chance to expire.
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined()],
        coverage_ledger=True,
    )

    off = carry_in(REPO, PR, SEAT, touched_files=["src/util.py"])
    back_on = carry_in(REPO, PR, SEAT, coverage_ledger=True)

    assert off.expired == 1
    assert back_on.coverage == 0 and back_on.text == ""


def test_an_empty_delta_expires_nothing(store):
    """A round whose delta touched no file must not discard the ledger wholesale.

    `expire_coverage(files=None)` is the wholesale case; the empty sequence is
    not the same thing, and conflating them would make an unparseable diff erase
    every assurance the seat holds.
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined()],
        coverage_ledger=True,
    )

    second = carry_in(REPO, PR, SEAT, touched_files=[], coverage_ledger=True)

    assert second.expired == 0 and second.coverage == 1


def test_a_bare_string_delta_reaches_the_stores_guard_intact():
    """`touched_files="src/app.py"` must not be normalised into a list of characters.

    A `sorted(set(...))` applied unconditionally would defeat the store's own
    guard (`_not_a_bare_string`, covered in test_review_state.py), which exists
    precisely because such a call expires nothing while reporting the same `0` as
    "there was no coverage for that file" -- a stale assurance kept.
    """
    assert ledger._expiry_targets("src/util.py") == "src/util.py"
    # Everything else is de-duplicated and ordered, so the store's parameter is a
    # function of the delta rather than of a set's iteration order.
    assert ledger._expiry_targets(frozenset({"b.py", "a.py"})) == ["a.py", "b.py"]
    assert ledger._expiry_targets(["a.py", "a.py"]) == ["a.py"]


def test_a_hollow_coverage_entry_is_dropped_rather_than_injected(store, capsys):
    """Mitigation 1: an entry nothing can retrace is the record this ledger must not carry.

    The schema can only require the KEYS -- `""` satisfies a required `str` -- so
    an entry with a conclusion, no `checked` and no evidence passes validation and
    is a bare clean bill of health. It is dropped on the way back out, loudly.
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[
            _examined(checked="   ", conclusion="error handling here is fine"),
            _examined(file="src/b.py", evidence=""),
            _examined(file="src/c.py"),
        ],
        coverage_ledger=True,
    )
    capsys.readouterr()

    second = carry_in(REPO, PR, SEAT, coverage_ledger=True)

    assert second.coverage == 1
    assert "src/c.py" in second.text
    assert "error handling here is fine" not in second.text and "src/b.py" not in second.text
    logged = [line for line in capsys.readouterr().err.splitlines() if "review-state" in line]
    assert len(logged) == 1 and "dropped 2 coverage entries" in logged[0]
    assert "/".join(EXAMINED_REQUIRED_FIELDS) in logged[0]


@pytest.mark.parametrize("padded", [" src/util.py", "src/util.py ", "\tsrc/util.py\n"])
def test_a_padded_coverage_path_is_stripped_so_a_delta_can_still_expire_it(store, padded):
    """`file` is the expiry MATCHING KEY, so a padded path is a permanent assurance.

    The delta arrives stripped from the diff parser and the store's `_clip`
    truncates without stripping, so an unnormalised path is recorded, rendered
    into later prompts, and matched by no delta that ever touches that file
    again -- the stale-assurance direction reached through the key rather than
    through the documented revert gap (CodeRabbit and `qwen-anthropic/qwen3.8-max`).
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined(file=padded)],
        coverage_ledger=True,
    )

    second = carry_in(REPO, PR, SEAT, touched_files=["src/util.py"], coverage_ledger=True)

    assert second.expired == 1 and second.coverage == 0


def test_a_coverage_entry_naming_no_file_is_dropped_rather_than_carried_forever(store, capsys):
    """A blank `file` is not merely unretraceable -- no delta can ever retire it.

    It is why `file` is one of `EXAMINED_REQUIRED_FIELDS`: the other three make an
    entry checkable, this one makes it mortal.
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined(file="   "), _examined(file="src/c.py")],
        coverage_ledger=True,
    )
    capsys.readouterr()

    second = carry_in(REPO, PR, SEAT, coverage_ledger=True)

    assert second.coverage == 1 and "src/c.py" in second.text
    logged = [line for line in capsys.readouterr().err.splitlines() if "review-state" in line]
    assert len(logged) == 1 and "dropped 1 coverage entry" in logged[0]


def test_the_receipt_counts_the_coverage_shown_not_the_rows_the_read_returned(store):
    """`carried.coverage` is what reached the prompt, and the renderer caps that.

    #157's rollout is scored on this receipt, so a seat holding more live entries
    than `MAX_PRIOR_COVERAGE` must not have its receipt claim a number the round
    never saw (`qwen-anthropic/qwen3.8-max`).
    """
    over = MAX_PRIOR_COVERAGE + 3
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[
            _examined(file=f"src/m{n}.py", evidence=f"src/m{n}.py:1-20") for n in range(over)
        ],
        coverage_ledger=True,
    )

    second = carry_in(REPO, PR, SEAT, coverage_ledger=True)

    assert second.coverage == MAX_PRIOR_COVERAGE
    assert f"{over - MAX_PRIOR_COVERAGE} older coverage entries were dropped" in second.text


def test_coverage_of_a_file_the_head_deleted_is_retired_against_the_tree(store, tmp_path):
    """The two shapes that differ MAXIMALLY from base are the two the delta omits.

    A deletion emits `+++ /dev/null` and a rename leaves nothing at the old path,
    so neither reaches `parse_diff`'s file set -- their coverage would otherwise
    survive every round to retention (`qwen-anthropic/qwen3.8-max` and
    `openrouter/upstage/solar-pro4`).
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.py").write_text("x = 1\n")
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[
            _examined(file="src/kept.py", evidence="src/kept.py:1-9"),
            _examined(file="src/deleted.py", evidence="src/deleted.py:1-9"),
        ],
        coverage_ledger=True,
    )

    second = carry_in(REPO, PR, SEAT, str(tmp_path), "head2", coverage_ledger=True)

    assert second.coverage == 1 and second.expired == 1
    assert "src/kept.py" in second.text and "src/deleted.py" not in second.text
    # Expired in the STORE, not merely skipped: the row is dead for every future
    # round, so the tree is not re-asked about it.
    assert [c["region"].file for c in store.coverage if not c["expired"]] == ["src/kept.py"]
    # And a round whose surviving coverage is all still in the tree writes
    # nothing: the pass costs a stat per entry, never an expiry.
    third = carry_in(REPO, PR, SEAT, str(tmp_path), "head3", coverage_ledger=True)
    assert third.coverage == 1 and third.expired == 0


@pytest.mark.parametrize("root", ["", "/definitely/not/a/checkout"])
def test_an_unusable_checkout_root_expires_no_coverage(store, root):
    """Under a root that does not exist every path is 'missing' -- expire none.

    The same fail-safe the findings half applies, and it matters more here: an
    expiry is a store write, so one bad argument would empty the ledger.
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined()],
        coverage_ledger=True,
    )

    second = carry_in(REPO, PR, SEAT, root, "head2", coverage_ledger=True)

    assert second.coverage == 1 and second.expired == 0
    assert all(not c["expired"] for c in store.coverage)


def test_consecutive_found_nothing_rounds_still_get_distinct_round_labels(store):
    """A round that recorded only coverage still happened (CodeRabbit).

    It is the shape a clean re-round takes -- nothing found, what was read
    recorded -- so a round number derived from the findings ledger alone would
    label a whole streak of them `1` and present each as the oldest.
    """
    for n in range(3):
        settle(
            carry_in(REPO, PR, SEAT, coverage_ledger=True),
            repo=REPO,
            pr=PR,
            seat=SEAT,
            head_sha=f"head{n}",
            examined=[_examined(file=f"src/m{n}.py", evidence=f"src/m{n}.py:1-20")],
            coverage_ledger=True,
        )

    carried = carry_in(REPO, PR, SEAT, coverage_ledger=True)

    assert carried.round == 4
    assert [line for line in carried.text.splitlines() if line.startswith("- src/m")] == [
        "- src/m2.py open_source -- round 3",
        "- src/m1.py open_source -- round 2",
        "- src/m0.py open_source -- round 1",
    ]


def test_two_seats_on_one_pr_never_read_each_others_coverage(store):
    """Per-seat, never shared (#160). Sharing would buy coverage with the second opinion.

    The seats overlap on most files, so a shared ledger looks like free coverage
    and is in fact the manufacture of exactly the correlation the second seat
    exists to break.
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined()],
        coverage_ledger=True,
    )

    other = carry_in(REPO, PR, "henry", coverage_ledger=True)

    assert other.coverage == 0 and other.text == ""
    assert carry_in(REPO, PR, SEAT, coverage_ledger=True).coverage == 1


def test_the_carried_coverage_block_is_advisory_and_never_says_to_pass_a_region_over(store):
    """Mitigation 3: an instruction to bypass turns a wrong row into a permanent blind spot.

    Asserted on fuko's own prose only -- a stored path or citation may legitimately
    contain any word, so a naive scan of the whole block would be a test of the
    model's vocabulary rather than of this contract.
    """
    settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined()],
        coverage_ledger=True,
    )

    text = carry_in(REPO, PR, SEAT, coverage_ledger=True).text

    assert COVERAGE_ADVISORY in text
    assert "skip" not in COVERAGE_ADVISORY.lower()
    assert "Deprioritise" in COVERAGE_ADVISORY
    # The three re-entry conditions, and the standing permission to disagree.
    assert "concrete reason to doubt" in COVERAGE_ADVISORY
    assert "not established fact" in COVERAGE_ADVISORY
    assert "never what was found to be sound" in COVERAGE_ADVISORY


def test_coverage_is_recorded_exactly_as_the_round_reported_it(store):
    """The write is unjudged: the table records what a round said, filtering happens on read.

    Re-deciding a conclusion on the way in would make the ledger disagree with
    the round it describes, and nothing about a coverage row is published, so
    there is no valve to route around here.
    """
    outcome = settle(
        carry_in(REPO, PR, SEAT, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        examined=[_examined(), _examined(file="src/b.py", evidence="")],
        coverage_ledger=True,
    )

    assert outcome.coverage == 2
    assert [row["region"].file for row in store.coverage] == ["src/util.py", "src/b.py"]


def test_no_store_configured_carries_nothing_and_records_nothing(monkeypatch):
    """The degradation criterion, against the real (unconfigured) primitives."""
    monkeypatch.setattr(review_state.settings, "database_url", "")

    carried = carry_in(REPO, PR, SEAT)
    outcome = settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        prior_status=[PriorFindingStatus(id="p1", status="fixed")],
        findings=[_finding()],
    )

    assert carried == CarriedState()
    assert carried.text == "" and not carried.state
    assert outcome == ledger.Settlement()


def test_default_seat_is_a_seat_not_a_missing_key():
    """A solo config is one seat; it must still get a ledger of its own."""
    assert ledger.DEFAULT_SEAT and os.sep not in ledger.DEFAULT_SEAT


def test_a_seat_with_the_findings_ledger_off_neither_reads_nor_writes_findings(monkeypatch, store):
    """#159's stateless arm: no read, no retirement, no verdicts, no write.

    The forbidden set is the whole Tier 1 surface, including ``settled_findings``
    -- the reopen path is a read of the closed ledger and a stateless arm must
    not take it either.
    """
    for name in ("open_findings", "record_findings", "touch_findings", "settled_findings"):
        monkeypatch.setattr(review_state, name, _forbidden(name))

    carried = carry_in(REPO, PR, SEAT, touched_files=["src/app.py"], findings_ledger=False)
    outcome = settle(
        carried,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding()],
        findings_ledger=False,
    )

    assert carried.text == ""
    assert carried.rows == {} and carried.truncated == 0
    assert outcome.recorded == 0 and outcome.closed == 0 and outcome.reasserted == 0


def test_the_findings_gate_is_the_only_difference_between_the_two_arms(store):
    """A round's own claim reaches the next round's prompt iff the flag is on."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding()],
    )

    assert _finding().title in carry_in(REPO, PR, SEAT).text
    assert carry_in(REPO, PR, SEAT, findings_ledger=False).text == ""


def test_turning_the_findings_ledger_off_loses_nothing_a_later_round_needs(store):
    """Off is a gate on THIS round, not a wipe: the rows are still there after."""
    settle(
        carry_in(REPO, PR, SEAT),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding()],
    )
    stateless = carry_in(REPO, PR, SEAT, findings_ledger=False)
    settle(
        stateless,
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head2",
        findings=[_finding(title="a second claim")],
        findings_ledger=False,
    )

    back_on = carry_in(REPO, PR, SEAT)
    assert _finding().title in back_on.text
    # The stateless round's OWN finding was never written, which is the point:
    # rows recorded while switched off would return as claims a flag-on round
    # could not have settled.
    assert "a second claim" not in back_on.text


def test_the_two_tiers_are_independent_switches(store):
    """Tier 2 on with Tier 1 off still records coverage, and expiry still runs."""
    outcome = settle(
        carry_in(REPO, PR, SEAT, findings_ledger=False, coverage_ledger=True),
        repo=REPO,
        pr=PR,
        seat=SEAT,
        head_sha="head1",
        findings=[_finding()],
        examined=[_examined()],
        findings_ledger=False,
        coverage_ledger=True,
    )

    assert outcome.coverage == 1 and outcome.recorded == 0
    # Expiry is unconditional, so the delta that touches the examined file kills
    # the entry even on a round whose Tier 1 is off.
    assert (
        carry_in(REPO, PR, SEAT, touched_files=[_examined().file], findings_ledger=False).expired
        == 1
    )
