"""Tests for the Tier-1 open-findings ledger policy (#156).

The store is faked in memory rather than mocked call-by-call: the acceptance
criterion is a ROUND TRIP ("a finding reported in round N and unaddressed is
present in round N+1's prompt"), which per-call assertions cannot express.
"""

import os

import pytest

from sidecar import review_state
from sidecar.reviewer import ledger
from sidecar.reviewer.ledger import CarriedState, carry_in, settle
from sidecar.reviewer.prompt import (
    AgenticFinding,
    PriorFinding,
    PriorFindingStatus,
    PriorState,
)

REPO, PR, SEAT = "o/r", 9, "dorian"


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
    """An in-memory stand-in for the five ``review_state`` primitives used here."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
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
            }
        return len(findings)

    def open_findings(self, repo, pr, seat, limit=200):
        return [
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
                ),
            )
            for row_id, row in self.rows.items()
            if row["key"] == (repo, pr, seat) and row["status"] == "open"
        ]

    def next_round(self, repo, pr, seat):
        rounds = [r["round"] for r in self.rows.values() if r["key"] == (repo, pr, seat)]
        return max(rounds, default=0) + 1

    def transition(self, finding_id, status, reason=""):
        row = self.rows.get(finding_id)
        if row is None or row["status"] != "open":
            return False
        row["status"], row["reason"] = status, reason
        return True

    def touch_findings(self, finding_ids):
        self.touched.extend(finding_ids)
        return len(finding_ids)


@pytest.fixture
def store(monkeypatch):
    """Replace the review-state primitives the ledger calls with a live fake."""
    fake = _Store()
    for name in ("record_findings", "open_findings", "next_round", "transition", "touch_findings"):
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
    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT)] == ["two", "three"]
    assert store.touched == [carried.rows["p2"]]


def test_re_reporting_a_carried_finding_touches_it_instead_of_duplicating_it(store):
    """A round that re-reports rather than settles must not mint a second row.

    Two open rows for one claim compound per round, and past the read cap the
    rows cut are the NEWEST -- never rendered, never minted an id, never touched,
    so they age out unseen. That is this module's own loss arriving by volume.
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

    assert outcome == ledger.Settlement(reasserted=1, recorded=1, deduped=1)
    assert sorted(f.prior.title for f in store.open_findings(REPO, PR, SEAT)) == [
        "drops the error",
        "leaks the handle",
    ]
    # Touched, not left to age out: a re-report is evidence a round looked at it.
    assert store.touched == [carried.rows["p1"]]


def test_a_settled_row_does_not_suppress_the_same_claim_recorded_again(store):
    """Dedup keys on rows this round LEFT OPEN, so a closed one cannot swallow a write."""
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
        prior_status=[PriorFindingStatus(id="p1", status="fixed", reason="closed in 9f2a1c")],
        findings=[_finding(file="src/a.py", title="leaks the handle")],
    )

    assert outcome == ledger.Settlement(closed=1, recorded=1)
    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT)] == ["leaks the handle"]


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
    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT)] == ["one"]
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

    assert [f.prior.title for f in store.open_findings(REPO, PR, SEAT)] == ["one"]
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
    assert [f.prior.file for f in store.open_findings(REPO, PR, SEAT)] == ["src/kept.py"]
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


def _forbidden(name):
    def _fail(*a, **kw):
        raise AssertionError(f"Tier 1 must not call {name}")

    return _fail


def test_tier_1_never_touches_the_coverage_ledger(monkeypatch, store):
    """Coverage is #157's: a half-wired assurance nothing expires is worse than none."""
    for name in ("record_coverage", "live_coverage", "expire_coverage"):
        monkeypatch.setattr(review_state, name, _forbidden(name))

    carried = carry_in(REPO, PR, SEAT)
    settle(carried, repo=REPO, pr=PR, seat=SEAT, head_sha="head1", findings=[_finding()])


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
