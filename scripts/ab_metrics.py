"""Score the stateful-vs-stateless review A/B from PR receipts (#159, epic #160).

Fetches each named PR's review receipts through BOTH channels the reviewers
publish FINDINGS on -- inline comments, and the review bodies that carry
unanchorable ones -- normalizes them with fuko's own recognizers, assigns each to
an ARM by its author login, and reports the metrics the epic is scored on:
findings per arm, distinct paths, path concentration, one-shot rate, cross-arm
agreement, and Chapman pool coverage. Beside the agreement figure it lists the
same-round same-file pairs the exact-title rule scored as disagreeing, because on
same-model arms that figure is a LOWER bound and the pairs are what a reader
adjudicates to find out how loose it is (#243).
The branch headers are read as a third channel, for their ``fuko-run:v1``
receipts alone: a round a failover backup answered is scored under the rescued
arm and is named in the output as a model confound (#204).
Token and cost per run come from ``review_runs`` (#152) and are printed only
when a database is configured.

The estimators live in :mod:`sidecar.abmetrics` and are unit-tested; this file
is the credentialed, I/O half -- what it adds is fetching, the login-to-arm
mapping, and formatting. Claim identity is the findings ledger's own RULE --
``(file, casefolded title)`` via :func:`sidecar.reviewer.ledger.claim_anchor` --
applied to the title recovered from what was published; a round is one head (an
inline comment's ``original_commit_id``, which GitHub does not rewrite when it
re-anchors an outdated thread). The rule is shared, the inputs are not: ``encode_marker``
omits title and body, so a published finding's title is re-derived from the
rendered markdown rather than read back from the ledger, and an anchor computed
here will not equal the one the ledger stored for the same finding. That is
harmless for every comparison this tool makes -- both arms are re-derived
identically -- and fatal to any future join against ledger rows, which is why it
is stated here rather than left to be discovered.

Arms are named EXPLICITLY on the command line rather than inferred, because on
this fleet they cannot be inferred: #159's control and treatment run the same
provider, the same model and the same ``role``, so the only thing separating
them on GitHub is which App identity posted. Worse, the App names no longer
describe the seats -- the control inherits the retired diff seat's identity --
so the mapping is a fact about the config at the time of the run, and belongs in
the invocation that reads that run's receipts.

This is a maintenance tool, not part of the runtime. It needs fuko-pr installed
(``pip install -e .``) and ``gh`` authenticated (or ``GITHUB_TOKEN`` set).

Usage::

    python scripts/ab_metrics.py <owner/repo>
        --arm control=fuko-dorian[bot] --arm treatment=fuko-gray[bot]
        --slot control=dorian --slot treatment=gray
        <pr> [<pr> ...]

    (one line in a real shell; wrapped here for width)

To regenerate the pre-A/B baselines under the SAME rule, run it over the older
PRs with the two seats that were live then. Comparing this tool's output against
the epic's published figures (4.7% / 86% / ~26% / 64%) is comparing two
different rules and is not evidence of anything.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from sidecar import run_metrics, runner
from sidecar.abmetrics import (
    Claim,
    TOP_PATHS,
    arm_metrics,
    backup_served_rounds,
    candidate_pairs,
    collect_claims,
    pair_metrics,
)
from sidecar.backends.base import PRRef

BASELINES = {
    "cross-arm agreement": "4.7%",
    "one-shot rate": "86%",
    "pool coverage (Chapman)": "~26%",
    "top-3-path share": "64%",
}
"""The epic's published figures, printed beside the run for orientation ONLY.

They were produced by scripts that no longer exist, under a claim-identity rule
nobody can now inspect. They are a sanity range, not a comparison: a difference
between a column here and one of these numbers may be the reviewer changing, or
it may be the rule changing, and this tool cannot tell the reader which.
"""


_API = "https://api.github.com"


def _token() -> str:
    """Return a GitHub token from ``GITHUB_TOKEN`` or the ``gh`` CLI."""
    env = os.environ.get("GITHUB_TOKEN")
    if env:
        return env
    return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()


def _pairs(values: list[str], flag: str) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` flags into a dict, failing loudly on a typo.

    A repeated NAME, or one VALUE given to two NAMEs, is rejected rather than
    resolved last-wins. That is the copy-paste typo this tool's charter has to
    catch: two arms handed the same bot login would score every claim under one
    of them and compare it against an arm holding none, which prints ``n/a``
    agreement and a power warning that reads as "no data" rather than "bad
    mapping" (CodeRabbit-class shape raised by the qwen3.8-max active seat).
    """
    out: dict[str, str] = {}
    for raw in values:
        name, sep, value = raw.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name or not value:
            raise SystemExit(f"{flag} expects NAME=VALUE, got {raw!r}")
        if name in out:
            raise SystemExit(f"{flag} {name!r} given more than once")
        if value in out.values():
            raise SystemExit(f"{flag} value {value!r} mapped to more than one name")
        out[name] = value
    return out


def _claims(
    repo: str, pr_num: int, token: str, arms: dict[str, str]
) -> tuple[list[Claim], set[tuple[str, str]], int]:
    """Fetch one PR's receipts through both channels and reduce them to claims.

    The reduction itself -- the arm/round rejoin, the two-channel union, the
    round key and the untitled drop -- is :func:`sidecar.abmetrics.collect_claims`,
    which is pure and unit-tested. All this adds is the two credentialed reads,
    which is the whole division of labour between this file and that module.
    """
    pr = PRRef(repo=repo, number=pr_num, url=f"https://github.com/{repo}/pull/{pr_num}")
    return collect_claims(
        runner.fetch_inline_comments(pr, token, _API),
        runner.fetch_reviews(pr, token, _API),
        arms,
        pr_num,
    )


def _backup_rounds(
    repo: str, pr_num: int, token: str, arms: dict[str, str]
) -> set[tuple[str, str]]:
    """Fetch one PR's branch headers and reduce them to its backup-served rounds.

    A third credentialed read beside the two in :func:`_claims`, on the third
    channel a run publishes: the branch header issue comments that carry the
    ``fuko-run:v1`` receipts. The reduction is
    :func:`sidecar.abmetrics.backup_served_rounds`, which is pure and unit-tested.
    """
    pr = PRRef(repo=repo, number=pr_num, url=f"https://github.com/{repo}/pull/{pr_num}")
    return backup_served_rounds(runner.fetch_issue_comments(pr, token, _API), arms, pr_num)


def _pct(value: float | None) -> str:
    """Format a fraction as a percentage, or ``n/a`` when it is undefined."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: int | None) -> str:
    """Format a count, or ``n/a`` when the aggregate was NULL.

    ``run_metrics`` keeps "not measured" distinct from zero and returns ``None``
    for the first (#152). Interpolating it with ``str`` would print the literal
    "None" into a report whose stated convention one column to the right is
    n/a-never-a-misleading-figure.
    """
    return "n/a" if value is None else str(value)


def _report_arms(claims: list[Claim], arms: dict[str, str], receipts: set[tuple[str, str]]) -> None:
    """Print the per-arm table."""
    print(
        f"\n{'arm':<12} {'rounds':>7} {'finds':>6} {'claims':>7} {'paths':>6} "
        f"{'re-rep':>7} {'one-shot':>9} {f'top{TOP_PATHS}':>7}"
    )
    for arm in arms:
        m = arm_metrics(arm, claims, sorted(receipts))
        print(
            f"{arm:<12} {m.rounds:>7} {m.findings:>6} {m.distinct_claims:>7} "
            f"{m.distinct_paths:>6} {m.re_reported:>7} {_pct(m.one_shot_rate):>9} "
            f"{_pct(m.top_paths_share):>7}"
        )
        for path, touches in m.top_paths:
            print(f"    {touches:>3}x {path}")


def _report_pair(claims: list[Claim], arms: dict[str, str], receipts: set[tuple[str, str]]) -> None:
    """Print the cross-arm agreement and Chapman coverage, or say why it cannot."""
    names = list(arms)
    if len(names) != 2:
        print("\npair metrics need exactly two arms; skipped")
        return
    p = pair_metrics(claims, names[0], names[1], sorted(receipts))
    print(
        f"\nrounds both arms reviewed: {p.rounds}\n"
        f"claims: {names[0]}={p.a_claims} {names[1]}={p.b_claims} "
        f"shared={p.shared} union={p.union}\n"
        f"cross-arm agreement: {_pct(p.agreement)}\n"
        f"estimated pool (Chapman): "
        f"{'n/a' if p.pool_estimate is None else format(p.pool_estimate, '.1f')}\n"
        f"estimated pool coverage: {_pct(p.coverage)}"
    )
    if p.rounds < 10:
        print(
            f"\nPOWER WARNING: {p.rounds} shared round(s). At a baseline agreement "
            "near 5% a handful of rounds cannot separate the arms; report this as "
            "a null result rather than a decision."
        )
    _report_candidates(claims, names[0], names[1], receipts)


def _report_candidates(claims: list[Claim], a: str, b: str, receipts: set[tuple[str, str]]) -> None:
    """List the same-round, same-file pairs the exact-title rule scored as disagreeing.

    Printed after the agreement figure because it is that figure's caveat: two
    arms on one model paraphrase, so the agreement above is a LOWER bound, and
    these are the pairs a reader has to adjudicate to know how loose the bound is
    (#243). Nothing here is counted, averaged, or fed back into a metric -- the
    adjudication is a human's, and a tool that pre-empted it would be the
    fitted-to-its-answer matcher #159 rules out.

    The empty line says only that no such PAIR exists, which is the one claim
    that holds however the list came out empty: the arms shared no round at all,
    every same-file claim matched exactly, or one arm carried an unmatched
    surplus the other had nothing to weigh against. Widening it to "nothing sits
    outside the exact-title match" would assert the second of those three in all
    three cases.
    """
    pairs = candidate_pairs(claims, a, b, sorted(receipts))
    if not pairs:
        print("\nno same-round same-file pair with unmatched claims on both arms")
        return
    print(
        f"\n{len(pairs)} same-round same-file pair(s) the exact-title rule scored as "
        "DISAGREEING. Cross-arm agreement above is a LOWER bound; adjudicate these by "
        "hand to see how loose it is (#243), and do not fold the verdict back into a "
        "matcher:"
    )
    for pair in pairs:
        print(f"    {pair.round_key} {pair.file}")
        for title in pair.a_titles:
            print(f"        {a}: {title}")
        for title in pair.b_titles:
            print(f"        {b}: {title}")


def _report_backup_rounds(rescued: set[tuple[str, str]], receipts: set[tuple[str, str]]) -> None:
    """Name the rounds a failover backup answered, so the model confound is visible.

    Printed rather than subtracted (#204): these rounds are attributed to the
    right arm and run the right ledger configuration, but a different model
    produced them, and a reader comparing two arms is entitled to know which
    rounds are not a clean single-variable comparison.

    Split against ``receipts`` -- the reviewed-round set every metric is built
    from -- so a reader can tell a key that locates its round from one that does
    not. Neither note asserts metric MEMBERSHIP, and that restraint is the point:
    a round key is the head the run STARTED on, while a round is keyed by the
    commit GitHub stamps at submission, so a push landing mid-run shifts the key
    off its own round (#210). It can then miss every reviewed round while the
    round itself is still scored under the submission-time head, or collide with
    a neighbouring round that was never rescued at all. Saying "included" or "in
    no metric" would be wrong in one of those cases whichever way it is phrased;
    what is always true is that nothing here is subtracted from any metric.
    """
    named = rescued & receipts
    unanchored = rescued - receipts
    if named:
        print(
            f"\nnote: {len(named)} round(s) were answered by a FAILOVER backup, not the "
            "arm's own model. The arm is the seat's, and so is the ledger configuration "
            "for any round run on a runner carrying #204; the model is a confound:"
        )
        for arm, round_key in sorted(named):
            print(f"    {arm}: {round_key}")
    if unanchored:
        print(
            f"\nnote: {len(unanchored)} further backup-answered round(s) carry a key "
            "matching no reviewed round here:"
        )
        for arm, round_key in sorted(unanchored):
            print(f"    {arm}: {round_key}")
    if rescued:
        print(
            "    (a key is the head its run STARTED on, not the one its review was "
            "submitted against, so a mid-run push can shift a key off its own round --"
            " #210. Rescued rounds are never subtracted from any metric.)"
        )


def _report_cost(repo: str, slots: dict[str, str], days: int) -> None:
    """Print per-arm token and cost totals from ``review_runs``, or why they are absent."""
    if not slots:
        print("\ntokens/cost: no --slot mapping given; skipped")
        return
    rows = {r["slot"]: r for r in run_metrics.slot_summary(repo, days)}
    if not rows:
        print("\ntokens/cost: no rows (FUKO_DATABASE_URL unset, or no runs in window)")
        return
    print(f"\n{'arm':<12} {'runs':>5} {'in':>10} {'cache-rd':>10} {'out':>9} {'cost$':>9}")
    for arm, slot in slots.items():
        r = rows.get(slot)
        if r is None:
            print(f"{arm:<12} {'-':>5} {'no runs recorded for slot ' + slot:>10}")
            continue
        cost = "n/a" if r["cost_usd"] is None else f"{r['cost_usd']:.4f}"
        print(
            f"{arm:<12} {r['runs']:>5} {_num(r['input_tokens']):>10} "
            f"{_num(r['cache_read_tokens']):>10} {_num(r['output_tokens']):>9} {cost:>9}"
        )


def main() -> None:
    """Parse the arguments, gather every PR's claims, and print the report."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("repo", help="owner/name")
    ap.add_argument("prs", nargs="+", type=int, help="PR numbers to score")
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=LOGIN",
        help="map a bot login onto an arm; repeat once per arm",
    )
    ap.add_argument(
        "--slot",
        action="append",
        default=[],
        metavar="NAME=SLOT",
        help="map an arm onto its review_runs slot, for the token column",
    )
    ap.add_argument("--days", type=int, default=30, help="review_runs window (default 30)")
    args = ap.parse_args()

    arms = _pairs(args.arm, "--arm")
    if not arms:
        raise SystemExit("at least one --arm NAME=LOGIN is required; arms are never inferred")
    token = _token()
    claims: list[Claim] = []
    receipts: set[tuple[str, str]] = set()
    rescued: set[tuple[str, str]] = set()
    untitled = 0
    for pr_num in args.prs:
        got, seen, blank = _claims(args.repo, pr_num, token, arms)
        claims += got
        receipts |= seen
        rescued |= _backup_rounds(args.repo, pr_num, token, arms)
        untitled += blank
        print(
            f"{args.repo}#{pr_num}: {len(got)} claim(s) over {len(seen)} reviewed round(s) "
            "across the named arms",
            file=sys.stderr,
        )

    print(f"\n=== {args.repo} PRs {args.prs} ===")
    if untitled:
        print(
            f"note: {untitled} finding(s) carried no usable title and were DROPPED from "
            "every metric below; review-body findings arrive titleless until #142 "
            "rehydrates them"
        )
    _report_arms(claims, arms, receipts)
    _report_backup_rounds(rescued, receipts)
    _report_pair(claims, arms, receipts)
    _report_cost(args.repo, _pairs(args.slot, "--slot"), args.days)
    print("\npublished baselines (different rule -- orientation only):")
    for name, value in BASELINES.items():
        print(f"    {name}: {value}")


if __name__ == "__main__":
    main()
