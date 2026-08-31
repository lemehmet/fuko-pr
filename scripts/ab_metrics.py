"""Score the stateful-vs-stateless review A/B from PR receipts (#159, epic #160).

Fetches each named PR's inline review comments, normalizes them with fuko's own
recognizers, assigns each finding to an ARM by its author login, and reports the
metrics the epic is scored on: findings per arm, distinct paths, path
concentration, one-shot rate, cross-arm agreement, and Chapman pool coverage.
Token and cost per run come from ``review_runs`` (#152) and are printed only
when a database is configured.

The estimators live in :mod:`sidecar.abmetrics` and are unit-tested; this file
is the credentialed, I/O half -- what it adds is fetching, the login-to-arm
mapping, and formatting. Claim identity is the findings ledger's own
``(file, casefolded title)``; a round is one head (the comment's ``commit_id``).

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
from sidecar.abmetrics import Claim, TOP_PATHS, arm_metrics, pair_metrics
from sidecar.backends.base import PRRef
from sidecar.normalizers import collect_signals

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


def _token() -> str:
    """Return a GitHub token from ``GITHUB_TOKEN`` or the ``gh`` CLI."""
    env = os.environ.get("GITHUB_TOKEN")
    if env:
        return env
    return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()


def _pairs(values: list[str], flag: str) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` flags into a dict, failing loudly on a typo."""
    out: dict[str, str] = {}
    for raw in values:
        name, sep, value = raw.partition("=")
        if not sep or not name.strip() or not value.strip():
            raise SystemExit(f"{flag} expects NAME=VALUE, got {raw!r}")
        out[name.strip()] = value.strip()
    return out


def _claims(repo: str, pr_num: int, token: str, arms: dict[str, str]) -> tuple[list[Claim], int]:
    """Return one PR's claims for the mapped arms, plus how many titles were empty.

    A signal reaches an arm only through the comment it came from: the normalized
    signal carries no author, so it is rejoined to its comment by ``thread_url``.
    A finding whose author is not one of the named arms is skipped -- CodeRabbit's
    findings are on the same PR and are not part of this experiment.

    An empty ``title`` is a known reviewer output (#142) and would collapse every
    such finding onto one anchor per file. The first line of the body stands in,
    and the count is returned so the caller can say how much of the run leaned on
    the fallback rather than burying it.
    """
    pr = PRRef(repo=repo, number=pr_num, url=f"https://github.com/{repo}/pull/{pr_num}")
    comments = runner.fetch_inline_comments(pr, token, "https://api.github.com")
    by_thread = {
        c.get("html_url"): (
            (c.get("user") or {}).get("login", ""),
            c.get("commit_id") or c.get("original_commit_id") or "",
        )
        for c in comments
    }
    login_to_arm = {login: arm for arm, login in arms.items()}
    out: list[Claim] = []
    untitled = 0
    for signal in collect_signals(comments):
        login, commit = by_thread.get(signal.thread_url or "", ("", ""))
        arm = login_to_arm.get(login)
        if arm is None or not signal.file:
            continue
        title = signal.title.strip()
        if not title:
            untitled += 1
            title = next((ln for ln in signal.body.splitlines() if ln.strip()), "")
        out.append(Claim(arm=arm, round_key=f"{pr_num}@{commit}", file=signal.file, title=title))
    return out, untitled


def _pct(value: float | None) -> str:
    """Format a fraction as a percentage, or ``n/a`` when it is undefined."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _report_arms(claims: list[Claim], arms: dict[str, str]) -> None:
    """Print the per-arm table."""
    print(
        f"\n{'arm':<12} {'rounds':>7} {'finds':>6} {'claims':>7} {'paths':>6} "
        f"{'re-rep':>7} {'one-shot':>9} {f'top{TOP_PATHS}':>7}"
    )
    for arm in arms:
        m = arm_metrics(arm, claims)
        print(
            f"{arm:<12} {m.rounds:>7} {m.findings:>6} {m.distinct_claims:>7} "
            f"{m.distinct_paths:>6} {m.re_reported:>7} {_pct(m.one_shot_rate):>9} "
            f"{_pct(m.top_paths_share):>7}"
        )
        for path, touches in m.top_paths:
            print(f"    {touches:>3}x {path}")


def _report_pair(claims: list[Claim], arms: dict[str, str]) -> None:
    """Print the cross-arm agreement and Chapman coverage, or say why it cannot."""
    names = list(arms)
    if len(names) != 2:
        print("\npair metrics need exactly two arms; skipped")
        return
    p = pair_metrics(claims, names[0], names[1])
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
            f"{arm:<12} {r['runs']:>5} {str(r['input_tokens']):>10} "
            f"{str(r['cache_read_tokens']):>10} {str(r['output_tokens']):>9} {cost:>9}"
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
    untitled = 0
    for pr_num in args.prs:
        got, blank = _claims(args.repo, pr_num, token, arms)
        claims += got
        untitled += blank
        print(f"{args.repo}#{pr_num}: {len(got)} claim(s) across the named arms", file=sys.stderr)

    print(f"\n=== {args.repo} PRs {args.prs} ===")
    if untitled:
        print(f"note: {untitled} finding(s) had an empty title; first body line used (#142)")
    _report_arms(claims, arms)
    _report_pair(claims, arms)
    _report_cost(args.repo, _pairs(args.slot, "--slot"), args.days)
    print("\npublished baselines (different rule -- orientation only):")
    for name, value in BASELINES.items():
        print(f"    {name}: {value}")


if __name__ == "__main__":
    main()
