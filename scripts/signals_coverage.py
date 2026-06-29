"""Coverage / drift guard for fuko's reviewer normalizers.

For each pull request, fetch its inline review comments, run ``collect_signals``,
and match the resulting signals back to comments by ``thread_url``. Report, per
PR, how many *top-level* bot findings exist versus how many became signals, and
list any that were MISSED — a finding a reviewer posted that fuko did not
normalize. A clean run (no misses) is evidence the normalizers still track what
CodeRabbit / Copilot / PR-Agent actually post; misses flag drift to fix (e.g. a
vendor changed its comment format).

This is a maintenance tool, not part of the runtime. It needs fuko-pr installed
(``pip install -e .`` or run via ``uvx``) and ``gh`` authenticated with read
access to the repo (or ``GITHUB_TOKEN`` set).

Usage:
    python scripts/signals_coverage.py <owner/repo> <pr> [<pr> ...]

For just emitting a PR's signals as JSON, use ``fuko signals --pr-url <url>``;
this script is the coverage analysis layered on top of the same machinery.
"""

from __future__ import annotations

import os
import subprocess
import sys

from sidecar import runner
from sidecar.backends.base import PRRef
from sidecar.normalizers import (
    collect_signals,
    is_coderabbit_comment,
    is_coderabbit_finding,
    is_copilot_comment,
    is_pragent_comment,
)

_BOT_MARKERS = ("coderabbit", "copilot", "fuko-")


def _token() -> str:
    """Return a GitHub token from ``GITHUB_TOKEN`` or the ``gh`` CLI."""
    env = os.environ.get("GITHUB_TOKEN")
    if env:
        return env
    return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()


def _is_bot(login: str) -> bool:
    """Return whether ``login`` is one of the reviewer bots fuko normalizes."""
    low = login.lower()
    return any(b in low for b in _BOT_MARKERS)


def _skip_reason(comment: dict) -> str:
    """Return why ``collect_signals`` would skip this comment, or ``""`` if it wouldn't."""
    body = comment.get("body", "") or ""
    if is_pragent_comment(body) or is_copilot_comment(comment):
        return ""
    if is_coderabbit_comment(comment):
        return "" if is_coderabbit_finding(body) else "cr-no-classification"
    return "unrecognized"


def analyze(repo: str, pr_num: int, token: str) -> int:
    """Print the coverage report for one PR; return the number of misses."""
    pr = PRRef(repo=repo, number=pr_num, url=f"https://github.com/{repo}/pull/{pr_num}")
    comments = runner.fetch_inline_comments(pr, token, "https://api.github.com")
    signals = collect_signals(comments)
    covered = {s.thread_url for s in signals if s.thread_url}

    findings = [
        c
        for c in comments
        if _is_bot((c.get("user") or {}).get("login", "")) and c.get("in_reply_to_id") is None
    ]
    misses = [c for c in findings if c.get("html_url") not in covered]
    print(
        f"\n=== {repo}#{pr_num} === inline: {len(comments)} | "
        f"top-level bot findings: {len(findings)} | signals: {len(signals)} | misses: {len(misses)}"
    )
    for c in misses:
        login = (c.get("user") or {}).get("login", "")
        print(
            f"    MISS [{login} | {_skip_reason(c) or 'unmatched'}] "
            f"{c.get('path')}:{c.get('line')}  {c.get('body', '')[:90]!r}"
        )
    return len(misses)


def main() -> None:
    """Run the coverage report for every PR named on the command line."""
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    repo = sys.argv[1]
    token = _token()
    total = sum(analyze(repo, int(n), token) for n in sys.argv[2:])
    print(f"\ntotal misses: {total}")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
