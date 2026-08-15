"""Classify a backend failure as provider throttling.

The pool fails over (and trips the breaker) only on a throttle or timeout; a
genuine error (PR-Agent bug, bad config, auth) should fail fast rather than burn
through every provider and trip every breaker on each run. A container timeout
(returncode 124) is throttle-class; otherwise the captured output is matched
against rate-limit signatures, with ``429`` bounded so it cannot hit unrelated
digits.
"""

import re

_THROTTLE_RE = re.compile(
    r"rate.?limit"
    r"|too.?many.?requests"
    r"|\b429\b"
    r"|quota"
    r"|overloaded"
    r"|over_?capacity"
    r"|resource[_ ]?exhausted"
    r"|insufficient_quota"
    r"|throttl"
    # Subscription/spend exhaustion, which reads nothing like "rate limit":
    # Claude Code says "You've hit your session limit" / "weekly limit", and
    # credit/spend caps are the API-key equivalent. All are "this credential is
    # spent, try another", i.e. exactly what failover is for.
    r"|you.?ve hit your \w+ limit"
    r"|usage limit"
    r"|credit balance is too low"
    r"|spend limit",
    re.IGNORECASE,
)

TIMEOUT_RETURNCODE = 124


def is_throttle(returncode: int, output: str) -> bool:
    """Return True if a non-zero result looks like provider throttling.

    A timed-out container (``returncode == 124``) counts as throttle-class: a
    stalled provider should fail over the same as a 429. Otherwise the captured
    stdout/stderr is scanned for a rate-limit signature.
    """
    if returncode == TIMEOUT_RETURNCODE:
        return True
    return bool(output) and _THROTTLE_RE.search(output) is not None
