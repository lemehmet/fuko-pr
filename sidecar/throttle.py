"""Classify a backend failure as provider throttling.

The pool fails over (and trips the breaker) only on a throttle or timeout; a
genuine error (PR-Agent bug, bad config, auth) should fail fast rather than burn
through every provider and trip every breaker on each run.
"""

import re

# Substrings LiteLLM / the providers emit on 429-class conditions. Matched
# case-insensitively against the captured backend output.
_THROTTLE_RE = re.compile(
    r"rate.?limit"
    r"|too.?many.?requests"
    r"|429"
    r"|quota"
    r"|overloaded"
    r"|over_?capacity"
    r"|resource[_ ]?exhausted"
    r"|insufficient_quota"
    r"|throttl",
    re.IGNORECASE,
)

# fuko's sentinel exit code for a container it killed after ``tool_timeout``.
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
