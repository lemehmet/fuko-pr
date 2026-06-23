"""Selection of learnings from pull-request review threads.

The valuable signal in a review thread is a *decline* — a trusted human pushing
back on a reviewer bot's finding and stating the project's actual convention
(e.g. "this is intentional because ..."). That is durable repo knowledge worth
injecting into future reviews.

Selection is an ALLOWLIST: a comment is kept only when it expresses a decline /
disagreement, matched by a small, stable set of stance markers ("Declining",
"Not addressing", "Intentional", "False positive", "Moot", ...). Everything else
is dropped. This is deliberately precision-favouring: fix acknowledgements use an
open-ended vocabulary ("Fixed in <sha>", "Added <test>", "Updated ...", "Good
catch — done") that a blocklist can never fully chase, whereas declines share a
narrow set of stances — so allowlisting the stances keeps the store clean across
repos at the cost of dropping the occasional unphrased decline. Deferrals ("filed
as #N", "deferring to a follow-up") are excluded even when they match a decline
marker, since they track the finding elsewhere rather than state a convention.

Resolution state is deliberately ignored: in the address-pr-reviews loop a fix
resolves the thread while a decline is often left unresolved, so gating on
``isResolved`` would bias toward the fix acknowledgements we want to exclude.
"""

import re

from .models import IngestItem

_DECLINE_RE = re.compile(
    r"\bdeclin"
    r"|\bnot\s+(?:address|add|chang|appl|need|actionabl|going|doing|relevant|required)"
    r"|\bno\s+change\s+(?:needed|required|necessary)"
    r"|\bnot\s+(?:a|an)\s+(?:bug|issue|problem|concern|regression)\b"
    r"|\bintentional|\bby design\b|\bdeliberate"
    r"|\bfalse positive\b|\bmoot\b|\binherent to\b"
    r"|\bwon'?t\s+(?:change|fix|be|add|do|touch|alter|modify)\b|\bwontfix\b|\bdisagree"
    r"|\bkeeping\b|\bleaving (?:it|this|that|them)?\s*as\b|\bas-is\b"
    r"|\bactually\s+(?:required|correct|fine|intended|right)\b"
    r"|\b(?:is|are|pin is)\s+correct\b|\bpremis",
    re.IGNORECASE,
)
_DEFERRAL_RE = re.compile(
    r"filed as #\d+"
    r"|tracked in #\d+"
    r"|\bfollow[- ]?up\b"
    r"|\bout[- ]of[- ]scope\b"
    r"|\bdefer(?:s|red|ring|ral|rals)?\b",
    re.IGNORECASE,
)
_MIN_LEARNING_CHARS = 40


def _is_human(comment: dict, bot_login: str | None) -> bool:
    """Return True for a comment authored by a real (non-bot) user."""
    login = (comment.get("author") or {}).get("login")
    if not login:
        return False
    if login.endswith("[bot]"):
        return False
    if bot_login and login == bot_login:
        return False
    return True


def _is_decline(body: str) -> bool:
    """Return True when the comment expresses a decline / disagreement stance."""
    return bool(_DECLINE_RE.search(body))


def _is_deferral(body: str) -> bool:
    """Return True for a deferral ("filed as #N", "deferring to a follow-up").

    A deferral can match a decline marker ("not addressing it in this PR"), so it
    is excluded explicitly — it tracks the finding elsewhere rather than stating
    the project's convention.
    """
    return bool(_DEFERRAL_RE.search(body))


def select_learning(thread: dict, bot_login: str | None = None) -> IngestItem | None:
    """Return a learning from a review thread's last human comment, or ``None``.

    Keeps the last non-bot comment only when it is long enough to carry meaning,
    expresses a decline stance, and is not a deferral — so a reviewer correction
    is captured while fix acknowledgements, deferrals, and neutral chatter are
    skipped. Scoped to the thread's file.
    """
    comments = (thread.get("comments") or {}).get("nodes") or []
    human = [c for c in comments if _is_human(c, bot_login)]
    if not human:
        return None
    last = human[-1]
    body = (last.get("body") or "").strip()
    if len(body) < _MIN_LEARNING_CHARS or not _is_decline(body) or _is_deferral(body):
        return None
    path = thread.get("path")
    return IngestItem(
        text=body,
        source="resolved_thread",
        source_url=last.get("url"),
        file_globs=[path] if path else [],
        topic="review decision",
    )
