"""Selection of learnings from pull-request review threads.

The valuable signal in a review thread is a *decline* — a trusted human pushing
back on a reviewer bot's finding and stating the project's actual convention
(e.g. "this is intentional because ..."). That is durable repo knowledge worth
injecting into future reviews. The other classes are NOT learnings and are
dropped: fix acknowledgements (a completion verb near a commit SHA — "Fixed in
<sha>", "Already addressed (<sha>)", "in place since <sha>"; the finding was
accepted and fixed), deferrals ("Filed as #N", the finding is valid but tracked
elsewhere), and comments too short to carry meaning. A genuine decline cites no
commit SHA, so it survives.

Resolution state is deliberately ignored: in the address-pr-reviews loop a fix
resolves the thread (last human comment is the fix-ack) while a decline leaves it
unresolved (last human comment is the correction) — so gating on ``isResolved``
would keep the noise and drop the signal.

The emitted ``source`` stays ``"resolved_thread"`` for backward compatibility
(the learnings ``source`` enum and ``/forget source=resolved_thread`` semantics),
even though an unresolved thread can now contribute a learning.
"""

import re

from .models import IngestItem

_FIX_ACK_RE = re.compile(
    r"\b(?:fix(?:ed)?|address(?:ed)?|resolved?|implement(?:ed)?"
    r"|appl(?:y|ied)|correct(?:ed)?|done|in place)\b"
    r"[^.\n]{0,40}?"
    r"\b[0-9a-f]{7,40}\b",
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


def _is_noise(body: str) -> bool:
    """Return True for a fix-ack or a deferral — not a learning."""
    return bool(_FIX_ACK_RE.search(body) or _DEFERRAL_RE.search(body))


def select_learning(thread: dict, bot_login: str | None = None) -> IngestItem | None:
    """Return a learning from a review thread's last human comment, or ``None``.

    Keeps the last non-bot comment unless it is a fix-ack, a deferral, or too
    short to carry meaning — so a decline/correction is captured while the
    "Fixed in <sha>" / "Filed as #N" replies are skipped. Scoped to the thread's
    file.
    """
    comments = (thread.get("comments") or {}).get("nodes") or []
    human = [c for c in comments if _is_human(c, bot_login)]
    if not human:
        return None
    last = human[-1]
    body = (last.get("body") or "").strip()
    if len(body) < _MIN_LEARNING_CHARS or _is_noise(body):
        return None
    path = thread.get("path")
    return IngestItem(
        text=body,
        source="resolved_thread",
        source_url=last.get("url"),
        file_globs=[path] if path else [],
        topic="review decision",
    )
