"""Selection of learnings from resolved pull-request review threads."""

from .models import IngestItem


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


def select_learning(thread: dict, bot_login: str | None = None) -> IngestItem | None:
    """Return a learning from a resolved thread's last human comment, or ``None``.

    A thread contributes a learning only when it is resolved and contains at
    least one non-bot comment; the last such comment is treated as the human
    correction that resolved the bot's suggestion, scoped to the thread's file.
    """
    if not thread.get("isResolved"):
        return None
    comments = (thread.get("comments") or {}).get("nodes") or []
    human = [c for c in comments if _is_human(c, bot_login)]
    if not human:
        return None
    body = (human[-1].get("body") or "").strip()
    if not body:
        return None
    path = thread.get("path")
    return IngestItem(
        text=body,
        source="resolved_thread",
        source_url=human[-1].get("url"),
        file_globs=[path] if path else [],
        topic="resolved thread",
    )
