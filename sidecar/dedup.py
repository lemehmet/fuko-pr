"""Duplicate filtering and bounded batching shared by the knowledge stores.

Both backends dedup on the ``(text, source)`` key *before* embedding, so
re-ingesting an already-stored backlog costs no embed calls. The same pass caps
how many genuinely new items reach the embedder, which is what lets a caller
drain a large backlog over several bounded requests instead of one request whose
embed work outruns its timeout.
"""

from .models import IngestItem


def partition(
    items: list[IngestItem],
    existing: set[tuple[str, str]],
    max_new: int | None = None,
) -> tuple[list[IngestItem], int, int]:
    """Split ``items`` into what to embed, the duplicate count, and the deferred count.

    A duplicate is any item whose ``(text, source)`` key is already stored or is
    repeated earlier in this batch. Of what remains, at most ``max_new`` items are
    returned for embedding and the rest are reported as deferred and left
    untouched — so a caller that re-sends the same batch picks them up on the next
    pass, once the ones embedded here dedup away.

    Args:
        items: Candidate learnings, in caller order.
        existing: ``(text, source)`` keys already stored for the repo.
        max_new: Cap on items handed to the embedder; ``None`` means no cap.

    Returns:
        A ``(to_embed, skipped, deferred)`` tuple.
    """
    to_embed: list[IngestItem] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    deferred = 0
    for item in items:
        key = (item.text, item.source)
        if key in existing or key in seen:
            skipped += 1
            continue
        seen.add(key)
        if max_new is not None and len(to_embed) >= max_new:
            deferred += 1
            continue
        to_embed.append(item)
    return to_embed, skipped, deferred
