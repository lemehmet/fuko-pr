"""Unit tests for the shared dedup/cap partition used by both stores."""

from sidecar.dedup import partition
from sidecar.models import IngestItem


def _items(*texts, source="review_thread"):
    return [IngestItem(text=t, source=source) for t in texts]


def test_no_cap_embeds_everything_new():
    to_embed, skipped, deferred = partition(_items("a", "b", "c"), set())
    assert [it.text for it in to_embed] == ["a", "b", "c"]
    assert (skipped, deferred) == (0, 0)


def test_existing_keys_are_skipped_not_embedded():
    existing = {("a", "review_thread")}
    to_embed, skipped, deferred = partition(_items("a", "b"), existing)
    assert [it.text for it in to_embed] == ["b"]
    assert (skipped, deferred) == (1, 0)


def test_same_text_different_source_is_not_a_duplicate():
    items = _items("a") + _items("a", source="remember")
    to_embed, skipped, deferred = partition(items, {("a", "review_thread")})
    assert [it.source for it in to_embed] == ["remember"]
    assert (skipped, deferred) == (1, 0)


def test_repeats_within_a_batch_are_skipped_once():
    to_embed, skipped, deferred = partition(_items("a", "a", "a"), set())
    assert len(to_embed) == 1
    assert (skipped, deferred) == (2, 0)


def test_cap_defers_the_overflow_in_caller_order():
    to_embed, skipped, deferred = partition(_items("a", "b", "c", "d"), set(), max_new=2)
    assert [it.text for it in to_embed] == ["a", "b"]
    assert (skipped, deferred) == (0, 2)


def test_cap_counts_only_new_items():
    existing = {("a", "review_thread"), ("b", "review_thread")}
    to_embed, skipped, deferred = partition(_items("a", "b", "c", "d"), existing, max_new=1)
    assert [it.text for it in to_embed] == ["c"]
    assert (skipped, deferred) == (2, 1)


def test_zero_cap_defers_every_new_item():
    to_embed, skipped, deferred = partition(_items("a", "b"), set(), max_new=0)
    assert to_embed == []
    assert (skipped, deferred) == (0, 2)


def test_deferred_item_repeated_later_in_batch_counts_as_duplicate():
    to_embed, skipped, deferred = partition(_items("a", "b", "b"), set(), max_new=1)
    assert [it.text for it in to_embed] == ["a"]
    assert (skipped, deferred) == (1, 1)


def test_totals_always_account_for_every_item():
    items = _items("a", "a", "b", "c", "d")
    to_embed, skipped, deferred = partition(items, {("d", "review_thread")}, max_new=1)
    assert len(to_embed) + skipped + deferred == len(items)
