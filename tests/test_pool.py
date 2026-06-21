"""Unit tests for provider-pool resolution and failover ordering."""

from sidecar.fukoconfig import ModelConfig, ReviewConfig
from sidecar.pool import order_pool, resolve_pool


def _m(provider, name="m", max_context=None):
    return ModelConfig(provider=provider, name=name, max_context=max_context)


def test_resolve_pool_uses_providers_when_present():
    review = ReviewConfig(providers=[_m("zai-coding"), _m("anthropic")])
    pool = resolve_pool(review)
    assert [m.provider for m in pool] == ["zai-coding", "anthropic"]


def test_resolve_pool_falls_back_to_single_model():
    review = ReviewConfig(model=_m("ollama", "kimi"))
    pool = resolve_pool(review)
    assert len(pool) == 1
    assert pool[0].provider == "ollama" and pool[0].name == "kimi"


def test_order_pool_priority_when_nothing_cooled():
    pool = [_m("zai-coding"), _m("anthropic"), _m("ollama")]
    assert [m.provider for m in order_pool(pool, set())] == [
        "zai-coding",
        "anthropic",
        "ollama",
    ]


def test_order_pool_puts_cooled_last():
    pool = [_m("zai-coding"), _m("anthropic"), _m("ollama")]
    ordered = order_pool(pool, {"zai-coding"})
    assert [m.provider for m in ordered] == ["anthropic", "ollama", "zai-coding"]


def test_order_pool_all_cooled_still_attempts_in_priority_order():
    pool = [_m("zai-coding"), _m("anthropic")]
    ordered = order_pool(pool, {"zai-coding", "anthropic"})
    assert [m.provider for m in ordered] == ["zai-coding", "anthropic"]


def test_order_pool_demotes_too_small_provider():
    pool = [_m("ollama", max_context=8000), _m("anthropic", max_context=200000)]
    ordered = order_pool(pool, set(), required_tokens=50000)
    assert [m.provider for m in ordered] == ["anthropic", "ollama"]


def test_order_pool_unknown_max_context_assumed_to_fit():
    pool = [_m("zai-coding", max_context=None), _m("ollama", max_context=8000)]
    ordered = order_pool(pool, set(), required_tokens=50000)
    assert [m.provider for m in ordered] == ["zai-coding", "ollama"]


def test_order_pool_fitting_but_cooled_beats_available_too_small():
    pool = [_m("zai-coding", max_context=128000), _m("ollama", max_context=8000)]
    ordered = order_pool(pool, {"zai-coding"}, required_tokens=50000)
    assert [m.provider for m in ordered] == ["zai-coding", "ollama"]
