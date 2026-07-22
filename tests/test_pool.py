"""Unit tests for unified model resolution and failover ordering."""

from sidecar.fukoconfig import CompareModel, ModelConfig, ReviewConfig, ReviewModel
from sidecar.pool import order_pool, partition_roles, resolve_models


def _m(provider, name="m", max_context=None):
    return ModelConfig(provider=provider, name=name, max_context=max_context)


def test_resolve_models_prefers_unified_list_over_legacy():
    review = ReviewConfig(
        models=[
            ReviewModel(provider="zai-coding", name="glm-5.2"),
            ReviewModel(provider="openrouter", name="fallback", role="backup"),
        ],
        providers=[_m("anthropic")],
        compare=[CompareModel(provider="ollama", name="q")],
    )
    models = resolve_models(review)
    assert [(m.provider, m.role) for m in models] == [
        ("zai-coding", "active"),
        ("openrouter", "backup"),
    ]


def test_resolve_models_maps_compare_to_all_active():
    review = ReviewConfig(
        compare=[
            CompareModel(provider="zai-coding", name="a", token_env="TOK_A"),
            CompareModel(provider="openrouter", name="b"),
        ]
    )
    models = resolve_models(review)
    assert [(m.provider, m.role, m.token_env) for m in models] == [
        ("zai-coding", "active", "TOK_A"),
        ("openrouter", "active", None),
    ]


def test_resolve_models_maps_providers_to_active_plus_backups():
    review = ReviewConfig(providers=[_m("zai-coding"), _m("anthropic"), _m("ollama")])
    models = resolve_models(review)
    assert [(m.provider, m.role) for m in models] == [
        ("zai-coding", "active"),
        ("anthropic", "backup"),
        ("ollama", "backup"),
    ]


def test_resolve_models_compare_precedes_providers():
    review = ReviewConfig(
        compare=[CompareModel(provider="anthropic", name="a")],
        providers=[_m("ollama")],
    )
    models = resolve_models(review)
    assert [(m.provider, m.role) for m in models] == [("anthropic", "active")]


def test_resolve_models_falls_back_to_single_model():
    review = ReviewConfig(model=_m("ollama", "kimi"))
    models = resolve_models(review)
    assert [(m.provider, m.name, m.role) for m in models] == [("ollama", "kimi", "active")]


def test_resolve_models_default_config_yields_one_active():
    models = resolve_models(ReviewConfig())
    assert [(m.provider, m.role) for m in models] == [("ollama", "active")]


def test_partition_roles_preserves_config_order():
    models = [
        ReviewModel(provider="a", role="active"),
        ReviewModel(provider="b", role="backup"),
        ReviewModel(provider="c", role="active"),
        ReviewModel(provider="d", role="backup"),
    ]
    actives, backups = partition_roles(models)
    assert [m.provider for m in actives] == ["a", "c"]
    assert [m.provider for m in backups] == ["b", "d"]


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
