"""Multi-credential providers (#298/#299): numbered sibling keys, drained in order."""

import pytest

from sidecar import run_metrics, runner
from sidecar.backends.agentic import AgenticBackend, _provider_key_vars, _transcript_secrets
from sidecar.backends.base import InvokeResult, PRRef
from sidecar.backends.pragent import PrAgentBackend
from sidecar.fukoconfig import ModelConfig, ReviewConfig, ReviewModel, load_config
from sidecar.pool import breaker_key, expand_credentials, order_pool
from sidecar.presets import (
    PRESETS,
    CredentialGapError,
    ProviderPreset,
    credential_env,
    get_preset,
    provider_key_vars,
    resolve_credentials,
)

# conftest stubs `runner._record_run` for every test; keep the real one.
_REAL_RECORD_RUN = runner._record_run

QWEN = get_preset("qwen-anthropic")
KEY = "QWEN_TOKEN_PLAN_KEY"


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch):
    """A developer's shell may export real keys and siblings; start from none."""
    import os

    for name in provider_key_vars(os.environ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_review_state(monkeypatch):
    """Same gate as ``tests/test_agentic_backend.py``: CI exports
    ``FUKO_DATABASE_URL``, and the tests below that drive ``invoke`` must not
    read or write the live test Postgres through the ledger."""
    from sidecar import review_state
    from sidecar import review_state_client as rsc

    monkeypatch.setattr(review_state.settings, "database_url", "")
    monkeypatch.setattr(rsc, "_transport_down", False)


def _qwen(**kw):
    return ReviewModel(provider="qwen-anthropic", name="qwen3.8-max", auth="api-key", **kw)


# --- resolution ---------------------------------------------------------------


def test_contiguous_siblings_resolve_in_order_and_stop_at_the_first_unset():
    env = {KEY: "k1", f"{KEY}_2": "k2"}
    assert resolve_credentials(QWEN, env) == ["k1", "k2"]


def test_a_single_key_resolves_to_itself():
    assert resolve_credentials(QWEN, {KEY: "k1"}) == ["k1"]


def test_an_empty_sibling_reads_as_absent():
    """GitHub renders an unset secret as an empty string."""
    assert resolve_credentials(QWEN, {KEY: "k1", f"{KEY}_2": "", f"{KEY}_3": ""}) == ["k1"]


def test_no_key_and_no_key_env_resolve_to_nothing():
    assert resolve_credentials(QWEN, {}) == []
    assert resolve_credentials(get_preset("ollama"), {KEY: "k1"}) == []


@pytest.mark.parametrize(
    "env",
    [{f"{KEY}_2": "k2"}, {KEY: "k1", f"{KEY}_3": "k3"}],
    ids=["base-unset", "interior-gap"],
)
def test_a_gap_is_a_config_error_naming_variables_not_values(env):
    with pytest.raises(CredentialGapError) as err:
        resolve_credentials(QWEN, env)
    assert KEY in str(err.value)
    assert not any(value in str(err.value) for value in env.values())


def test_names_that_merely_resemble_a_sibling_are_not_credentials():
    env = {KEY: "k1", f"{KEY}_02": "x", f"{KEY}_1": "x", f"{KEY}_ALT": "x", f"X{KEY}_2": "x"}
    assert resolve_credentials(QWEN, env) == ["k1"]


def test_credential_env_spells_the_first_key_bare():
    assert credential_env(KEY, 1) == KEY
    assert credential_env(KEY, 3) == f"{KEY}_3"


# --- pool expansion and ordering ------------------------------------------------


def test_expansion_keeps_credentials_adjacent_and_single_entries_untouched():
    qwen, other = _qwen(), ModelConfig(provider="openrouter", name="m")
    counts = {"qwen-anthropic": 2, "openrouter": 1}
    pool = expand_credentials([qwen, other], lambda m: counts[m.provider])
    assert [(m.provider, m.credential) for m in pool] == [
        ("qwen-anthropic", 1),
        ("qwen-anthropic", 2),
        ("openrouter", 1),
    ]
    # Byte-identical for a single-key entry, and identity-preserving for the
    # branch entry: `_with_branch_ledger` recognises the branch by `is`.
    assert pool[0] is qwen and pool[2] is other


def test_an_entry_with_no_key_is_still_one_attempt():
    model = ModelConfig(provider="ollama", name="m")
    assert expand_credentials([model], lambda m: 0) == [model]


def test_the_index_survives_a_copy_and_never_serializes():
    second = _qwen().on_credential(2)
    assert second.model_copy(update={"findings_ledger": False}).credential == 2
    assert "credential" not in second.model_dump()


def test_the_config_file_cannot_choose_a_credential(tmp_path):
    """Keys live in the environment only -- no `.fuko.toml` surface (#298)."""
    path = tmp_path / ".fuko.toml"
    path.write_text(
        '[[review.models]]\nprovider = "qwen-anthropic"\nname = "q"\n'
        "credential = 2\n_credential = 2\n"
    )
    assert load_config(path).review.models[0].credential == 1


def test_a_cooling_credential_does_not_demote_its_sibling():
    first = _qwen()
    second = first.on_credential(2)
    assert breaker_key(first) == "qwen-anthropic"
    assert breaker_key(second) == "qwen-anthropic#2"
    assert order_pool([first, second], {"qwen-anthropic"}) == [second, first]
    assert order_pool([first, second], {"qwen-anthropic#2"}) == [first, second]


# --- failover through the runner --------------------------------------------------


class _Backend:
    """Records each attempt's credential; answers from a scripted list."""

    def __init__(self, results):
        self.results = list(results)
        self.seen: list[int] = []

    def build_env(self, preset, model, knowledge, tools):
        self.seen.append(model.credential)
        return {}

    def invoke(self, pr, env, tools):
        return self.results.pop(0)


def _run(monkeypatch, backend, pool, cooled=frozenset()):
    recorded, trips = [], []
    monkeypatch.setattr(runner, "_normalize", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "_record_transcript", lambda *a, **k: None)
    monkeypatch.setattr(
        runner, "_record_run", lambda pr, model, **kw: recorded.append((model, kw["outcome"]))
    )
    monkeypatch.setattr(runner, "_cb_trip", lambda key, *a, **k: trips.append(key))
    result = runner._run_pool(
        backend,
        PRRef("o/r", 8, "u"),
        "",
        {},
        ReviewConfig(tool_timeout=5),
        pool,
        set(cooled),
        None,
        tools=["review"],
    )
    return result, recorded, trips


_OK = InvokeResult(returncode=0)
_THROTTLED = InvokeResult(returncode=1, throttled=True, detail="429 quota")
_REJECTED = InvokeResult(returncode=1, auth_failed=True, detail="401")


def test_a_healthy_first_credential_is_the_only_one_ever_used(monkeypatch):
    """Sequential drain: credential 2's quota clock must stay unstarted."""
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_2", "k2")
    backend = _Backend([_OK])
    result, recorded, trips = _run(monkeypatch, backend, [_qwen()])
    assert backend.seen == [1] and trips == []
    assert result.credential == 1
    assert recorded[0][0].credential == 1


def test_a_throttle_fails_over_to_the_next_credential_of_the_same_entry(monkeypatch, capsys):
    monkeypatch.setenv(KEY, "k1-secret")
    monkeypatch.setenv(f"{KEY}_2", "k2-secret")
    backend = _Backend([_THROTTLED, _OK])
    result, recorded, trips = _run(monkeypatch, backend, [_qwen()])
    assert backend.seen == [1, 2]
    # Only the throttled credential cools; the label stays the pool entry's, so
    # the receipt does not read a second key as a promoted backup.
    assert trips == ["qwen-anthropic"]
    assert result.returncode == 0 and result.credential == 2
    assert result.provider == "qwen-anthropic/qwen3.8-max"
    assert [(m.credential, outcome) for m, outcome in recorded] == [(2, "ok")]
    log = capsys.readouterr().err
    assert "attempt 2/2 via qwen-anthropic/qwen3.8-max (credential #2)" in log
    assert "k1-secret" not in log and "k2-secret" not in log


def test_every_credential_throttling_exhausts_the_pool_and_trips_each(monkeypatch):
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_2", "k2")
    result, recorded, trips = _run(monkeypatch, _Backend([_THROTTLED, _THROTTLED]), [_qwen()])
    assert result.throttled
    assert trips == ["qwen-anthropic", "qwen-anthropic#2"]
    assert [(m.credential, outcome) for m, outcome in recorded] == [(2, "throttled_out")]


def test_a_cooling_first_credential_is_tried_after_its_sibling(monkeypatch):
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_2", "k2")
    backend = _Backend([_OK])
    _run(monkeypatch, backend, [_qwen()], cooled={"qwen-anthropic"})
    assert backend.seen == [2]


def test_a_single_key_entry_runs_exactly_as_before(monkeypatch, capsys):
    monkeypatch.setenv(KEY, "k1")
    backend = _Backend([_THROTTLED])
    result, _, trips = _run(monkeypatch, backend, [_qwen()])
    assert backend.seen == [1] and trips == ["qwen-anthropic"]
    assert "credential #" not in capsys.readouterr().err
    assert result.credential == 1


def test_a_rejected_key_moves_on_to_its_sibling_without_tripping(monkeypatch):
    """A revoked key 1 must not take the seat down while key 2 is good."""
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_2", "k2")
    backend = _Backend([_REJECTED, _OK])
    result, _, trips = _run(monkeypatch, backend, [_qwen()])
    assert backend.seen == [1, 2] and trips == []
    assert result.returncode == 0 and result.credential == 2


def test_a_rejected_key_with_no_sibling_left_still_fails_fast(monkeypatch):
    """Auth failures never burn the pool across PROVIDERS."""
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv("ANTHROPIC_KEY", "a1")
    backup = ReviewModel(provider="anthropic", name="b", auth="api-key", role="backup")
    backend = _Backend([_REJECTED, _OK])
    result, recorded, _ = _run(monkeypatch, backend, [_qwen(), backup])
    assert backend.seen == [1] and result.returncode == 1
    assert recorded[0][1] == "failed"


def test_a_gap_fails_the_pool_that_uses_the_preset_and_no_other(monkeypatch):
    monkeypatch.setenv(f"{KEY}_2", "k2")
    monkeypatch.setenv("ANTHROPIC_KEY", "a1")
    with pytest.raises(CredentialGapError):
        _run(monkeypatch, _Backend([_OK]), [_qwen()])
    other = ReviewModel(provider="anthropic", name="b", auth="api-key")
    result, _, _ = _run(monkeypatch, _Backend([_OK]), [other])
    assert result.returncode == 0


def test_the_metrics_row_names_the_credential_only_when_it_is_not_the_first(monkeypatch):
    posted = []

    class _Resp:
        def raise_for_status(self):
            return None

    monkeypatch.setenv("FUKO_URL", "http://fuko")
    monkeypatch.setattr(
        runner.httpx, "post", lambda url, json, **kw: posted.append(json) or _Resp()
    )
    for model in (_qwen(), _qwen().on_credential(2)):
        _REAL_RECORD_RUN(
            PRRef("o/r", 8, "u"),
            model,
            slot=None,
            duration_s=1,
            attempts=1,
            outcome="ok",
            findings=0,
            detail="",
        )
    assert "credential" not in posted[0]
    assert posted[1]["credential"] == 2


def test_record_writes_the_credential_column(monkeypatch):
    statements = []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=()):
            statements.append((sql, params))

    monkeypatch.setattr(run_metrics, "_enabled", lambda: True)
    monkeypatch.setattr("sidecar.db.db_best_effort", lambda: _Conn())
    run_metrics.record("o/r", 7, "qwen-anthropic", "m", credential=2)
    sql, params = statements[-1]
    columns = sql.split("(", 1)[1].split(")", 1)[0].replace('" "', "").split(", ")
    assert params[[c.strip('" ') for c in columns].index("credential")] == 2


def test_the_receipt_carries_the_index():
    from sidecar.signals import RunReceipt

    receipt = RunReceipt(label="p/m", head_sha="abc", credential=2)
    assert RunReceipt.model_validate_json(receipt.model_dump_json()).credential == 2
    assert RunReceipt(label="p/m", head_sha="abc").credential == 1


# --- backends ---------------------------------------------------------------------


def test_agentic_injects_the_credential_the_pool_entry_names(monkeypatch):
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_2", "k2")
    backend = AgenticBackend()
    assert backend.build_env(QWEN, _qwen(), "", ["review"])["ANTHROPIC_API_KEY"] == "k1"
    second = backend.build_env(QWEN, _qwen().on_credential(2), "", ["review"])
    assert second["ANTHROPIC_API_KEY"] == "k2"
    assert backend.credential_count(QWEN, _qwen()) == 2


def test_agentic_subscription_entries_are_never_expanded(monkeypatch):
    """Subscription sends no key, so N exported keys must not mean N attempts."""
    monkeypatch.setenv("ANTHROPIC_KEY", "a1")
    monkeypatch.setenv("ANTHROPIC_KEY_2", "a2")
    model = ReviewModel(provider="anthropic", name="claude", auth="subscription")
    assert AgenticBackend().credential_count(get_preset("anthropic"), model) == 0


def test_agentic_auto_auth_with_only_a_sibling_set_reports_the_gap(monkeypatch):
    """`auto` must not read "base unset" as a subscription runner and review
    under the runner's own login while a key sits exported."""
    monkeypatch.setenv("ANTHROPIC_KEY_2", "a2")
    model = ReviewModel(provider="anthropic", name="claude")
    with pytest.raises(CredentialGapError):
        AgenticBackend().build_env(get_preset("anthropic"), model, "", ["review"])


def test_pragent_reads_the_credential_the_pool_entry_names(monkeypatch):
    monkeypatch.setenv("ZAI_KEY", "z1")
    monkeypatch.setenv("ZAI_KEY_2", "z2")
    preset = get_preset("zai-coding")
    model = ModelConfig(provider="zai-coding", name="glm")
    backend = PrAgentBackend()
    assert backend.build_env(preset, model, "", ["review"])["OPENAI__KEY"] == "z1"
    assert backend.build_env(preset, model.on_credential(2), "", ["review"])["OPENAI__KEY"] == "z2"


# --- the security boundary: strip and scrub ---------------------------------------


def test_every_presets_siblings_are_stripped_and_scrubbed(monkeypatch):
    """Derived from `PRESETS`, not one named variable: a sibling that escapes
    either list is a live billable key in PR-author-visible output."""
    monkeypatch.setitem(
        PRESETS, "future", ProviderPreset(litellm_prefix="openai/", key_env="FUTURE_KEY")
    )
    expected = {}
    for preset in PRESETS.values():
        if preset.key_env:
            for index in (1, 2, 3):
                name = credential_env(preset.key_env, index)
                expected[name] = f"secret-{name}"
                monkeypatch.setenv(name, expected[name])
    assert set(expected) <= _provider_key_vars()
    assert expected.items() <= dict(_transcript_secrets({}, "")).items()


def test_a_sibling_stranded_past_a_gap_is_still_stripped_and_scrubbed(monkeypatch):
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_3", "k3-stranded")
    assert f"{KEY}_3" in _provider_key_vars()
    assert dict(_transcript_secrets({}, ""))[f"{KEY}_3"] == "k3-stranded"


def test_the_harness_on_credential_one_holds_no_sibling(monkeypatch):
    import tests.test_agentic_backend as helpers

    monkeypatch.setenv(KEY, "k1-secret")
    monkeypatch.setenv(f"{KEY}_2", "k2-secret")
    _, captured = helpers._invoke(
        monkeypatch,
        AgenticBackend(),
        helpers.HarnessResult(0, helpers.REVIEW_JSON),
        env={
            "FUKO_AGENTIC_MODEL": "qwen3.8-max",
            "FUKO_AGENTIC_AUTH": "api-key",
            "ANTHROPIC_API_KEY": "k1-secret",
        },
    )
    assert KEY not in captured["env"] and f"{KEY}_2" not in captured["env"]
    assert "k2-secret" not in captured["env"].values()
    assert captured["env"]["ANTHROPIC_API_KEY"] == "k1-secret"


def test_the_agentic_driver_flags_a_rejected_key_and_nothing_else(monkeypatch):
    import tests.test_agentic_backend as helpers

    rejected, _ = helpers._invoke(
        monkeypatch, AgenticBackend(), helpers.HarnessResult(1, "", stderr="invalid api key")
    )
    assert rejected.auth_failed and not rejected.throttled
    throttled, _ = helpers._invoke(
        monkeypatch, AgenticBackend(), helpers.HarnessResult(1, "", stderr="429 rate limit")
    )
    assert throttled.throttled and not throttled.auth_failed


# --- review round 1 (#311) ------------------------------------------------------------


def test_a_rejection_jumps_to_its_sibling_past_an_intervening_backup(monkeypatch):
    """A cooling sibling is ordered BEHIND an available backup. A rejected key
    is not a throttle, so the backup must not run on the way to the sibling."""
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_2", "k2")
    monkeypatch.setenv("ANTHROPIC_KEY", "a1")
    backup = ReviewModel(provider="anthropic", name="b", auth="api-key", role="backup")
    attempts: list[tuple[str, int]] = []

    class Backend(_Backend):
        def build_env(self, preset, model, knowledge, tools):
            attempts.append((model.provider, model.credential))
            return {}

    result, _, trips = _run(
        monkeypatch, Backend([_REJECTED, _OK]), [_qwen(), backup], cooled={"qwen-anthropic#2"}
    )
    assert attempts == [("qwen-anthropic", 1), ("qwen-anthropic", 2)]
    assert result.returncode == 0 and result.credential == 2 and trips == []


def test_a_rejected_login_does_not_walk_into_a_same_provider_backups_keys(monkeypatch):
    """A subscription seat has no sibling: its entry is never expanded. A
    backup on the same provider with two keys is a different entry."""
    monkeypatch.setenv("ANTHROPIC_KEY", "a1")
    monkeypatch.setenv("ANTHROPIC_KEY_2", "a2")
    seat = ReviewModel(provider="anthropic", name="claude", auth="subscription")
    backup = ReviewModel(provider="anthropic", name="claude", auth="api-key", role="backup")
    counts = {"subscription": 0, "api-key": 2}

    class Backend(_Backend):
        def credential_count(self, preset, model):
            return counts[model.auth]

    backend = Backend([_REJECTED, _OK])
    result, recorded, _ = _run(monkeypatch, backend, [seat, backup])
    assert backend.seen == [1] and result.returncode == 1
    assert recorded[0][1] == "failed"


def test_a_rejected_key_is_not_retried_under_an_identical_entry(monkeypatch):
    """Two entries on one provider hold the SAME keys: once both are rejected
    the pool stops instead of re-sending each key a second time."""
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_2", "k2")
    backend = _Backend([_REJECTED, _REJECTED, _OK, _OK])
    result, _, _ = _run(monkeypatch, backend, [_qwen(), _qwen(role="backup")])
    assert backend.seen == [1, 2] and result.returncode == 1


def test_misnamed_first_keys_are_stripped_and_scrubbed_but_never_used(monkeypatch):
    """`_1` and `_01` are the natural misnamings of the first key. They are not
    credentials to the resolver, and they still hold one."""
    monkeypatch.setenv(f"{KEY}_1", "k1-misnamed")
    monkeypatch.setenv(f"{KEY}_01", "k1-padded")
    monkeypatch.setenv(f"{KEY}_ALT", "not-a-sibling")
    assert resolve_credentials(QWEN) == []
    assert {f"{KEY}_1", f"{KEY}_01"} <= _provider_key_vars()
    assert f"{KEY}_ALT" not in _provider_key_vars()
    scrubbed = dict(_transcript_secrets({}, ""))
    assert scrubbed[f"{KEY}_1"] == "k1-misnamed" and scrubbed[f"{KEY}_01"] == "k1-padded"


@pytest.mark.parametrize(
    "stderr",
    [
        "API Error: 403 Forbidden",
        '{"type":"error","error":{"type":"permission_error"}}',
        "Your API key has expired.",
        "the token was revoked",
    ],
)
def test_a_dead_key_reads_as_rejected_in_every_wording(monkeypatch, stderr):
    import tests.test_agentic_backend as helpers

    result, _ = helpers._invoke(
        monkeypatch, AgenticBackend(), helpers.HarnessResult(1, "", stderr=stderr)
    )
    assert result.auth_failed and not result.throttled


@pytest.mark.parametrize("stderr", ["403 insufficient quota", "session expired, 429 rate limit"])
def test_a_quota_refusal_stays_a_throttle(monkeypatch, stderr):
    import tests.test_agentic_backend as helpers

    result, _ = helpers._invoke(
        monkeypatch, AgenticBackend(), helpers.HarnessResult(1, "", stderr=stderr)
    )
    assert result.throttled and not result.auth_failed


# --- review round 2 (#311) ------------------------------------------------------------


def _attempt_log(results):
    attempts: list[tuple[str, int]] = []

    class Backend(_Backend):
        def build_env(self, preset, model, knowledge, tools):
            attempts.append((model.provider, model.credential))
            return {}

    return Backend(results), attempts


def test_a_backup_passed_over_for_the_sibling_still_rescues_a_throttled_sibling(monkeypatch):
    """Jumping to the sibling defers the backup; it must not consume it."""
    monkeypatch.setenv(KEY, "k1")
    monkeypatch.setenv(f"{KEY}_2", "k2")
    monkeypatch.setenv("ANTHROPIC_KEY", "a1")
    backup = ReviewModel(provider="anthropic", name="b", auth="api-key", role="backup")
    backend, attempts = _attempt_log([_REJECTED, _THROTTLED, _OK])
    result, _, trips = _run(monkeypatch, backend, [_qwen(), backup], cooled={"qwen-anthropic#2"})
    assert attempts == [("qwen-anthropic", 1), ("qwen-anthropic", 2), ("anthropic", 1)]
    assert result.returncode == 0 and trips == ["qwen-anthropic#2"]


def test_a_rejected_key_does_not_drop_a_subscription_backup_on_the_same_provider(monkeypatch):
    """A subscription entry shares the provider's breaker key and none of its
    credentials, so a rejected key 1 says nothing about it."""
    monkeypatch.setenv("ANTHROPIC_KEY", "a1")
    monkeypatch.setenv("ANTHROPIC_KEY_2", "a2")
    seat = ReviewModel(provider="anthropic", name="claude", auth="api-key")
    backup = ReviewModel(provider="anthropic", name="claude", auth="subscription", role="backup")
    names: list[str] = []
    counts = {"subscription": 0, "api-key": 2}

    class Backend(_Backend):
        def credential_count(self, preset, model):
            return counts[model.auth]

        def build_env(self, preset, model, knowledge, tools):
            names.append(f"{model.auth}#{model.credential}")
            return {}

    result, _, _ = _run(monkeypatch, Backend([_REJECTED, _THROTTLED, _OK]), [seat, backup])
    assert names == ["api-key#1", "api-key#2", "subscription#1"]
    assert result.returncode == 0


def test_a_quota_refusal_sent_as_403_forbidden_stays_a_throttle(monkeypatch):
    import tests.test_agentic_backend as helpers

    result, _ = helpers._invoke(
        monkeypatch,
        AgenticBackend(),
        helpers.HarnessResult(1, "", stderr="API Error: 403 Forbidden: insufficient quota"),
    )
    assert result.throttled and not result.auth_failed


def test_review_prose_about_expired_tokens_is_not_a_rejected_key(monkeypatch):
    """The agent's own text is PR-author influenced; only stderr may say the
    key was refused in the weaker wordings."""
    import tests.test_agentic_backend as helpers

    prose = "The token has expired branch never refreshes; permission_error is swallowed."
    result, _ = helpers._invoke(
        monkeypatch, AgenticBackend(), helpers.HarnessResult(1, prose, stderr="429 rate limit")
    )
    assert result.throttled and not result.auth_failed
    plain, _ = helpers._invoke(
        monkeypatch, AgenticBackend(), helpers.HarnessResult(1, prose, stderr="boom")
    )
    assert not plain.auth_failed
