"""Tests for reviewer-health persistence, the escalation policy, and its endpoints."""

from fastapi.testclient import TestClient

from sidecar import main, reviewer_health, runner
from sidecar.backends.base import InvokeResult
from sidecar.status import escalation_needed

_TOKEN = "test-token"

_REAL_RH_STATES = runner._rh_states
_REAL_OBSERVE = runner._observe_reviewer_health


def _client(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    return TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})


def test_escalation_needed_on_each_degraded_state():
    for state in ("rate_limited", "paused", "unavailable"):
        assert escalation_needed([{"backend": "coderabbit", "state": state}])


def test_escalation_not_needed_on_healthy_states():
    rows = [
        {"backend": "coderabbit", "state": "done"},
        {"backend": "copilot", "state": "pending"},
        {"backend": "x", "state": "in_progress"},
        {"backend": "y", "state": "none"},
    ]
    assert not escalation_needed(rows)
    assert not escalation_needed([])


def test_escalation_accepts_stored_row_shape():
    rows = [
        {"reviewer": "copilot", "state": "unavailable", "observed_at": "x", "pr": 7, "detail": ""}
    ]
    assert escalation_needed(rows)


def test_reviewer_health_no_ops_without_database(monkeypatch):
    monkeypatch.setattr(reviewer_health.settings, "database_url", "")
    assert reviewer_health.observe("o/r", "coderabbit", "rate_limited", 7, "429") is None
    assert reviewer_health.states("o/r") == []


def test_rh_state_endpoint(monkeypatch):
    rows = [
        {
            "reviewer": "coderabbit",
            "state": "rate_limited",
            "observed_at": "2026-07-22T12:00:00+00:00",
            "pr": 7,
            "detail": "rate-limit notice",
        }
    ]
    monkeypatch.setattr(reviewer_health, "states", lambda repo: rows if repo == "o/r" else [])
    resp = _client(monkeypatch).get("/rh/state", params={"repo": "o/r"})
    assert resp.status_code == 200
    assert resp.json() == {"reviewers": rows}


def test_rh_observe_endpoint(monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", "")
    seen = []
    monkeypatch.setattr(
        reviewer_health,
        "observe",
        lambda repo, rev, state, pr, detail: seen.append((repo, rev, state, pr, detail)),
    )
    resp = _client(monkeypatch).post(
        "/rh/observe",
        json={
            "repo": "o/r",
            "pr": 7,
            "observations": [
                {"reviewer": "coderabbit", "state": "done", "detail": "scanned HEAD"},
                {"reviewer": "copilot", "state": "unavailable", "detail": "quota"},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"recorded": 2, "persisted": False}
    assert seen == [
        ("o/r", "coderabbit", "done", 7, "scanned HEAD"),
        ("o/r", "copilot", "unavailable", 7, "quota"),
    ]


def test_rh_observe_endpoint_reports_persisted_with_database(monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", "postgresql://x/y")
    monkeypatch.setattr(reviewer_health, "observe", lambda *a: None)
    resp = _client(monkeypatch).post(
        "/rh/observe",
        json={"repo": "o/r", "pr": 7, "observations": [{"reviewer": "copilot", "state": "done"}]},
    )
    assert resp.json() == {"recorded": 1, "persisted": True}


def test_rh_states_reads_sidecar_over_http(monkeypatch):
    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")
    monkeypatch.setenv("FUKO_TOKEN", "t")

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"reviewers": [{"reviewer": "coderabbit", "state": "paused"}]}

    monkeypatch.setattr(
        runner.httpx, "get", lambda url, params=None, headers=None, timeout=None: _Resp()
    )
    assert _REAL_RH_STATES("o/r") == [{"reviewer": "coderabbit", "state": "paused"}]


def test_rh_states_empty_on_http_error(monkeypatch):
    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")

    def boom(*a, **k):
        raise RuntimeError("sidecar down")

    monkeypatch.setattr(runner.httpx, "get", boom)
    assert _REAL_RH_STATES("o/r") == []


def _wire_review(monkeypatch, cfg_path, backend):
    monkeypatch.setattr(runner, "build_knowledge", lambda *a: "")
    monkeypatch.setattr(runner, "_cb_cooldowns", lambda: set())
    monkeypatch.setattr(runner, "_estimate_required_context", lambda *a: None)
    monkeypatch.setattr(runner, "get_backend", lambda name, config=None: backend)
    monkeypatch.setattr(runner, "_post_branch_header", lambda *a, **k: (None, None))


def _one_active_one_backup(tmp_path):
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "anthropic"\nname = "a"\n'
        "[[review.models]]\n"
        'provider = "ollama"\nname = "b"\nrole = "backup"\n',
        encoding="utf-8",
    )
    return cfg


class _CountingBackend:
    def __init__(self):
        self.models = []

    def build_env(self, preset, model, knowledge, tools):
        self.models.append(f"{model.provider}/{model.name}")
        return {}

    def invoke(self, pr, env, tools):
        return InvokeResult(returncode=0)

    def normalize_output(self, pr, model="", *, compare_label=None, **_kw):
        return []


def test_review_promotes_backups_when_degraded(monkeypatch, tmp_path, capsys):
    cfg = _one_active_one_backup(tmp_path)
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    backend = _CountingBackend()
    _wire_review(monkeypatch, cfg, backend)
    monkeypatch.setattr(
        runner, "_rh_states", lambda repo: [{"reviewer": "coderabbit", "state": "rate_limited"}]
    )
    observed = []
    monkeypatch.setattr(
        runner, "_observe_reviewer_health", lambda pr, token, api_url: observed.append(pr.repo)
    )

    result = runner.review("https://github.com/o/r/pull/7", str(cfg))

    assert result.returncode == 0
    assert backend.models == ["anthropic/a", "ollama/b"]
    assert observed == ["o/r"]
    err = capsys.readouterr().err
    assert "promoted ollama/b" in err
    # The budget arithmetic is PRINTED, not carried in a comment: it is a function
    # of the backup count and has gone stale every time a human held it (#106).
    assert "sequential worst case" in err
    assert "branches x" in err


# --- escalation invariants (#106) --------------------------------------------
#
# These assert a PROPERTY of the branch set the runner actually builds -- every
# entry that starts a branch resolves to a distinct identity, evaluated after
# role partitioning and after promotion -- rather than the shape of any one
# `.fuko.toml`. A config-shape assertion would have passed on the run that broke:
# the config was fine, and escalation introduced the identity-less branch at
# runtime.


def _models(*specs):
    """Build ReviewModel entries from ``(provider, name, role, token_env)`` tuples."""
    from sidecar.fukoconfig import ReviewModel

    return [ReviewModel(provider=p, name=n, role=r, token_env=t) for p, n, r, t in specs]


def _identities_are_distinct(entries) -> bool:
    """The invariant: every branch-starting entry has its own resolvable identity."""
    envs = [e.token_env for e in entries]
    if not all(envs):
        return False
    values = [runner.os.environ.get(e, "") for e in envs]
    return all(values) and len(set(values)) == len(values)


def test_escalation_skips_a_promotion_that_would_collapse_concurrency(monkeypatch):
    """The case that failed live: promotion adds a branch with no identity.

    Concurrency is all-or-nothing, so ONE token-less branch collapses every
    branch onto a single token — the escalation meant to ADD a reviewer removes
    identity separation from the whole fleet. The invariant must hold by SKIPPING
    the promotion, not by degrading the run to sequential.
    """
    from sidecar.fukoconfig import ReviewConfig

    monkeypatch.setenv("T_DORIAN", "tok-a")
    monkeypatch.setenv("T_GRAY", "tok-b")
    actives = _models(
        ("zai-coding", "glm-5.2", "active", "T_DORIAN"),
        ("openrouter", "qwen/qwen3.8-max", "active", "T_GRAY"),
    )
    backups = _models(("ollama-cloud", "glm-5.2:cloud", "backup", None))

    promote, reasons = runner.plan_escalation(actives, backups, ReviewConfig(), concurrent=True)

    assert promote == []
    assert reasons and "collapse" in reasons[0]
    # The invariant holds over the post-promotion branch set.
    assert _identities_are_distinct([*actives, *promote])


def test_escalation_promotes_a_backup_that_has_its_own_identity(monkeypatch):
    """The invariant permits the promotion when it does not break identity."""
    from sidecar.fukoconfig import ReviewConfig

    monkeypatch.setenv("T_DORIAN", "tok-a")
    monkeypatch.setenv("T_GRAY", "tok-b")
    monkeypatch.setenv("T_BASIL", "tok-c")
    actives = _models(
        ("zai-coding", "glm-5.2", "active", "T_DORIAN"),
        ("openrouter", "qwen/qwen3.8-max", "active", "T_GRAY"),
    )
    backups = _models(("ollama-cloud", "glm-5.2:cloud", "backup", "T_BASIL"))

    promote, reasons = runner.plan_escalation(actives, backups, ReviewConfig(), concurrent=True)

    assert [f"{m.provider}/{m.name}" for m in promote] == ["ollama-cloud/glm-5.2:cloud"]
    assert reasons == []
    assert promote[0].role == "active" and promote[0].promoted is True
    assert _identities_are_distinct([*actives, *promote])


def test_escalation_rescues_in_a_total_outage(monkeypatch):
    """With no surviving active, a backup still promotes -- the case rescue is for.

    An empty `existing` used to fail the same-backend gate's `len != 1` and block
    every promotion with a misleading "does not match []" reason, defeating
    escalation exactly when it matters most.
    """
    from sidecar.fukoconfig import ReviewConfig

    monkeypatch.setenv("T_BASIL", "tok-c")
    backups = _models(("ollama-cloud", "glm-5.2:cloud", "backup", "T_BASIL"))

    promote, reasons = runner.plan_escalation([], backups, ReviewConfig(), concurrent=True)

    assert [f"{m.provider}/{m.name}" for m in promote] == ["ollama-cloud/glm-5.2:cloud"]
    assert reasons == []


def test_escalation_keeps_a_total_outage_round_single_backend():
    """The first promoted backup defines the round's backend; later ones must match.

    Without this, an empty `existing` would let two different-backend backups both
    promote into one mixed-driver round -- the very thing the same-backend gate
    exists to prevent.
    """
    from sidecar.fukoconfig import ReviewConfig, ReviewModel

    backups = [
        ReviewModel(provider="zai-coding", name="glm-5.2", role="backup"),  # inherits pr-agent
        ReviewModel(provider="anthropic", name="claude-x", role="backup", backend="agentic"),
    ]

    promote, reasons = runner.plan_escalation([], backups, ReviewConfig(), concurrent=False)

    assert [f"{m.provider}/{m.name}" for m in promote] == ["zai-coding/glm-5.2"]
    assert len(reasons) == 1 and "does not match" in reasons[0]


def test_escalation_promotes_identity_less_backup_when_already_sequential(monkeypatch):
    """If the actives cannot run concurrently anyway, promotion costs nothing.

    The identity rule exists to protect concurrency; with none to protect,
    declining the extra reviewer would be a pure loss.
    """
    from sidecar.fukoconfig import ReviewConfig

    actives = _models(("zai-coding", "glm-5.2", "active", None))
    backups = _models(("ollama-cloud", "glm-5.2:cloud", "backup", None))

    promote, reasons = runner.plan_escalation(actives, backups, ReviewConfig(), concurrent=False)

    assert len(promote) == 1
    assert reasons == []


def test_escalation_counts_trials_as_branches(monkeypatch):
    """A trial starts its own branch, so it spends budget like an active does.

    Planning against `actives` alone undercounts branches — the runner builds
    `[*actives, *trials]` — so a budget check would pass a promotion the job
    cannot actually hold.
    """
    from sidecar.fukoconfig import ReviewConfig

    for name in ("T_A", "T_T", "T_C"):
        monkeypatch.setenv(name, f"tok-{name}")
    existing = _models(("p", "a", "active", "T_A"), ("p", "t", "trial", "T_T"))
    backups = _models(("p", "c", "backup", "T_C"))
    # 2 existing + 1 promoted = 3 branches x 2 x 600s = 60m, over a 50m budget.
    review = ReviewConfig(tool_timeout=600, job_budget_minutes=50)

    promote, reasons = runner.plan_escalation(existing, backups, review, concurrent=True)

    assert promote == []
    assert reasons and "over the 50m job budget" in reasons[0]


def test_escalation_refuses_a_promotion_the_job_budget_cannot_hold(monkeypatch):
    """Budget is computed, not remembered: branches x tools x tool_timeout."""
    from sidecar.fukoconfig import ReviewConfig

    monkeypatch.setenv("T_A", "tok-a")
    monkeypatch.setenv("T_B", "tok-b")
    monkeypatch.setenv("T_C", "tok-c")
    actives = _models(
        ("p", "a", "active", "T_A"),
        ("p", "b", "active", "T_B"),
    )
    backups = _models(("p", "c", "backup", "T_C"))
    # 3 branches x 2 tools x 600s = 60m, over a 50m budget.
    review = ReviewConfig(tool_timeout=600, job_budget_minutes=50)

    promote, reasons = runner.plan_escalation(actives, backups, review, concurrent=True)

    assert promote == []
    assert reasons and "over the 50m job budget" in reasons[0]


def test_escalation_promotes_what_the_budget_holds_not_all_or_nothing(monkeypatch):
    """A budget that holds one of two backups promotes one rather than refusing both."""
    from sidecar.fukoconfig import ReviewConfig

    for name in ("T_A", "T_B", "T_C", "T_D"):
        monkeypatch.setenv(name, f"tok-{name}")
    actives = _models(("p", "a", "active", "T_A"), ("p", "b", "active", "T_B"))
    backups = _models(("p", "c", "backup", "T_C"), ("p", "d", "backup", "T_D"))
    # 3 branches x 2 x 600s = 60m fits; 4 branches = 80m does not.
    review = ReviewConfig(tool_timeout=600, job_budget_minutes=70)

    promote, reasons = runner.plan_escalation(actives, backups, review, concurrent=True)

    assert [m.name for m in promote] == ["c"]
    assert len(reasons) == 1 and "over the 70m job budget" in reasons[0]


def test_escalation_without_a_budget_promotes_but_still_reports_the_cost(monkeypatch):
    """An unset budget must not mean an unmeasured one."""
    from sidecar.fukoconfig import ReviewConfig

    monkeypatch.setenv("T_A", "tok-a")
    monkeypatch.setenv("T_C", "tok-c")
    actives = _models(("p", "a", "active", "T_A"))
    backups = _models(("p", "c", "backup", "T_C"))

    promote, reasons = runner.plan_escalation(
        actives, backups, ReviewConfig(job_budget_minutes=None), concurrent=True
    )

    assert len(promote) == 1 and reasons == []
    assert runner.sequential_cost_minutes(2, 2, 600) == 40.0


def test_config_parsed_backup_with_token_env_is_promoted(monkeypatch, tmp_path):
    """#114 end-to-end: a role="backup" entry may carry token_env through TOML
    parsing, and such an identity'd backup is then promoted under escalation.

    The runtime for this shipped in #123 (identity-aware promotion, plus the
    direct-object tests above). What was NOT pinned: those tests hand-build
    ReviewModel objects, so none prove token_env survives config PARSING on a
    backup entry -- the literal precondition for the owner's 3rd-App provisioning
    to pay off. Without this, "provisioned the App, escalation still promotes
    nothing" is a silent, assumed-path-never-exercised failure. This walks the
    real path: load_config -> resolve_models -> partition_roles -> plan_escalation.

    NOTE for #99 Part A: promotion eligibility becomes `identity AND same-backend`.
    Every entry here is left on the default (pr-agent) backend, so the backup
    matches its actives' backend and stays promotable under the stricter rule --
    Part A should EXTEND this test, not contradict it.
    """
    from sidecar.fukoconfig import load_config
    from sidecar.pool import partition_roles, resolve_models

    monkeypatch.setenv("FUKO_GITHUB_TOKEN_DORIAN", "tok-a")
    monkeypatch.setenv("FUKO_GITHUB_TOKEN_GRAY", "tok-b")
    monkeypatch.setenv("FUKO_GITHUB_TOKEN_BASIL", "tok-c")
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "zai-coding"\nname = "glm-5.2"\nrole = "active"\n'
        'token_env = "FUKO_GITHUB_TOKEN_DORIAN"\n'
        "[[review.models]]\n"
        'provider = "openrouter"\nname = "qwen/qwen3.8-max"\nrole = "active"\n'
        'token_env = "FUKO_GITHUB_TOKEN_GRAY"\n'
        "[[review.models]]\n"
        'provider = "ollama-cloud"\nname = "glm-5.2:cloud"\nrole = "backup"\n'
        'token_env = "FUKO_GITHUB_TOKEN_BASIL"\n',
        encoding="utf-8",
    )
    review = load_config(cfg).review
    actives, backups, trials = partition_roles(resolve_models(review))
    # token_env is inherited onto backup entries via CompareModel (#114 confirms it
    # is already allowed and must not be re-added) -- assert it parsed, end to end.
    assert backups[0].token_env == "FUKO_GITHUB_TOKEN_BASIL"

    existing = [*actives, *trials]
    promote, reasons = runner.plan_escalation(existing, backups, review, concurrent=True)

    assert [f"{m.provider}/{m.name}" for m in promote] == ["ollama-cloud/glm-5.2:cloud"]
    assert reasons == []
    assert promote[0].role == "active" and promote[0].promoted is True
    assert _identities_are_distinct([*existing, *promote])


def test_config_parsed_backup_without_token_env_still_skips(monkeypatch, tmp_path):
    """The identity-less counterpart, through the same config path, still skips.

    Guards the invariant from the other side: a parsed backup with no token_env
    must not be promoted into a concurrent round (it would collapse every branch
    onto one identity, #106).
    """
    from sidecar.fukoconfig import load_config
    from sidecar.pool import partition_roles, resolve_models

    monkeypatch.setenv("FUKO_GITHUB_TOKEN_DORIAN", "tok-a")
    monkeypatch.setenv("FUKO_GITHUB_TOKEN_GRAY", "tok-b")
    cfg = tmp_path / ".fuko.toml"
    cfg.write_text(
        "[[review.models]]\n"
        'provider = "zai-coding"\nname = "glm-5.2"\nrole = "active"\n'
        'token_env = "FUKO_GITHUB_TOKEN_DORIAN"\n'
        "[[review.models]]\n"
        'provider = "openrouter"\nname = "qwen/qwen3.8-max"\nrole = "active"\n'
        'token_env = "FUKO_GITHUB_TOKEN_GRAY"\n'
        "[[review.models]]\n"
        'provider = "ollama-cloud"\nname = "glm-5.2:cloud"\nrole = "backup"\n',
        encoding="utf-8",
    )
    review = load_config(cfg).review
    actives, backups, trials = partition_roles(resolve_models(review))
    assert backups[0].token_env is None

    existing = [*actives, *trials]
    promote, reasons = runner.plan_escalation(existing, backups, review, concurrent=True)

    assert promote == []
    assert reasons and "collapse" in reasons[0]


def test_escalation_skips_backup_on_backend_mismatch(monkeypatch):
    """#99 same-backend gate: an identity'd backup on a different driver is skipped.

    A promoted backup starts its own branch, which must run under the SAME driver
    as the branches it joins -- a backup rescued into a round of pr-agent branches
    cannot run as an agentic branch. Identity alone is no longer sufficient.
    """
    from sidecar.fukoconfig import ReviewConfig, ReviewModel

    monkeypatch.setenv("T_DORIAN", "tok-a")
    monkeypatch.setenv("T_GRAY", "tok-b")
    monkeypatch.setenv("T_BASIL", "tok-c")
    actives = [
        ReviewModel(provider="zai-coding", name="glm-5.2", role="active", token_env="T_DORIAN"),
        ReviewModel(
            provider="openrouter", name="qwen/qwen3.8-max", role="active", token_env="T_GRAY"
        ),
    ]
    # Identity resolves (T_BASIL set) so the identity gate passes; the backend gate
    # is the one that must skip it.
    backups = [
        ReviewModel(
            provider="ollama-cloud",
            name="glm-5.2:cloud",
            role="backup",
            token_env="T_BASIL",
            backend="agentic",
        )
    ]

    promote, reasons = runner.plan_escalation(actives, backups, ReviewConfig(), concurrent=True)

    assert promote == []
    assert reasons and "does not match" in reasons[0]


def test_escalation_skips_all_when_actives_span_multiple_backends(monkeypatch):
    """Fail-closed: with no single active backend to join, no backup is promoted."""
    from sidecar.fukoconfig import ReviewConfig, ReviewModel

    monkeypatch.setenv("T_DORIAN", "tok-a")
    monkeypatch.setenv("T_GRAY", "tok-b")
    monkeypatch.setenv("T_BASIL", "tok-c")
    actives = [
        ReviewModel(
            provider="zai-coding",
            name="glm-5.2",
            role="active",
            token_env="T_DORIAN",
            backend="agentic",
        ),
        ReviewModel(
            provider="openrouter",
            name="qwen/qwen3.8-max",
            role="active",
            token_env="T_GRAY",
        ),  # inherits pr-agent -> fleet spans two backends
    ]
    backups = [
        ReviewModel(
            provider="ollama-cloud",
            name="glm-5.2:cloud",
            role="backup",
            token_env="T_BASIL",
            backend="agentic",
        )
    ]

    promote, reasons = runner.plan_escalation(actives, backups, ReviewConfig(), concurrent=True)

    assert promote == []
    assert reasons and "does not match" in reasons[0]


def test_escalation_promotes_backup_when_all_share_a_backend(monkeypatch):
    """The rule is symmetric: when the fleet and the backup share ONE backend,
    an identity'd backup promotes (here every entry is on the agentic driver)."""
    from sidecar.fukoconfig import ReviewConfig, ReviewModel

    monkeypatch.setenv("T_DORIAN", "tok-a")
    monkeypatch.setenv("T_GRAY", "tok-b")
    monkeypatch.setenv("T_BASIL", "tok-c")
    actives = [
        ReviewModel(
            provider="zai-coding",
            name="glm-5.2",
            role="active",
            token_env="T_DORIAN",
            backend="agentic",
        ),
        ReviewModel(
            provider="openrouter",
            name="qwen/qwen3.8-max",
            role="active",
            token_env="T_GRAY",
            backend="agentic",
        ),
    ]
    backups = [
        ReviewModel(
            provider="ollama-cloud",
            name="glm-5.2:cloud",
            role="backup",
            token_env="T_BASIL",
            backend="agentic",
        )
    ]

    promote, reasons = runner.plan_escalation(actives, backups, ReviewConfig(), concurrent=True)

    assert [f"{m.provider}/{m.name}" for m in promote] == ["ollama-cloud/glm-5.2:cloud"]
    assert reasons == []


def test_review_keeps_backups_in_reserve_when_healthy(monkeypatch, tmp_path):
    cfg = _one_active_one_backup(tmp_path)
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    backend = _CountingBackend()
    _wire_review(monkeypatch, cfg, backend)
    monkeypatch.setattr(
        runner, "_rh_states", lambda repo: [{"reviewer": "coderabbit", "state": "done"}]
    )
    monkeypatch.setattr(runner, "_observe_reviewer_health", lambda pr, token, api_url: None)

    result = runner.review("https://github.com/o/r/pull/7", str(cfg))

    assert result.returncode == 0
    assert backend.models == ["anthropic/a"]


def test_observe_reviewer_health_persists_rows_locally(monkeypatch):
    from sidecar import reviewer_health as rh
    from sidecar.backends.base import PRRef

    monkeypatch.delenv("FUKO_URL", raising=False)
    monkeypatch.setattr(runner, "fetch_pr_head", lambda pr, token, api: "headsha")
    monkeypatch.setattr(runner, "fetch_issue_comments", lambda pr, token, api: [])
    monkeypatch.setattr(runner, "fetch_reviews", lambda pr, token, api: [])
    monkeypatch.setattr(runner, "fetch_check_runs", lambda pr, ref, token, api: [])
    seen = []
    monkeypatch.setattr(
        rh, "observe", lambda repo, rev, state, pr, detail: seen.append((repo, rev, state, pr))
    )

    _REAL_OBSERVE(PRRef("o/r", 7, "u"), "tok", "https://api.github.com")

    assert ("o/r", "coderabbit", "none", 7) in seen
    assert ("o/r", "copilot", "none", 7) in seen


def test_observe_reviewer_health_swallows_fetch_errors(monkeypatch, capsys):
    from sidecar.backends.base import PRRef

    def boom(*a, **k):
        raise RuntimeError("github down")

    monkeypatch.setattr(runner, "fetch_pr_head", boom)
    _REAL_OBSERVE(PRRef("o/r", 7, "u"), "tok", "https://api.github.com")
    assert "observation skipped" in capsys.readouterr().err
