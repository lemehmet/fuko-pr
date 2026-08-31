"""Shared test fixtures."""

import pytest

from sidecar import runner


@pytest.fixture(autouse=True)
def _quiet_reviewer_health(monkeypatch):
    """Neutralize the reviewer-health hooks for every runner test by default.

    ``review()`` reads persisted reviewer health at start (escalation) and
    observes the PR's live reviewer states at the end — both would otherwise
    reach for the network/database in unit tests. Tests that exercise these
    paths re-patch the same attributes explicitly.
    """
    monkeypatch.setattr(runner, "_rh_states", lambda repo: [])
    monkeypatch.setattr(runner, "_observe_reviewer_health", lambda pr, token, api_url: None)
    monkeypatch.setattr(runner, "_record_run", lambda pr, model, **kw: None)


@pytest.fixture(autouse=True)
def _no_ambient_sidecar(monkeypatch):
    """Unset the sidecar env for every test, so no suite can reach a real one.

    Since #171 the ledger picks its transport from ``FUKO_URL`` at CALL time
    (:func:`sidecar.review_state_client._remote`), which makes an operator's own
    exported environment part of the test fixture: on a developer laptop with
    ``FUKO_URL`` set, the ledger and agentic suites would post their fake rounds
    to the live homelab sidecar -- writing junk into a real ledger, and routing
    past the in-memory fakes that are supposed to be answering.

    Autouse and repo-wide rather than per-suite because the hazard is ambient by
    definition: a test that never mentions the ledger can still reach it through
    a call chain, and the failure is silent on the machine that has no
    ``FUKO_URL`` set (CI) and only fires on the one that does.

    Tests that exercise the HTTP branch set these back themselves.
    """
    monkeypatch.delenv("FUKO_URL", raising=False)
    monkeypatch.delenv("FUKO_TOKEN", raising=False)
