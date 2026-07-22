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
