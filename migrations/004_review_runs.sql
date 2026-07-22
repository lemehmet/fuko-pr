-- Per-branch review-run metrics: one row per model branch per review round.
--
-- Written by the runner at the end of each branch (best-effort) so model
-- performance is queryable per provider+model+slot: duration, failover
-- attempts, outcome, and findings count. Token/cost columns are nullable —
-- PR-Agent does not currently expose usage or generation ids to the runner,
-- so they stay NULL until a capture path exists (OpenRouter /generation
-- lookup or PR-Agent verbose-output parsing).
CREATE TABLE IF NOT EXISTS review_runs (
    id                BIGSERIAL PRIMARY KEY,
    repo              TEXT NOT NULL,
    pr                INTEGER NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    slot              TEXT,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_s        REAL NOT NULL,
    attempts          INTEGER NOT NULL DEFAULT 1,
    outcome           TEXT NOT NULL,
    findings          INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          NUMERIC(10, 4),
    detail            TEXT
);

CREATE INDEX IF NOT EXISTS review_runs_repo_started_idx
    ON review_runs (repo, started_at);

CREATE INDEX IF NOT EXISTS review_runs_provider_model_idx
    ON review_runs (provider, model);
