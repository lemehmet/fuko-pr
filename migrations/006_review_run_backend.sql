-- Attribute each review_runs row to the review DRIVER (backend) that produced it
-- (#99). Two harnesses are indistinguishable receipts-only otherwise, and scoring
-- is receipts-only by rule. Every historical run predates per-backend selection and
-- was pr-agent, so the NOT NULL DEFAULT backfills existing rows to 'pr-agent' in one
-- statement (Postgres applies a non-volatile column default to existing rows).
-- Idempotent (ADD COLUMN IF NOT EXISTS is a no-op on re-apply), matching every
-- other migration -- they are re-run on each pool creation.
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS backend TEXT NOT NULL DEFAULT 'pr-agent';

CREATE INDEX IF NOT EXISTS review_runs_backend_idx ON review_runs (backend);
