-- Record WHICH of a provider's credentials answered a review run (#299). A
-- preset may be backed by numbered sibling keys (KEY, KEY_2, ...), each its own
-- quota bucket, drained in order; provider+model cannot say which bucket paid.
-- The 1-based INDEX only -- never the key, a hash, or a prefix of it.
-- Every historical row predates siblings and ran on the only key there was, so
-- the NOT NULL DEFAULT backfills them to 1, the value an omitting caller writes.
-- Idempotent (ADD COLUMN IF NOT EXISTS is a no-op on re-apply), matching every
-- other migration -- they are re-run on each pool creation.
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS credential INTEGER NOT NULL DEFAULT 1;
