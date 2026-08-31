-- Make a model-driven closure reversible (#177, epic #160).
--
-- 009 left `fixed` and `rejected` terminal: `transition` matches
-- `status = 'open'`, so once a verdict closed a row nothing could reach it
-- again. That is the one irreversible write the stateful-review design makes
-- from model text produced while reading a contributor-controlled checkout, and
-- a wrong closure -- adversarial or merely mistaken -- was permanent.
--
-- The reversal is driven by a later round's own PUBLISHED finding, not by the
-- fenced verdict channel, and it needs exactly one new column: the count of how
-- many times a closed row was re-raised. The count is the audit trail #177 asks
-- for -- a row at `reopened > 0` is one a round declared settled and a later
-- round contradicted, which is the anomaly an operator wants to see. WHO closed
-- it and why is already in `status`/`status_reason`, which the reopen carries
-- forward into its own reason line rather than discarding.
--
-- No new index: the settled read filters on (repo, pr, seat, status), which is
-- exactly what `review_findings_pr_seat_status_idx` from 009 already covers,
-- and it is bounded by a LIMIT well under one PR's ledger.
--
-- Idempotent like every migration here (re-applied on each pool creation, see
-- `db._migration_sql`).

ALTER TABLE review_findings
    ADD COLUMN IF NOT EXISTS reopened INTEGER NOT NULL DEFAULT 0;
