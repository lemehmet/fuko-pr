-- Per-PR, per-seat review state: the open-findings ledger and the coverage
-- ledger of the stateful-review epic (#155, epic #160).
--
-- Numbered 009 rather than the 008 the issue proposed: #152's token/cost
-- columns took 008 first, and the issue says whichever lands second renumbers.
--
-- WHY NOT `learnings`. That table is repo-scoped, semantic and vector-retrieved,
-- built to dedup durable conventions on ingest (UNIQUE (repo, text, source)).
-- This state is scoped to (repo, pr, seat), dies with the pull request, is
-- looked up by exact key rather than by embedding similarity, and MUTATES
-- (open -> fixed). Storing it in `learnings` would pay an embed round-trip to
-- read back one's own last round and would mix PR-lifetime rows into the
-- knowledge base -- the population mistake that left the stores full of noise.
--
-- `seat` is the seat LABEL (`henry`, `sybil`), never `provider/model`: seats are
-- model-agnostic by design and swapping the model behind a seat must not orphan
-- its ledger. Same reasoning as `run_metrics.slot_summary()` grouping by slot.
--
-- Every statement is idempotent: migrations are re-applied on each pool
-- creation (see `db._migration_sql`), so the indexes are explicitly NAMED with
-- IF NOT EXISTS -- an unnamed `CREATE INDEX ON ...` would mint a new
-- auto-named duplicate index on every startup.

CREATE TABLE IF NOT EXISTS review_findings (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo          TEXT NOT NULL,
    pr            INTEGER NOT NULL,
    seat          TEXT NOT NULL,
    round         INTEGER NOT NULL,
    head_sha      TEXT NOT NULL,
    file          TEXT NOT NULL,
    line          INTEGER,
    severity      TEXT NOT NULL,
    category      TEXT NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL,
    evidence      TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'open'
                  CHECK (status IN ('open', 'fixed', 'rejected', 'stale')),
    status_reason TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The one read the reviewer makes every round: the open ledger for this seat.
CREATE INDEX IF NOT EXISTS review_findings_pr_seat_status_idx
    ON review_findings (repo, pr, seat, status);

-- `severity` and `category` carry NO check constraint, unlike `status`. Those
-- two are the model's own words about a finding, so pinning them here would let
-- a vocabulary drift fail the write that records a real finding; `status` is
-- fuko's own state machine and is pinned. The store gates it in Python too --
-- this constraint is the backstop, not the gate.

CREATE TABLE IF NOT EXISTS review_coverage (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo       TEXT NOT NULL,
    pr         INTEGER NOT NULL,
    seat       TEXT NOT NULL,
    round      INTEGER NOT NULL,
    head_sha   TEXT NOT NULL,
    file       TEXT NOT NULL,
    region     TEXT NOT NULL DEFAULT '',
    checked    TEXT NOT NULL,
    conclusion TEXT NOT NULL,
    evidence   TEXT NOT NULL DEFAULT '',
    expired_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Partial on purpose: expired rows are kept as history but are never read back
-- into a prompt, so the index only covers the live ledger.
CREATE INDEX IF NOT EXISTS review_coverage_live_idx
    ON review_coverage (repo, pr, seat) WHERE expired_at IS NULL;

-- `expired_at` is the asymmetry the epic insists on: a coverage claim is an
-- ASSURANCE and dies when the delta touches its file, while a finding is a
-- CLAIM and survives -- it stays open until a round settles it with a reason.
