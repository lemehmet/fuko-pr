-- One queryable row per stored session transcript, plus the reference that
-- ties a review run to the blob it produced (#239, epic #236).
--
-- `review_runs` answers how much a run spent -- tokens, cache behaviour, cost,
-- turns (008) -- and nothing about WHAT it spent it on. These figures close
-- that gap: per-tool call counts, the bytes tool results returned, and how many
-- files the run read more than once. The hand-count that motivated the epic
-- (182 reads across 24 runs) becomes something the database can produce.
--
-- WHY A SEPARATE TABLE. The alternative is five more columns on `review_runs`,
-- which #239 rules out and which would be wrong in both directions: these
-- figures describe the TRANSCRIPT, not the run's outcome, and a run without a
-- transcript would then carry five more NULLs on the row every summary reads.
--
-- WHY `key` IS THE PRIMARY KEY, and not `review_runs.id`. `run_metrics.record()`
-- inserts the run row AFTER the run finishes and never asks for `RETURNING id`,
-- so that id does not exist while the transcript is being written. The key is
-- minted at run start instead (`reviewer.transcript.mint_key`) and is the
-- transcript's own identity in every layer: it names the blob in the store, it
-- is the path component of `POST /transcripts/{key}`, and it is this row.
--
-- WHY NO FOREIGN KEY between `review_runs.transcript_key` and this table. The
-- two rows are written in separate transactions, index row first, and the
-- reference is only written when that insert already succeeded -- so the
-- invariant a foreign key would state (`transcript_key IS NOT NULL` implies a
-- row here) is held by write order. Declaring it as a constraint would add the
-- one failure mode the ordering exists to avoid: a transcript-side problem
-- rejecting the `review_runs` insert, costing the duration, outcome, attempts
-- and token counts beside it. Same blast-radius argument as
-- `run_metrics._storable_cost`. The guarantee is scoped to a failure the
-- STATEMENT earned; a connection-level one latches `db_best_effort` for the
-- whole process, and then nothing lands either way (see `index_transcript`).
--
-- The columns here are NOT NULL because the row's EXISTENCE is the measurement:
-- it is written only for a run whose transcript was captured, from that run's
-- own feed. That is why nothing is backfilled -- for the reason 008 gives about
-- the token columns, applied to tools. A backfilled 0 would read as "this run
-- used no tools", which is never true of an agentic review; a pr-agent run and
-- every run predating this migration has no transcript at all, and says so by
-- having no row and a NULL reference.
--
-- A transcript whose feed was cut short (a `tool_timeout` kill, a sidecar
-- death) still gets a row, marked `complete = false`: the figures then describe
-- the prefix that was stored, which is a real measurement of a real run, and
-- dropping it would silently bias the corpus towards runs that finished.
--
-- Every statement is idempotent and every index explicitly named, because
-- migrations are re-applied on each pool creation (see `db._migration_sql`).
CREATE TABLE IF NOT EXISTS review_transcripts (
    key                 TEXT PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Whether the feed reached its terminal `result` event.
    complete            BOOLEAN NOT NULL,
    -- Call counts keyed by tool name, e.g. {"Read": 182, "Grep": 9}. JSONB
    -- rather than a child table: the tool vocabulary is the harness's and
    -- grows without a migration, and every question asked of it so far is one
    -- key at a time (`tool_calls->>'Read'`).
    tool_calls          JSONB NOT NULL,
    -- Total UTF-8 bytes of tool-result content the run was fed. BIGINT because
    -- a long agentic review's tool results are megabytes.
    tool_result_bytes   BIGINT NOT NULL,
    -- Distinct files READ more than once in this run -- one file read three
    -- times counts once. The re-read figure #159 needs to decide stateful
    -- versus stateless review from receipts.
    repeated_read_files INTEGER NOT NULL
);

-- The reader's ordering (#240, #241): newest transcripts first.
CREATE INDEX IF NOT EXISTS review_transcripts_created_at_idx
    ON review_transcripts (created_at DESC);

-- NULLABLE and deliberately NOT backfilled: a placeholder here would read as a
-- transcript that had gone missing rather than one that was never captured.
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS transcript_key TEXT;

-- PARTIAL, because the overwhelming majority of rows are NULL (every pr-agent
-- run, every run before this landed) and the only query that uses this column
-- joins a transcript back to the run that produced it.
CREATE INDEX IF NOT EXISTS review_runs_transcript_key_idx
    ON review_runs (transcript_key) WHERE transcript_key IS NOT NULL;
