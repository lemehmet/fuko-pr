-- What a review run actually spent: tokens, cache behaviour, and turns (#152).
--
-- Until now `review_runs` could answer "how long" and "how many findings" but
-- not "how much", so seats were priced and models promoted on duration alone.
-- The agentic backend's harness already parsed the CLI's terminal `result`
-- event and discarded its `usage` block; these columns are where it lands.
--
-- NULLABLE, and deliberately NOT backfilled to 0 -- unlike the `backend` (006)
-- and `endpoint` (007) columns, which had one honest historical value each.
-- Here there is none: every pre-existing row, and every pr-agent row ever, has
-- an unknown cost, and a 0 would read as "this review was free" in the
-- summaries -- the one answer that is always wrong.
--
-- `input_tokens` is the Anthropic-shaped count: FRESH input only, with cache
-- reads and writes reported alongside rather than inside it. That is what makes
-- the pair below answer the question this migration exists for -- whether a
-- gateway honours prompt caching at all, which moves the bill for one review by
-- roughly 25x. It is therefore NOT the same quantity as the `prompt_tokens` /
-- `completion_tokens` columns reserved by 004, which are OpenAI-shaped
-- (cache-inclusive) and still reserved for a future PR-Agent capture path;
-- reusing them would silently conflate the two definitions in one column.
--
-- `cost_usd` already exists from 004 (NUMERIC(10,4)) and is now written for the
-- first time; it is not re-added here.
--
-- Idempotent (ADD COLUMN IF NOT EXISTS is a no-op on re-apply), matching every
-- other migration -- they are re-run on each pool creation.
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS input_tokens BIGINT;
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS output_tokens BIGINT;
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS cache_read_tokens BIGINT;
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS cache_write_tokens BIGINT;
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS turns INTEGER;
