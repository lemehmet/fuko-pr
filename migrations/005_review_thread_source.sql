-- Rename the swept-thread learning source: resolved_thread -> review_thread (#83).
--
-- Selection never depended on a thread being resolved: select_learning() reads
-- the last trusted decline and ignores isResolved entirely, so the old name
-- described a filter the sidecar does not apply. The source vocabulary is pinned
-- by a CHECK constraint, so it has to move with the name.
--
-- Kept as plain statements rather than a DO block: the migration runner splits
-- files on ';', which would shred a block body. Idempotent by construction --
-- DROP ... IF EXISTS followed by ADD leaves the same constraint on every run.

ALTER TABLE learnings DROP CONSTRAINT IF EXISTS learnings_source_check;

UPDATE learnings SET source = 'review_thread' WHERE source = 'resolved_thread';

ALTER TABLE learnings ADD CONSTRAINT learnings_source_check
    CHECK (source IN ('remember', 'review_thread', 'docs'));
