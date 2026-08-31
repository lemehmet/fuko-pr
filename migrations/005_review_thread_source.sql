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
--
-- 'digest' (#158) is listed here even though 011 is what adds it, and this is
-- not redundancy. There is no applied-migrations table: sidecar/db.py re-runs
-- every file, in filename order, on every pool creation. So this ADD runs
-- before 011 on each startup, and ADD CONSTRAINT validates existing rows -- one
-- stored digest would make it fail, aborting migration before 011 and leaving
-- the pool unopenable on every restart thereafter. THE INVARIANT: every
-- historical re-add of this constraint must list the CURRENT vocabulary. The
-- same reasoning is why 001's inline CHECK already said 'review_thread'.

ALTER TABLE learnings DROP CONSTRAINT IF EXISTS learnings_source_check;

UPDATE learnings SET source = 'review_thread' WHERE source = 'resolved_thread';

ALTER TABLE learnings ADD CONSTRAINT learnings_source_check
    CHECK (source IN ('remember', 'review_thread', 'docs', 'digest'));
