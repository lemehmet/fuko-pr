-- Admit the 'digest' learning source (#158).
--
-- A digest is a mechanically derived structural index of one large file, keyed
-- on the hash of the blob it describes and scoped by file_globs to that file's
-- own path. It rides the existing knowledge channel rather than a table of its
-- own, so the only schema change it needs is the source vocabulary.
--
-- Kept as plain statements rather than a DO block for the same reason as
-- migration 005: the migration runner splits files on ';', which would shred a
-- block body. Idempotent by construction -- DROP ... IF EXISTS followed by ADD
-- leaves the same constraint on every run.
--
-- This file is what upgrades an already-deployed database. It is NOT what makes
-- 'digest' storable at runtime: sidecar/db.py keeps no applied-migrations table
-- and re-runs every file on every pool creation, so 001 and 005 re-assert the
-- constraint before this one gets a turn. Both were widened to include 'digest'
-- for that reason. Anyone adding a fifth source must widen all three -- adding
-- only a migration 012 would leave the sidecar unable to start as soon as one
-- row used the new source.

ALTER TABLE learnings DROP CONSTRAINT IF EXISTS learnings_source_check;

ALTER TABLE learnings ADD CONSTRAINT learnings_source_check
    CHECK (source IN ('remember', 'review_thread', 'docs', 'digest'));
