-- Records which embedding model produced the vectors in `learnings`.
--
-- The dimension alone is not enough to tell two embedding spaces apart. A
-- same-dimension model swap -- bge-m3 to Qwen3-Embedding-0.6B, both 1024 --
-- slips past the dimension check in db.py, and the store then holds vectors
-- from two different spaces at once. That does not fail: it retrieves, badly,
-- which is the worst failure mode a knowledge base has.
--
-- Deliberately a generic key/value table rather than an `embed_model` column:
-- it mirrors the sqlite-vec store's existing `meta` table, so the two backends
-- keep the same shape.
CREATE TABLE IF NOT EXISTS meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);
