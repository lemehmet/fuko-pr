CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS learnings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo        TEXT NOT NULL,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('remember', 'resolved_thread', 'docs')),
    source_url  TEXT,
    file_globs  TEXT[] NOT NULL DEFAULT '{}',
    topic       TEXT,
    embedding   vector(1024) NOT NULL,
    origin_user TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ,
    UNIQUE (repo, text, source)
);

CREATE INDEX IF NOT EXISTS learnings_repo_idx ON learnings (repo);
CREATE INDEX IF NOT EXISTS learnings_embedding_idx
    ON learnings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS learnings_file_globs_idx
    ON learnings USING gin (file_globs);
