"""sqlite-vec knowledge store, synced through object storage.

A single embedded sqlite file (sqlite-vec ``vec0`` for vectors) is the whole
knowledge base. Each operation downloads the file, runs locally, and -- for
writes -- uploads it back under optimistic concurrency, retrying if it loses a
race. This is the server-free deployment: no Postgres, no always-on sidecar.

Note: retrieval ranks the semantic top ``candidate_k`` and then applies file-glob
scoping, like the Postgres store. For knowledge bases larger than ``candidate_k``,
a file-scoped learning outside that semantic window is not separately boosted (the
Postgres store does a second scoped pass); at typical repo scale the window covers
the whole base, so the two agree.
"""

from __future__ import annotations

import fnmatch
import json
import sqlite3
import struct
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .embed import get_embedder
from .fukoconfig import KnowledgeConfig
from .ingest import _parse_dt
from .models import IngestItem
from .objectstore import PreconditionFailed, make_object_store
from .retrieve import _build_query

_MAX_RETRIES = 5


def _pack(vec: list[float]) -> bytes:
    # Little-endian float32 (sqlite-vec's format), explicit so a db synced across
    # architectures via object storage is read back consistently.
    return struct.pack(f"<{len(vec)}f", *vec)


def _norm_expires(value: str | None) -> str | None:
    """Normalize ``expires_at`` to a UTC ISO-8601 string (NULL on parse failure).

    Matches the Postgres store, so the lexicographic ``expires_at > now`` filter
    is correct regardless of what a client supplied.
    """
    dt = _parse_dt(value)
    return dt.isoformat() if dt else None


class SqliteVecStore:
    """Store backed by a sqlite-vec file in object storage."""

    def __init__(self, knowledge: KnowledgeConfig) -> None:
        """Build the object-store sync layer from ``knowledge.object_store``."""
        if knowledge.object_store is None:
            raise ValueError("sqlite-vec store requires a [knowledge.object_store] section")
        self._obj = make_object_store(knowledge.object_store)
        self._dim: int | None = None

    def _ensure_dim(self) -> int:
        if self._dim is None:
            self._dim = get_embedder().probe_dim()
        return self._dim

    def _vec_ddl(self, dim: int) -> str:
        return (
            "USING vec0(repo TEXT partition key, lid TEXT, "
            f"embedding float[{dim}] distance_metric=cosine)"
        )

    def _open(self, path: str, dim: int) -> tuple[sqlite3.Connection, bool]:
        """Open the db (loading sqlite-vec), ensure schema, and migrate on a dim change.

        Returns ``(conn, migrated)``; ``migrated`` is True when the embedding model's
        dimension changed and every learning was re-embedded + the vector table rebuilt.
        """
        import sqlite_vec

        conn = sqlite3.connect(path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS learnings("
            "lid TEXT PRIMARY KEY, vec_rowid INTEGER, repo TEXT, text TEXT, source TEXT, "
            "source_url TEXT, file_globs TEXT, topic TEXT, origin_user TEXT, expires_at TEXT, "
            "UNIQUE(repo, text, source))"
        )
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_learnings {self._vec_ddl(dim)}")

        row = conn.execute("SELECT value FROM meta WHERE key = 'embed_dim'").fetchone()
        stored = int(row[0]) if row else None
        migrated = False
        if stored is None:
            conn.execute("INSERT INTO meta(key, value) VALUES ('embed_dim', ?)", (str(dim),))
        elif stored != dim:
            self._migrate_dim(conn, dim)
            conn.execute("UPDATE meta SET value = ? WHERE key = 'embed_dim'", (str(dim),))
            migrated = True
        conn.commit()
        return conn, migrated

    def _migrate_dim(self, conn: sqlite3.Connection, dim: int) -> None:
        """Re-embed every learning with the current model and rebuild the vector table."""
        rows = conn.execute("SELECT lid, repo, text FROM learnings").fetchall()
        conn.execute("DROP TABLE vec_learnings")
        conn.execute(f"CREATE VIRTUAL TABLE vec_learnings {self._vec_ddl(dim)}")
        if not rows:
            return
        embeddings = get_embedder().embed([text for _, _, text in rows])
        for (lid, repo, _text), emb in zip(rows, embeddings):
            cur = conn.execute(
                "INSERT INTO vec_learnings(repo, lid, embedding) VALUES (?, ?, ?)",
                (repo, lid, _pack(emb)),
            )
            conn.execute("UPDATE learnings SET vec_rowid = ? WHERE lid = ?", (cur.lastrowid, lid))

    def _read(self, fn):
        data, token = self._obj.load()
        dim = self._ensure_dim()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "kb.db"
            if data is not None:
                path.write_bytes(data)
            conn, migrated = self._open(str(path), dim)
            try:
                result = fn(conn)
            finally:
                conn.close()
            if migrated:
                # persist the one-time re-embed so later reads don't repeat it
                try:
                    self._obj.save(path.read_bytes(), token)
                except PreconditionFailed:
                    pass
            return result

    def _mutate(self, fn):
        dim = self._ensure_dim()
        for _ in range(_MAX_RETRIES):
            data, token = self._obj.load()
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "kb.db"
                if data is not None:
                    path.write_bytes(data)
                conn, _migrated = self._open(str(path), dim)
                try:
                    result = fn(conn)
                    conn.commit()
                finally:
                    conn.close()
                new_bytes = path.read_bytes()
            try:
                self._obj.save(new_bytes, token)
                return result
            except PreconditionFailed:
                continue
        raise PreconditionFailed("knowledge store write lost too many races")

    def ingest(self, repo: str, items: list[IngestItem]) -> tuple[int, int]:
        """Embed and insert learnings, skipping exact duplicates."""
        if not items:
            return 0, 0
        embeddings = get_embedder().embed([it.text for it in items])

        def fn(conn: sqlite3.Connection) -> tuple[int, int]:
            inserted = skipped = 0
            for item, emb in zip(items, embeddings):
                lid = uuid.uuid4().hex
                cur = conn.execute(
                    "INSERT OR IGNORE INTO learnings"
                    "(lid, repo, text, source, source_url, file_globs, topic, "
                    "origin_user, expires_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        lid,
                        repo,
                        item.text,
                        item.source,
                        item.source_url,
                        json.dumps(item.file_globs),
                        item.topic,
                        item.origin_user,
                        _norm_expires(item.expires_at),
                    ),
                )
                if cur.rowcount != 1:
                    skipped += 1
                    continue
                vec_cur = conn.execute(
                    "INSERT INTO vec_learnings(repo, lid, embedding) VALUES (?, ?, ?)",
                    (repo, lid, _pack(emb)),
                )
                conn.execute(
                    "UPDATE learnings SET vec_rowid = ? WHERE lid = ?", (vec_cur.lastrowid, lid)
                )
                inserted += 1
            return inserted, skipped

        return self._mutate(fn)

    def query(
        self,
        repo: str,
        files: list[str],
        pr_body: str | None = None,
        query_text: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Return the learnings most relevant to the given PR context."""
        q = _build_query(files, pr_body, query_text)
        if not q:
            return []
        vec = _pack(get_embedder().embed_one(q))
        k = top_k or settings.top_k
        cand = settings.candidate_k
        now = datetime.now(timezone.utc).isoformat()

        def fn(conn: sqlite3.Connection) -> list[dict]:
            knn = conn.execute(
                "SELECT lid, distance FROM vec_learnings "
                "WHERE repo = ? AND embedding MATCH ? AND k = ?",
                (repo, vec, cand),
            ).fetchall()
            dist = {lid: d for lid, d in knn}
            if not dist:
                return []
            marks = ",".join("?" * len(dist))
            rows = conn.execute(
                f"SELECT lid, text, source, source_url, file_globs, topic FROM learnings "
                f"WHERE lid IN ({marks}) AND (expires_at IS NULL OR expires_at > ?)",
                (*dist.keys(), now),
            ).fetchall()
            results: list[dict] = []
            for lid, text, source, source_url, file_globs, topic in rows:
                globs = json.loads(file_globs) if file_globs else []
                if globs and not any(fnmatch.fnmatch(f, p) for f in files for p in globs):
                    continue
                results.append(
                    {
                        "id": lid,
                        "text": text,
                        "source": source,
                        "source_url": source_url,
                        "file_globs": list(globs),
                        "topic": topic,
                        "score": 1.0 - float(dist[lid]),
                    }
                )
            results.sort(key=lambda r: r["score"], reverse=True)
            return results[:k]

        return self._read(fn)

    def forget(
        self,
        repo: str,
        *,
        id: str | None = None,
        source: str | None = None,
        all: bool = False,
    ) -> int:
        """Delete learnings by id, source, or wholesale; return the count removed."""
        if id:
            where, params = "lid = ? AND repo = ?", (id, repo)
        elif source:
            where, params = "repo = ? AND source = ?", (repo, source)
        elif all:
            where, params = "repo = ?", (repo,)
        else:
            return 0

        def fn(conn: sqlite3.Connection) -> int:
            rows = conn.execute(f"SELECT vec_rowid FROM learnings WHERE {where}", params).fetchall()
            if not rows:
                return 0
            for (vec_rowid,) in rows:
                if vec_rowid is not None:
                    conn.execute("DELETE FROM vec_learnings WHERE rowid = ?", (vec_rowid,))
            conn.execute(f"DELETE FROM learnings WHERE {where}", params)
            return len(rows)

        return self._mutate(fn)
