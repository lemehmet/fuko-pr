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
from .dedup import partition
from .embed import get_embedder
from .fukoconfig import KnowledgeConfig
from .ingest import _UPDATABLE, _parse_dt, checked_expires
from .models import UNSET, DuplicateLearningError, IngestItem, check_source, check_text
from .objectstore import PreconditionFailed, make_object_store
from .retrieve import _build_query, fold_repo_counts, like_escape

_MAX_RETRIES = 5

_VAR_BATCH = 500

_ROW_COLUMNS = "lid, repo, text, source, source_url, file_globs, topic, expires_at"


def _row_to_dict(row: tuple) -> dict:
    """Shape one ``_ROW_COLUMNS`` row like the Postgres store's rows.

    ``created_at`` is always ``None``: this schema orders by insertion rowid and
    never recorded a timestamp, and inventing one here would make the two stores
    disagree about what they know.
    """
    return {
        "id": row[0],
        "repo": row[1],
        "text": row[2],
        "source": row[3],
        "source_url": row[4],
        "file_globs": json.loads(row[5]) if row[5] else [],
        "topic": row[6],
        "created_at": None,
        "expires_at": row[7],
    }


def _unicode_lower(value):
    """Case-fold with Python's Unicode rules, overriding sqlite's ASCII-only ``lower``.

    Both sqlite's ``LIKE`` and its built-in ``lower()`` fold ASCII only, so
    ``'ÄPFEL'`` never matches a search for ``'äpfel'`` -- while Postgres ``ILIKE``
    does match it. Registering this keeps the two stores' search agreeing on text
    that is not plain ASCII, without needing an ICU-enabled sqlite build.
    """
    return value.lower() if isinstance(value, str) else value


def _pack(vec: list[float]) -> bytes:
    """Pack a vector as little-endian float32 (sqlite-vec's portable wire format)."""
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
        conn.create_function("lower", 1, _unicode_lower, deterministic=True)
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
        for (lid, repo, _text), emb in zip(rows, embeddings, strict=True):
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

    def _existing_keys(self, repo: str, items: list[IngestItem]) -> set[tuple[str, str]]:
        candidates = {(it.text, it.source) for it in items}
        texts = list({it.text for it in items})
        if not texts:
            return set()

        def fn(conn: sqlite3.Connection) -> list[tuple[str, str]]:
            rows: list[tuple[str, str]] = []
            for i in range(0, len(texts), _VAR_BATCH):
                batch = texts[i : i + _VAR_BATCH]
                placeholders = ",".join("?" * len(batch))
                rows.extend(
                    conn.execute(
                        f"SELECT text, source FROM learnings "
                        f"WHERE repo = ? AND text IN ({placeholders})",
                        (repo, *batch),
                    ).fetchall()
                )
            return rows

        return {(text, source) for text, source in self._read(fn) if (text, source) in candidates}

    def ingest(
        self, repo: str, items: list[IngestItem], *, max_new: int | None = None
    ) -> tuple[int, int]:
        """Embed and insert learnings, skipping exact duplicates.

        Duplicates of the ``(repo, text, source)`` key are filtered out *before*
        embedding, so re-sweeping an already-ingested backlog costs no embed
        calls; only genuinely new learnings reach the (potentially slow)
        embedder. The ``INSERT OR IGNORE`` remains as a backstop for races.

        Args:
            repo: Repository the learnings belong to.
            items: Candidate learnings.
            max_new: Cap on how many new learnings are embedded in this call.
                Items past the cap are neither inserted nor skipped, so a caller
                can detect them as ``len(items) - inserted - skipped`` and re-send
                the same batch to drain the rest. ``None`` embeds everything new.

        Returns:
            A ``(inserted, skipped)`` tuple.
        """
        if not items:
            return 0, 0
        existing = self._existing_keys(repo, items)
        to_embed, skipped, _deferred = partition(items, existing, max_new)
        if not to_embed:
            return 0, skipped
        embeddings = get_embedder().embed([it.text for it in to_embed])

        def fn(conn: sqlite3.Connection) -> tuple[int, int]:
            inserted = inner_skipped = 0
            for item, emb in zip(to_embed, embeddings, strict=True):
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
                    inner_skipped += 1
                    continue
                vec_cur = conn.execute(
                    "INSERT INTO vec_learnings(repo, lid, embedding) VALUES (?, ?, ?)",
                    (repo, lid, _pack(emb)),
                )
                conn.execute(
                    "UPDATE learnings SET vec_rowid = ? WHERE lid = ?", (vec_cur.lastrowid, lid)
                )
                inserted += 1
            return inserted, inner_skipped

        inserted, inner_skipped = self._mutate(fn)
        return inserted, skipped + inner_skipped

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

    def list_learnings(
        self,
        repo: str | None = None,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
        q: str | None = None,
        include_expired: bool = False,
    ) -> tuple[list[dict], int]:
        """Return a page of learnings (newest insert first) plus the total match count."""
        where: list[str] = []
        params: list = []
        if not include_expired:
            where.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.now(timezone.utc).isoformat())
        if repo:
            where.append("repo = ?")
            params.append(repo)
        if source:
            where.append("source = ?")
            params.append(source)
        if q:
            where.append(
                r"(lower(text) LIKE ? ESCAPE '\' OR lower(coalesce(topic, '')) LIKE ? ESCAPE '\')"
            )
            pattern = f"%{like_escape(q.lower())}%"
            params.extend([pattern, pattern])
        clause = " AND ".join(where) if where else "1"
        page_sql = (
            f"SELECT {_ROW_COLUMNS} "
            f"FROM learnings WHERE {clause} ORDER BY rowid DESC LIMIT ? OFFSET ?"
        )

        def fn(conn: sqlite3.Connection) -> tuple[list[dict], int]:
            total = conn.execute(
                f"SELECT count(*) FROM learnings WHERE {clause}", params
            ).fetchone()[0]
            rows = conn.execute(page_sql, [*params, limit, offset]).fetchall()
            return [_row_to_dict(row) for row in rows], int(total)

        return self._read(fn)

    def get_learning(self, repo: str, id: str) -> dict | None:
        """Return one learning by id within ``repo`` (expired included), else ``None``."""

        def fn(conn: sqlite3.Connection) -> dict | None:
            row = conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM learnings WHERE repo = ? AND lid = ?", (repo, id)
            ).fetchone()
            return _row_to_dict(row) if row else None

        return self._read(fn)

    def update_learning(
        self,
        repo: str,
        id: str,
        *,
        text: str = UNSET,
        source: str = UNSET,
        source_url: str | None = UNSET,
        file_globs: list[str] = UNSET,
        topic: str | None = UNSET,
        expires_at: str | None = UNSET,
    ) -> dict | None:
        """Apply the supplied fields to one learning; re-embeds only on a text change.

        The re-embed decision reads the stored text inside the mutation's own
        connection, not from an earlier read: ``_mutate`` re-runs its callback
        after losing an optimistic-concurrency race, and a decision made against
        the pre-race snapshot could leave the row holding this write's text with
        the winner's embedding. The computed vector is memoized across those
        retries, so a lost race never pays the (slow) embedder twice.
        """
        supplied = {
            name: value
            for name, value in zip(
                _UPDATABLE,
                (text, source, source_url, file_globs, topic, expires_at),
                strict=True,
            )
            if value is not UNSET
        }
        if "source" in supplied:
            check_source(supplied["source"])
        if "text" in supplied:
            check_text(supplied["text"])
        if "expires_at" in supplied:
            parsed = checked_expires(supplied["expires_at"])
            supplied["expires_at"] = parsed.isoformat() if parsed else None
        if not supplied:
            return self.get_learning(repo, id)

        assignments: list[str] = []
        params: list = []
        for name, value in supplied.items():
            assignments.append(f"{name} = ?")
            if name == "file_globs":
                params.append(json.dumps(value or []))
            else:
                params.append(value)

        memo: list[bytes] = []

        def embedding_for(new_text: str) -> bytes:
            if not memo:
                memo.append(_pack(get_embedder().embed_one(new_text)))
            return memo[0]

        def fn(conn: sqlite3.Connection) -> dict | None:
            row = conn.execute(
                "SELECT vec_rowid, text FROM learnings WHERE repo = ? AND lid = ?", (repo, id)
            ).fetchone()
            if row is None:
                return None
            embedding = (
                embedding_for(supplied["text"]) if supplied.get("text", row[1]) != row[1] else None
            )
            try:
                conn.execute(
                    f"UPDATE learnings SET {', '.join(assignments)} WHERE repo = ? AND lid = ?",
                    (*params, repo, id),
                )
            except sqlite3.IntegrityError as e:
                raise DuplicateLearningError(
                    "another learning in this repo already has that (text, source)"
                ) from e
            if embedding is not None and row[0] is not None:
                conn.execute(
                    "UPDATE vec_learnings SET embedding = ? WHERE rowid = ?", (embedding, row[0])
                )
            updated = conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM learnings WHERE repo = ? AND lid = ?", (repo, id)
            ).fetchone()
            return _row_to_dict(updated) if updated else None

        return self._mutate(fn)

    def repos(self) -> list[dict]:
        """Return the per-repository footprint of live learnings, repo-sorted."""
        now = datetime.now(timezone.utc).isoformat()

        def fn(conn: sqlite3.Connection) -> list[dict]:
            return fold_repo_counts(
                conn.execute(
                    "SELECT repo, source, count(*) FROM learnings "
                    "WHERE expires_at IS NULL OR expires_at > ? GROUP BY repo, source",
                    (now,),
                ).fetchall()
            )

        return self._read(fn)
