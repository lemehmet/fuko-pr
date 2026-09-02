# Deployment

fuko-pr's knowledge store is pluggable. Pick the mode that matches where your CI
runs. In every mode the store lives in **your** infrastructure — fuko never hosts
a multitenant knowledge base.

| Mode | Store | Server? | Embeddings | Concurrent writers |
|------|-------|---------|------------|--------------------|
| Homelab / self-host | Postgres + pgvector | sidecar (`fuko serve`) | local Ollama | yes |
| Managed DB | Neon / Supabase pgvector | optional | remote provider | yes |
| Server-free | sqlite-vec file in S3/R2 | none | remote provider | no (single-writer) |

If several instances will **write** to one shared knowledge base — a fleet whose
repos share a store, or overlapping `/remember` and thread-sweep jobs — pick a
Postgres mode. The server-free sqlite-vec store is single-writer (reads, including
a multi-model A/B review, are always safe); see the caveat under
[Server-free](#server-free-sqlite-vec-in-s3r2) below.

## Homelab / self-host (Postgres)

Run pgvector and (optionally) the `fuko serve` sidecar somewhere your runners can
reach privately:

```bash
docker run -d --name fuko-pg -p 5432:5432 \
  -e POSTGRES_USER=fuko -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=fuko \
  pgvector/pgvector:pg16
```

`.fuko.toml`: `[knowledge] store = "postgres"`. Set `FUKO_DATABASE_URL`. The schema
is created on first connection. Embeddings can be a local Ollama model since the
runners and the model are on the same private network.

## Managed DB (Neon / Supabase)

Same as above with a hosted pgvector connection string in `FUKO_DATABASE_URL`. The
sidecar is optional — `fuko review` connects to the DB directly. On hosted runners
without a local Ollama, point `FUKO_EMBED_BASE_URL`/`FUKO_EMBED_MODEL` at a remote
provider (see below).

## Server-free (sqlite-vec in S3/R2)

The whole knowledge base is a single sqlite-vec file in your own bucket. No
Postgres, no always-on sidecar. `fuko review` downloads the file, queries it
in-process, and (on writes) uploads it back with optimistic-concurrency conditional
writes (retrying if it loses a race).

Requires the extra: `pip install "fuko-pr[sqlite]"` (sqlite-vec + boto3).

> **Single-writer by design.** The store is one file guarded by optimistic
> concurrency: a write downloads the file, mutates it locally, and conditionally
> uploads it back, retrying only if it lost a race (5 attempts, no backoff, no
> locking; on exhaustion the write raises and the learning is dropped). Reads are
> always safe — `fuko review` only queries the store, so any number of reviewers
> (including a multi-model **A/B** comparison on one PR) can run at once. The limit
> is concurrent *writers*: the KB is written only out of band — `/remember` and
> `/forget` commands, the resolved-thread sweep, and `ingest-docs` — and if two of
> those overlap on one
> shared file (a fleet whose repos share a bucket, or a sweep landing during a
> `/remember`) the loser exhausts its retries and drops the learning. For a
> shared, multi-repo knowledge base, use the **Postgres** mode below: it is a real
> concurrent store (pooled connections, row-level dedup, a shared provider cooldown
> table) and stays correct when more than one writer commits at the same time.

```toml
[knowledge]
store = "sqlite-vec"

[knowledge.object_store]
backend = "s3"                 # s3 | r2 | file
bucket = "my-fuko-kb"
key = "owner/repo.db"
# endpoint_url = "https://<accountid>.r2.cloudflarestorage.com"   # for R2
creds_env_prefix = "FUKO_S3"   # reads FUKO_S3_ACCESS_KEY_ID, FUKO_S3_SECRET_ACCESS_KEY, FUKO_S3_REGION
```

### Bucket setup (free / cheap)

- **Cloudflare R2** — no egress fees; create a bucket + an API token scoped to it,
  set `endpoint_url` to your account's R2 endpoint, region `auto`.
- **AWS S3** — create a private bucket + an IAM user limited to
  `s3:GetObject`/`s3:PutObject` on `arn:aws:s3:::my-fuko-kb/*`. Conditional writes
  (`If-Match`/`If-None-Match`) are supported, which fuko uses for safe write-back.

Provide creds to the runner as the env vars named by `creds_env_prefix`
(`FUKO_S3_ACCESS_KEY_ID`, `FUKO_S3_SECRET_ACCESS_KEY`, optional `FUKO_S3_REGION`).
The bucket is private; keep it that way.

### Embeddings on hosted runners

A hosted runner has no local Ollama, so pair the server-free store with a **remote
embedding provider** (any OpenAI-compatible `/embeddings` endpoint — e.g. Voyage,
Jina, or BigModel). The embedding endpoint is environment-only — `.fuko.toml` has
no say in it (#216) — so set it on the runner:

```bash
FUKO_EMBED_BASE_URL=https://api.example.com/v1
FUKO_EMBED_MODEL=<the provider's embedding model>
FUKO_EMBED_API_KEY=<the provider's key>
FUKO_EMBED_QUERY_PREFIX=          # empty unless the model is asymmetric
```

`FUKO_EMBED_MODEL` is also the provenance marker for the stored vectors, so
changing it re-embeds the knowledge base. Embeddings are cheap (pennies), and the
file is small, so each run's download/query/upload is fast.

## Session transcripts in object storage (optional)

Two object stores, and they are unrelated. The one above holds **the knowledge
base** as a single mutable sqlite file and only exists in the server-free
deployment. This one holds **agentic session transcripts** as many write-once
blobs, and it is newly relevant to a **Postgres** deployment, which has never
needed object storage before (#238).

It is entirely optional. Leave `FUKO_TRANSCRIPT_STORE_BACKEND` unset and the
sidecar starts, reviews run and no transcript is ever stored *here* — no error,
no change from before. Note what that does **not** mean: a runner with
`FUKO_TRANSCRIPT_DIR` set still writes its own local transcript, exactly as it
did before this feature existed. Unset means shared-store persistence is off,
not that capture is; the runner-resident copy of the reviewed repository is
governed by `FUKO_TRANSCRIPT_DIR` alone. Turn shipping on only if you have also
turned capture on there (see [`agentic-reviewer.md`](agentic-reviewer.md), which
is where the whole feature and its privacy properties are documented).

Configure it **on the sidecar**, through the environment — the deployed sidecar
image has no repo checkout and therefore no `.fuko.toml` to read (the same
reason the embedding endpoint is environment-only, #216):

```bash
FUKO_TRANSCRIPT_STORE_BACKEND=r2          # "" (off) | file | s3 | r2
FUKO_TRANSCRIPT_STORE_BUCKET=fuko-transcripts
FUKO_TRANSCRIPT_STORE_PREFIX=transcripts               # optional
FUKO_TRANSCRIPT_STORE_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
FUKO_TRANSCRIPT_STORE_CREDS_ENV_PREFIX=FUKO_S3         # reads <prefix>_ACCESS_KEY_ID / _SECRET_ACCESS_KEY / _REGION
```

or, for a single host with no bucket:

```bash
FUKO_TRANSCRIPT_STORE_BACKEND=file
FUKO_TRANSCRIPT_STORE_ROOT=/var/lib/fuko/transcript-blobs
```

The **`s3`/`r2` backends need `boto3`**, which `docker/Dockerfile.sidecar`
installs via the `s3` extra (`pip install ".[s3]"`). Running the sidecar from a
plain `pip install fuko-pr` instead? Install `fuko-pr[s3]` or the bucket
backends answer `503 transcript store unusable: No module named 'boto3'` on
every upload, and log the same line on the sidecar. The `file` backend and the
unconfigured default need nothing.

One exception to configuring it on the sidecar alone: if you run a
`file`-backend sidecar as a plain process **on a host that also runs agentic
reviews**, export `FUKO_TRANSCRIPT_STORE_BACKEND` / `_ROOT` to the runner as
well. The reviewer's read denylist is built from the runner process's own
settings, so otherwise the blob corpus sits on the harness's filesystem with no
rule covering it (see [`agentic-reviewer.md`](agentic-reviewer.md)). The
containerized deployment below is insulated by the container boundary.

**Runners need nothing.** They ship what they captured to the sidecar they
already talk to, over `POST /transcripts/<key>` with the `FUKO_TOKEN` they
already hold; no storage credentials are added to any workflow. An IAM user
limited to `s3:PutObject`/`s3:GetObject` on the bucket is enough, and the bucket
should be private: a transcript holds the reviewed repository as the agent read
it.

`FUKO_TRANSCRIPT_MAX_BYTES` (default 256 MiB) bounds what one upload can make
the sidecar hold; an upload over it is refused whole (`413`) rather than stored
truncated, since a partial blob under a write-once key could never be corrected.
It is a memory ceiling, not a retention policy — real sessions are tens of MB.

**Growth is unbounded by design** — every agentic seat, every push, kept
forever (epic #236). Budget for it.

Renaming `FUKO_TRANSCRIPT_STORE_CREDS_ENV_PREFIX` needs no source edit: the
driver derives `<prefix>_ACCESS_KEY_ID` / `<prefix>_SECRET_ACCESS_KEY` from the
setting at run time and both strips them from the agent's environment and
scrubs them by value from transcripts.

**Turning capture on before storage is a supported order.** A runner that ships
to a sidecar with `FUKO_TRANSCRIPT_STORE_BACKEND` unset gets a `503` marked
`X-Fuko-Transcript-Store: unconfigured`, treats it as the off state, and says
nothing — you do not get a failure line per run while you stage the rollout. A
store that was *meant* to work and does not (unknown backend, missing bucket,
absent `boto3`) is a different `503`: the sidecar logs it and the runner reports
it, once, on stderr. Neither ever faults the review.

## Ollama in Docker

PR-Agent runs in a container; for a host Ollama, set the review model's
`base_url = "http://host.docker.internal:11434"` and, on Linux, add
`docker_extra_args = ["--add-host", "host.docker.internal:host-gateway"]` under
`[review]`.
