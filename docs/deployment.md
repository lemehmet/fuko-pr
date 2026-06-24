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
without a local Ollama, point `[embedding]` at a remote provider (see below).

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
Jina, or BigModel). Set `[embedding] base_url`/`model` and the key env var.
Embeddings are cheap (pennies), and the file is small, so each run's
download/query/upload is fast.

## Ollama in Docker

PR-Agent runs in a container; for a host Ollama, set the review model's
`base_url = "http://host.docker.internal:11434"` and, on Linux, add
`docker_extra_args = ["--add-host", "host.docker.internal:host-gateway"]` under
`[review]`.
