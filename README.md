# fuko-pr

![CI](https://github.com/lemehmet/fuko-pr/actions/workflows/ci.yml/badge.svg)
[![coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-success)](./CONTRIBUTING.md)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)

**A vendor-neutral layer over PR review bots, with a knowledge base you own.**

Closed reviewers (CodeRabbit, Copilot) give sustained quality, but their
performance and rate limits are volatile and their closed nature makes switching
costly. fuko-pr puts a thin, swappable layer in front of an open reviewer
([PR-Agent](https://github.com/the-pr-agent/pr-agent) today; more later) so you
can:

- **choose your review model/provider** (z.ai/GLM, Anthropic, Ollama, …) with one
  config edit — no relearning each bot's settings,
- **own your knowledge base** — repo-specific learnings live in *your* store
  (Postgres or a sqlite-vec file in your own S3/R2 bucket) and survive switching
  the reviewer or the model underneath,
- **read every reviewer's output as one schema** — `fuko signals` normalizes
  PR-Agent, Copilot, and CodeRabbit findings into a single deterministic format.

It's [Terraform](https://terraform.io) for review bots: a uniform config + a
driver per backend, plus a portable knowledge base.

## How it works

```
 .fuko.toml ──► fuko review ──────────────────────────────────────────────┐
 (backend +     1. retrieve repo knowledge  ── your Store (pgvector|sqlite-vec)
  model +       2. translate config ► backend env        (Postgres | S3/R2 file)
  keys via      3. invoke the reviewer (PR-Agent, in Docker)
  env)          4. normalize its output ► Review Signal v1  ──► fuko signals (JSON)
                                                                            ┘
 KNOWLEDGE IN:  /remember comments · resolved review threads · docs/ADRs ─► Store
```

The knowledge base is the constant: steps 1–2 don't know which backend/model runs
in 3–4, and only the driver knows how a given reviewer is configured and parsed.

## Quickstart (local, fully offline)

Reviews a PR on your machine using a local Ollama model and a local knowledge
file — no paid APIs, no server.

1. **Install** (with the server-free store extra):

   ```bash
   pip install -e ".[sqlite]"        # or: pip install "fuko-pr[sqlite]" once published
   ```

2. **Pull a local model for embeddings** (the knowledge base) and, optionally, for
   the review itself:

   ```bash
   ollama pull bge-m3                 # embeddings
   ollama pull qwen2.5-coder:32b      # a local review model (optional)
   ```

3. **Configure** — copy the example and pick a provider:

   ```bash
   cp .fuko.toml.example .fuko.toml
   ```

   ```toml
   [review]
   backend = "pr-agent"
   # PR-Agent runs from its Docker image (it is not pip-installable). The public
   # codiumai image lags (stops at 0.34), so prefer the pinned GHCR image this
   # repo builds (see "Deploying as a GitHub Action" below):
   image = "ghcr.io/OWNER/pr-agent:0.41.0"   # pin the matching @sha256 digest from your build

   [[review.models]]
   provider = "ollama"                  # zai-coding | openrouter | lemonade | anthropic | openai | ollama
   name = "qwen2.5-coder:32b"
   base_url = "http://host.docker.internal:11434"  # reach host Ollama from the container
   # Add more entries to scale up: every active entry (the default role) reviews
   # each PR — two or more actives run as an A/B comparison — while entries with
   # role = "backup" are shared failover targets used when a provider throttles.
   # role = "trial" runs a candidate model alongside the actives and surfaces its
   # output for evaluation, but non-gating (consumers don't block or gate on it).

   [knowledge]
   store = "sqlite-vec"
   [knowledge.object_store]
   backend = "file"
   key = ".fuko/kb.db"                  # a local file; use s3/r2 for CI runners

   [embedding]
   provider = "ollama"
   model = "bge-m3"
   base_url = "http://localhost:11434/v1"
   ```

   > **The embedding endpoint is configured by environment, not by this
   > section.** `[embedding]` is parsed but not yet consumed (#216) — the
   > sidecar reads `FUKO_EMBED_*` only. For the local Ollama setup above that
   > means, in your `.env` (see [`.env.example`](./.env.example)):
   >
   > ```bash
   > FUKO_EMBED_BASE_URL=http://localhost:11434/v1
   > FUKO_EMBED_MODEL=bge-m3
   > FUKO_EMBED_QUERY_PREFIX=
   > ```
   >
   > Both lines matter. `FUKO_EMBED_MODEL` defaults to
   > `qwen3-embedding-0.6b`, which Ollama does not serve here, and it doubles
   > as the provenance marker for the stored vectors — changing it re-embeds
   > the knowledge base. `FUKO_EMBED_QUERY_PREFIX` defaults to that model's
   > task instruction, which a symmetric model like bge-m3 must not receive.

4. **Seed knowledge** and **review a PR**:

   ```bash
   fuko ingest-docs docs/*.md --repo owner/repo      # optional: seed from docs
   export GITHUB_TOKEN=...                            # a token that can comment
   fuko review --pr-url https://github.com/owner/repo/pull/123
   fuko signals --pr-url https://github.com/owner/repo/pull/123   # findings as JSON
   ```

Switching the review model later is two lines in `.fuko.toml` plus the matching
key secret — e.g. `provider = "anthropic"`, `name = "claude-sonnet-4-6"`,
`ANTHROPIC_KEY=…`. No other changes. (The pre-unification `[review.model]`,
`[[review.providers]]`, and `[[review.compare]]` sections still parse — they map
onto `[[review.models]]` with a deprecation nudge on stderr.)

## Deploying as a GitHub Action

Copy `workflows/pr-review.yml` into your app repo as
`.github/workflows/pr-review.yml`, commit a `.fuko.toml`, and add the secrets your
config needs (the model provider's key, e.g. `ZAI_KEY` or `ANTHROPIC_KEY`; plus
`FUKO_URL`/`FUKO_TOKEN` if you run a knowledge sidecar). The workflow installs
fuko-pr and calls `fuko review`; PR-Agent runs from its Docker image, so the
runner only needs Docker — not a working PR-Agent Python environment.

Optionally post reviews as a **"Fuko PR Review" GitHub App** instead of
`github-actions[bot]`: create the App (Pull requests RW, Issues RW, Contents R),
install it, then set repo **variable** `FUKO_APP_ID` + secret `FUKO_APP_PRIVATE_KEY`.

PR-Agent isn't published as a usable pip package (its pins conflict) and the public
image lags; `.github/workflows/pr-agent-image.yml` builds a pinned, multi-arch
`pr-agent` image and pushes it to your GHCR, which you then reference as
`[review].image`.

## Deployment modes

| Mode | Store | Server? | Embeddings | Best for |
|------|-------|---------|------------|----------|
| **Homelab / self-host** | Postgres + pgvector | sidecar | local Ollama | a private fleet you control |
| **Managed DB** | Neon / Supabase pgvector | optional | remote provider | SaaS runners, fine with a DB |
| **Server-free** | sqlite-vec file in S3/R2 | none | remote provider | SaaS runners, no infra |

See [`docs/deployment.md`](docs/deployment.md) for the server-free S3/R2 setup and
the trade-offs.

## The knowledge base

Learnings come from four sources and live in your store:

- **`/remember <text>`** on a PR comment — stores a repo learning. Add a trailing
  `paths: src/**/*.py` line to scope it to files. (`workflows/ingest-comment.yml`)
- **PR review threads** — an hourly sweep of merged PRs keeps a thread's last
  trusted-author comment when it declines a reviewer finding and states the
  project's convention, scoped to its file. Resolution state is not a filter:
  the merge settles the thread, and a decline is typically left unresolved while
  a fix resolves it. (`workflows/sweep-threads.yml`)
- **Docs / ADRs** — `fuko ingest-docs <globs> --repo owner/repo`.
- **File structure indexes** — `fuko digest <paths> --repo owner/repo`, run in a
  checkout. For every file at or above `--min-bytes` (64 KB by default) it stores
  a map of what the file declares and at which lines, scoped to that file's own
  path and keyed on the hash of the blob it describes, so an edit supersedes its
  own index. The point is that a reviewer facing a 400 KB source file can read
  the two hundred lines it needs instead of the whole thing. Paths are stored
  relative to the working directory and anything outside it is skipped with a
  warning — retrieval matches these against the repository-relative paths a pull
  request reports, so an index keyed any other way could never be found.

  The index is extracted mechanically (Python via `ast`, everything else via a
  declaration scan) rather than written by a model, and it carries identifiers
  and line numbers only — no prose, no doc comments. That is deliberate: an
  index that cannot express an opinion cannot smuggle in "this file is fine",
  and it shares nothing across reviewers that the source file does not already.

  **Retrieval is off by default.** Indexes are stored but invisible to reviews
  until `FUKO_DIGEST_RETRIEVAL=1` is set on the deployment that serves them.

On each review, `fuko review` retrieves the most relevant learnings (semantic
top-N by cosine distance plus any file-scoped ones matching the changed paths) and
feeds them to the reviewer. Changing the embedding model re-embeds everything and
rebuilds the vector index automatically — no manual migration.

## Reading reviewer output: `fuko signals`

`fuko signals --pr-url <url>` emits every reviewer's findings on a PR as one
canonical JSON schema — **fuko Review Signal v1** — so a downstream
"address-the-reviews" tool reads one shape instead of sniffing each vendor's
markdown. PR-Agent declares severity/category; Copilot and CodeRabbit are detected
by author and mapped best-effort (`severity_source` records which). See
[`docs/review-signal-v1.md`](docs/review-signal-v1.md).

## Optional: the knowledge sidecar

`fuko serve` runs a small FastAPI service (`/ingest`, `/query`, `/learnings`,
`/forget`, `/healthz`, `/comment`, `/ingest-threads`) over your store — useful when you want a
shared, always-on knowledge endpoint for a fleet. Set `FUKO_AUTH_TOKEN` to require
`Authorization: Bearer <token>`. In server-free and managed-DB modes the sidecar is
optional; `fuko review` talks to the store directly.

To browse a running sidecar's store from any machine, `fuko kb` is an HTTP client
over `FUKO_URL` + `FUKO_AUTH_TOKEN`:

```bash
fuko kb repos                       # every repo with a KB, and its source mix
fuko kb count                       # totals + breakdown by repo/source
fuko kb list --repo owner/name -q "glob" --full     # --include-expired to see expired ones
fuko kb query owner/name --files path/to/changed.py --text "topic"
fuko kb edit owner/name <uuid> --topic "Migrations" --globs "migrations/*.sql"
fuko kb forget owner/name --id <uuid>
```

(`fuko query`/`fuko forget` do the same against the *local* store via `.fuko.toml`.)

## Configuration

- **`.fuko.toml`** (committed, per-repo): backend, model provider, tools, store,
  embedding. See `.fuko.toml.example`. Secrets are never in this file — each
  provider preset declares the env var that holds its key.
- **`FUKO_*` env** (runtime/server settings): `FUKO_DATABASE_URL`, `FUKO_EMBED_*`,
  `FUKO_AUTH_TOKEN`, etc. See `.env.example`.

Design and contracts: [`docs/design.md`](docs/design.md).

## Browser UI

The sidecar serves utility pages under `/ui`:

- **`/ui/metrics`** — per-model and per-slot review aggregates, recent runs,
  reviewer health, open provider cooldowns.
- **`/ui/kb`** — the knowledge-base console. Pick a repository, search and page
  through its learnings, fix a mis-scoped glob or a bad text, add one by hand,
  upload design docs (same chunking as `fuko ingest-docs`), purge in bulk, and
  **preview retrieval** — the query a review would run, so you can check a
  learning will actually reach the reviewer.

Browsing is unauthenticated, for a LAN-only deployment. Editing needs the
sidecar's `FUKO_AUTH_TOKEN`, exchanged once at `/ui/login` for a signed
`HttpOnly`, `SameSite=Strict` session cookie; with no token configured, every
editing action is refused. Adding a page: [`docs/web-ui.md`](docs/web-ui.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). In short: `ruff check`, `ruff format
--check`, and `pytest` (≥ 80% coverage over `sidecar`) must pass; conventions are
in [`AGENTS.md`](AGENTS.md). Security policy: [`SECURITY.md`](SECURITY.md).

## License

[Apache-2.0](LICENSE).
