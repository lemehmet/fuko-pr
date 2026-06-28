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
   # PR-Agent runs from its Docker image (it is not pip-installable):
   image = "codiumai/pr-agent:latest"   # or your pinned ghcr.io/OWNER/pr-agent:0.38.0

   [review.model]
   provider = "ollama"                  # zai-coding | anthropic | openai | ollama
   name = "qwen2.5-coder:32b"
   base_url = "http://host.docker.internal:11434"  # reach host Ollama from the container

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

4. **Seed knowledge** and **review a PR**:

   ```bash
   fuko ingest-docs docs/*.md --repo owner/repo      # optional: seed from docs
   export GITHUB_TOKEN=...                            # a token that can comment
   fuko review --pr-url https://github.com/owner/repo/pull/123
   fuko signals --pr-url https://github.com/owner/repo/pull/123   # findings as JSON
   ```

Switching the review model later is two lines in `.fuko.toml` plus the matching
key secret — e.g. `provider = "anthropic"`, `name = "claude-sonnet-4-6"`,
`ANTHROPIC_KEY=…`. No other changes.

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

Learnings come from three sources and live in your store:

- **`/remember <text>`** on a PR comment — stores a repo learning. Add a trailing
  `paths: src/**/*.py` line to scope it to files. (`workflows/ingest-comment.yml`)
- **Resolved review threads** — an hourly sweep keeps the last human comment of
  each resolved thread as a learning, scoped to its file. (`workflows/sweep-threads.yml`)
- **Docs / ADRs** — `fuko ingest-docs <globs> --repo owner/repo`.

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
fuko kb count                       # totals + breakdown by repo/source
fuko kb list --repo owner/name --full
fuko kb query owner/name --files path/to/changed.py --text "topic"
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

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). In short: `ruff check`, `ruff format
--check`, and `pytest` (≥ 80% coverage over `sidecar`) must pass; conventions are
in [`AGENTS.md`](AGENTS.md). Security policy: [`SECURITY.md`](SECURITY.md).

## License

[Apache-2.0](LICENSE).
