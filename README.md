# fuko-pr

![CI](https://github.com/<OWNER>/fuko-pr/actions/workflows/ci.yml/badge.svg)
[![coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-success)](#contributing)

A thin RAG **knowledge sidecar** for [PR-Agent](https://github.com/The-PR-Agent/pr-agent).

fuko gives PR-Agent a repo-specific memory: it ingests learnings (from `/remember`
commands, resolved review threads, and your docs/ADRs) into a pgvector store, and
serves the most relevant ones into PR-Agent's `extra_instructions` on each PR review.

PR-Agent stays the reviewer; fuko just gives it memory. PR-Agent chats with
**GLM-5.2 via z.ai**; fuko pulls embeddings from any OpenAI-compatible source
(default: a local [Ollama](https://ollama.com) model, since z.ai exposes no
`/embeddings` endpoint).

## Status

- [x] Sidecar: FastAPI over pgvector (`/ingest`, `/query`, `/forget`, `/healthz`)
- [x] CLI: `serve`, `ingest-docs`, `query`, `forget`, `retrieve`
- [x] z.ai-compatible embeddings via any OpenAI-compatible `/embeddings` (default: local Ollama `bge-m3`)
- [x] Path A review workflow (GitHub Action pre-step → PR-Agent)
- [x] `/remember` + `/forget` on `issue_comment`
- [x] Resolved-thread capture (GraphQL scheduled sweep)

## How it works

```
 INGEST                         STORE                    RETRIEVE & INJECT
 /remember cmds   ┐          ┌──────────────┐         ┌────────────────────────┐
 resolved threads │  embed → │  pgvector    │ ← query │ on PR review:          │
 docs / ADRs      ┘          │  + metadata  │         │ top-k learnings        │
                             └──────────────┘         │ → PR-Agent extra_instr │
```

## Quickstart (local)

1. **Run pgvector**

   ```bash
   docker run -d --name fuko-pg -p 5432:5432 \
     -e POSTGRES_USER=fuko -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=fuko \
     pgvector/pgvector:pg16
   ```

2. **Run an embeddings model** (default: Ollama with `bge-m3`)

   ```bash
   ollama pull bge-m3
   ```

   z.ai has no embeddings endpoint, so use a local model or point `FUKO_EMBED_BASE_URL`
   at any OpenAI-compatible provider (see `.env.example`).

3. **Configure**

   ```bash
   cp .env.example .env
   # set FUKO_DATABASE_URL (embeddings default to local Ollama)
   ```

4. **Install**

   ```bash
   pip install -e ".[dev]"
   ```

5. **Seed knowledge from your docs**

   ```bash
   fuko ingest-docs docs/architecture.md 'ADR/*.md' --repo owner/repo
   ```

6. **Query**

   ```bash
   fuko query --repo owner/repo --file src/auth/login.ts --file src/auth/session.ts
   ```

7. **Run the sidecar**

   ```bash
   fuko serve          # http://localhost:8000, docs at /docs
   curl localhost:8000/healthz
   ```

The schema is created automatically on first connection (see
`migrations/001_init.sql`). The `embedding` column is sized to whatever the model
returns (probed at startup). If you change the embedding model to one with a
different dimension, fuko re-embeds every stored learning and rebuilds the vector
column/index automatically on the next startup — a one-time, potentially slow
operation, but no manual migration is needed.

## API

| Method | Path       | Body                                                      |
|--------|------------|-----------------------------------------------------------|
| POST   | `/ingest`  | `{repo, items:[{text, source, source_url?, file_globs?, topic?}]}` |
| POST   | `/query`   | `{repo, files:[...], pr_body?, query_text?, top_k?}`      |
| POST   | `/forget`  | `{repo, id? \| source? \| all?}`                          |
| GET    | `/healthz` | —                                                         |

Set `FUKO_AUTH_TOKEN` to require `Authorization: Bearer <token>` on requests.

## Path A: wiring it into PR-Agent

`workflows/pr-review.yml` is a drop-in workflow that, on each PR:

1. fetches the PR's changed files + body via the GitHub API,
2. asks the fuko sidecar `/query` for relevant learnings,
3. feeds them into PR-Agent via `pr_reviewer.extra_instructions` /
   `pr_code_suggestions.extra_instructions` (the no-fork seam), and
4. runs PR-Agent with GLM-5.2 via z.ai.

Deploy it:

1. Copy `workflows/pr-review.yml` into your app repo as
   `.github/workflows/pr-review.yml`.
2. Run the fuko sidecar somewhere reachable from your self-hosted runners
   (`docker/Dockerfile.sidecar`).
3. Add repo secrets:
   - `FUKO_URL` — sidecar base URL (e.g. `http://fuko.internal:8000`)
   - `FUKO_TOKEN` — optional, only if `FUKO_AUTH_TOKEN` is set on the sidecar
   - `ZAI_KEY` — your z.ai key. The GLM **Coding-plan** subscription is served
     from the **coding** endpoint (`OPENAI__API_BASE: https://api.z.ai/api/coding/paas/v4`,
     already set in the workflow); the standard `/api/paas/v4` is pay-per-token.
4. Optional — post reviews as a **"Fuko PR Review" GitHub App** instead of
   `github-actions[bot]`: create the App (Pull requests RW, Issues RW, Contents R),
   install it on the repo, then set repo **variable** `FUKO_APP_ID` and secret
   `FUKO_APP_PRIVATE_KEY`. (App ID is a *variable* because the `secrets` context is
   not allowed in step `if:`.) Without these the workflow falls back to the
   default token.
5. Tune `runs-on` to match your runner fleet. If the sidecar is briefly
   unreachable, the workflow logs a warning and reviews without injected knowledge.

> **PR-Agent settings use dynaconf dunder keys.** Nested settings must be passed
> as `SECTION__KEY` env vars (e.g. `CONFIG__MODEL`, `OPENAI__API_BASE`,
> `PR_REVIEWER__EXTRA_INSTRUCTIONS`) — dotted keys like `config.model` are
> silently ignored. GLM-5.2 also needs `CONFIG__CUSTOM_MODEL_MAX_TOKENS` (it's not
> in PR-Agent's built-in table) and a raised `CONFIG__AI_TIMEOUT`.

## Interactive commands

`workflows/pr-command.yml` runs PR-Agent tools on demand from a PR comment —
`/review`, `/improve`, `/ask <q>`, `/describe`, `/add_docs`, `/update_changelog`,
`/generate_labels`, `/similar_issue`, `/help`, `/config`. Gated on a repo-trusted
`author_association` and a job-level same-repo guard (a GitHub-hosted `guard` job
the self-hosted job `needs:`), so fork PRs never schedule on the self-hosted fleet.

## Adding knowledge from PR comments

`workflows/ingest-comment.yml` listens for PR comments and forwards commands to
the sidecar's `POST /comment` endpoint:

- `/remember <text>` — stores a repo learning. Add a trailing
  `paths: src/**/*.py, tests/*.py` line to scope it to specific files. Reacts 👍.
- `/forget all` · `/forget source=<docs|remember|resolved_thread>` ·
  `/forget <id>` — removes learnings. Reacts 🚀 (or 👎 if nothing matched).

Bot comments are ignored (no feedback loops). Drop the file into your app repo as
`.github/workflows/ingest-comment.yml`; it uses the same `FUKO_URL`/`FUKO_TOKEN`
secrets as the review workflow.

## Learning from resolved review threads

`workflows/sweep-threads.yml` runs hourly (scheduled workflows run on the default
branch; also triggerable manually via `workflow_dispatch`). It pulls the resolved
review threads from the 20 most-recently-updated open PRs via GraphQL and forwards
them to `POST /ingest-threads`. The sidecar keeps the **last human comment** of
each resolved thread as a learning (`source=resolved_thread`), scoped to the
thread's file and linked to the comment permalink. Bot comments (`*[bot]`) are
excluded; set `FUKO_BOT_LOGIN` to also exclude a non-`[bot]` reviewer app.
Re-sweeping is a no-op thanks to the `UNIQUE(repo, text, source)` dedup.

## The retrieval flywheel

`/query` does two passes: a **semantic** top-N by cosine distance, plus any
explicitly **file-scoped** learnings (`file_globs != '{}'`); then filters the
scoped ones with `fnmatch` against the PR's changed paths, and returns the
top-k. Repo-global learnings (empty `file_globs`) always pass the filter.

## Contributing

CI runs `ruff check`, `ruff format --check`, and `pytest` with a `--cov-fail-under=80`
gate over the `sidecar` package (pgvector + Ollama are provisioned in the `test` job).

Local dev:

```bash
make up                # pgvector + Ollama via docker compose
make ollama-pull       # pull bge-m3 into the Ollama container
cp .env.example .env   # set FUKO_DATABASE_URL
make install
make test              # pytest + coverage gate
make lint              # ruff check
```

All code conventions (coverage ≥ 80%, mandatory docstrings, stdlib-first, etc.) are
documented in [`AGENTS.md`](./AGENTS.md).

## License

Apache-2.0
