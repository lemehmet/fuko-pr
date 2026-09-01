# Sidecar & runner setup

How to run the **fuko-pr sidecar stack** and point an application repository's
workflows at it.

## Pick a topology

The sidecar is an HTTP service; nothing requires it to share a host with the
runner. Two arrangements work, and they differ only in what `FUKO_URL` points at.

| | Co-located | Dedicated host |
| --- | --- | --- |
| Sidecar runs on | the runner itself | its own box / VM / container |
| `FUKO_URL` | `http://localhost:8000` | `http://<host>:8000` |
| Compose file | [`docker/runner-compose.yml`](./docker/runner-compose.yml) | your own, or config management |
| Good for | a single runner, trying things out | several runners or repos sharing one knowledge base |

The knowledge base is per-repo *inside* one store, so a dedicated host lets every
repo and every runner share a single sidecar. That is how the reference
deployment runs; see [Dedicated host](#dedicated-host) below.

## Prerequisites

A Linux x64 host with:

- Docker + Docker Compose v2
- ~3 GB free for the Ollama `bge-m3` model
- Outbound access to your chat provider (e.g. `api.z.ai`, `openrouter.ai`) and to GitHub
- For the co-located setup: a GitHub self-hosted runner agent registered to the repo or org
- **For the agentic backend only** (any seat with `backend = "agentic"`): the
  `claude` CLI (Claude Code) on the RUNNER's `PATH` — the driver runs it as a
  subprocess of the review job, not in Docker (containerizing it is #102). A
  missing binary fails that branch with `HarnessNotAvailableError` and does NOT
  fail over; it is a provisioning gap, not a throttle. Install the native build
  version-pinned and keep it off auto-update so review behavior only changes
  when you choose:

  ```bash
  curl -fsSL https://claude.ai/install.sh | bash -s -- <pinned-version>
  export DISABLE_AUTOUPDATER=1   # in the runner service environment
  ```

  The seat's model key (e.g. `QWEN_TOKEN_PLAN_KEY` for the `qwen-anthropic`
  preset) must be exported in the workflow's `env:` block from a repo secret —
  the driver reads it on the runner, not on the sidecar.

## 1. Register the GitHub runner

Follow [GitHub's self-hosted runner guide](https://docs.github.com/en/actions/hosting-your-own-runners).
When configuring, attach whatever labels your workflows select on and install it
as a service so it survives reboots:

```bash
./config.sh \
  --url https://github.com/<owner>/<app-repo> \
  --token <REGISTRATION_TOKEN> \
  --labels "<your-fleet-label>" \
  --unattended
sudo ./svc.sh install && sudo ./svc.sh start
```

`<REGISTRATION_TOKEN>` comes from the repo/org **Settings → Actions → Runners →
New self-hosted runner**. `self-hosted` is added automatically, as are the OS/arch
labels (matching is case-insensitive).

The shipped workflows use a bare `runs-on: self-hosted` for every job that reaches
the sidecar; the lightweight permission-guard jobs run on `ubuntu-latest`. Narrow
the `self-hosted` ones to your own labels when you copy them into an app repo.

## 2. Start the fuko stack

Pick a stable checkout directory you can write to — `$HOME/fuko-pr` needs no
privileges, `/opt/fuko-pr` is the conventional service location but is
root-owned by default (`sudo install -d -o "$USER" /opt/fuko-pr` first). The
rest of this page writes `$FUKO_SRC`; substitute your own.

```bash
export FUKO_SRC=/opt/fuko-pr                     # or $HOME/fuko-pr
git clone <fuko-pr-url> "$FUKO_SRC"
cd "$FUKO_SRC"
export COMPOSE_PROJECT_NAME=fuko                 # see the warning below
export FUKO_AUTH_TOKEN=$(openssl rand -hex 16)   # workflows send this as FUKO_TOKEN
docker compose -f docker/runner-compose.yml up -d --build
docker compose -f docker/runner-compose.yml exec ollama ollama pull bge-m3
```

[`docker/runner-compose.yml`](./docker/runner-compose.yml) wires:

- `pg` — pgvector knowledge store
- `ollama` — local embeddings backend (`bge-m3`, 1024-dim)
- `sidecar` — FastAPI service on host port `8000`, auth via `FUKO_AUTH_TOKEN`

The compose file pins `FUKO_EMBED_MODEL: bge-m3` and `FUKO_EMBED_QUERY_PREFIX:
""` on the sidecar, so this stack works as written. Both are overrides, not
defaults: the sidecar's built-in default is `qwen3-embedding-0.6b`, which this
Ollama service does not serve, and its query instruction is wrong for a
symmetric model like bge-m3. **If you deploy the sidecar without this compose
file** — the [dedicated-host](#dedicated-host) path, or your own unit — set both
yourself, together. Only the model is tracked in `meta`, so a mismatched prefix
degrades retrieval with nothing in the logs to show for it; see
[Changing the embedding model](#6-updating-the-stack).

> **Pin `COMPOSE_PROJECT_NAME`, or always invoke compose identically.**
> The project name decides which volumes you get, and a different name means a
> second, empty knowledge base plus a port-8000 collision with the stack you
> already had. Compose derives it from the base name of the directory holding
> the first `-f` file — **not** your shell's working directory — so the commands
> above yield the project `docker`, and moving the compose file elsewhere
> silently changes it. Export `COMPOSE_PROJECT_NAME=fuko` (or pass `-p fuko`)
> once and use it for every invocation.
>
> Already running a stack? Check its name with `docker compose ls` first.
> Adopting a different project name orphans the volumes holding your existing
> knowledge base rather than migrating them.

All services use `restart: unless-stopped`. Ensure Docker starts on boot so the
sidecar returns after a host reboot:

```bash
sudo systemctl enable docker
```

## 3. Verify

`/healthz` deliberately touches neither auth nor the database, so on its own it
can green-light a sidecar whose Postgres never came up. Check an authenticated,
DB-backed endpoint too:

```bash
curl -s localhost:8000/healthz                                   # {"ok":true}
curl -s -H "Authorization: Bearer $FUKO_AUTH_TOKEN" \
  -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"repo":"<owner>/<app-repo>"}'                             # {"results":[]}
```

## 4. Configure the application repository

**Settings → Secrets and variables → Actions.**

Secrets:

| Secret           | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| `FUKO_URL`       | `http://localhost:8000`, or `http://<host>:8000`          |
| `FUKO_TOKEN`     | the `FUKO_AUTH_TOKEN` generated above                     |
| `ZAI_KEY`        | your z.ai API key (if a configured model uses that provider) |
| `OPENROUTER_KEY` | your OpenRouter API key (likewise)                        |

`GITHUB_TOKEN` is provided automatically.

Variables (all optional — the defaults are fine for most repos):

| Variable           | Effect                                                                 |
| ------------------ | ---------------------------------------------------------------------- |
| `FUKO_CHUNK_SIZE`  | threads per `/ingest-threads` request in the sweep (1-100, default 25)  |
| `FUKO_BOT_LOGIN`   | also exclude a reviewer service account whose login has no `[bot]` suffix |

## 5. Sidecar settings

Read from the environment with a `FUKO_` prefix (see [`.env.example`](./.env.example)):

| Setting                | Default | Effect                                                            |
| ---------------------- | ------- | ------------------------------------------------------------------ |
| `FUKO_AUTH_TOKEN`      | unset   | bearer token for every write/read endpoint; unset **refuses** them   |
| `FUKO_TOP_K`           | `6`     | learnings injected into a review                                    |
| `FUKO_INGEST_MAX_NEW`  | `10`    | new learnings embedded per `/ingest-threads` call                   |
| `FUKO_EMBED_BASE_URL`  | Ollama  | any OpenAI-compatible `/embeddings` endpoint                        |
| `FUKO_EMBED_MODEL`     | `qwen3-embedding-0.6b` | embedding model, and the provenance marker for the stored vectors |
| `FUKO_EMBED_QUERY_PREFIX` | Qwen3 task instruction | prepended to *queries* only; set to empty for a symmetric model |

`FUKO_EMBED_MODEL` doubles as the provenance marker: changing it re-embeds the
whole knowledge base on the next startup, because two models at the same
dimension produce incomparable vectors and nothing else would notice. It must
name whatever the endpoint actually serves — for the compose stack below that
is `bge-m3`, so `docker/runner-compose.yml` pins both it and an empty
`FUKO_EMBED_QUERY_PREFIX`. Only the model is tracked; a query prefix that does
not match it degrades retrieval silently, which is why the two move together.

`FUKO_INGEST_MAX_NEW` bounds how long a single ingest request can take, not how
much a sweep can ingest: the sweep re-posts a batch until the sidecar reports
`remaining: 0`, and already-stored learnings dedup away for free. Lower it if a
slow embedder still makes the sweep's POST time out.

## 6. Updating the stack

```bash
cd "$FUKO_SRC"
git pull
docker compose -f docker/runner-compose.yml up -d --build   # rebuilds sidecar image
```

Migrations in `migrations/*.sql` are idempotent and apply themselves at sidecar
startup, so a newer build needs no manual database step.

Rotate the auth token: change `FUKO_AUTH_TOKEN`, restart the sidecar, and update
the `FUKO_TOKEN` secret in each app repo:

```bash
docker compose -f docker/runner-compose.yml restart sidecar
```

**Changing the embedding model needs no manual migration.** Point
`FUKO_EMBED_MODEL` (and `FUKO_EMBED_BASE_URL`) at the new one, **set
`FUKO_EMBED_QUERY_PREFIX` to match it**, and restart. The sidecar re-embeds
every learning and rebuilds the vector column and index itself — a one-time and
potentially slow startup, but automatic. Do not drop the `learnings` table by
hand; see [`AGENTS.md`](./AGENTS.md).

Two triggers, not one. A **dimension** change is visible in the schema. A
**model** change at the same dimension is not — bge-m3 and Qwen3-Embedding-0.6B
are both 1024-wide — so the sidecar records the model that produced the stored
vectors in `meta.embed_model` and re-embeds when that changes too. An absent
marker counts as a change, so the first restart after upgrading to a build that
has this table re-embeds once by design.

The query prefix is the half that is **not** covered by the marker, because it
changes nothing about the stored vectors and must not trigger a re-embed. It is
also the half that fails silently: a query embedded with an instruction the
documents never carried still returns a well-formed vector and a plausible
ranking. So move it with the model, in the same edit — empty for a symmetric
model such as bge-m3, the model's own task instruction for an asymmetric one
such as Qwen3-Embedding.

## Dedicated host

Nothing in the sidecar cares whether a runner is present. Put the same three
containers on their own box, expose port 8000 to the runners, and set each app
repo's `FUKO_URL` secret to `http://<host>:8000` instead of `localhost`.

Worth doing there:

- **Keep secrets off the compose file.** Render them to a root-owned env file and
  reference it with `env_file:`, so rebuilds and config management never bake a
  token into the image or the repo.
- **Bind deliberately.** Port 8000 carries your whole knowledge base behind a
  single bearer token. Expose it on a trusted network only.
- **Back up the `pg` volume.** It *is* the knowledge base; the containers are
  disposable, that volume is not.
- **Smoke an authed endpoint after every deploy**, not just `/healthz` — see
  [Verify](#3-verify).

## Troubleshooting

- **"context build failed" in the review workflow** — the sidecar isn't reachable from
  the job. Confirm `curl <FUKO_URL>/healthz` works from the runner and that the job
  actually ran on the runner you expected (labels matched).
- **Embedding 400 / model not found** — `ollama pull bge-m3` not run, or
  `FUKO_EMBED_MODEL`/`FUKO_EMBED_BASE_URL` mismatch. `FUKO_EMBED_MODEL` defaults
  to `qwen3-embedding-0.6b`, which this Ollama stack does not serve, so it has
  to be pinned to `bge-m3` (the compose file does) rather than left unset.
- **Retrieval got worse after an embedding-model change, with nothing in the
  logs** — `FUKO_EMBED_QUERY_PREFIX` was not moved with `FUKO_EMBED_MODEL`. Only
  the model is tracked in `meta`, so a mismatched query prefix embeds the query
  side into a different shape than the documents and still returns well-formed
  vectors. Clear it for a symmetric model (bge-m3), set the model's own
  instruction for an asymmetric one (Qwen3-Embedding).
- **The sweep reports `chunk N failed: timed out`** — the sidecar took too long to
  embed a batch. It retries on the next hourly sweep by itself; if it persists,
  lower `FUKO_INGEST_MAX_NEW` on the sidecar, or `FUKO_CHUNK_SIZE` on the repo.
- **The sweep runs clean but stores nothing** — expected when no review thread
  ends in a *decline*. Only a trusted author pushing back on a finding and stating
  the convention is kept; fix acknowledgements and chatter are dropped by design.
- **Empty knowledge base after a restart** — compose almost certainly picked a
  different project name and therefore a different volume. See the warning in
  [Start the fuko stack](#2-start-the-fuko-stack).
- **PR-Agent model error** — a provider key is missing/invalid, the provider is
  unreachable, or the model config in `.fuko.toml` is wrong.
