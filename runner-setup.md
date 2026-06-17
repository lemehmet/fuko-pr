# Runner setup

How to prepare a self-hosted runner to host the **fuko-pr sidecar stack** and run
the fuko-pr workflows against an application repository.

The workflows (`workflows/*.yml`) target a runner labeled
`[self-hosted, mepro, X64, Linux]`. The fuko sidecar (pgvector + Ollama + the
FastAPI service) runs as containers **on the same host**; the workflows reach it at
`http://localhost:8000`.

## Prerequisites

A Linux x64 host with:

- Docker + Docker Compose v2
- ~3 GB free for the Ollama `bge-m3` model
- Outbound access to `api.z.ai` (GLM-5.2 chat) and to GitHub
- A GitHub self-hosted runner agent registered to the repo or org

## 1. Register the GitHub runner

Follow [GitHub's self-hosted runner guide](https://docs.github.com/en/actions/hosting-your-own-runners).
When configuring, attach the custom labels and install it as a service so it
survives reboots:

```bash
./config.sh \
  --url https://github.com/<owner>/<app-repo> \
  --token <REGISTRATION_TOKEN> \
  --labels "mepro,X64,Linux" \
  --unattended
sudo ./svc.sh install && sudo ./svc.sh start
```

`<REGISTRATION_TOKEN>` comes from the repo/org **Settings → Actions → Runners →
New self-hosted runner**. `self-hosted` is added automatically; `Linux`/`X64` match
the auto-added OS/arch labels (matching is case-insensitive); `mepro` targets this
specific fleet.

## 2. Start the fuko stack

On the runner, clone this repo and bring up the stack:

```bash
git clone <fuko-pr-url> ~/fuko-pr
cd ~/fuko-pr
export FUKO_AUTH_TOKEN=$(openssl rand -hex 16)   # workflows send this as FUKO_TOKEN
docker compose -f docker/runner-compose.yml up -d --build
docker compose -f docker/runner-compose.yml exec ollama ollama pull bge-m3
```

[`docker/runner-compose.yml`](./docker/runner-compose.yml) wires:

- `pg` — pgvector knowledge store
- `ollama` — local embeddings backend (`bge-m3`, 1024-dim)
- `sidecar` — FastAPI service on host port `8000`, auth via `FUKO_AUTH_TOKEN`

All services use `restart: unless-stopped`. Ensure Docker starts on boot so the
sidecar returns after a host reboot:

```bash
sudo systemctl enable docker
```

## 3. Verify

```bash
curl -s localhost:8000/healthz                                   # {"ok":true}
curl -s -H "Authorization: Bearer $FUKO_AUTH_TOKEN" \
  -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"repo":"<owner>/<app-repo>"}'                             # {"results":[]}
```

## 4. Add secrets to the application repository

In the target repo (e.g. `lemehmet/pomotodo`) → **Settings → Secrets and variables →
Actions**:

| Secret       | Value                                              |
| ------------ | -------------------------------------------------- |
| `FUKO_URL`   | `http://localhost:8000`                            |
| `FUKO_TOKEN` | the `FUKO_AUTH_TOKEN` generated above              |
| `ZAI_KEY`    | your z.ai API key (for GLM-5.2 chat)               |

`GITHUB_TOKEN` is provided automatically.

## 5. Updating the stack

Deploy a newer `fuko-pr`:

```bash
cd ~/fuko-pr
git pull
docker compose -f docker/runner-compose.yml up -d --build   # rebuilds sidecar image
```

Rotate the auth token: change `FUKO_AUTH_TOKEN`, restart the sidecar, and update the
`FUKO_TOKEN` secret in each app repo:

```bash
docker compose -f docker/runner-compose.yml restart sidecar
```

Change the embedding model: update `FUKO_EMBED_MODEL`/`FUKO_EMBED_DIM` in
`docker/runner-compose.yml`, **and** the `vector()` dimension in
`migrations/001_init.sql`, then drop & recreate the `learnings` table (the migration
is `IF NOT EXISTS`) — see [`AGENTS.md`](./AGENTS.md).

## Troubleshooting

- **"context build failed" in the review workflow** — the sidecar isn't reachable from
  the job. Confirm `curl localhost:8000/healthz` works on the host and that the job
  actually ran on this runner (labels matched).
- **Embedding 400 / model not found** — `ollama pull bge-m3` not run, or
  `FUKO_EMBED_MODEL`/`FUKO_EMBED_BASE_URL` mismatch.
- **PR-Agent model error** — `ZAI_KEY` missing/invalid, or `config.model` /
  `OPENAI.API_BASE` changed in the review workflow.
