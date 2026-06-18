# fuko-pr design

> Status: design draft (pre-implementation). This document defines the contracts;
> code should follow it, and changes to the contracts should land here first.

## What fuko-pr is

A **vendor-neutral abstraction layer over PR review bots**, with a **portable,
user-owned knowledge base** as the durable asset.

Closed reviewers (CodeRabbit, Copilot) deliver sustained quality but their
performance and rate limits are volatile, and their closed nature makes switching
vendors hard. PR-Agent cracks that open, and more PR-Agent-likes will follow.
fuko-pr's job is to let an engineer:

- choose the **review backend** (PR-Agent today; others later),
- choose the **underlying model/provider** (z.ai/GLM, Ollama, OpenAI, Anthropic, …),

with minimal reconfiguration — just API keys — and **own their knowledge base**,
which survives swapping the backend or the model underneath.

The analogy is **Terraform/LiteLLM**, not OpenRouter: a uniform config + workflow
surface with a *driver per backend*, accepting that each backend exposes some
backend-specific capability. fuko-pr stays a loosely-coupled companion; there is no
intent to upstream it into PR-Agent.

## Architecture: a bidirectional layer over a pluggable store

```
            .fuko.yaml (unified config, keys via env)
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                      fuko review (runner)                      │
  │                                                                │
  │  1. retrieve knowledge ── Store.query() ──┐                    │
  │  2. driver.build_env(preset, model, kb) ──┤  INGRESS           │
  │  3. driver.invoke(pr_url, env) ── backend subprocess           │
  │  4. driver.normalize_output(pr) ──────────┘  EGRESS            │
  │         → fuko Review Signal v1 (markers on threads)           │
  └──────────────────────────────────────────────────────────────┘
                         │                         │
                         ▼                         ▼
                 Store (pluggable)        canonical signals
            Postgres+pgvector | sqlite-vec     consumed by the
            (in user's own infra)          address-pr-reviews skill
```

Two seams, symmetric:

- **Ingress** — unified config → backend-specific config. `driver.build_env()`.
- **Egress** — backend output → one canonical schema. `driver.normalize_output()`.

The **knowledge base is the constant**: steps 1–2 don't know which backend/model
runs in 3–4, and only the driver knows the backend's injection seam and output shape.

## `.fuko.toml`

Committed to the repo. The single surface an engineer edits. Secrets are never in
this file — each provider preset declares the env var that holds its key. TOML
(not YAML) so it parses with the stdlib `tomllib` — no new dependency, per the
stdlib-first rule — and matches PR-Agent's own config convention.

```toml
[review]
backend = "pr-agent"              # which driver
tools = ["review", "improve"]     # backend tools to run

[review.model]
provider = "zai-coding"           # a preset NAME (see presets)
name = "glm-5.2"

[knowledge]
store = "postgres"                # postgres | sqlite-vec

# [knowledge.object_store]        # when store = "sqlite-vec"
# backend = "s3"                  # s3 | r2 | file
# bucket = "my-fuko-kb"
# key = "my-repo/knowledge.db"
# creds_env_prefix = "FUKO_S3"    # FUKO_S3_ACCESS_KEY_ID, ...

[embedding]
provider = "ollama"               # offline default (self-hosted runners)
model = "bge-m3"
base_url = "http://localhost:11434/v1"
# On SaaS runners, prefer a remote embedding provider (see Storage notes).
```

Switching the review model is a two-line edit plus a key:

```toml
[review.model]
provider = "ollama"
name = "qwen2.5-coder:32b"
```

No dunder keys, no knowing the coding-vs-paas endpoint, no max-tokens landmine —
all absorbed by the driver.

## The `fuko review` runner

Replaces today's 195-line `pr-review.yml`. The GitHub workflow shrinks to a trigger:

```yaml
- run: fuko review --pr-url ${{ github.event.pull_request.html_url }}
  env:
    GITHUB_TOKEN: ${{ steps.app-token.outputs.token || github.token }}
    ZAI_KEY: ${{ secrets.ZAI_KEY }}
```

Pipeline:

1. **Resolve config** — load `.fuko.toml`, resolve the provider preset, read keys
   from the declared env vars.
2. **Build knowledge** — fetch the PR's changed files + body (GitHub API,
   paginated), `Store.query(repo, files, pr_body)`, format top-k into the injection
   blob. Backend-agnostic.
3. **Translate (ingress)** — `driver.build_env(preset, model, knowledge, tools)`
   → the backend-specific config (env dict for PR-Agent).
4. **Invoke** — `driver.invoke(pr, env, tools)` runs the backend. For PR-Agent this
   is its **Docker image** (`docker run --rm -e <KEY>... <image> --pr_url <url>
   <tool>`), once per tool. PR-Agent is *not* pip-installable — its pinned deps are
   mutually unsatisfiable (`google-cloud-storage==2.10.0` vs
   `google-cloud-aiplatform==1.154.0` needing `>=3.10.0`), so the official image is
   the only reliable runtime. Env vars are forwarded by name (`-e KEY`) so secrets
   and multiline `extra_instructions` stay out of argv. The image + extra
   `docker run` args are configurable in `[review]`.
5. **Normalize (egress)** — `driver.normalize_output(pr)` reads back the bot's
   posted comments, maps them to fuko Review Signal v1, and injects markers.

The same binary runs from GitHub Actions, GitLab CI, or locally; the runner needs
only Docker (for PR-Agent) — not a working PR-Agent pip environment.

### Consumer interface: `fuko signals`

`fuko signals --pr-url <url>` emits the canonical Review Signal v1 list for a PR as
JSON, normalizing **every recognized reviewer** (PR-Agent, Copilot, CodeRabbit) via
`normalizers.collect_signals`. This is the deterministic seam a downstream
addresser consumes — one schema instead of per-vendor sniffing. Detection is per
comment: PR-Agent by format (it posts under whatever token runs it), Copilot and
CodeRabbit by author. PR-Agent carries declared severity/category; free-form
reviewers get an inferred best-effort mapping (`severity_source` records which).

## Contracts

### ProviderPreset (data, not code)

Adding a model provider = adding an entry. This is where "natural selection" lives.

```python
@dataclass(frozen=True)
class ProviderPreset:
    litellm_prefix: str            # "openai/", "ollama/", "anthropic/"
    base_url: str | None           # provider endpoint, or None for SDK default
    key_env: str | None            # env var holding the API key, or None (local)
    quirks: dict[str, object]      # e.g. {"custom_model_max_tokens": 128000, "ai_timeout": 300}
```

Initial presets (the ones we actually use):

| preset       | litellm_prefix | base_url                                  | key_env        | quirks                          |
|--------------|----------------|-------------------------------------------|----------------|---------------------------------|
| `zai-coding` | `openai/`      | `https://api.z.ai/api/coding/paas/v4`     | `ZAI_KEY`      | max_tokens 128000, timeout 300  |
| `ollama`     | `ollama/`      | `http://localhost:11434`                  | —              | —                               |
| `openai`     | `openai/`      | (SDK default)                             | `OPENAI_KEY`   | —                               |
| `anthropic`  | `anthropic/`   | (SDK default)                             | `ANTHROPIC_KEY`| —                               |

### ReviewBackend driver

```python
class ReviewBackend(Protocol):
    name: str

    # capabilities — the runner degrades gracefully on what a backend lacks
    supports_inline_suggestions: bool
    injection: Literal["extra_instructions", "api", "prompt"]

    def build_env(self, preset: ProviderPreset, model: ModelConfig,
                  knowledge: str, tools: list[str]) -> dict[str, str]: ...

    def invoke(self, pr_url: str, env: dict[str, str]) -> InvokeResult: ...

    def normalize_output(self, pr: PRRef) -> list[ReviewSignal]: ...
```

The **pr-agent driver** (first and only implementation for now):

- `build_env` emits dynaconf **dunder** env vars (`CONFIG__MODEL`,
  `CONFIG__CUSTOM_MODEL_MAX_TOKENS`, `CONFIG__AI_TIMEOUT`, `OPENAI__API_BASE` /
  `OLLAMA__API_BASE`, `OPENAI__KEY`, `PR_REVIEWER__EXTRA_INSTRUCTIONS`,
  `PR_CODE_SUGGESTIONS__EXTRA_INSTRUCTIONS`, tool flags). This is the tested home
  for every landmine currently hand-tuned in YAML.
- `invoke` runs PR-Agent as a subprocess (`python -m pr_agent.cli --pr_url <url>
  <tool>`), with PR-Agent as an **optional, pinned** extra: `pip install
  fuko-pr[pr-agent]`. Subprocess (not `import`) keeps coupling loose and isolates us
  from PR-Agent's version churn.
- `normalize_output` reads back the comments PR-Agent posted (author = the bot/app),
  maps them to signals, and edits each thread to embed a marker.

**Do not build a second backend driver until a real one exists.** The contract is
defined independently so PR-Agent's shape doesn't leak into "generic".

### Store

```python
class Store(Protocol):
    def ingest(self, repo: str, items: list[Learning]) -> tuple[int, int]: ...  # (inserted, skipped)
    def query(self, repo: str, files: list[str], pr_body: str,
              query_text: str | None, top_k: int) -> list[Learning]: ...
    def forget(self, repo: str, *, id=None, source=None, all=False) -> int: ...
```

Today's `ingest.py` / `retrieve.py` become `PostgresStore`. The data is tiny
(hundreds–low-thousands of learnings), so brute-force cosine is sub-millisecond;
pgvector/HNSW is not required by scale, only convenient where Postgres already runs.

**Principle: the store always lives in the *user's own* account/infra** (their
Postgres, their bucket). fuko never hosts a multitenant KB — that would rebuild the
lock-in we're escaping. "Own your knowledge base" is literal.

### fuko Review Signal v1

The canonical egress schema. The `address-pr-reviews` consumer reads only this.

```jsonc
{ "v": 1,
  "id": "fk_a1b2",                 // stable id for dedup / threading
  "file": "src/auth/login.py", "line": 42, "end_line": 48,
  "severity": "high",              // info | low | medium | high | critical
  "severity_source": "declared",   // declared | inferred (mapping is lossy; flag it)
  "category": "bug",               // bug | security | perf | style | test | docs | design
  "title": "…", "body": "…",
  "suggestion": true,              // is there an applyable code block?
  "thread_url": "…",
  "backend": "pr-agent", "model": "glm-5.2",
  "kb_refs": ["resolved_thread:…"] // which learnings drove this finding
}
```

Transport: an invisible HTML-comment marker injected per thread (renders nothing on
GitHub and GitLab, survives round-trips):

```
<!-- fuko-signal:v1 {"id":"fk_a1b2","severity":"high",...} -->
```

The consumer `grep`s `fuko-signal:v1` and parses the JSON; thread resolution state
still comes from the API, but *what each thread is* becomes self-describing.

## Storage backends

### Postgres + pgvector (self-host / managed)

- **Homelab / self-host:** the current sidecar + pgvector. Unchanged.
- **Managed (Neon / Supabase free tier):** direct connection from the runner in
  "local mode" — the FastAPI sidecar is optional. One connection-string secret.

### sqlite-vec + object storage (server-free flagship)

For SaaS runners (GitHub-hosted, GitLab SaaS) that should not have to run a server.

- The KB is a **single sqlite-vec file** (`pip install sqlite-vec` bundles the
  extension binary). A few MB at realistic scale.
- The file lives in **object storage (S3 / R2)** in the *user's own* bucket.
- **Read path** (`fuko review`): download the `.db`, query in-process.
- **Write path** (`/remember`, resolved-thread sweep): download → mutate → upload.

**Write-back concurrency** — concurrent writers can clobber each other
(last-upload-wins → lost writes). Resolve with **conditional PUT**: download with the
object ETag, mutate, `PUT` with `If-Match: <etag>`; on `412 Precondition Failed`,
re-download and retry. Both S3 and R2 support conditional writes; no lock service
needed. Correct enough for the low write volume of PR review.

**Embeddings on SaaS runners** — the offline-Ollama default assumes a machine that
has Ollama, which is true in a homelab and false on a GitHub-hosted runner. The
server-free path therefore pairs with a **remote embedding provider**
(e.g. OpenAI `text-embedding-3-small`; embeddings cost pennies). Offline Ollama
remains the default for self-hosted runners. Document this so the two "server-free"
choices don't contradict each other.

> Embedding dimension: changing the embedding model changes the vector dimension.
> The store must handle a dimension change without "drop the table by hand" — probe
> the dim at startup and size/migrate the column (Postgres) or recreate the virtual
> table (sqlite-vec) accordingly.

## Deployment modes

| Mode                         | Store                         | Sidecar  | Embeddings        |
|------------------------------|-------------------------------|----------|-------------------|
| Homelab / self-host          | Postgres+pgvector             | yes      | local Ollama      |
| SaaS runners, server-free    | sqlite-vec in S3/R2           | no       | remote provider   |
| SaaS runners, managed DB     | managed pgvector (Neon/Supabase) | optional | remote provider |

## Sequencing

0. **Define contracts** (this doc) — config schema, ProviderPreset, ReviewBackend,
   Store, Signal v1.
1. **`fuko review` runner + pr-agent driver** — port today's workflow logic into
   tested code; provider presets with the gotchas baked in. Unit-test
   "preset + model → env dict".
2. **Egress normalization** — pr-agent `normalize_output` + signal markers. **Early
   incremental win:** ship *read-only* normalizer drivers for CodeRabbit/Copilot
   (no invocation, just output → Signal v1) so the `address-pr-reviews` skill gets
   lighter and more reliable before the full runner exists.
3. **sqlite-vec Store + object-storage backend** — conditional-PUT write-back; fix
   the embedding-dimension migration edge.
4. **OSS polish** — de-vendor defaults (provider-neutral first, z.ai as one
   example), working clean-clone quickstart, S3 setup guides, CONTRIBUTING,
   architecture docs.

Backends grow by adding drivers; providers grow by adding `ProviderPreset` data.
Edge cases are left to natural selection — the architecture stays flat.

## Non-goals

- fuko does **not** become a standalone reviewer that calls the chat model itself —
  that reimplements PR-Agent.
- fuko does **not** host a multitenant knowledge base.
- No uniform "every backend is identical" API — backend abstraction is leakier than
  the model one; capability flags + graceful degradation instead.
