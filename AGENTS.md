# AGENTS.md

Guidance for AI coding agents working on fuko-pr.

## Project

fuko-pr is a thin RAG knowledge sidecar for PR-Agent (see `README.md`).
Python 3.11+, FastAPI + psycopg3/pgvector, embeddings via any OpenAI-compatible
endpoint (default: local Ollama `bge-m3`).

## Rules (mandatory)

- **Coverage gate: >= 80%.** Line coverage of the `sidecar` package must stay at or
  above 80%, enforced by `pytest` (`--cov=sidecar --cov-fail-under=80`). Do not lower
  `fail_under`, and do not commit code that drops coverage below the threshold.
- **Docstrings are mandatory.** Every module, class, and public function/method has a
  docstring (Google convention). Enforced by `ruff check` (pydocstyle `D` rules).
  Private helpers (leading underscore) and code under `tests/` are exempt.
- **Prefer self-documenting code; comment the non-obvious _why_.** Don't narrate
  _what_ the code does (that's redundant with readable code) — but a short comment
  is welcome where it captures a non-obvious rationale the code can't: an ordering
  constraint, a subtle invariant, a fail-safe direction, or a "looks wrong but
  isn't". When the explanation is really about a data field's meaning, prefer a
  `Field(description=...)` (structured, shows up in the schema) over a loose comment.
- **Stdlib-first.** Only add a dependency when the stdlib genuinely cannot do it.
- **Embedding-dimension changes are handled automatically.** If the embedding model's
  dimension changes, the store re-embeds every learning and rebuilds the vector
  column/table on next startup (Postgres) or next access (sqlite-vec) — a one-time,
  potentially slow cost. Do not reintroduce a manual drop-&-recreate step.
- **Embeddings are provider-agnostic.** z.ai exposes no `/embeddings` endpoint, so
  embeddings always go through an OpenAI-compatible HTTP endpoint. Chat (PR-Agent)
  is configured separately.
- **HTML lives in `sidecar/web/`.** Browser-facing pages are served under `/ui` by
  that package; no other module builds HTML. Adding one is a registry entry plus a
  module — the pattern and its conventions (route fetches and degrades, render
  function is pure, everything escaped through `components`, no build step) are in
  [`docs/web-ui.md`](docs/web-ui.md).

## Commands

- Install (dev): `pip install -e ".[dev]"`
- Tests + coverage gate: `pytest`
- Lint + docstring gate: `ruff check .`
- Format: `ruff format .`
- Run sidecar: `fuko serve`
