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
- **No inline comments** unless explicitly requested. Prefer self-documenting code.
- **Stdlib-first.** Only add a dependency when the stdlib genuinely cannot do it.
- **Keep model and migration in sync.** If the embedding dimension changes, update
  both `FUKO_EMBED_DIM` and the `vector()` column in `migrations/001_init.sql`
  (drop & recreate the table — the migration is `IF NOT EXISTS`).
- **Embeddings are provider-agnostic.** z.ai exposes no `/embeddings` endpoint, so
  embeddings always go through an OpenAI-compatible HTTP endpoint. Chat (PR-Agent)
  is configured separately.

## Commands

- Install (dev): `pip install -e ".[dev]"`
- Tests + coverage gate: `pytest`
- Lint + docstring gate: `ruff check .`
- Format: `ruff format .`
- Run sidecar: `fuko serve`
