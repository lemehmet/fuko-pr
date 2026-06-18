# Contributing to fuko-pr

Thanks for your interest! fuko-pr is a small, deliberately stdlib-first Python
project. The full conventions live in [`AGENTS.md`](AGENTS.md); this is the short
version.

## Dev setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # 3.11–3.13
pip install -e ".[dev,sqlite]"        # sqlite extra adds sqlite-vec + boto3
make up                               # pgvector + Ollama via docker compose (optional)
make ollama-pull                      # pull bge-m3 into the Ollama container
```

`make test`, `make lint`, `make up` are conveniences; see the `Makefile`.

## The checks (all must pass)

```bash
ruff check .
ruff format --check .
pytest                                # ≥ 80% coverage over `sidecar` is enforced
```

- **Coverage ≥ 80%** over the `sidecar` package (`--cov-fail-under=80`). Don't lower
  the gate. The DB- and embeddings-backed tests are skipped unless
  `FUKO_DATABASE_URL` (and an embeddings backend) are set; CI provisions pgvector +
  Ollama, so coverage is measured with them running.
- **Docstrings** on every module/class/public function (Google convention, enforced
  by ruff `D`). Private helpers and `tests/` are exempt.
- **No inline comments** unless explicitly requested — prefer self-documenting code.
- **Stdlib-first** — add a dependency only when the stdlib genuinely can't do it.

## Architecture

See [`docs/design.md`](docs/design.md) for the contracts (config schema, the
`ReviewBackend` and `Store` protocols, provider presets, and the Review Signal v1
spec). New behavior should fit those seams: a new model provider is a
`ProviderPreset` entry (data), a new reviewer is a backend driver, a new store is a
`Store` implementation.

## Pull requests

1. Branch off `main` (don't push to `main` directly).
2. Keep PRs focused; include tests for new behavior.
3. Ensure the checks above pass locally.
4. Open a PR; address review feedback (including bot reviewers) before merge.

## License

By contributing you agree your contributions are licensed under
[Apache-2.0](LICENSE).
