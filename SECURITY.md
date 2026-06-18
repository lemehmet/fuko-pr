# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's **"Report a vulnerability"**
(Security ▸ Advisories) on this repository, or by email to the maintainer. Do not
open a public issue for an undisclosed vulnerability. We aim to acknowledge within
a few days and will coordinate a fix and disclosure timeline with you.

## How fuko-pr handles secrets

- **Keys live in the environment, never in config.** `.fuko.toml` selects a
  provider; each provider preset declares the *name* of the env var that holds its
  key. `.fuko.toml` and `.env` are git-ignored.
- **Secrets stay out of process arguments.** When invoking the reviewer container,
  env vars are forwarded by name (`docker run -e KEY`), so values are read from the
  environment rather than placed on the command line.
- **Secrets stay out of logs and signals.** The review workflow logs only counts,
  not injected knowledge; Review Signal markers carry machine fields only, never
  arbitrary comment text that could leak internal links.
- **Your knowledge base is yours.** It lives in your Postgres or your S3/R2 bucket;
  fuko never hosts a multitenant store. Keep object-storage buckets private and
  scope credentials to the single bucket/key.

## Operational notes

- Fork-PR safety: the provided workflows gate self-hosted jobs behind a same-repo
  guard so fork PRs never run on your runners.
- Knowledge is attacker-influenceable (via `/remember`), so it is treated as
  untrusted input to the reviewer. `fuko review` forwards it to the reviewer
  container through an environment variable — not via GitHub Actions output and not
  on the command line — so it cannot inject into the Actions runner or process
  arguments. The amount injected is bounded by the configured retrieval `top_k`.
  (It does become part of the model's instructions, which is its intended use.)
