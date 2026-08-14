# The agentic reviewer (`backend = "agentic"`)

fuko's own PR reviewer, driven as a review backend alongside `pr-agent`. Instead
of sending one compressed diff prompt to a model, it gives the model a real
checkout of the PR head plus read-only code-navigation tools and requires it to
**verify findings by reading the surrounding code** before reporting them.

## Why it exists

Same-pipeline reviewers correlate by construction: every model behind one
single-shot harness sees the same diff presentation under the same prompt
scaffold, so they converge on the same salient findings (measured on a consumer
repo: 62–67% overlap between same-harness instances, versus 83% uniqueness for
a reviewer with different context construction). Model diversity alone cannot
fix that — context-construction diversity can. This backend is the
different-context reviewer, and it doubles as the proof that the
`ReviewBackend` seam is real (it is the second driver behind it).

## Architecture

```
sidecar/reviewer/            the reviewer (harness-agnostic core)
  checkout.py                PR head checkout + API diff fetch
  prompt.py                  review strategy + JSON output contract   <- the value
  harness.py                 agent runtimes (headless Claude Code today)
sidecar/backends/agentic.py  the fuko driver (ReviewBackend protocol)
```

The flow, per branch:

1. **invoke** — fetch PR metadata and unified diff from the GitHub API; fetch
   `pull/N/head` at depth 1 into a temp directory; run the agent over that
   checkout with the review prompt; parse its final message as structured
   findings; stash them in memory. Nothing is posted.
2. **normalize_output** — turn the stash into Review Signals (v1) carrying the
   branch's true `role`, and post ONE pull-request review under the branch's
   own token: anchored findings as inline comments with their invisible
   markers (and the visible A/B label) attached **at creation**, unanchored
   findings and the summary in the review body.

This inverts pr-agent's egress: there is no published-markdown scraping and no
marker PATCH pass, because the output is born in fuko's canonical format. A
finding that never reaches GitHub is never returned as a signal (a failed post
degrades to zero findings, not phantom ones).

## Output contract

The agent must answer with a single JSON object (`sidecar/reviewer/prompt.py`
is authoritative): a `summary` plus up to 10 `findings`, each with
`file`/`line` (new-file side, inside a diff hunk), `severity`, `category`,
`title`, `body`, `evidence` (what it read to verify — paths/symbols beyond the
hunk), and `confidence`. Low-confidence findings are dropped before posting and
counted in the review body ("N withheld"), so the agent has a pressure valve
that is not "report it anyway".

## Security model

The reviewed checkout is an **arbitrary contributor's code**, and the reviewer
runs on a self-hosted runner. That makes one non-obvious vector the dominant
concern.

### The working directory is the attack surface

Claude Code loads project configuration from its working directory —
`.claude/settings.json` **hooks** (arbitrary commands) and `.mcp.json` servers
— and a `-p` session shows no workspace-trust dialog to gate it. This is
documented behavior, and it was **verified against Claude Code 2.1.232**: a
`SessionStart` hook placed in the working directory executed. A hostile PR
adding `.claude/settings.json` would therefore have run code on the runner.

Mitigations, each independently sufficient for its vector:

1. **The agent never runs from the checkout.** `cwd` is a clean scratch
   directory; the checkout is mounted as an additional readable root
   (`--add-dir`). Settings, hooks, and MCP servers load from the working
   directory, not from `--add-dir` roots — verified: the same hook does not
   fire under this arrangement, while `Read`/`Grep`/`Glob` still work.
2. **`--setting-sources user`** — project settings are never a source.
3. **`--settings '{"disableAllHooks":true}'`** — hooks off regardless.
4. **`--strict-mcp-config`** — no MCP server starts that was not configured
   here (none is).
5. **The checkout is stripped** of `.claude/`, `.mcp.json`, and friends before
   the run (`strip_agent_config`), because skills and subagents *are* read from
   additional roots. The files remain in the diff, so a PR that edits them is
   still reviewable — and worth flagging, which the strategy prompt asks for.

`--bare` would harden further, but its help is explicit that it never reads
OAuth or the keychain — it would force API-key billing and break subscription
auth, so it is deliberately not used.

### Everything else

- **Read-only tool surface.** Headless mode cannot prompt for permissions, so
  the allowlist (`Read,Grep,Glob`) *is* the tool surface: no Bash, no writes,
  no network. Repository code is never executed — git does not run
  repo-shipped hooks on clone/fetch/checkout either.
- **Prompt-injection posture.** The diff and repository contents are declared
  untrusted in the strategy prompt; instruction-like text inside them
  (including text addressed to AI reviewers) must be ignored and *reported as
  a security finding*. The blast radius of a successful injection is bounded
  by the tool surface: wrong review text, not actions.
- **Credential hygiene.** The agent subprocess environment strips
  `GITHUB_TOKEN` / `GITHUB__USER_TOKEN` / `FUKO_GITHUB_*` and every Anthropic
  credential, then injects exactly the one its auth mode uses (below). The
  checkout's fetch auth rides in `GIT_CONFIG_*` environment (not argv, not the
  remote URL), scoped to the one fetch.

## Authentication

Two modes, chosen per model entry with `auth`:

```toml
[[review.models]]
provider = "anthropic"
name = "claude-sonnet-5"
auth = "subscription"   # "auto" (default) | "subscription" | "api-key"
```

- **`subscription`** — the agent runs as the runner's own logged-in Claude
  session; no key is passed. On a CI runner, generate a long-lived token with
  `claude setup-token` and export it as `CLAUDE_CODE_OAUTH_TOKEN`; an
  interactive login works too (credentials live under `HOME` /
  `CLAUDE_CONFIG_DIR`, both of which pass through to the agent untouched).
  A logged-out runner is caught by a preflight (`claude auth status`) and
  fails that branch immediately, before any clone.
- **`api-key`** — `ANTHROPIC_KEY` (the preset's env var) is passed through as
  `ANTHROPIC_API_KEY`, with `ANTHROPIC_BASE_URL` for a gateway. A missing key
  is a config error, raised before the run.
- **`auto`** (default) — api-key when `ANTHROPIC_KEY` is set, else
  subscription.

**Why the modes are mutually exclusive at the environment level:** Claude
Code's credential precedence is `ANTHROPIC_AUTH_TOKEN` > `ANTHROPIC_API_KEY` >
`apiKeyHelper` > `CLAUDE_CODE_OAUTH_TOKEN` > the interactive login. An ambient
`ANTHROPIC_API_KEY` therefore *silently* moves billing off a subscription —
verified: with the key exported, `claude auth status` reports
`apiKeySource: ANTHROPIC_API_KEY` and `subscriptionType: null`. So all
Anthropic credentials are stripped from the inherited environment and each
mode injects only its own. Pin `auth` explicitly on any runner where both
exist.

An exhausted plan window ("You've hit your session/weekly limit") is
classified as throttling, so the branch fails over to a backup entry; an
authentication failure is deliberately **not**, because failing over would
burn every provider in the pool on what is a one-line runner fix.

## Configuration

```toml
[review]
backend = "agentic"        # global until fuko-pr #99 lands per-model backend

[[review.models]]
provider = "anthropic"
name = "claude-sonnet-5"
auth = "subscription"      # or "api-key" (ANTHROPIC_KEY); see Authentication
# extra_instructions = """
# ...per-entry steering, same field the pr-agent backend uses (#98)."""
```

Current limits, on purpose:

- **Runner prerequisite:** the `claude` CLI must be installed on the runner
  and authenticated per the mode above; a missing binary fails that branch
  with a clear message instead of throttling.
- **`anthropic` preset only.** The headless-Claude harness authenticates via
  `ANTHROPIC_API_KEY`. Other model families arrive with an OSS agentic harness
  implementing the same `run_review` signature — the strategy and driver do
  not change for it.
- **Global `backend` scalar.** Mixing agentic and pr-agent entries in one
  fleet needs per-model backend selection + backend-attributed receipts,
  tracked as #99. Until then a repo opts in wholesale (or dogfoods it solo).
- **One tool.** `review` only; `improve`/`describe` in `[review].tools` are
  ignored for this backend.
- Failover pairs an agentic branch with whatever backups the pool holds; a
  pr-agent backup rescuing an agentic branch works (different harness, same
  receipts attribution) but mixes harnesses within one branch — #99's
  per-model backend field is also the place that gets a per-entry say.

## Cost & pacing

An agentic review is multi-turn (up to 50 tool turns) against a frontier
model: expect noticeably higher per-review cost and latency than a single-shot
pr-agent pass. `[review].tool_timeout` bounds wall-clock exactly as it does
for pr-agent containers (a timeout classifies as throttle-class and fails
over). Trial-seat first: run it as a non-gating `role = "trial"` entry and
score marginal uniqueness receipts-only before letting it gate.
