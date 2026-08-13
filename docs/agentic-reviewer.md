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

- **Read-only agent.** Headless mode cannot prompt for permissions, so the
  tool allowlist (`Read,Grep,Glob`) is the entire tool surface: no Bash, no
  file writes, no network tools. Repository code is never executed — git does
  not run repo-shipped hooks on clone/fetch/checkout, and the reviewer treats
  the tree purely as data.
- **Prompt-injection posture.** The diff and repository contents are declared
  untrusted in the strategy prompt; instruction-like text inside them
  (including text addressed to AI reviewers) must be ignored and *reported as
  a security finding*. The blast radius of a successful injection is bounded
  by the tool surface: wrong review text, not actions.
- **Credential hygiene.** The agent subprocess environment strips
  `GITHUB_TOKEN` / `GITHUB__USER_TOKEN` / `FUKO_GITHUB_*`; it receives only
  the Anthropic credentials. The checkout's fetch auth rides in `GIT_CONFIG_*`
  environment (not argv, not the remote URL), scoped to the one fetch.

## Configuration

```toml
[review]
backend = "agentic"        # global until fuko-pr #99 lands per-model backend

[[review.models]]
provider = "anthropic"     # key from env ANTHROPIC_KEY
name = "claude-sonnet-5"
# extra_instructions = """
# ...per-entry steering, same field the pr-agent backend uses (#98)."""
```

Current limits, on purpose:

- **Runner prerequisite:** the `claude` CLI must be installed on the runner
  (`npm install -g @anthropic-ai/claude-code`); a missing binary fails that
  branch with a clear message instead of throttling.
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
