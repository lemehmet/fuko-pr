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

```text
sidecar/reviewer/            the reviewer (harness-agnostic core)
  checkout.py                PR head checkout + API diff fetch
  prompt.py                  review strategy + JSON output contract   <- the value
  harness.py                 agent runtimes (headless Claude Code today)
  ledger.py                  per-seat findings + coverage policy (carry in / settle)
  transcript.py              scrubbed, streaming capture of the harness event feed
  transcript_client.py       how a finished transcript leaves the runner
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

## Rounds remember: the open-findings ledger

Measured on mepro, **86% of findings are one-shot** — reported once, never
re-noticed. A finding the author does not act on in the round it appears is
simply lost, because the next round has no idea it was ever made. The ledger
(#156, epic #160) closes that: when a review-state store is configured, each
round loads its own seat's still-open findings, renders them into the prompt
behind the `prior-review-state` fence, applies the verdicts the agent returns
on them, and records what it published as the next round's open ledger.

The rules that matter:

- **Keyed per seat.** `(repo, pr, seat)`, where the seat is the branch's slot
  label (`dorian`, `gray`) — model-agnostic, so swapping the model behind a
  seat keeps its ledger. There is deliberately no shared cross-seat ledger: it
  would raise fleet coverage at the cost of the independent second opinion that
  is the reason for running two seats.

  The runner resolves one lane per branch for the whole run, and they are always
  distinct. A branch keeps its slot when the slot is its own; when two branches
  project onto the same slot — the same `token_env`, or names differing only in
  case, since the slot is lowercased — both fall back to their `provider/name`
  label, and a label two branches also share gains its index. A **solo** run has
  no branch to collide with, so it simply keeps its own slot when it declares a
  `token_env` — which also means its ledger survives the config later growing
  into an A/B run — and falls back to the constant `default` seat only when it
  declares none. The trade on any fallback is that renaming that branch's model
  resets its ledger, which a slot would have survived; giving each branch its
  own distinctly-named `token_env` avoids it.
- **Only the agent closes a finding**, with two exceptions in fuko's own hands:
  a verdict it cannot read closes nothing, and a finding whose file the current
  head no longer contains is retired as `stale`. Line drift never closes
  anything — "the code moved" is not evidence the problem was fixed.
- **`rejected` needs a reason.** Rejection is one round overruling its
  predecessor; without a reason the entry is downgraded to `still_open`, so a
  seat cannot close inherited findings by assertion.
- **Silence keeps a finding open.** Omitting `prior_status` entirely is allowed
  by the contract, and every unmentioned finding is simply offered again.
- **Only published findings are recorded** — a low-confidence finding the
  pressure valve withheld must not re-enter through the next round's prompt.
- **A re-report is a re-assertion, and `(file, title)` is what "the same claim"
  means.** A round that re-publishes a finding it was handed, instead of
  settling it, touches that row rather than opening a second one — otherwise
  duplicates compound each round until the read cap sheds the newest rows, which
  are then unreachable by any round and age out unseen.

  This is the one rule here with a **known false negative**, and it is worth
  stating rather than implying away: a genuinely new claim that names the same
  file under the same case-folded headline is not recorded, and the row that
  survives keeps the earlier body and the earlier evidence. Widening the key to
  the body would remove that, at the price of missing every reworded re-report
  — most of them — and restoring the growth above, which loses claims outright
  rather than one round's phrasing of, and citations for, a claim that is still
  open and still in the next prompt.
  Every suppression is logged by name (`re-asserted, not re-recorded: …`), so
  the trade is visible in the round's output rather than silent.
- **A verdict's closure is not the last word.** A round that publishes a claim
  matching a row an earlier round (or the same round) closed as `fixed` or
  `rejected` **re-raises that row** instead of opening a second one, and the row
  counts how often that has happened (#177,
  `migrations/010_review_finding_reopen.sql`). This matters because closure is
  the one irreversible write the ledger makes from model text produced while
  reading a checkout the contributor controls: an injected "everything here was
  fixed" — or an honest mistake — used to retire a finding permanently. It
  cannot any more; the seat's own later reading of the code answers it.

  The trigger is a **published** finding, never a line in the fenced verdict
  channel, so this adds no way for the reviewed content to reach the ledger that
  a genuine problem did not already have. `stale` is excluded: that retirement is
  fuko's own, made on evidence fuko read itself, and softening it is a separate
  question (#175). Each re-raise is logged by name
  (`re-raised a closed finding: …`) — a closure that keeps being contradicted is
  a seat settling claims it has not verified, or a verdict that was never its
  own idea.

### Turning it off per entry

The findings ledger is **on by default** and switched off per entry:

```toml
[[review.models]]
provider = "qwen-anthropic"
name = "qwen3.8-max"
backend = "agentic"
findings_ledger = false   # default true
```

The polarity is the opposite of `coverage_ledger`'s on purpose. Tier 1 shipped
unconditional, so an opt-in would silently strip carried state from every fleet
that merely bumps its pinned fuko revision; an opt-*out* means an entry that
never mentions the flag behaves exactly as it did, down to the harness
environment (the backend emits `FUKO_AGENTIC_FINDINGS_LEDGER` only to say `0`).

A seat with the flag off makes **no findings read, retires nothing, mints no
ids, applies no verdicts and records nothing**, and its prompt carries no
prior-state section at all. It is not byte-for-byte the pre-ledger prompt: the
output contract still asks for `examined` and `prior_status` unconditionally. It
is asked for in both arms, so nothing state-dependent separates them — which is
what makes a genuinely stateless arm expressible, and which
is why the flag exists (#159): before it, both arms of a stateful-vs-stateless
A/B carried findings. Turning it back on **self-heals**: the first flag-on round
retires rows whose files the head no longer carries, against the checkout it
already holds. That recoverability is also why the findings retirement is gated
while the coverage expiry beside it is not — expiry consumes *this* round's
delta, which no later round can reconstruct, whereas retirement is checked
against a current tree every later round has. And a control that kept writing
`stale` closures to the table it is meant to be ignoring would not be stateless
in the first place.

Storage is Postgres-only (`FUKO_DATABASE_URL`, `migrations/009_review_state.sql`)
and entirely best-effort: with no store, an unreachable one, or a sqlite-vec
deployment, every ledger call is a no-op and the round builds exactly the prompt
it built before the ledger existed.

The runner reaches that store the same way it reaches every other piece of
shared state — the **sidecar over HTTP** when `FUKO_URL` is set, else the local
Postgres (#171). A review runner in the homelab deployment holds `FUKO_URL` +
`FUKO_TOKEN` and no connection string, so before this seam existed every ledger
call took the no-op path and each round was byte-for-byte a pre-ledger one.
`sidecar/review_state_client.py` chooses the transport; the endpoints are
`/rs/findings`, `/rs/round`, `/rs/settled`, `/rs/coverage` and their writes,
behind the same bearer-token dependency as `/cb/*` and `/rh/*`. Requests carry
`(repo, pr, seat)` on every call — including the id-addressed writes, which the
store matches in SQL, so no request can settle, re-raise or touch a row outside
its own seat's lane. A sidecar that does not answer latches the run onto the
local branch after one timeout rather than paying one per call (#170).

## Rounds aim: the coverage ledger

Measured on mepro, the fleet behaves like a **sampler**, not a reviewer: two
seats on the same head agree on 4.7% of findings, estimated pool coverage is
**~26%**, 64% of all file-touches land on three paths, and one 428KB file was
read **182 times** across 24 seat-runs. Every round re-reads the same few huge
files and draws a different sample. The coverage ledger (#157) is the lever on
that number: each round records the regions it examined, and a later round of the
**same seat** is shown them so its budget goes to surface nobody has covered.

It is off by default and enabled per entry:

```toml
[[review.models]]
provider = "qwen-anthropic"
name = "qwen3.8-max"
backend = "agentic"
coverage_ledger = true    # default false
```

Staged deliberately: coverage state changes *what the reviewer looks at*, so it
is scored on a non-gating (`role = "trial"`) seat's receipts before it reaches a
gating one. A seat with the flag off neither reads nor records coverage and
builds exactly the prompt it built before. It does still *expire* coverage its
delta invalidates — expiry is the one half that runs on every seat, for the
reason given under "Coverage expires; findings survive" below — so a flag-off
seat writes `review_coverage.expired_at` and nothing else.

Coverage is the ledger with the real carry-forward hazard — a wrong recorded
conclusion does not merely mislead the next round, it *suppresses the
re-examination that would have corrected it* — so four rules are load-bearing
rather than stylistic:

- **What was looked at, never what was found to be fine.** A coverage entry
  records the question a round asked of the code (`checked`) and what reading it
  established (`conclusion`), and the strategy forbids a clean bill of health
  outright. The schema can only require the *keys* — `""` satisfies a required
  string — so an entry whose `file`, `checked`, `conclusion` or `evidence` is
  blank is **dropped on the way back out** and logged, rather than rendered as
  though a round had established it. `file` is in that set because it is the key
  expiry matches on: an entry naming no file is one no delta can ever
  invalidate. That shape (a conclusion, no question, no citation)
  is precisely the unfalsifiable clean bill the epic prohibits.
- **Coverage expires; findings survive.** A finding is a *claim* and stays open
  until a round settles it with a reason. A coverage entry is an *assurance*: the
  moment a round's delta touches its file, the tree it described is gone and the
  entry is expired (`review_coverage.expired_at`) before it can reach another
  prompt. This is the one place in the epic that consults the delta at all, and
  it consults it to **invalidate**, never to scope — the round still reviews the
  whole change.

  The delta used is the current base→head diff, which over-expires and is the
  safe error: coverage of a file that appears in that diff never survives a
  round, so what carries is coverage of the *unchanged* surface a round reads to
  verify the diff — the callers, callees and invariants that were being re-read
  182 times. Expiry runs on every seat, flag or no flag, because it can only ever
  remove a stale assurance and gating it would let a seat toggled off and back on
  carry entries no round in between could expire.

  A deleted or renamed-away file never reaches the *parsed file set* expiry
  matches against — `parse_diff` collects a path only from a `+++ b/` header, and
  a deletion emits `+++ /dev/null` while a rename leaves nothing at the old path
  — even though both differ maximally from base. (The raw diff still describes
  both; it is the set derived from it that does not.) So the read path retires
  those against the checkout, the same way the findings ledger retires a finding
  whose file is gone. The residual gap left is a file modified, examined, then reverted to its
  base content: it leaves the diff and is still in the tree, so neither pass
  expires its entry. What makes that survivable is the next rule.
- **Advisory, never binding.** The block is introduced by fixed prose
  (`COVERAGE_ADVISORY`) that says *deprioritise* and names three conditions for
  going back to a region anyway — this round's changes touch it, it is on the
  path of something being verified, or there is concrete reason to doubt what is
  recorded — and states that a recorded conclusion is an earlier round's
  inference, not established fact. There is deliberately no instruction to pass a
  region over: an imperative to that effect is what would convert one round's
  mistake into a permanent blind spot.
- **Per seat, never shared.** Two seats on one PR keep disjoint coverage. Sharing
  would raise fleet coverage — the seats overlap on 45 files — by manufacturing
  exactly the correlation across *different* models that the second seat exists
  to break.

Evidence on a carried row is bounded per row (`MAX_PRIOR_EVIDENCE`) on both
ledgers, and the coverage list is capped newest-round-first at
`MAX_PRIOR_COVERAGE` with the cut stated in-band — including that absence from
the list is not evidence a region is unexamined.

## Scoring the state tiers: `scripts/ab_metrics.py`

Whether carried state actually buys coverage is an empirical question, and it is
scored from receipts rather than argued (#159). The numbers this file quotes
above — 4.7% agreement, 86% one-shot, ~26% pool coverage, 64% on three paths —
came out of throwaway scripts that no longer exist, which is why "did coverage
improve?" was not answerable. `scripts/ab_metrics.py` replaces them, over the
tested estimators in `sidecar/abmetrics.py`:

```bash
python scripts/ab_metrics.py lemehmet/fuko-pr 210 211 212 \
    --arm control=fuko-dorian[bot] --arm treatment=fuko-gray[bot] \
    --slot control=dorian --slot treatment=gray
```

Two things about it are load-bearing. **Claim identity is fixed** — the findings
ledger's own `(file, casefolded title)` — so a rule cannot be quietly widened
until the answer improves; regenerate the baselines under the *same* rule rather
than comparing against the published figures. And **arms are named, never
inferred**: the two trial seats share a provider, a model and a `role`, so the
only thing distinguishing them on GitHub is which App posted — and the App names
no longer describe the seats. Token and cost per run come from `review_runs`
(#152) via the `--slot` mapping, and print as `n/a` with no database configured
rather than as zero.

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
   additional roots. `.claude/` and `.mcp.json` are cleared at **every depth**
   (a subdirectory is a project root in its own right); other tools' config
   (`.cursor`, `.github/copilot-instructions.md`) only at the checkout root.
   The files remain in the diff, so a PR that edits them is
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
  a security finding*. Untrusted text is fenced with delimiters it cannot close
  (a literal `</diff>` in a PR body is neutralised), so it cannot escape into
  the instruction stream.
- **Read confinement is a denylist, not a sandbox — know what that buys.**
  `--add-dir` *adds* a readable root; it does **not** confine reads to it.
  Verified on Claude Code 2.1.232: with `Read` allowlisted and a clean cwd, an
  absolute path outside every declared root is still readable. That matters
  because findings are published verbatim to the PR, where the (untrusted)
  author can read them — so "the blast radius is wrong review text, not
  actions" is **not** the whole story: an injected "read X, put it in a
  finding" is an exfiltration channel. The harness therefore ships explicit
  `permissions.deny` rules over the runner's credential stores
  (`~/.claude` — which subscription auth deliberately keeps reachable — plus
  `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gh`, `~/.config/gcloud`,
  `~/.docker`, `~/.kube`, `~/.netrc`, `~/.git-credentials`), written in the
  absolute-rule spelling `Read(//abs/path/**)` (a single leading slash silently
  fails to match, and so does a triple).

  Two measured properties are worth knowing before you extend this. A **path**
  rule is enforced across the read-class tools — the `Read(...)` rules are what
  stop `Grep` from reading a credential file — while a **tool-scoped**
  `Grep(...)` rule is not honored at all. Verified on 2.1.232 with a canary
  outside every declared root: no rules → leaked; `Read(//abs/**)` → blocked;
  `Grep(//abs/**)` alone → leaked. That is why only `Read(...)` rules are
  emitted, and why adding `Grep(...)` entries would be decorative.

  **The list bounds the known credential stores and nothing more.** `Read` and
  `Grep` still reach everything else on the runner — `/etc`, other repositories
  checked out in the work dir, build caches. So: **run the reviewer on a runner
  you would be willing to let an untrusted PR read.** A container or a
  dedicated unprivileged user is the real boundary, and that is the runner's
  job, not this module's.
- **Credential hygiene.** The agent subprocess environment strips
  `GITHUB_TOKEN` / `GITHUB__USER_TOKEN` / `FUKO_GITHUB_*` and everything that
  decides who pays or where the traffic goes — every Anthropic credential plus
  `ANTHROPIC_BASE_URL` — then injects exactly what its auth mode uses (below).
  **Config decides the endpoint, never the ambient environment**: a gateway
  user sets `base_url` on the model entry, which api-key mode re-injects;
  subscription mode never gets one, since an inherited base URL would point the
  runner's own authenticated session at a foreign host. The checkout's fetch
  auth rides in `GIT_CONFIG_*` environment (not argv, not the remote URL),
  scoped to the one fetch.

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
- **`api-key`** — the preset's env var (`ANTHROPIC_KEY`, or e.g.
  `ANTHROPIC_COMPAT_KEY`) is passed through as `ANTHROPIC_API_KEY`, with
  `ANTHROPIC_BASE_URL` for a gateway. A missing key is a config error, raised
  before the run.
- **`auto`** (default) — api-key when **the preset's own key env var** is set,
  else subscription. Not `ANTHROPIC_KEY` specifically: `_resolve_auth` reads
  `preset.key_env`, which is `QWEN_TOKEN_PLAN_KEY` for `qwen-anthropic`,
  `ANTHROPIC_COMPAT_KEY` for `anthropic-compatible`, and so on. Exporting a
  real `ANTHROPIC_KEY` does not make `auto` resolve to api-key for a gateway
  entry — it resolves to subscription, which for a `requires_base_url` preset
  is refused outright (see below).

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

### A self-hosted gateway: `anthropic-compatible`

Any endpoint that speaks the Anthropic Messages API can host a seat — a LiteLLM
or vLLM in front of local weights, a rented box, a provider without a preset of
its own:

```toml
[[review.models]]
provider = "anthropic-compatible"
name = "Qwen3.8-Flash-Next-GGUF"   # the gateway's slug, verbatim
base_url = "https://llm.example.internal/anthropic"
auth = "api-key"                   # key from ANTHROPIC_COMPAT_KEY
backend = "agentic"
max_context = 262144
role = "trial"
```

`base_url` and `auth = "api-key"` are both **required**, and each is a config
error rather than a default. The preset has no endpoint of its own, so a
missing `base_url` falls back to `api.anthropic.com`; and subscription mode
injects no base URL at all, so it reaches that same endpoint even when the
entry names a gateway. Either way, if the entry's `name` happens to be a slug
Anthropic serves, that is not a failed run but a *successful* one against the
wrong model, published under this entry's label — a substitution the receipt
cannot detect, because the label and the requested model still agree.

The auth rule matters more than it looks, because the realistic way to hit it
is not writing `auth = "subscription"`. It is leaving `auth` at its `auto`
default and forgetting to export the key: `auto` reads a missing key as "this
is a subscription runner", which is correct for every other preset and wrong
for this one. Both refusals therefore run before the auth branch, not inside
it.

The preset deliberately carries no `small_model` quirk, so the entry's own
model serves the harness's background haiku-class and subagent calls too. The
vendor presets name a cheap tier there to keep those calls off an expensive
model; a single-model deployment has no cheap tier, and naming a second slug
would make the gateway swap models mid-review.

## Session transcripts (`FUKO_TRANSCRIPT_DIR`)

The harness reads the CLI's whole NDJSON event feed and folds it away, keeping
only `assistant` and `result` events — so the `user` events, where tool
**results** live (~91% of a review's token spend), were read and discarded in
the same pass. Setting `FUKO_TRANSCRIPT_DIR` tees that feed to
`<dir>/<key>.ndjson` as it streams (#237, epic #236):

```bash
FUKO_TRANSCRIPT_DIR=/var/lib/fuko/transcripts   # unset = capture off
```

Properties worth knowing before turning it on:

- **Off by default, and off means free.** With the variable unset the feed is
  iterated exactly as before — no tee, no per-event cost. There is deliberately
  no default path: this writes a file per agentic branch per push and keeps it.
- **Credentials are scrubbed at capture, irreversibly.** The driver hands the
  transcript the exact values it holds — the checkout's GitHub App token, the
  seat's injected provider credential, and the ambient secrets it strips from
  the harness environment — and each is replaced by `[REDACTED:<VAR>]` before
  any byte reaches disk, including inside a tool result that echoed it. Values
  the driver does not hold are written verbatim: nothing is inferred from a
  string's shape, because an over-broad rule silently corrupts the corpus and
  cannot be undone.

  Exact-value replacement is per **whole line**, and there is exactly one line
  that may not be whole: when `tool_timeout` kills the harness (or the child
  dies mid-write) the pipe yields a trailing line with no newline, which can
  hold a *prefix* of a credential that no exact rule matches (#251). That line
  gets a second pass which redacts a trailing suffix of it that is a proper
  prefix of a known value. It is a no-op on a clean run: a complete NDJSON
  event ends in `}` and no credential begins with one, so the guard costs no
  corpus fidelity and, unlike dropping the line, does not throw away a complete
  final `result` event.
- **Streamed, never buffered.** One line at a time, line-buffered, so peak
  memory does not track transcript size and a run killed at `tool_timeout`
  keeps everything that arrived before the cut.
- **Denied to the reviewing agent.** The destination is added to the harness
  read denylist whenever it is configured — including on runs that do not
  capture, since the archive earlier rounds left behind is the part worth
  reading. The path is canonicalized first (`expanduser().resolve()`), so a
  relative or symlinked destination cannot render a rule that misses what is
  actually written; `/var/lib/fuko/transcripts` becomes the rule
  `Read(//var/lib/fuko/transcripts/**)` in the absolute-rule spelling described
  above. This is not covered by the file mode: the agent is spawned
  as the same uid, so `0600` stops other *users*, not this reader. Without the
  rule a transcript is a durable, cross-repo record of everything past runs
  read, sitting where `Glob` can find it, reachable by an agent whose findings
  are published verbatim to an untrusted PR author. `FUKO_TRANSCRIPT_DIR` is
  stripped with the rest of the `FUKO_` namespace before the spawn, so the path
  is handed over under its own name (`FUKO_TRANSCRIPT_DENY_DIR`) purely to be
  denied.

  **The rule follows the setting, not the files.** It is emitted for whatever
  `FUKO_TRANSCRIPT_DIR` names *right now*, so **unsetting the variable or
  pointing it at a new directory un-denies the corpus already on disk** — the
  files stay where they are, and from the next run on nothing stops the agent
  reading them. Turning capture off for privacy, or moving the destination to a
  bigger disk, is therefore not a retreat: **delete the old directory**, or keep
  its path denied by other means. There is deliberately no memory of previously
  configured destinations — a deny list that accumulated paths nobody could see
  or clear would be its own hazard — so this one is on the operator.
- **Owner-only on disk.** The directory is created `0700` and each file `0600`,
  set as the file is created rather than chmod'ed after. What survives the
  scrub is the reviewed repository as the agent read it — the same content the
  checkout gets a `0700` temp dir for, except a transcript is kept rather than
  deleted, so on a shared runner the durable copy must not be the readable one.
- **Capture never fails a review.** An unwritable destination, a full disk, a
  misconfigured path: one stderr line, an inert transcript, and a review whose
  text, `usage`, `cost_usd`, `turns` and `subtype` are identical to a run with
  no capture at all.
- **The key is the transcript's own.** Minted at run start (`review_runs` is
  inserted afterwards and never returns its id), so later work can key a stored
  blob and an index row on it.

### Shipping them off the runner (`FUKO_TRANSCRIPT_STORE_*`)

A transcript on one runner's disk is only reachable from that runner. Point the
**sidecar** at object storage and every runner's transcript lands in one place,
keyed by the same key capture minted (#238):

```bash
# On the sidecar, not the runner.
FUKO_TRANSCRIPT_STORE_BACKEND=s3          # unset/empty = no store, transcripts stay local
FUKO_TRANSCRIPT_STORE_BUCKET=fuko-transcripts
FUKO_TRANSCRIPT_STORE_PREFIX=transcripts  # optional, inside the bucket
FUKO_TRANSCRIPT_STORE_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com   # R2
FUKO_TRANSCRIPT_STORE_CREDS_ENV_PREFIX=FUKO_S3   # reads FUKO_S3_ACCESS_KEY_ID / _SECRET_ACCESS_KEY / _REGION
```

`backend = file` with `FUKO_TRANSCRIPT_STORE_ROOT=/var/lib/fuko/blobs` is the
serverless variant — one directory of blobs, useful for a single-host fleet and
for trying the path out before there is a bucket.

- **The runner holds no storage credentials.** It ships to the sidecar it
  already has (`FUKO_URL` + `FUKO_TOKEN`) over a dedicated `POST
  /transcripts/<key>`, and the sidecar writes the blob. Nothing new goes into a
  consuming repo's workflow secrets. On a host with no `FUKO_URL` (a laptop
  `fuko review`) the same `FUKO_TRANSCRIPT_STORE_*` variables are read locally
  and the write goes straight to the store.
- **Environment, not `.fuko.toml`.** The deployed sidecar image carries no repo
  checkout, so it has no config file to read — the same reason the embedding
  endpoint is environment-only (#216).
- **Blobs are write-once.** A key is created or refused (`409`), never
  overwritten, so a re-delivery cannot clobber a stored session. Capture keeps
  the local file too — the deployed copy is a second home, not a move — so a
  workflow artifact upload still works.
- **Unconfigured stays working.** No `FUKO_TRANSCRIPT_STORE_BACKEND` means the
  sidecar starts, reviews run, and transcripts are simply absent from storage.
  A runner with no destination at all (no `FUKO_URL`, no local store) does not
  even wrap the sink, so it never attempts an upload. A runner that ships to a
  sidecar whose store is unconfigured attempts once, gets a `503`, and treats
  that as the off state — silently, so staging capture ahead of storage does
  not print a line per run. Any *other* rejection is one stderr line and a
  review identical to one with no capture at all.
- **It costs the review at most one timeout.** The upload happens once, at
  close, on the finished file, streamed off disk in 64 KiB chunks under an
  absolute 120-second body deadline — an order of magnitude above
  `/metrics/run`'s 10 seconds, which is why it is not that endpoint. `httpx`
  has no request lifetime, only per-phase timeouts, so the deadline rides on
  the body stream itself; the phases it cannot interleave with (connect, the
  one write already in flight, the response) are bounded separately, and the
  true worst case is `transcript_client.UPLOAD_CEILING_S` — 160 seconds, which
  holds because the response body is never read (a read to completion is
  bounded per chunk, not in total). There
  is no retry: the blob is write-once, so a retry after a client-side timeout
  would race an upload that may already have landed.
- **A local blob store is denied to the agent.** With the `file` backend on the
  host that runs the harness, the store root is a second, longer-lived copy of
  the transcript corpus, so it goes into the reviewer's read denylist beside
  `FUKO_TRANSCRIPT_DIR` — same reasoning, same rule. The rule is built from the
  **runner process's own** `FUKO_TRANSCRIPT_STORE_*`, because that is the only
  configuration it can see. In the containerized deployment the sidecar's store
  lives behind a container boundary and the question does not arise; but if you
  run a `file`-backend sidecar as a plain process **on the runner host**, export
  the same `FUKO_TRANSCRIPT_STORE_BACKEND` / `_ROOT` to the runner too, or the
  corpus is written where no deny rule reaches it.

### What a run spent its turns on (`review_transcripts`)

Every captured transcript also gets a row in `review_transcripts`
(`migrations/013`, #239), and the run's `review_runs` row gains one nullable
column — `transcript_key` — pointing at it:

| column | what it holds |
| --- | --- |
| `key` | the transcript's own key; names the blob in the store |
| `complete` | whether the feed reached its terminal `result` event |
| `tool_calls` | call counts by tool name, e.g. `{"Read": 182, "Grep": 9}` |
| `tool_result_bytes` | total UTF-8 bytes of tool-result content the run was fed |
| `repeated_read_files` | distinct files read more than once — one file read three times counts **once** |

- **Derived at capture, not from the blob.** The figures are folded out of the
  same lines the tee is already writing, so nothing is re-downloaded and nothing
  is held: peak memory stays one event plus a counter per tool and per distinct
  file read. They are metered off the **scrubbed** text, so a reader that
  recomputes them from the stored object gets the same numbers.
- **A cut-short feed is still indexed**, with `complete = false`. Dropping it
  would bias the corpus towards runs that finished.
- **Nothing is backfilled.** A pr-agent run, and every run predating this, has
  no row and a NULL reference — `migrations/008`'s reasoning about cost applied
  to tools, where a 0 would read as "this run used no tools".
- **The transcript can never cost the metrics row.** The index row is written
  first, in its own transaction; the reference is written only if it landed.
  There is no foreign key, deliberately: the invariant is held by write order,
  and a constraint would let a transcript-side failure reject the run row's
  duration, outcome, attempts and token counts too. A failed capture or upload
  simply records no reference.

The readers (#240, #241) are the rest of the epic; nothing reads the blobs yet.

## Configuration

```toml
[review]
backend = "agentic"        # fleet default; any entry may override it

[[review.models]]
provider = "anthropic"
name = "claude-sonnet-5"
auth = "subscription"      # or "api-key"; see Authentication
# backend = "pr-agent"     # per-entry override (#99, landed)
# extra_instructions = """
# ...per-entry steering, same field the pr-agent backend uses (#98)."""
```

Per-entry `backend` landed with #99 — `ReviewModel.backend`, validated at
config-parse time and resolved per branch by `_backend_for`, with the driver
recorded on the receipt. `[review].backend` is the fleet default an entry may
override, which is what lets one fleet mix harnesses (the correlation fix the
2026-07-31 audit motivated) and what a `role = "trial"` seat on a new driver
rides in on.

Current limits, on purpose:

- **Runner prerequisite:** the `claude` CLI must be installed on the runner
  and authenticated per the mode above; a missing binary fails that branch
  with a clear message instead of throttling.
- **Anthropic-protocol presets only.** The gate is the preset's
  `litellm_prefix`, not its name: any preset whose prefix is `anthropic/`
  qualifies, because the harness is headless Claude Code and it speaks that
  protocol. That is `anthropic` itself, the vendor gateways
  (`qwen-anthropic`, `zai-anthropic`) and `anthropic-compatible` — the model
  behind those endpoints is not Claude. Other model families arrive with an
  OSS agentic harness implementing the same `run_review` signature — the
  strategy and driver do not change for it.
- **One tool.** `review` only; `improve`/`describe` in `[review].tools` are
  ignored for this backend — accepted rather than rejected so a shared list
  keeps working. Worth knowing when sizing a job: the budget arithmetic
  (`fleet_sequential_cost_minutes`) charges every branch `len(tools)`, so an
  all-agentic fleet with `tools = ["review", "improve"]` books twice the
  wall-clock it can actually spend.
- **Failover stays inside a branch's own driver.** `backend` is per entry
  (below), but a pool is not mixed: `_compatible_backups` filters a branch's
  failover targets to entries on the same driver, so an agentic branch fails
  over only to another *agentic* entry. A pr-agent backup cannot rescue an
  agentic branch, and a promotion across drivers is refused too (#132). That
  is a deliberate rule about what a substitute may be, not a missing feature.

## Cost & pacing

An agentic review is multi-turn against a frontier model: expect noticeably
higher per-review cost and latency than a single-shot pr-agent pass.
`[review].tool_timeout` bounds wall-clock exactly as it does for pr-agent
containers (a timeout classifies as throttle-class and fails over), and it is
the bound that binds first — the turn cap (`DEFAULT_MAX_TURNS`, 250) sits
above what that budget buys at the observed ~5 turns/min, so it is a backstop
against a pathological loop rather than a review-length limit.

That ordering assumes a seat running at that rate. `tool_timeout` bounds the
whole agentic invocation — this driver runs one process per branch, not one per
tool — so a seat that paces itself between tool calls reaches that bound long
before 250 turns: the cap never binds, and the seat ends at the timeout's kill
(throttle-class, so the branch fails over) rather than at the `error_max_turns`
ending #213 made diagnosable. So the cap is configurable (#229):
`[review].max_turns` sets the fleet default and a per-entry
`[[review.models]].max_turns` overrides it for that branch — the same precedence
and branch scope as `tool_timeout`, including the backups that answer a failover
inside that branch's pool. Unset at both levels means `DEFAULT_MAX_TURNS`.

Set the two knobs together, because the smaller bound is the one that fires. A
paced seat that should finish its review needs its `tool_timeout` raised to cover
its paced wall-clock first; only then is `max_turns` the bound that binds, and
its number then has to sit under what that budget buys. Leave `tool_timeout` at a
value the seat's pacing overruns and the seat still dies at the whole-run kill
whatever `max_turns` says — which reads like a gateway outage rather than a
capped review. The turns-to-wall-clock mapping is per-seat and unmeasured, so
derive a seat's number from its own observed pacing, never the fleet's.
Trial-seat
first: run it as a non-gating `role = "trial"` entry and
score marginal uniqueness receipts-only before letting it gate.
