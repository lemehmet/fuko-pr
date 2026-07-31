# fuko Review Signal v1

A small, stable schema for one PR-review finding. Every reviewer fuko knows about
(PR-Agent, Copilot, CodeRabbit) is normalized into this shape, so a consumer reads
**one** deterministic format instead of sniffing each vendor's ad-hoc markdown.

`fuko signals --pr-url <url>` emits a JSON array of these for a pull request.

## Schema

| Field | Type | Notes |
|-------|------|-------|
| `v` | int | Schema version. `1`. |
| `id` | string | Stable id (`fk_<hash>`), derived from file+line+title, for dedup/threading. |
| `file` | string \| null | Path the finding is about. |
| `line` | int \| null | Start line (or the line, for single-line findings). |
| `end_line` | int \| null | End line for a multi-line range, else null. |
| `severity` | enum | `info` \| `low` \| `medium` \| `high` \| `critical`. |
| `severity_source` | enum | `declared` (the reviewer stated it) \| `inferred` (fuko mapped it). |
| `category` | enum | `bug` \| `security` \| `perf` \| `style` \| `test` \| `docs` \| `design`. |
| `title` | string | Short headline. |
| `body` | string | Full finding text (the human-facing comment). |
| `suggestion` | bool | Whether the finding carries an applicable code suggestion. |
| `thread_url` | string \| null | Where to reply / resolve. |
| `backend` | string | `pr-agent` \| `copilot` \| `coderabbit`. |
| `model` | string | Review model id when known (e.g. `anthropic/claude-sonnet-4-6`), else `""`. |
| `kb_refs` | string[] | Knowledge-base learnings that drove the finding, when known. |

`severity`/`category` are **closed enums** on purpose: a deterministic consumer
beats perfect fidelity. When a reviewer doesn't state severity, fuko infers it and
sets `severity_source: "inferred"` so the consumer can treat it with appropriate
skepticism.

## Marker (optional, for self-describing threads)

A signal can travel inside an invisible HTML comment appended to the PR comment it
describes:

```
<!-- fuko-signal:v1 {"v":1,"id":"fk_a1b2","severity":"high",...} -->
```

It renders as nothing on GitHub/GitLab and survives round-trips, so a consumer can
`grep` `fuko-signal:v1` and parse the JSON. The marker carries **machine fields
only** — `title`/`body` stay in the visible comment and are excluded, and any `>`
in a value is JSON-escaped, so the payload can never contain `-->` and break the
comment.

## How each reviewer maps in

- **PR-Agent** — detected by format (it posts under whatever token runs it). It
  declares a `[label, importance: N]` tag, so severity (N→scale) and category are
  `declared`.
- **Copilot** — detected by author; free-form prose → `inferred` severity/category
  (light keyword inference).
- **CodeRabbit** — detected by author **and** a finding classification line
  (`_⚠️ Potential issue_ | _🔴 Critical_`); its chat replies and rate-limit
  notices carry no classification and are skipped. Severity/category are `declared`.

## Run receipts (`fuko-run:v1`) — coverage, not findings

A Review Signal says *what a reviewer found*. It cannot say *whether a reviewer
ran*: an instance that reviewed the HEAD and found nothing emits exactly what an
instance that never started emits — nothing. A consumer gating a merge on "the
reviewer went quiet" therefore cannot tell review coverage from a missing API
key, a throttled provider, or a branch that crashed. The ambiguity resolves in
the unsafe direction: it merges unreviewed code.

So each fuko branch writes a **run receipt** into its own header comment:

```
<!-- fuko-run:v1 {"v":1,"label":"openrouter/x-ai/grok-4.5","role":"active",
     "head_sha":"def5678…","state":"done","model":"openrouter/x-ai/grok-4.5"} -->
```

| field | meaning |
|---|---|
| `label` | `provider/name` of the branch's configured **primary** entry |
| `role` | `active` (gating), `trial` (surfaced, non-gating), `backup` |
| `slot` | A/B slot, when the branch occupies one |
| `head_sha` | the PR HEAD this branch reviewed |
| `state` | `in_progress` → `done` \| `failed` |
| `model` | who **actually** answered — differs from `label` when a backup was promoted |
| `findings` | signals produced, when counted |
| `detail` | human-readable outcome or failure reason |

The receipt is written when the branch starts and **rewritten in place** when it
ends, so there is exactly one receipt per instance per PR — a consumer reads
coverage without deduplicating.

`fuko status` surfaces these as `fuko:<label>` rows alongside the CodeRabbit and
Copilot rows. Every ambiguous case reports as *not* `done`:

- a receipt stuck at `in_progress` (the branch died before finalizing) stays
  `in_progress`, so the consumer's own timeout governs rather than an implied
  never-ending run;
- a receipt for an older commit, or one with no `head_sha` (HEAD was
  unresolvable at run time), reports `pending`;
- a `failed` receipt reports `unavailable`, which is a **degraded** state, so
  `escalation_needed` fires and backups are promoted next round.

## Stability

`v1` is additive-only: new optional fields may appear; existing fields keep their
meaning. A breaking change bumps the version (`fuko-signal:v2` / `fuko-run:v2`)
and consumers can branch on `v`.
