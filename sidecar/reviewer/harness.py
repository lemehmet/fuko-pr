"""Agent runtimes that can execute the review strategy.

A harness takes a prepared prompt plus the checkout to review and returns the
agent's final text; :func:`sidecar.reviewer.prompt.parse_review` turns that into
structured findings. The first (and currently only) harness drives headless
Claude Code (``claude -p``), which must be installed on the runner. Other
runtimes -- e.g. an OSS agentic harness fronting non-Anthropic models --
implement the same ``run_review`` signature and slot in without touching the
strategy or the driver.

Two isolation properties are load-bearing, because the checkout is an untrusted
contributor's pull request:

* **The agent never runs FROM the checkout.** ``cwd`` is a clean, empty
  directory and the checkout is exposed as an additional readable root
  (``--add-dir``). Claude Code loads project configuration -- including
  ``.claude/settings.json`` **hooks**, which are arbitrary commands -- from its
  working directory, and headless mode skips the workspace-trust prompt that
  would otherwise gate it. Verified against Claude Code 2.1.232: a hook shipped
  in the working directory executes, and the same hook does not execute when
  the directory is passed via ``--add-dir`` instead.
* **The tool surface is pinned to read-only navigation**
  (:data:`ALLOWED_TOOLS`), by ``--tools`` -- which selects the built-in set the
  session HAS -- and not by ``--allowedTools``, which only pre-approves tools
  the session already has. That distinction was a live vulnerability
  (GHSA-wc47-w25x-54fc): with ``--allowedTools Read,Grep,Glob`` alone the agent
  still had ``Bash`` and used it, executing shell against an untrusted checkout
  on a runner holding the seat's provider key and a GitHub App token. Verified
  on Claude Code 2.1.251, clean cwd, empty user-scope ``permissions``:

  ===================================================  ==========
  flags                                                Bash
  ===================================================  ==========
  ``--allowedTools "Read,Grep,Glob"``                  **RUNS**
  ``+ permissions.deny ["Bash", ...]``                 blocked
  ``--tools "Read,Grep,Glob"``                         not offered
  ===================================================  ==========

  Both mechanisms are emitted, and they fail in different directions on
  purpose: ``--tools`` never presents the tool (so a denied call cannot even
  cost a turn), while the ``permissions.deny`` entries in
  :data:`DENIED_TOOLS` still hold if a future CLI renames or drops ``--tools``.
  Neither is a sandbox -- ``Read`` and ``Grep`` still reach the whole runner
  (see :data:`SENSITIVE_HOME_DIRS`); this closes the arbitrary-execution and
  network vectors, not the read surface.
"""

from __future__ import annotations

import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..throttle import TIMEOUT_RETURNCODE
from .transcript import Transcript

ALLOWED_TOOLS = "Read,Grep,Glob"

#: Built-in tools denied by NAME in the ``--settings`` payload, as the standing
#: half of the two-mechanism surface described in the module docstring.
#:
#: This is an enumeration, so it is only as complete as the CLI's built-in set
#: at the time of writing -- which is precisely why it is the SECONDARY
#: mechanism. ``--tools`` is authoritative (it closes the set by construction
#: and needs no maintenance as tools are added); this list is what survives
#: that flag being renamed or dropped, and it covers the tools whose loss of
#: containment would matter: arbitrary execution, writes, and network egress.
#:
#: Deliberately NOT the complement of :data:`ALLOWED_TOOLS`: naming a tool that
#: does not exist in some CLI version is inert, while omitting a dangerous one
#: is not, so this errs toward listing too much.
DENIED_TOOLS = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Task",
    "SlashCommand",
)

#: Set by a caller that REPLACES ``CLAUDE_CONFIG_DIR`` for isolation, carrying
#: the value it replaced so the read denylist can still cover it. Consumed only
#: by :func:`_permission_settings`; the harness itself never reads it, and
#: Claude Code does not know the name.
_ENV_AMBIENT_CONFIG_DIR = "FUKO_AMBIENT_CLAUDE_CONFIG_DIR"

#: Set by a caller that captures session transcripts, carrying every directory
#: a transcript can land in purely so the read denylist can cover them all.
#: Consumed only by :func:`_permission_settings`; nothing here writes through it.
#:
#: NEWLINE-separated, because there is more than one such directory once #238
#: adds a local blob store: the capture destination and, where the ``file``
#: backend is configured on this host, the store root the finished transcripts
#: are shipped into. A newline rather than ``os.pathsep``, whose POSIX value
#: ``:`` also separates a Windows drive letter from its path.
#:
#: It has its own name rather than riding ``FUKO_TRANSCRIPT_DIR`` because that
#: one is stripped with the rest of the ``FUKO_`` namespace before the spawn, so
#: by the time the settings payload is built the destination is invisible --
#: which is how the directory came to be undenied in the first place.
_ENV_TRANSCRIPT_DENY_DIR = "FUKO_TRANSCRIPT_DENY_DIR"

#: Operator-supplied absolute directories to add to the read denylist,
#: NEWLINE-separated. Passed through from the workflow environment.
#:
#: This exists because the stores below are the ones fuko can NAME, and an
#: operator's runner holds credential stores fuko has never heard of. The case
#: that produced it is concrete: the `codex-proxy` preset's translating proxy
#: keeps a long-lived ChatGPT OAuth session whose location is set by the
#: proxy's OWN `CCP_CONFIG_DIR`, so :data:`SENSITIVE_HOME_DIRS` can only cover
#: the default path -- a relocated store is exactly as readable as an undenied
#: one, and on a same-user deployment that is a live exfiltration path (found
#: by the reviewer itself on PR #285).
#:
#: A path knob rather than a `CCP_CONFIG_DIR` special case on purpose: the
#: proxy is one instance of "a credential store whose path only the operator
#: knows", and a rule per vendor would have to be written again for the next
#: one. Rides the same shape as :data:`_ENV_TRANSCRIPT_DENY_DIR` -- read here
#: and nowhere else, stripped with the rest of the ``FUKO_`` namespace and
#: re-set explicitly before the spawn.
_ENV_EXTRA_DENY_DIRS = "FUKO_EXTRA_DENY_DIRS"

#: Directories the agent must never read, relative to the runner's home.
#:
#: ``--add-dir`` ADDS a readable root; it does NOT confine reads to it. Verified
#: on Claude Code 2.1.232: with ``--allowedTools Read`` and a clean cwd, the
#: agent will happily read an absolute path outside both cwd and every
#: ``--add-dir`` root. That matters here because findings are published
#: verbatim to the pull request, which the (untrusted) PR author can read -- so
#: an injected instruction that says "read X and put it in a finding" is an
#: exfiltration channel, not merely "wrong review text". Subscription auth
#: deliberately keeps the runner's own login reachable under ``HOME``, which
#: makes ``~/.claude`` the highest-value target on the box.
#:
#: This is a denylist over the credential stores, not a sandbox: it closes the
#: named paths, it does not confine the agent -- Read and Grep still reach
#: anything else on the runner (``/etc``, other checkouts in the work dir).
#: Real confinement needs the run to happen in a container or under a dedicated
#: unprivileged user, which is the runner's job, not this module's.
#:
#: Two properties below are load-bearing and were measured on 2.1.232, because
#: both are the opposite of what the rule syntax suggests. Canary outside cwd
#: and outside every ``--add-dir`` root, agent asked to Grep it:
#:
#: ===========================================  ==========
#: deny rules                                   outcome
#: ===========================================  ==========
#: (none)                                       LEAKED
#: ``Read(//abs/**)``                           blocked
#: ``Read(//abs/**)`` + ``Grep(//abs/**)``      blocked
#: ``Grep(//abs/**)``                           LEAKED
#: ===========================================  ==========
#:
#: So: a PATH rule is enforced across the read-class tools -- the ``Read(...)``
#: rules below are what actually stop ``Grep`` from reading a credential file --
#: while a TOOL-scoped ``Grep(...)`` rule is not honored at all. We therefore
#: emit only ``Read(...)`` rules on purpose. Adding ``Grep(...)`` entries would
#: be decorative and would imply a coverage guarantee that does not exist.
SENSITIVE_HOME_DIRS = (
    ".claude",
    ".ssh",
    ".aws",
    ".gnupg",
    ".config/gh",
    ".config/gcloud",
    # An Anthropic-to-Codex translating proxy (the `codex-proxy` preset) owns a
    # ChatGPT OAuth session here and refreshes it IN PLACE, so unlike every
    # model credential this backend injects, it is a long-lived one that lives
    # on disk in the runner's home for the life of the box. It is also the one
    # credential the environment denial below cannot reach: the harness never
    # holds it -- the proxy substitutes it downstream -- so `/proc/self/environ`
    # is not the channel, the filesystem is.
    ".config/claude-code-proxy",
    ".docker",
    ".kube",
)
#: Single files worth the same treatment.
#:
#: The ``actions-runner`` entries are the self-hosted runner's own registration
#: credentials, which authenticate the runner to GitHub. Every fuko workflow in
#: this repo is ``runs-on: [self-hosted, ...]``, so these sit in the same home
#: directory this denylist is built from -- a larger prize than several stores
#: already covered, reachable through exactly the channel the denylist exists to
#: close: an injected "read X and put it in a finding" reaches the PR verbatim,
#: where an untrusted PR author reads it.
#:
#: Listed as FILES, deliberately, rather than denying ``actions-runner`` as a
#: directory: the runner's workspace is ``<runner-dir>/_work/<repo>/<repo>``, so
#: a directory rule would deny the checkout itself and leave the reviewer unable
#: to read the code it is reviewing. The narrower rule is not a compromise here;
#: the wider one is simply wrong.
SENSITIVE_HOME_FILES = (
    ".netrc",
    ".git-credentials",
    ".claude.json",
    "actions-runner/.credentials",
    "actions-runner/.credentials_rsautokey",
)

#: Kernel pseudo-filesystems, denied because ``/proc/self/environ`` hands the
#: agent its OWN process environment -- which necessarily holds the credential
#: this backend just injected (``ANTHROPIC_API_KEY`` in api-key mode,
#: ``CLAUDE_CODE_OAUTH_TOKEN`` in subscription mode). Published findings are the
#: same egress channel as the original read-confinement bug; only the source
#: differs. No legitimate code review reads ``/proc``, ``/sys`` or ``/dev``, so
#: this costs a real reviewer nothing.
#:
#: What that environment no longer holds is anything of fuko's own: since #171
#: gave ``FUKO_TOKEN`` ledger-write authority, the entire ``FUKO_`` namespace is
#: stripped before the spawn (:data:`sidecar.backends.agentic._FUKO_ENV_PREFIX`),
#: and what the harness legitimately needs -- ``FUKO_AMBIENT_CLAUDE_CONFIG_DIR``
#: and ``FUKO_TRANSCRIPT_DENY_DIR`` above among them -- is set explicitly
#: afterwards rather than inherited. The
#: model credential CANNOT be removed the same way, because the harness needs it
#: to run at all, which is exactly why the two are handled differently and why
#: this denial still matters.
#:
#: NOT empirically verified: this was developed on darwin, which has no
#: ``/proc``. The rules use the ``Read(//abs/**)`` spelling that WAS verified
#: (see the matrix above), and path rules were measured to cover ``Grep`` too,
#: so one rule closes both tools -- but the specific ``/proc`` denial is
#: reasoned, not measured, and should be checked on a Linux runner.
#:
#: This is also precisely why fuko-pr#102 (running the reviewer in a container)
#: matters: a denylist over ``/proc`` still leaves the credential sitting in
#: the agent's own environment, reachable by any path we failed to enumerate.
#: Only a boundary fixes the class; this closes the instance.
SENSITIVE_SYSTEM_DIRS = ("/proc", "/sys", "/dev")


def _unrepresentable(path: str, *, root_is_a_parent: bool = False) -> str | None:
    """Why this rule builder cannot express ``path``, or ``None`` if it can.

    The same family ``transcript_dir`` and the blob-root validator refuse at
    the WRITING end, applied here at the declaring end -- where fuko cannot
    rename the operator's directory and can only decline to pretend it is
    covered. Returned as a reason rather than raised, because a declaration
    fuko cannot honour must not fail the review; it must be announced.

    ``root_is_a_parent`` says the path only anchors children (``HOME``), where
    the root is a perfectly good parent and is not a reason to refuse.
    """
    if os.name == "posix" and "\\" in path:
        # A backslash is an ORDINARY POSIX filename character. Rewriting it to
        # `/` -- what this builder used to do for every candidate, until #285
        # (operator-declared) and #271 (HOME and config dirs) -- silently names
        # a DIFFERENT directory: `/srv/cred\store` becomes `/srv/cred/store`.
        return (
            "it contains a backslash, an ordinary POSIX filename character "
            "that this rule syntax cannot carry"
        )
    if root_is_a_parent:
        return None
    if path.rstrip("/") == "":
        # `rstrip("/")` reduces the root to the empty string, which is then
        # dropped before the non-POSIX report can see it.
        return "it is the filesystem root"
    if posixpath.normpath(path).rstrip("/") == "":
        # `/.` and `/tmp/..` are the root by another spelling (#286). Decided
        # LEXICALLY, not with `realpath`: the question is whether a rule
        # rendered from this value -- `Read(//./**)`, `Read(//tmp/../**)` --
        # collapses to `Read(//**)` under a path normalizer and blinds the
        # reviewer to the checkout, and that does not depend on what the
        # kernel says `/tmp` is on this host (a symlink into `/private` on
        # darwin, where `realpath("/tmp/..")` is not the root at all).
        return f"it is the filesystem root ({path!r} normalizes to '/')"
    return None


def _environment_root(env: dict[str, str], key: str, *, root_is_a_parent: bool) -> str | None:
    """The directory ``env[key]`` names, normalized for rule building, or ``None``.

    ``None`` means either the variable is unset or empty (nothing to say) or
    its value cannot anchor a rule -- in which case the refusal is announced on
    stderr, because a candidate that is dropped before the ``unusable`` report
    can see it is exactly the silent skip this builder exists to avoid (#271).

    ``root_is_a_parent`` says whether the value is denied ITSELF (a config dir,
    where ``/`` would render ``Read(//**)`` and blind the reviewer to the
    checkout) or only anchors children (``HOME``, whose ``/`` is a perfectly
    good parent for ``/.claude`` and ``/.ssh``). Only in the first case is the
    filesystem root a reason to refuse.

    The value comes back as DECLARED (``/.`` stays ``/.``, an alias stays an
    alias). Its canonical spelling is a second rule the caller adds beside it
    via :func:`_canonical_twin`, never a replacement for it.
    """
    value = env.get(key) or ""
    if not value:
        return None
    if os.name != "posix":
        # Off POSIX the backslash IS the separator, and the rewrite is what lets
        # a Windows-shaped value render at all -- into the `unusable` report,
        # since `C:/Users/...` is not POSIX-absolute. On POSIX the same byte is
        # an ordinary filename character, and rewriting it would name a
        # directory that does not exist; that case is refused below instead.
        value = value.replace("\\", "/")
    reason = _unrepresentable(value, root_is_a_parent=root_is_a_parent)
    if reason:
        print(
            f"fuko: credential denylist NOT applied for {key}={value!r} -- {reason}. "
            "Every deny rule derived from it is missing, so what lives under it is "
            "readable by the reviewer; rename or relocate it.",
            file=sys.stderr,
        )
        return None
    return value.rstrip("/")


def _canonical_twin(path: str, declared_as: str) -> str | None:
    """The canonical spelling of ``path`` to deny BESIDE it, or ``None``.

    A rule is matched against the path the agent spells, so a rule for an alias
    leaves the store readable under its real name -- the bypass ``transcript_dir``
    resolves away before its paths ever reach this builder, and the one #286
    found still open for the HOME-derived stores (a symlinked ``HOME``, a
    symlinked ``~/.ssh``, a ``HOME`` of ``/.``). Both spellings are emitted
    rather than the declared one replaced: whether the CLI resolves a path
    before matching is its business and could change between versions, and an
    extra inert rule costs nothing where a missing one costs the credential.

    ``None`` when ``path`` is already canonical (nothing to add), when it is not
    POSIX-absolute (resolving a relative or Windows-shaped value against the
    working directory would invent a rule nobody declared; the ``unusable``
    report handles it), or when the target cannot be resolved or expressed --
    both announced, because a target that is NOT additionally denied is exactly
    the readable-under-its-real-name hole, and the operator should hear so.
    The canonical target gets the SAME representability check as the declared
    spelling (#285 r4): a clean alias can resolve to a target holding a
    backslash, or to ``/``, and appending either unchecked reintroduces the
    wrong-rule bug one indirection later.
    """
    if not path.startswith("/"):
        return None
    try:
        # Non-strict: the prefix is resolved even where the leaf does not exist
        # yet, which is the normal state of a store the runner has not created.
        resolved = os.path.realpath(path)
    except (OSError, RuntimeError):
        # A resolution loop or an unreadable parent. The declared rule is
        # already emitted; losing the canonical one is worth a sentence on
        # stderr, not a failed review.
        print(
            f"fuko: could not canonicalize {declared_as}; only the declared spelling is denied.",
            file=sys.stderr,
        )
        return None
    if resolved.rstrip("/") == path.rstrip("/"):
        return None
    reason = _unrepresentable(resolved)
    if reason:
        print(
            f"fuko: {declared_as} resolves to {resolved!r}, which is NOT additionally denied "
            f"-- {reason}. Only the declared spelling is covered; reads through the "
            "canonical path are not.",
            file=sys.stderr,
        )
        return None
    return resolved.rstrip("/")


def _permission_settings(env: dict[str, str]) -> str:
    """Build the ``--settings`` payload: hooks off, credential stores unreadable.

    Paths use Claude Code's absolute-rule spelling ``Read(//abs/path/**)`` (a
    leading ``//``), which is what actually matches an absolute path -- a
    single-slash rule silently fails to match and the read goes through.

    Only POSIX-absolute roots produce rules. A Windows-shaped home
    (a ``C:`` drive path) would otherwise render as ``Read(/C:/Users/...)``,
    which is not the verified spelling and would silently match nothing -- a
    denylist that looks present and protects nothing is worse than none at all.
    Such a root is skipped and announced on stderr instead, so the operator
    learns the credential denylist is not in force on that runner rather than
    discovering it from a leaked review.

    The same holds for a root this syntax cannot carry at all: a ``HOME`` or
    config dir holding a backslash on POSIX, or a config dir that IS the
    filesystem root. Neither is rewritten into a rule for some other directory;
    each is refused and announced (#271). A ``HOME`` of ``/`` is not in that
    set -- its stores are ``/.claude``, ``/.ssh``, ..., which are ordinary
    absolute paths and get ordinary rules.

    Every path rule is matched against the path the agent SPELLS, so a store
    reachable under two names needs two rules: each HOME-derived store and each
    config dir is denied under its declared spelling AND its canonical one
    (#286), the way the operator-declared paths already were. A symlinked
    ``HOME``, a symlinked ``~/.ssh`` under a real one, and a ``HOME`` of ``/.``
    all read the same to the kernel and differently to a rule; the resolved
    spelling goes through the same representability check as the declared one.
    """
    candidates: list[tuple[str, bool]] = []  # (path, is_directory)
    home_key = "HOME" if env.get("HOME") else "USERPROFILE"
    home = _environment_root(env, home_key, root_is_a_parent=True)
    if home is not None:
        # Per STORE rather than once for HOME: resolving the root would cover a
        # symlinked home and miss a symlinked `~/.ssh` under a real one, and
        # the per-store resolution covers both, since the home prefix is part
        # of every store's path.
        stores = [(f"{home}/{d}", True) for d in SENSITIVE_HOME_DIRS]
        stores += [(f"{home}/{f}", False) for f in SENSITIVE_HOME_FILES]
        for store, is_dir in stores:
            candidates.append((store, is_dir))
            twin = _canonical_twin(store, f"{home_key}-derived path {store!r}")
            if twin:
                candidates.append((twin, is_dir))
    # Both the config dir the harness will USE and any ambient one it replaced.
    #
    # The caller may redirect CLAUDE_CONFIG_DIR at a private per-branch
    # directory (the agentic backend does, to stop concurrent branches sharing
    # one ~/.claude). Denying only the effective value would then silently drop
    # the rule that had covered the RUNNER's real config dir, leaving an
    # operator's credentials readable by an agent whose findings are published
    # verbatim to an untrusted PR author. Deny both: the replacement, because
    # it accumulates this run's own session state, and the ambient one,
    # because it is the higher-value target and the reason this rule exists.
    for key in ("CLAUDE_CONFIG_DIR", _ENV_AMBIENT_CONFIG_DIR):
        config_dir = _environment_root(env, key, root_is_a_parent=False)
        if config_dir:
            candidates.append((config_dir, True))
            twin = _canonical_twin(config_dir, f"{key}={config_dir!r}")
            if twin:
                candidates.append((twin, True))
    # The session-transcript corpus (#237). Same reasoning as the config dirs
    # one line up, and a higher-value target than either: a transcript holds
    # every `user` tool-result event of a past run -- the full contents of every
    # file that run's agent read -- plus the prompt, the injected knowledge base
    # and the carried prior findings, it is kept rather than deleted, and it
    # accumulates across every repository the runner serves. `0600` does not
    # cover this: the agent is spawned as the same uid, so the mode stops other
    # USERS and not this reader. `Glob` is in :data:`ALLOWED_TOOLS`, so a
    # directory nobody names is still found by `**/*.ndjson`.
    #
    # Denied whenever a destination is configured, not only when THIS run
    # captures: the files an earlier round left behind are the ones worth
    # reading, and they outlive the run that wrote them.
    #
    # EVERY directory, not just the capture one: with #238's `file` backend on
    # this host the shipped blobs are a second, longer-lived copy of the same
    # corpus, and a rule covering only `FUKO_TRANSCRIPT_DIR` would leave it
    # exactly as readable as the capture directory was before #237 denied it.
    for entry in (env.get(_ENV_TRANSCRIPT_DENY_DIR) or "").split("\n"):
        transcript_deny = entry.strip().replace("\\", "/").rstrip("/")
        if transcript_deny:
            candidates.append((transcript_deny, True))
    # Whatever else the operator knows lives on this runner and must not be
    # read. Unlike every rule above, fuko cannot derive these -- see
    # :data:`_ENV_EXTRA_DENY_DIRS`. Non-absolute entries fall into the same
    # `unusable` report below as a Windows-shaped HOME, so a typo announces
    # itself instead of quietly denying nothing.
    for raw in (env.get(_ENV_EXTRA_DENY_DIRS) or "").split("\n"):
        entry = raw.strip()
        if not entry:
            continue
        # A path whose real name carries leading or trailing whitespace cannot
        # be declared through this channel, and the operator has to hear it
        # (#285 r4). `transcript_dir` REFUSES a padded value outright, which is
        # right for a single-value setting where padding is unambiguous. This
        # is a newline-separated LIST, where indentation is ordinary formatting
        # and refusing it would break the natural way to write more than one
        # entry -- so the strip stays and the ambiguity is announced instead.
        # What must not survive either way is the silent wrong rule: `/srv/oauth `
        # is a different directory from `/srv/oauth`, and denying the latter
        # while the former holds the session is the failure this whole branch
        # exists to prevent.
        if raw.strip("\r") != entry:
            print(
                f"fuko: declared deny path {raw!r} was read as {entry!r} -- surrounding "
                "whitespace is treated as list formatting. If the directory's real name "
                "has whitespace at either end it is NOT covered; rename it.",
                file=sys.stderr,
            )
        # REJECT the spellings this rule syntax cannot carry, rather than
        # rewriting them into a rule for some OTHER directory (#285 r3). The
        # transcript and blob-root validators already enumerate this family and
        # refuse it at the writing end; this is the same taxonomy at the
        # declaring end, where fuko cannot rename the operator's directory and
        # so can only decline to pretend it is covered.
        #
        # A silent wrong rule is the worst outcome available here: the operator
        # sees a declaration, the denylist reports no problem, and the store
        # stays readable. Every refusal below is therefore announced.
        unrepresentable_reason = _unrepresentable(entry)
        if unrepresentable_reason:
            print(
                f"fuko: NOT denying declared path {entry!r} -- {unrepresentable_reason}. "
                "That store is readable by the reviewer; move it or rename it.",
                file=sys.stderr,
            )
            continue
        extra_deny = entry.rstrip("/")
        candidates.append((extra_deny, True))
        # ALSO deny the canonical target -- the operator declaring a symlinked
        # or `..`-containing store is the realistic case, not the adversarial
        # one. See `_canonical_twin` for why both spellings rather than one.
        twin = _canonical_twin(extra_deny, f"declared path {entry!r}")
        if twin:
            candidates.append((twin, True))
    # Unconditional: these do not depend on HOME, and on a runner without one
    # they are the only rules that remain.
    candidates += [(d, True) for d in SENSITIVE_SYSTEM_DIRS]

    # Build the rule from a normalized path rather than by concatenating onto
    # whatever the environment held: POSIX permits a leading `//` with
    # implementation-defined meaning, so a HOME of `//home/runner` would
    # otherwise render `Read(///home/...)` -- the silently non-matching form,
    # leaving the credential stores undenied with nothing to show for it.
    # Stripping and re-adding makes exactly one `//` prefix by construction, so
    # the broken spelling cannot be produced at all rather than merely asserted
    # against.
    # Bare tool names first, so the execution/egress denial is present even on a
    # runner where every path rule is dropped as non-POSIX (see below). Those
    # two failures are independent: a Windows-shaped HOME must not silently take
    # the arbitrary-execution denial down with the credential denylist.
    deny = list(DENIED_TOOLS)
    deny += [
        f"Read(//{path.lstrip('/')}/**)" if is_dir else f"Read(//{path.lstrip('/')})"
        for path, is_dir in candidates
        if path.startswith("/")
    ]
    # Two spellings can render as one rule (`//home/x` and its canonical
    # `/home/x` both lose their leading slashes here); the duplicate is inert
    # but reads as a mistake in the payload.
    deny = list(dict.fromkeys(deny))
    unusable = sorted({path for path, _ in candidates if not path.startswith("/")})
    if unusable:
        print(
            "fuko: credential denylist NOT applied -- these paths are not "
            f"POSIX-absolute, so no verified deny rule exists for them: {', '.join(unusable)}. "
            "The agentic reviewer's read denylist is inert on this runner; run it "
            "in a container or under a dedicated unprivileged user.",
            file=sys.stderr,
        )
    return json.dumps({"disableAllHooks": True, "permissions": {"deny": deny}})


# Not listed in `claude --help` for 2.1.232, but accepted (verified: unknown
# options exit with "error: unknown option", this one runs) and documented in
# the CLI reference. It bounds a pathological tool loop; the wall-clock
# `tool_timeout` is the outer bound that does not depend on this flag.
#
# 250 is chosen so `tool_timeout` binds FIRST: at the observed ~5 turns/min a
# 2700s budget is ~225 turns, and fuko-pr's own 1800s is ~150, so in both cases
# a runaway hits the wall-clock bound the budget arithmetic actually reasons
# about. That keeps this a backstop against a pathological loop rather than a
# review-length limit -- at 50 it was the latter, and exhausting it ENDS a
# review with exit 1 and an empty stderr, indistinguishable from a crash (#149).
#
# That derivation holds only for a seat running at that RATE. `tool_timeout`
# bounds the whole agentic invocation -- this driver runs one process per branch,
# not one per tool -- so a seat that paces itself between tool calls reaches that
# bound long before 250 turns and the cap never binds: it dies by the timeout's
# kill rather than at the `error_max_turns` ending, which is the diagnosable one.
# Restoring that ordering is a two-knob job -- the smaller bound fires, so such a
# seat needs `tool_timeout` raised to cover its pacing AND a cap under what that
# budget then buys -- which is why 250 is a DEFAULT and not the number:
# `[review].max_turns` sets the fleet's, a per-entry
# `[[review.models]].max_turns` sets a seat's (#229).
DEFAULT_MAX_TURNS = 250

# "Not logged in · Please run /login" and the API-key equivalents. Auth failure
# must NOT be treated as throttling: failing over to the next provider would
# burn the whole pool on what is a one-line runner fix.
_AUTH_FAILURE_RE = re.compile(
    r"not logged in|please run /login|invalid api key|authentication_error|"
    r"\bunauthorized\b|\b401\b",
    re.IGNORECASE,
)


class HarnessNotAvailableError(RuntimeError):
    """Raised when the harness binary is not installed on this runner."""


@dataclass(frozen=True)
class HarnessResult:
    """The raw outcome of one agent run.

    ``usage`` / ``cost_usd`` / ``turns`` are the terminal ``result`` event's own
    accounting (#152), and are ``None`` whenever that event did not arrive or did
    not carry them -- a killed run, CLI schema drift, a harness with no usage
    feed at all. ``usage`` stays the raw mapping the CLI emitted rather than a
    normalized one, so a field this repo does not yet read (thinking tokens, the
    1h/5m cache split) is preserved for whoever adds the column;
    :func:`usage_tokens` is the mapping onto the fields the metrics row stores.
    """

    returncode: int
    text: str
    stderr: str = ""
    timed_out: bool = False
    usage: dict | None = None
    cost_usd: float | None = None
    turns: int | None = None


#: ``review_runs`` token column -> the key it reads from a ``result`` event's
#: ``usage`` mapping.
#:
#: The spellings differ on purpose and the difference is the whole point of the
#: pairing: an Anthropic-shaped ``input_tokens`` counts only the tokens actually
#: billed as fresh input -- cache reads and cache writes are reported ALONGSIDE
#: it, not inside it -- whereas an OpenAI-shaped ``prompt_tokens`` (the columns
#: migration 004 reserved for a PR-Agent capture path) is the inclusive total.
#: Storing them under distinct names keeps "what did this cost" and "is the
#: gateway honouring prompt caching" separable, which is the question #152 exists
#: to answer.
_USAGE_FIELDS = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_tokens": "cache_read_input_tokens",
    "cache_write_tokens": "cache_creation_input_tokens",
}


def _as_int(value: object) -> int | None:
    """A non-negative int from an untrusted event field, else None.

    ``bool`` is excluded explicitly because it is an ``int`` subclass, so a CLI
    that ever emitted ``true`` for a count would otherwise be recorded as one
    token rather than as "not reported".
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _as_float(value: object) -> float | None:
    """A non-negative FINITE float from an untrusted event field, else None.

    ``json.loads`` accepts the bare ``NaN`` / ``Infinity`` / ``-Infinity``
    literals, and ``NaN`` compares false against every bound -- so a plain
    ``value < 0`` guard admits both. Neither survives the trip: over HTTP
    ``NaN`` fails ``RunMetricRequest``'s ``ge=0`` and infinity overflows
    ``NUMERIC(10,4)``, either of which loses the WHOLE metrics row rather than
    just the cost, and on the direct path a stored ``NaN`` poisons every
    ``sum(cost_usd)`` group it lands in, permanently. Rejecting here keeps a
    garbled figure degrading to "not measured" like any other field.

    The conversion is guarded because it is the one that can raise: an
    arbitrarily large JSON integer parses to a Python ``int`` that ``float()``
    cannot represent, and an ``OverflowError`` out of a best-effort accounting
    helper would kill the review it is only supposed to be measuring.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def usage_tokens(usage: dict | None) -> dict[str, int | None]:
    """Map a ``result`` event's ``usage`` onto the metrics token fields (#152).

    Every field degrades to ``None`` independently: a usage mapping that is
    absent, not a mapping, or missing/garbled in one key yields ``None`` for
    that key rather than a zero. The distinction is load-bearing downstream --
    ``NULL`` reads as "not measured" while ``0`` reads as "free".
    """
    reported = usage if isinstance(usage, dict) else {}
    return {column: _as_int(reported.get(key)) for column, key in _USAGE_FIELDS.items()}


def is_auth_failure(output: str) -> bool:
    """Return whether ``output`` looks like a credential problem, not a capacity one."""
    return bool(output) and _AUTH_FAILURE_RE.search(output) is not None


def check_auth(env: dict[str, str]) -> dict | None:
    """Return ``claude auth status`` as parsed JSON, or None if it cannot be read.

    Used as a preflight for subscription mode, where a lapsed login on the
    runner would otherwise surface as a confusing mid-review failure on every
    PR. Returns None (rather than raising) when the binary is missing, the
    probe fails, or the output is not a JSON **object** -- an unreadable probe
    is not evidence of a broken login, so the review proceeds and the run
    itself reports the truth. The object check matters because ``json.loads``
    happily returns a bare ``null``/number/string for malformed-but-valid JSON,
    which the caller would then treat as a status mapping and crash on.
    """
    binary = shutil.which("claude", path=env.get("PATH"))
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        parsed = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def run_review(
    prompt: str,
    repo_dir: Path,
    *,
    cwd: Path,
    model: str,
    env: dict[str, str],
    timeout: int,
    max_turns: int = DEFAULT_MAX_TURNS,
    transcript: Transcript | None = None,
) -> HarnessResult:
    """Run headless Claude Code over ``repo_dir`` and return its final text.

    The agent runs from ``cwd`` -- which must be a clean directory the
    repository does not control -- with ``repo_dir`` mounted as an additional
    readable root, so no project configuration from the reviewed code is
    loaded (see the module docstring).

    The prompt goes over stdin (it embeds a full diff -- argv has size limits
    and shows up in process listings). ``--output-format stream-json`` (which
    print mode requires ``--verbose`` for) turns stdout into an NDJSON event
    feed consumed INCREMENTALLY: each ``tool_use`` block becomes one compact
    progress line on this process's stderr as it happens (mepro asked for
    this after a fleet of 15-30 min reviews whose only log lines were start
    and end -- an 1800s kill now shows the last tool the seat was on), and
    the agent's final message is lifted from the terminal ``result`` event,
    so the downstream contract is unchanged: the returned ``text`` is still
    the final message the strategy constrains to a bare JSON object. Event
    parsing is TOLERANT -- unknown or non-JSON lines are skipped, and if the
    ``result`` event never arrives (CLI schema drift, mid-stream kill) the
    last assistant text block stands in, so drift degrades to the old
    behavior rather than a hard failure. That same terminal event carries the
    run's token usage, dollar cost and turn count, which ride back on the
    result (#152) for the metrics row -- best-effort, ``None`` when absent.
    ``env`` is the harness process
    environment (the caller decides exactly which credentials it carries) and
    is used as given rather than merged with this process's.

    A timeout maps to :data:`sidecar.throttle.TIMEOUT_RETURNCODE` so the driver
    classifies a hung run as throttle-class, same as a hung PR-Agent container.

    An optional ``transcript`` (:mod:`sidecar.reviewer.transcript`) tees the raw
    feed to durable storage as it streams, INCLUDING the ``user`` tool-result
    events this fold skips (#237). It sits beside the fold rather than inside
    it: what comes back here is byte-identical whether capture is on, off, or
    failing, and passing ``None`` leaves the stream untouched.
    """
    binary = shutil.which("claude", path=env.get("PATH"))
    if binary is None:
        raise HarnessNotAvailableError(
            "the 'claude' CLI is not on PATH; install Claude Code on this "
            "runner or switch this model entry to the pr-agent backend"
        )
    cmd = [
        binary,
        "-p",
        "--model",
        model,
        "--output-format",
        "stream-json",
        # Print mode refuses stream-json without it; it gates the event feed,
        # not log chattiness.
        "--verbose",
        # `--tools` selects the built-in set the session HAS; `--allowedTools`
        # only pre-approves tools it already has. Only the first is a boundary
        # (GHSA-wc47-w25x-54fc) -- keep both: the allowlist is what stops the
        # permitted three from prompting, which headless mode cannot answer.
        "--tools",
        ALLOWED_TOOLS,
        "--allowedTools",
        ALLOWED_TOOLS,
        "--add-dir",
        str(repo_dir),
        # Belt and braces on top of the clean cwd, each independently sufficient
        # for its vector: load settings from the USER scope only (never the
        # reviewed project's), refuse hooks outright, and start no MCP server
        # that was not explicitly configured here (none is).
        "--setting-sources",
        "user",
        "--settings",
        _permission_settings(env),
        "--strict-mcp-config",
        "--max-turns",
        str(max_turns),
    ]
    start = time.monotonic()

    def _emit(turn: int, tool: str, arg: str) -> None:
        mins = int((time.monotonic() - start) // 60)
        print(
            f"fuko: agentic {model} [{mins}m t{turn}] {tool} {arg}",
            file=sys.stderr,
            flush=True,
        )

    returncode, outcome, stderr, timed_out = _drive(
        cmd, prompt=prompt, cwd=cwd, env=env, timeout=timeout, emit=_emit, transcript=transcript
    )
    # Only a non-success subtype is announced: a clean run stays quiet so the
    # line is a signal and not noise. It is keyed on the subtype alone, not on
    # the returncode, because the returncode is exactly what cannot tell these
    # endings apart -- `error_max_turns` exits 1 with an empty stderr (#149).
    # The MODEL rides the line for the same reason it rides `_emit` above:
    # concurrent branches interleave on ONE stderr, so a line without it is
    # unassignable when two seats end badly in the same window.
    #
    # The subtype is FLATTENED for the same reason `_emit`'s argument is: it is
    # a stream-derived string, so a value carrying a line break would put chosen
    # text at column 0 of its own line and forge a gate downstream log consumers
    # anchor on (#147). The guard tests the RAW value on purpose -- flattening
    # first would let a crafted `success\n...` go quiet, which is backwards.
    if outcome.subtype and outcome.subtype != "success":
        print(
            f"fuko: agentic {model} harness ended with result subtype={_flatten(outcome.subtype)}",
            file=sys.stderr,
            flush=True,
        )
    if timed_out:
        # The text is deliberately dropped (a killed run has no verdict), but the
        # accounting is not: a run killed at 30 minutes is the single most
        # expensive shape this fleet produces, and if the terminal event did
        # arrive before the kill its tokens were still spent and still billed.
        return HarnessResult(
            returncode=TIMEOUT_RETURNCODE,
            text="",
            stderr=stderr or f"review timed out after {timeout}s",
            timed_out=True,
            usage=outcome.usage,
            cost_usd=outcome.cost_usd,
            turns=outcome.turns,
        )
    return HarnessResult(
        returncode=returncode,
        text=outcome.text,
        stderr=stderr,
        usage=outcome.usage,
        cost_usd=outcome.cost_usd,
        turns=outcome.turns,
    )


def _flatten(value: str) -> str:
    r"""One PHYSICAL log line, always.

    The argument is reviewer-chosen (and PR-author-influenced — seats grep for
    strings drawn from the diff), and downstream log gates anchor on line
    starts, so an embedded newline must not let an argument place chosen text
    at column 0 of its own line (mepro PR #2014 r2).

    Flattened via ``splitlines()`` rather than by replacing ``\r``/``\n``:
    Python breaks lines on eight further characters (``\x0b``, ``\x0c``,
    ``\x1c``-``\x1e``, ``\x85``, ``\u2028``, ``\u2029``), so the replace form
    left a crafted argument looking flat while any consumer splitting by the
    normal rule still saw two lines — the very forgery this guards against
    (fuko-henry, #147). Truncation happens AFTER flattening so a cut cannot
    resurrect a break.
    """
    return " ".join(value.splitlines())[:100]


def _tool_arg(tool_input: dict) -> str:
    """The one argument worth showing for a tool call, truncated for a log line."""
    for key in ("file_path", "pattern", "query", "command", "url", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return _flatten(value)
    for value in tool_input.values():
        if isinstance(value, str) and value:
            return _flatten(value)
    return ""


@dataclass(frozen=True)
class _StreamOutcome:
    """What folding one event feed yielded: the final text plus its accounting."""

    text: str
    saw_result: bool
    usage: dict | None = None
    cost_usd: float | None = None
    turns: int | None = None
    subtype: str | None = None


def _consume_stream(lines, emit) -> _StreamOutcome:
    """Fold the harness's NDJSON event feed into a :class:`_StreamOutcome`.

    ``emit(turn, tool, arg)`` fires once per ``tool_use`` block as it streams.
    Tolerant by design: blank/non-JSON/unknown lines are skipped, because a
    progress feature must never be the thing that kills a review. When no
    ``result`` event arrives, the last assistant text block stands in -- for a
    healthy run they are the same message.

    The terminal event's ``usage`` / ``total_cost_usd`` / ``num_turns`` ride
    along (#152); they were previously parsed and dropped, and they are the only
    place a run's real token and cash cost is stated. They are read whether or
    not that event's ``result`` field is usable text, because a schema change
    that costs us the final message should not also cost us the bill.

    ``turns`` is the event's own ``num_turns`` and nothing else. The assistant
    messages counted here for the progress lines are a DIFFERENT quantity, so
    substituting them would put two definitions in one column; a run whose
    terminal event never arrived honestly reports ``None``.

    ``subtype`` is the terminal event's own verdict on how the run ended, and is
    the ONLY place a turn-cap exhaustion says so: it exits 1 with nothing on
    stderr but a benign startup warning, so a review that simply stopped was
    read as a provider fault for a full day (#149). Folding the feed away used
    to discard it.
    """
    result_text: str | None = None
    last_assistant_text = ""
    reported_usage: dict | None = None
    reported_cost: float | None = None
    reported_turns: int | None = None
    reported_subtype: str | None = None
    turns = 0
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "assistant":
            turns += 1
            message = event.get("message") or {}
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    emit(turns, block.get("name") or "?", _tool_arg(block.get("input") or {}))
                elif block.get("type") == "text" and block.get("text"):
                    last_assistant_text = block["text"]
        elif kind == "result":
            usage = event.get("usage")
            if isinstance(usage, dict):
                reported_usage = usage
            reported_cost = _as_float(event.get("total_cost_usd"))
            reported_turns = _as_int(event.get("num_turns"))
            if isinstance(event.get("subtype"), str):
                reported_subtype = event["subtype"]
            if isinstance(event.get("result"), str):
                result_text = event["result"]
    return _StreamOutcome(
        text=last_assistant_text if result_text is None else result_text,
        saw_result=result_text is not None,
        usage=reported_usage,
        cost_usd=reported_cost,
        turns=reported_turns,
        subtype=reported_subtype,
    )


def _tee(lines, transcript: Transcript):
    """Yield each raw feed line after handing a copy to ``transcript``.

    A GENERATOR, so the fold downstream keeps iterating the pipe lazily and the
    capture holds one line at a time -- collecting the feed to write it after
    the run would reintroduce exactly the memory cost the streaming design
    avoids, on the runs that are largest (#237).

    The copy is written BEFORE the line is yielded, so whatever the fold has
    seen is already durable: a mid-stream kill can lose the line that was in
    flight, never one the review already acted on.
    """
    for raw in lines:
        transcript.write(raw)
        yield raw


def _drive(
    cmd: list[str],
    *,
    prompt: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    emit,
    transcript: Transcript | None = None,
) -> tuple[int, _StreamOutcome, str, bool]:
    """Run ``cmd`` streaming stdout through :func:`_consume_stream`.

    With a ``transcript``, the pipe is wrapped in :func:`_tee` on the way to the
    fold; without one it is passed through untouched, so capture that is off
    costs nothing per event rather than a no-op call per event.

    Returns ``(returncode, outcome, stderr, timed_out)``. Three pipes need
    three actors to avoid deadlock on a large prompt or chatty child: stdin is
    fed from its own thread (the child may start emitting before it finishes
    reading a multi-megabyte diff), stderr drains on a second thread, and the
    main thread consumes the stdout event feed so progress lines appear the
    moment the child writes them. The timeout is a Timer that kills the
    process outright -- the driver classifies that as throttle-class via
    TIMEOUT_RETURNCODE exactly as the old ``subprocess.run(timeout=...)`` did.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # The feed is NDJSON, which is UTF-8 by its own definition -- so say so,
        # rather than letting the runner's locale decide how the child's bytes
        # are read. And decode TOLERANTLY (#258): the timeout kills the child
        # outright, and a kill landing mid multibyte character leaves a partial
        # sequence in the pipe that strict decoding raises `UnicodeDecodeError`
        # on -- a `ValueError`, so it escapes `invoke`'s `OSError` handler,
        # reaches the runner's branch-level `except`, and costs that attempt its
        # whole metrics row while its transcript is already in the store.
        #
        # Replacing folds that into the tolerance the feed already has: the
        # mangled line fails `json.loads` and is skipped by `_consume_stream` and
        # by the transcript meter alike (both return on a `ValueError` rather
        # than going inert), so a cut-short run is indexed with `complete=False`
        # -- which is what it is -- instead of vanishing.
        #
        # It applies to stdin too, and that is also an improvement: an
        # unencodable character in the prompt used to raise `UnicodeEncodeError`
        # in `_feed`, which catches only `BrokenPipeError`/`OSError`, killing the
        # feeder thread with stdin never closed and hanging the child until this
        # timer fires.
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd),
        env=env,
    )
    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, _kill)
    timer.start()
    stderr_chunks: list[str] = []
    stderr_thread = threading.Thread(target=lambda: stderr_chunks.extend(proc.stderr), daemon=True)
    stderr_thread.start()

    def _feed() -> None:
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # child died first; its returncode tells the story

    stdin_thread = threading.Thread(target=_feed, daemon=True)
    stdin_thread.start()
    try:
        stream = proc.stdout if transcript is None else _tee(proc.stdout, transcript)
        outcome = _consume_stream(stream, emit)
        proc.wait()
    finally:
        timer.cancel()
        # Closed here rather than only by the caller: the feed is over the
        # moment the fold returns, on the kill path as much as the clean one.
        # `close()` is idempotent, so the driver's own `finally` -- which also
        # covers the paths that never reach this function -- still holds.
        if transcript is not None:
            transcript.close()
    stderr_thread.join(timeout=5)
    return proc.returncode, outcome, "".join(stderr_chunks), timed_out.is_set()
