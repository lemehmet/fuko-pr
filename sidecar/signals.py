"""The canonical fuko Review Signal schema (v1) and its comment marker.

Every backend normalizes its reviewer's output into this shape, so a consumer
(e.g. an address-PR-reviews tool) reads one deterministic schema instead of
sniffing each vendor's ad-hoc format. A signal travels inside an *invisible* HTML
comment marker (``<!-- fuko-signal:v1 {json} -->``) appended to the PR comment it
describes: it renders as nothing on GitHub/GitLab and survives round-trips, so the
consumer can ``grep`` the marker and parse the JSON deterministically.

The marker carries only machine fields -- the human-facing ``title``/``body`` stay
in the visible comment and are excluded -- and any ``>`` in a field value is
JSON-escaped, so the serialized payload can never contain ``-->`` and prematurely
terminate the HTML comment, whatever the field values.
"""

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "low", "medium", "high", "critical"]
Category = Literal["bug", "security", "perf", "style", "test", "docs", "design"]

_MARKER_TAG = "fuko-signal:v1"
_MARKER_RE = re.compile(r"<!--\s*fuko-signal:v1\s+(.*?)\s*-->")
_MARKER_STRIP_RE = re.compile(r"\n*<!--\s*fuko-signal:v1\s+.*?\s*-->\n*")

_RUN_TAG = "fuko-run:v1"
_RUN_RE = re.compile(r"<!--\s*fuko-run:v1\s+(.*?)\s*-->")
_RUN_STRIP_RE = re.compile(r"\n*<!--\s*fuko-run:v1\s+.*?\s*-->\n*")

RunState = Literal["in_progress", "done", "failed"]


class ReviewSignal(BaseModel):
    """A single normalized review finding."""

    v: int = 1
    id: str
    file: str | None = None
    line: int | None = None
    end_line: int | None = None
    severity: Severity = "medium"
    severity_source: Literal["declared", "inferred"] = "inferred"
    category: Category = "bug"
    title: str = ""
    body: str = ""
    suggestion: bool = False
    suppressed: bool = Field(
        default=False,
        description=(
            "Whether the reviewer demoted this finding out of its promoted output -- "
            "today, Copilot's collapsed 'Suppressed comments' block, which rides in a "
            "review body whose prose says it generated no new comments. Surfaced as a "
            "distinct sub-source rather than merged in: the reviewer demoted it for a "
            "reason and a consumer may weight it lower, but it has to SEE the finding "
            "before it can weigh it."
        ),
    )
    thread_url: str | None = None
    backend: str = ""
    model: str = ""
    role: str = Field(
        default="active",
        description=(
            "Producing fuko branch role: 'active' (gating), 'trial' (surfaced but "
            "non-gating), or 'backup' (a promoted failover, inheriting its branch's "
            "role). Defaults to 'active' so external reviewers and pre-role markers "
            "decode as gating. Kept a plain str, not a Literal, so a future role "
            "value still round-trips through the marker instead of failing decode; "
            "values only ever come from the config-validated role, and consumers "
            "fail toward gating (anything != 'trial' gates)."
        ),
    )
    kb_refs: list[str] = Field(default_factory=list)


class RunReceipt(BaseModel):
    """A per-branch record that a fuko instance ran against a specific HEAD.

    A Review Signal says *what a reviewer found*; a receipt says *whether that
    reviewer ran at all*. Without one, a fuko instance that found nothing is
    indistinguishable from one that never started -- both produce zero signals --
    so a consumer gating a merge on "the instance went quiet" cannot tell review
    coverage from a silently broken key, a throttled provider, or a crashed
    branch. That ambiguity is unsafe in exactly one direction: it merges
    unreviewed code.

    The receipt travels in the branch's own header issue comment, written when the
    branch starts (``in_progress``) and rewritten in place when it ends, so each
    instance has exactly one receipt per PR that a consumer can read alongside
    :func:`sidecar.status.reviewer_states`.
    """

    v: int = 1
    label: str = Field(description="`provider/name` of the branch's configured PRIMARY entry.")
    role: str = Field(
        default="active",
        description=(
            "The branch's configured role: 'active' (gating), 'trial' (surfaced but "
            "non-gating), or 'backup'. Mirrors ReviewSignal.role so a consumer can "
            "apply one gating rule to findings and coverage alike."
        ),
    )
    slot: str | None = Field(
        default=None, description="A/B slot identifier, when the branch occupies one."
    )
    promoted: bool = Field(
        default=False,
        description=(
            "Whether this branch is a backup that escalation promoted to active for "
            "the round. Such a branch has no slot of its own, so without this a "
            "consumer sees a null slot beside role='active' and cannot tell a "
            "promoted backup from a misconfigured active."
        ),
    )
    head_sha: str = Field(
        default="",
        description=(
            "The PR HEAD this branch reviewed. A receipt for an older HEAD means the "
            "instance has not yet reviewed the current one -- the same staleness rule "
            "the CodeRabbit/Copilot rows use."
        ),
    )
    state: RunState = Field(
        default="in_progress",
        description=(
            "'in_progress' once the branch starts, then 'done' (a review was posted) "
            "or 'failed' (every model in the branch's pool, primary plus backups, was "
            "exhausted). A receipt stuck at 'in_progress' means the branch died before "
            "it could finalize -- which reads as NOT done, the fail-safe direction."
        ),
    )
    model: str = Field(
        default="",
        description=(
            "`provider/name` that actually produced the review, which differs from "
            "`label` when the primary throttled and a backup was promoted. This is the "
            "attribution a consumer needs to score a model's findings."
        ),
    )
    backend: str = Field(
        default="pr-agent",
        description=(
            "The review DRIVER (harness) that produced this run -- 'pr-agent' or "
            "'agentic'. Two harnesses are otherwise indistinguishable receipts-only, "
            "and scoring is receipts-only by rule (#99). Defaults to 'pr-agent' so a "
            "receipt written before this field existed decodes as the only backend "
            "that could have produced it, matching the review_runs backfill."
        ),
    )
    findings: int | None = Field(
        default=None, description="Signals this branch produced; None when not counted."
    )
    channels: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-tool terminal outcome ('review' -> 'done', 'improve' -> "
            "'killed:timeout', ...). A seat publishes on more than one channel, so "
            "a single branch-level state cannot say that the guide posted while the "
            "suggestions channel died -- and an optional tool's failure leaves the "
            "branch at 'done'. Anything other than 'done' here is REDUCED COVERAGE, "
            "not a clean pass. Empty means NOT REPORTED, never 'every channel was "
            "healthy': it is what an in-flight receipt carries before any tool has "
            "finished, what a backend that does not report per-channel outcomes "
            "produces, and what a receipt written before this field existed has. A "
            "consumer therefore cannot detect a dead channel on such a receipt -- "
            "that is a gap to close per backend, not an assertion that none exists."
        ),
    )
    detail: str = Field(default="", description="Human-readable outcome or failure reason.")


def make_id(*parts: str) -> str:
    """Return a stable ``fk_`` id derived from ``parts`` (same inputs -> same id)."""
    digest = hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:10]
    return "fk_" + digest


def encode_marker(signal: ReviewSignal) -> str:
    """Render ``signal`` as an invisible HTML-comment marker (machine fields only).

    Any ``>`` in a field value is JSON-escaped so the payload can never contain
    ``-->``; ``model_validate_json`` decodes the escape on the way back.
    """
    payload = signal.model_dump_json(exclude={"title", "body"}).replace(">", "\\u003e")
    return f"<!-- {_MARKER_TAG} {payload} -->"


def extract_markers(text: str) -> list[ReviewSignal]:
    """Parse all fuko-signal markers from ``text``, skipping malformed ones."""
    out: list[ReviewSignal] = []
    for m in _MARKER_RE.finditer(text or ""):
        try:
            out.append(ReviewSignal.model_validate_json(m.group(1)))
        except ValueError:
            continue
    return out


def strip_markers(text: str) -> str:
    """Remove any fuko-signal markers (and their surrounding blank lines) from ``text``."""
    return _MARKER_STRIP_RE.sub("", text or "")


def with_marker(body: str, signal: ReviewSignal) -> str:
    """Return ``body`` with ``signal``'s marker appended, replacing any existing marker."""
    return strip_markers(body).rstrip() + "\n\n" + encode_marker(signal)


def encode_run_marker(receipt: RunReceipt) -> str:
    """Render ``receipt`` as an invisible HTML-comment marker.

    Escapes ``>`` exactly as :func:`encode_marker` does, so no field value can
    close the HTML comment early.
    """
    payload = receipt.model_dump_json().replace(">", "\\u003e")
    return f"<!-- {_RUN_TAG} {payload} -->"


def extract_run_receipts(text: str) -> list[RunReceipt]:
    """Parse all fuko-run receipts from ``text``, skipping malformed ones."""
    out: list[RunReceipt] = []
    for m in _RUN_RE.finditer(text or ""):
        try:
            out.append(RunReceipt.model_validate_json(m.group(1)))
        except ValueError:
            continue
    return out


def with_run_receipt(body: str, receipt: RunReceipt) -> str:
    """Return ``body`` carrying ``receipt``, replacing any receipt already present.

    Replacing rather than appending is what keeps the branch header rewritable in
    place: the same comment is edited from ``in_progress`` to its final state, so a
    consumer always reads exactly one receipt per instance instead of having to
    pick the newest of a growing pile.
    """
    return _RUN_STRIP_RE.sub("", body or "").rstrip() + "\n\n" + encode_run_marker(receipt)


def with_markers(body: str, signals: list[ReviewSignal]) -> str:
    """Return ``body`` with every signal's marker appended, replacing existing markers.

    Unlike :func:`with_marker` (one marker per inline comment), this supports comments
    that carry SEVERAL findings — e.g. PR-Agent's "PR Reviewer Guide" issue comment,
    where each security concern and focus area gets its own marker. All existing fuko
    markers are stripped first and the fresh set appended, so re-running with a
    deterministically re-derived set is idempotent (same signals -> same body).
    """
    out = strip_markers(body).rstrip()
    for signal in signals:
        out += "\n\n" + encode_marker(signal)
    return out


_VISIBLE_LABEL_RE = re.compile(r"^🤖 `[^`]+`\n\n")


def visible_label(label: str) -> str:
    """Return the compact visible model tag prepended to A/B inline comments."""
    return f"🤖 `{label}`"


def strip_visible_label(text: str) -> str:
    """Remove a leading visible model tag from ``text``, if present.

    Anchored like :data:`_VISIBLE_LABEL_RE` itself: only a tag at the very start is
    publisher decoration; one appearing later is content.
    """
    return _VISIBLE_LABEL_RE.sub("", (text or "").lstrip("\n"))


def with_visible_label(body: str, label: str, signal: ReviewSignal) -> str:
    """Return ``body`` tagged with a visible ``label`` and ``signal``'s invisible marker.

    The visible tag makes the producing model legible on the diff (where both A/B
    branches attach to the same lines), while the marker keeps machine attribution
    intact. Any prior visible tag is stripped first so re-tagging stays idempotent.

    ``_VISIBLE_LABEL_RE`` is anchored to the absolute start of the string (no
    ``MULTILINE``): the tag is only ever prepended at the very beginning, so a
    ``🤖 `...` `` line appearing later in the body (e.g. quoted inside a suggestion)
    is legitimate content and is left intact. Only leading newlines are stripped
    before re-tagging — never indentation — so the anchored pattern reliably matches
    a prior tag while preserving any meaningful leading whitespace in the suggestion.
    """
    tagged = _VISIBLE_LABEL_RE.sub("", strip_markers(body).lstrip("\n"))
    return visible_label(label) + "\n\n" + with_marker(tagged, signal)
