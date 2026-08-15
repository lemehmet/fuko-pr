"""Tests for egress normalization: vendor comments -> Review Signals.

Fixtures are real comment bodies captured from a live PR (pr-agent + Copilot).
"""

import httpx
import pytest

from sidecar.backends import pragent
from sidecar.backends.base import PRRef
from sidecar.backends.pragent import PrAgentBackend
from sidecar.normalizers import (
    coderabbit_signal,
    collect_issue_comment_signals,
    collect_review_signals,
    collect_signals,
    copilot_signal,
    copilot_suppressed_signals,
    guide_signals,
    is_coderabbit_comment,
    is_coderabbit_finding,
    is_copilot_comment,
    is_guide_comment,
    is_pragent_comment,
    pragent_signal,
    pragent_signals,
    unrecognized_comments,
)
from sidecar.signals import (
    ReviewSignal,
    encode_marker,
    make_id,
    with_markers,
    with_visible_label,
)

PRAGENT = {
    "id": 111,
    "path": "src/lib/breakLogic.ts",
    "line": 4,
    "start_line": None,
    "html_url": "https://github.com/o/r/pull/8#discussion_r111",
    "user": {"login": "lemehmet", "id": 7},
    "body": (
        "**Suggestion:** When `completedFocusCount` is 0, the modulo condition is "
        "true (0 % 4 === 0), so the very first call returns a long break of 15 "
        "minutes instead of a short break. Add a guard to ensure the count is "
        "positive before applying the long-break rule. [possible issue, importance: 7]\n"
        "```suggestion\n  if (completedFocusCount > 0 && completedFocusCount % 4 === 0) "
        "return 15\n```"
    ),
}

COPILOT = {
    "id": 222,
    "path": "src/lib/breakLogic.ts",
    "line": 6,
    "html_url": "https://github.com/o/r/pull/8#discussion_r222",
    "user": {"login": "Copilot"},
    "body": "`completedFocusCount % 4 == 0` treats `0` as a long break, likely incorrect.",
}


def test_is_pragent_comment():
    assert is_pragent_comment(PRAGENT["body"])
    assert not is_pragent_comment(COPILOT["body"])
    assert not is_pragent_comment("")


def test_pragent_signal_maps_declared_fields():
    sig = pragent_signal(PRAGENT, model="anthropic/claude-sonnet-4-6")
    assert sig.file == "src/lib/breakLogic.ts"
    assert (sig.line, sig.end_line) == (4, None)
    assert sig.category == "bug"  # "possible issue"
    assert sig.severity == "high"  # importance 7
    assert sig.severity_source == "declared"
    assert sig.suggestion is True
    assert sig.backend == "pr-agent"
    assert sig.model == "anthropic/claude-sonnet-4-6"
    assert sig.title.startswith("When `completedFocusCount` is 0")
    assert "[possible issue, importance: 7]" not in sig.title  # label trimmed off
    assert sig.thread_url == PRAGENT["html_url"]


@pytest.mark.parametrize(
    "imp,expected",
    [(2, "low"), (5, "medium"), (7, "high"), (9, "critical")],
)
def test_severity_from_importance(imp, expected):
    c = dict(PRAGENT, body=f"**Suggestion:** x [possible issue, importance: {imp}]")
    assert pragent_signal(c).severity == expected


@pytest.mark.parametrize(
    "label,expected",
    [("security", "security"), ("performance", "perf"), ("best practice", "style")],
)
def test_category_mapping(label, expected):
    c = dict(PRAGENT, body=f"**Suggestion:** x [{label}, importance: 5]")
    assert pragent_signal(c).category == expected


def test_pragent_signal_without_label_is_inferred():
    c = dict(PRAGENT, body="**Suggestion:** tighten this type")
    sig = pragent_signal(c)
    assert sig.severity_source == "inferred"
    assert sig.severity == "medium"
    assert sig.category == "bug"


def test_pragent_signal_multiline_range():
    c = dict(PRAGENT, start_line=4, line=8)
    sig = pragent_signal(c)
    assert sig.line == 4
    assert sig.end_line == 8


def test_pragent_signals_filters_foreign_comments():
    pairs = pragent_signals([PRAGENT, COPILOT], model="m")
    assert len(pairs) == 1
    assert pairs[0]["comment"]["id"] == 111


def test_is_copilot_comment():
    assert is_copilot_comment(COPILOT)
    assert is_copilot_comment({"user": {"login": "copilot-pull-request-reviewer[bot]"}})
    assert not is_copilot_comment(PRAGENT)
    assert not is_copilot_comment({})


def test_copilot_signal_inferred_fields():
    sig = copilot_signal(COPILOT)
    assert sig.backend == "copilot"
    assert sig.model == ""
    assert sig.severity_source == "inferred"
    assert sig.severity == "medium"
    assert sig.file == "src/lib/breakLogic.ts"
    assert sig.line == 6
    assert sig.suggestion is False
    assert sig.title.startswith("`completedFocusCount % 4 == 0`")


# --- Copilot's collapsed "Suppressed comments" block (#109) -------------------
#
# The body below is TRIMMED FROM A REAL REVIEW on this repo's PR #100, where
# Copilot's visible prose read "generated no new comments" while the collapsed
# block carried three real findings — one of them a symlink-attack security
# issue. That review is why this is not a hypothetical channel.

_REAL_SUPPRESSED_BODY = """\
## Pull request overview

Copilot reviewed 13 out of 13 changed files in this pull request and generated \
no new comments.

<details>
<summary>Suppressed comments (3)</summary>

**sidecar/reviewer/checkout.py:6**
* Module docstring says the checkout is a "blob-filtered fetch", but \
`checkout_pr_head()` explicitly avoids blob filters. This is misleading.
```
sites and verify claims). The checkout is a shallow, blob-filtered fetch of the
```
**sidecar/reviewer/checkout.py:137**
* `strip_agent_config()` can follow symlinks: `Path.is_dir()` follows symlinks \
by default, so a malicious PR could commit `.claude` as a symlink to some \
sensitive directory and `shutil.rmtree()` would delete the target outside the \
checkout. This is a classic symlink attack on cleanup code.
**sidecar/backends/agentic.py:420**
* Unanchored findings are only rendered as plain markdown bullets.
</details>
"""


def _copilot_review(body):
    return {
        "user": {"login": "copilot-pull-request-reviewer[bot]"},
        "body": body,
        "html_url": "https://github.com/o/r/pull/100#pullrequestreview-1",
    }


def test_copilot_suppressed_block_yields_every_finding():
    """A review saying "no new comments" is not clean until this block is read."""
    signals = copilot_suppressed_signals(_copilot_review(_REAL_SUPPRESSED_BODY))
    assert len(signals) == 3
    assert [s.file for s in signals] == [
        "sidecar/reviewer/checkout.py",
        "sidecar/reviewer/checkout.py",
        "sidecar/backends/agentic.py",
    ]
    assert [s.line for s in signals] == [6, 137, 420]
    assert all(s.backend == "copilot" for s in signals)


def test_copilot_suppressed_findings_are_marked_suppressed():
    """Visible as a distinct sub-source: weighable, but not hideable."""
    signals = copilot_suppressed_signals(_copilot_review(_REAL_SUPPRESSED_BODY))
    assert all(s.suppressed is True for s in signals)


def test_copilot_suppressed_security_finding_infers_security():
    """The symlink-attack item is the one that most needed surfacing."""
    signals = copilot_suppressed_signals(_copilot_review(_REAL_SUPPRESSED_BODY))
    symlink = signals[1]
    assert symlink.category == "security"
    assert "symlink" in symlink.body


def test_copilot_suppressed_entry_body_stops_at_the_next_anchor():
    """Each finding gets its own text, not the remainder of the block."""
    signals = copilot_suppressed_signals(_copilot_review(_REAL_SUPPRESSED_BODY))
    assert "agentic.py" not in signals[1].body
    assert signals[0].title.startswith("Module docstring says")
    # The bullet marker is presentation; it must not leak into the title.
    assert not signals[0].title.startswith("*")


def test_copilot_review_without_a_suppressed_block_yields_nothing():
    body = "## Pull request overview\n\nCopilot reviewed 2 files and generated 1 comment."
    assert copilot_suppressed_signals(_copilot_review(body)) == []


def test_collect_review_signals_ignores_non_copilot_reviews():
    """CodeRabbit's review bodies are handled elsewhere; this channel is Copilot's."""
    cr = {"user": {"login": "coderabbitai[bot]"}, "body": _REAL_SUPPRESSED_BODY}
    assert collect_review_signals([cr]) == []
    assert len(collect_review_signals([_copilot_review(_REAL_SUPPRESSED_BODY)])) == 3


def test_collect_review_signals_recovers_fuko_markers_from_a_review_body():
    """#100's third suppressed finding, verified still live before this fix.

    The agentic backend marks every finding BEFORE choosing inline vs body, so an
    unanchored finding is posted marked in the review body — but nothing read
    review bodies back, so `normalize_output` returned it as a signal that could
    never be recovered from the PR.
    """
    sig = ReviewSignal(id="fk_unanchored", file="src/x.py", title="t", body="b", backend="agentic")
    review = {
        "user": {"login": "fuko-dorian[bot]"},
        "html_url": "https://github.com/o/r/pull/9#pullrequestreview-3",
        "body": (
            "## fuko agentic review\n\nsummary\n\n### Findings without a diff anchor\n\n"
            + with_markers("**Location:** `src/x.py`\n\nfinding text", [sig])
        ),
    }
    (out,) = collect_review_signals([review])
    assert out.id == "fk_unanchored"
    assert out.backend == "agentic"
    assert out.thread_url == review["html_url"]


def test_suppressed_signal_ids_are_stable_and_distinct():
    """Same inputs -> same id, so a consumer can dedupe across runs."""
    first = copilot_suppressed_signals(_copilot_review(_REAL_SUPPRESSED_BODY))
    second = copilot_suppressed_signals(_copilot_review(_REAL_SUPPRESSED_BODY))
    assert [s.id for s in first] == [s.id for s in second]
    assert len({s.id for s in first}) == 3


def test_copilot_category_inference():
    sec = dict(COPILOT, body="This is a SQL injection vulnerability.")
    perf = dict(COPILOT, body="This causes an N+1 query and is slow.")
    plain = dict(COPILOT, body="Rename this variable for clarity.")
    assert copilot_signal(sec).category == "security"
    assert copilot_signal(perf).category == "perf"
    assert copilot_signal(plain).category == "bug"


def test_collect_signals_dispatches_per_vendor():
    other = {"user": {"login": "some-human"}, "body": "lgtm", "path": "x", "line": 1}
    signals = collect_signals([PRAGENT, COPILOT, other], model="anthropic/claude")
    assert [s.backend for s in signals] == ["pr-agent", "copilot"]
    assert signals[0].severity_source == "declared"
    assert signals[1].severity_source == "inferred"


CODERABBIT = {
    "id": 333,
    "path": "apps/web/src/hooks/use-webrtc.ts",
    "line": 87,
    "start_line": None,
    "html_url": "https://github.com/o/r/pull/9#discussion_r333",
    "user": {"login": "coderabbitai[bot]"},
    "body": (
        "_⚠️ Potential issue_ | _🔴 Critical_\n\n"
        "**Make media-resolution gating reactive, or calls can still stall.**\n\n"
        "`localMediaResolvedRef` updates won't re-run Effect 3.\n\n"
        "<details>\n<summary>Suggested fix</summary>\n\n```diff\n- a\n+ b\n```\n</details>\n\n"
        "<!-- This is an auto-generated comment by CodeRabbit -->"
    ),
}

CODERABBIT_CHAT = {
    "id": 334,
    "user": {"login": "coderabbitai[bot]"},
    "body": "`@lemehmet`, you're right — my concern was incorrect. Thanks for the clarification.",
}


def test_is_coderabbit_comment_and_finding():
    assert is_coderabbit_comment(CODERABBIT)
    assert is_coderabbit_finding(CODERABBIT["body"])
    # author matches but it's a chat reply, not a finding
    assert is_coderabbit_comment(CODERABBIT_CHAT)
    assert not is_coderabbit_finding(CODERABBIT_CHAT["body"])
    assert not is_coderabbit_comment(COPILOT)


def test_coderabbit_signal_declared_fields():
    sig = coderabbit_signal(CODERABBIT)
    assert sig.backend == "coderabbit"
    assert sig.severity == "critical"
    assert sig.severity_source == "declared"
    assert sig.category == "bug"  # "Potential issue"
    assert sig.suggestion is True  # "Suggested fix"
    assert sig.file == "apps/web/src/hooks/use-webrtc.ts"
    assert sig.line == 87
    assert sig.title == "Make media-resolution gating reactive, or calls can still stall."


@pytest.mark.parametrize(
    "cls,severity,category",
    [
        ("_⚠️ Potential issue_ | _🟠 Major_ | _⚡ Quick win_", "high", "bug"),
        ("_⚠️ Potential issue_ | _🟡 Minor_", "medium", "bug"),
        ("_🧹 Nitpick_ | _🔵 Trivial_", "low", "style"),
        ("_🛠️ Refactor suggestion_ | _🟠 Major_", "high", "design"),
        ("_🔒 Security_ | _🔴 Critical_", "critical", "security"),
        ("_🐢 Performance issue_ | _🟡 Minor_", "medium", "perf"),
        ("_✏️ Typo_ | _🔵 Trivial_", "low", "docs"),
    ],
)
def test_coderabbit_severity_and_category_mapping(cls, severity, category):
    c = dict(CODERABBIT, body=f"{cls}\n\n**t**\n\nbody")
    sig = coderabbit_signal(c)
    assert sig.severity == severity
    assert sig.category == category
    assert sig.severity_source == "declared"


def test_collect_signals_includes_coderabbit_findings_only():
    signals = collect_signals([PRAGENT, COPILOT, CODERABBIT, CODERABBIT_CHAT], model="m")
    assert [s.backend for s in signals] == ["pr-agent", "copilot", "coderabbit"]


def test_collect_signals_prefers_embedded_marker():
    # A fuko-pr comment carrying a review-time marker (model glm-5.2, severity high)
    marker = encode_marker(
        ReviewSignal(
            id="fk_reviewtime",
            file="x.py",
            line=10,
            severity="high",
            severity_source="declared",
            category="security",
            backend="pr-agent",
            model="openai/glm-5.2",
        )
    )
    body = (
        "**Suggestion:** tighten this input handling [possible issue, importance: 4]\n"
        "```suggestion\nfix\n```\n\n" + marker
    )
    c = {
        "path": "x.py",
        "line": 10,
        "html_url": "u",
        "user": {"login": "fuko-pr-review[bot]"},
        "body": body,
    }

    # run with the WRONG local default model — the marker must win
    [sig] = collect_signals([c], model="ollama/qwen2.5-coder")
    assert sig.model == "openai/glm-5.2"  # from the marker, not the local config
    assert sig.id == "fk_reviewtime"
    assert sig.severity == "high"  # marker, not importance-4-derived "medium"
    assert sig.category == "security"
    # human fields are kept from the live parse (the marker excludes them)
    assert sig.title.startswith("tighten this input handling")
    assert "**Suggestion:**" in sig.body


def test_collect_signals_without_marker_uses_local_model():
    # no marker -> model comes from the passed config (unchanged behavior)
    [sig] = collect_signals([PRAGENT], model="anthropic/claude")
    assert sig.model == "anthropic/claude"


def _published(body: str, signal: ReviewSignal, label: str = "openrouter/x-ai/grok-4.5") -> dict:
    """Build a comment exactly as fuko publishes it: visible label + marker."""
    return {
        "path": signal.file,
        "line": signal.line,
        "html_url": "https://github.com/o/r/pull/8#discussion_r999",
        "user": {"login": "fuko-sybil[bot]"},
        "body": with_visible_label(body, label, signal),
    }


REVIEW_TIME = ReviewSignal(
    id="fk_published",
    file="x.py",
    line=10,
    severity="high",
    severity_source="declared",
    category="security",
    backend="pr-agent",
    model="openrouter/x-ai/grok-4.5",
)


def test_is_pragent_comment_tolerates_publisher_decoration():
    # fuko prepends a visible model label before posting; the suggestion is no longer
    # at position 0. Anchoring there dropped 39/39 of fuko's own findings in mepro #1629.
    decorated = with_visible_label(PRAGENT["body"], "openrouter/x-ai/grok-4.5", REVIEW_TIME)
    assert not decorated.lstrip().startswith("**Suggestion:**")  # the shape that broke it
    assert is_pragent_comment(decorated)


def test_collect_signals_admits_published_fuko_comment():
    c = _published(PRAGENT["body"], REVIEW_TIME)
    [sig] = collect_signals([c], model="ollama/qwen2.5-coder")
    assert sig.id == "fk_published"
    assert sig.model == "openrouter/x-ai/grok-4.5"  # marker wins over local config
    assert sig.severity == "high"
    assert "modulo condition" in sig.body


def test_collect_signals_admits_marker_with_unrecognizable_prose():
    # The whole point of the marker: prose format may drift arbitrarily, but a
    # comment fuko wrote must never silently vanish from the signal set.
    c = _published("Completely unrecognizable prose in a future format.", REVIEW_TIME)
    [sig] = collect_signals([c], model="ollama/qwen2.5-coder")
    assert sig.id == "fk_published"
    assert sig.severity == "high"
    assert sig.category == "security"
    assert sig.title == "Completely unrecognizable prose in a future format."
    assert "fuko-signal" not in sig.body  # marker stripped from the human-facing text
    assert sig.thread_url == "https://github.com/o/r/pull/8#discussion_r999"


def test_collect_signals_marker_admission_does_not_resurrect_coderabbit_chat():
    # CodeRabbit chat is dropped by design, not by recognizer failure — it stays dropped.
    assert collect_signals([CODERABBIT_CHAT], model="m") == []


def test_prefer_marker_keeps_thread_url_when_marker_lacks_one():
    bare = REVIEW_TIME.model_copy(update={"thread_url": None})
    c = _published(PRAGENT["body"], bare)
    [sig] = collect_signals([c], model="m")
    assert sig.thread_url == "https://github.com/o/r/pull/8#discussion_r999"


def test_collect_signals_skips_replies_that_quote_a_finding():
    # Marker admission makes this reachable: a reply quoting a finding's body carries
    # that finding's marker verbatim, and would be collected a second time.
    original = _published(PRAGENT["body"], REVIEW_TIME)
    quoting_reply = {
        "id": 777,
        "in_reply_to_id": 999,
        "user": {"login": "lemehmet"},
        "body": "> " + original["body"].replace("\n", "\n> ") + "\n\nFixed in abc1234.",
    }
    assert len(collect_signals([original, quoting_reply], model="m")) == 1


def test_is_pragent_comment_ignores_a_blockquoted_suggestion():
    # Quoting a finding is not posting one. Matching inside a blockquote is what
    # made the duplicate-reply case reachable in the first place.
    assert not is_pragent_comment("> **Suggestion:** quoted from someone else\n\nmy reply")
    assert is_pragent_comment("🤖 `m`\n\n**Suggestion:** a real one")


def test_unrecognized_comments_excludes_recognized_but_skipped_coderabbit_chat():
    # CR chat/rate-limit notices are claimed by a recognizer and then dropped on
    # purpose. Reporting them as unreadable is inaccurate and makes the warning noise.
    assert unrecognized_comments([CODERABBIT_CHAT], model="m") == []


def test_unrecognized_comments_reports_only_unclaimed_top_level():
    human = {
        "id": 900,
        "html_url": "u",
        "user": {"login": "lemehmet"},
        "body": "looks good to me",
    }
    reply = {
        "id": 901,
        "in_reply_to_id": 900,
        "user": {"login": "lemehmet"},
        "body": "agreed",
    }
    dropped = unrecognized_comments(
        [PRAGENT, COPILOT, _published(PRAGENT["body"], REVIEW_TIME), human, reply], model="m"
    )
    assert [c["id"] for c in dropped] == [900]


def test_normalize_output_returns_only_pragent_signals(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghtok")
    monkeypatch.setattr(
        PrAgentBackend, "_fetch_review_comments", lambda self, a, p, h: [PRAGENT, COPILOT]
    )
    monkeypatch.setattr(PrAgentBackend, "_fetch_issue_comments", lambda self, a, p, h: [])
    injected = []
    monkeypatch.setattr(
        PrAgentBackend,
        "_inject_markers",
        lambda self, a, p, h, pairs, label=None, actor=None: injected.append((pairs, label)),
    )
    sigs = PrAgentBackend().normalize_output(PRRef("o/r", 8, "u"), model="anthropic/claude")
    assert [s.severity for s in sigs] == ["high"]
    assert len(injected) == 1
    assert len(injected[0][0]) == 1
    assert injected[0][1] is None


def test_normalize_output_excludes_comments_an_earlier_round_already_marked(monkeypatch):
    # `findings` from normalize_output feeds the per-run review_runs metric, so a
    # comment this branch posted AND marked on a previous round must not be counted
    # again. Before #1629 the strict `**Suggestion:**` anchor excluded these by
    # accident (a marked comment also carries a visible label); now it is explicit.
    already_marked = _published(PRAGENT["body"], REVIEW_TIME)
    fresh = dict(PRAGENT)
    monkeypatch.setenv("GITHUB_TOKEN", "ghtok")
    monkeypatch.setattr(
        PrAgentBackend, "_fetch_review_comments", lambda self, a, p, h: [already_marked, fresh]
    )
    monkeypatch.setattr(PrAgentBackend, "_fetch_issue_comments", lambda self, a, p, h: [])
    monkeypatch.setattr(
        PrAgentBackend, "_inject_markers", lambda self, a, p, h, pairs, label=None, actor=None: None
    )
    sigs = PrAgentBackend().normalize_output(PRRef("o/r", 8, "u"), model="anthropic/claude")
    assert len(sigs) == 1  # only the fresh one; the marked one is a previous round's


def test_normalize_output_degrades_when_fetch_fails(monkeypatch):
    def boom(self, a, p, h):
        raise httpx.HTTPError("nope")

    monkeypatch.setattr(PrAgentBackend, "_fetch_review_comments", boom)
    assert PrAgentBackend().normalize_output(PRRef("o/r", 8, "u")) == []


class _PatchClient:
    """Fake httpx.Client capturing PATCH calls.

    ``GET /user`` resolves the marking identity to actor id ``7`` -- the same id as
    the ``PRAGENT`` fixture's author -- so the author-filter in ``_inject_markers``
    keeps that comment. ``actor_id`` can be overridden per test to simulate marking
    under a *different* identity (where the sibling's comment is skipped).
    """

    calls: list = []
    actor_id: int = 7

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        assert url.endswith("/user")

        class _R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"id": _PatchClient.actor_id}

        return _R()

    def patch(self, url, json):
        _PatchClient.calls.append((url, json))

        class _R:
            def raise_for_status(self):
                return None

        return _R()


_AUTH = {"Authorization": "Bearer t"}


def test_inject_markers_patches_unmarked(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    pairs = pragent_signals([PRAGENT], model="m")
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, pairs)
    assert len(_PatchClient.calls) == 1
    url, payload = _PatchClient.calls[0]
    assert url.endswith("/pulls/comments/111")
    assert "fuko-signal:v1" in payload["body"]


def test_inject_markers_skips_already_marked(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    sig = pragent_signal(PRAGENT, model="m")
    marked = dict(PRAGENT, body=PRAGENT["body"] + "\n\n" + encode_marker(sig))
    PrAgentBackend()._inject_markers(
        "https://api", PRRef("o/r", 8, "u"), _AUTH, [{"comment": marked, "signal": sig}]
    )
    assert _PatchClient.calls == []


def test_inject_markers_skips_marker_from_another_model(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    other = pragent_signal(PRAGENT, model="model-a")
    marked = dict(PRAGENT, body=PRAGENT["body"] + "\n\n" + encode_marker(other))
    mine = pragent_signal(marked, model="model-b")
    PrAgentBackend()._inject_markers(
        "https://api", PRRef("o/r", 8, "u"), _AUTH, [{"comment": marked, "signal": mine}]
    )
    assert _PatchClient.calls == []


def test_inject_markers_adds_visible_label_in_compare_mode(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    pairs = pragent_signals([PRAGENT], model="anthropic/claude")
    PrAgentBackend()._inject_markers(
        "https://api", PRRef("o/r", 8, "u"), _AUTH, pairs, label="anthropic/claude"
    )
    assert len(_PatchClient.calls) == 1
    body = _PatchClient.calls[0][1]["body"]
    assert body.startswith("🤖 `anthropic/claude`\n\n")
    assert "fuko-signal:v1" in body


def test_inject_markers_no_visible_label_without_compare(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    pairs = pragent_signals([PRAGENT], model="anthropic/claude")
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, pairs)
    body = _PatchClient.calls[0][1]["body"]
    assert "🤖" not in body
    assert "fuko-signal:v1" in body


def test_normalize_output_passes_compare_label_through(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghtok")
    monkeypatch.setattr(PrAgentBackend, "_fetch_review_comments", lambda self, a, p, h: [PRAGENT])
    monkeypatch.setattr(PrAgentBackend, "_fetch_issue_comments", lambda self, a, p, h: [])
    seen = []
    monkeypatch.setattr(
        PrAgentBackend,
        "_inject_markers",
        lambda self, a, p, h, pairs, label=None, actor=None: seen.append(label),
    )
    backend = PrAgentBackend()
    # No compare_label → no visible tag. A compare_label distinct from ``model``
    # (the marker id) is passed through verbatim as the visible label, so a
    # ``zai-coding`` branch tags ``zai-coding/glm`` rather than its litellm
    # alias ``openai/glm``.
    backend.normalize_output(PRRef("o/r", 8, "u"), model="openai/glm")
    backend.normalize_output(
        PRRef("o/r", 8, "u"), model="openai/glm", compare_label="zai-coding/glm"
    )
    assert seen == [None, "zai-coding/glm"]


def test_inject_markers_empty_is_noop(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, [])
    assert _PatchClient.calls == []


def test_inject_markers_skips_when_unauthenticated(monkeypatch):
    _PatchClient.calls = []
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    pairs = pragent_signals([PRAGENT], model="m")
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), {}, pairs)
    assert _PatchClient.calls == []


def test_inject_markers_skips_on_patch_error(monkeypatch):
    class _ErrClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, params=None):
            class _R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"id": 7}  # matches the PRAGENT author, so it isn't filtered

            return _R()

        def patch(self, url, json):
            raise httpx.HTTPError("403 not your comment")

    monkeypatch.setattr(pragent.httpx, "Client", _ErrClient)
    pairs = pragent_signals([PRAGENT], model="m")
    PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, pairs)


def test_inject_markers_skips_sibling_authored_comment(monkeypatch):
    # Concurrent A/B mode: marking under actor id 99 must NOT PATCH the PRAGENT
    # comment authored by actor id 7 (a sibling branch's). Without the author
    # filter every such PATCH would 403 and burn API quota.
    _PatchClient.calls = []
    _PatchClient.actor_id = 99
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    try:
        pairs = pragent_signals([PRAGENT], model="m")
        PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, pairs)
        assert _PatchClient.calls == []
    finally:
        _PatchClient.actor_id = 7


def test_inject_markers_marks_all_when_actor_unresolved(monkeypatch):
    # If GET /user can't resolve the identity, fall back to marking best-effort
    # across all comments (prior behavior) rather than skip everything.
    _PatchClient.calls = []
    _PatchClient.actor_id = None
    monkeypatch.setattr(pragent.httpx, "Client", _PatchClient)
    try:
        pairs = pragent_signals([PRAGENT], model="m")
        PrAgentBackend()._inject_markers("https://api", PRRef("o/r", 8, "u"), _AUTH, pairs)
        assert len(_PatchClient.calls) == 1
    finally:
        _PatchClient.actor_id = 7


class _GetClient:
    def __init__(self, pages):
        self._pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params):
        batch = self._pages.get(params["page"], [])

        class _R:
            def raise_for_status(self):
                return None

            def json(self):
                return batch

        return _R()


def test_fetch_review_comments_paginates(monkeypatch):
    pages = {1: [{"id": i} for i in range(100)], 2: [{"id": 999}]}
    monkeypatch.setattr(pragent.httpx, "Client", lambda *a, **k: _GetClient(pages))
    out = PrAgentBackend()._fetch_review_comments("https://api", PRRef("o/r", 8, "u"), {})
    assert len(out) == 101
    assert out[-1]["id"] == 999


def test_fetch_review_comments_empty(monkeypatch):
    monkeypatch.setattr(pragent.httpx, "Client", lambda *a, **k: _GetClient({1: []}))
    out = PrAgentBackend()._fetch_review_comments("https://api", PRRef("o/r", 8, "u"), {})
    assert out == []


def test_fetch_issue_comments_paginates(monkeypatch):
    pages = {1: [{"id": i} for i in range(100)], 2: [{"id": 555}]}
    monkeypatch.setattr(pragent.httpx, "Client", lambda *a, **k: _GetClient(pages))
    out = PrAgentBackend()._fetch_issue_comments("https://api", PRRef("o/r", 8, "u"), {})
    assert len(out) == 101
    assert out[-1]["id"] == 555


# --- PR Reviewer Guide (issue comment) parsing -------------------------------

GUIDE = {
    "id": 555,
    "html_url": "https://github.com/o/r/pull/8#issuecomment-555",
    "user": {"login": "fuko-pr-review[bot]", "id": 7},
    "body": (
        "## PR Reviewer Guide 🔍\n\n"
        "Here are some key observations to aid the review process:\n\n"
        "<table>\n"
        "<tr><td>⏱️&nbsp;<strong>Estimated effort to review</strong>: 4 🔵🔵🔵🔵⚪</td></tr>\n"
        "<tr><td>🧪&nbsp;<strong>PR contains tests</strong></td></tr>\n"
        "<tr><td>🔒&nbsp;<strong>Security concerns</strong><br><br>"
        '<strong>Possible incomplete redaction:</strong><br> the `Fill "hunter2"`-style '
        "test values may leak into workflow logs.</td></tr>\n"
        "<tr><td>⚡&nbsp;<strong>Recommended focus areas for review</strong><br><br>"
        "<details><summary><a href='https://github.com/o/r/pull/8/files#diff-abc123R10-R20'>"
        "<strong>Regex Gap</strong></a>\n\n"
        "The redaction regex misses multiline values in sidecar/redact.py:42 and beyond.\n"
        "</summary>\n\n"
        '```txt\nFill "hunter2"\n```\n\n'
        "</details>"
        "<details><summary><strong>Race Condition</strong>\n\n"
        "Concurrent writers may clobber the sink registry.\n"
        "</summary>\n\n"
        "```txt\nregistry[key] = sink\n```\n\n"
        "</details>\n\n"
        "</td></tr>\n"
        "</table>"
    ),
}


def test_is_guide_comment():
    assert is_guide_comment(GUIDE["body"])
    assert not is_guide_comment(PRAGENT["body"])
    assert not is_guide_comment("")


def test_guide_signals_security_cell():
    sigs = guide_signals(GUIDE, model="anthropic/claude")
    [sec] = [s for s in sigs if s.category == "security"]
    assert sec.title == "Possible incomplete redaction"
    assert sec.severity == "medium"
    assert sec.severity_source == "inferred"
    assert "may leak into workflow logs" in sec.body
    assert sec.thread_url == GUIDE["html_url"]
    assert sec.backend == "pr-agent"
    assert sec.model == "anthropic/claude"
    assert sec.id == make_id("guide", "555", "Possible incomplete redaction")


def test_guide_signals_focus_areas():
    sigs = guide_signals(GUIDE, model="m")
    focus = [s for s in sigs if s.category == "bug"]
    assert [s.title for s in focus] == ["Regex Gap", "Race Condition"]
    linked, plain = focus
    # href fragment can't yield a path, but a literal path:line in the text can
    assert (linked.file, linked.line) == ("sidecar/redact.py", 42)
    assert "misses multiline values" in linked.body
    assert (plain.file, plain.line) == (None, None)
    assert "clobber the sink registry" in plain.body
    for s in focus:
        assert (s.severity, s.severity_source) == ("medium", "inferred")
        assert s.backend == "pr-agent"
        assert s.thread_url == GUIDE["html_url"]


def test_guide_signals_are_deterministic():
    a = guide_signals(GUIDE, model="m")
    b = guide_signals(GUIDE, model="m")
    assert [s.id for s in a] == [s.id for s in b]
    assert len({s.id for s in a}) == 3


def test_guide_signals_no_op_variants_emit_nothing():
    body = (
        "## PR Reviewer Guide 🔍\n\n<table>\n"
        "<tr><td>🔒&nbsp;<strong>No security concerns identified</strong></td></tr>\n"
        "<tr><td>⚡&nbsp;<strong>No major issues detected</strong></td></tr>\n"
        "</table>"
    )
    assert guide_signals(dict(GUIDE, body=body)) == []


def test_guide_signals_security_without_lead_phrase_gets_default_title():
    body = (
        "## PR Reviewer Guide 🔍\n\n<table>\n"
        "<tr><td>🔒&nbsp;<strong>Security concerns</strong><br><br>"
        "Plain free text without a bolded lead.</td></tr>\n</table>"
    )
    [sig] = guide_signals(dict(GUIDE, body=body))
    assert sig.title == "Security concern"
    assert sig.body == "Plain free text without a bolded lead."


def test_guide_signals_empty_security_cell_emits_nothing():
    body = (
        "## PR Reviewer Guide 🔍\n\n<table>\n"
        "<tr><td>🔒&nbsp;<strong>Security concerns</strong><br><br></td></tr>\n</table>"
    )
    assert guide_signals(dict(GUIDE, body=body)) == []


def test_guide_signals_tolerates_malformed_bodies():
    for body in [
        "not a guide at all",
        "",
        "## PR Reviewer Guide 🔍\n\nno table here",
        "## PR Reviewer Guide 🔍\n\n<table><tr><td>broken",
        "## PR Reviewer Guide 🔍\n\n<table><tr><td>⚡&nbsp;"
        "<strong>Recommended focus areas for review</strong><br><br>"
        "<details>no summary tag</details></td></tr></table>",
        "## PR Reviewer Guide 🔍\n\n<table><tr><td>⚡&nbsp;"
        "<strong>Recommended focus areas for review</strong><br><br>"
        "<details><summary><strong></strong></summary></details></td></tr></table>",
    ]:
        assert guide_signals(dict(GUIDE, body=body)) == []
    assert guide_signals({}) == []
    # a non-string body must be swallowed too, never raise
    assert guide_signals(dict(GUIDE, body=12345)) == []


def test_guide_signals_summary_without_strong_uses_first_line():
    body = (
        "## PR Reviewer Guide 🔍\n\n<table>\n"
        "<tr><td>⚡&nbsp;<strong>Recommended focus areas for review</strong><br><br>"
        "<details><summary>Plain Title\n\nThe description follows the title line.\n"
        "</summary>\n\n</details></td></tr>\n</table>"
    )
    [sig] = guide_signals(dict(GUIDE, body=body))
    assert sig.title == "Plain Title"
    assert "description follows" in sig.body


def test_guide_signals_tolerate_tag_attributes():
    # PR-Agent may emit attributes on the structural tags (<td align=...>,
    # <details open>); parsing must not silently yield nothing (Copilot, #73).
    body = (
        "## PR Reviewer Guide 🔍\n\n<table>\n"
        '<tr><td align="left">⚡&nbsp;<strong>Recommended focus areas for review</strong><br><br>'
        '<details open><summary class="x"><strong>Attr Tag</strong>\n\n'
        "Body under an attributed summary.\n</summary>\n\n</details></td></tr>\n</table>"
    )
    [sig] = guide_signals(dict(GUIDE, body=body))
    assert sig.title == "Attr Tag"
    assert "attributed summary" in sig.body


def test_authored_by_matches_int_user_id():
    from sidecar.backends.pragent import PrAgentBackend

    assert PrAgentBackend._authored_by({"user": {"id": 42}}, "42") is True
    assert PrAgentBackend._authored_by({"user": {"id": 99}}, "42") is False
    assert PrAgentBackend._authored_by({"user": {"id": 42}}, None) is False
    assert PrAgentBackend._authored_by({}, "42") is False


def test_collect_issue_comment_signals_rehydrates_guide_titles():
    sigs = guide_signals(GUIDE, model="openai/glm-5.2")
    marked = dict(GUIDE, body=with_markers(GUIDE["body"], sigs))
    # collected under a DIFFERENT local model — marker attribution must win
    out = collect_issue_comment_signals([marked], model="ollama/qwen")
    assert len(out) == 3
    assert {s.model for s in out} == {"openai/glm-5.2"}
    assert [s.title for s in out] == [
        "Possible incomplete redaction",
        "Regex Gap",
        "Race Condition",
    ]
    assert all(s.body for s in out)
    assert {s.id for s in out} == {s.id for s in sigs}


def test_collect_issue_comment_signals_marker_only_comment():
    marker_sig = ReviewSignal(id="fk_solo", backend="pr-agent", model="m", category="security")
    body = "some other bot output\n\n" + encode_marker(marker_sig)
    [out] = collect_issue_comment_signals(
        [{"id": 1, "html_url": "https://x/#c1", "body": body}], model=""
    )
    assert out.id == "fk_solo"
    assert out.thread_url == "https://x/#c1"  # filled from the comment payload
    assert out.title == ""  # no fresh parse available to rehydrate


def test_collect_issue_comment_signals_skips_unmarked():
    assert collect_issue_comment_signals([{"id": 1, "body": "walkthrough"}, {}]) == []


def test_normalize_output_includes_guide_signals(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghtok")
    monkeypatch.setattr(PrAgentBackend, "_fetch_review_comments", lambda self, a, p, h: [PRAGENT])
    walkthrough = {"id": 9, "body": "a non-guide issue comment (e.g. CodeRabbit walkthrough)"}
    empty_guide = {"id": 10, "body": "## PR Reviewer Guide 🔍\n\nno table -> no signals"}
    monkeypatch.setattr(
        PrAgentBackend,
        "_fetch_issue_comments",
        lambda self, a, p, h: [walkthrough, GUIDE, empty_guide],
    )
    monkeypatch.setattr(
        PrAgentBackend, "_inject_markers", lambda self, a, p, h, pairs, label=None, actor=None: None
    )
    marked = []
    monkeypatch.setattr(
        PrAgentBackend,
        "_mark_guide_comments",
        lambda self, a, p, h, pairs, label=None, actor=None: marked.append((pairs, label)),
    )
    sigs = PrAgentBackend().normalize_output(PRRef("o/r", 8, "u"), model="anthropic/claude")
    assert len(sigs) == 4  # 1 inline + 1 security + 2 focus areas
    assert {s.backend for s in sigs} == {"pr-agent"}
    assert [s.category for s in sigs[1:]] == ["security", "bug", "bug"]
    # the guide pairs handed to the marking step carry the same parsed signals
    assert len(marked) == 1
    assert len(marked[0][0]) == 1
    assert len(marked[0][0][0]["signals"]) == 3


def test_normalize_output_degrades_when_issue_fetch_fails(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "ghtok")
    monkeypatch.setattr(PrAgentBackend, "_fetch_review_comments", lambda self, a, p, h: [PRAGENT])
    monkeypatch.setattr(
        PrAgentBackend, "_inject_markers", lambda self, a, p, h, pairs, label=None, actor=None: None
    )

    def boom(self, a, p, h):
        raise httpx.HTTPError("nope")

    monkeypatch.setattr(PrAgentBackend, "_fetch_issue_comments", boom)
    sigs = PrAgentBackend().normalize_output(PRRef("o/r", 8, "u"), model="m")
    assert [s.file for s in sigs] == ["src/lib/breakLogic.ts"]  # inline survives
    assert "could not read issue comments" in capsys.readouterr().err
