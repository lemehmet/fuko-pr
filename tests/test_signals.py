"""Unit tests for Review Signal v1 markers and stable ids."""

from sidecar.signals import (
    ReviewSignal,
    encode_marker,
    extract_markers,
    make_id,
    strip_markers,
    with_marker,
)


def _signal(**kw):
    base = dict(
        id="fk_abc123",
        file="src/auth/login.py",
        line=42,
        end_line=48,
        severity="high",
        severity_source="declared",
        category="security",
        title="SQL injection",
        body="Concatenated query; use parameters.",
        suggestion=True,
        thread_url="https://github.com/o/r/pull/1#discussion_r1",
        backend="pr-agent",
        model="anthropic/claude-sonnet-4-6",
        kb_refs=["resolved_thread:9"],
    )
    base.update(kw)
    return ReviewSignal(**base)


def test_make_id_is_stable_and_prefixed():
    a = make_id("src/x.py", "42", "title")
    b = make_id("src/x.py", "42", "title")
    assert a == b
    assert a.startswith("fk_")
    assert make_id("src/x.py", "43", "title") != a


def test_marker_round_trips_machine_fields():
    sig = _signal()
    body = with_marker("Here is the finding.", sig)
    [got] = extract_markers(body)
    assert got.id == sig.id
    assert got.severity == "high"
    assert got.severity_source == "declared"
    assert got.category == "security"
    assert got.file == "src/auth/login.py"
    assert (got.line, got.end_line) == (42, 48)
    assert got.suggestion is True
    assert got.thread_url == sig.thread_url
    assert got.backend == "pr-agent"
    assert got.kb_refs == ["resolved_thread:9"]


def test_marker_excludes_human_text():
    marker = encode_marker(_signal())
    assert "SQL injection" not in marker
    assert "parameters" not in marker
    # extracted signal has empty human fields (they live in the visible comment)
    [got] = extract_markers(marker)
    assert got.title == ""
    assert got.body == ""


def test_marker_escapes_arrow_in_field_values():
    # a field value containing '-->' must not terminate the HTML comment early
    sig = _signal(id="fk_z", thread_url="https://x/y?q=a-->b", file="weird-->name.py")
    marker = encode_marker(sig)
    # only the trailing comment terminator is a literal '-->'
    assert marker.count("-->") == 1
    assert marker.endswith("-->")
    body = with_marker("text", sig)
    [got] = extract_markers(body)
    assert got.thread_url == "https://x/y?q=a-->b"
    assert got.file == "weird-->name.py"


def test_extract_multiple_and_ignores_foreign_comments():
    text = "\n".join(
        [
            encode_marker(_signal(id="fk_one")),
            "<!-- some other html comment -->",
            "<!-- fuko-signal:v1 {not valid json} -->",
            encode_marker(_signal(id="fk_two")),
        ]
    )
    ids = sorted(s.id for s in extract_markers(text))
    assert ids == ["fk_one", "fk_two"]


def test_extract_handles_no_markers():
    assert extract_markers("just a normal comment") == []
    assert extract_markers("") == []


def test_strip_markers_removes_marker():
    body = with_marker("visible text", _signal())
    stripped = strip_markers(body)
    assert "fuko-signal" not in stripped
    assert "visible text" in stripped


def test_with_marker_is_idempotent():
    once = with_marker("visible text", _signal(id="fk_x"))
    twice = with_marker(once, _signal(id="fk_x"))
    assert twice.count("fuko-signal:v1") == 1
    assert extract_markers(twice)[0].id == "fk_x"
