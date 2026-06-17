"""Unit tests for ``/remember`` and ``/forget`` comment parsing."""

from sidecar.commands import parse_forget, parse_remember


def test_parse_remember_basic():
    assert parse_remember("/remember use absolute imports") == ("use absolute imports", [])


def test_parse_remember_with_paths():
    result = parse_remember("/remember: do X\npaths: src/**/*.py, tests/*.py")
    assert result == ("do X", ["src/**/*.py", "tests/*.py"])


def test_parse_remember_separators():
    assert parse_remember("/remember: hi") == ("hi", [])
    assert parse_remember("/remember - hi") == ("hi", [])


def test_parse_remember_empty_or_paths_only():
    assert parse_remember("/remember") is None
    assert parse_remember("/remember paths: x") is None


def test_parse_remember_not_a_command():
    assert parse_remember("nice work") is None


def test_parse_forget_all():
    assert parse_forget("/forget all") == {"all": True}


def test_parse_forget_source():
    assert parse_forget("/forget source=docs") == {"source": "docs"}


def test_parse_forget_id():
    uid = "11111111-2222-3333-4444-555555555555"
    assert parse_forget("/forget " + uid) == {"id": uid}


def test_parse_forget_empty_or_other():
    assert parse_forget("/forget") is None
    assert parse_forget("hello") is None
