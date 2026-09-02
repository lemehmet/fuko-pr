"""Tests for ``fuko transcripts`` (the network call is faked), #240."""

import argparse
import http.client
import io
import json
import urllib.error

import pytest

from sidecar import cli, transcriptscli

_ROW = {
    "key": "20260901T120000Z-a1b2c3d4e5f6",
    "created_at": "2026-09-01T12:00:00+00:00",
    "complete": True,
    "tool_calls": {"Read": 182, "Grep": 9, "Bash": 23, "Edit": 2, "Glob": 1},
    "tool_result_bytes": 41_943_040,
    "repeated_read_files": 37,
    "repo": "lemehmet/mepro",
    "pr": 42,
    "seat": "dorian",
    "provider": "openrouter",
    "model": "qwen3.8-max",
    "backend": "agentic",
    "outcome": "ok",
    "started_at": "2026-09-01T11:58:00+00:00",
    "duration_s": 612.5,
}


def _ns(**kw):
    defaults = dict(
        repo=None,
        pr=None,
        seat=None,
        since=None,
        until=None,
        limit=50,
        offset=0,
        full=False,
        json=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _DyingResponse(_FakeResponse):
    """A response that opens fine and then dies partway through its body.

    The failure urllib does NOT wrap: everything after the headers is a plain
    socket read, so a stall or a reset arrives as ``TimeoutError`` /
    ``ConnectionResetError`` / ``IncompleteRead`` rather than as the
    ``URLError`` the open-time handler knows.
    """

    def __init__(self, data=b"", error=None, chunks=0):
        super().__init__(data)
        self._error = error or TimeoutError("timed out")
        self._left = chunks

    def read(self, size=-1):
        if self._left <= 0:
            raise self._error
        self._left -= 1
        return super().read(size)


class _ClosedPipe:
    """A stdout whose reader has gone away, the way ``| head`` leaves it."""

    def write(self, data):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):  # pragma: no cover - the write raises first
        raise BrokenPipeError(32, "Broken pipe")


class _PipedStdout:
    buffer = _ClosedPipe()

    def fileno(self):
        # What a captured stream does, and the branch that keeps the redirect
        # from being the thing that fails.
        raise io.UnsupportedOperation("fileno")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FUKO_AUTH_TOKEN", "t")
    monkeypatch.setenv("FUKO_URL", "http://sidecar:8000")


# --- listing.


def test_list_calls_the_endpoint_and_prints_the_figures(monkeypatch, capsys):
    seen = {}

    def fake(path, params=None):
        seen.update(path=path, params=params)
        return {"transcripts": [_ROW], "count": 7}

    monkeypatch.setattr(transcriptscli, "_call", fake)
    transcriptscli._list(_ns())
    out = capsys.readouterr().out
    assert seen["path"] == "/transcripts"
    assert "1 shown · 7 total" in out
    assert "lemehmet/mepro#42" in out and "dorian" in out
    assert "complete" in out and "INCOMPLETE" not in out
    assert "217 calls" in out and "Read=182" in out
    assert "40.0 MiB results" in out
    assert "37 files re-read" in out


def test_list_caps_the_tool_list_unless_full(monkeypatch, capsys):
    monkeypatch.setattr(transcriptscli, "_call", lambda p, params=None: {"transcripts": [_ROW]})
    transcriptscli._list(_ns())
    capped = capsys.readouterr().out
    assert "+1 more" in capped and "Glob=1" not in capped

    transcriptscli._list(_ns(full=True))
    full = capsys.readouterr().out
    assert "Glob=1" in full and "+1 more" not in full
    # --full also shows how the run itself ended.
    assert "ok in 612.5s via openrouter (agentic)" in full


def test_list_flags_an_incomplete_transcript(monkeypatch, capsys):
    row = {**_ROW, "complete": False}
    monkeypatch.setattr(transcriptscli, "_call", lambda p, params=None: {"transcripts": [row]})
    transcriptscli._list(_ns())
    assert "INCOMPLETE" in capsys.readouterr().out


def test_list_names_a_transcript_no_run_row_claims(monkeypatch, capsys):
    row = {**_ROW, "repo": None, "pr": None, "seat": None, "model": None, "started_at": None}
    monkeypatch.setattr(transcriptscli, "_call", lambda p, params=None: {"transcripts": [row]})
    transcriptscli._list(_ns())
    out = capsys.readouterr().out
    assert "(no run row)" in out and "None" not in out


def test_list_forwards_every_filter(monkeypatch):
    seen = {}

    def fake(path, params=None):
        seen.update(params)
        return {"transcripts": [], "count": 0}

    monkeypatch.setattr(transcriptscli, "_call", fake)
    transcriptscli._list(
        _ns(
            repo="o/r", pr=7, seat="gray", since="2026-08-01", until="2026-09-01", limit=5, offset=2
        )
    )
    assert seen == {
        "repo": "o/r",
        "pr": 7,
        "seat": "gray",
        "since": "2026-08-01",
        "until": "2026-09-01",
        "limit": 5,
        "offset": 2,
    }


def test_list_json_prints_the_body_unchanged(monkeypatch, capsys):
    body = {"transcripts": [_ROW], "count": 1}
    monkeypatch.setattr(transcriptscli, "_call", lambda p, params=None: body)
    transcriptscli._list(_ns(json=True))
    assert json.loads(capsys.readouterr().out) == body


def test_a_run_with_no_tool_calls_says_so_rather_than_showing_nothing(monkeypatch, capsys):
    row = {**_ROW, "tool_calls": {}}
    monkeypatch.setattr(transcriptscli, "_call", lambda p, params=None: {"transcripts": [row]})
    transcriptscli._list(_ns())
    assert "no tool calls" in capsys.readouterr().out


# --- the transport's failure modes: none of them may look like an empty corpus.


def test_a_missing_token_is_fatal_before_any_request(monkeypatch):
    monkeypatch.delenv("FUKO_AUTH_TOKEN")
    with pytest.raises(SystemExit) as e:
        transcriptscli._url("/transcripts")
    assert "FUKO_AUTH_TOKEN" in str(e.value)


def test_an_unreachable_sidecar_exits_non_zero(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(transcriptscli.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    assert "cannot reach http://sidecar:8000" in str(e.value)


@pytest.mark.parametrize(
    ("code", "detail", "expected"),
    [
        (503, "transcript index needs the Postgres store", "503"),
        (404, "no transcript stored under abc123", "404"),
        (400, "invalid blob key '..'", "400"),
    ],
)
def test_an_http_error_exits_with_the_sidecars_own_taxonomy(monkeypatch, code, detail, expected):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://sidecar:8000/transcripts",
            code,
            "Service Unavailable",
            {},
            io.BytesIO(json.dumps({"detail": detail}).encode()),
        )

    monkeypatch.setattr(transcriptscli.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    message = str(e.value)
    assert expected in message and detail in message


def test_a_non_json_body_is_reported_verbatim(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://sidecar:8000/transcripts", 502, "Bad Gateway", {}, io.BytesIO(b"<html>nginx")
        )

    monkeypatch.setattr(transcriptscli.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    assert "nginx" in str(e.value)


def test_a_non_json_success_body_is_fatal_not_an_empty_listing(monkeypatch):
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"nope")
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    assert "non-JSON" in str(e.value)


# --- fetch.


def test_get_writes_the_stored_bytes_to_stdout(monkeypatch, capsysbinary):
    body = b'{"type":"assistant"}\n{"type":"result"}\n'
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(body)
    )
    transcriptscli._get(argparse.Namespace(key="abc123", out=None))
    assert capsysbinary.readouterr().out == body


def test_get_writes_to_a_file_when_asked(monkeypatch, tmp_path, capsys):
    body = b'{"type":"result"}\n' * 3
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(body)
    )
    out = tmp_path / "session.ndjson"
    transcriptscli._get(argparse.Namespace(key="abc123", out=str(out)))
    assert out.read_bytes() == body
    # The receipt goes to stderr so `--out` still leaves stdout clean.
    assert f"wrote {len(body)} bytes" in capsys.readouterr().err


def test_get_url_encodes_the_key(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        transcriptscli.urllib.request,
        "urlopen",
        lambda req, timeout=None: (seen.update(url=req.full_url), _FakeResponse(b""))[1],
    )
    transcriptscli._get(argparse.Namespace(key="../etc/passwd", out=None))
    assert seen["url"] == "http://sidecar:8000/transcripts/..%2Fetc%2Fpasswd"


def test_get_of_an_absent_key_exits_non_zero(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://sidecar:8000/transcripts/x",
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"detail":"no transcript stored under x"}'),
        )

    monkeypatch.setattr(transcriptscli.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="x", out=None))
    assert "404" in str(e.value)


# --- failures that happen AFTER the response opened.


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        ConnectionResetError(104, "Connection reset by peer"),
        http.client.IncompleteRead(b"half"),
    ],
)
def test_a_body_that_dies_mid_read_is_named_not_a_traceback(monkeypatch, error):
    monkeypatch.setattr(
        transcriptscli.urllib.request,
        "urlopen",
        lambda req, timeout=None: _DyingResponse(b"{}", error=error),
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    assert "failed mid-body" in str(e.value)


def test_get_names_a_transfer_that_dies_mid_body(monkeypatch, capsysbinary):
    monkeypatch.setattr(
        transcriptscli.urllib.request,
        "urlopen",
        lambda req, timeout=None: _DyingResponse(b'{"type":"assistant"}\n', chunks=1),
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="abc123", out=None))
    assert "failed mid-body" in str(e.value)


def test_get_removes_the_partial_file_when_the_transfer_dies(monkeypatch, tmp_path):
    # The one failure here that outlives the command: a short file that reads
    # as a whole session next time somebody greps it.
    monkeypatch.setattr(
        transcriptscli.urllib.request,
        "urlopen",
        lambda req, timeout=None: _DyingResponse(b'{"type":"assistant"}\n', chunks=1),
    )
    out = tmp_path / "session.ndjson"
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="abc123", out=str(out)))
    assert "failed mid-body" in str(e.value)
    assert not out.exists()


def test_a_partial_file_that_cannot_be_removed_still_reports_the_transfer(monkeypatch, tmp_path):
    # The cleanup is best-effort: whatever stops the unlink, the fault the
    # operator needs to see is the one that killed the download.
    def dying(req, timeout=None):
        return _DyingResponse(b'{"type":"assistant"}\n', chunks=1)

    monkeypatch.setattr(transcriptscli.urllib.request, "urlopen", dying)
    monkeypatch.setattr(
        transcriptscli.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("read-only"))
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="abc123", out=str(tmp_path / "s.ndjson")))
    assert "failed mid-body" in str(e.value)


def test_a_local_write_failure_blames_the_disk_not_the_sidecar(monkeypatch, tmp_path):
    # Both raise OSError, and this command writes multi-megabyte files onto
    # boxes chosen for having less memory than the sidecar -- so ENOSPC here
    # must not send the operator to look at the network.
    monkeypatch.setattr(
        transcriptscli.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(b'{"type":"assistant"}\n'),
    )
    out = tmp_path / "session.ndjson"
    real_open = open

    def full_disk(path, mode="r", *a, **kw):
        handle = real_open(path, mode, *a, **kw)
        if "w" in mode:
            handle.write = lambda data: (_ for _ in ()).throw(OSError(28, "No space left"))
        return handle

    monkeypatch.setattr("builtins.open", full_disk)
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="abc123", out=str(out)))
    message = str(e.value)
    assert "cannot write" in message and "No space left" in message
    assert "failed mid-body" not in message
    assert not out.exists()


def test_a_flush_failure_when_the_file_closes_also_blames_the_disk(monkeypatch, tmp_path):
    # Where ENOSPC most often actually lands: every `write` sat in the buffer
    # and the whole thing fails at close.
    class _FullOnClose:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            raise OSError(28, "No space left on device")

        def write(self, data):
            return len(data)

    monkeypatch.setattr(
        transcriptscli.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(b'{"type":"result"}\n'),
    )
    monkeypatch.setattr("builtins.open", lambda *a, **kw: _FullOnClose())
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="abc123", out=str(tmp_path / "s.ndjson")))
    assert "cannot write" in str(e.value)


def test_get_names_an_out_path_it_cannot_open(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"{}")
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="abc123", out=str(tmp_path / "no" / "such")))
    assert "cannot write" in str(e.value)


def test_a_closed_stdout_pipe_exits_zero_and_leaves_nothing_buffered(monkeypatch):
    # `fuko transcripts get KEY | head` is the documented use, so it must exit
    # 0 -- and must not hand a populated buffer to the interpreter's final
    # flush, which would retry the dead pipe and report it after the fact.
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"body")
    )
    monkeypatch.setattr(transcriptscli.sys, "stdout", _PipedStdout())
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="abc123", out=None))
    assert e.value.code == 0


def test_a_closed_stdout_pipe_redirects_the_fd_before_leaving(monkeypatch, tmp_path):
    # The redirect is the whole fix: without it the buffered tail is flushed at
    # shutdown against the closed pipe. Assert the dup2 actually happened.
    seen = {}
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"body")
    )

    class _RealFd(_PipedStdout):
        def fileno(self):
            return 4321

    monkeypatch.setattr(transcriptscli.sys, "stdout", _RealFd())
    monkeypatch.setattr(transcriptscli.os, "dup2", lambda a, b: seen.update(target=b))
    with pytest.raises(SystemExit) as e:
        transcriptscli._get(argparse.Namespace(key="abc123", out=None))
    assert e.value.code == 0
    assert seen["target"] == 4321


@pytest.mark.parametrize(
    "error",
    [
        http.client.RemoteDisconnected("peer closed"),
        http.client.BadStatusLine("garbage"),
        TimeoutError("timed out waiting for headers"),
    ],
)
def test_a_connection_that_dies_before_the_headers_is_named(monkeypatch, error):
    # urllib wraps only what its handler raises around `h.request`; the
    # `h.getresponse()` after it is unguarded, so these arrive bare.
    def boom(req, timeout=None):
        raise error

    monkeypatch.setattr(transcriptscli.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    assert "cannot reach http://sidecar:8000" in str(e.value)


def test_an_error_body_that_cannot_be_read_still_names_the_status(monkeypatch):
    class _DeadBody(io.BytesIO):
        def read(self, size=-1):
            raise ConnectionResetError(104, "Connection reset by peer")

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://sidecar:8000/transcripts", 502, "Bad Gateway", {}, _DeadBody(b"")
        )

    monkeypatch.setattr(transcriptscli.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    assert "502" in str(e.value)


# --- the success body's shape.


@pytest.mark.parametrize("body", [b"null", b"[1, 2]", b'"a string"', b"42"])
def test_a_200_body_that_is_not_an_object_is_fatal_not_an_empty_listing(monkeypatch, body):
    # The one outcome #240 exists to prevent: something other than the sidecar
    # answering 200, and the CLI rendering it as "the corpus is empty".
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(body)
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    assert "not an object" in str(e.value)


def test_a_200_object_without_the_expected_field_is_fatal(monkeypatch):
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"{}")
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._list(_ns())
    assert "'transcripts'" in str(e.value)


@pytest.mark.parametrize("value", [None, {"a": 1}, "rows", 3])
def test_a_transcripts_field_that_is_not_a_list_is_fatal(monkeypatch, value):
    # `null` matters most: `or []` would have turned it into the healthy-empty
    # page, which is the answer this endpoint must never fake.
    body = json.dumps({"transcripts": value, "count": 0}).encode()
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(body)
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._list(_ns())
    assert "not a list" in str(e.value)


@pytest.mark.parametrize(
    "row",
    [
        None,
        "a string",
        7,
        [],
        {"no": "key"},
        {"key": "k", "tool_calls": ["Read"]},
        {"key": "k", "tool_result_bytes": "big"},
        {"key": "k", "outcome": "ok", "duration_s": "long"},
    ],
)
def test_a_row_the_reader_cannot_read_is_named_not_a_traceback(monkeypatch, row):
    body = json.dumps({"transcripts": [row], "count": 1}).encode()
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(body)
    )
    with pytest.raises(SystemExit) as e:
        transcriptscli._list(_ns(full=True))
    assert "cannot read" in str(e.value)


def test_a_closed_pipe_on_the_listing_exits_zero_like_the_fetch_path(monkeypatch):
    body = json.dumps({"transcripts": [_ROW], "count": 1}).encode()
    monkeypatch.setattr(
        transcriptscli.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(body)
    )

    class _PipedText(_PipedStdout):
        def write(self, data):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

        def isatty(self):
            return False

    monkeypatch.setattr(transcriptscli.sys, "stdout", _PipedText())
    with pytest.raises(SystemExit) as e:
        transcriptscli._list(_ns())
    assert e.value.code == 0


@pytest.mark.parametrize("body", [b"null", b"[1, 2]", b'"a string"', b"42"])
def test_a_json_error_body_that_is_not_an_object_is_still_named(monkeypatch, body):
    # A proxy between the operator and the sidecar can answer with valid JSON
    # that is not the sidecar's {"detail": ...} object. Decoding it must not be
    # the thing that crashes the one path whose job is to name the fault.
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://sidecar:8000/transcripts", 502, "Bad Gateway", {}, io.BytesIO(body)
        )

    monkeypatch.setattr(transcriptscli.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit) as e:
        transcriptscli._call("/transcripts")
    message = str(e.value)
    assert "502" in message and body.decode() in message


# --- registration.


def test_the_subcommand_is_registered_and_parses(monkeypatch):
    called = {}
    monkeypatch.setattr(transcriptscli, "_list", lambda args: called.update(vars(args)))
    monkeypatch.setattr(
        cli.sys, "argv", ["fuko", "transcripts", "list", "--repo", "o/r", "--pr", "9"]
    )
    cli.main()
    assert called["repo"] == "o/r" and called["pr"] == 9


def test_get_is_registered(monkeypatch):
    called = {}
    monkeypatch.setattr(transcriptscli, "_get", lambda args: called.update(vars(args)))
    monkeypatch.setattr(cli.sys, "argv", ["fuko", "transcripts", "get", "k1", "-o", "f.ndjson"])
    cli.main()
    assert called["key"] == "k1" and called["out"] == "f.ndjson"
    assert called["transcripts_cmd"] == "get"
