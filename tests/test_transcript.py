"""Tests for the streaming, credential-scrubbed transcript tee (#237)."""

import json
import os
import re
import sys

import pytest

from sidecar.backends.agentic import _transcript_secrets
from sidecar.config import settings
from sidecar.reviewer import harness as harness_mod
from sidecar.reviewer import transcript as transcript_mod
from sidecar.reviewer.harness import run_review
from sidecar.reviewer.transcript import (
    FileTranscriptSink,
    Scrubber,
    Transcript,
    mint_key,
    open_transcript,
)

SECRET = "ghs_0123456789abcdefghijABCDEFGHIJ"


class _RecordingSink:
    def __init__(self, fail_at=None):
        self.lines = []
        self.closes = 0
        self._fail_at = fail_at

    def write(self, line):
        if self._fail_at is not None and len(self.lines) + 1 == self._fail_at:
            raise OSError("no space left on device")
        self.lines.append(line)

    def close(self):
        self.closes += 1


def _transcript(secrets=(), sink=None):
    sink = sink if sink is not None else _RecordingSink()
    return Transcript("key-1", sink, Scrubber.for_secrets(secrets)), sink


def _feed_script(directory, events, tail=""):
    """A fake `claude` binary that prints `events` as NDJSON, then runs `tail`."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "claude"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "sys.stdin.read()\n"
        f"for event in {events!r}:\n"
        "    print(json.dumps(event), flush=True)\n" + tail
    )
    script.chmod(0o755)
    return script


RESULT_EVENT = {"type": "result", "subtype": "success", "result": '{"findings": []}'}


# --- scrubbing -------------------------------------------------------------


def test_scrubs_an_injected_value_inside_a_tool_result_event():
    """The `user` events the fold skips are where tool results -- and therefore
    an echoed credential -- actually live."""
    transcript, sink = _transcript([("ANTHROPIC_API_KEY", SECRET)])
    line = json.dumps(
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "content": f"ANTHROPIC_API_KEY={SECRET}\n"}]
            },
        }
    )
    transcript.write(line)
    assert SECRET not in sink.lines[0]
    assert "[REDACTED:ANTHROPIC_API_KEY]" in sink.lines[0]
    # Still one parseable NDJSON event: the marker carries no quote or backslash.
    assert json.loads(sink.lines[0])["type"] == "user"


def test_scrubs_the_json_escaped_spelling_of_a_value():
    """The feed is NDJSON, so a value carrying a quote arrives escaped."""
    secret = 'pw"with\\quote-chars'
    transcript, sink = _transcript([("FUKO_DATABASE_URL", secret)])
    transcript.write(json.dumps({"type": "assistant", "text": f"saw {secret} here"}))
    assert json.dumps(secret)[1:-1] not in sink.lines[0]
    assert secret not in sink.lines[0]


def test_a_credential_shaped_string_that_was_never_injected_is_written_verbatim():
    """The rule is EXACT injected values, decided rather than inherited.

    Shape-matching would be a second source of truth beside the driver's
    credential lists, and scrubbing is irreversible: a rule that fires on
    anything token-shaped silently corrupts the corpus the epic exists to build
    (a PR that legitimately quotes a fake key, a test fixture, a docs example).
    """
    transcript, sink = _transcript([("ANTHROPIC_API_KEY", SECRET)])
    look_alike = "ghp_ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"
    transcript.write(json.dumps({"type": "user", "text": look_alike}))
    assert look_alike in sink.lines[0]


def test_a_short_value_is_still_scrubbed():
    """The caller supplies only values of variables it already decided are
    credential-bearing, so a short one is a short SECRET -- an operator's brief
    `FUKO_TOKEN` -- not a boolean flag. A length floor here would write it to
    durable storage verbatim."""
    transcript, sink = _transcript([("FUKO_TOKEN", "abc")])
    transcript.write(json.dumps({"type": "assistant", "text": "abc appears in prose"}))
    assert "abc" not in sink.lines[0]
    assert "[REDACTED:FUKO_TOKEN]" in sink.lines[0]


def test_a_secret_containing_another_secret_is_replaced_whole():
    """Longest needle first, or the shorter rule fires inside the longer value
    and leaves the remainder of the longer credential on disk."""
    inner = "abcdefghij"
    outer = inner + "0123456789"
    transcript, sink = _transcript([("SHORT_KEY", inner), ("LONG_KEY", outer)])
    transcript.write(json.dumps({"type": "assistant", "text": outer}))
    assert inner not in sink.lines[0]
    assert "[REDACTED:LONG_KEY]" in sink.lines[0]


def test_the_marker_names_the_variable_and_never_the_value():
    scrubbed = Scrubber.for_secrets([("ZAI_KEY", SECRET)]).scrub(f"x {SECRET} y")
    assert scrubbed == "x [REDACTED:ZAI_KEY] y"
    assert str(len(SECRET)) not in scrubbed


# --- the local file sink ---------------------------------------------------


def test_the_file_is_created_lazily_and_flushed_per_line(tmp_path):
    """A run that streams nothing leaves no file; a run cut short keeps its
    tail, which needs the line already on disk rather than in a buffer."""
    path = tmp_path / "nested" / "run.ndjson"
    sink = FileTranscriptSink(path)
    assert not path.exists()
    sink.write('{"type": "system"}')
    # Read it back while the sink is still open: no close, no flush call.
    assert path.read_text() == '{"type": "system"}\n'
    sink.write('{"type": "result"}\n')
    assert path.read_text().splitlines() == ['{"type": "system"}', '{"type": "result"}']
    sink.close()
    sink.close()


def test_the_transcript_file_and_directory_are_owner_only(tmp_path):
    """The scrub drops the driver's credential values and nothing else, so what
    is left is the reviewed repository as the agent read it -- the content the
    checkout gets `mkdtemp`'s 0o700 for, kept durably rather than deleted."""
    path = tmp_path / "nested" / "run.ndjson"
    sink = FileTranscriptSink(path)
    sink.write('{"type": "system"}')
    sink.close()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


# --- the tee ---------------------------------------------------------------


def test_the_tee_writes_through_rather_than_accumulating():
    """Each line is durable before the fold sees the next one.

    Asserted from the producer side: the generator refuses to yield event N+1
    until the sink already holds event N, so a tee that collected the feed and
    wrote it at the end fails here deterministically.
    """
    transcript, sink = _transcript()

    def feed():
        for index in range(5):
            assert len(sink.lines) == index
            yield json.dumps({"type": "assistant", "n": index})

    seen = list(harness_mod._tee(feed(), transcript))
    assert seen == sink.lines
    assert len(sink.lines) == 5


def test_blank_lines_are_dropped_but_unknown_events_are_kept():
    """Everything the fold skips still belongs in the transcript -- only
    whitespace, which frames no event, does not."""
    transcript, sink = _transcript()
    transcript.write("   \n")
    transcript.write('{"type": "future_event_kind"}')
    transcript.write("not json at all\n")
    assert sink.lines == ['{"type": "future_event_kind"}', "not json at all\n"]


def test_run_review_captures_every_event_including_the_ones_the_fold_skips(tmp_path):
    events = [
        {"type": "system", "subtype": "init"},
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]},
        },
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "x" * 50}]}},
        RESULT_EVENT,
    ]
    assert _feed_script(tmp_path, events).exists()
    transcript = open_transcript([], directory=str(tmp_path / "transcripts"))
    result = run_review(
        "p",
        tmp_path,
        cwd=tmp_path,
        model="m",
        env={**os.environ, "PATH": str(tmp_path)},
        timeout=30,
        transcript=transcript,
    )
    assert result.text == '{"findings": []}'
    written = (tmp_path / "transcripts" / f"{transcript.key}.ndjson").read_text().splitlines()
    assert [json.loads(line) for line in written] == events


def test_a_feed_cut_short_by_the_timeout_kill_keeps_what_arrived(tmp_path):
    """The production shape of a truncated run: `tool_timeout` kills a harness
    that is still streaming, and everything before the cut must survive."""
    events = [{"type": "assistant", "n": index} for index in range(3)]
    assert _feed_script(tmp_path, events, tail="time.sleep(60)\n").exists()
    transcript = open_transcript([], directory=str(tmp_path / "transcripts"))
    result = run_review(
        "p",
        tmp_path,
        cwd=tmp_path,
        model="m",
        env={**os.environ, "PATH": str(tmp_path)},
        timeout=2,
        transcript=transcript,
    )
    assert result.timed_out is True
    written = (tmp_path / "transcripts" / f"{transcript.key}.ndjson").read_text().splitlines()
    assert [json.loads(line) for line in written] == events


# --- capture never fails a review -----------------------------------------


def _run_with(directory, transcript):
    assert _feed_script(directory, [{"type": "assistant", "n": 1}, RESULT_EVENT]).exists()
    return run_review(
        "p",
        directory,
        cwd=directory,
        model="m",
        env={**os.environ, "PATH": str(directory)},
        timeout=30,
        transcript=transcript,
    )


def test_every_capture_failure_leaves_the_result_identical_to_no_capture(
    tmp_path, monkeypatch, capsys
):
    """Byte-identical `HarnessResult` across: no capture, an unwritable
    destination, a sink that raises mid-stream, and capture disabled."""
    baseline = _run_with(tmp_path / "none", None)

    occupied = tmp_path / "occupied"
    occupied.write_text("this is a file, not a directory")
    unwritable = open_transcript([], directory=str(occupied / "transcripts"))
    assert _run_with(tmp_path / "unwritable", unwritable) == baseline

    raising, _sink = _transcript(sink=_RecordingSink(fail_at=1))
    assert _run_with(tmp_path / "raising", raising) == baseline

    monkeypatch.setattr(settings, "transcript_dir", "")
    assert open_transcript([]) is None
    assert _run_with(tmp_path / "disabled", None) == baseline

    err = capsys.readouterr().err
    assert err.count("capture failed") == 2


def test_a_failed_capture_reports_once_and_goes_inert(capsys):
    """One line, not one per event: a disk that filled at event 900 would
    otherwise flood the log of the run that is already the largest."""
    transcript, sink = _transcript(sink=_RecordingSink(fail_at=2))
    for index in range(50):
        transcript.write(json.dumps({"n": index}))
    err = capsys.readouterr().err
    assert err.count("fuko: transcript key-1 capture failed") == 1
    assert len(sink.lines) == 1
    transcript.close()
    transcript.close()
    assert sink.closes == 1


def test_capture_disabled_never_wraps_the_stream(tmp_path, monkeypatch):
    """`None` rather than an inert object: the fold iterates the pipe directly,
    so a fleet that never turns capture on pays nothing per event."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("the tee must not be built when capture is off")

    monkeypatch.setattr(harness_mod, "_tee", _forbidden)
    returncode, outcome, _stderr, _timed_out = harness_mod._drive(
        [sys.executable, str(_feed_script(tmp_path, [RESULT_EVENT]))],
        prompt="p",
        cwd=tmp_path,
        env=dict(os.environ),
        timeout=30,
        emit=lambda *a: None,
    )
    assert returncode == 0 and outcome.text == '{"findings": []}'


# --- configuration and identity -------------------------------------------


def test_capture_is_off_until_a_destination_is_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transcript_dir", "")
    assert open_transcript([]) is None
    monkeypatch.setattr(settings, "transcript_dir", str(tmp_path / "configured"))
    transcript = open_transcript([], label="seat-model")
    assert transcript is not None
    transcript.write('{"type": "system"}')
    transcript.close()
    assert (tmp_path / "configured" / f"{transcript.key}.ndjson").exists()


def test_opening_a_transcript_announces_where_it_went(tmp_path, capsys):
    transcript = open_transcript([], label="seat-model", directory=str(tmp_path))
    err = capsys.readouterr().err
    assert f"fuko: agentic seat-model transcript {tmp_path}" in err
    assert transcript.key in err


def test_the_transcript_mints_its_own_key():
    """`review_runs` cannot supply one -- its row is inserted after the run and
    its id is never returned (#236) -- so the key is minted at run start."""
    first, second = mint_key(), mint_key()
    assert first != second
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{12}", first)


def test_a_relative_destination_is_made_absolute(tmp_path, monkeypatch):
    """`_permission_settings` drops any path that is not POSIX-absolute, so a
    relative destination would render NO deny rule while the sink still wrote
    there."""
    monkeypatch.chdir(tmp_path)
    assert transcript_mod.transcript_dir("corpus") == (tmp_path / "corpus").resolve()


def test_a_symlinked_destination_resolves_to_its_target(tmp_path):
    """A rule for the alias leaves the archive readable under its real name."""
    target = tmp_path / "real"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    assert transcript_mod.transcript_dir(str(alias)) == target.resolve()


@pytest.mark.parametrize("root", ["/", "//", "/.."])
def test_the_filesystem_root_is_refused_as_a_destination(root):
    """`_permission_settings` normalizes a candidate with `rstrip("/")`, which
    turns "/" into "" -- dropped without even reaching the non-POSIX
    announcement. A root destination would therefore write transcripts that no
    deny rule covers and nothing reports."""
    with pytest.raises(ValueError, match="filesystem root"):
        transcript_mod.transcript_dir(root)


def test_a_root_destination_leaves_both_capture_and_deny_off(capsys):
    """Refusing has to degrade like every other misconfiguration, on BOTH
    sides: the one state worth ruling out is a capture that opened against a
    path no rule covers."""
    assert open_transcript([], directory="/") is None
    assert "transcript capture unavailable" in capsys.readouterr().err


def test_no_destination_is_not_a_path(monkeypatch):
    """Capture off returns None rather than the current directory."""
    monkeypatch.setattr(settings, "transcript_dir", "", raising=False)
    assert transcript_mod.transcript_dir() is None


def test_a_misconfigured_destination_degrades_to_no_capture(monkeypatch, capsys):
    monkeypatch.setattr(transcript_mod, "mint_key", lambda: 1 / 0)
    assert open_transcript([], directory="/tmp/whatever") is None
    assert "transcript capture unavailable" in capsys.readouterr().err


# --- the driver's secret set ----------------------------------------------


def test_the_secret_set_covers_the_checkout_token_and_both_environments(monkeypatch):
    """The GitHub App token is the case an env-derived set would miss: it never
    enters the harness environment (every spelling is stripped), it belongs to
    the checkout."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-github-token")
    monkeypatch.setenv("ZAI_KEY", "other-seats-provider-key")
    monkeypatch.setenv("FUKO_TOKEN", "fuko-ledger-write-token")
    secrets = _transcript_secrets(
        {"ANTHROPIC_API_KEY": "the-seats-own-key", "ANTHROPIC_BASE_URL": "https://gw.example"},
        "the-checkout-token",
    )
    values = dict(secrets)
    assert values["GITHUB_APP_TOKEN"] == "the-checkout-token"
    assert values["ANTHROPIC_API_KEY"] == "the-seats-own-key"
    assert values["GITHUB_TOKEN"] == "ambient-github-token"
    assert values["ZAI_KEY"] == "other-seats-provider-key"
    assert values["FUKO_TOKEN"] == "fuko-ledger-write-token"
    # Routing, not a credential: scrubbing it would replace a URL that
    # legitimately appears in the feed.
    assert "ANTHROPIC_BASE_URL" not in values


def test_the_secret_set_covers_the_sibling_seats_app_tokens(monkeypatch):
    """A seat's run holds every OTHER seat's write-scoped App token: the review
    workflow mints one per App and exports them all into the one process. Only
    the branch's own arrives as the `token` argument, so the siblings reach the
    transcript unless the prefix is scrubbed."""
    monkeypatch.setenv("FUKO_GITHUB_TOKEN_HENRY", "ghs_henrys-own-app-token")
    monkeypatch.setenv("FUKO_GITHUB_TOKEN_GRAY", "ghs_grays-app-token")
    monkeypatch.setenv("FUKO_GITHUB_TOKEN_DORIAN", "ghs_dorians-app-token")
    values = dict(_transcript_secrets({}, "ghs_henrys-own-app-token"))
    assert values["FUKO_GITHUB_TOKEN_GRAY"] == "ghs_grays-app-token"
    assert values["FUKO_GITHUB_TOKEN_DORIAN"] == "ghs_dorians-app-token"


def test_the_secret_set_covers_the_object_store_credentials(monkeypatch):
    """Namespace-stripped like the rest of `FUKO_`, and a credential."""
    monkeypatch.setenv("FUKO_S3_ACCESS_KEY_ID", "the-access-key-id")
    monkeypatch.setenv("FUKO_S3_SECRET_ACCESS_KEY", "the-secret-access-key")
    monkeypatch.setenv("FUKO_S3_REGION", "auto")
    values = dict(_transcript_secrets({}, ""))
    assert values["FUKO_S3_ACCESS_KEY_ID"] == "the-access-key-id"
    assert values["FUKO_S3_SECRET_ACCESS_KEY"] == "the-secret-access-key"
    # A region code is not a credential; scrubbing "auto" would hit prose.
    assert "FUKO_S3_REGION" not in values


def test_the_secret_set_covers_the_runners_own_anthropic_credential(monkeypatch):
    """`_ANTHROPIC_INHERITED_VARS` strips the ambient value so config alone
    decides the seat's auth -- which means it never appears in `harness_env`,
    and a set read only from there would miss the credential this process
    actually holds."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "the-runners-own-key")
    secrets = _transcript_secrets({"ANTHROPIC_API_KEY": "the-seats-injected-key"}, "")
    # BOTH spellings are live at once and both must go: the seat's injected
    # credential and the runner's ambient one are different secrets.
    scrubbed = Scrubber.for_secrets(secrets).scrub(
        "saw the-runners-own-key and the-seats-injected-key"
    )
    assert "the-runners-own-key" not in scrubbed
    assert "the-seats-injected-key" not in scrubbed


def test_a_run_with_no_checkout_token_registers_no_empty_secret(monkeypatch):
    """An empty needle would match between every pair of characters; the
    scrubber drops it, but the set should not carry it at all."""
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "FUKO_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert not any(value == "" for _name, value in _transcript_secrets({}, ""))


@pytest.mark.parametrize("value", ["", None, 1234])
def test_scrubber_skips_values_it_must_not_replace(value):
    """Only the two that cannot name a credential occurrence: an empty needle
    (matches everywhere) and a non-string (has no spelling on the wire)."""
    assert Scrubber.for_secrets([("X", value)]).replacements == ()
