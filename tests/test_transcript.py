"""Tests for the streaming, credential-scrubbed transcript tee (#237)."""

import json
import os
import re
import sys

import httpx
import pytest

from sidecar.backends import agentic
from sidecar.backends.agentic import _transcript_secrets
from sidecar.backends.agentic import settings as agentic_settings
from sidecar.config import settings
from sidecar.reviewer import harness as harness_mod
from sidecar.reviewer import transcript as transcript_mod
from sidecar.reviewer.harness import run_review
from sidecar.reviewer.transcript import (
    FileTranscriptSink,
    Scrubber,
    ShippingTranscriptSink,
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
        # Stands in for a sink WITH a destination: `Transcript.index` writes a
        # row only for bytes a sink affirms it stored, so a double that stayed
        # silent about it would suppress every index in this file for the wrong
        # reason.
        self.stored = True

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


def test_a_renamed_store_credential_prefix_is_still_stripped_and_scrubbed(monkeypatch):
    """`FUKO_TRANSCRIPT_STORE_CREDS_ENV_PREFIX` is a supported setting, and a
    value outside the `FUKO_` namespace is neither stripped by the namespace
    filter nor covered by the literal default names -- so the credential that
    stores the transcript would be written into it."""
    monkeypatch.setattr(agentic_settings, "transcript_store_creds_env_prefix", "MYCO_S3")
    monkeypatch.setenv("MYCO_S3_ACCESS_KEY_ID", "the-access-key-id")
    monkeypatch.setenv("MYCO_S3_SECRET_ACCESS_KEY", "the-secret-access-key")

    assert agentic._store_credential_vars() == {
        "MYCO_S3_ACCESS_KEY_ID",
        "MYCO_S3_SECRET_ACCESS_KEY",
    }
    values = dict(_transcript_secrets({}, ""))
    assert values["MYCO_S3_ACCESS_KEY_ID"] == "the-access-key-id"
    assert values["MYCO_S3_SECRET_ACCESS_KEY"] == "the-secret-access-key"


def test_boto3s_default_credential_chain_is_stripped_and_scrubbed(monkeypatch):
    """The `FUKO_URL`-unset path writes straight to the store, so a bucket
    credential is in THIS process for the first time -- and `_s3_client` passes
    `None` when the explicit names are unset, which makes botocore resolve
    these. The agent has no use for any of them."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAtheaccesskey")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "thesecretaccesskey")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "thesessiontoken")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "thelegacytoken")
    monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN", "thecontainertoken")
    values = dict(_transcript_secrets({}, ""))
    assert values["AWS_ACCESS_KEY_ID"] == "AKIAtheaccesskey"
    assert values["AWS_SECRET_ACCESS_KEY"] == "thesecretaccesskey"
    assert values["AWS_SESSION_TOKEN"] == "thesessiontoken"
    # botocore checks the legacy spelling FIRST, so a deployment that sets it
    # is the one actually using it.
    assert values["AWS_SECURITY_TOKEN"] == "thelegacytoken"
    assert values["AWS_CONTAINER_AUTHORIZATION_TOKEN"] == "thecontainertoken"


def test_a_credential_prefix_is_normalized_the_same_way_on_both_sides(monkeypatch):
    """Normalizing on only one side IS the leak: the store would read the
    padded spelling while the strip and scrub lists named the trimmed one."""
    from sidecar.objectstore import BlobStoreConfig

    monkeypatch.setattr(agentic_settings, "transcript_store_creds_env_prefix", "  MYCO_S3  ")
    assert BlobStoreConfig.from_settings().creds_env_prefix == "MYCO_S3"
    assert agentic._store_credential_vars() == {
        "MYCO_S3_ACCESS_KEY_ID",
        "MYCO_S3_SECRET_ACCESS_KEY",
    }


def test_a_destination_holding_a_newline_is_refused(monkeypatch, tmp_path):
    """The deny-dir hand-off is newline-separated, so a directory name holding
    one is split into a rule for somewhere else plus a tail dropped as
    non-POSIX -- while the transcript still lands at the real path."""
    from sidecar.objectstore import BlobStoreConfig, local_blob_root

    bad = tmp_path / "one\ntwo"
    with pytest.raises(ValueError, match="newline"):
        transcript_mod.transcript_dir(str(bad))
    with pytest.raises(ValueError, match="newline"):
        local_blob_root(BlobStoreConfig(backend="file", root=str(bad)))


def test_a_destination_padded_with_whitespace_is_refused(tmp_path):
    """`_permission_settings` strips each deny candidate, so a directory whose
    name ends in whitespace is denied under its TRIMMED name while the
    transcript lands at the padded one -- undenied, and nothing says so."""
    from sidecar.objectstore import BlobStoreConfig, local_blob_root

    bad = tmp_path / "corpus "
    assert str(bad).endswith(" ")
    with pytest.raises(ValueError, match="whitespace"):
        transcript_mod.transcript_dir(str(bad))
    with pytest.raises(ValueError, match="whitespace"):
        local_blob_root(BlobStoreConfig(backend="file", root=str(bad)))


@pytest.mark.skipif(os.name != "posix", reason="a backslash is a path separator off POSIX")
def test_a_destination_holding_a_backslash_is_refused_on_posix(tmp_path):
    """`_permission_settings` rewrites backslashes to `/` so a Windows-shaped
    path renders at all; on POSIX, where a backslash is an ordinary filename
    character, that rewrite names a different directory."""
    from sidecar.objectstore import BlobStoreConfig, local_blob_root

    bad = tmp_path / "cor\\pus"
    with pytest.raises(ValueError, match="backslash"):
        transcript_mod.transcript_dir(str(bad))
    with pytest.raises(ValueError, match="backslash"):
        local_blob_root(BlobStoreConfig(backend="file", root=str(bad)))


def test_an_unset_store_credential_prefix_names_nothing(monkeypatch):
    """An empty prefix must not resolve to bare `_ACCESS_KEY_ID`."""
    monkeypatch.setattr(agentic_settings, "transcript_store_creds_env_prefix", "")
    assert agentic._store_credential_vars() == set()


@pytest.mark.parametrize("value", ["", None, 1234])
def test_scrubber_skips_values_it_must_not_replace(value):
    """Only the two that cannot name a credential occurrence: an empty needle
    (matches everywhere) and a non-string (has no spelling on the wire)."""
    assert Scrubber.for_secrets([("X", value)]).replacements == ()


# --- a truncated line cannot keep a credential fragment (#251) --------------


def test_a_credential_prefix_left_by_a_cut_is_redacted_off_a_truncated_line():
    """Exact-substring scrubbing matches whole values only, so a line cut
    mid-credential holds a prefix nothing matches (#251)."""
    scrubber = Scrubber.for_secrets([("ANTHROPIC_API_KEY", SECRET)])
    cut = '{"type":"user","content":"token=' + SECRET[:20]
    assert scrubber.scrub(cut) == cut  # the gap #251 documents
    assert (
        scrubber.scrub_partial(cut)
        == '{"type":"user","content":"token=[REDACTED:ANTHROPIC_API_KEY]'
    )


def test_a_complete_event_is_untouched_by_the_truncated_line_pass():
    """The real protection against over-redaction: a whole NDJSON event ends in
    `}`, and no credential begins with one."""
    scrubber = Scrubber.for_secrets([("ANTHROPIC_API_KEY", SECRET)])
    line = json.dumps({"type": "result", "ok": True})
    assert scrubber.scrub_partial(line) == line


def test_the_longest_fragment_wins_rather_than_the_first_needle_to_match():
    """`replacements` is ordered by needle LENGTH, which is not match length.
    Returning on the first needle that matches at all would redact four
    characters of a coincidence and leave the rest of a genuinely truncated
    second credential on disk, under the wrong marker."""
    cut_secret = "B" * 30
    long_secret = "BBBB" + "z" * 40  # sorts first: the longest needle
    scrubber = Scrubber.for_secrets([("LONG_KEY", long_secret), ("CUT_KEY", cut_secret)])
    # The cut left 20 characters of CUT_KEY on the line; their last four also
    # happen to be LONG_KEY's first four, which is all a first-match-wins scan
    # would redact -- leaving 16 characters of a live credential behind it.
    line = "trailing " + cut_secret[:20]
    assert scrubber.scrub_partial(line) == "trailing [REDACTED:CUT_KEY]"

    # And the ordinary case: the longer of two genuine fragments is taken.
    other = "B" * 10 + "y" * 20
    scrubber = Scrubber.for_secrets([("OTHER", other), ("CUT_KEY", cut_secret)])
    assert scrubber.scrub_partial("trailing " + cut_secret[:20]) == ("trailing [REDACTED:CUT_KEY]")


def test_a_fragment_shorter_than_the_minimum_is_left_alone():
    scrubber = Scrubber.for_secrets([("ANTHROPIC_API_KEY", SECRET)])
    short = "trailing text ending in " + SECRET[: transcript_mod.MIN_FRAGMENT - 1]
    assert scrubber.scrub_partial(short) == short


def test_a_line_with_a_newline_never_takes_the_truncated_pass(monkeypatch):
    """Only the last line a pipe yields can lack its newline, so the extra pass
    costs at most one line per run."""
    calls = []
    transcript, sink = _transcript([("ANTHROPIC_API_KEY", SECRET)])
    monkeypatch.setattr(
        transcript_mod.Scrubber, "scrub_partial", lambda self, text: calls.append(text) or text
    )
    transcript.write('{"a": 1}\n')
    assert calls == []
    transcript.write('{"a": 1')
    assert calls == ['{"a": 1']


def test_a_timeout_kill_that_cuts_a_credential_leaves_no_fragment_on_disk(tmp_path):
    """#251's acceptance, driven through the real kill path: the harness is
    killed while a credential is half-written, and nothing of it reaches disk."""
    events = [{"type": "assistant", "n": index} for index in range(3)]
    straddle = '{"type":"user","content":"export TOKEN=' + SECRET[:20]
    assert _feed_script(
        tmp_path,
        events,
        tail=f"sys.stdout.write({straddle!r})\nsys.stdout.flush()\ntime.sleep(60)\n",
    ).exists()
    transcript = open_transcript(
        [("GITHUB_APP_TOKEN", SECRET)], directory=str(tmp_path / "transcripts")
    )
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
    written = (tmp_path / "transcripts" / f"{transcript.key}.ndjson").read_text()
    assert SECRET[: transcript_mod.MIN_FRAGMENT] not in written
    assert "[REDACTED:GITHUB_APP_TOKEN]" in written
    # Everything that streamed before the cut still survives (#236).
    assert [json.loads(line) for line in written.splitlines()[:3]] == events


def test_a_clean_run_keeps_a_final_event_that_ends_without_a_newline(tmp_path):
    """The companion #251 asks for: the guard must not cost a complete final
    event, which is why the line is kept and only the fragment removed."""
    assert _feed_script(
        tmp_path,
        [{"type": "assistant", "n": 0}],
        tail=f"sys.stdout.write(json.dumps({RESULT_EVENT!r}))\n",
    ).exists()
    transcript = open_transcript(
        [("GITHUB_APP_TOKEN", SECRET)], directory=str(tmp_path / "transcripts")
    )
    result = run_review(
        "p",
        tmp_path,
        cwd=tmp_path,
        model="m",
        env={**os.environ, "PATH": str(tmp_path)},
        timeout=30,
        transcript=transcript,
    )
    assert result.timed_out is False
    written = (tmp_path / "transcripts" / f"{transcript.key}.ndjson").read_text().splitlines()
    assert json.loads(written[-1]) == RESULT_EVENT


# --- shipping the finished file off the runner (#238) ----------------------


def test_the_finished_file_is_shipped_on_close_and_only_then(tmp_path):
    shipped = []
    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    sink = ShippingTranscriptSink(
        inner, "k", lambda key, path: shipped.append((key, path.read_bytes()))
    )
    sink.write('{"a": 1}')
    assert shipped == []
    sink.close()
    assert shipped == [("k", b'{"a": 1}\n')]


def test_a_run_that_wrote_nothing_ships_nothing(tmp_path):
    shipped = []
    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    ShippingTranscriptSink(inner, "k", lambda key, path: shipped.append(key)).close()
    assert shipped == []
    assert not (tmp_path / "t.ndjson").exists()


def test_closing_twice_ships_once(tmp_path):
    """`TranscriptSink.close` promises repeat calls are safe, and the key is
    write-once -- a second ship would answer 409 and turn a harmless repeat
    call into a reported capture failure."""
    shipped = []
    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    sink = ShippingTranscriptSink(inner, "k", lambda key, path: shipped.append(key))
    sink.write('{"a": 1}')
    sink.close()
    sink.close()
    assert shipped == ["k"]


def test_a_close_after_a_failed_ship_does_not_retry_it(tmp_path):
    """One attempt is the decision this module already made: the blob is
    write-once, so a retry races an upload that may already have landed."""
    calls = []

    def _boom(key, path):
        calls.append(key)
        raise RuntimeError("sidecar unreachable")

    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    sink = ShippingTranscriptSink(inner, "k", _boom)
    sink.write('{"a": 1}')
    with pytest.raises(RuntimeError):
        sink.close()
    sink.close()
    assert calls == ["k"]


def test_a_store_that_declined_the_bytes_indexes_nothing(tmp_path):
    """Shipping to a sidecar whose store is unconfigured is a SILENT success --
    that silence is the point of the off state -- but nothing was stored, so a
    reference would name a blob that only ever existed on this runner's disk."""
    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    sink = ShippingTranscriptSink(inner, "k", lambda key, path: False)
    transcript = Transcript("k", sink, Scrubber())
    transcript.write(json.dumps(_tool_use("Read", file_path="a.py")) + "\n")
    transcript.close()
    assert sink.stored is False
    assert transcript.index() is None


def test_a_store_that_accepted_the_bytes_is_indexed(tmp_path):
    """The other half of the same flag, and the shape a ship callable that
    reports nothing still lands in: raising is how this interface reports a
    failure, so a silent return means stored."""
    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    sink = ShippingTranscriptSink(inner, "k", lambda key, path: None)
    transcript = Transcript("k", sink, Scrubber())
    transcript.write(json.dumps(_tool_use("Read", file_path="a.py")) + "\n")
    transcript.close()
    assert sink.stored is True
    assert transcript.index().tool_calls == {"Read": 1}


def test_a_first_write_that_fails_ships_nothing(tmp_path, monkeypatch):
    """`opened` has to mean a line LANDED, not that the file was created:
    otherwise a failure on the very first write ships an empty blob under a
    write-once key, contradicting "a run that wrote nothing ships nothing"."""
    shipped = []
    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    sink = ShippingTranscriptSink(inner, "k", lambda key, path: shipped.append(key))
    transcript = Transcript("k", sink, Scrubber())

    monkeypatch.setattr(FileTranscriptSink, "write", lambda self, line: _fail_after_create(self))
    transcript.write('{"a": 1}')
    monkeypatch.undo()
    transcript.close()
    assert shipped == []


def _fail_after_create(sink):
    """Create the file the way `write` does, then fail before any line lands."""
    sink.path.parent.mkdir(mode=sink.DIR_MODE, parents=True, exist_ok=True)
    sink.path.touch(mode=sink.FILE_MODE)
    raise OSError("no space left on device")


def test_a_transcript_that_went_inert_is_still_shipped(tmp_path):
    """What is on disk is a clean PREFIX -- capture goes inert on the first
    failure rather than skipping a line and resuming -- which is the shape #236
    keeps."""
    shipped = []
    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    sink = ShippingTranscriptSink(inner, "k", lambda key, path: shipped.append(key))
    transcript = Transcript("k", sink, Scrubber())
    transcript.write('{"a": 1}')
    transcript._fail(OSError("no space left on device"))
    transcript.write('{"a": 2}')
    transcript.close()
    assert shipped == ["k"]
    assert (tmp_path / "t.ndjson").read_text() == '{"a": 1}\n'


def test_a_failed_upload_is_one_stderr_line_and_never_a_fault(tmp_path, capsys):
    def _boom(key, path):
        raise httpx.ConnectError("sidecar unreachable")

    inner = FileTranscriptSink(tmp_path / "t.ndjson")
    transcript = Transcript("k", ShippingTranscriptSink(inner, "k", _boom), Scrubber())
    transcript.write('{"a": 1}')
    transcript.close()
    err = capsys.readouterr().err
    assert err.count("fuko: transcript k capture failed") == 1
    assert "sidecar unreachable" in err


def test_capture_only_deployments_never_wrap_the_sink(tmp_path, monkeypatch):
    """With nowhere to ship, #237's behaviour is unchanged -- a local file and
    no attempt, rather than a failure reported once a run."""
    monkeypatch.setattr(transcript_mod, "upload_target", lambda: "")
    transcript = open_transcript([], directory=str(tmp_path))
    assert isinstance(transcript._sink, FileTranscriptSink)


def test_a_reachable_destination_wraps_the_sink_and_is_announced(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(transcript_mod, "upload_target", lambda: "sidecar")
    transcript = open_transcript([], directory=str(tmp_path))
    assert isinstance(transcript._sink, ShippingTranscriptSink)
    assert "-> sidecar" in capsys.readouterr().err


# --- The per-run index the capture derives as it stores (#239).


def _tool_use(name, **tool_input):
    return {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": name, "input": tool_input}]},
    }


def _tool_result(content):
    return {"type": "user", "message": {"content": [{"type": "tool_result", "content": content}]}}


#: One file read three times, two others read once each, plus a non-read tool
#: and two tool results in the two shapes the CLI spells them.
_INDEX_FEED = [
    {"type": "system", "subtype": "init"},
    _tool_use("Read", file_path="src/app.py"),
    _tool_result("x" * 40),
    _tool_use("Read", file_path="src/app.py", offset=200),
    _tool_result([{"type": "text", "text": "y" * 10}]),
    _tool_use("Read", file_path="src/app.py"),
    _tool_use("Read", file_path="src/other.py"),
    _tool_use("Read", file_path="docs/readme.md"),
    _tool_use("Grep", pattern="def "),
    RESULT_EVENT,
]


def _indexed(events):
    transcript, _ = _transcript()
    for event in events:
        transcript.write(json.dumps(event) + "\n")
    transcript.close()
    return transcript.index()


def test_the_index_counts_calls_per_tool_result_bytes_and_repeated_reads():
    """#239's fixture feed: five Read calls over three files, one Grep, and 50
    bytes of tool-result text across both content shapes."""
    index = _indexed(_INDEX_FEED)
    assert index.key == "key-1"
    assert index.tool_calls == {"Read": 5, "Grep": 1}
    assert index.tool_result_bytes == 50
    # ONE file, not three reads and not three files -- a re-read at a different
    # offset is still a second trip through the same file.
    assert index.repeated_read_files == 1
    assert index.complete is True


def test_the_index_travels_as_a_plain_mapping():
    """Both metrics transports carry it identically, so neither has to know the
    dataclass."""
    assert _indexed(_INDEX_FEED).as_dict() == {
        "key": "key-1",
        "complete": True,
        "tool_calls": {"Read": 5, "Grep": 1},
        "tool_result_bytes": 50,
        "repeated_read_files": 1,
    }


def test_a_feed_cut_short_is_indexed_as_incomplete():
    """A run killed at `tool_timeout` never emits its terminal `result` event.
    Its prefix is a real measurement of a real run, so it is indexed -- and
    marked, rather than dropped or recorded as if it had finished."""
    index = _indexed(_INDEX_FEED[:-1])
    assert index.complete is False
    assert index.tool_calls == {"Read": 5, "Grep": 1}


def test_a_run_that_stored_nothing_has_no_index():
    """No file, nothing shipped -- an index row would name a blob that does not
    exist."""
    transcript, _ = _transcript()
    transcript.write("   \n")
    transcript.close()
    assert transcript.index() is None


def test_a_capture_that_failed_has_no_index():
    """Conservative on purpose: the same flag covers a failed upload, and a
    reference to a blob that was never stored is worse than a missing row."""
    transcript, _ = _transcript(sink=_RecordingSink(fail_at=2))
    transcript.write(json.dumps(_tool_use("Read", file_path="a.py")) + "\n")
    transcript.write(json.dumps(RESULT_EVENT) + "\n")
    transcript.close()
    assert transcript.index() is None


def test_the_index_measures_only_the_lines_the_sink_accepted():
    """A line lost to a full disk is not one the index row claims."""
    transcript, sink = _transcript(sink=_RecordingSink(fail_at=3))
    transcript.write(json.dumps(_tool_use("Read", file_path="a.py")) + "\n")
    transcript.write(json.dumps(_tool_result("z" * 7)) + "\n")
    transcript.write(json.dumps(_tool_use("Grep", pattern="x")) + "\n")
    assert len(sink.lines) == 2
    assert transcript._meter.tool_calls == {"Read": 1}
    assert transcript._meter.tool_result_bytes == 7


def test_the_index_is_derived_from_the_scrubbed_bytes():
    """Metering the stored text is what makes a reader recomputing the figures
    from the blob (#240) agree with the ones written at capture."""
    transcript, sink = _transcript(secrets=[("TOKEN", SECRET)])
    transcript.write(json.dumps(_tool_result(f"key={SECRET}")) + "\n")
    transcript.close()
    assert SECRET not in sink.lines[0]
    assert transcript.index().tool_result_bytes == len("key=[REDACTED:TOKEN]".encode())


def test_the_meter_skips_what_it_cannot_parse():
    """Same tolerance the progress fold has: a blank, non-JSON or non-object
    line -- including the truncated final event `scrub_partial` leaves -- costs
    a figure, never a review."""
    transcript, _ = _transcript()
    transcript.write("not json at all\n")
    transcript.write("[1, 2, 3]\n")
    transcript.write('{"type": "assistant", "message": "not a dict"}\n')
    transcript.write('{"type": "future_event_kind"}\n')
    # An assistant turn of prose, and a content list holding something that is
    # not a block at all: neither is a tool call.
    transcript.write(
        json.dumps(
            {"type": "assistant", "message": {"content": ["nope", {"type": "text", "text": "hi"}]}}
        )
        + "\n"
    )
    transcript.write('{"type": "assistant", "message": {"content": [{"type": "tool_use"')
    transcript.close()
    index = transcript.index()
    assert index.tool_calls == {} and index.tool_result_bytes == 0
    assert index.repeated_read_files == 0 and index.complete is False


def test_a_nameless_tool_use_counts_under_the_folds_own_placeholder():
    transcript, _ = _transcript()
    transcript.write(
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use"}]}}) + "\n"
    )
    transcript.close()
    assert transcript.index().tool_calls == {"?": 1}


@pytest.mark.parametrize("name", [["Read"], {"tool": "Read"}, 7, ""])
def test_an_unhashable_or_non_string_tool_name_folds_into_the_placeholder(name):
    """The name is a dict KEY, so a feed spelling it as a list or an object
    would raise `TypeError` out of `write()` -- past the sink's own guard, into
    the review path this module promises never to fault."""
    transcript, _ = _transcript()
    transcript.write(
        json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name}]}}
        )
        + "\n"
    )
    transcript.close()
    assert transcript.index().tool_calls == {"?": 1}


def test_a_line_that_cannot_be_metered_costs_the_figures_and_not_the_review():
    """`write()` calls the meter OUTSIDE the guard protecting the sink, so a
    shape the meter cannot fold would fault the review path. An unpaired
    surrogate escape is legal JSON that `json.loads` hands back as a lone
    surrogate, which `.encode("utf-8")` refuses -- one of the two shapes that
    got past enumeration, which is why the guard is blanket."""
    transcript, sink = _transcript()
    line = json.dumps(_tool_result("x")).replace('"x"', '"\\ud800"')
    transcript.write(line + "\n")
    transcript.write(json.dumps(_tool_use("Read", file_path="a.py")) + "\n")
    transcript.close()
    # The bytes still reached the sink; only the measurement of them is lost.
    assert len(sink.lines) == 2
    assert transcript.index() is None


def test_a_non_text_tool_result_block_contributes_no_guessed_bytes():
    """This figure is what the run was fed IN TEXT; an image's size is not
    stated, and inventing one would put two definitions in one column."""
    transcript, _ = _transcript()
    transcript.write(
        json.dumps(_tool_result([{"type": "image", "source": {"data": "..."}}])) + "\n"
    )
    transcript.write(json.dumps(_tool_result({"unexpected": "shape"})) + "\n")
    transcript.close()
    assert transcript.index().tool_result_bytes == 0


def test_a_read_without_a_usable_path_is_counted_but_not_tracked():
    transcript, _ = _transcript()
    transcript.write(json.dumps(_tool_use("Read")) + "\n")
    transcript.write(json.dumps(_tool_use("Read", file_path="")) + "\n")
    transcript.close()
    index = transcript.index()
    assert index.tool_calls == {"Read": 2} and index.repeated_read_files == 0


def test_a_capture_with_nowhere_to_ship_is_not_indexed(tmp_path, monkeypatch):
    """The `fuko review` laptop case: no `FUKO_URL`, no store backend, so the
    transcript is a local file. The figures are real, but nothing else can fetch
    what they describe, and a reference is a promise that it can."""
    # Both, and in this order: `upload_target()` answers "sidecar" on `FUKO_URL`
    # alone, so an ambient one would give this run a destination and the
    # assertion would be testing the environment rather than the code.
    monkeypatch.delenv("FUKO_URL", raising=False)
    monkeypatch.setattr(settings, "transcript_store_backend", "")
    transcript = open_transcript([], directory=str(tmp_path / "transcripts"))
    transcript.write(json.dumps(_tool_use("Read", file_path="a.py")) + "\n")
    transcript.close()
    assert transcript.index() is None


def test_run_review_indexes_the_feed_it_captured(tmp_path, monkeypatch):
    """End to end through the real harness and a real store: the figures come
    off the same feed the tee stored, with nothing re-downloaded."""
    # `FUKO_URL` outranks the configured store in `upload_target()`, so an
    # ambient one would ship this over HTTP and leave the file store unexercised.
    monkeypatch.delenv("FUKO_URL", raising=False)
    monkeypatch.setattr(settings, "transcript_store_backend", "file")
    monkeypatch.setattr(settings, "transcript_store_root", str(tmp_path / "blobs"))
    assert _feed_script(tmp_path, _INDEX_FEED).exists()
    transcript = open_transcript([], directory=str(tmp_path / "transcripts"))
    run_review(
        "p",
        tmp_path,
        cwd=tmp_path,
        model="m",
        env={**os.environ, "PATH": str(tmp_path)},
        timeout=30,
        transcript=transcript,
    )
    index = transcript.index()
    assert index.key == transcript.key
    assert index.tool_calls == {"Read": 5, "Grep": 1}
    assert index.tool_result_bytes == 50 and index.repeated_read_files == 1
    assert index.complete is True
