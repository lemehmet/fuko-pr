"""Streaming, credential-scrubbed capture of the agentic harness's event feed.

The capture half of the transcript epic (#236, sub-issue #237). The harness
already receives the CLI's whole NDJSON event feed and folds it away
(:func:`sidecar.reviewer.harness._consume_stream` handles ``assistant`` and
``result`` only), so ``user`` events -- where TOOL RESULTS live, ~91% of a
review's token spend -- are read and discarded in the same pass. This module is
the tee that keeps them.

Three properties are load-bearing, and each of them is a constraint the epic
settled rather than a preference:

* **Write-through, never buffered.** The fold iterates ``proc.stdout`` lazily on
  purpose: a 30-minute review whose tool results are megabytes of file content
  never has the feed in memory at once. The tee therefore writes each line as it
  arrives and holds none of them, so peak memory is one event rather than one
  session, and a run cut short (a kill, a timeout, sidecar death) leaves
  everything that streamed before the cut on disk. That is also why the file is
  line-buffered: a tail lost to a buffer is the case this exists for.
* **Scrubbed before any byte is written, irreversibly.** Scrubbing lives HERE,
  inside :class:`Transcript`, rather than in whichever sink is configured, so
  the remote sink (#238) plugs in BELOW the scrub and cannot forget it.
  Exact-value replacement is per WHOLE line, and the one line that may not be
  whole -- a trailing newline-less one, which only a kill or a mid-write death
  can produce -- additionally has a trailing credential FRAGMENT redacted; see
  :meth:`Scrubber.scrub_partial` and #251.
* **Never fails a review.** Same direction as
  :func:`sidecar.review_state._best_effort`: a misconfigured destination, a
  disk that fills mid-stream, a sink that raises -- each degrades to one stderr
  line and an inert transcript, never to a fault in the review path.

The capture also METERS itself as it goes (:class:`_Meter`, #239): the figures
that say what a run spent its turns on -- per-tool call counts, tool-result
bytes, re-read files -- are folded out of the same lines on their way to the
sink, so nothing is re-downloaded and nothing is held. They ride to the metrics
row as a :class:`TranscriptIndex`.

**What gets scrubbed, stated as a rule:** the EXACT credential values the driver
holds, and nothing inferred. :meth:`Scrubber.for_secrets` is handed
``(name, value)`` pairs by :mod:`sidecar.backends.agentic`, which is the one
place that knows both what it injected into the harness environment and what it
deliberately stripped from it. A credential-SHAPED string that was never
injected is written through verbatim: shape-matching is at best a second layer
(the epic's decision), and a heuristic here would be a second source of truth
next to the driver's credential lists -- the two would drift, and the direction
they drift in is silent corpus corruption, since scrubbing cannot be undone.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ..config import settings
from .transcript_client import ship, upload_target


#: Shortest credential PREFIX :meth:`Scrubber.scrub_partial` will redact off the
#: end of a truncated line. Four characters is short enough that no fragment
#: worth having survives and long enough that a coincidental tail in
#: non-JSON output is not silently eaten; see that method for why the real
#: protection against over-redaction is the shape of a complete event, not this
#: number.
MIN_FRAGMENT = 4


def _redaction(name: str) -> str:
    """The marker a scrubbed value is replaced by.

    Names the variable and never the value or its length: which credential
    appeared where is the useful half for debugging, and a length is a hint at
    the secret. The marker is deliberately free of quotes and backslashes so
    substituting it inside a JSON string leaves the line parseable.
    """
    return f"[REDACTED:{name}]"


@dataclass(frozen=True)
class Scrubber:
    """Replaces known credential values in a line of the event feed."""

    replacements: tuple[tuple[str, str], ...] = ()

    @classmethod
    def for_secrets(cls, secrets: Sequence[tuple[str, str]]) -> Scrubber:
        """Build a scrubber for ``(name, value)`` credential pairs.

        Each value is registered twice: verbatim, and in its JSON-escaped
        spelling, because the feed is NDJSON and a value carrying a quote or a
        backslash appears on the wire escaped rather than raw. Only empty and
        non-string values are skipped -- an empty needle matches between every
        pair of characters, so it would redact the whole feed.

        There is deliberately NO minimum length. A short value here is still a
        credential: the caller hands this method the exact values of variables
        it already decided are credential-bearing, so the "an env var holding
        ``1`` or ``true`` is not a secret" hazard belongs to a scrubber that
        reads the whole environment, which this one never does. Length would
        instead decide that an operator's short ``FUKO_TOKEN`` is written to
        durable storage verbatim -- and between a leaked credential and a
        transcript with an over-eager redaction in it, only one is recoverable.

        Longest needle first, so a credential that happens to contain another
        one cannot be half-replaced into a string that no longer matches the
        longer rule.
        """
        needles: dict[str, str] = {}
        for name, value in secrets:
            if not isinstance(value, str) or not value:
                continue
            marker = _redaction(name)
            for needle in (value, json.dumps(value)[1:-1]):
                needles.setdefault(needle, marker)
        ordered = sorted(needles.items(), key=lambda item: len(item[0]), reverse=True)
        return cls(tuple(ordered))

    def scrub(self, text: str) -> str:
        """Return ``text`` with every known credential value replaced."""
        for needle, marker in self.replacements:
            if needle in text:
                text = text.replace(needle, marker)
        return text

    def scrub_partial(self, text: str) -> str:
        """Scrub a line that may have been CUT MID-VALUE (#251).

        Exact-substring replacement holds for every whole line and fails for
        exactly one: when ``tool_timeout`` kills the harness, or the child dies
        mid-write, iterating the pipe yields a trailing line with no newline. If
        the cut landed inside a credential the line holds only a PREFIX of the
        needle, no registered value matches, and the fragment would be written
        durably. So after the exact pass this also redacts a trailing suffix of
        the line that is a proper prefix of some needle.

        #237 accepted that residual on four mitigations, two of which -- a
        ``0600`` file in a ``0700`` directory, and the harness read denylist --
        are properties of the RUNNER'S COPY. #238 ships the bytes to shared
        object storage, where neither applies, so the residual widens exactly
        where it stops being bounded to one owner-only file. That is why the
        guard lands here rather than at the upload: by upload time the signal is
        gone, because :meth:`FileTranscriptSink.write` normalizes every line by
        appending a newline, so the file never ends without one.

        Chosen over #251's option 2 (drop the newline-less line on a timeout)
        because dropping it would also discard a COMPLETE final event -- the
        ``result`` event among them -- and "a run cut short keeps everything
        that streamed before the cut" is a settled decision of #236. This keeps
        the line and removes only the fragment.

        Over-redaction is bounded to nothing in practice: a complete NDJSON
        event ends in ``}``, and no credential value begins with ``}``, so on a
        clean run whose last write merely lacked a newline every candidate
        suffix fails at its first character. :data:`MIN_FRAGMENT` is the guard
        for the other case -- a final line that is not JSON at all -- where a
        one- or two-character coincidence would otherwise eat real text.

        The LONGEST match across every needle wins, not the first needle that
        matches at all. ``replacements`` is ordered by needle length, which is
        not the same as match length: a long secret whose first four characters
        happen to end the line would otherwise claim the redaction and leave the
        rest of a different, genuinely truncated secret on disk -- under the
        wrong marker. Scanning all of them costs one pass over a handful of
        registered values, once per run.
        """
        text = self.scrub(text)
        best_size, best_marker = 0, ""
        for needle, marker in self.replacements:
            # A full match was already handled by `scrub`, hence `len(needle) - 1`.
            for size in range(min(len(text), len(needle) - 1), best_size, -1):
                if size >= MIN_FRAGMENT and text.endswith(needle[:size]):
                    best_size, best_marker = size, marker
                    break
        if best_size:
            return text[:-best_size] + best_marker
        return text


class TranscriptSink(Protocol):
    """Where scrubbed transcript lines go.

    The seam #238 extends: a shipped-off-the-runner implementation is another
    class satisfying this protocol, not a rewrite of the capture path. Sinks
    receive ALREADY-SCRUBBED text and may assume nothing else about it.
    """

    def write(self, line: str) -> None:
        """Append one already-scrubbed event line."""
        ...

    def close(self) -> None:
        """Release the sink; called once, and safe to call again."""
        ...


class FileTranscriptSink:
    """Appends lines to one local NDJSON file.

    The file is opened on the FIRST write, so a run that streams no events
    leaves no empty file behind and a capture nobody exercises costs no inode.
    It is line-buffered because the tail is the interesting part of a cut-short
    run: a killed process's buffer is not flushed by anyone.

    Owner-only, not umask's choice. Scrubbing removes the credential values the
    driver holds and nothing else, so what stays is the whole reviewed
    repository as the agent read it -- which is the same content the checkout
    itself carries, and ``invoke`` gives that ``mkdtemp``'s ``0o700``. A
    transcript outlives the checkout by design (the corpus is kept), so leaving
    it at a default ``0o644`` would make the durable copy the readable one.
    """

    DIR_MODE = 0o700
    FILE_MODE = 0o600

    def __init__(self, path: Path) -> None:
        """Record the destination path; nothing is created until first write."""
        self.path = path
        #: Whether a line ever LANDED -- set after the first successful write,
        #: not merely after the file is created, so a first write that raises
        #: leaves an empty file that a wrapping sink will not ship (#238).
        #: Survives `close()`, which clears the handle, so that sink can tell
        #: "empty run" from "closed".
        self.opened = False
        self._handle = None

    def write(self, line: str) -> None:
        """Append ``line``, creating the file and its parents on first use."""
        if self._handle is None:
            self.path.parent.mkdir(mode=self.DIR_MODE, parents=True, exist_ok=True)
            # `os.open` rather than `Path.open`, because the mode has to be set
            # AS the file is created: a chmod afterwards leaves a window in
            # which the file exists world-readable. `mode` is still subject to
            # the umask, which can only narrow it further -- never widen it.
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, self.FILE_MODE)
            self._handle = open(fd, "a", encoding="utf-8", buffering=1)
        self._handle.write(line if line.endswith("\n") else line + "\n")
        self.opened = True

    def close(self) -> None:
        """Close the file if it was ever opened."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class ShippingTranscriptSink:
    """A local sink whose finished file is sent to shared storage on close (#238).

    A DECORATOR rather than a replacement, because the local file is not a
    staging artifact: #237 ships with a workflow uploading it as a run artifact,
    and the run-order-sortable directory is what an operator greps on the box.
    So the write path is unchanged -- line-buffered, write-through, one event of
    peak memory -- and shipping is one bounded act at the end, on bytes that are
    already whole and already scrubbed.

    Why close-time rather than per line: a keyed write-once blob is created by
    one request, and there is no append. Streaming events individually would
    mean either a request per event or an interface with a commit, and the
    upload would then straddle the review instead of following it.

    An INERT transcript (a sink failure partway through, see
    :meth:`Transcript._fail`) is still shipped. What is on disk in that case is
    a clean PREFIX -- capture goes inert on the first failure rather than
    skipping a line and resuming -- which is the same shape a kill leaves, and
    #236 keeps those. A run that wrote nothing at all ships nothing: there is no
    file, and an empty blob would only make ``#239``'s index row point at one.
    """

    def __init__(self, inner: FileTranscriptSink, key: str, ship) -> None:
        """Wrap ``inner``, shipping its file under ``key`` via the ``ship`` callable."""
        self.path = inner.path
        self._inner = inner
        self._key = key
        self._ship = ship
        self._closed = False
        #: Whether shared storage ACCEPTED these bytes; ``None`` until
        #: :meth:`close` has tried. This attribute EXISTING is itself half the
        #: signal :meth:`Transcript.index` reads (#239): a sink that cannot
        #: affirm storage is read as not having stored, so a bare
        #: :class:`FileTranscriptSink` -- a capture with nowhere to ship --
        #: never becomes a reference. The other half is this value, because
        #: shipping can also succeed silently without storing anything (the
        #: sidecar's marked off state).
        self.stored: bool | None = None

    def write(self, line: str) -> None:
        """Append one already-scrubbed line to the local file."""
        self._inner.write(line)

    def close(self) -> None:
        """Close the local file, then ship it; any failure propagates to the caller.

        Closing FIRST is load-bearing: the handle is line-buffered, so the last
        line is only guaranteed on disk once it is closed, and shipping before
        that would upload a file missing its own tail.

        Exceptions are deliberately not swallowed here. :meth:`Transcript.close`
        is the one place that decides what a capture failure costs -- one stderr
        line and an inert transcript, never a fault in the review path -- and a
        second, quieter handler here would report the same failure twice or not
        at all.

        Idempotent, as :class:`TranscriptSink` promises: the key is write-once,
        so a second close that shipped again would answer ``409`` and turn a
        harmless repeat call into a reported failure. The flag is set BEFORE the
        upload, so a failed ship is not retried by a later close either -- one
        attempt is the decision this module already made.
        """
        if self._closed:
            return
        self._closed = True
        self._inner.close()
        if self._inner.opened:
            # `is not False`, not `bool(...)`: only an EXPLICIT refusal marks
            # these bytes unstored. A ship callable that returns nothing has
            # returned without raising, and raising is how this interface
            # reports every real failure -- so silence means it stored.
            self.stored = self._ship(self._key, self.path) is not False


#: The tool whose calls the repeated-read figure is derived from.
#:
#: One tool, named exactly, rather than "anything that could read a file". A
#: ``Grep`` with a path, or a ``Bash`` running ``cat``, also puts file content
#: in the context, but neither states WHICH file in a field that survives a
#: schema change, and counting them would make the number a heuristic rather
#: than a count. #159 asks whether a run re-reads what it already read; ``Read``
#: is where that is answerable exactly.
READ_TOOL = "Read"


@dataclass(frozen=True)
class TranscriptIndex:
    """What one captured transcript is indexed by (#239).

    The figures a run's own feed can state about itself: which tools it called
    and how often, how many bytes of tool results it was fed, how many files it
    read more than once, and whether the feed reached its end. They exist to
    answer what a run spent its turns ON -- ``review_runs`` already answers how
    much it spent.
    """

    key: str
    complete: bool
    tool_calls: dict[str, int]
    tool_result_bytes: int
    repeated_read_files: int

    def as_dict(self) -> dict:
        """This index as the plain mapping the metrics transport carries.

        A mapping rather than the dataclass itself, for the same reason the
        token/cost figures travel as one (:func:`sidecar.runner._costs_of`):
        both metrics transports -- the HTTP body and the direct Postgres call --
        then carry it identically, and neither has to know this class.
        """
        return {
            "key": self.key,
            "complete": self.complete,
            "tool_calls": dict(self.tool_calls),
            "tool_result_bytes": self.tool_result_bytes,
            "repeated_read_files": self.repeated_read_files,
        }


def _blocks(event: dict) -> list:
    """The content blocks of one feed event, or an empty list."""
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return content if isinstance(content, list) else []


def _text_bytes(content) -> int:
    """UTF-8 bytes of a ``tool_result``'s content, whatever shape it arrived in.

    The CLI spells one result either as a bare string or as a list of typed
    blocks, and both are common in a single feed. Non-text blocks (an image, a
    shape the schema grows later) contribute nothing rather than a guess: this
    figure is what the run was FED IN TEXT, and inventing a byte count for
    something whose size is not stated would put two definitions in one column.
    """
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    total = 0
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                total += len(block["text"].encode("utf-8"))
    return total


class _Meter:
    """Folds a stored feed into the per-run figures :class:`TranscriptIndex` holds.

    Fed the SCRUBBED line, after the sink accepted it, so the figures describe
    exactly the bytes that were stored: a reader recomputing them from the blob
    (#240's option) gets the same numbers, and a capture that went inert
    part-way measures the prefix that survived rather than the run that did not.

    Tolerant in the same direction as
    :func:`sidecar.reviewer.harness._consume_stream`: a blank, non-JSON or
    unrecognised line is skipped. That covers the one line
    :meth:`Scrubber.scrub_partial` can leave un-parseable -- a truncated final
    event -- and it means a schema the CLI grows costs a figure, never a
    review.

    Memory is a counter per tool plus a counter per distinct file READ, which
    is bounded by the repository rather than by the run: the whole point of the
    streaming capture is that no run holds its feed at once, and this holds
    less than the run's file list.
    """

    def __init__(self) -> None:
        """Start every figure at nothing measured yet."""
        self.tool_calls: dict[str, int] = {}
        self.tool_result_bytes = 0
        self.complete = False
        self._reads: dict[str, int] = {}

    @property
    def repeated_read_files(self) -> int:
        """How many distinct files were read more than once.

        FILES, not reads: a file read three times is one repeated file, which
        is the quantity #159 asks about -- whether a stateless run re-derives
        what it already had.
        """
        return sum(1 for count in self._reads.values() if count > 1)

    def feed(self, line: str) -> None:
        """Fold one stored line into the running figures."""
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "assistant":
            self._assistant(event)
        elif kind == "user":
            self._tool_results(event)
        elif kind == "result":
            # The terminal event, and the only evidence the feed is whole. A
            # transcript without it was cut short -- by the `tool_timeout` kill,
            # a sidecar death, or a disk that filled -- and the index row says
            # so rather than letting a partial run read as a finished one.
            self.complete = True

    def _assistant(self, event: dict) -> None:
        for block in _blocks(event):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            # `"?"` matches the progress fold's own placeholder for a nameless
            # tool_use block, so one vocabulary describes both. The isinstance
            # test is what keeps it a placeholder rather than a crash: a feed
            # naming a tool with a list or an object would otherwise be used as
            # a dict key and raise TypeError out of `write()`, past the sink's
            # own guard, into the review path this module promises never to
            # fault. An unrecognised shape costs a tool name, never a review.
            raw_name = block.get("name")
            name = raw_name if isinstance(raw_name, str) and raw_name else "?"
            self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
            if name == READ_TOOL:
                self._read(block.get("input"))

    def _read(self, tool_input) -> None:
        path = tool_input.get("file_path") if isinstance(tool_input, dict) else None
        if isinstance(path, str) and path:
            # The path AS THE AGENT WROTE IT, unnormalized. Two spellings of one
            # file would undercount, but normalizing here would invent a
            # filesystem the sidecar cannot see (the checkout is long gone by
            # the time this row is read), and a re-read at a different offset
            # counts as a re-read -- it is a second trip through the same file,
            # which is the cost being measured.
            self._reads[path] = self._reads.get(path, 0) + 1

    def _tool_results(self, event: dict) -> None:
        for block in _blocks(event):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                self.tool_result_bytes += _text_bytes(block.get("content"))


class Transcript:
    """One run's capture: an identity, a scrubber, and a sink.

    ``key`` is minted at run start and is the transcript's own identity, not a
    borrowed one -- ``review_runs`` cannot supply it, because that row is
    inserted after the run finishes and its id is never returned (#236). Later
    sub-issues key the stored blob and the index row by this value.

    Every sink call is guarded. The FIRST failure reports one stderr line and
    makes the transcript permanently inert: a disk that filled at event 900
    would otherwise report on every event after it, turning a degraded capture
    into a log flood on the runs that are already the largest.
    """

    def __init__(self, key: str, sink: TranscriptSink, scrubber: Scrubber) -> None:
        """Bind the run's key to the sink that stores it and the scrubber guarding it."""
        self.key = key
        self._sink = sink
        self._scrubber = scrubber
        self._failed = False
        self._closed = False
        self._meter = _Meter()
        #: Lines the sink ACCEPTED. Counted here rather than read off the sink,
        #: because :class:`ShippingTranscriptSink` wraps the file sink and
        #: ``TranscriptSink`` promises nothing about what landed -- and because
        #: this has to survive `close()`, which clears the file handle.
        self._stored = 0

    def write(self, line: str) -> None:
        """Scrub one raw feed line and hand it to the sink.

        Blank lines are dropped rather than stored: the feed's own framing is
        one event per line, so whitespace carries no event, and passing it on
        would put unparseable lines in an NDJSON file for no gain. Everything
        else is written AS RECEIVED, including lines the fold skips -- the
        ``user`` tool-result events, and anything the CLI's schema grows next.

        A line WITHOUT a trailing newline is scrubbed as a possibly-truncated
        one (:meth:`Scrubber.scrub_partial`, #251). Only the last line a pipe
        yields can lack its newline, so this costs the extra pass at most once
        per run and needs nothing plumbed in about how the run ended -- which is
        also what makes it cover a mid-write death, not just the timeout kill.
        """
        if self._failed or self._closed or not line.strip():
            return
        scrub = self._scrubber.scrub if line.endswith("\n") else self._scrubber.scrub_partial
        scrubbed = scrub(line)
        try:
            self._sink.write(scrubbed)
        except Exception as e:
            self._fail(e)
            return
        # METERED AFTER the sink accepted it (#239), so the figures describe
        # what was stored rather than what was offered: a line lost to a full
        # disk is not one the index row claims.
        self._stored += 1
        self._meter.feed(scrubbed)

    def close(self) -> None:
        """Close the sink; idempotent, and never raises into the review path.

        With a shipping sink configured this is where the transcript leaves the
        runner, so it may do bounded network I/O -- bounded by that sink's own
        timeout, which is the whole latency #238 allows the review's completion
        path to take on.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._sink.close()
        except Exception as e:  # pragma: no cover - defensive, same guard as write
            self._fail(e)

    def index(self) -> TranscriptIndex | None:
        """This capture's index row (#239), or ``None`` when there is nothing to index.

        Call it AFTER :meth:`close`, which is where a shipping sink either
        delivers the bytes or fails; the figures themselves are complete from
        the last :meth:`write` on, and closing clears no state this reads.

        The rule is ONE thing, stated once: a row is written only for bytes that
        reached SHARED storage, because the key it is written under is what a
        reader fetches them by (``migrations/013``). So the sink has to AFFIRM
        that it stored them -- anything that does not is read as "not stored",
        which is why this is ``not getattr(...)`` and not a test against
        ``False``. ``None`` therefore in four cases, all of which leave the
        reference on the ``review_runs`` row NULL rather than naming a blob:

        * **Nothing landed.** A run that streamed no events leaves no file and
          ships nothing (:class:`ShippingTranscriptSink`), so an index row would
          point at a blob that does not exist.
        * **There was nowhere to ship it.** With no destination configured
          (:func:`sidecar.reviewer.transcript_client.upload_target`) the sink is
          a bare :class:`FileTranscriptSink` and the transcript is a local file
          on this runner -- the ``fuko review`` laptop case. The figures are
          real, but nothing else can fetch what they describe.
        * **Storage declined the bytes.** Shipping to a sidecar whose blob store
          is unconfigured is a SILENT success -- deliberately, so staging
          capture ahead of storage does not print a line per run -- but nothing
          was stored, and that is the configuration the deployment docs
          recommend passing through.
        * **The capture failed.** Conservative on purpose, and knowingly
          over-suppressing one case: a sink that failed at event 900 still has
          its clean prefix shipped by ``close``, and that prefix would be
          indexable. But the same ``None`` covers a failure IN ``close`` -- an
          upload that never landed -- and from here the two are one flag. A
          missing index row costs an observability row; a reference to a blob
          that was never stored costs a reader that cannot tell a lost
          transcript from an unstored one, which is the distinction #236 built
          the nullable reference to preserve.

        A transcript whose feed was CUT SHORT is not one of those cases: it is
        indexed, with ``complete`` false. What streamed before the cut is a real
        measurement of a real run, and dropping it would bias the corpus towards
        runs that finished.
        """
        if self._failed or not self._stored:
            return None
        if not getattr(self._sink, "stored", False):
            return None
        return TranscriptIndex(
            key=self.key,
            complete=self._meter.complete,
            tool_calls=dict(self._meter.tool_calls),
            tool_result_bytes=self._meter.tool_result_bytes,
            repeated_read_files=self._meter.repeated_read_files,
        )

    def _fail(self, error: Exception) -> None:
        """Report one stderr line and go inert for the rest of the run."""
        if not self._failed:
            self._failed = True
            print(f"fuko: transcript {self.key} capture failed: {error}", file=sys.stderr)


def mint_key() -> str:
    """A fresh transcript key: UTC timestamp first, then a random suffix.

    Sortable by eye and by name (the runner's transcripts list in run order),
    and unique without coordination -- concurrent seats are threads in one
    process and mint keys milliseconds apart.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:12]}"


def transcript_dir(directory: str | None = None) -> Path | None:
    """The configured transcript destination, resolved, or ``None`` when unset.

    Shared with the driver so the READ-DENY rule the harness is given covers the
    path this module actually writes: the corpus is a durable, cross-repo
    archive of everything past runs read, and the reviewing agent keeps
    ``Read``/``Grep``/``Glob`` over the whole runner, so an unmatched deny rule
    would leave it readable by an agent whose findings are published verbatim to
    an untrusted PR author. Two spellings of the same path would be exactly that
    (see :func:`sidecar.reviewer.harness._permission_settings`).

    CANONICAL, not merely expanded, because that function drops any path which
    is not POSIX-absolute and matches literally otherwise -- so the two ways a
    destination can be written and not denied are both silent:

    * a RELATIVE destination renders no rule at all (the non-absolute branch
      announces it, but the sink still writes there), and
    * a SYMLINKED one renders a rule for the alias while the sink writes through
      to the target, which the agent can then read under its real name.

    ``resolve()`` is non-strict, so a destination that does not exist yet still
    comes back absolute rather than raising -- the first capture creates it.

    The filesystem ROOT is rejected outright, and that is a third spelling of
    the same failure rather than a tidiness rule: ``_permission_settings``
    normalizes a candidate with ``rstrip("/")``, which turns ``"/"`` into the
    empty string, and an empty candidate is dropped WITHOUT reaching the
    non-POSIX announcement -- so a root destination would write transcripts that
    no rule covers and nothing reports. Refusing it here keeps the two sides
    consistent: the driver gets no deny path and the capture does not open, so
    the "written but undenied" state cannot be reached at all.
    """
    destination = settings.transcript_dir if directory is None else directory
    if not destination:
        return None
    resolved = Path(destination).expanduser().resolve()
    if resolved.parent == resolved:
        raise ValueError(
            f"transcript_dir {resolved} is the filesystem root; "
            "set FUKO_TRANSCRIPT_DIR to a dedicated directory"
        )
    if "\n" in str(resolved):
        # A FOURTH spelling of the same failure, and one #238 introduced: the
        # driver hands the deny paths over newline-separated (a directory name
        # may legally contain `:`, so `os.pathsep` was worse), and a name
        # holding a newline is split into two candidates -- a rule for some
        # other directory, and a tail dropped as non-POSIX -- while the
        # transcript still lands at the real path. Refused for the same reason
        # the root is: the capture must not open where no rule reaches.
        raise ValueError(
            "transcript_dir contains a newline, which the read-denylist "
            "hand-off cannot represent; rename the directory"
        )
    return resolved


def open_transcript(
    secrets: Sequence[tuple[str, str]],
    *,
    label: str = "",
    directory: str | None = None,
) -> Transcript | None:
    """Open this run's transcript, or return ``None`` when capture is off.

    Capture is off unless a destination is configured (``FUKO_TRANSCRIPT_DIR``,
    or an explicit ``directory``): defaulting it on would write to an
    unconfigured path on every runner, which is exactly what #237 rules out.
    ``None`` -- rather than an inert object -- is what lets the harness skip the
    tee entirely, so a fleet that never turns this on pays nothing per event.

    ``secrets`` are ``(name, value)`` pairs; see the module docstring for the
    rule they follow. ``label`` names the seat on the announcement line, which
    is what makes a transcript findable (a workflow uploads it as an artifact).

    The local file is always written; shipping it to shared storage (#238) is
    added on top when there is somewhere to ship it
    (:func:`sidecar.reviewer.transcript_client.upload_target`). Gating on a
    destination rather than on a second opt-in switch is what keeps a
    capture-only deployment silent: with no store reachable there is no attempt
    and therefore no failure to report.
    """
    try:
        destination = transcript_dir(directory)
        if destination is None:
            return None
        key = mint_key()
        path = destination / f"{key}.ndjson"
        local = FileTranscriptSink(path)
        target = upload_target()
        sink: TranscriptSink = ShippingTranscriptSink(local, key, ship) if target else local
        transcript = Transcript(key, sink, Scrubber.for_secrets(secrets))
    except Exception as e:
        # Misconfiguration must degrade like every other capture failure: the
        # review runs, and the operator gets a line saying why it has no
        # transcript.
        print(f"fuko: transcript capture unavailable: {e}", file=sys.stderr)
        return None
    where = f"{path} -> {target}" if target else str(path)
    print(
        f"fuko: agentic {label} transcript {where}" if label else f"fuko: transcript {where}",
        file=sys.stderr,
    )
    return transcript
