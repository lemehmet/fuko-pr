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
  inside :class:`Transcript`, rather than in whichever sink is configured, so a
  later remote sink (#238) plugs in BELOW the scrub and cannot forget it.
* **Never fails a review.** Same direction as
  :func:`sidecar.review_state._best_effort`: a misconfigured destination, a
  disk that fills mid-stream, a sink that raises -- each degrades to one stderr
  line and an inert transcript, never to a fault in the review path.

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

    def close(self) -> None:
        """Close the file if it was ever opened."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


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

    def write(self, line: str) -> None:
        """Scrub one raw feed line and hand it to the sink.

        Blank lines are dropped rather than stored: the feed's own framing is
        one event per line, so whitespace carries no event, and passing it on
        would put unparseable lines in an NDJSON file for no gain. Everything
        else is written AS RECEIVED, including lines the fold skips -- the
        ``user`` tool-result events, and anything the CLI's schema grows next.
        """
        if self._failed or self._closed or not line.strip():
            return
        try:
            self._sink.write(self._scrubber.scrub(line))
        except Exception as e:
            self._fail(e)

    def close(self) -> None:
        """Close the sink; idempotent, and never raises into the review path."""
        if self._closed:
            return
        self._closed = True
        try:
            self._sink.close()
        except Exception as e:  # pragma: no cover - defensive, same guard as write
            self._fail(e)

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
    """
    destination = settings.transcript_dir if directory is None else directory
    return Path(destination).expanduser() if destination else None


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
    """
    try:
        destination = transcript_dir(directory)
        if destination is None:
            return None
        key = mint_key()
        path = destination / f"{key}.ndjson"
        transcript = Transcript(key, FileTranscriptSink(path), Scrubber.for_secrets(secrets))
    except Exception as e:
        # Misconfiguration must degrade like every other capture failure: the
        # review runs, and the operator gets a line saying why it has no
        # transcript.
        print(f"fuko: transcript capture unavailable: {e}", file=sys.stderr)
        return None
    print(
        f"fuko: agentic {label} transcript {path}" if label else f"fuko: transcript {path}",
        file=sys.stderr,
    )
    return transcript
