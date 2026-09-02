"""``fuko transcripts`` — list captured agentic sessions and fetch one (#240).

The first reader over the transcript corpus epic #236 builds: an operator with
SSH finds the runs that reviewed a pull request, sees what each spent its turns
on, and pulls one session down to read or grep.

An HTTP client over the running sidecar, the same shape and the same two
environment variables as ``fuko kb`` — the sidecar is what holds both the index
(Postgres) and the blobs (object storage), and neither is reachable from a
laptop:

  ``FUKO_URL``         base URL of the sidecar (default ``http://localhost:8000``)
  ``FUKO_AUTH_TOKEN``  bearer token (required)

Every failure exits NON-ZERO with a message naming it. That is the point of the
sub-issue rather than a nicety: a store that is unconfigured or unreachable must
never print the empty listing a healthy, empty corpus prints.
"""

import http.client
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

#: Read budget for one call. Generous next to ``fuko kb``'s 30s because a fetch
#: streams a whole session — a long agentic review's transcript is megabytes,
#: and the sidecar may be pulling it from object storage first.
_TIMEOUT_S = 300

#: Tools named in a listing line before it says "and N more". The vocabulary is
#: the harness's and grows without a migration, so a run that touched a dozen
#: tools must not push the figures that matter off the line; ``--full`` shows
#: every one.
_TOOLS_SHOWN = 4


def _color(value, code: str) -> str:
    """Wrap ``value`` in an ANSI code when stdout is a TTY, else return it plain."""
    return f"{code}{value}{_RESET}" if sys.stdout.isatty() else str(value)


def _url(path: str, params: dict | None = None) -> tuple[str, str]:
    """Return the sidecar base URL and the full URL for ``path``.

    Exits when no token is configured: an unauthenticated call would come back
    401 and read like a store that refused, when the fix is local.
    """
    if not os.environ.get("FUKO_AUTH_TOKEN"):
        sys.exit("fuko transcripts: set FUKO_AUTH_TOKEN (the sidecar bearer token)")
    base = os.environ.get("FUKO_URL", "http://localhost:8000").rstrip("/")
    url = base + path
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        if query:
            url += "?" + query
    return base, url


#: What a transfer raises when it fails anywhere urllib does not reach. urllib
#: wraps only what its own handler raises around ``h.request``, so both the
#: ``h.getresponse()`` that follows and every read after the response opened
#: surface bare: ``TimeoutError``/``ConnectionResetError`` (both ``OSError``
#: since 3.10), or an ``http.client`` ``IncompleteRead`` / ``BadStatusLine``.
_BODY_FAILURES = (OSError, http.client.HTTPException)


def _open(url: str, base: str):
    """Open an authenticated GET, mapping every failure to a fatal message.

    The HTTP status is preserved in the message rather than collapsed, because
    the sidecar's taxonomy is what tells the operator which fault this is: 404
    is a key that holds nothing, 503 is a store that could not be looked in at
    all, and #240 exists so those two never look the same.
    """
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", "Bearer " + os.environ["FUKO_AUTH_TOKEN"])
    try:
        return urllib.request.urlopen(req, timeout=_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        # `e.read()` is itself a socket read, so it gets the same treatment the
        # body reads below do: a peer that dies while we are collecting its
        # error text must still exit with a message rather than a traceback.
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except _BODY_FAILURES:
            detail = "(error body could not be read)"
        # The body is decoded ONLY when it is the object the sidecar's own
        # HTTPException produces. Valid JSON that is not a mapping -- `null`, a
        # list, a bare string, all of which an intermediary proxy may return --
        # decodes without raising, so a bare `.get` would be an AttributeError
        # escaping as a traceback out of the one path whose whole job is to name
        # the fault. The undecoded text is a fine message; a crash is not.
        try:
            parsed = json.loads(detail)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            detail = parsed.get("detail", detail)
        sys.exit(f"fuko transcripts: {e.code} {e.reason} — {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"fuko transcripts: cannot reach {base} ({e.reason})")
    except ValueError as e:
        sys.exit(f"fuko transcripts: invalid URL {url!r} ({e})")
    except _BODY_FAILURES as e:
        # urllib wraps only what its OWN handler raises around `h.request`; the
        # `h.getresponse()` that follows is unguarded, so a peer that hangs up
        # or stalls while sending its status line arrives here as a bare
        # `RemoteDisconnected`, `BadStatusLine` or `TimeoutError` rather than as
        # a `URLError`. Same fault the operator sees for a refused connection,
        # so it gets the same named exit.
        sys.exit(f"fuko transcripts: cannot reach {base} ({e})")


def _body_failed(base: str, error: Exception) -> None:
    """Exit fatally on a transfer that died mid-body, naming the fault.

    The module promises that every failure exits non-zero with a message naming
    it, and a transcript is exactly the payload long enough for the connection
    to die halfway through it. Without this the operator gets a traceback from
    the middle of a read loop, which names Python's frames rather than the
    outage.
    """
    sys.exit(f"fuko transcripts: transfer from {base} failed mid-body ({error})")


def _partial_gone(path: str) -> None:
    """Remove a half-written ``--out`` file, best effort.

    A truncated transcript is the one failure here that outlives the command:
    every other one is a message on the way out, but a short file stays on disk
    reading like a whole session, and the operator greps it a week later and
    concludes the run did less than it did. Whatever stops the unlink, the
    fault the caller is about to report is the one worth seeing.
    """
    try:
        os.unlink(path)
    except OSError:
        pass


def _exit_closed_pipe() -> None:
    """Exit 0 after stdout's reader went away, with no second error on the way out.

    ``fuko transcripts get KEY | head`` closes the pipe early, which is the
    intended use. But the ``EPIPE`` surfaces from a flush with the tail still in
    ``sys.stdout``'s buffer, so leaving on ``sys.exit`` alone hands that same
    buffer to the interpreter's final flush, which retries the dead pipe and
    prints ``Exception ignored ... BrokenPipeError`` while exiting non-zero.
    Pointing the fd at ``/dev/null`` first -- CPython's own prescription -- lets
    that last flush succeed silently, so the quiet exit the pipe case deserves
    is the one it gets.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except OSError:
        # No real stdout fd to redirect (a captured or already-closed stream).
        # Nothing is buffered against a pipe in that case either.
        pass
    sys.exit(0)


def _call(path: str, params: dict | None = None) -> dict:
    """GET ``path`` and return the decoded JSON body, which must BE an object.

    The type is checked here rather than left to the caller's first `.get`: a
    200 carrying valid JSON that is not a mapping -- ``null``, a list, a bare
    string, all of which an intermediary can answer with -- would otherwise
    reach the listing as an ``AttributeError`` traceback out of the module that
    promises to name every fault.
    """
    base, url = _url(path, params)
    with _open(url, base) as resp:
        try:
            body = json.load(resp)
        except ValueError as e:
            sys.exit(f"fuko transcripts: non-JSON response from {base} ({e})")
        except _BODY_FAILURES as e:
            _body_failed(base, e)
    if not isinstance(body, dict):
        sys.exit(f"fuko transcripts: {base} answered with JSON that is not an object")
    return body


def _human_bytes(count: int) -> str:
    """Render a byte count for a listing line (binary units, one decimal)."""
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"  # pragma: no cover - the loop always returns


def _where(item: dict) -> str:
    """The ``repo#pr`` this transcript belongs to, or a marker that nothing claims it.

    A transcript with no run row is not an error and not a gap in the corpus:
    the reference is written in a separate transaction after the index row, so a
    lost metrics post leaves a real stored session that nothing names. Saying so
    is better than printing ``None#None``, which reads as a bug.
    """
    if not item.get("repo"):
        return "(no run row)"
    pr = item.get("pr")
    return f"{item['repo']}#{pr}" if pr is not None else item["repo"]


def _plural(count: int, noun: str) -> str:
    """``count`` with ``noun``, pluralized by the naive ``s`` rule these two need."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _tools(item: dict, full: bool) -> str:
    """Per-tool call counts, busiest first, capped unless ``full``."""
    calls = item.get("tool_calls") or {}
    if not calls:
        return "no tool calls"
    ordered = sorted(calls.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ordered if full else ordered[:_TOOLS_SHOWN]
    rendered = " ".join(f"{name}={count}" for name, count in shown)
    if len(ordered) > len(shown):
        rendered += f" +{len(ordered) - len(shown)} more"
    return f"{_plural(sum(calls.values()), 'call')} ({rendered})"


def _print_run(item: dict, full: bool) -> None:
    """Print one transcript as a two-line block: identity, then what it spent."""
    seat = item.get("seat") or "—"
    model = item.get("model") or "—"
    when = item.get("started_at") or item.get("created_at") or "—"
    head = (
        f"{_color(item['key'], _CYAN)}  {_where(item)}  "
        f"{seat}  {_color(model, _DIM)}  {_color(when, _DIM)}"
    )
    # INCOMPLETE is called out rather than left to be inferred from a small
    # figure: an acceptance criterion of #240, so a session cut short is never
    # mistaken for a cheap one.
    state = "complete" if item.get("complete") else _color("INCOMPLETE", _BOLD)
    figures = (
        f"{state}  {_tools(item, full)}  "
        f"{_human_bytes(int(item.get('tool_result_bytes') or 0))} results  "
        f"{_plural(int(item.get('repeated_read_files') or 0), 'file')} re-read"
    )
    print(f"• {head}\n  {figures}")
    if full and item.get("outcome"):
        duration = item.get("duration_s")
        extra = f"{item['outcome']}"
        if duration is not None:
            extra += f" in {float(duration):.1f}s"
        if item.get("provider"):
            extra += f" via {item['provider']}"
        if item.get("backend"):
            extra += f" ({item['backend']})"
        print(f"  {_color(extra, _DIM)}")


def _list(args) -> None:
    resp = _call(
        "/transcripts",
        params={
            "repo": args.repo,
            "pr": args.pr,
            "seat": args.seat,
            "since": args.since,
            "until": args.until,
            "limit": args.limit,
            "offset": args.offset,
        },
    )
    # A body with no `transcripts` key at all is not an empty corpus, it is
    # something other than this sidecar answering -- and "0 shown · 0 total" is
    # exactly the rendering #240 exists to keep faults out of. Checked before
    # anything is printed, so the operator never sees a listing at all.
    # A body with no `transcripts` key at all -- or one holding anything but a
    # list -- is not an empty corpus, it is something other than this sidecar
    # answering, and "0 shown · 0 total" is exactly the rendering #240 exists to
    # keep faults out of. `null` is called out separately because `or []` would
    # otherwise turn it into the healthy-empty page rather than a fault.
    if "transcripts" not in resp:
        sys.exit("fuko transcripts: the listing response carried no 'transcripts' field")
    if not isinstance(resp["transcripts"], list):
        sys.exit("fuko transcripts: the listing response's 'transcripts' field is not a list")
    # The same closed-pipe contract `get` has: `list | head` and
    # `list --json | jq -e ... | head` are documented uses, and a reader that
    # walked away is not a fault to report.
    try:
        if args.json:
            # The raw body, so `| jq` works on the listing the same way `get`
            # makes it work on the session. Nothing is reformatted, same reason.
            json.dump(resp, sys.stdout, indent=2)
            print()
            sys.stdout.flush()
            return
        items = resp["transcripts"]
        print(_color(f"{len(items)} shown · {resp.get('count', 0)} total\n", _BOLD))
        for item in items:
            _print_run(item, args.full)
        sys.stdout.flush()
    except BrokenPipeError:
        _exit_closed_pipe()
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        # The render itself, rather than each field it touches. A row is read
        # through a dozen accessors and coercions, and enumerating the shapes
        # each one rejects would be a schema validator that still missed the
        # next one. Whatever a row turns out not to be, the operator gets the
        # named exit this module promises instead of a traceback -- and this is
        # the last layer where that can still go wrong, since everything below
        # is `.get(...)` with a default.
        sys.exit(f"fuko transcripts: the listing carried a row this reader cannot read ({e})")


def _get(args) -> None:
    base, url = _url(f"/transcripts/{urllib.parse.quote(args.key, safe='')}")
    with _open(url, base) as resp:
        if args.out:
            written = 0
            try:
                handle = open(args.out, "wb")
            except OSError as e:
                sys.exit(f"fuko transcripts: cannot write {args.out} ({e})")
            try:
                with handle:
                    # Copied in chunks rather than read whole: a long session is
                    # megabytes, and this command exists to make one greppable
                    # on a box that may have far less memory than the sidecar.
                    #
                    # The read and the write are caught SEPARATELY even though
                    # both raise `OSError`: this command writes multi-megabyte
                    # files onto boxes chosen for having less memory than the
                    # sidecar, so ENOSPC/EDQUOT is a plausible failure here, and
                    # blaming the transfer for a full local disk sends the
                    # operator to look at the network for a fault on their own
                    # machine.
                    while True:
                        try:
                            chunk = resp.read(1 << 20)
                        except _BODY_FAILURES as e:
                            _partial_gone(args.out)
                            _body_failed(base, e)
                        if not chunk:
                            break
                        try:
                            handle.write(chunk)
                        except OSError as e:
                            _partial_gone(args.out)
                            sys.exit(f"fuko transcripts: cannot write {args.out} ({e})")
                        written += len(chunk)
            except OSError as e:
                # The buffered flush `with handle:` performs on the way out --
                # a local write like any other, and the one place ENOSPC most
                # often actually surfaces.
                _partial_gone(args.out)
                sys.exit(f"fuko transcripts: cannot write {args.out} ({e})")
            print(f"wrote {written} bytes to {args.out}", file=sys.stderr)
            return
        out = sys.stdout.buffer
        try:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
            out.flush()
        except BrokenPipeError:
            # Caught BEFORE the transfer failures below: `BrokenPipeError` is an
            # `OSError`, and a reader that walked away is the one case here that
            # is not a fault at all.
            _exit_closed_pipe()
        except _BODY_FAILURES as e:
            _body_failed(base, e)


def add_parser(sub) -> None:
    """Register the ``transcripts`` subcommand group on the top-level subparsers."""
    p = sub.add_parser(
        "transcripts",
        help="list captured agentic sessions and fetch one, over HTTP",
        description="Reads a running sidecar's transcript corpus via FUKO_URL + "
        "FUKO_AUTH_TOKEN. Exits non-zero when the store is unconfigured or "
        "unreachable, so that never looks like an empty corpus.",
    )
    t = p.add_subparsers(dest="transcripts_cmd", required=True)

    pl = t.add_parser("list", help="list captured transcripts (newest first)")
    pl.add_argument("--repo", help="owner/name")
    pl.add_argument("--pr", type=int, help="pull request number")
    pl.add_argument("--seat", help="seat (slot) label the model occupied")
    pl.add_argument("--since", help="ISO date or timestamp; inclusive, UTC when bare")
    pl.add_argument(
        "--until",
        help="ISO date or timestamp; EXCLUSIVE, UTC when bare, so a bare date "
        "ends the window at the start of that day",
    )
    pl.add_argument("--limit", type=int, default=50)
    pl.add_argument("--offset", type=int, default=0)
    pl.add_argument("--full", action="store_true", help="every tool, plus the run's outcome")
    pl.add_argument("--json", action="store_true", help="print the raw response body")
    pl.set_defaults(transcripts_fn=_list)

    pg = t.add_parser(
        "get",
        help="write one transcript's stored bytes to stdout (or --out)",
        description="The stored NDJSON verbatim, for a pager, a grep or a jq. "
        "Nothing is reformatted.",
    )
    pg.add_argument("key", help="the transcript key, as shown by `fuko transcripts list`")
    pg.add_argument("-o", "--out", help="write to this file instead of stdout")
    pg.set_defaults(transcripts_fn=_get)


def dispatch(args) -> None:
    """Run the selected ``transcripts`` subcommand."""
    args.transcripts_fn(args)
