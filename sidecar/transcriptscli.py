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
        detail = e.read().decode("utf-8", "replace")[:500]
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        sys.exit(f"fuko transcripts: {e.code} {e.reason} — {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"fuko transcripts: cannot reach {base} ({e.reason})")
    except ValueError as e:
        sys.exit(f"fuko transcripts: invalid URL {url!r} ({e})")


def _call(path: str, params: dict | None = None) -> dict:
    """GET ``path`` and return the decoded JSON body."""
    base, url = _url(path, params)
    with _open(url, base) as resp:
        try:
            return json.load(resp)
        except ValueError as e:
            sys.exit(f"fuko transcripts: non-JSON response from {base} ({e})")


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
    if args.json:
        # The raw body, so `| jq` works on the listing the same way `get` makes
        # it work on the session. Nothing is reformatted for the same reason.
        json.dump(resp, sys.stdout, indent=2)
        print()
        return
    items = resp.get("transcripts") or []
    print(_color(f"{len(items)} shown · {resp.get('count', 0)} total\n", _BOLD))
    for item in items:
        _print_run(item, args.full)


def _get(args) -> None:
    base, url = _url(f"/transcripts/{urllib.parse.quote(args.key, safe='')}")
    with _open(url, base) as resp:
        if args.out:
            written = 0
            with open(args.out, "wb") as handle:
                # Copied in chunks rather than read whole: a long session is
                # megabytes, and this command exists to make one greppable on a
                # box that may have far less memory than the sidecar does.
                while chunk := resp.read(1 << 20):
                    handle.write(chunk)
                    written += len(chunk)
            print(f"wrote {written} bytes to {args.out}", file=sys.stderr)
            return
        out = sys.stdout.buffer
        try:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
            out.flush()
        except BrokenPipeError:
            # `fuko transcripts get KEY | head` closes the pipe early, which is
            # the intended use, not a failure. Left to Python's own exit path
            # rather than reported.
            sys.exit(0)


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
