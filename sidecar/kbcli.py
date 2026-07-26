"""``fuko kb`` — HTTP-client subcommands to browse a running sidecar's knowledge base.

Unlike ``fuko query``/``fuko forget`` (which open the local store directly via
``.fuko.toml``), these talk to a running sidecar over HTTP, so they work from any
machine. Configuration is two environment variables:

  ``FUKO_URL``         base URL of the sidecar (default ``http://localhost:8000``)
  ``FUKO_AUTH_TOKEN``  bearer token (required)
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from .models import SOURCES

_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _color(value, code: str) -> str:
    """Wrap ``value`` in an ANSI code when stdout is a TTY, else return it plain."""
    return f"{code}{value}{_RESET}" if sys.stdout.isatty() else str(value)


def _call(method: str, path: str, params: dict | None = None, body: dict | None = None) -> dict:
    """Make an authenticated request to the sidecar and return the decoded JSON.

    Exits with a clear message on a missing token, an HTTP error, or an
    unreachable host rather than raising a traceback.
    """
    token = os.environ.get("FUKO_AUTH_TOKEN")
    if not token:
        sys.exit("fuko kb: set FUKO_AUTH_TOKEN (the sidecar bearer token)")
    base = os.environ.get("FUKO_URL", "http://localhost:8000").rstrip("/")
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"fuko kb: {e.code} {e.reason} — {e.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"fuko kb: cannot reach {base} ({e.reason})")
    except ValueError as e:
        sys.exit(f"fuko kb: invalid URL or non-JSON response from {base} ({e})")


def _print_learning(item: dict, full: bool) -> None:
    """Print one learning as a two-line block (head + text), with extras when ``full``."""
    globs = ",".join(item.get("file_globs") or []) or "—"
    head = f"{_color(item['repo'], _CYAN)}  {_color(item['source'], _DIM)}  {_color(globs, _DIM)}"
    text = item["text"].replace("\n", " ")
    if not full:
        text = text[:140] + ("…" if len(item["text"]) > 140 else "")
    print(f"• {head}\n  {text}")
    if full:
        meta = " ".join(p for p in (item.get("id", ""), item.get("source_url") or "") if p)
        if meta:
            print(f"  {_color(meta, _DIM)}")


def _list(args) -> None:
    resp = _call(
        "GET",
        "/learnings",
        params={
            "repo": args.repo,
            "source": args.source,
            "limit": args.limit,
            "offset": args.offset,
            "q": args.q,
            "include_expired": args.include_expired or None,
        },
    )
    items = resp["learnings"]
    print(_color(f"{len(items)} shown · {resp['count']} total\n", _BOLD))
    for item in items:
        _print_learning(item, args.full)


def _repos(args) -> None:
    repos = _call("GET", "/repos")["repos"]
    total = sum(r["count"] for r in repos)
    print(_color(f"{len(repos)} repo(s) · {total} learnings\n", _BOLD))
    for entry in repos:
        breakdown = " ".join(f"{s}={n}" for s, n in sorted(entry["sources"].items()))
        print(f"  {entry['repo']:28} {entry['count']:>6}  {_color(breakdown, _DIM)}")


def _count(args) -> None:
    repos = _call("GET", "/repos")["repos"]
    if args.repo:
        repos = [r for r in repos if r["repo"] == args.repo]
    rows = [
        (entry["repo"], source, n)
        for entry in repos
        for source, n in sorted(entry["sources"].items())
        if not args.source or source == args.source
    ]
    print(_color(f"{sum(n for _, _, n in rows)} total\n", _BOLD))
    for repo, source, n in sorted(rows):
        print(f"  {repo:24} {_color(f'{source:16}', _DIM)} {n}")


_CLEARABLE = ("source_url", "topic", "expires_at")


def _edit(args) -> None:
    changes: dict = {}
    for name, value in (
        ("text", args.text),
        ("source", args.source),
        ("source_url", args.source_url),
        ("topic", args.topic),
        ("expires_at", args.expires_at),
    ):
        if value is None:
            continue
        changes[name] = None if (value == "" and name in _CLEARABLE) else value
    if args.globs is not None:
        changes["file_globs"] = args.globs
    if not changes:
        sys.exit("fuko kb: nothing to change; pass at least one field")
    updated = _call("PATCH", f"/learnings/{args.id}", body={"repo": args.repo, **changes})
    print(_color("updated", _BOLD))
    _print_learning(updated, full=True)


def _query(args) -> None:
    body = {"repo": args.repo, "files": args.files or [], "query_text": args.text}
    if args.pr_body:
        body["pr_body"] = args.pr_body
    if args.top_k is not None:
        body["top_k"] = args.top_k
    resp = _call("POST", "/query", body=body)
    results = resp["results"]
    print(_color(f"{len(results)} result(s)\n", _BOLD))
    for item in results:
        print(_color(f"• score {item['score']:.3f}", _DIM))
        _print_learning({**item, "repo": args.repo}, full=True)


def _forget(args) -> None:
    if not (args.id or args.source or args.all):
        sys.exit("fuko kb: provide --id, --source, or --all")
    if (args.source or args.all) and not args.yes:
        what = "ALL learnings" if args.all else f"all source={args.source!r} learnings"
        if input(f"Delete {what} for {args.repo}? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("aborted")
    selector = {"repo": args.repo, "id": args.id, "source": args.source, "all": args.all}
    resp = _call("POST", "/forget", body={k: v for k, v in selector.items() if v})
    print(_color(f"deleted {resp['deleted']}", _BOLD))


def add_parser(sub) -> None:
    """Register the ``kb`` subcommand group on the top-level subparsers."""
    p = sub.add_parser("kb", help="browse/query a running sidecar's KB over HTTP")
    kb = p.add_subparsers(dest="kb_cmd", required=True)

    pl = kb.add_parser("list", help="list stored learnings (newest first)")
    pl.add_argument("--repo")
    pl.add_argument("--source")
    pl.add_argument("--limit", type=int, default=100)
    pl.add_argument("--offset", type=int, default=0)
    pl.add_argument("-q", "--query", dest="q", help="case-insensitive substring of text or topic")
    pl.add_argument(
        "--include-expired", dest="include_expired", action="store_true", help="show expired too"
    )
    pl.add_argument("--full", action="store_true", help="print full text, id, and source url")
    pl.set_defaults(kb_fn=_list)

    pr = kb.add_parser("repos", help="every repo with a knowledge base, and its source mix")
    pr.set_defaults(kb_fn=_repos)

    pc = kb.add_parser("count", help="total plus a breakdown by repo and source")
    pc.add_argument("--repo")
    pc.add_argument("--source")
    pc.set_defaults(kb_fn=_count)

    pe = kb.add_parser("edit", help="change fields of one stored learning")
    pe.add_argument("repo")
    pe.add_argument("id")
    pe.add_argument("--text")
    pe.add_argument("--source", choices=SOURCES)
    pe.add_argument("--source-url", dest="source_url", help='pass "" to clear')
    pe.add_argument("--topic", help='pass "" to clear')
    pe.add_argument("--expires-at", dest="expires_at", help='pass "" to clear')
    pe.add_argument("--globs", nargs="*", help="replace the file globs (pass none to clear)")
    pe.set_defaults(kb_fn=_edit)

    pq = kb.add_parser("query", help="semantic, file-scoped retrieval (what a review sees)")
    pq.add_argument("repo")
    pq.add_argument("--files", nargs="*", help="changed file paths to scope retrieval")
    pq.add_argument("--text", help="query text")
    pq.add_argument("--pr-body", dest="pr_body")
    pq.add_argument("--top-k", dest="top_k", type=int)
    pq.set_defaults(kb_fn=_query)

    pf = kb.add_parser("forget", help="delete learnings by id, source, or all (repo-scoped)")
    pf.add_argument("repo")
    selector = pf.add_mutually_exclusive_group(required=True)
    selector.add_argument("--id")
    selector.add_argument("--source")
    selector.add_argument("--all", action="store_true")
    pf.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    pf.set_defaults(kb_fn=_forget)


def dispatch(args) -> None:
    """Run the selected ``kb`` subcommand."""
    args.kb_fn(args)
