"""Command-line interface for the fuko-pr sidecar."""

import argparse
import glob as globmod
import os
import re
import sys
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*)")


def main() -> None:
    """Parse command-line arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(prog="fuko", description="fuko-pr sidecar CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="run the HTTP sidecar")

    p_review = sub.add_parser("review", help="review a PR through the configured backend")
    p_review.add_argument("--pr-url", required=True, help="full pull request URL")
    p_review.add_argument("--config", default=".fuko.toml", help="path to .fuko.toml")

    p_signals = sub.add_parser(
        "signals", help="emit canonical Review Signals (v1) for a PR as JSON"
    )
    p_signals.add_argument("--pr-url", required=True, help="full pull request URL")
    p_signals.add_argument("--config", default=".fuko.toml", help="path to .fuko.toml")

    p_status = sub.add_parser(
        "status", help="emit per-reviewer review state on a PR's HEAD as JSON"
    )
    p_status.add_argument("--pr-url", required=True, help="full pull request URL")

    p_query = sub.add_parser("query", help="query learnings for a set of changed files")
    p_query.add_argument("--repo", required=True)
    p_query.add_argument("--file", action="append", default=[], help="changed file path")
    p_query.add_argument("--text", help="explicit query text")
    p_query.add_argument("--top-k", type=int, default=None)
    p_query.add_argument("--config", default=".fuko.toml", help="path to .fuko.toml")

    p_docs = sub.add_parser("ingest-docs", help="ingest markdown/text docs as learnings")
    p_docs.add_argument("paths", nargs="+", help="files or globs to ingest")
    p_docs.add_argument("--repo", required=True)
    p_docs.add_argument("--glob", action="append", default=[], help="file_globs to attach")
    p_docs.add_argument("--source-url", default=None)
    p_docs.add_argument("--config", default=".fuko.toml", help="path to .fuko.toml")

    p_forget = sub.add_parser("forget", help="remove learnings")
    p_forget.add_argument("--repo", required=True)
    p_forget.add_argument("--id", default=None)
    p_forget.add_argument("--source", default=None)
    p_forget.add_argument("--all", action="store_true")
    p_forget.add_argument("--config", default=".fuko.toml", help="path to .fuko.toml")

    p_retrieve = sub.add_parser("retrieve", help="build extra_instructions markdown for PR-Agent")
    p_retrieve.add_argument("--repo", required=True)
    p_retrieve.add_argument("--out", default="extra.md")
    p_retrieve.add_argument(
        "--files-file", default=None, help="newline-separated paths file (default: stdin)"
    )
    p_retrieve.add_argument("--pr-body", default=None)
    p_retrieve.add_argument("--config", default=".fuko.toml", help="path to .fuko.toml")

    args = parser.parse_args()
    {
        "serve": _cmd_serve,
        "review": _cmd_review,
        "signals": _cmd_signals,
        "status": _cmd_status,
        "query": _cmd_query,
        "ingest-docs": _cmd_ingest_docs,
        "forget": _cmd_forget,
        "retrieve": _cmd_retrieve,
    }[args.cmd](args)


def _cmd_review(args) -> None:
    from . import runner

    result = runner.review(args.pr_url, args.config)
    if result.detail:
        label = "review backend failed" if result.returncode != 0 else "review backend warning"
        print(f"{label}: {result.detail}", file=sys.stderr)
    if result.returncode != 0:
        sys.exit(1)


def _cmd_signals(args) -> None:
    import json

    import httpx

    from . import runner
    from .fukoconfig import load_config
    from .normalizers import collect_signals
    from .presets import UnknownPresetError, get_preset

    cfg = load_config(args.config)
    pr = runner.parse_pr_url(args.pr_url)
    token = os.environ.get("GITHUB_TOKEN", "")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")

    try:
        preset = get_preset(cfg.review.model.provider)
        model = preset.litellm_prefix + cfg.review.model.name
    except UnknownPresetError:
        model = ""

    try:
        comments = runner.fetch_inline_comments(pr, token, api_url)
    except httpx.HTTPStatusError as e:
        _exit_on_auth_error(e, pr, token)

    signals = collect_signals(comments, model)
    print(json.dumps([s.model_dump() for s in signals], indent=2))


def _exit_on_auth_error(exc, pr, token: str) -> None:
    """On a 401/403/404 from a GitHub fetch, print a clear message and exit; else re-raise."""
    status = exc.response.status_code
    if status in (401, 403, 404):
        hint = "GITHUB_TOKEN is not set" if not token else "the token lacks access"
        print(
            f"fuko: cannot read {pr.repo}#{pr.number} (HTTP {status}; {hint}). "
            "Set GITHUB_TOKEN to a token with 'Pull requests: Read' on this repository "
            "(a private repo returns 404 when the request is unauthorized).",
            file=sys.stderr,
        )
        sys.exit(1)
    raise exc


def _cmd_status(args) -> None:
    import json

    import httpx

    from . import runner
    from .status import reviewer_states

    pr = runner.parse_pr_url(args.pr_url)
    token = os.environ.get("GITHUB_TOKEN", "")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    try:
        head = runner.fetch_pr_head(pr, token, api_url)
        issue_comments = runner.fetch_issue_comments(pr, token, api_url)
        reviews = runner.fetch_reviews(pr, token, api_url)
    except httpx.HTTPStatusError as e:
        _exit_on_auth_error(e, pr, token)

    try:
        check_runs = runner.fetch_check_runs(pr, head, token, api_url)
    except httpx.HTTPStatusError:
        check_runs = None

    print(json.dumps(reviewer_states(head, issue_comments, reviews, check_runs), indent=2))


def _store(config_path: str):
    """Return the knowledge store selected by the config at ``config_path``."""
    from .fukoconfig import load_config
    from .stores import get_store

    return get_store(load_config(config_path).knowledge)


def _cmd_serve(_args) -> None:
    import uvicorn

    from .config import settings

    uvicorn.run("sidecar.main:app", host=settings.host, port=settings.port, reload=False)


def _cmd_query(args) -> None:
    results = _store(args.config).query(args.repo, args.file, None, args.text, args.top_k)
    if not results:
        print("(no learnings matched)")
        return
    for r in results:
        print(f"[{r['score']:.3f}] ({r['source']}) {r['topic'] or ''}".rstrip())
        print(f"    {r['text'][:200].replace(chr(10), ' ')}")
        if r["source_url"]:
            print(f"    -> {r['source_url']}")


def _collect_files(patterns: list[str]) -> list[str]:
    collected: list[str] = []

    def add_path(p: str) -> None:
        path = Path(p)
        if path.is_dir():
            collected.extend(str(f) for f in path.rglob("*") if f.is_file())
        elif path.is_file():
            collected.append(p)

    for pat in patterns:
        matches = globmod.glob(pat, recursive=True)
        if matches:
            for m in matches:
                add_path(m)
            continue
        add_path(pat)
        if not (Path(pat).is_dir() or Path(pat).is_file()):
            print(f"warning: no matches for '{pat}', skipping", file=sys.stderr)

    seen: set[str] = set()
    out: list[str] = []
    for f in collected:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _cmd_ingest_docs(args) -> None:
    from . import models as M

    files = _collect_files(args.paths)
    if not files:
        print("no files found; nothing to ingest", file=sys.stderr)
        return

    items: list[M.IngestItem] = []
    for fp in files:
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            print(f"warning: could not read {fp}: {e}; skipping", file=sys.stderr)
            continue
        for chunk_text, heading in chunk_markdown(text):
            body = chunk_text.strip()
            if not body:
                continue
            items.append(
                M.IngestItem(
                    text=body,
                    source="docs",
                    source_url=args.source_url,
                    file_globs=list(args.glob),
                    topic=heading,
                )
            )

    if not items:
        print("no chunks produced", file=sys.stderr)
        return

    inserted, skipped = _store(args.config).ingest(args.repo, items)
    print(f"ingested {inserted} chunks (skipped {skipped}) from {len(files)} file(s)")


def _cmd_forget(args) -> None:
    if not (args.id or args.source or args.all):
        print("provide --id, --source, or --all", file=sys.stderr)
        sys.exit(2)
    deleted = _store(args.config).forget(args.repo, id=args.id, source=args.source, all=args.all)
    print(f"deleted {deleted}")


def _cmd_retrieve(args) -> None:
    if args.files_file:
        raw = Path(args.files_file).read_text().splitlines()
    else:
        raw = sys.stdin.read().splitlines()
    files = [line.strip() for line in raw if line.strip()]

    results = _store(args.config).query(args.repo, files, args.pr_body, None, None)
    md = format_extra_instructions(results)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"\n(wrote {len(results)} learnings to {args.out})", file=sys.stderr)


def format_extra_instructions(results: list[dict]) -> str:
    """Render retrieved learnings as a PR-Agent ``extra_instructions`` markdown block."""
    if not results:
        return ""
    lines = [
        "## Repository knowledge (from fuko-pr)",
        (
            "Apply the following repo-specific learnings where relevant to this PR. "
            "Cite the source link when acting on a learning."
        ),
        "",
    ]
    for r in results:
        cite = f" (source: {r['source_url']})" if r["source_url"] else f" (source: {r['source']})"
        globs = f" [applies to: {', '.join(r['file_globs'])}]" if r["file_globs"] else ""
        lines.append(f"- {r['text']}{cite}{globs}")
    return "\n".join(lines) + "\n"


def _split_paragraphs(body: str, max_len: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", body):
        if not cur or len(cur) + len(para) + 2 <= max_len:
            cur = (cur + "\n\n" + para) if cur else para
        else:
            out.append(cur)
            cur = para
        if len(cur) > max_len:
            out.append(cur[:max_len])
            cur = ""
    if cur:
        out.append(cur)
    return out or [body[:max_len]]


def chunk_markdown(text: str, max_len: int = 1500) -> list[tuple[str, str]]:
    """Split ``text`` into ``(chunk, heading)`` pairs, capping each chunk near ``max_len``."""
    chunks: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        for part in _split_paragraphs(body, max_len):
            chunks.append((part, heading))

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            heading = m.group(2).strip()
            buf = [line]
        else:
            buf.append(line)
    flush()
    return chunks or [(text.strip()[:max_len], "")]


if __name__ == "__main__":
    main()
