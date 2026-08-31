"""Command-line interface for the fuko-pr sidecar."""

import argparse
import glob as globmod
import os
import sys
from pathlib import Path

from . import digest
from .chunking import chunk_markdown
from .digest import DIGEST_SOURCE


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
        "status",
        help="emit per-reviewer review state on a PR's HEAD as JSON "
        "(external bots plus fuko's own instances)",
    )
    p_status.add_argument("--pr-url", required=True, help="full pull request URL")
    p_status.add_argument(
        "--config",
        default=".fuko.toml",
        help="path to .fuko.toml; its configured seat labels let a receipt whose "
        "seat was renamed or removed report 'superseded' instead of a stuck 'pending'",
    )

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

    p_digest = sub.add_parser(
        "digest",
        help="index large files in a checkout as 'digest' learnings",
        description="Run this from the root of a checkout. Each index is scoped to the "
        "path it was collected under, and retrieval matches those paths against the "
        "files a pull request changes, so an index collected under an absolute path "
        "can never be retrieved.",
    )
    p_digest.add_argument(
        "paths", nargs="*", default=["."], help="files, globs, or directories (default: .)"
    )
    p_digest.add_argument("--repo", required=True)
    p_digest.add_argument(
        "--min-bytes",
        type=int,
        default=digest.MIN_BYTES,
        help="only index files at least this large (default: %(default)s)",
    )
    p_digest.add_argument(
        "--max-chars",
        type=int,
        default=digest.MAX_CHARS,
        help="cap on one rendered index (default: %(default)s)",
    )
    p_digest.add_argument(
        "--dry-run", action="store_true", help="print the indexes instead of storing them"
    )
    p_digest.add_argument("--config", default=".fuko.toml", help="path to .fuko.toml")

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

    from . import kbcli

    kbcli.add_parser(sub)

    args = parser.parse_args()
    {
        "serve": _cmd_serve,
        "review": _cmd_review,
        "signals": _cmd_signals,
        "status": _cmd_status,
        "query": _cmd_query,
        "ingest-docs": _cmd_ingest_docs,
        "digest": _cmd_digest,
        "forget": _cmd_forget,
        "retrieve": _cmd_retrieve,
        "kb": kbcli.dispatch,
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
    from .normalizers import (
        collect_issue_comment_signals,
        collect_review_signals,
        collect_signals,
    )
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
        issue_comments = runner.fetch_issue_comments(pr, token, api_url)
        # Review BODIES are a third channel, not a duplicate of the first two:
        # Copilot's suppressed-comments block lives only here (#109).
        reviews = runner.fetch_reviews(pr, token, api_url)
    except httpx.HTTPStatusError as e:
        _exit_on_auth_error(e, pr, token)

    signals = (
        collect_signals(comments, model)
        + collect_issue_comment_signals(issue_comments, model)
        + collect_review_signals(reviews, model)
    )
    _warn_on_dropped_comments(comments, model)
    print(json.dumps([s.model_dump() for s in signals], indent=2))


def _is_bot_author(comment: dict) -> bool:
    """Return whether ``comment`` was posted by a bot rather than a person.

    ``user.type`` is authoritative when GitHub sends it; the login suffix is the
    fallback for trimmed payloads and fixtures.
    """
    user = comment.get("user") or {}
    return (user.get("type") or "").lower() == "bot" or str(user.get("login", "")).endswith("[bot]")


def _warn_on_dropped_comments(comments: list[dict], model: str) -> None:
    """Warn on stderr when inline comments produced no signal, naming their authors.

    Consumers triage from this output, so a silent shortfall reads as "no findings"
    rather than "findings not recognized" -- the failure mode this warning exists to
    make impossible.

    Scoped to bot authors on purpose. A human's inline note is legitimately not a
    finding, and warning about those every run would train the reader to skip the
    line -- which is the same silence, just louder. A *bot* comment that no
    recognizer can read is the thing that must never pass unnoticed. Reports rather
    than fails: not every bot comment is a finding either.

    ``unrecognized_comments`` already excludes comments a recognizer claimed and
    then deliberately skipped (CodeRabbit chat, rate-limit notices), so those never
    reach this warning -- they were understood, not missed.
    """
    from collections import Counter

    from .normalizers import unrecognized_comments

    dropped = [c for c in unrecognized_comments(comments, model) if _is_bot_author(c)]
    if not dropped:
        return
    by_author = Counter((c.get("user") or {}).get("login", "?") for c in dropped)
    breakdown = ", ".join(f"{login}: {n}" for login, n in sorted(by_author.items()))
    print(
        f"fuko: warning: {len(dropped)} bot inline comment(s) matched no reviewer "
        f"format and carried no fuko-signal marker, so they are NOT in the output "
        f"({breakdown}). Read the raw comment list before concluding they are not "
        f"findings.",
        file=sys.stderr,
    )


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
    except httpx.HTTPError:
        check_runs = None

    # Cross-reference receipt labels against the currently-configured seats so a
    # renamed/removed seat reports `superseded` rather than a stuck `pending`
    # (#116). On ANY failure to load config, pass None: `fuko_states` then keeps
    # today's behavior instead of dropping a genuinely-pending row -- erring
    # toward the gating state is the safe direction when the config is unreadable.
    configured_labels = _configured_seat_labels(args.config)

    print(
        json.dumps(
            reviewer_states(
                head, issue_comments, reviews, check_runs, configured_labels=configured_labels
            ),
            indent=2,
        )
    )


def _configured_seat_labels(config_path: str) -> list[str] | None:
    """Return the ``provider/name`` label of every configured review entry, or None.

    None is returned when the config file is absent, cannot be parsed, or
    resolves to no entries, so :func:`sidecar.status.fuko_states` skips its
    cross-reference and keeps every receipt row -- the fail-safe direction, since
    dropping a real pending row on a config-read failure could let a merge
    proceed past an unreviewed seat. A label matches a receipt when it equals the
    receipt's ``provider/name`` (see :func:`sidecar.runner`), so every role is
    included: membership only needs to prove the seat still exists, and including
    backups avoids ever misreading a still-configured seat as removed.

    A MISSING file must map to None, not to the built-in defaults: ``load_config``
    returns a default ``FukoConfig()`` when the path does not exist, whose single
    fallback entry would then wrongly supersede every real seat. So absence is
    checked explicitly here rather than trusting the loader's defaults.
    """
    from pathlib import Path

    from .fukoconfig import load_config
    from .pool import resolve_models

    if not Path(config_path).exists():
        print(
            f"fuko: warning: {config_path} not found for seat cross-reference; "
            "receipts for renamed/removed seats will still report 'pending'.",
            file=sys.stderr,
        )
        return None
    # The label extraction is inside the try on purpose: if resolve_models returns
    # an unexpected value (e.g. None) or a model lacks provider/name, building the
    # labels must fail safe to None rather than crash `fuko status`.
    #
    # An empty `provider`/`name` is rejected rather than emitted: pydantic types
    # them as `str` but does not forbid `""`, so a malformed entry (`provider = ""`)
    # would otherwise yield a junk label like `"/name"` or `"/"`. A junk label that
    # matches no real receipt would then supersede every genuine seat (a receipt is
    # superseded when its label is NOT in the configured set) -- the exact
    # merge-past-unreviewed-seat direction this function fails safe against. So a
    # blank component raises, and the `except` below turns it into a safe `None`.
    try:
        models = resolve_models(load_config(config_path).review)
        labels = []
        for m in models:
            provider = (m.provider or "").strip()
            name = (m.name or "").strip()
            if not provider or not name:
                raise ValueError(f"configured model has a blank provider/name: {m!r}")
            labels.append(f"{provider}/{name}")
    except Exception as e:  # noqa: BLE001 -- any load/parse failure must fail safe to None
        print(
            f"fuko: warning: could not load {config_path} for seat cross-reference "
            f"({e}); receipts for renamed/removed seats will still report 'pending'.",
            file=sys.stderr,
        )
        return None
    return labels or None


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


_DIGEST_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "target",
        "dist",
        "build",
        "vendor",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".next",
    }
)


def _digest_candidates(patterns: list[str], min_bytes: int, root: Path) -> list[str]:
    """Return the readable files under ``patterns`` that are at least ``min_bytes``.

    Generated and vendored trees are skipped: an index of a checked-in bundle
    would displace real knowledge from a review's ``top_k`` budget while
    describing code nobody reviews. The skip set names directories *inside* a
    repository, so it is tested against the ``root``-relative path -- matching it
    against a candidate's absolute ancestors would drop every file of a checkout
    that merely happens to live under ``/srv/build``. A candidate outside
    ``root`` is kept here so :func:`_cmd_digest` can report it as unreachable
    rather than have it vanish at the wrong check.
    """
    out: list[str] = []
    for candidate in _collect_files(patterns):
        rel = _index_path(candidate, root)
        parts = set(Path(rel).parts) if rel is not None else set()
        if parts & _DIGEST_SKIP_DIRS:
            continue
        try:
            if Path(candidate).stat().st_size < min_bytes:
                continue
        except OSError as e:
            print(f"warning: could not stat {candidate}: {e}; skipping", file=sys.stderr)
            continue
        out.append(candidate)
    return out


def _index_path(candidate: str, root: Path) -> str | None:
    """Return ``candidate`` as a ``root``-relative POSIX path, or ``None`` if it escapes.

    An index is retrieved by matching its glob against the paths GitHub reports
    for a pull request, and those are repository-relative POSIX paths. Any other
    spelling -- an absolute path, or a ``../`` one reached from the wrong
    directory -- produces a row that is stored, embedded, and can never match
    anything, so the path is canonicalised here and a file outside the checkout
    is skipped rather than indexed unreachably.
    """
    try:
        rel = Path(candidate).resolve().relative_to(root)
    except (OSError, ValueError):
        return None
    return rel.as_posix()


def _forget_superseded(store, repo: str, current: dict[str, str]) -> int:
    """Delete every stored index of ``current``'s paths that is not the current one.

    ``current`` maps an indexed path to the exact text just stored for it, and
    the predicate is inequality against that text -- not against the topic. The
    topic carries only ``<path>@<blob hash>``, so two renderings of the *same*
    blob (a different ``--max-chars``, or any later change to what ``render``
    emits) share a topic while ingest, which dedups on text, inserts the second
    one beside the first. Keying supersession on the topic left both rows in
    place, reported ``0 superseded``, and no later run could ever collapse them.
    Comparing text makes the invariant the one that was always intended: one
    index row per indexed path.

    Called *after* the new digests are inserted, not before: the current text is
    then already stored, so the predicate cannot delete it. For a single run that
    means an indexed file is never left without an index -- the cost of the other
    order, a failed insert leaving the file unindexed, is worse than this order's
    cost of a few seconds with two indexes of one file. It is *not* a guarantee
    across concurrent runs: two overlapping runs whose renderings of one file
    differ each store their own row and each collects the other's as stale from a
    snapshot taken before any delete, so the pair can remove both. Recovering is
    a re-run, and closing it properly needs a delete that carries the predicate
    rather than an id (see #199, which changes that call shape anyway).
    """
    stale: list[str] = []
    offset = 0
    while True:
        rows, total = store.list_learnings(
            repo=repo, source=DIGEST_SOURCE, limit=200, offset=offset, include_expired=True
        )
        if not rows:
            break
        for row in rows:
            path = digest.topic_path(row.get("topic"))
            if path in current and row.get("text") != current[path]:
                stale.append(row["id"])
        offset += len(rows)
        if offset >= total:
            break
    for learning_id in stale:
        store.forget(repo, id=learning_id)
    return len(stale)


def _cmd_digest(args) -> None:
    items = []
    paths: list[str] = []
    featureless: list[str] = []
    root = Path.cwd().resolve()
    if not (root / ".git").exists():
        # The complementary direction of the outside-the-checkout warning below,
        # and the silent one: a file *under* a subdirectory is happily keyed
        # relative to that subdirectory, so it stores a path GitHub never reports
        # for a pull request and is just as unreachable -- with nothing to see.
        print(
            f"fuko: warning: no .git in {root}, which may not be the checkout root. "
            "Indexes are keyed relative to the working directory and matched against "
            "the repository-relative paths a pull request reports, so one collected "
            "from a subdirectory can never be retrieved.",
            file=sys.stderr,
        )
    candidates = _digest_candidates(args.paths, args.min_bytes, root)
    outside = [p for p in candidates if _index_path(p, root) is None]
    if outside:
        print(
            f"fuko: warning: skipped {len(outside)} file(s) outside the checkout "
            f"(e.g. {outside[0]}); an index is matched against the repository-relative "
            "paths a pull request reports, so those could never be retrieved. Run "
            "`fuko digest` from the root of the checkout.",
            file=sys.stderr,
        )
    for fp in candidates:
        rel = _index_path(fp, root)
        if rel is None:
            continue
        try:
            # Decoded from the raw bytes, not `read_text`: text mode normalises
            # newlines, which would make the rendered blob hash and size describe
            # something no `sha256sum` of the file on disk can reproduce -- and
            # the index tells its reader to check exactly that.
            text = Path(fp).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"warning: could not read {fp}: {e}; skipping", file=sys.stderr)
            continue
        try:
            item = digest.build_item(rel, text, args.max_chars)
        except ValueError as e:
            print(f"warning: cannot index {rel}: {e}; skipping", file=sys.stderr)
            continue
        if item is None:
            featureless.append(rel)
            continue
        items.append(item)
        paths.append(rel)

    if featureless:
        print(
            f"fuko: skipped {len(featureless)} file(s) with no recognised declarations "
            f"(e.g. {featureless[0]}); an index of a lockfile or a data dump has nothing "
            "to navigate to, and storing one would spend an embedding and a retrieval "
            "slot on it.",
            file=sys.stderr,
        )

    if not items:
        print(
            f"no readable files at or above {args.min_bytes} bytes; nothing to index",
            file=sys.stderr,
        )
        return

    if args.dry_run:
        for item in items:
            print(item.text)
            print()
        print(f"(dry run: {len(items)} index(es), nothing stored)", file=sys.stderr)
        return

    store = _store(args.config)
    inserted, skipped = store.ingest(args.repo, items)
    # Keyed on the literal path, not on ``file_globs`` -- the stored glob is
    # escaped for fnmatch, while ``topic_path`` returns the path verbatim.
    superseded = _forget_superseded(
        store, args.repo, {path: item.text for path, item in zip(paths, items, strict=True)}
    )
    print(
        f"indexed {len(items)} file(s): {inserted} new, {skipped} unchanged, "
        f"{superseded} superseded"
    )


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
    """Render retrieved learnings as a PR-Agent ``extra_instructions`` markdown block.

    File digests (#158) get their own section rather than a bullet in the
    learnings list. Two reasons, both about not misrepresenting them: a digest
    is a multi-line table that would be unreadable as a list item, and the
    learnings section presents its entries as conventions to apply, which a
    structural index is not.
    """
    digests = [r for r in results if r["source"] == DIGEST_SOURCE]
    learnings = [r for r in results if r["source"] != DIGEST_SOURCE]
    if not results:
        return ""
    lines: list[str] = []
    if learnings:
        lines += [
            "## Repository knowledge (from fuko-pr)",
            (
                "Apply the following repo-specific learnings where relevant to this PR. "
                "Cite the source link when acting on a learning."
            ),
            "",
        ]
        for r in learnings:
            cite = (
                f" (source: {r['source_url']})" if r["source_url"] else f" (source: {r['source']})"
            )
            globs = f" [applies to: {', '.join(r['file_globs'])}]" if r["file_globs"] else ""
            lines.append(f"- {r['text']}{cite}{globs}")
    if digests:
        if lines:
            lines.append("")
        lines += [
            "## File structure index (from fuko-pr)",
            (
                "Mechanically extracted maps of large files this PR touches: what "
                "each file declares and at which lines. Use them to read the "
                "specific ranges you need instead of whole files. They are "
                "navigation aids, NOT review conclusions -- they say nothing "
                "about whether any of this code is correct, an index may lag the "
                "checkout (each names the blob hash it was built from), and a "
                "declaration missing from an index is not a declaration missing "
                "from the file."
            ),
            "",
        ]
        for r in digests:
            lines.append(r["text"])
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


if __name__ == "__main__":
    main()
