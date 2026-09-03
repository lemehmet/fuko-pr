"""The knowledge-base console: browse, edit, add, delete, purge, upload, preview.

The knowledge base is the durable asset, and until now it was write-only in
practice -- the sweep fed it, review read it, and the only human window was
``fuko kb list`` over SSH. This page is that window.

Reads are open, matching the metrics page's stance on a LAN-only deployment.
Every write goes through :mod:`sidecar.web.security`: a session cookie minted
from ``FUKO_AUTH_TOKEN`` plus a session-bound CSRF token on each form.
"""

from __future__ import annotations

import sys

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse

from ..chunking import chunk_markdown
from ..models import SOURCES, DuplicateLearningError, IngestItem, UnknownSourceError
from ..stores import current_store
from . import components as c
from . import security
from .layout import document, page

PAGE = page("kb")

router = APIRouter()

PAGE_SIZE = 25

_BROWSE = PAGE.path

_EDIT = f"{PAGE.path}/edit"

_DELETE = f"{PAGE.path}/delete"

_TOOLS = f"{PAGE.path}/tools"

_PURGE = f"{PAGE.path}/purge"

_UPLOAD = f"{PAGE.path}/upload"

_PREVIEW = f"{PAGE.path}/preview"

_SOURCE_OPTIONS = [("", "any source"), *((s, s) for s in SOURCES)]

_TEXT_PREVIEW_CHARS = 160


def _shell(request: Request, title: str, body: str) -> str:
    """Wrap a rendered body in the shared document, with the session indicator in nav."""
    return document(title=title, body=body, active=PAGE.slug, extra_nav=security.nav_extra(request))


def _flash(msg: str | None, err: str | None) -> str:
    """Render the post-redirect banner carried in the query string."""
    if err:
        return c.notice(err, kind="danger")
    return c.notice(msg, kind="ok") if msg else ""


def _degraded(error: str) -> str:
    return c.notice(f"Knowledge store unreachable — {error}", kind="danger")


def _split_globs(raw: str | None) -> list[str]:
    """Parse a comma- or whitespace-separated glob list from a form field."""
    if not raw:
        return []
    return [part for part in (p.strip() for p in raw.replace(",", " ").split()) if part]


def _split_files(raw: str | None) -> list[str]:
    """Parse the preview's changed-file list (one per line, or comma separated)."""
    if not raw:
        return []
    return [part for part in (p.strip() for p in raw.replace(",", "\n").splitlines()) if part]


def _url(path: str, **params: object) -> str:
    """Build one of this page's URLs with its query string."""
    return f"{path}{c.query_string(params)}"


def _redirect(path: str, **params: object) -> RedirectResponse:
    """Post/redirect/get back to a page, carrying a flash message in the query.

    Stamped ``no-store`` here rather than at each of the seven writes: every
    caller is a route behind :func:`~sidecar.web.security.require`, and one rule
    applied in one place is what #266 asked for. A 303 is not heuristically
    cacheable in the first place -- this is the cheap end of making "anything
    behind ``require`` is ``no-store``" true of every response, not most of them.
    """
    return security.no_store(
        RedirectResponse(_url(path, **params), status_code=status.HTTP_303_SEE_OTHER)
    )


def render_picker(repos: list[dict], *, error: str = "") -> str:
    """Render the repository chooser (pure)."""
    rows = [
        "<tr>"
        + c.raw_cell(c.link(_url(_BROWSE, repo=entry["repo"]), entry["repo"]))
        + c.cell(entry["count"], numeric=True)
        + c.raw_cell(
            " ".join(c.badge(f"{name} {n}") for name, n in sorted(entry["sources"].items()))
        )
        + "</tr>"
        for entry in repos
    ]
    body = [
        "<h1>Knowledge base</h1>",
        _degraded(error) if error else "",
        c.section(
            "Repositories",
            c.table(
                [("repo", False), ("learnings", True), ("sources", False)],
                rows,
                "no repository has any learnings yet",
            ),
        ),
    ]
    return "".join(body)


def _learning_row(item: dict, *, repo: str, signed_in: bool) -> str:
    text = item["text"]
    head = text[:_TEXT_PREVIEW_CHARS].replace("\n", " ")
    body = (
        f"<details><summary>{c.esc(head)}{'…' if len(text) > _TEXT_PREVIEW_CHARS else ''}"
        f"</summary><pre>{c.esc(text)}</pre></details>"
    )
    actions = (
        c.link(_url(_EDIT, repo=repo, id=item["id"]), "edit")
        + " · "
        + c.link(_url(_DELETE, repo=repo, id=item["id"]), "delete")
        if signed_in
        else '<span class="muted">sign in</span>'
    )
    source = c.badge(item["source"])
    if item["source_url"]:
        source += " " + c.link(item["source_url"], "↗")
    return (
        "<tr>"
        + c.raw_cell(body)
        + c.raw_cell(source)
        + c.cell(item["topic"] or "—")
        + c.cell(", ".join(item["file_globs"]) or "—", css="muted")
        + c.cell((item["created_at"] or "")[:10], css="muted")
        + c.raw_cell(actions, css="nowrap")
        + "</tr>"
    )


def render_browse(
    *,
    repo: str,
    items: list[dict],
    total: int,
    source: str | None,
    q: str | None,
    include_expired: bool,
    offset: int,
    signed_in: bool,
    flash: str = "",
    error: str = "",
) -> str:
    """Render one page of a repository's learnings (pure)."""
    filters = {"repo": repo, "source": source, "q": q, "include_expired": include_expired}
    tools = (
        c.link(_url(_EDIT, repo=repo), "add a learning")
        + " · "
        + c.link(_url(_TOOLS, repo=repo), "upload docs / purge")
        if signed_in
        else ""
    )
    body = [
        f"<h1>{c.esc(repo)}</h1>",
        "<p>"
        + c.link(_BROWSE, "← all repositories")
        + " · "
        + c.link(_url(_PREVIEW, repo=repo), "retrieval preview")
        + ((" · " + tools) if tools else "")
        + "</p>",
        flash,
        _degraded(error) if error else "",
        '<form method="get" action="'
        + _BROWSE
        + '">'
        + c.hidden(repo=repo)
        + c.field("search", "q", q or "", placeholder="text or topic", size=28)
        + c.select("source", "source", _SOURCE_OPTIONS, source or "")
        + '<label class="row"><input type="checkbox" name="include_expired" value="1"'
        + (" checked" if include_expired else "")
        + "> include expired</label>"
        + "<button>filter</button></form>",
        c.table(
            [
                ("learning", False),
                ("source", False),
                ("topic", False),
                ("globs", False),
                ("added", False),
                ("", False),
            ],
            [_learning_row(item, repo=repo, signed_in=signed_in) for item in items],
            "nothing matches these filters",
        ),
        c.pager(_BROWSE, filters, offset=offset, limit=PAGE_SIZE, total=total),
    ]
    return "".join(body)


def render_edit(
    *, repo: str, learning: dict | None, session: str, error: str = "", draft: dict | None = None
) -> str:
    """Render the add/edit form (pure); ``learning`` is ``None`` for a new one."""
    values = {**(learning or {}), **(draft or {})}
    is_new = learning is None
    body = [
        f"<h1>{'Add a learning' if is_new else 'Edit learning'}</h1>",
        "<p>" + c.link(_url(_BROWSE, repo=repo), "← " + repo) + "</p>",
        c.notice(error, kind="danger") if error else "",
        '<form method="post" class="stacked" action="' + _EDIT + '">',
        security.csrf_field(session),
        c.hidden(repo=repo, id=values.get("id")),
        c.textarea("text", "text", values.get("text", ""), required=True),
        "<div class=fields>",
        c.select(
            "source",
            "source",
            [(s, s) for s in SOURCES],
            values.get("source") or ("remember" if is_new else ""),
        ),
        c.field("topic", "topic", values.get("topic") or "", size=24),
        c.field("source url", "source_url", values.get("source_url") or "", size=32),
        c.field(
            "file globs",
            "file_globs",
            ", ".join(values.get("file_globs") or []),
            size=32,
            placeholder="src/**/*.py, docs/*.md",
        ),
        c.field(
            "expires at",
            "expires_at",
            values.get("expires_at") or "",
            size=24,
            placeholder="2027-01-01T00:00:00Z",
        ),
        "</div>",
        "<button>save</button></form>",
    ]
    if not is_new:
        body.append(
            f'<p class="muted">id {c.esc(values.get("id"))}</p>',
        )
    return "".join(body)


def render_delete(*, repo: str, learning: dict, session: str) -> str:
    """Render the delete confirmation (pure). Deletion has no undo, so it gets a step."""
    return "".join(
        [
            "<h1>Delete this learning?</h1>",
            c.notice("This cannot be undone.", kind="warn"),
            f"<pre>{c.esc(learning['text'])}</pre>",
            f'<p class="muted">{c.badge(learning["source"])} '
            f"{c.esc(learning.get('topic') or '')}</p>",
            '<form method="post" action="' + _DELETE + '">',
            security.csrf_field(session),
            c.hidden(repo=repo, id=learning["id"]),
            "<button>delete</button></form>",
            "<p>" + c.link(_url(_BROWSE, repo=repo), "cancel") + "</p>",
        ]
    )


def render_tools(*, repo: str, session: str, flash: str = "", error: str = "") -> str:
    """Render the doc-upload and purge forms (pure)."""
    return "".join(
        [
            f"<h1>{c.esc(repo)} · tools</h1>",
            "<p>" + c.link(_url(_BROWSE, repo=repo), "← back to browsing") + "</p>",
            flash,
            c.notice(error, kind="danger") if error else "",
            c.section(
                "Upload documents",
                '<p class="muted">Markdown or text; each file is split on headings into '
                "learnings stored under the <code>docs</code> source — the same chunking "
                "<code>fuko ingest-docs</code> uses.</p>"
                '<form method="post" class="stacked" action="'
                + _UPLOAD
                + '" enctype="multipart/form-data">'
                + security.csrf_field(session)
                + c.hidden(repo=repo)
                + "<div class=fields>"
                + c.field("files", "files", type="file", multiple=True, accept=".md,.txt,.markdown")
                + c.field("source url", "source_url", "", size=32)
                + c.field("file globs", "file_globs", "", size=28, placeholder="src/**/*.py")
                + "</div><button>upload</button></form>",
            ),
            c.section(
                "Purge",
                c.notice(
                    "Deletes in bulk and cannot be undone. Type the repository name to confirm.",
                    kind="danger",
                )
                + '<form method="post" action="'
                + _PURGE
                + '">'
                + security.csrf_field(session)
                + c.hidden(repo=repo)
                + c.select("what", "source", [("", "everything"), *((s, s) for s in SOURCES)], "")
                + c.field("type the repo name", "confirm", "", size=28, placeholder=repo)
                + "<button>purge</button></form>",
            ),
        ]
    )


def render_preview(
    *, repo: str, query_text: str, files: list[str], results: list[dict], error: str = ""
) -> str:
    """Render the retrieval preview (pure): what a review would actually be shown."""
    rows = [
        "<tr>"
        + c.cell(f"{r['score']:.3f}", numeric=True)
        + c.raw_cell(
            f"<details><summary>{c.esc(r['text'][:_TEXT_PREVIEW_CHARS])}</summary>"
            f"<pre>{c.esc(r['text'])}</pre></details>"
        )
        + c.raw_cell(c.badge(r["source"]))
        + c.cell(", ".join(r["file_globs"]) or "—", css="muted")
        + "</tr>"
        for r in results
    ]
    return "".join(
        [
            f"<h1>{c.esc(repo)} · retrieval preview</h1>",
            "<p>" + c.link(_url(_BROWSE, repo=repo), "← back to browsing") + "</p>",
            '<p class="muted">Runs the same retrieval a review does: semantic match plus '
            "file-glob scoping. This is the check for whether a learning will actually "
            "reach the reviewer.</p>",
            c.notice(error, kind="danger") if error else "",
            '<form method="get" class="stacked" action="'
            + _PREVIEW
            + '">'
            + c.hidden(repo=repo)
            + "<div class=fields>"
            + c.field("query text", "text", query_text, size=40)
            + c.field(
                "changed files",
                "files",
                ", ".join(files),
                size=40,
                placeholder="src/auth/login.py, README.md",
            )
            + "</div><button>preview</button></form>",
            c.table(
                [("score", True), ("learning", False), ("source", False), ("globs", False)],
                rows,
                "nothing retrieved for this query",
            ),
        ]
    )


@router.get(_BROWSE, response_class=HTMLResponse, include_in_schema=False)
def browse(
    request: Request,
    response: Response,
    repo: str | None = None,
    source: str | None = None,
    q: str | None = None,
    include_expired: bool = False,
    offset: int = 0,
    msg: str | None = None,
    err: str | None = None,
) -> str:
    """Serve the repository picker, or one repository's learnings."""
    security.vary_by_cookie(response)
    store = current_store()
    if not repo:
        repos, error = _read(store.repos, [])
        return _shell(request, "fuko knowledge base", render_picker(repos, error=error))

    offset = max(0, offset)
    (items, total), error = _read(
        lambda: store.list_learnings(
            repo=repo,
            source=source or None,
            limit=PAGE_SIZE,
            offset=offset,
            q=q or None,
            include_expired=include_expired,
        ),
        ([], 0),
    )
    return _shell(
        request,
        f"fuko knowledge base · {repo}",
        render_browse(
            repo=repo,
            items=items,
            total=total,
            source=source,
            q=q,
            include_expired=include_expired,
            offset=offset,
            signed_in=security.session_of(request) is not None,
            flash=_flash(msg, err),
            error=error,
        ),
    )


@router.get(_EDIT, response_class=HTMLResponse, include_in_schema=False)
def edit_form(request: Request, response: Response, repo: str, id: str | None = None) -> str:
    """Serve the add form (no ``id``) or the edit form for one learning."""
    session = security.require(request)
    security.no_store(response)
    learning = None
    if id:
        learning = current_store().get_learning(repo, id)
        if learning is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such learning in that repo")
    return _shell(
        request,
        "fuko knowledge base · edit",
        render_edit(repo=repo, learning=learning, session=session),
    )


@router.post(_EDIT, include_in_schema=False)
def edit_submit(
    request: Request,
    repo: str = Form(...),
    text: str = Form(...),
    source: str = Form(default="remember"),
    topic: str = Form(default=""),
    source_url: str = Form(default=""),
    file_globs: str = Form(default=""),
    expires_at: str = Form(default=""),
    id: str = Form(default=""),
    csrf: str = Form(default=""),
) -> Response:
    """Create or update one learning, then redirect back to the listing.

    A rejected write re-renders the form with the submitted values intact --
    losing a long edit to a collision would be its own bug.
    """
    session = security.require(request)
    security.check_csrf(session, csrf)
    store = current_store()
    fields = {
        "text": text,
        "source": source,
        "topic": topic or None,
        "source_url": source_url or None,
        "file_globs": _split_globs(file_globs),
        "expires_at": expires_at or None,
    }
    try:
        if id:
            updated = store.update_learning(repo, id, **fields)
            if updated is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no such learning in that repo")
            return _redirect(_BROWSE, repo=repo, msg="Saved.")
        inserted, _skipped = store.ingest(repo, [IngestItem(**fields)])
        note = "Added." if inserted else "Already stored — nothing added."
        return _redirect(_BROWSE, repo=repo, msg=note)
    except (DuplicateLearningError, UnknownSourceError) as e:
        draft = {**fields, "id": id or None}
        # Stamped like the redirects around it: this body carries the viewer's
        # CSRF token and their unsaved draft, so it is the one write response
        # that would actually hurt in a shared cache.
        return security.no_store(
            HTMLResponse(
                _shell(
                    request,
                    "fuko knowledge base · edit",
                    render_edit(
                        repo=repo,
                        learning={"id": id} if id else None,
                        session=session,
                        error=str(e),
                        draft=draft,
                    ),
                ),
                status_code=status.HTTP_409_CONFLICT
                if isinstance(e, DuplicateLearningError)
                else status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        )


@router.get(_DELETE, response_class=HTMLResponse, include_in_schema=False)
def delete_form(request: Request, response: Response, repo: str, id: str) -> str:
    """Serve the delete confirmation for one learning."""
    session = security.require(request)
    security.no_store(response)
    learning = current_store().get_learning(repo, id)
    if learning is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such learning in that repo")
    return _shell(
        request,
        "fuko knowledge base · delete",
        render_delete(repo=repo, learning=learning, session=session),
    )


@router.post(_DELETE, include_in_schema=False)
def delete_submit(
    request: Request,
    repo: str = Form(...),
    id: str = Form(...),
    csrf: str = Form(default=""),
) -> RedirectResponse:
    """Delete one learning and redirect back to the listing."""
    session = security.require(request)
    security.check_csrf(session, csrf)
    deleted = current_store().forget(repo, id=id)
    note = "Deleted." if deleted else "Nothing deleted — it was already gone."
    return _redirect(_BROWSE, repo=repo, msg=note)


@router.get(_TOOLS, response_class=HTMLResponse, include_in_schema=False)
def tools_form(
    request: Request,
    response: Response,
    repo: str,
    msg: str | None = None,
    err: str | None = None,
) -> str:
    """Serve the doc-upload and purge forms for one repository."""
    session = security.require(request)
    security.no_store(response)
    return _shell(
        request,
        f"fuko knowledge base · {repo} tools",
        render_tools(repo=repo, session=session, flash=_flash(msg, err)),
    )


@router.post(_UPLOAD, include_in_schema=False)
async def upload_submit(
    request: Request,
    repo: str = Form(...),
    source_url: str = Form(default=""),
    file_globs: str = Form(default=""),
    csrf: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
) -> RedirectResponse:
    """Chunk uploaded markdown into ``docs`` learnings and ingest them.

    Uses the same :func:`~sidecar.chunking.chunk_markdown` split as
    ``fuko ingest-docs``, so a file ingested here and the same file ingested
    from the CLI produce identical learnings and dedup against each other.
    """
    session = security.require(request)
    security.check_csrf(session, csrf)

    globs = _split_globs(file_globs)
    items: list[IngestItem] = []
    for upload in files:
        raw = await upload.read()
        if not raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        for chunk, heading in chunk_markdown(text):
            body = chunk.strip()
            if body:
                items.append(
                    IngestItem(
                        text=body,
                        source="docs",
                        source_url=source_url or None,
                        file_globs=globs,
                        topic=heading or upload.filename,
                    )
                )
    if not items:
        return _redirect(_TOOLS, repo=repo, err="No readable content in that upload.")
    inserted, skipped = current_store().ingest(repo, items)
    return _redirect(
        _TOOLS,
        repo=repo,
        msg=f"Ingested {inserted} chunk(s) from {len(files)} file(s); {skipped} already stored.",
    )


@router.post(_PURGE, include_in_schema=False)
def purge_submit(
    request: Request,
    repo: str = Form(...),
    source: str = Form(default=""),
    confirm: str = Form(default=""),
    csrf: str = Form(default=""),
) -> RedirectResponse:
    """Delete a repository's learnings in bulk, gated on typing the repository name."""
    session = security.require(request)
    security.check_csrf(session, csrf)
    if confirm.strip() != repo:
        return _redirect(_TOOLS, repo=repo, err="Type the repository name exactly to confirm.")
    store = current_store()
    deleted = store.forget(repo, source=source) if source else store.forget(repo, all=True)
    what = f"source={source}" if source else "the whole repository"
    return _redirect(_TOOLS, repo=repo, msg=f"Purged {deleted} learning(s) from {what}.")


@router.get(_PREVIEW, response_class=HTMLResponse, include_in_schema=False)
def preview(
    request: Request, response: Response, repo: str, text: str = "", files: str = ""
) -> str:
    """Show what retrieval would return for a hypothetical PR (read-only, open)."""
    security.vary_by_cookie(response)
    paths = _split_files(files)
    results: list[dict] = []
    error = ""
    if text or paths:
        results, error = _read(
            lambda: current_store().query(repo, paths, None, text or None, None), []
        )
    return _shell(
        request,
        f"fuko knowledge base · {repo} preview",
        render_preview(repo=repo, query_text=text, files=paths, results=results, error=error),
    )


def _read(fetch, fallback):
    """Run a store read, degrading to ``fallback`` plus a message instead of a 500.

    Same contract the metrics page follows: the store being unreachable must not
    cost the operator the page, only its contents.

    The visitor gets a GENERIC message; the exception text goes to stderr only.
    Every caller of this helper is an UNAUTHENTICATED route (browse and preview
    are the two pages without ``security.require``), so whatever a store
    exception happens to contain would otherwise be rendered to anyone who can
    reach the console. Store errors are exactly the kind that quote what they
    failed to connect to -- DSNs, hosts, and, for a URL carrying inline
    credentials, the credentials themselves. The operator loses nothing: the
    detail is already logged next to a timestamp on the box that has it.
    """
    try:
        return fetch(), ""
    except Exception as e:
        print(f"fuko: kb console degraded (store unreachable?): {e}", file=sys.stderr)
        return fallback, "see the sidecar log for details"
