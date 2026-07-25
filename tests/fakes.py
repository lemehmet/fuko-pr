"""Shared test doubles."""

import io
import subprocess


class FakePopen:
    """Stands in for ``subprocess.Popen`` in ``invoke()`` tests.

    Scripted per-instance: exit code, merged output, and an optional hang that
    raises ``TimeoutExpired`` from ``wait()`` once (the second wait — after the
    docker kill — returns).
    """

    def __init__(self, cmd, *, rc=0, output="", hang=False, env=None):
        self.cmd = cmd
        self.env = env
        self.returncode = rc
        self._hang = int(hang)
        self.killed = False
        self.stdout = io.StringIO(output)

    def wait(self, timeout=None):
        if self._hang > 0:
            self._hang -= 1
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        return self.returncode

    def kill(self):
        self.killed = True


def popen_factory(recorder=None, behavior=None):
    """Build a ``Popen``-compatible callable for monkeypatching.

    ``behavior(tool)`` maps the docker command's trailing tool argument to
    ``dict(rc=..., output=..., hang=...)``; ``recorder`` (a list) collects
    ``(cmd, env)`` tuples.
    """

    def _factory(cmd, env=None, stdout=None, stderr=None, text=None, bufsize=None):
        if recorder is not None:
            recorder.append((cmd, env))
        kw = behavior(cmd[-1]) if behavior else {}
        return FakePopen(cmd, env=env, **kw)

    return _factory


class FakeStore:
    """In-memory stand-in for the knowledge Store protocol, for console tests.

    Records every call in ``seen`` and can be told to raise from any method, so
    a test can drive both the happy path and the degraded one.
    """

    def __init__(self, items=None):
        self.items = list(items or [])
        self.seen = {}
        self.raises = None
        self.read_raises = None

    def _check(self):
        if self.raises:
            raise self.raises

    def repos(self):
        if self.read_raises:
            raise self.read_raises
        summary = {}
        for item in self.items:
            entry = summary.setdefault(
                item["repo"], {"repo": item["repo"], "count": 0, "sources": {}}
            )
            entry["count"] += 1
            entry["sources"][item["source"]] = entry["sources"].get(item["source"], 0) + 1
        return [summary[k] for k in sorted(summary)]

    def list_learnings(self, **kw):
        if self.read_raises:
            raise self.read_raises
        self.seen["list"] = kw
        rows = [i for i in self.items if i["repo"] == kw.get("repo")]
        if kw.get("source"):
            rows = [i for i in rows if i["source"] == kw["source"]]
        if kw.get("q"):
            needle = kw["q"].lower()
            rows = [i for i in rows if needle in i["text"].lower()]
        offset, limit = kw.get("offset", 0), kw.get("limit", 25)
        return rows[offset : offset + limit], len(rows)

    def get_learning(self, repo, id):
        self.seen["get"] = (repo, id)
        return next((i for i in self.items if i["repo"] == repo and i["id"] == id), None)

    def update_learning(self, repo, id, **changes):
        self.seen["update"] = (repo, id, changes)
        self._check()
        current = self.get_learning(repo, id)
        if current is None:
            return None
        current.update(changes)
        return current

    def ingest(self, repo, items, *, max_new=None):
        self.seen["ingest"] = (repo, items)
        self._check()
        for item in items:
            self.items.append({"repo": repo, "id": f"new-{len(self.items)}", **item.model_dump()})
        return len(items), 0

    def forget(self, repo, *, id=None, source=None, all=False):
        self.seen["forget"] = (repo, id, source, all)
        self._check()
        before = len(self.items)
        if id:
            self.items = [i for i in self.items if not (i["repo"] == repo and i["id"] == id)]
        elif source:
            self.items = [
                i for i in self.items if not (i["repo"] == repo and i["source"] == source)
            ]
        elif all:
            self.items = [i for i in self.items if i["repo"] != repo]
        return before - len(self.items)

    def query(self, repo, files, pr_body=None, query_text=None, top_k=None):
        self.seen["query"] = (repo, files, query_text)
        if self.read_raises:
            raise self.read_raises
        return [
            {**i, "score": 0.9}
            for i in self.items
            if i["repo"] == repo and (not query_text or query_text.lower() in i["text"].lower())
        ]
