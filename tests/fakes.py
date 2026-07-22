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
