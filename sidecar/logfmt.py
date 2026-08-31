r"""One rule for putting foreign text on a log line, shared by every writer of one.

Downstream log gates are ``^``-anchored, so any value that reaches stderr with a
line break in it hands the text after the break column 0 of its own line and lets
it forge a gate line. The repo's answer is :func:`flatten_for_log`, and it lives
here -- importing nothing -- so that the reviewer backend and the ledger seam can
share the one definition instead of each keeping a copy: the hazard is precisely
that two "flatten" implementations disagree about what a line is, and a rule that
exists once cannot.
"""

from __future__ import annotations


def flatten_for_log(value: str) -> str:
    r"""Collapse to ONE physical line: log gates downstream are ``^``-anchored.

    Flattens using ``splitlines()`` -- the SAME rule the dump splits on -- rather
    than replacing ``\r``/``\n``. Python breaks lines on eight more characters
    than those two (``\x0b``, ``\x0c``, ``\x1c``-``\x1e``, ``\x85``,
    ``\u2028``, ``\u2029``), so a hand-rolled replace leaves a crafted payload
    looking flat here while still splitting downstream -- reopening the column-0
    forgery this exists to close (fuko-henry, #147). Defining "one line" by the
    splitter's own definition makes the two agree by construction.
    """
    return " ".join(value.strip().splitlines())
