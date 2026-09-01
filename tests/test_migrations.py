"""Guards on the migration files themselves.

``sidecar.db.get_pool`` keeps no applied-migrations table: it re-runs every
``migrations/*.sql`` in filename order on each pool creation. That makes a
constraint's *history* live code, not archaeology.
"""

import re
from pathlib import Path

from sidecar.models import SOURCES

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def test_every_created_object_is_re_runnable():
    """Replay-on-every-startup means a bare ``CREATE`` bricks the second boot.

    ``get_pool`` keeps no applied-migrations table, so every file runs again on
    each pool creation. A ``CREATE TABLE``/``CREATE INDEX`` without
    ``IF NOT EXISTS`` therefore succeeds exactly once and aborts the whole
    migration pass from then on -- including the files after it, which is how a
    single non-idempotent statement takes the sidecar down permanently rather
    than failing its own migration. Asserted structurally because the failure
    only shows up on a *second* startup against a store the first one created,
    which no unit test naturally reproduces.
    """
    offenders: list[str] = []
    for path in sorted(MIGRATIONS.glob("*.sql")):
        sql = re.sub(r"--[^\n]*", "", path.read_text())
        for match in re.finditer(
            r"CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+(?!IF\s+NOT\s+EXISTS)",
            sql,
            re.IGNORECASE,
        ):
            offenders.append(f"{path.name}: {sql[match.start() : match.start() + 60].strip()}")
    assert not offenders, (
        "migrations replay on every pool creation, so each CREATE must be "
        f"IF NOT EXISTS: {offenders}"
    )


def _source_checks() -> list[tuple[str, set[str]]]:
    found: list[tuple[str, set[str]]] = []
    for path in sorted(MIGRATIONS.glob("*.sql")):
        sql = re.sub(r"--[^\n]*", "", path.read_text())
        for match in re.finditer(r"source\s+IN\s*\(([^)]*)\)", sql, re.IGNORECASE):
            found.append((path.name, set(re.findall(r"'([^']+)'", match.group(1)))))
    return found


def test_every_source_check_lists_the_current_vocabulary():
    """A stale CHECK in an OLD migration bricks startup, not just that migration.

    Migrations replay on every pool creation, in filename order, and
    ``ADD CONSTRAINT`` validates the rows already in the table. So an earlier
    file that re-adds ``learnings_source_check`` without the newest source fails
    the moment one row uses it -- aborting the whole migration pass before the
    file that would have widened the constraint ever runs, and leaving the
    sidecar unable to open its pool on every restart from then on. Adding a
    source therefore means widening every historical re-add of the constraint,
    not only appending a new migration.
    """
    checks = _source_checks()
    assert checks, "no learnings source CHECK found in migrations/"
    for name, listed in checks:
        assert listed == set(SOURCES), (
            f"{name} pins the source vocabulary to {sorted(listed)}, but the "
            f"current vocabulary is {sorted(SOURCES)}. Widen it: migrations "
            f"replay in filename order on every startup."
        )
