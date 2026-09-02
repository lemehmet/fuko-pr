"""Per-branch review-run metrics, shared across runs via Postgres.

One row per model branch per review round (see ``migrations/004``): which
provider+model ran under which slot, how long it took, how many failover
attempts it needed, how it ended, how many findings it produced -- and, for a
backend whose driver reports it, what it spent (``migrations/008``). This is
the evidence base for promoting experimental models to known-good and for
comparing slots by cost.

The token and cost figures are nullable end to end and stay that way: only the
agentic backend has a usage feed, so a pr-agent row -- and every row written
before #152 -- reports ``NULL``, not ``0``. The aggregates below preserve that
distinction (a group with nothing measured sums to ``None``), because a zero
here would read as "these reviews were free".

What a run spent its turns ON is the sibling question, and it lives in
``review_transcripts`` (``migrations/013``, #239) rather than in more columns
here: per-tool call counts, tool-result bytes, and how many files a run read
more than once, keyed by the transcript's own key. ``review_runs`` gains only
the nullable reference to it, written after that row lands.

Best-effort by design, mirroring :mod:`sidecar.circuit_breaker`: with no
Postgres configured (``FUKO_DATABASE_URL`` unset) these functions degrade to
no-ops -- metrics must never block or fail a review.
"""

from __future__ import annotations

import json
import math
import sys

from .config import settings


def _enabled() -> bool:
    """Run-metrics persistence requires the shared Postgres store."""
    return bool(settings.database_url)


#: Largest value ``review_runs.cost_usd`` can hold: the column is
#: ``NUMERIC(10, 4)`` (``migrations/004``), so ten digits with four after the
#: point.
_MAX_COST_USD = 999_999.9999


def _storable_cost(value: float | None) -> float | None:
    """Drop a ``cost_usd`` the column cannot hold, so the ROW still lands.

    A figure outside the column's range is not a bigger bill, it is a garbled
    one -- a unit or schema drift somewhere upstream; a bounded review cannot
    legitimately spend a million dollars. What makes it worth a guard is not its
    likelihood but its blast radius: the insert would raise ``numeric field
    overflow``, and both transports discard the whole row on error (the HTTP
    path at ``raise_for_status``, the direct path in ``_record_run``'s blanket
    ``except``), so one unusable figure costs the duration, outcome, attempts
    and token counts beside it too.

    ``NaN`` is rejected for the opposite reason -- Postgres accepts it into a
    ``NUMERIC``, after which every ``sum(cost_usd)`` group containing that row
    is ``NaN`` for good. The harness already rejects both at capture; this is
    the guard at the boundary that owns the column, so it holds for callers
    that never went through a harness.
    """
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= _MAX_COST_USD:
        return None
    return number


#: The cost aggregates every per-group summary selects, as its trailing columns.
#:
#: Summed WITHOUT ``coalesce``: ``sum()`` over an all-NULL group is NULL, and
#: that is the answer we want to keep. Wrapping these in ``coalesce(.., 0)`` --
#: as the neighbouring ``findings`` sum legitimately does, findings being a
#: count everyone can produce -- would render a fleet of unmeasured pr-agent
#: runs as a fleet that cost nothing.
_COST_AGGREGATES = (
    "sum(input_tokens), sum(output_tokens), sum(cache_read_tokens), "
    "sum(cache_write_tokens), sum(cost_usd), sum(turns)"
)


def _int_or_none(value) -> int | None:
    """Coerce one aggregate to ``int``, preserving "nothing was measured"."""
    return None if value is None else int(value)


def _costs(row) -> dict:
    """Map the trailing :data:`_COST_AGGREGATES` columns of a row onto their keys.

    ``cost_usd`` is a ``NUMERIC`` and comes back as ``Decimal``, which does not
    survive JSON serialization; it is floated at its stored scale.
    """
    input_tokens, output_tokens, cache_read, cache_write, cost_usd, turns = row
    return {
        "input_tokens": _int_or_none(input_tokens),
        "output_tokens": _int_or_none(output_tokens),
        "cache_read_tokens": _int_or_none(cache_read),
        "cache_write_tokens": _int_or_none(cache_write),
        "cost_usd": None if cost_usd is None else round(float(cost_usd), 4),
        "turns": _int_or_none(turns),
    }


def _tool_calls(value) -> dict:
    """The call-count mapping this column will accept, entry by entry (#239).

    ``TranscriptIndexRequest`` states the same contract for the HTTP hop --
    ``dict[str, NonNegativeCount]`` -- but :func:`record` is also called
    DIRECTLY, with a plain mapping that no request model ever saw, so the two
    transports would otherwise disagree about what reaches a column documented
    as counts. Restated here so they cannot.

    An entry that does not fit is dropped rather than clamped or rejected: a
    count is not recoverable by guessing, and one unusable entry must not cost
    the rest of the row -- the same reason an unusable key costs only the
    reference (:func:`_index_transcript`) and an unstorable cost only itself
    (:func:`_storable_cost`). ``bool`` is excluded despite being an ``int``:
    ``True`` as a call count is a shape error, not one call.
    """
    if not isinstance(value, dict):
        return {}
    return {
        name: count
        for name, count in value.items()
        if isinstance(name, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
    }


def _index_transcript(transcript) -> str | None:
    """Write this run's transcript index row and return the key to reference (#239).

    ``None`` on every path that must leave ``review_runs.transcript_key`` NULL:
    no transcript, a key the blob store could never have stored, or an index
    write that did not land. A reference is only written for a row that exists,
    which is the invariant ``migrations/013`` states and holds by WRITE ORDER
    rather than by a foreign key -- so a transcript-side failure costs the
    reference and never the metrics row beside it.

    Its OWN ``db_best_effort`` block, deliberately, and that is the ordering
    constraint the acceptance criterion turns on: one block is one transaction,
    so writing both rows inside it would let a failing index insert roll the
    ``review_runs`` row back with it -- losing duration, outcome, attempts and
    token counts to an observability write.

    ``ON CONFLICT DO NOTHING`` because a re-delivered metrics post must not fail
    on a key it already stored; the blob is write-once and so is this row, so
    "already there" is success, not a conflict to resolve.

    Accepts a mapping or any object with the same attributes, so the endpoint
    can hand over its pydantic model and the direct path its dict without either
    side converting.
    """
    if not transcript:
        return None
    if not isinstance(transcript, dict):
        transcript = {
            field: getattr(transcript, field, None)
            for field in (
                "key",
                "complete",
                "tool_calls",
                "tool_result_bytes",
                "repeated_read_files",
            )
        }
    from .db import db_best_effort
    from .objectstore import BLOB_KEY_RE

    key = transcript.get("key")
    if not isinstance(key, str) or not BLOB_KEY_RE.fullmatch(key):
        # Validated HERE rather than by the request model: a 422 would reject the
        # whole body, so one unusable key would cost the run row too -- the same
        # blast-radius argument `_storable_cost` makes about an unstorable cost.
        print(f"fuko: transcript index skipped, invalid key {key!r}", file=sys.stderr)
        return None
    tool_calls = _tool_calls(transcript.get("tool_calls"))
    try:
        with db_best_effort() as conn:
            conn.execute(
                "INSERT INTO review_transcripts "
                "(key, complete, tool_calls, tool_result_bytes, repeated_read_files) "
                "VALUES (%s, %s, %s::jsonb, %s, %s) ON CONFLICT (key) DO NOTHING",
                (
                    key,
                    bool(transcript.get("complete")),
                    # Serialized here rather than passed as a dict: psycopg does
                    # not adapt a mapping to `jsonb` on its own, and json.dumps
                    # keeps this module free of a psycopg import it otherwise
                    # has no use for.
                    json.dumps(tool_calls),
                    int(transcript.get("tool_result_bytes") or 0),
                    int(transcript.get("repeated_read_files") or 0),
                ),
            )
    except Exception as e:
        print(f"fuko: transcript index write failed (continuing): {e}", file=sys.stderr)
        return None
    return key


def record(
    repo: str,
    pr: int,
    provider: str,
    model: str,
    *,
    slot: str | None = None,
    duration_s: float = 0.0,
    attempts: int = 1,
    outcome: str = "ok",
    findings: int | None = None,
    detail: str = "",
    backend: str = "pr-agent",
    endpoint: str = "",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    cost_usd: float | None = None,
    turns: int | None = None,
    transcript=None,
) -> None:
    """Insert one review-run row (no-op when persistence is disabled).

    ``backend`` is the driver that produced the run (#99); it defaults to
    ``"pr-agent"`` so an omitting caller writes the same value the schema backfill
    applied to pre-existing rows. ``endpoint`` is the base URL the answering
    entry was configured to reach (see ``RunReceipt.endpoint``); it defaults to
    ``""`` -- the SDK-default-endpoint value the backfill applied -- so an
    omitting caller stays consistent with pre-existing rows.

    The token/cost arguments (#152) default to ``None`` for the same reason the
    columns are nullable: a caller that cannot measure a figure must write "not
    measured", never a zero that would later be read as "free". ``cost_usd``
    passes :func:`_storable_cost` on the way in, so a figure the column cannot
    hold costs only itself rather than the whole row.

    ``transcript`` (#239) is this run's session-transcript index -- its key plus
    the per-tool figures derived from the feed. It is written FIRST, into its own
    table and its own transaction (:func:`_index_transcript`), and only a row
    that landed becomes the reference on this one; anything else records NULL,
    which is what a run with no transcript honestly has.
    """
    if not _enabled():
        return
    from .db import db_best_effort

    transcript_key = _index_transcript(transcript)
    with db_best_effort() as conn:
        conn.execute(
            "INSERT INTO review_runs "
            "(repo, pr, provider, model, slot, duration_s, attempts, outcome, findings, "
            "detail, backend, endpoint, input_tokens, output_tokens, cache_read_tokens, "
            "cache_write_tokens, cost_usd, turns, transcript_key) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                repo,
                pr,
                provider,
                model,
                slot,
                duration_s,
                attempts,
                outcome,
                findings,
                (detail or "")[:500],
                backend,
                endpoint,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                _storable_cost(cost_usd),
                turns,
                transcript_key,
            ),
        )


def slot_summary(repo: str | None = None, days: int = 30) -> list[dict]:
    """Aggregate runs per SLOT over the last ``days`` (empty when disabled).

    The slot view shows lane health independent of which model currently
    occupies it (slots are model-agnostic by design); rows without a slot
    (solo configs, rescued-by-backup rows keep their branch slot) are skipped.
    Token and cost totals ride along and are ``None`` for a lane whose runs
    reported none (see the module docstring).
    """
    if not _enabled():
        return []
    from .db import db_best_effort

    where = "WHERE slot IS NOT NULL AND started_at > now() - make_interval(days => %s)"
    params: list = [min(max(1, days), 3650)]
    if repo:
        where += " AND repo = %s"
        params.append(repo)

    with db_best_effort() as conn:
        rows = conn.execute(
            "SELECT slot, count(*), "
            "count(*) FILTER (WHERE outcome = 'ok'), "
            "count(*) FILTER (WHERE outcome != 'ok'), "
            "avg(duration_s), coalesce(sum(findings), 0), "
            f"{_COST_AGGREGATES} "
            f"FROM review_runs {where} "
            "GROUP BY slot ORDER BY slot",
            params,
        ).fetchall()
    return [
        {
            "slot": slot,
            "runs": runs,
            "ok": ok,
            "not_ok": not_ok,
            "avg_duration_s": round(float(avg_duration), 1) if avg_duration is not None else None,
            "findings": int(findings),
            **_costs(costs),
        }
        for slot, runs, ok, not_ok, avg_duration, findings, *costs in rows
    ]


def recent_runs(repo: str | None = None, limit: int = 50) -> list[dict]:
    """Return the newest run rows, bounded (empty when disabled).

    ``limit`` is clamped to [1, 200] so the viewer can never issue an
    unbounded read as ``review_runs`` grows.
    """
    if not _enabled():
        return []
    from .db import db_best_effort

    where = ""
    params: list = []
    if repo:
        where = "WHERE repo = %s"
        params.append(repo)
    params.append(min(max(1, limit), 200))

    with db_best_effort() as conn:
        rows = conn.execute(
            "SELECT repo, pr, provider, model, slot, started_at, duration_s, "
            "attempts, outcome, findings, backend "
            f"FROM review_runs {where} ORDER BY started_at DESC LIMIT %s",
            params,
        ).fetchall()
    return [
        {
            "repo": repo_,
            "pr": pr,
            "provider": provider,
            "model": model,
            "slot": slot,
            "started_at": started_at.isoformat(),
            "duration_s": float(duration_s),
            "attempts": attempts,
            "outcome": outcome,
            "findings": findings,
            "backend": backend,
        }
        for (
            repo_,
            pr,
            provider,
            model,
            slot,
            started_at,
            duration_s,
            attempts,
            outcome,
            findings,
            backend,
        ) in rows
    ]


def summary(repo: str | None = None, days: int = 30) -> list[dict]:
    """Aggregate runs per provider+model over the last ``days`` (empty when disabled).

    The grouping the whole exercise exists for: runs, outcomes, average
    duration, total findings, and what they spent per model -- filtered to one
    repo when given. Token and cost totals are ``None`` for a model whose runs
    reported none (see the module docstring).
    """
    if not _enabled():
        return []
    from .db import db_best_effort

    where = "WHERE started_at > now() - make_interval(days => %s)"
    params: list = [min(max(1, days), 3650)]
    if repo:
        where += " AND repo = %s"
        params.append(repo)

    with db_best_effort() as conn:
        rows = conn.execute(
            "SELECT provider, model, count(*), "
            "count(*) FILTER (WHERE outcome = 'ok'), "
            "count(*) FILTER (WHERE outcome != 'ok'), "
            "avg(duration_s), coalesce(sum(findings), 0), "
            f"{_COST_AGGREGATES} "
            f"FROM review_runs {where} "
            "GROUP BY provider, model ORDER BY count(*) DESC",
            params,
        ).fetchall()
    return [
        {
            "provider": provider,
            "model": model,
            "runs": runs,
            "ok": ok,
            "not_ok": not_ok,
            "avg_duration_s": round(float(avg_duration), 1) if avg_duration is not None else None,
            "findings": int(findings),
            **_costs(costs),
        }
        for provider, model, runs, ok, not_ok, avg_duration, findings, *costs in rows
    ]
