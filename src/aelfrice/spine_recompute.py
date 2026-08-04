"""Recompute the temporal spine from the log, and report the divergence (#1283).

The ratified contract (#1283, 2026-08-01; `PHILOSOPHY.md` and
`docs/design/write-log-as-truth.md`) says `edges` are **log-derived**:
the `TEMPORAL_NEXT` set is to be reproducible from `ingest_log`, ordered
by `(created_at, ingest_log ULID)`. The shipped writer instead orders by
`(created_at, rowid)`, and `rowid` is implicit — `VACUUM` may renumber
it — which is precisely why the ratified key is the log's ULID rather
than anything read off the belief table.

This module is the recompute half. It does **not** change the writer, so
running it against a live store measures a *gap*, not drift. Closing the
gap is the writer change tracked as item (3) of the 2026-08-04 ruling;
until writer and recompute share a key, a divergence report that read
zero would mean the recompute had been fitted to the defect.

## The three rules, and why each is not a heuristic

**Migration-synth log rows are excluded by `source_kind`.** The #263
legacy log synthesis minted 20,852 rows inside a 201 ms window whose
ULID prefix is migration wall-clock and whose order is `beliefs.rowid`
relabelled. They are identified by `source_kind = 'legacy_unknown'` — a
stated durable column, exact in both directions on the development store
(every such row falls inside the window, and every row in the window
carries it). Two heuristic detectors were tried first and both failed
loudly: "the ULID prefix disagrees with `ts` by more than a day" flags
legitimately delayed derivation and collapses reproduction to 34.75%,
and "exclude the largest ULID-prefix date cluster" would drop 51.8% of
the log — the 2026-07-07 bulk backfill, which is a real mint carrying
20,095 session-scoped beliefs. See
`benchmarks/ingest_log_ulid_clusters.py`.

**A belief takes its earliest qualifying log ULID.** A belief can be
corroborated by later rows; the first one is the one that records its
insertion, and it is the only one whose order says anything about when
the belief entered the store.

**Beliefs with no qualifying log row sort last within their
`created_at` group, then by id.** This is a *forward convention*, and
the distinction is load-bearing: it is **not** a recovery of historical
order, because there is none to recover. On the development store 2,426
session-scoped beliefs have no log row and they carry only 433 distinct
`created_at` values, so 93.9% of them sit inside a tie. Inside a tie the
shipped spine ordered by `rowid`; the durable columns available here are
`created_at` (tied, by construction) and `id` (content-addressed, so its
order is arbitrary with respect to insertion). There is no third column.
Three placement rules were measured — last, first, and id-interleaved —
and produced one link of spread between them. **The historical ordering
of the no-log bucket is unreconstructible.** The convention exists so
that the recompute is deterministic, not so that it is right about the
past.

Read-only: nothing here writes. Stdlib-only apart from the store handle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from aelfrice.models import EDGE_TEMPORAL_NEXT, INGEST_SOURCE_LEGACY_UNKNOWN

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aelfrice.store import MemoryStore

__all__ = [
    "SYNTH_SOURCE_KIND",
    "SpineDivergence",
    "recompute_spine_edges",
    "spine_divergence",
]

# The #263 legacy log synthesis stamps its rows with this `source_kind`.
# Rows carrying it are excluded from supplying an ordering key: their
# ULID prefix is migration wall-clock and their order is `beliefs.rowid`
# relabelled, so keying on them would launder rowid order into a key the
# contract calls durable.
SYNTH_SOURCE_KIND: Final[str] = INGEST_SOURCE_LEGACY_UNKNOWN

# Sorts after every real ULID (Crockford base32 tops out at 'Z'), so the
# no-log convention needs no branch in the sort key itself.
_NO_LOG_SENTINEL: Final[str] = "~"


@dataclass(frozen=True)
class SpineDivergence:
    """What a recompute reproduces, and what it does not.

    ``n_shipped`` / ``n_recomputed``
        Edge counts on each side.

    ``n_reproduced``
        Shipped edges the recompute also produces.

    ``missing_touching_no_log``
        Shipped edges the recompute misses where either endpoint has no
        qualifying log row. **Unreconstructible** — see the module
        docstring. Reported separately so it is never mistaken for
        drift.

    ``missing_fan_in``
        Shipped edges the recompute misses where the successor carries
        more than one predecessor edge. A chain gives each successor
        exactly one, so these are a writer defect, not a key
        disagreement. Counted, and expected to be non-increasing.

    ``missing_other``
        Everything else. This is the only bucket that should move when
        the key is wrong, which is why it is worth having on its own.
    """

    n_shipped: int
    n_recomputed: int
    n_reproduced: int
    missing_touching_no_log: int
    missing_fan_in: int
    missing_other: int

    @property
    def reproduced_share(self) -> float:
        """Fraction of the shipped spine the recompute reproduces."""
        if self.n_shipped == 0:
            return 1.0
        return self.n_reproduced / self.n_shipped


def _log_sort_keys(store: "MemoryStore") -> dict[str, str]:
    """`belief_id -> earliest qualifying ingest_log ULID`.

    Synth rows supply no key, so a belief covered only by them is
    treated as having no log row at all — which is correct: what it has
    is a row whose order is `rowid` wearing a ULID.
    """
    keys: dict[str, str] = {}
    cur = store._conn.execute(  # noqa: SLF001 - read-only, module is store-adjacent
        "SELECT id, derived_belief_ids FROM ingest_log "
        "WHERE derived_belief_ids IS NOT NULL AND source_kind != ? "
        "ORDER BY id ASC",
        (SYNTH_SOURCE_KIND,),
    )
    for row in cur:
        try:
            belief_ids = json.loads(row["derived_belief_ids"]) or []
        except (ValueError, TypeError):
            continue
        log_id = str(row["id"])
        for raw in belief_ids:
            # Rows arrive in ULID order, so the first write wins and is
            # the earliest. setdefault rather than a min() so the pass
            # stays linear.
            keys.setdefault(str(raw), log_id)
    return keys


def recompute_spine_edges(
    store: "MemoryStore",
) -> tuple[set[tuple[str, str]], set[str]]:
    """The `TEMPORAL_NEXT` set implied by the log, as `{(src, dst)}`.

    `src` is the successor and `dst` its predecessor, matching
    `temporal_spine.write_temporal_spine`.

    Also returns the set of belief ids that had no qualifying log row,
    so a caller can attribute divergence to that bucket without
    recomputing the join.

    Deterministic over a fixed store: the sort key is `(created_at, log
    ULID or sentinel, id)` and re-running it reproduces the same edge
    set. Durability of the ULID component is an **observed property, not
    a guarantee**, and the operator's #1283 constraint (3) forbids
    wording that implies otherwise: `ulid.make_generator` is monotone
    only *within* a process, and `ulid.py`'s own docstring records that
    "cross-process drift is possible but tolerated" because the
    `ingest_log` primary key requires only uniqueness. Measured on the
    development store, 0.017% of session-scoped log rows (10 groups,
    20 rows) share a millisecond across two distinct writer processes;
    their intra-millisecond order is 80 random bits rather than a write
    order, and today they decide zero spine links. A hard guarantee
    needs a deterministic intra-millisecond tiebreak, which this does
    not add.

    Soft-deleted beliefs stay in the chain, matching
    `session_predecessor_id` — spine integrity has to survive GC
    (#1064), and skip-but-continue happens at traversal time.
    """
    keys = _log_sort_keys(store)
    rows = store._conn.execute(  # noqa: SLF001 - read-only
        "SELECT id, session_id, created_at FROM beliefs "
        "WHERE session_id IS NOT NULL AND session_id != ''"
    ).fetchall()

    by_session: dict[str, list[tuple[str, str, str]]] = {}
    no_log: set[str] = set()
    for row in rows:
        bid = str(row["id"])
        sort_key = keys.get(bid)
        if sort_key is None:
            no_log.add(bid)
            sort_key = _NO_LOG_SENTINEL
        by_session.setdefault(str(row["session_id"]), []).append(
            (str(row["created_at"]), sort_key, bid)
        )

    edges: set[tuple[str, str]] = set()
    for members in by_session.values():
        members.sort()
        for predecessor, successor in zip(members, members[1:]):
            edges.add((successor[2], predecessor[2]))
    return edges, no_log


def spine_divergence(store: "MemoryStore") -> SpineDivergence:
    """Compare the shipped `TEMPORAL_NEXT` set against the recompute.

    Reports the misses in three buckets rather than as one percentage,
    because they have different meanings and only one of them is a
    defect anybody can fix. A single number would let the
    unreconstructible bucket mask a real key disagreement.
    """
    recomputed, no_log = recompute_spine_edges(store)
    shipped_rows = store._conn.execute(  # noqa: SLF001 - read-only
        "SELECT src, dst FROM edges WHERE type = ?", (EDGE_TEMPORAL_NEXT,)
    ).fetchall()
    shipped = {(str(r["src"]), str(r["dst"])) for r in shipped_rows}

    fan_in: dict[str, int] = {}
    for src, _dst in shipped:
        fan_in[src] = fan_in.get(src, 0) + 1

    reproduced = shipped & recomputed
    touching_no_log = 0
    fan_in_misses = 0
    other = 0
    for src, dst in shipped - recomputed:
        if src in no_log or dst in no_log:
            touching_no_log += 1
        elif fan_in.get(src, 0) > 1:
            fan_in_misses += 1
        else:
            other += 1

    return SpineDivergence(
        n_shipped=len(shipped),
        n_recomputed=len(recomputed),
        n_reproduced=len(reproduced),
        missing_touching_no_log=touching_no_log,
        missing_fan_in=fan_in_misses,
        missing_other=other,
    )
