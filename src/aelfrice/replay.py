"""v2.0 #205 ingest_log validation harness.

Two checks per the spec at docs/design/write-log-as-truth.md:

1. **Reachability** (cheap): every belief in the canonical store has at
   least one ingest_log row that references its id in
   `derived_belief_ids`. This is the v2.0 contract guarantee — no
   orphan beliefs. Runs by default in `aelf doctor`.

2. **Full equality** (expensive, opt-in): re-run classifier over each
   `ingest_log.raw_text` and compare to canonical `beliefs`. This is
   the v2.x flip-readiness probe. Stubbed in v2.0 first slice;
   surfaced via `aelf doctor --replay` once the derivation function
   is factored out.

Per memo D5(C). Per memo D3, beliefs whose only log rows have
`source_kind=legacy_unknown` are excluded from full-equality checks
(they have no `raw_text` that the current classifier can re-derive).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aelfrice.models import INGEST_SOURCE_LEGACY_UNKNOWN
from aelfrice.store import MemoryStore


@dataclass(frozen=True)
class ReachabilityReport:
    """Result of the reachability check.

    `total_beliefs`: count of canonical beliefs in the store.
    `reachable`: count of beliefs with ≥1 log row pointing at them.
    `orphan_belief_ids`: beliefs with zero log rows. v2.0 contract
        requires this to be empty for stores that started life on
        v2.0; pre-v2.0 stores legitimately have orphans until the
        legacy_unknown migration runs (not shipped in this slice).
    """
    total_beliefs: int
    reachable: int
    orphan_belief_ids: list[str] = field(default_factory=list)

    @property
    def all_reachable(self) -> bool:
        return self.total_beliefs == self.reachable


def check_log_reachability(store: MemoryStore) -> ReachabilityReport:
    """Hypothesis-check the reachability contract.

    For every belief in `store`, query `iter_ingest_log_for_belief`.
    Any belief with zero log rows is an orphan — a violation of the
    spec's acceptance criterion #1.

    Cost: O(n_beliefs × n_log) in the linear-scan implementation
    (`iter_ingest_log_for_belief` walks all log rows). Acceptable for
    a doctor-tier check; the validation harness is not on the
    interactive path.
    """
    belief_ids = store.list_belief_ids()
    orphans: list[str] = []
    reachable = 0
    for bid in belief_ids:
        if store.iter_ingest_log_for_belief(bid):
            reachable += 1
        else:
            orphans.append(bid)
    return ReachabilityReport(
        total_beliefs=len(belief_ids),
        reachable=reachable,
        orphan_belief_ids=orphans,
    )


@dataclass(frozen=True)
class FullEqualityReport:
    """Stub. v2.0 first slice does not implement full-equality replay.

    Wired through so callers can detect "not implemented" without
    raising; the spec's acceptance criterion #3 is partially met by
    reachability, with full-equality landing in v2.x.
    """
    implemented: bool
    excluded_legacy_unknown: int


def replay_full_equality(store: MemoryStore) -> FullEqualityReport:
    """v2.x flip-readiness probe. Not implemented in v2.0 first slice.

    Returns `implemented=False` plus a count of legacy_unknown log
    rows that would be excluded from the comparison anyway. The
    intent is documented so a reviewer can grep this surface for the
    next slice's wiring.
    """
    cur = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log WHERE source_kind = ?",
        (INGEST_SOURCE_LEGACY_UNKNOWN,),
    )
    legacy_n = int(cur.fetchone()["n"])
    return FullEqualityReport(implemented=False, excluded_legacy_unknown=legacy_n)
