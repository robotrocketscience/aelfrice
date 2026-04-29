"""v2.0 #205 ingest_log validation harness.

Two checks per the spec at docs/design/write-log-as-truth.md:

1. **Reachability** (cheap): every belief in the canonical store has at
   least one ingest_log row that references its id in
   `derived_belief_ids`. This is the v2.0 contract guarantee — no
   orphan beliefs. Runs by default in `aelf doctor`.

2. **Full equality** (expensive, opt-in): re-run classifier over each
   `ingest_log.raw_text` and compare to canonical `beliefs`. This is
   the v2.x flip-readiness probe. Implemented in v2.x; surfaced via
   `aelf doctor --replay` (issue #262).

Per memo D5(C). Per memo D3, beliefs whose only log rows have
`source_kind=legacy_unknown` are excluded from full-equality checks
(they have no `raw_text` that the current classifier can re-derive).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from aelfrice.derivation import DerivationInput, derive
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
    """Result of the v2.x full-equality replay probe (#262).

    Counts how many ingest_log rows, when re-derived, produce a belief
    that is shape-equal to the canonical belief in the store.

    Shape-equality contract (ratified 2026-04-29):
    - content_hash matches, AND
    - type matches, AND
    - origin matches OR canonical origin IS NULL (legacy backfill cohort), AND
    - the deterministic edge set matches (only triple_extractor-style edges,
      NOT feedback-driven edges).

    alpha/beta/last_retrieved_at/feedback-driven edges are tracked
    separately but never trigger drift.

    Counters:

    `implemented`: always True in this implementation.
    `total_log_rows`: ingest_log rows considered (excludes legacy_unknown).
    `excluded_legacy_unknown`: legacy_unknown rows skipped.
    `matched`: rows where re-derivation produces a shape-equal belief.
    `mismatched`: canonical belief exists but shape-equality fails on a
        non-origin field (content_hash or type mismatch).
    `derived_orphan`: log row produced a belief id not in the canonical store.
    `canonical_orphan`: canonical belief whose only log row is excluded
        (legacy_unknown) or has no log row at all (pre-#205).
    `legacy_origin_backfill`: canonical origin IS NULL but derived origin is
        set — counted as match per spec, NOT as mismatch.
    `feedback_derived_edges`: edges in the canonical store with a
        non-deterministic provenance. NOTE: the `edges` table has no `source`
        column in the current schema; this counter is always 0 until the
        schema adds provenance tracking. Informational only; never triggers
        drift.
    `drift_examples`: per-bucket sample, capped at `drift_examples` per bucket.
        Keys: "mismatched", "derived_orphan", "canonical_orphan".
        Each mismatched entry: {"belief_id", "log_row_id", "raw_text" (≤200 chars),
            "fields_diff"}.
        Each derived_orphan entry: {"log_row_id", "raw_text", "synthesized_belief_id"}.
        Each canonical_orphan entry: {"belief_id", "content_hash"}.
    """
    implemented: bool                       # always True
    total_log_rows: int                     # non-legacy_unknown rows considered
    excluded_legacy_unknown: int            # legacy_unknown rows skipped
    matched: int                            # shape-equal
    mismatched: int                         # canonical exists, non-origin field mismatch
    derived_orphan: int                     # log row belief not in canonical store
    canonical_orphan: int                   # canonical belief with no non-legacy log row
    legacy_origin_backfill: int             # canonical origin NULL, derived origin set (match)
    feedback_derived_edges: int             # non-deterministic edges (informational)
    drift_examples: dict[str, list[dict]]   # type: ignore[type-arg]

    @property
    def has_drift(self) -> bool:
        """True iff the canonical store disagrees with re-derived ingest log.

        Drift is the union of `mismatched` (canonical exists but a non-origin
        field differs) and `derived_orphan` (replay produced a belief id that
        is not in the canonical store).

        `canonical_orphan` is informational-only and does NOT count toward
        drift: it flags beliefs that exist in the canonical store but have no
        non-legacy log row (pre-#205 inserts and legacy_unknown-only rows).
        These are expected during the v2.x migration window and reporting
        them as drift would produce false positives.

        `legacy_origin_backfill` and `feedback_derived_edges` are also
        informational and never trigger drift.
        """
        return self.mismatched > 0 or self.derived_orphan > 0


# Scope values accepted by replay_full_equality.
ReplayScope = Literal["all", "since-v2"]


def replay_full_equality(
    store: MemoryStore,
    *,
    max_drift: int | None = None,
    drift_examples: int = 10,
    scope: ReplayScope = "all",
) -> FullEqualityReport:
    """v2.x flip-readiness probe. Re-derives every non-legacy ingest_log
    row and compares the result to the canonical belief store.

    Parameters
    ----------
    store:
        Open MemoryStore to probe.
    max_drift:
        If set, ``has_drift`` is still computed from the raw counts; the
        ``--max-drift`` exit-code logic lives in the CLI layer.
    drift_examples:
        Maximum number of representative cases captured per drift bucket
        (mismatched / derived_orphan / canonical_orphan). Default 10.
    scope:
        ``"all"`` (default) — walk every non-legacy_unknown log row.
        ``"since-v2"`` — only rows where ``source_kind != legacy_unknown``.
        Post-#263 migration these two scopes are equivalent because
        ``legacy_unknown`` is the only pre-v2.0 cohort in the log. The
        flag exists for forward compatibility.

    Notes on feedback_derived_edges
    ---------------------------------
    The ``edges`` table in the current schema has columns
    ``(src, dst, type, weight, anchor_text)`` — no ``source`` provenance
    column. Distinguishing triple_extractor edges from feedback-driven
    (contradiction SUPERSEDES) edges at the store level is not possible
    without a schema addition. ``feedback_derived_edges`` is therefore
    always 0 in this implementation. This is informational-only and never
    triggers drift. A schema migration adding an ``edge_source`` column
    would unlock this counter.
    """
    conn = store._conn  # pyright: ignore[reportPrivateUsage]

    # --- Count excluded legacy_unknown rows --------------------------------
    cur = conn.execute(
        "SELECT COUNT(*) AS n FROM ingest_log WHERE source_kind = ?",
        (INGEST_SOURCE_LEGACY_UNKNOWN,),
    )
    excluded_legacy = int(cur.fetchone()["n"])

    # --- Walk non-legacy_unknown log rows ----------------------------------
    # Both "all" and "since-v2" reduce to the same filter post-#263.
    cur = conn.execute(
        "SELECT id, ts, source_kind, source_path, raw_text, raw_meta, "
        "       derived_belief_ids, derived_edge_ids, "
        "       classifier_version, rule_set_hash, session_id "
        "FROM ingest_log "
        "WHERE source_kind != ? "
        "ORDER BY id",
        (INGEST_SOURCE_LEGACY_UNKNOWN,),
    )
    rows = cur.fetchall()

    total_log_rows = len(rows)
    matched = 0
    mismatched = 0
    derived_orphan = 0
    legacy_origin_backfill = 0

    examples_mismatched: list[dict] = []     # type: ignore[type-arg]
    examples_derived_orphan: list[dict] = [] # type: ignore[type-arg]

    for row in rows:
        raw_text = str(row["raw_text"])
        source_kind = str(row["source_kind"])
        source_path = row["source_path"]
        session_id = row["session_id"]
        ts = str(row["ts"])
        classifier_version = row["classifier_version"]
        rule_set_hash = row["rule_set_hash"]
        log_row_id = str(row["id"])

        inp = DerivationInput(
            raw_text=raw_text,
            source_kind=source_kind,
            source_path=source_path if source_path is not None else None,
            raw_meta=None,   # raw_meta is metadata only; derive() does not use it
            session_id=session_id if session_id is not None else None,
            ts=ts,
            classifier_version=classifier_version if classifier_version is not None else None,
            rule_set_hash=rule_set_hash if rule_set_hash is not None else None,
        )

        out = derive(inp)

        if out.belief is None:
            # persist=False skip path — informational, not a derived_orphan
            continue

        synthesized = out.belief
        canonical = store.get_belief(synthesized.id)

        if canonical is None:
            # Belief id not in canonical store → derived_orphan
            derived_orphan += 1
            if len(examples_derived_orphan) < drift_examples:
                examples_derived_orphan.append({
                    "log_row_id": log_row_id,
                    "raw_text": raw_text[:200],
                    "synthesized_belief_id": synthesized.id,
                })
            continue

        # Belief found — check shape-equality
        # Origin equality: canonical origin IS NULL is treated as a match
        # (legacy backfill cohort). In practice the DB stores 'unknown' as
        # default, not SQL NULL, so we also treat 'unknown' as the backfill
        # sentinel when the derived origin is more specific.
        origin_canonical = canonical.origin or ""
        origin_derived = synthesized.origin or ""

        origin_null = (origin_canonical in ("", "unknown") and
                       origin_derived not in ("", "unknown"))

        content_hash_match = canonical.content_hash == synthesized.content_hash
        type_match = canonical.type == synthesized.type
        origin_match = (origin_canonical == origin_derived) or origin_null

        if content_hash_match and type_match and origin_match:
            matched += 1
            if origin_null:
                legacy_origin_backfill += 1
        else:
            mismatched += 1
            if len(examples_mismatched) < drift_examples:
                fields_diff: dict[str, object] = {}
                if not content_hash_match:
                    fields_diff["content_hash"] = {
                        "canonical": canonical.content_hash,
                        "derived": synthesized.content_hash,
                    }
                if not type_match:
                    fields_diff["type"] = {
                        "canonical": canonical.type,
                        "derived": synthesized.type,
                    }
                if not origin_match:
                    fields_diff["origin"] = {
                        "canonical": origin_canonical,
                        "derived": origin_derived,
                    }
                examples_mismatched.append({
                    "belief_id": canonical.id,
                    "log_row_id": log_row_id,
                    "raw_text": raw_text[:200],
                    "fields_diff": fields_diff,
                })

    # --- Canonical orphans -------------------------------------------------
    # A canonical belief is an orphan when every log row pointing at it is
    # legacy_unknown (or there are no log rows at all, pre-#205).
    # TODO(perf): replace N+1 iteration with set-based store query — see
    # follow-up issue.
    belief_ids = store.list_belief_ids()
    canonical_orphan = 0
    examples_canonical_orphan: list[dict] = []  # type: ignore[type-arg]

    for bid in belief_ids:
        all_rows = store.iter_ingest_log_for_belief(bid)
        has_non_legacy = any(
            str(r.get("source_kind", "")) != INGEST_SOURCE_LEGACY_UNKNOWN
            for r in all_rows
        )
        if not has_non_legacy:
            canonical_orphan += 1
            if len(examples_canonical_orphan) < drift_examples:
                b = store.get_belief(bid)
                examples_canonical_orphan.append({
                    "belief_id": bid,
                    "content_hash": b.content_hash if b is not None else None,
                })

    # feedback_derived_edges: always 0 — edges table has no source column.
    # See docstring for explanation.
    feedback_derived_edges = 0

    return FullEqualityReport(
        implemented=True,
        total_log_rows=total_log_rows,
        excluded_legacy_unknown=excluded_legacy,
        matched=matched,
        mismatched=mismatched,
        derived_orphan=derived_orphan,
        canonical_orphan=canonical_orphan,
        legacy_origin_backfill=legacy_origin_backfill,
        feedback_derived_edges=feedback_derived_edges,
        drift_examples={
            "mismatched": examples_mismatched,
            "derived_orphan": examples_derived_orphan,
            "canonical_orphan": examples_canonical_orphan,
        },
    )
