"""v2.0 #205 ingest_log validation harness.

Two checks per the spec at docs/design/write-log-as-truth.md:

1. **Reachability** (cheap): every belief in the canonical store has at
   least one ingest_log row that references its id in
   `derived_belief_ids`. This is the v2.0 contract guarantee — no
   orphan beliefs. Not currently wired into any CLI entry point; call
   `check_log_reachability(store)` directly.

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
from aelfrice.derivation_worker import _route_overrides_from_raw_meta
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
    - origin matches OR canonical origin IS NULL (legacy backfill cohort).

    These three are the fields `derive()` alone determines and that no
    post-ingest operation rewrites, so a divergence is unambiguously a
    derivation regression. Only they trigger `has_drift`.

    Mutable divergence (#1167). The remaining belief columns are compared
    too, but reported in a separate, non-drift-triggering bucket:

        alpha, beta, lock_level, retention_class, scope,
        last_retrieved_at

    Each of these is legitimately rewritten after ingest — corroboration
    and feedback move (alpha, beta); `aelf lock` moves lock_level; the
    snapshot lifecycle moves retention_class; federation moves scope;
    retrieval stamps last_retrieved_at. None is reconstructible from the
    write log as it exists today, so counting them as drift would report
    a false positive on every live store. They were previously dropped
    silently, which is how the #1167 5x alpha deflation stayed invisible
    while its origin half was flagged. Making them visible-but-not-drift
    is the honest reading of "full equality" until the log is total
    (#1157), at which point they can be promoted into the strict contract.

    Edge set (#1354). `derive()` now emits DERIVED_FROM edges from the
    row's own `raw_meta`, so the edge set — unlike the mutable fields
    above — IS log-derivable, and a disagreement with the logged
    `derived_edge_ids` is a derivation regression. It therefore counts
    toward drift, in its own `edge_set_divergence` counter rather than
    inside `mutable_divergence`.

    The comparison is watermarked on the raw column value. SQL NULL means
    no edge-aware writer ever stamped the row — the pre-#1354 cohort, or
    a call site that writes a log row without going through
    `run_worker` — and is exempt; `'[]'` means stamped-and-derived-none
    and is compared. Without that split, promoting the edge set would
    report the entire pre-#1354 history as drift.

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
    `mutable_divergence`: log rows where the canonical belief matched on the
        strict contract but differed on ≥1 mutable field. Informational;
        never triggers drift.
    `mutable_field_counts`: per-field row counts behind `mutable_divergence`.
        Keys are drawn from `MUTABLE_FIELDS`; absent keys mean zero.
    `edge_set_divergence`: log rows whose stamped `derived_edge_ids`
        disagrees with the re-derived edge set. Rows with a NULL column
        are exempt. DOES trigger drift.
    `drift_examples`: per-bucket sample, capped at `drift_examples` per bucket.
        Keys: "mismatched", "derived_orphan", "canonical_orphan",
        "mutable_divergence", "edge_set".
        Each mismatched entry: {"belief_id", "log_row_id", "raw_text" (≤200 chars),
            "fields_diff"}.
        Each derived_orphan entry: {"log_row_id", "raw_text", "synthesized_belief_id"}.
        Each canonical_orphan entry: {"belief_id", "content_hash"}.
        Each mutable_divergence entry: {"belief_id", "log_row_id", "fields_diff"}.
        Each edge_set entry: {"belief_id", "log_row_id", "canonical", "derived"}.
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
    mutable_divergence: int = 0             # ≥1 mutable field differs (informational)
    mutable_field_counts: dict[str, int] = field(default_factory=dict)
    edge_set_divergence: int = 0            # logged edges != re-derived edges (DRIFT)

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

        `edge_set_divergence` DOES count toward drift (#1354). Unlike the
        mutable fields, the edge set is log-derivable: `derive()` emits it
        from the row's own `raw_meta`, so a disagreement with the logged
        `derived_edge_ids` is a derivation regression, not a legitimate
        post-ingest rewrite. Rows predating the edge-aware writer carry
        SQL NULL and are exempt — see the comparison site.

        `legacy_origin_backfill`, `feedback_derived_edges` and
        `mutable_divergence` are also informational and never trigger
        drift. See the class docstring for why the mutable fields are
        reported rather than enforced.
        """
        return (
            self.mismatched > 0
            or self.derived_orphan > 0
            or self.edge_set_divergence > 0
        )


# Scope values accepted by replay_full_equality.
ReplayScope = Literal["all", "since-v2"]

# Belief fields compared outside the strict shape-equality contract (#1167).
# Order is the reporting order; see FullEqualityReport for why each one is
# informational rather than drift-triggering.
MUTABLE_FIELDS: tuple[str, ...] = (
    "alpha",
    "beta",
    "lock_level",
    "retention_class",
    "scope",
    "last_retrieved_at",
    "locked_at",
    "lock_tier",
    "lock_expires_at",
    "last_confirmed_at",
    "valid_to",
    "corroboration_count",
    "hibernation_score",
    "activation_condition",
    "session_id",
    "project_context",
)

# The strict shape-equality contract, named rather than left implicit
# (#1345). These are the fields `derive()` alone determines and that no
# post-ingest operation rewrites, so a divergence is unambiguously a
# derivation regression -- and only these set `has_drift`.
STRICT_FIELDS: tuple[str, ...] = ("content_hash", "type", "origin")

# Belief fields deliberately compared by neither set, each with the reason
# (#1345). This exists so "not compared" is a decision on the record: the
# gap this issue closed was 13 fields that nothing had ever classified,
# and an enumeration without a home for the exclusions would just move the
# silence one level up.
EXCLUDED_FIELDS: dict[str, str] = {
    "id": (
        "content-addressed identity. It is a function of the same inputs "
        "as content_hash, which is in the strict set, so comparing it "
        "again reports the same divergence twice."
    ),
    "created_at": (
        "the wall clock at insert, not a re-derivable value. Measured "
        "rather than reasoned: comparing it reported divergence on 25 of "
        "25 rows of a clean synthetic store, because the canonical value "
        "is when the writer ran and the re-derived one is the log row's "
        "ts. A field that diverges on every row is not a signal, it is a "
        "constant that hides the fields that are. Reconstructing insert "
        "order from the log is #1283, and it is bounded there, not here."
    ),
    "content": (
        "the input `content_hash` is computed over. A content difference "
        "is already a strict-contract failure; comparing the text as well "
        "adds a second report of one defect and puts belief text into the "
        "drift examples."
    ),
}

# Absolute tolerance for the alpha/beta comparison. derive() recomputes the
# prior from the same table the writer used, so an exact match is expected;
# the epsilon only absorbs SQLite REAL round-trip noise.
_POSTERIOR_EPSILON: float = 1e-9

# Mutable fields compared with the epsilon rather than by equality, because
# they round-trip through SQLite REAL.
_FLOAT_MUTABLE_FIELDS: frozenset[str] = frozenset(
    {"alpha", "beta", "hibernation_score"}
)


def _mutable_fields_diff(
    canonical: object,
    synthesized: object,
) -> dict[str, object]:
    """Return the per-field diff for the mutable (non-log-derivable) set.

    Empty dict means the canonical belief agrees with the re-derived one on
    every field outside the strict contract.
    """
    diff: dict[str, object] = {}
    # One pass, in MUTABLE_FIELDS order. The order matters: `diff` is a dict,
    # it is rendered verbatim into the drift examples `aelf doctor replay`
    # prints, and iterating `_FLOAT_MUTABLE_FIELDS` (a frozenset) instead
    # would key that output on string hash randomisation — a different field
    # order every process, for the same store. Nondeterministic output from
    # the instrument that exists to falsify the determinism claim.
    for name in MUTABLE_FIELDS:
        c_val = getattr(canonical, name)
        d_val = getattr(synthesized, name)
        if name in _FLOAT_MUTABLE_FIELDS:
            if c_val is None or d_val is None:
                if c_val != d_val:
                    diff[name] = {"canonical": c_val, "derived": d_val}
                continue
            c = float(c_val)
            d = float(d_val)
            if abs(c - d) > _POSTERIOR_EPSILON:
                diff[name] = {"canonical": c, "derived": d}
            continue
        if c_val != d_val:
            diff[name] = {"canonical": c_val, "derived": d_val}
    return diff


def _logged_edge_set(blob: object) -> set[tuple[str, str, str]]:
    """Decode an ingest_log `derived_edge_ids` blob into (src, dst, type).

    The column is JSON TEXT holding a list of 3-element lists. NULL, invalid
    JSON, and rows of the wrong shape all decode to the empty set — this is a
    reporting probe, not a validator, and a malformed row must not abort the
    walk.
    """
    import json as _json  # noqa: PLC0415 - keep replay's stdlib footprint local

    if not blob:
        return set()
    try:
        decoded = _json.loads(blob)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return set()
    if not isinstance(decoded, list):
        return set()
    out: set[tuple[str, str, str]] = set()
    for item in decoded:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            out.add((str(item[0]), str(item[1]), str(item[2])))
    return out


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

    import json as _json  # noqa: PLC0415 - keep replay's stdlib footprint local
    _META_OVERRIDE_BELIEF_TYPE = "override_belief_type"

    total_log_rows = len(rows)
    matched = 0
    mismatched = 0
    derived_orphan = 0
    legacy_origin_backfill = 0
    mutable_divergence = 0
    mutable_field_counts: dict[str, int] = {}
    edge_set_divergence = 0

    examples_mismatched: list[dict] = []     # type: ignore[type-arg]
    examples_derived_orphan: list[dict] = [] # type: ignore[type-arg]
    examples_mutable: list[dict] = []        # type: ignore[type-arg]
    examples_edge_set: list[dict] = []       # type: ignore[type-arg]

    for row in rows:
        raw_text = str(row["raw_text"])
        source_kind = str(row["source_kind"])
        source_path = row["source_path"]
        session_id = row["session_id"]
        ts = str(row["ts"])
        classifier_version = row["classifier_version"]
        rule_set_hash = row["rule_set_hash"]
        log_row_id = str(row["id"])

        # Reconstruct override_belief_type and route_overrides from
        # raw_meta when present — mirrors
        # derivation_worker._derivation_input_from_row so replay equality
        # holds for #264 slice 2 host-classified rows AND for #265 PR-B
        # LLM-routed rows. `route_overrides` are frozen at ingest time
        # per the memo at docs/design/v2_view_flip_scanner_call_site.md;
        # replay must re-apply them verbatim, otherwise the canonical
        # belief (which carries the router's fields) drifts from the
        # re-derived shell (which would carry derive()'s defaults).
        raw_meta_blob = row["raw_meta"]
        override_belief_type: str | None = None
        meta_obj: object | None = None
        if raw_meta_blob:
            try:
                meta_obj = _json.loads(raw_meta_blob)
            except _json.JSONDecodeError:
                meta_obj = None
            if isinstance(meta_obj, dict):
                ov = meta_obj.get(_META_OVERRIDE_BELIEF_TYPE)
                if isinstance(ov, str) and ov:
                    override_belief_type = ov
        meta_dict = meta_obj if isinstance(meta_obj, dict) else None
        route_overrides = _route_overrides_from_raw_meta(meta_dict)
        inp = DerivationInput(
            raw_text=raw_text,
            source_kind=source_kind,
            source_path=source_path if source_path is not None else None,
            # #1167: raw_meta must round-trip verbatim. `derive()` reads
            # `raw_meta["role"]` to route source_kind=transcript user
            # turns to the undeflated USER_SOURCE prior and
            # ORIGIN_USER_TRANSCRIPT (derivation.py, #888/#1089).
            # Nulling it here re-derived every user-typed belief as
            # agent_inferred at 1/5 the alpha, so `aelf doctor --replay`
            # reported drift on every healthy store.
            raw_meta=meta_dict,
            session_id=session_id if session_id is not None else None,
            ts=ts,
            classifier_version=classifier_version if classifier_version is not None else None,
            rule_set_hash=rule_set_hash if rule_set_hash is not None else None,
            override_belief_type=override_belief_type,
            route_overrides=route_overrides,
        )

        out = derive(inp)

        if out.belief is None:
            # persist=False skip path — informational, not a derived_orphan
            continue

        synthesized = out.belief
        # include_retired (#1210): this reconciles ingest_log against the
        # rows that exist, and a tombstone is still a row. Under the default
        # every retired belief would reclassify as a derived_orphan and
        # inflate the drift counts the replay-soak gate reads.
        canonical = store.get_belief(synthesized.id, include_retired=True)

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

        # #1354: the edge set, which unlike the mutable fields IS
        # log-derivable, so a divergence is drift.
        #
        # The guard tests the raw blob for SQL NULL, not the decoded set
        # and not truthiness. `_logged_edge_set` collapses NULL, '[]',
        # malformed JSON and wrong-shape rows all to the empty set, so
        # guarding on its output would silently exempt corruption as
        # well as history. NULL means no edge-aware writer ever stamped
        # this row (the pre-#1354 cohort, or a call site that writes the
        # log row without run_worker); '[]' means stamped-and-empty and
        # IS compared.
        logged_blob = row["derived_edge_ids"]
        if logged_blob is not None:
            derived_edges = {(e.src, e.dst, e.type) for e in out.edges}
            logged_edges = _logged_edge_set(logged_blob)
            if logged_edges != derived_edges:
                edge_set_divergence += 1
                if len(examples_edge_set) < drift_examples:
                    examples_edge_set.append({
                        "belief_id": canonical.id,
                        "log_row_id": log_row_id,
                        "canonical": sorted(logged_edges),
                        "derived": sorted(derived_edges),
                    })

        # #1167: compare the fields outside the strict contract too. Counted
        # and exampled separately; never promoted into `has_drift`.
        mutable_diff = _mutable_fields_diff(canonical, synthesized)
        if mutable_diff:
            mutable_divergence += 1
            for name in mutable_diff:
                mutable_field_counts[name] = mutable_field_counts.get(name, 0) + 1
            if len(examples_mutable) < drift_examples:
                examples_mutable.append({
                    "belief_id": canonical.id,
                    "log_row_id": log_row_id,
                    "fields_diff": mutable_diff,
                })

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
    all_orphans = store.list_canonical_orphans()
    canonical_orphan = len(all_orphans)
    examples_canonical_orphan: list[dict] = []  # type: ignore[type-arg]
    for bid, content_hash in all_orphans[:drift_examples]:
        examples_canonical_orphan.append({
            "belief_id": bid,
            "content_hash": content_hash,
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
        mutable_divergence=mutable_divergence,
        mutable_field_counts=mutable_field_counts,
        edge_set_divergence=edge_set_divergence,
        drift_examples={
            "mismatched": examples_mismatched,
            "derived_orphan": examples_derived_orphan,
            "canonical_orphan": examples_canonical_orphan,
            "mutable_divergence": examples_mutable,
            "edge_set": examples_edge_set,
        },
    )
