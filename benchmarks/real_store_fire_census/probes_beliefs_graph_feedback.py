"""Census probes for subsystem beliefs-graph-feedback (A001-A037).

Every probe reads one store through `ctx.db` (read-only) or the JSONL
sidecars beside it, and returns integers and fixed labels only. Writers
each probe keys on, read from src/aelfrice at main ad14ef545:

  feedback_history.source
    'user'                       cli._cmd_feedback default --source
    'user_confirmed'             cli._cmd_confirm default --source
    'propagation:<src>'          feedback.apply_feedback, PROPAGATION_SOURCE_PREFIX
    'hook'                       hook_search.record_retrieval (exposure)
    'sentiment_inferred'         sentiment_feedback.apply_sentiment_to_pending
    'promotion:user_validated'   promotion.promote via cli._cmd_validate
    'promotion:<label>'          same, with --source <label>
    'promotion:phantom_lock_match'  cli._cmd_lock Surface B
    'promotion:revert_to_agent_inferred', 'promotion:revert_evidence'  cli._cmd_demote
    'promotion:evidence'         phantom_promotion_opportunity (other subsystem)
    'lock:unlock'                promotion.unlock via aelf unlock AND aelf demote on a lock
    'scope:<old>-><new>'         cli._apply_scope_change (validate/promote AND demote)
    'contradiction_tiebreaker:<rule>'  contradiction.resolve_contradiction
  edges.type
    CONTRADICTS weight 1.0, anchor_text NULL, src<dst  relationship_detector.write_semantic_edges
    CONTRADICTS with anchor_text                        triple_extractor (commit hook)
    POTENTIALLY_STALE weight 1.0                        relationship_detector.write_potentially_stale_edges
    SUPERSEDES + RESOLVES (weight 0.0)                  contradiction.resolve_contradiction
  deferred_feedback_queue event_type='retrieval_exposure'  retrieval.retrieve when enqueue on
  belief_support_evidence kind='reworded_restatement'       supports_log.log_reworded_support
  beliefs.type='correction'    classification_core step 4 (correction.detect_correction),
                               or override_belief_type / route_overrides in ingest_log.raw_meta
  hook_audit.jsonl rows hook='sentiment_feedback'  hook._write_sentiment_feedback_audit
  feed.jsonl rows event/kind   cli._feed_log_event

Some stores carry a schema older than aelfrice (beliefs with belief_type,
beta_param and locked; edges with from_id, to_id and edge_type). No
aelfrice code reads or writes that shape, so a probe whose writer needs
the current columns reports `untraceable` there instead of failing.

Release dates used to split rows written by an older default from rows
the current code wrote come from the CHANGELOG entry named beside each.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import core as cc

Ctx = cc.Ctx
Result = cc.Result

# --- helpers ---------------------------------------------------------------


def _store_probe(fn: Callable[[Ctx], Result]) -> Callable[[Ctx], Result]:
    """Run fn only when the context has a db; else silent with no count."""
    def wrapped(ctx: Ctx) -> Result:
        if ctx.db is None:
            return cc.silent(None, note="no db")
        return fn(ctx)
    wrapped.__name__ = fn.__name__
    return wrapped


def _n(ctx: Ctx, table: str, sql: str, params: tuple[Any, ...] = ()) -> int:
    """ctx.count, or 0 when `table` is absent from this store."""
    if not ctx.table_exists(table):
        return 0
    return ctx.count(sql, params)


def _fb(ctx: Ctx, where: str, params: tuple[Any, ...] = ()) -> int:
    return _n(ctx, "feedback_history",
              f"SELECT COUNT(*) FROM feedback_history WHERE {where}", params)


def _audit_rows(ctx: Ctx) -> list[Path]:
    return cc.jsonl_paths(ctx.sidecar("hook_audit.jsonl"))


def _sentiment_rows(ctx: Ctx, pred: Callable[[dict[str, Any]], bool]) -> int:
    return cc.jsonl_count(_audit_rows(ctx),
                          lambda r: r.get("hook") == "sentiment_feedback" and pred(r))


def _feed_count(ctx: Ctx, event: str, kind: str | None = None) -> int:
    return cc.jsonl_count(
        ctx.sidecar("feed.jsonl"),
        lambda r: r.get("event") == event and (kind is None or r.get("kind") == kind))


def _edges(ctx: Ctx, where: str) -> int:
    return _n(ctx, "edges", f"SELECT COUNT(*) FROM edges WHERE {where}")


def _seq_gap(ctx: Ctx, table: str) -> int:
    """AUTOINCREMENT high-water mark minus live rows: rows ever deleted."""
    if not ctx.table_exists("sqlite_sequence") or not ctx.table_exists(table):
        return 0
    seq = ctx.count("SELECT COALESCE(MAX(seq), 0) FROM sqlite_sequence WHERE name = ?",
                    (table,))
    live = ctx.count(f"SELECT COUNT(*) FROM {table}")
    return max(seq - live, 0)


def _typed_edges(ctx: Ctx) -> bool:
    """The current edges schema: src, dst and type columns."""
    return all(ctx.column_exists("edges", c) for c in ("src", "dst", "type"))


def _after(col: str) -> str:
    """SQL test that a text or epoch timestamp column is on or after (?, ?)."""
    return (f"((typeof({col}) = 'text' AND substr({col}, 1, 10) >= ?) OR "
            f"(typeof({col}) IN ('integer', 'real') AND {col} >= ?))")


def _rows_after(ctx: Ctx, table: str, day: str, epoch: int) -> int:
    if not ctx.column_exists(table, "created_at"):
        return 0
    return ctx.count(f"SELECT COUNT(*) FROM {table} WHERE " + _after("created_at"),
                     (day, epoch))


def _days_vs(ctx: Ctx, table: str, col: str, day: str, agg: str = "MAX") -> int:
    """Whole days between `day` and agg(col) in table (positive when later)."""
    return ctx.count(
        f"SELECT CAST(julianday(substr({agg}({col}),1,10)) - julianday(?) AS INTEGER) "
        f"FROM {table}", (day,))


# Rules written by contradiction._pick_winner: <w>_beats_<l>,
# <c>_beats_<c>_by_recency, <c>_beats_<c>_by_id.
_CLASSES = ("user_stated", "user_corrected", "user_validated", "user_transcript",
            "document_recent", "agent_inferred")
_RULES_CROSS = frozenset(f"{w}_beats_{l}" for w in _CLASSES for l in _CLASSES if w != l)
_RULES_TIE = frozenset(f"{c}_beats_{c}_by_{k}" for c in _CLASSES for k in ("recency", "id"))
_TIEBREAK_PREFIX = "contradiction_tiebreaker:"

_PROMOTION_OTHER = (
    "promotion:phantom_lock_match", "promotion:evidence",
    "promotion:revert_to_agent_inferred", "promotion:revert_evidence",
)

_EDGE_TYPES_LOWER = frozenset({
    "supports", "cites", "contradicts", "supersedes", "relates_to", "derived_from",
    "implements", "temporal_next", "tests", "resolves", "potentially_stale",
})

# CHANGELOG/v1.md 1.0.3: contradiction tie-breaker and aelf resolve.
RESOLVE_SHIP_DAY = "2026-04-27"
RESOLVE_SHIP_EPOCH = 1777248000
# CHANGELOG/v4.md 4.0.0: valence propagation is live (#1058).
PROPAGATION_SHIP_DAY = "2026-07-07"
PROPAGATION_SHIP_EPOCH = 1783382400
# 4.0.0 shipped 2026-07-07; release-day rows still show the old exposure bump.
EXPOSURE_RELEASE_DAY = "2026-07-08"
# 4.3.0 flipped retrieval-exposure enqueue off (#1162) and added the lock floor (#1168).
ENQUEUE_FLIP_DAY = "2026-08-13"
ENQUEUE_FLIP_GRACE_DAY = "2026-08-20"
LOCK_FLOOR_RELEASE_DAY = "2026-08-13"
# 5.1.0 removed the relevance sweeper, the only writer of injection_events.referenced (#1655).
SWEEPER_REMOVAL_DAY = "2026-10-06"
# 5.2.0 is the first release with the RESOLVES writer (#1658).
RESOLVES_RELEASE_DAY = "2026-10-22"

# --- probes ----------------------------------------------------------------


@_store_probe
def p_a002(ctx: Ctx) -> Result:
    """aelf confirm: feedback rows plus feed rows; the lock join needs lock columns."""
    n = _fb(ctx, "source = 'user_confirmed'")
    has_lock_cols = (ctx.column_exists("beliefs", "lock_level")
                     and ctx.column_exists("beliefs", "locked_at"))
    on_lock = 0
    if has_lock_cols:
        on_lock = _n(
            ctx, "feedback_history",
            "SELECT COUNT(*) FROM feedback_history f JOIN beliefs b ON b.id = f.belief_id "
            "WHERE f.source = 'user_confirmed' AND b.lock_level = 'user' "
            "AND substr(f.created_at, 1, 19) >= substr(b.locked_at, 1, 19)")
    feed = _feed_count(ctx, "feedback.applied", "confirm")
    return cc.from_count(n + feed, {
        "fb_user_confirmed": n,
        "on_locked_after_lock": on_lock,
        "feed_confirm_rows": feed,
        "aelfrice_lock_schema": int(has_lock_cols),
        "beliefs_rows": _n(ctx, "beliefs", "SELECT COUNT(*) FROM beliefs"),
        "fb_rows": _n(ctx, "feedback_history", "SELECT COUNT(*) FROM feedback_history"),
    }, note="count = feedback_history user_confirmed + feed confirm rows")


@_store_probe
def p_a003(ctx: Ctx) -> Result:
    a = _fb(ctx, "source = 'promotion:revert_to_agent_inferred'")
    b = _fb(ctx, "source = 'promotion:revert_evidence'")
    return cc.from_count(a + b, {"revert_to_agent_inferred": a, "revert_evidence": b},
                         note="demote on a lock writes lock:unlock (A008); --to-scope writes scope rows (A009)")


def _a005_legacy(ctx: Ctx) -> Result:
    legacy = ctx.column_exists("edges", "edge_type")
    detail = {
        "legacy_schema": int(legacy),
        "edges_rows": ctx.count("SELECT COUNT(*) FROM edges"),
        "beliefs_rows_after_ship": _rows_after(ctx, "beliefs", RESOLVE_SHIP_DAY,
                                               RESOLVE_SHIP_EPOCH),
        "edges_rows_after_ship": _rows_after(ctx, "edges", RESOLVE_SHIP_DAY,
                                             RESOLVE_SHIP_EPOCH),
    }
    if legacy:
        for t in ("CONTRADICTS", "SUPERSEDES", "RESOLVES"):
            detail[f"legacy_{t.lower()}_edges"] = ctx.count(
                "SELECT COUNT(*) FROM edges WHERE upper(edge_type) = ?", (t,))
    return cc.Result("untraceable", None, detail,
                     "pre-v1 edges schema (from_id,to_id,edge_type); resolver reads src,dst,type")


@_store_probe
def p_a005(ctx: Ctx) -> Result:
    """aelf resolve: tie-breaker audit rows. The resolver needs typed edges."""
    if not ctx.table_exists("edges"):
        return cc.silent(0, {"edges_table": 0}, note="no edges table")
    if not (ctx.column_exists("edges", "src") and ctx.column_exists("edges", "type")):
        return _a005_legacy(ctx)
    fb = 0
    if ctx.column_exists("feedback_history", "source"):
        fb = _fb(ctx, "source LIKE 'contradiction_tiebreaker:%'")
    unresolved = ctx.count(
        "SELECT COUNT(*) FROM (SELECT DISTINCT min(src, dst) a, max(src, dst) b "
        "FROM edges WHERE type = 'CONTRADICTS' AND src <> dst) p "
        "WHERE NOT EXISTS (SELECT 1 FROM edges s WHERE s.type = 'SUPERSEDES' AND "
        "((s.src = p.a AND s.dst = p.b) OR (s.src = p.b AND s.dst = p.a)))")
    return cc.from_count(fb, {
        "resolves_edges": _edges(ctx, "type = 'RESOLVES'"),
        "supersedes_edges": _edges(ctx, "type = 'SUPERSEDES'"),
        "contradicts_edges": _edges(ctx, "type = 'CONTRADICTS'"),
        "unresolved_pairs": unresolved,
    })


@_store_probe
def p_a006(ctx: Ctx) -> Result:
    gap = _seq_gap(ctx, "deferred_feedback_queue")
    return cc.from_shared_count(gap, {
        "queue_rows": _n(ctx, "deferred_feedback_queue",
                         "SELECT COUNT(*) FROM deferred_feedback_queue"),
        "belief_delete_audit_rows": _fb(ctx, "source LIKE 'user_deleted%'"),
    }, note="ambiguous: id gap also counts queue rows cascaded by belief deletes")


@_store_probe
def p_a008(ctx: Ctx) -> Result:
    return cc.from_shared_count(_fb(ctx, "source = 'lock:unlock'"),
                         note="ambiguous: aelf demote on a locked belief writes the same row")


@_store_probe
def p_a009(ctx: Ctx) -> Result:
    marks = ",".join("?" * len(_PROMOTION_OTHER))
    n = _fb(ctx, f"source LIKE 'promotion:%' AND source NOT IN ({marks})", _PROMOTION_OTHER)
    uv = _fb(ctx, "source = 'promotion:user_validated'")
    return cc.from_count(n, {
        "user_validated": uv, "custom_label": n - uv,
        "scope_rows": _fb(ctx, "source LIKE 'scope:%'"),
        "feed_promoted_rows": _feed_count(ctx, "wonder.promoted"),
    }, note="scope_rows shared with aelf demote --to-scope")


@_store_probe
def p_a011(ctx: Ctx) -> Result:
    rules: dict[str, int] = {}
    if ctx.table_exists("feedback_history"):
        for (src, n) in ctx.require_db().execute(
                "SELECT source, COUNT(*) FROM feedback_history "
                "WHERE source LIKE 'contradiction_tiebreaker:%' GROUP BY 1"):
            rule = str(src)[len(_TIEBREAK_PREFIX):]
            label = rule if rule in _RULES_CROSS or rule in _RULES_TIE else "other"
            rules[label] = rules.get(label, 0) + int(n)
    cross = sum(v for k, v in rules.items() if k in _RULES_CROSS)
    tie = sum(v for k, v in rules.items() if k in _RULES_TIE)
    detail = dict(rules)
    detail["by_class"] = cross
    detail["by_tiebreak"] = tie
    return cc.from_count(cross, detail, note="count = resolutions decided by precedence class")


_A012_ORIGINS = frozenset({"agent_inferred", "user_transcript", "user_stated",
                           "user_corrected", "user_validated", "unknown"})
# CHANGELOG/v0.md: correction detector in aelfrice.correction (#22).
CORRECTION_SHIP_DAY = "2026-04-26"


def _a012_override_ids(ctx: Ctx) -> set[str]:
    out: set[str] = set()
    if not (ctx.column_exists("ingest_log", "raw_meta")
            and ctx.column_exists("ingest_log", "derived_belief_ids")):
        return out
    for (raw,) in ctx.require_db().execute(
            "SELECT derived_belief_ids FROM ingest_log WHERE raw_meta LIKE ? OR raw_meta LIKE ?",
            ("%override_belief_type%", "%route_overrides%")):
        try:
            ids: Any = json.loads(raw) if raw else []
        except ValueError:
            continue
        if isinstance(ids, list):
            out.update(str(i) for i in ids)  # pyright: ignore[reportUnknownVariableType]
    return out


def _a012_legacy(ctx: Ctx) -> Result:
    detail = {"rows": ctx.count("SELECT COUNT(*) FROM beliefs")}
    if ctx.column_exists("beliefs", "created_at"):
        detail["rows_on_or_after_ship"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE substr(created_at,1,10) >= ?",
            (CORRECTION_SHIP_DAY,))
    if ctx.column_exists("beliefs", "belief_type"):
        detail["predecessor_belief_type_correction"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE belief_type = 'correction'")
    return cc.Result("untraceable", None, detail, "pre-aelfrice schema: no beliefs.type column")


@_store_probe
def p_a012(ctx: Ctx) -> Result:
    """Correction detector: type='correction' rows not set by an LLM override."""
    if not ctx.table_exists("beliefs"):
        return cc.silent(0, note="no beliefs table")
    if not ctx.column_exists("beliefs", "type"):
        return _a012_legacy(ctx)
    override_ids = _a012_override_ids(ctx)
    active_sql = "valid_to IS NULL" if ctx.column_exists("beliefs", "valid_to") else "1"
    origin_sql = "origin" if ctx.column_exists("beliefs", "origin") else "NULL"
    total = active = via_override = 0
    by_origin: dict[str, int] = {}
    for bid, origin, is_active in ctx.require_db().execute(
            f"SELECT id, {origin_sql}, {active_sql} FROM beliefs WHERE type = 'correction'"):
        if bid in override_ids:
            via_override += 1
            continue
        total += 1
        active += int(is_active or 0)
        label = origin if isinstance(origin, str) and origin in _A012_ORIGINS else "other"
        by_origin["origin_" + label] = by_origin.get("origin_" + label, 0) + 1
    return cc.from_count(total, {"active": active, "via_llm_override": via_override, **by_origin},
                         note="excludes override_belief_type and route_overrides rows")


@_store_probe
def p_a014(ctx: Ctx) -> Result:
    """Enqueue on retrieve: only rows enqueued on or after the 4.3.0 default-off flip."""
    if not ctx.table_exists("deferred_feedback_queue"):
        return cc.silent(0, note="no queue table")
    q = ("SELECT COUNT(*), "
         "SUM(CASE WHEN substr(enqueued_at,1,10) < ? THEN 1 ELSE 0 END), "
         "SUM(CASE WHEN substr(enqueued_at,1,10) >= ? THEN 1 ELSE 0 END), "
         "SUM(CASE WHEN substr(enqueued_at,1,10) >= ? THEN 1 ELSE 0 END), "
         "CAST(julianday(substr(MAX(enqueued_at),1,10)) - julianday(?) AS INTEGER), "
         "CAST(julianday(substr(MIN(enqueued_at),1,10)) - julianday(?) AS INTEGER) "
         "FROM deferred_feedback_queue WHERE event_type = 'retrieval_exposure'")
    total, pre, post, post_grace, max_d, min_d = ctx.counts(
        q, (ENQUEUE_FLIP_DAY, ENQUEUE_FLIP_DAY, ENQUEUE_FLIP_GRACE_DAY,
            ENQUEUE_FLIP_DAY, ENQUEUE_FLIP_DAY))
    detail = {"total": total, "pre_flip": pre, "post_flip": post,
              "post_flip_plus7d": post_grace}
    if total:
        detail["max_days_vs_flip"] = max_d
        detail["min_days_vs_flip"] = min_d
    if ctx.column_exists("beliefs", "created_at"):
        detail["newest_belief_days_vs_flip"] = _days_vs(ctx, "beliefs", "created_at",
                                                        ENQUEUE_FLIP_DAY)
    return cc.from_count(post, detail, note="count = rows enqueued on or after the flip day")


@_store_probe
def p_a016(ctx: Ctx) -> Result:
    pos = _fb(ctx, "source = 'user' AND valence > 0")
    neg = _fb(ctx, "source = 'user' AND valence < 0")
    return cc.from_count(pos + neg, {
        "used": pos, "harmful": neg,
        "feed_applied": _feed_count(ctx, "feedback.applied", "feedback"),
        "feed_locked_not_applied": _feed_count(ctx, "feedback.locked_not_applied", "feedback"),
    }, note="custom --source labels counted only via feed rows")


@_store_probe
def p_a017(ctx: Ctx) -> Result:
    if not ctx.column_exists("edges", "anchor_text"):
        return cc.silent(0, note="edges has no anchor_text")
    n = _edges(ctx, "type = 'CONTRADICTS' AND anchor_text IS NULL AND src < dst AND weight = 1.0")
    return cc.from_count(n, {
        "contradicts_total": _edges(ctx, "type = 'CONTRADICTS'"),
        "contradicts_with_anchor": _edges(ctx, "type = 'CONTRADICTS' AND anchor_text IS NOT NULL"),
    }, note="pre-anchor_text triple edges with src<dst would also match")


_A018_LOCK_COLS = ("lock_level", "locked_at", "alpha", "beta")
_A018_FLOOR_SRC = ("(f.source IN ('user', 'sentiment_inferred') "
                   "OR f.source LIKE 'propagation:%')")
_A018_JOIN = ("SELECT COUNT(*) FROM feedback_history f JOIN beliefs b ON b.id = f.belief_id "
              "WHERE b.lock_level = 'user' AND f.valence <> 0 ")
_A018_POST = "AND substr(f.created_at, 1, 19) >= substr(b.locked_at, 1, 19)"


@_store_probe
def p_a018(ctx: Ctx) -> Result:
    """Lock floor: passive feedback on a user lock dated at or after locked_at.

    Exposure rows (source 'hook') are detail only: since #1086 they call
    apply_feedback with update_posterior=False, so the floor is not what
    holds the posterior.
    """
    if not all(ctx.column_exists("beliefs", c) for c in _A018_LOCK_COLS):
        return cc.untraceable(note="pre-lock schema: no lock_level, locked_at or beta")
    floor_sql = _A018_JOIN + "AND " + _A018_FLOOR_SRC
    floor = _n(ctx, "feedback_history", floor_sql + " " + _A018_POST)
    return cc.from_count(floor, {
        "user_locks": _n(ctx, "beliefs",
                         "SELECT COUNT(*) FROM beliefs WHERE lock_level = 'user'"),
        "floor_rows_any_time": _n(ctx, "feedback_history", floor_sql),
        "floor_rows_post_4_3_0": _n(
            ctx, "feedback_history",
            floor_sql + " " + _A018_POST + " AND substr(f.created_at, 1, 10) >= ?",
            (LOCK_FLOOR_RELEASE_DAY,)),
        "exposure_rows_on_locks_post_lock": _n(
            ctx, "feedback_history", _A018_JOIN + "AND f.source = 'hook' " + _A018_POST),
        "locked_posterior_off_prior": _n(
            ctx, "beliefs", "SELECT COUNT(*) FROM beliefs WHERE lock_level = 'user' "
            "AND (alpha <> 9.0 OR beta <> 0.5)"),
    }, note="count = passive feedback rows on a lock dated after its locked_at")


@_store_probe
def p_a022(ctx: Ctx) -> Result:
    applied = _fb(ctx, "source = 'sentiment_inferred' AND valence < 0")
    disabled = _sentiment_rows(ctx, lambda r: r.get("sentiment") == "negative"
                               and r.get("abstained") == "negative_disabled")
    other = _sentiment_rows(ctx, lambda r: r.get("sentiment") == "negative"
                            and r.get("abstained") != "negative_disabled")
    return cc.from_count(applied + disabled, {
        "store_negative_rows": applied, "audit_negative_disabled": disabled,
        "audit_negative_other": other,
    }, note="count = applied negative rows + negative_disabled audit rows")


@_store_probe
def p_a023(ctx: Ctx) -> Result:
    return cc.from_count(_fb(ctx, "source = 'promotion:phantom_lock_match'"))


def _positive(r: dict[str, Any]) -> bool:
    return r.get("sentiment") == "positive"


@_store_probe
def p_a024(ctx: Ctx) -> Result:
    n = _fb(ctx, "source = 'sentiment_inferred' AND valence > 0")
    return cc.from_count(n, {
        "audit_positive_applied": _sentiment_rows(
            ctx, lambda r: _positive(r) and "abstained" not in r),
        "audit_no_prior_injection": _sentiment_rows(
            ctx, lambda r: _positive(r) and r.get("abstained") == "no_prior_injection"),
        "audit_candidates_gone": _sentiment_rows(
            ctx, lambda r: _positive(r) and r.get("abstained") == "candidates_gone"),
    })


@_store_probe
def p_a025(ctx: Ctx) -> Result:
    """POTENTIALLY_STALE edges. A store without edges.type cannot hold the trace."""
    if not ctx.table_exists("edges"):
        return cc.silent(0, note="no edges table")
    if not ctx.column_exists("edges", "type"):
        marker = 0
        if ctx.column_exists("edges", "edge_type"):
            marker = ctx.count(
                "SELECT COUNT(*) FROM edges WHERE UPPER(edge_type) LIKE '%STALE%'")
        return cc.silent(None, {"legacy_schema_stale_like": marker},
                         note="foreign legacy schema; aelfrice never writes it")
    n = _edges(ctx, "type = 'POTENTIALLY_STALE'")
    return cc.from_count(n, {"weight_1": _edges(ctx, "type = 'POTENTIALLY_STALE' AND weight = 1.0")})


# Edge types with a nonzero valence in feedback.EDGE_VALENCE.
_NONZERO_EDGE_TYPES = ("SUPPORTS", "CITES", "CONTRADICTS", "RELATES_TO", "DERIVED_FROM",
                       "IMPLEMENTS", "TEMPORAL_NEXT", "TESTS")
# Nonzero-valence rows that never reach the propagating branch: hook exposure
# (audit-only, #1086), sentiment (propagate=False, #1291) and the CLI
# delete, retire and restore audit rows, which call insert_feedback_event.
_NON_TRIGGER_SOURCES = ("hook", "sentiment_inferred", "user_deleted", "user_deleted_force",
                        "user_retired", "user_retired_force", "user_restored")


def _a026_triggers(ctx: Ctx) -> dict[str, int]:
    """Eligible propagation triggers after ship, and those with a live out-edge."""
    ph = ",".join("?" * len(_NON_TRIGGER_SOURCES))
    base = ("FROM feedback_history f WHERE f.valence != 0 "
            f"AND f.source NOT IN ({ph}) AND f.source NOT LIKE 'propagation:%' "
            "AND f.source NOT LIKE 'promotion:%' AND " + _after("f.created_at"))
    bparams = (*_NON_TRIGGER_SOURCES, PROPAGATION_SHIP_DAY, PROPAGATION_SHIP_EPOCH)
    out = {"nonzero_explicit_after_ship": ctx.count("SELECT COUNT(*) " + base, bparams)}
    if not (_typed_edges(ctx) and ctx.table_exists("beliefs")):
        return out
    tph = ",".join("?" * len(_NONZERO_EDGE_TYPES))
    lock_ok = ""
    if ctx.column_exists("beliefs", "lock_level"):
        # aelf feedback on a user lock moves nothing; aelf confirm does.
        lock_ok = (" AND NOT (COALESCE((SELECT lock_level FROM beliefs b "
                   "WHERE b.id = f.belief_id), 'none') = 'user' "
                   "AND f.source != 'user_confirmed')")
    out["eligible_triggers_with_edges"] = ctx.count(
        "SELECT COUNT(*) " + base + lock_ok +
        " AND EXISTS (SELECT 1 FROM edges e JOIN beliefs d ON d.id = e.dst "
        f"WHERE e.src = f.belief_id AND e.type IN ({tph}))",
        (*bparams, *_NONZERO_EDGE_TYPES))
    out["nonzero_valence_edges"] = ctx.count(
        f"SELECT COUNT(*) FROM edges WHERE type IN ({tph})", _NONZERO_EDGE_TYPES)
    return out


@_store_probe
def p_a026(ctx: Ctx) -> Result:
    """Propagation rows by magnitude, plus the trigger rows that could have walked."""
    if not ctx.table_exists("feedback_history"):
        return cc.silent(0, note="no feedback_history")
    ge05, mid, lt005 = ctx.counts(
        "SELECT SUM(ABS(valence) >= 0.5), SUM(ABS(valence) < 0.5 AND ABS(valence) >= 0.05), "
        "SUM(ABS(valence) < 0.05) FROM feedback_history WHERE source LIKE 'propagation:%'")
    prop = ge05 + mid + lt005
    prop_after = ctx.count(
        "SELECT COUNT(*) FROM feedback_history f WHERE f.source LIKE 'propagation:%' AND "
        + _after("f.created_at"), (PROPAGATION_SHIP_DAY, PROPAGATION_SHIP_EPOCH))
    detail = {"abs_ge_0_5": ge05, "abs_0_05_to_0_5": mid, "abs_lt_0_05": lt005,
              "propagation_rows_after_ship": prop_after, **_a026_triggers(ctx)}
    note = "same rows as A037; abs_lt_0_05 should be 0 under min 0.05"
    if not _typed_edges(ctx):
        note = "edges not current schema"
    return cc.from_count(prop, detail, note=note)


@_store_probe
def p_a027(ctx: Ctx) -> Result:
    if not ctx.column_exists("edges", "anchor_text"):
        return cc.silent(0, note="edges has no anchor_text")
    det_where = "type = 'CONTRADICTS' AND anchor_text IS NULL AND src < dst AND weight = 1.0"
    det = _edges(ctx, det_where)
    stale = _edges(ctx, "type = 'POTENTIALLY_STALE'")
    max_deg, at_cap = ctx.counts(
        "SELECT COALESCE(MAX(n), 0), COALESCE(SUM(n >= 8), 0) FROM ("
        f"SELECT x, COUNT(*) n FROM (SELECT src x FROM edges WHERE {det_where} "
        f"UNION ALL SELECT dst x FROM edges WHERE {det_where}) GROUP BY x)")
    return cc.from_count(det + stale, {
        "detector_contradicts": det, "potentially_stale": stale,
        "max_contradicts_per_belief": max_deg, "beliefs_at_cap_8": at_cap,
    })


@_store_probe
def p_a028(ctx: Ctx) -> Result:
    """Relevance sweeper (removed): only rows scored on or after its removal fire."""
    if not ctx.column_exists("injection_events", "referenced"):
        return cc.silent(0, note="no referenced column")
    d = SWEEPER_REMOVAL_DAY
    (total, scored, ref1, scored_no_ts, pre, post, inj_post, inj_post_scored,
     max_ref_d, max_inj_d) = ctx.counts(
        "SELECT COUNT(*), "
        "SUM(referenced IS NOT NULL), SUM(referenced = 1), "
        "SUM(referenced IS NOT NULL AND referenced_at IS NULL), "
        "SUM(referenced IS NOT NULL AND substr(referenced_at,1,10) < ?), "
        "SUM(referenced IS NOT NULL AND substr(referenced_at,1,10) >= ?), "
        "SUM(substr(injected_at,1,10) >= ?), "
        "SUM(substr(injected_at,1,10) >= ? AND referenced IS NOT NULL), "
        "CAST(julianday(substr(MAX(referenced_at),1,10)) - julianday(?) AS INTEGER), "
        "CAST(julianday(substr(MAX(injected_at),1,10)) - julianday(?) AS INTEGER) "
        "FROM injection_events", (d,) * 6)
    detail = {"rows": total, "scored": scored, "referenced_1": ref1,
              "scored_no_ts": scored_no_ts, "scored_pre_removal": pre,
              "scored_post_removal": post, "injected_post_removal": inj_post,
              "injected_post_removal_scored": inj_post_scored}
    if scored:
        detail["newest_scored_days_vs_removal"] = max_ref_d
    if total:
        detail["newest_injected_days_vs_removal"] = max_inj_d
    return cc.from_count(post + scored_no_ts, detail,
                         note="fired = scored on or after removal day")


def _cols(ctx: Ctx, table: str) -> set[str]:
    if not ctx.table_exists(table):
        return set()
    return {str(r[1]) for r in ctx.require_db().execute(f"PRAGMA table_info({table})")}


@_store_probe
def p_a029(ctx: Ctx) -> Result:
    """RESOLVES edges, with tie-breaker rows and edge counts as context."""
    ecols = _cols(ctx, "edges")
    if not ecols:
        return cc.silent(0, note="no edges table")
    rel = RESOLVES_RELEASE_DAY
    bcols = _cols(ctx, "beliefs")
    if "type" not in ecols and "edge_type" in ecols:
        post = 0
        if "created_at" in bcols:
            post = ctx.count("SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (rel,))
        return cc.Result("untraceable", None, {
            "legacy_edge_type_resolves": ctx.count(
                "SELECT COUNT(*) FROM edges WHERE edge_type = 'RESOLVES'"),
            "beliefs_created_after_5_2_0": post,
        }, "legacy pre-rewrite schema; edges has edge_type not type")
    d = {
        "weight_0": _edges(ctx, "type = 'RESOLVES' AND weight = 0.0"),
        "supersedes": _edges(ctx, "type = 'SUPERSEDES'"),
        "contradicts": _edges(ctx, "type = 'CONTRADICTS'"),
        "tiebreak_rows_total": _fb(ctx, "source LIKE 'contradiction_tiebreaker:%'"),
        "tiebreak_rows_after_5_2_0": _fb(
            ctx, "source LIKE 'contradiction_tiebreaker:%' AND created_at >= ?", (rel,)),
    }
    if "created_at" in bcols:
        d["beliefs_created_after_5_2_0"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (rel,))
    return cc.from_count(_edges(ctx, "type = 'RESOLVES'"), d)


_A030_STEP = 0.1
_A030_EPS = 1e-6


def _a030_baselines(ctx: Ctx) -> dict[tuple[Any, Any], tuple[float, float]]:
    """Modal (alpha, beta) per (type, origin) among beliefs with no feedback row."""
    tally: dict[tuple[Any, Any], Counter[tuple[float, float]]] = {}
    for t, o, a, be in ctx.require_db().execute(
            "SELECT b.type, b.origin, b.alpha, b.beta FROM beliefs b WHERE NOT EXISTS "
            "(SELECT 1 FROM feedback_history f WHERE f.belief_id = b.id)"):
        tally.setdefault((t, o), Counter())[(round(float(a), 6), round(float(be), 6))] += 1
    return {k: c.most_common(1)[0][0] for k, c in tally.items()}


def _a030_classify(ctx: Ctx, pre: str, post: str) -> Counter[str]:
    base = _a030_baselines(ctx)
    c: Counter[str] = Counter()
    for t, o, a, be, kpre, kpost in ctx.require_db().execute(
            "SELECT b.type, b.origin, b.alpha, b.beta, "
            f"SUM(CASE WHEN {pre.replace('created_at', 'f.created_at')} THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN {post.replace('created_at', 'f.created_at')} THEN 1 ELSE 0 END) "
            "FROM beliefs b JOIN feedback_history f ON f.belief_id = b.id "
            "GROUP BY b.id HAVING SUM(CASE WHEN f.source <> 'hook' THEN 1 ELSE 0 END) = 0"):
        c["hook_only_beliefs"] += 1
        if kpost:
            c["hook_only_with_post_rows"] += 1
        b0 = base.get((t, o))
        if b0 is None:
            c["no_baseline"] += 1
            continue
        da, dbeta = float(a) - b0[0], float(be) - b0[1]
        if abs(dbeta) > _A030_EPS:
            c["beta_off_baseline"] += 1
        elif abs(da) < _A030_EPS:
            c["at_prior"] += 1
        elif kpost and abs(da - _A030_STEP * (kpre + kpost)) < _A030_EPS:
            c["bumped_incl_post_rows"] += 1
        elif kpre and abs(da - _A030_STEP * kpre) < _A030_EPS:
            c["bumped_pre_rows_only"] += 1
        else:
            c["alpha_off_other"] += 1
    return c


@_store_probe
def p_a030(ctx: Ctx) -> Result:
    """Legacy exposure-updates-posterior mode, from its two traces after 4.0.0.

    Hook rows are written in both modes, so they measure exposure, not the
    knob. AELFRICE_EXPOSURE_UPDATES_POSTERIOR=1 bumps alpha by 0.1 per hook
    row on a hook-only belief and writes 'propagation:hook' rows. Before
    the release day the bump was the default, so only later rows count.
    """
    if not ctx.table_exists("feedback_history"):
        return cc.silent(0, note="no feedback_history")
    if not all(ctx.column_exists("beliefs", c) for c in ("type", "origin", "alpha", "beta")):
        return cc.untraceable(note="pre-origin schema: no per-origin prior baseline")
    pre = f"substr(created_at, 1, 10) < '{EXPOSURE_RELEASE_DAY}'"
    post = f"substr(created_at, 1, 10) >= '{EXPOSURE_RELEASE_DAY}'"
    hook_post = _fb(ctx, f"source = 'hook' AND {post}")
    prop_post = _fb(ctx, f"source = 'propagation:hook' AND {post}")
    c = _a030_classify(ctx, pre, post)
    detail = {
        "hook_rows_pre_4_0_0": _fb(ctx, f"source = 'hook' AND {pre}"),
        "hook_rows_post_4_0_0": hook_post,
        "prop_hook_pre": _fb(ctx, f"source = 'propagation:hook' AND {pre}"),
        "prop_hook_post": prop_post,
        **{k: int(v) for k, v in c.items()},
    }
    if hook_post == 0 and prop_post == 0:
        return cc.silent(0, detail, note="no post-4.0.0 exposure rows; mode unobservable")
    return cc.from_count(c["bumped_incl_post_rows"] + prop_post, detail,
                         note="count = legacy-mode bumps on post-4.0.0 rows + post-4.0.0 prop rows")


@_store_probe
def p_a031(ctx: Ctx) -> Result:
    if not ctx.table_exists("belief_support_evidence"):
        return cc.silent(0)
    n, applied, supporters = ctx.counts(
        "SELECT COUNT(*), SUM(applied), COUNT(DISTINCT supporter_id) "
        "FROM belief_support_evidence WHERE kind = 'reworded_restatement'")
    return cc.from_count(n, {"applied": applied, "distinct_supporters": supporters})


@_store_probe
def p_a033(ctx: Ctx) -> Result:
    if not ctx.table_exists("feedback_history"):
        return cc.silent(0)
    n, fires = ctx.counts(
        "SELECT COUNT(*), COUNT(DISTINCT created_at) FROM feedback_history "
        "WHERE source = 'sentiment_inferred'")
    fractional = _fb(ctx, "source = 'sentiment_inferred' AND ABS(valence) NOT IN (1.0, 1.5, 2.0)")
    one_unit = _n(ctx, "feedback_history",
                  "SELECT COUNT(*) FROM (SELECT ABS(SUM(valence)) s FROM feedback_history "
                  "WHERE source = 'sentiment_inferred' GROUP BY created_at) "
                  "WHERE ABS(s - 1.0) < 0.001 OR ABS(s - 1.5) < 0.001")
    prop = _fb(ctx, "source = 'propagation:sentiment_inferred'")
    return cc.from_count(n, {"fires_by_timestamp": fires, "groups_summing_one_unit": one_unit,
                             "rows_fractional_share": fractional, "propagation_rows": prop},
                         note="rows shared with A022 A024; split shape in detail")


@_store_probe
def p_a034(ctx: Ctx) -> Result:
    strong = _sentiment_rows(ctx, lambda r: r.get("confidence") == 0.9)
    base = _sentiment_rows(ctx, lambda r: r.get("confidence") == 0.6)
    other = _sentiment_rows(ctx, lambda r: r.get("confidence") not in (0.9, 0.6))
    return cc.from_count(strong + base + other, {
        "strong_1_5": strong, "base_1_0": base, "other_tier": other,
        "store_rows_abs_1_5": _fb(ctx, "source = 'sentiment_inferred' AND ABS(valence) = 1.5"),
        "store_rows_abs_1_0": _fb(ctx, "source = 'sentiment_inferred' AND ABS(valence) = 1.0"),
    }, note="tier read from hook_audit; store shares are split by pack size")


_A037_TRIGGER_SOURCES = ("user", "user_confirmed")
_A037_ZERO_VALENCE = ("supersedes", "resolves")


def _a037_edge_cols(ctx: Ctx) -> tuple[str, str]:
    """(type column, source column) for the edges schema, or ('', '')."""
    if ctx.column_exists("edges", "type"):
        return "type", "src"
    if ctx.column_exists("edges", "edge_type"):
        return "edge_type", "from_id"
    return "", ""


def _a037_triggers(ctx: Ctx, ecol: str, esrc: str) -> dict[str, int]:
    ph = ",".join("?" * len(_A037_TRIGGER_SOURCES))
    tot, post = ctx.counts(
        "SELECT COUNT(*), SUM(CASE WHEN substr(created_at,1,10) >= ? THEN 1 ELSE 0 END) "
        f"FROM feedback_history WHERE source IN ({ph})",
        (PROPAGATION_SHIP_DAY, *_A037_TRIGGER_SOURCES))
    out = {"trig_total": tot, "trig_post_ship": post}
    if ecol and post:
        zph = ",".join("?" * len(_A037_ZERO_VALENCE))
        out["trig_post_ship_with_out_edge"] = ctx.count(
            f"SELECT COUNT(*) FROM feedback_history f WHERE f.source IN ({ph}) "
            "AND substr(f.created_at,1,10) >= ? AND EXISTS (SELECT 1 FROM edges e "
            f"WHERE e.{esrc} = f.belief_id AND lower(e.{ecol}) NOT IN ({zph}))",
            (*_A037_TRIGGER_SOURCES, PROPAGATION_SHIP_DAY, *_A037_ZERO_VALENCE))
    return out


@_store_probe
def p_a037(ctx: Ctx) -> Result:
    """Propagation rows, with the trigger rows (aelf feedback, aelf confirm) beside them.

    Sentiment passes propagate=False and hook exposure defaults to
    update_posterior=False, so a store with no trigger row is silent by
    absence of input.
    """
    detail: dict[str, int] = {}
    ecol, esrc = _a037_edge_cols(ctx)
    if ecol:
        e = ctx.group_count(f"SELECT lower({ecol}), COUNT(*) FROM edges GROUP BY 1",
                            _EDGE_TYPES_LOWER)
        detail.update({"edges_" + k: v for k, v in e.items()})
    if ctx.table_exists("beliefs"):
        detail["beliefs"] = ctx.count("SELECT COUNT(*) FROM beliefs")
    if not ctx.table_exists("feedback_history"):
        return cc.silent(0, detail, note="no feedback_history table")
    detail["fh_rows"] = ctx.count("SELECT COUNT(*) FROM feedback_history")
    prop = _fb(ctx, "source LIKE 'propagation:%'")
    g = ctx.group_count(
        "SELECT source, COUNT(*) FROM feedback_history WHERE source LIKE 'propagation:%' "
        "GROUP BY 1", {"propagation:user", "propagation:user_confirmed", "propagation:hook"})
    detail.update({"src_" + k.replace("propagation:", ""): v for k, v in g.items()})
    trig = _a037_triggers(ctx, ecol, esrc)
    detail.update(trig)
    if ctx.column_exists("beliefs", "created_at"):
        detail["newest_belief_days_vs_ship"] = _days_vs(ctx, "beliefs", "created_at",
                                                        PROPAGATION_SHIP_DAY)
    note = "" if prop or trig["trig_post_ship"] else "no trigger rows after ship"
    return cc.from_count(prop, detail, note=note)


# --- registry --------------------------------------------------------------

P = cc.Probe
U = cc.untraceable_probe

PROBES: list[cc.Probe] = [
    U("A001", "aelf cadence-score",
      "read-only report over cadence_shadow rows; writes no file or row"),
    P("A002", "aelf confirm (lock-floor exemption)", "store", p_a002),
    P("A003", "aelf demote", "store", p_a003),
    U("A004", "aelf doctor relationships audit",
      "read-only audit; prints a report and writes nothing"),
    P("A005", "aelf resolve contradiction tie-breaker", "store", p_a005),
    P("A006", "aelf sweep-feedback --gc", "store", p_a006),
    U("A007", "aelf sweep-feedback audit",
      "audit-only since #1162; SweepResult.mutated is False and nothing is written"),
    P("A008", "aelf unlock", "store", p_a008),
    P("A009", "aelf validate / promote", "store", p_a009),
    U("A010", "Consolidation cluster report",
      "read-only audit; no edges, retirements or log rows"),
    P("A011", "Contradiction precedence classes", "store", p_a011),
    P("A012", "Correction detector sets type=correction", "store", p_a012),
    U("A013", "Correction-frequency escalator",
      "no caller; escalated shares would be indistinguishable from base shares"),
    P("A014", "Deferred-feedback enqueue on retrieve", "store", p_a014),
    U("A015", "Detector-threshold manifest and version guard",
      "constants module and CI guard; edges carry no version field"),
    P("A016", "Explicit feedback (aelf feedback used|harmful)", "store", p_a016),
    P("A017", "Ingest-time CONTRADICTS writer", "store", p_a017),
    P("A018", "Lock floor on passive feedback", "store", p_a018),
    U("A019", "Lock near-duplicate warning on aelf lock",
      "warning goes to stdout only; no row or log"),
    U("A020", "Near-duplicate audit (aelf doctor dedup)",
      "read-only audit; prints clusters and writes nothing"),
    U("A021", "Negation hyphen bound in correction detector",
      "acts by not matching; no row records a suppressed signal"),
    P("A022", "Negative sentiment feedback and abstain audit", "store", p_a022),
    P("A023", "Phantom lock-match promotion on aelf lock", "store", p_a023),
    P("A024", "Positive sentiment-from-prose feedback", "store", p_a024),
    P("A025", "POTENTIALLY_STALE writer", "store", p_a025),
    P("A026", "Propagation walk parameters and EDGE_VALENCE", "store", p_a026),
    P("A027", "Relationship detector thresholds and write cap", "store", p_a027),
    P("A028", "Relevance (reference) detector, removed", "store", p_a028),
    P("A029", "RESOLVES edge writer", "store", p_a029),
    P("A030", "Retrieval exposure is audit-only", "store", p_a030),
    P("A031", "Reworded matcher thresholds and vetoes", "store", p_a031),
    U("A032", "Sentiment length guard and harness-prefix filter",
      "guard returns None before any audit row; leaves no trace"),
    P("A033", "Sentiment equal split, no propagation, lock-aware divisor", "store", p_a033),
    P("A034", "Sentiment pattern tiers and magnitudes", "sidecar", p_a034),
    U("A035", "Slot-conflict flag on aelf search",
      "marker printed in search output only; no row or log"),
    U("A036", "Typed-slot value-comparison contradiction gate",
      "no production caller passes use_value_comparison=True; edges match detector edges"),
    P("A037", "Valence propagation through edges", "store", p_a037),
]

TRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind != "untraceable")
UNTRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind == "untraceable")
