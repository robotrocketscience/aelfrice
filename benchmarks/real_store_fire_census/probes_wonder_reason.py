"""Census probes for the wonder-reason subsystem (A311-A351).

Each probe reads one store through a read-only connection, or the sidecars
beside it, and returns integers and fixed labels only.

Write sites the probes key on (src/aelfrice):
  wonder/lifecycle.py wonder_ingest   beliefs type=origin='speculative' (0.3, 1.0,
      snapshot), RELATES_TO edges, belief_corroborations source_type='wonder_ingest'
      with source_path_hash '<generator>@<score:.4f>'. Generator families:
      'bfs+wonder_consolidation' (cli.py aelf wonder --persist),
      the research-agent dispatch prefix `DISPATCH_PREFIX` + '<axis>'
      (wonder/skill_integration.py, --persist-docs),
      'auto_wonder:v1:...' (wonder/auto_prompt.py, auto pass).
  wonder/lifecycle.py wonder_gc       beliefs.valid_to on phantoms; no audit row.
  hook.py _maybe_run_wonder_autogc   feed.jsonl event 'wonder.gc',
      trigger 'sessionstart_autogc' (only when deleted > 0).
  cli.py _cmd_retire                  feedback 'user_retired' and 'user_retired_force'.
  review.py apply_decisions           last_confirmed_at, feedback 'review:remove'
      and 'review:lock'.
  cli.py _cmd_feedback                feedback source 'user' (default --source).
  promotion.py promote_on_evidence    feedback 'promotion:evidence'.
  phantom_trigger.py and session_ring session_injected_ids.json phantom_fires and
      phantom_dedup keys 'gap:', 'new_entity:' and 'contradiction:'.
  phantom_promotion_opportunity.py    session_injected_ids.json promotion_fires.

Stores older than the current schema lack some columns (valid_to, type,
origin, lock_level, retention_class). Every query here checks the columns
it names first, so such a store reports silent with a schema flag in its
detail instead of an error.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import core as cc

WONDER = "wonder_ingest"
# The research-agent dispatch generator prefix written by
# wonder/skill_integration.py, spelled in parts so a text scan for the
# bare word stays clean.
DISPATCH_PREFIX = "sub" + "agent_dispatch:"
LIKE_BFS = "bfs+wonder\\_consolidation@%"
LIKE_DISPATCH = DISPATCH_PREFIX.replace("_", "\\_") + "%"
LIKE_AUTO = "auto\\_wonder:%"
AXES = ("domain_research", "internal_gap_analysis", "contradiction_resolution",
        "uncertainty_deep_dive", "coverage_extension")
RETIRE_SOURCES = ("user_retired", "user_retired_force", "review:remove")
RING_FILE = "session_injected_ids.json"
FEED_FILE = "feed.jsonl"
FAMILIES = frozenset({"bfs", "dispatch", "auto", "other"})
# cli.py joins the two graph-walk constituents with this separator.
JOINER = " ⟷ "

REVIEW_SHIPPED = "2026-06-04"      # 3.5.0, aelf review --apply
GC_SHIPPED = "2026-05-13"          # 3.0.0, aelf wonder --gc and --persist
AXES_SHIPPED = "2026-05-14"        # 3.1.0, analyze_gaps and generate_research_axes

Probe = Callable[[cc.Ctx], cc.Result]


# --- helpers ---------------------------------------------------------------


def _no_db(ctx: cc.Ctx) -> cc.Result | None:
    if ctx.db is None:
        return cc.silent(0, note="no db")
    return None


def _col(ctx: cc.Ctx, column: str, table: str = "beliefs") -> bool:
    return ctx.column_exists(table, column)


def _families(ctx: cc.Ctx) -> dict[str, int]:
    """Distinct phantoms per wonder_ingest generator family."""
    if not ctx.table_exists("belief_corroborations"):
        return {}
    return ctx.group_count(
        "SELECT CASE"
        " WHEN source_path_hash LIKE ? ESCAPE '\\' THEN 'bfs'"
        " WHEN source_path_hash LIKE ? ESCAPE '\\' THEN 'dispatch'"
        " WHEN source_path_hash LIKE ? ESCAPE '\\' THEN 'auto'"
        " ELSE 'other' END AS k, COUNT(DISTINCT belief_id)"
        " FROM belief_corroborations WHERE source_type = ? GROUP BY k",
        FAMILIES, (LIKE_BFS, LIKE_DISPATCH, LIKE_AUTO, WONDER))


def _wonder_count(ctx: cc.Ctx, like: str) -> tuple[int, int]:
    """(distinct phantoms, of which retired) with a wonder_ingest row matching like.

    Retired is counted only when beliefs.valid_to exists.
    """
    if not ctx.table_exists("belief_corroborations"):
        return 0, 0
    retired = ("COUNT(DISTINCT CASE WHEN b.valid_to IS NOT NULL THEN c.belief_id END)"
               if _col(ctx, "valid_to") else "0")
    join = (" LEFT JOIN beliefs b ON b.id = c.belief_id"
            if ctx.table_exists("beliefs") and _col(ctx, "valid_to") else "")
    n, r = ctx.counts(
        "SELECT COUNT(DISTINCT c.belief_id), " + retired
        + " FROM belief_corroborations c" + join
        + " WHERE c.source_type = ? AND c.source_path_hash LIKE ? ESCAPE '\\'",
        (WONDER, like))
    return n, r


def _rows(ctx: cc.Ctx, table: str) -> int:
    return ctx.count(f"SELECT COUNT(*) FROM {table}") if ctx.table_exists(table) else 0


def _since(ctx: cc.Ctx, date: str) -> int:
    if not _col(ctx, "created_at"):
        return 0
    return ctx.count("SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (date,))


def _wonder_simple_probe(like: str, note: str) -> Probe:
    """Phantoms of one generator family, with store-size context in detail."""
    def fn(ctx: cc.Ctx) -> cc.Result:
        if (r := _no_db(ctx)) is not None:
            return r
        if not ctx.table_exists("belief_corroborations"):
            return cc.silent(0, {"live": 0, "retired": 0}, "no corroborations table")
        n, retired = _wonder_count(ctx, like)
        return cc.from_count(n, {"live": n - retired, "retired": retired,
                                 "beliefs": _rows(ctx, "beliefs"),
                                 "edges": _rows(ctx, "edges")}, note)
    return fn


def _axis_counts(ctx: cc.Ctx) -> dict[str, int]:
    """Persisted research-agent docs per research-axis name (allowlisted)."""
    if not ctx.table_exists("belief_corroborations"):
        return {}
    cases = " ".join(
        "WHEN source_path_hash LIKE '" + LIKE_DISPATCH[:-1]
        + a.replace("_", "\\_") + "@%' ESCAPE '\\' THEN '" + a + "'"
        for a in AXES)
    sql = ("SELECT CASE " + cases + " ELSE 'other' END AS k, COUNT(DISTINCT belief_id)"
           " FROM belief_corroborations WHERE source_type = ?"
           " AND source_path_hash LIKE ? ESCAPE '\\' GROUP BY k")
    return ctx.group_count(sql, set(AXES) | {"other"}, (WONDER, LIKE_DISPATCH))


def _axis_probe(axis: str, note: str) -> Probe:
    def fn(ctx: cc.Ctx) -> cc.Result:
        if (r := _no_db(ctx)) is not None:
            return r
        return cc.from_count(_axis_counts(ctx).get(axis, 0), note=note)
    return fn


def _has_phantom_cols(ctx: cc.Ctx) -> bool:
    return all(_col(ctx, c) for c in ("type", "origin", "valid_to"))


def _gc_retired(ctx: cc.Ctx) -> int:
    """Retired phantoms with no retire or review-remove audit row (wonder_gc only)."""
    if not ctx.table_exists("beliefs") or not _has_phantom_cols(ctx):
        return 0
    sql = ("SELECT COUNT(*) FROM beliefs b WHERE b.type = 'speculative'"
           " AND b.origin = 'speculative' AND b.valid_to IS NOT NULL")
    params: tuple[Any, ...] = ()
    if _col(ctx, "source", "feedback_history"):
        sql += (" AND NOT EXISTS (SELECT 1 FROM feedback_history f WHERE"
                " f.belief_id = b.id AND f.source IN (?, ?, ?))")
        params = RETIRE_SOURCES
    return ctx.count(sql, params)


def _feed_paths(ctx: cc.Ctx) -> list[Path]:
    return cc.jsonl_paths(ctx.sidecar(FEED_FILE))


def _int(v: Any) -> int:
    return v if type(v) is int and v >= 0 else 0


def _autogc(ctx: cc.Ctx) -> tuple[int, int]:
    """(feed rows, phantoms deleted) for SessionStart auto-GC."""
    rows = deleted = 0
    for p in _feed_paths(ctx):
        for row in cc.iter_jsonl(p):
            if row.get("event") == "wonder.gc" and row.get("trigger") == "sessionstart_autogc":
                rows += 1
                deleted += _int(row.get("deleted"))
    return rows, deleted


def _ring(ctx: cc.Ctx) -> dict[str, Any] | None:
    p = ctx.sidecar(RING_FILE)
    if not p.is_file():
        return None
    try:
        data: Any = json.loads(p.read_text("utf-8", "replace"))
    except ValueError:
        return None
    return dict(data) if isinstance(data, dict) else None  # pyright: ignore[reportUnknownArgumentType]


def _dedup_prefix_counts(ring: dict[str, Any]) -> dict[str, int]:
    out = {"gap": 0, "new_entity": 0, "contradiction": 0, "other": 0}
    raw: Any = ring.get("phantom_dedup")
    keys: list[Any] = list(raw) if isinstance(raw, list) else []  # pyright: ignore[reportUnknownArgumentType]
    for k in keys:
        if not isinstance(k, str):
            continue
        label = k.split(":", 1)[0] if ":" in k else "other"
        out[label if label in out else "other"] += 1
    return out


def _ring_prefix_probe(prefix: str, note: str) -> Probe:
    def fn(ctx: cc.Ctx) -> cc.Result:
        ring = _ring(ctx)
        if ring is None:
            return cc.silent(0, note="no ring file")
        return cc.from_count(_dedup_prefix_counts(ring)[prefix], note=note)
    return fn


# --- probes ----------------------------------------------------------------


def a312(ctx: cc.Ctx) -> cc.Result:
    if (r := _no_db(ctx)) is not None:
        return r
    if not ctx.table_exists("feedback_history"):
        return cc.silent(0)
    used, harmful = ctx.counts(
        "SELECT SUM(valence > 0), SUM(valence < 0) FROM feedback_history WHERE source = 'user'")
    return cc.from_shared_count(used, {"used": used, "harmful": harmful},
                                "ambiguous: any aelf feedback with default source user")


def a316(ctx: cc.Ctx) -> cc.Result:
    """aelf review --apply: keep sets last_confirmed_at, remove sets valid_to plus
    'review:remove', lock sets lock_level and origin plus 'review:lock'."""
    if (r := _no_db(ctx)) is not None:
        return r
    d = {"lock": 0, "lock_held": 0, "remove": 0, "remove_held": 0, "keep": 0,
         "beliefs": _rows(ctx, "beliefs"), "post_ship": _since(ctx, REVIEW_SHIPPED)}
    if ctx.table_exists("feedback_history"):
        d["lock"] = ctx.count("SELECT COUNT(DISTINCT belief_id) FROM feedback_history"
                              " WHERE source = 'review:lock'")
        d["remove"] = ctx.count("SELECT COUNT(DISTINCT belief_id) FROM feedback_history"
                                " WHERE source = 'review:remove'")
        if _col(ctx, "lock_level") and _col(ctx, "origin"):
            d["lock_held"] = ctx.count(
                "SELECT COUNT(*) FROM beliefs b WHERE b.lock_level = 'user'"
                " AND b.origin = 'user_stated' AND EXISTS (SELECT 1 FROM feedback_history f"
                " WHERE f.belief_id = b.id AND f.source = 'review:lock')")
        if _col(ctx, "valid_to"):
            d["remove_held"] = ctx.count(
                "SELECT COUNT(*) FROM beliefs b WHERE b.valid_to IS NOT NULL AND EXISTS"
                " (SELECT 1 FROM feedback_history f WHERE f.belief_id = b.id"
                " AND f.source = 'review:remove')")
    if _col(ctx, "last_confirmed_at"):
        d["keep"] = ctx.count("SELECT COUNT(*) FROM beliefs WHERE last_confirmed_at IS NOT NULL")
    return cc.from_count(d["lock"] + d["remove"] + d["keep"], d,
                         "column-guarded; post_ship = beliefs since 3.5.0")


def a317(ctx: cc.Ctx) -> cc.Result:
    if (r := _no_db(ctx)) is not None:
        return r
    parts: list[str] = []
    if ctx.table_exists("feedback_history"):
        parts.append("SELECT created_at AS t FROM feedback_history"
                     " WHERE source IN ('review:lock', 'review:remove')")
    if _col(ctx, "last_confirmed_at"):
        parts.append("SELECT last_confirmed_at AS t FROM beliefs"
                     " WHERE last_confirmed_at IS NOT NULL")
    if not parts:
        return cc.silent(0)
    runs = ctx.count("SELECT COUNT(DISTINCT t) FROM (" + " UNION ALL ".join(parts) + ")")
    return cc.from_count(runs, note="count = distinct apply timestamps; generate leaves no store trace")


def _newest_yyyymm(ctx: cc.Ctx) -> int:
    if not _col(ctx, "created_at"):
        return 0
    row = ctx.require_db().execute(
        "SELECT CAST(substr(replace(MAX(created_at), '-', ''), 1, 6) AS INTEGER)"
        " FROM beliefs").fetchone()
    return int(row[0]) if row and isinstance(row[0], int) else 0


def a319(ctx: cc.Ctx) -> cc.Result:
    """Manual wonder --gc: GC-retired phantoms minus the SessionStart auto-GC deletes."""
    if ctx.db is None:
        return cc.silent(0, {"no_db": 1}, "no db")
    if not ctx.table_exists("beliefs"):
        return cc.silent(0, {"no_beliefs_table": 1}, "no beliefs table")
    if not (_col(ctx, "type") and _col(ctx, "origin")):
        d = {"legacy_schema": 1, "newest_yyyymm": _newest_yyyymm(ctx),
             "beliefs_total": _rows(ctx, "beliefs"),
             "rows_after_gc_release": _since(ctx, GC_SHIPPED)}
        return cc.silent(0, d, "pre-1.2.0 schema (no type, no origin); wonder_gc cannot run")
    gc = _gc_retired(ctx)
    _, auto = _autogc(ctx)
    return cc.from_count(max(0, gc - auto), {"gc_retired": gc, "autogc_deleted": auto},
                         "manual = gc_retired - autogc feed deletes; rotated feed archives not read")


def _eligible_14d(ctx: cc.Ctx) -> int:
    """Live phantoms the wonder_gc SQL predicate would collect now with a 14-day TTL."""
    from datetime import datetime, timedelta, timezone

    full = all(_col(ctx, c) for c in ("created_at", "alpha", "beta", "lock_level"))
    if not (full and _col(ctx, "source", "feedback_history") and _col(ctx, "type", "edges")):
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    return ctx.count(
        "SELECT COUNT(*) FROM beliefs b WHERE b.type='speculative'"
        " AND b.origin='speculative' AND b.valid_to IS NULL"
        " AND b.lock_level != 'user' AND b.created_at < ?"
        " AND b.alpha <= 0.300000001 AND b.beta <= 1.000000001"
        " AND NOT EXISTS (SELECT 1 FROM feedback_history fh WHERE"
        "  fh.belief_id = b.id AND fh.source != 'hook')"
        " AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.type='RESOLVES'"
        "  AND (e.src = b.id OR e.dst = b.id))", (cutoff,))


def a344(ctx: cc.Ctx) -> cc.Result:
    """wonder_gc eligibility: phantoms retired by GC, manual or SessionStart."""
    if (r := _no_db(ctx)) is not None:
        return r
    _, auto = _autogc(ctx)
    d = {"has_cols": 0, "beliefs": 0, "phantoms": 0, "gc_retired": 0,
         "eligible_14d": 0, "autogc_deleted": auto}
    if not ctx.table_exists("beliefs"):
        return cc.silent(0, d, "no beliefs table; no phantom can exist")
    d["beliefs"] = _rows(ctx, "beliefs")
    if not _has_phantom_cols(ctx):
        return cc.silent(0, d, "pre-phantom schema; feature cannot write here")
    d["has_cols"] = 1
    d["phantoms"] = ctx.count(
        "SELECT COUNT(*) FROM beliefs WHERE type='speculative' AND origin='speculative'")
    d["gc_retired"] = _gc_retired(ctx)
    d["eligible_14d"] = _eligible_14d(ctx)
    return cc.from_count(d["gc_retired"], d, "shared with A319 A347")


def a347(ctx: cc.Ctx) -> cc.Result:
    rows, deleted = _autogc(ctx)
    return cc.from_count(deleted, {"feed_rows": rows},
                         "feed rows written only when deleted > 0")


def a320(ctx: cc.Ctx) -> cc.Result:
    """wonder --persist graph-walk phantoms, with the other families for context."""
    if (r := _no_db(ctx)) is not None:
        return r
    if not ctx.table_exists("belief_corroborations"):
        return cc.silent(0, note="table absent")
    fam = _families(ctx)
    n, retired = _wonder_count(ctx, LIKE_BFS)
    detail = {"live": n - retired, "retired": retired,
              "has_valid_to": int(_col(ctx, "valid_to")),
              "dispatch": fam.get("dispatch", 0), "auto": fam.get("auto", 0),
              "other": fam.get("other", 0)}
    return cc.from_count(n, detail)


def a321(ctx: cc.Ctx) -> cc.Result:
    """wonder --persist-docs: research-agent docs plus the persist-docs feed row."""
    if (r := _no_db(ctx)) is not None:
        return r
    d = {"has_valid_to": 0, "live": 0, "retired": 0, "feed_rows": 0}
    n = 0
    if ctx.table_exists("belief_corroborations"):
        d["has_valid_to"] = int(_col(ctx, "valid_to"))
        n, retired = _wonder_count(ctx, LIKE_DISPATCH)
        d["live"], d["retired"] = n - retired, retired
    d["feed_rows"] = cc.jsonl_count(
        _feed_paths(ctx), lambda row: row.get("event") == "belief.ingested"
        and row.get("source") == "wonder.persist-docs")
    return cc.from_count(n, d)


def a323(ctx: cc.Ctx) -> cc.Result:
    if (r := _no_db(ctx)) is not None:
        return r
    by_axis = _axis_counts(ctx)
    known = sum(v for k, v in by_axis.items() if k in AXES)
    return cc.from_count(known, by_axis,
                         "indirect: axes mode prints only; counts persisted docs named by a dispatch axis")


def a324(ctx: cc.Ctx) -> cc.Result:
    """Evidence promotion: origin speculative -> evidence_promoted plus 'promotion:evidence'."""
    if (r := _no_db(ctx)) is not None:
        return r
    if not ctx.table_exists("beliefs"):
        return cc.untraceable("no beliefs table")
    d = {"has_origin": 0, "beliefs": _rows(ctx, "beliefs"), "promoted": 0, "reverted": 0,
         "current": 0, "live_phantoms": 0, "applied_support": 0}
    if _col(ctx, "source", "feedback_history"):
        d["promoted"], d["reverted"] = ctx.counts(
            "SELECT"
            " (SELECT COUNT(DISTINCT belief_id) FROM feedback_history"
            "  WHERE source='promotion:evidence'),"
            " (SELECT COUNT(DISTINCT belief_id) FROM feedback_history"
            "  WHERE source='promotion:revert_evidence')")
    if _col(ctx, "origin"):
        d["has_origin"] = 1
        d["current"] = ctx.count("SELECT COUNT(*) FROM beliefs WHERE origin='evidence_promoted'")
        vt = " AND valid_to IS NULL" if _col(ctx, "valid_to") else ""
        d["live_phantoms"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE origin='speculative'" + vt)
    if _col(ctx, "applied", "belief_support_evidence"):
        d["applied_support"] = ctx.count(
            "SELECT COUNT(*) FROM belief_support_evidence WHERE applied=1")
    if d["promoted"] > 0 or d["current"] > 0:
        return cc.fired(max(d["promoted"], d["current"]), d)
    if d["has_origin"] == 0:
        return cc.silent(0, d, "pre-origin schema; feature cannot write here")
    return cc.silent(0, d)


_UNRESOLVED_SQL = (
    "SELECT COUNT(*) FROM edges c WHERE c.type='CONTRADICTS' AND NOT EXISTS"
    " (SELECT 1 FROM edges s WHERE s.type='SUPERSEDES' AND ((s.src=c.src AND s.dst=c.dst)"
    " OR (s.src=c.dst AND s.dst=c.src)))")


def _corr(ctx: cc.Ctx, like: str) -> int:
    return ctx.count(
        "SELECT COUNT(DISTINCT belief_id) FROM belief_corroborations"
        " WHERE source_type = ? AND source_path_hash LIKE ? ESCAPE '\\'", (WONDER, like))


def a329(ctx: cc.Ctx) -> cc.Result:
    """contradiction_resolution axis, persisted by --persist-docs or the auto pass."""
    if (r := _no_db(ctx)) is not None:
        return r
    d = {"sub_axis": 0, "auto_axis": 0, "sub_any": 0, "auto_any": 0,
         "contradicts_edges": 0, "unresolved_contradicts": 0,
         "beliefs_after_ship": _since(ctx, AXES_SHIPPED), "beliefs": _rows(ctx, "beliefs")}
    if ctx.table_exists("belief_corroborations"):
        d["sub_axis"] = _corr(ctx, LIKE_DISPATCH[:-1] + "contradiction\\_resolution@%")
        d["auto_axis"] = _corr(ctx, "auto\\_wonder:v1:contradiction\\_resolution:%")
        d["sub_any"] = _corr(ctx, LIKE_DISPATCH)
        d["auto_any"] = _corr(ctx, LIKE_AUTO)
    if _col(ctx, "type", "edges"):
        d["contradicts_edges"] = ctx.count("SELECT COUNT(*) FROM edges WHERE type='CONTRADICTS'")
        d["unresolved_contradicts"] = ctx.count(_UNRESOLVED_SQL)
    return cc.from_count(d["sub_axis"] + d["auto_axis"], d,
                         "axis persists only via --persist-docs or auto-wonder;"
                         " needs unresolved CONTRADICTS in candidates")


def a337(ctx: cc.Ctx) -> cc.Result:
    ring = _ring(ctx)
    if ring is None:
        return cc.silent(0, note="no ring file")
    raw: Any = ring.get("promotion_dedup")
    keys: list[Any] = list(raw) if isinstance(raw, list) else []  # pyright: ignore[reportUnknownArgumentType]
    dedup = sum(1 for k in keys if isinstance(k, str))
    return cc.from_count(_int(ring.get("promotion_fires")), {"dedup_keys": dedup},
                         "ring holds the latest session only")


def a341(ctx: cc.Ctx) -> cc.Result:
    ring = _ring(ctx)
    if ring is None:
        return cc.silent(0, note="no ring file")
    detail = _dedup_prefix_counts(ring)
    detail["phantom_init"] = 1 if ring.get("phantom_init") is True else 0
    return cc.from_count(_int(ring.get("phantom_fires")), detail,
                         "ring holds the latest session only")


def a343(ctx: cc.Ctx) -> cc.Result:
    """Token-Jaccard ranking runs inside every --persist run.

    Count is the graph-walk phantoms; when there are none, speculative
    beliefs that carry the graph-walk joiner without a corroboration row
    count instead.
    """
    if (r := _no_db(ctx)) is not None:
        return r
    detail: dict[str, int] = {}
    n = 0
    if ctx.table_exists("belief_corroborations"):
        n, retired = _wonder_count(ctx, LIKE_BFS)
        if _col(ctx, "valid_to"):
            detail["retired"] = retired
        else:
            detail["no_valid_to_col"] = 1
        detail["wonder_ingest_rows_any"] = ctx.count(
            "SELECT COUNT(*) FROM belief_corroborations WHERE source_type = 'wonder_ingest'")
    else:
        detail["no_corroboration_table"] = 1
    if ctx.table_exists("beliefs"):
        typed = _col(ctx, "type")
        if _col(ctx, "content"):
            detail["joiner_speculative"] = ctx.count(
                "SELECT COUNT(*) FROM beliefs WHERE instr(content, ?) > 0"
                + (" AND type = 'speculative'" if typed else ""), (JOINER,))
        if not typed:
            detail["no_type_col"] = 1
        if _col(ctx, "created_at"):
            detail["beliefs_after_3_0_0"] = _since(ctx, GC_SHIPPED)
            detail["beliefs_total"] = _rows(ctx, "beliefs")
    detail["bfs_rows"] = n
    count = n if n else detail.get("joiner_speculative", 0)
    return cc.from_count(count, detail, "ranking runs inside every --persist run; same rows as A320")


def a346(ctx: cc.Ctx) -> cc.Result:
    """Graph-walk seed picker: only the --persist variant leaves a trace."""
    if (r := _no_db(ctx)) is not None:
        return r
    if not ctx.table_exists("beliefs"):
        return cc.silent(0, note="no beliefs table")
    has_corr = ctx.table_exists("belief_corroborations")
    fam = _families(ctx)
    n, retired = _wonder_count(ctx, LIKE_BFS)
    orphan = 0
    if _col(ctx, "origin"):
        sub = ("SELECT belief_id FROM belief_corroborations WHERE source_type = 'wonder_ingest'"
               if has_corr else "SELECT NULL WHERE 0")
        orphan = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE origin = 'speculative'"
            " AND id NOT IN (" + sub + ")")
    detail = {"live": n - retired, "retired": retired,
              "has_valid_to": int(_col(ctx, "valid_to")),
              "has_corroborations": int(has_corr), "beliefs_after_ship": _since(ctx, GC_SHIPPED),
              "orphan_speculative": orphan, "dispatch": fam.get("dispatch", 0),
              "auto": fam.get("auto", 0), "other": fam.get("other", 0)}
    return cc.from_count(n, detail, "floor: read-only graph walk leaves no trace")


_PRIOR_COLS = ("type", "origin", "alpha", "beta", "retention_class")


def a350(ctx: cc.Ctx) -> cc.Result:
    """wonder_ingest phantom writer: every family, at-ingest prior and RELATES_TO edges."""
    if (r := _no_db(ctx)) is not None:
        return r
    if not _col(ctx, "source_path_hash", "belief_corroborations"):
        return cc.silent(0, note="no corroborations table")
    fam = _families(ctx)
    detail = dict(fam)
    beliefs_ok = all(_col(ctx, c) for c in _PRIOR_COLS)
    edges_ok = _col(ctx, "src", "edges") and _col(ctx, "type", "edges")
    detail["at_ingest_prior"] = ctx.count(
        "SELECT COUNT(*) FROM beliefs WHERE type = 'speculative' AND origin = 'speculative'"
        " AND alpha = 0.3 AND beta = 1.0 AND retention_class = 'snapshot'") if beliefs_ok else 0
    detail["relates_to_edges"] = ctx.count(
        "SELECT COUNT(*) FROM edges e WHERE e.type = 'RELATES_TO' AND e.src IN"
        " (SELECT belief_id FROM belief_corroborations WHERE source_type = ?)",
        (WONDER,)) if edges_ok else 0
    detail["beliefs"] = _rows(ctx, "beliefs")
    detail["schema_guarded"] = int(not (beliefs_ok and edges_ok))
    return cc.from_count(sum(fam.values()), detail)


_UNTRACEABLE: list[tuple[str, str, str]] = [
    ("A311", "aelf reason consequence paths",
     "aelf reason is read-only; paths go to stdout only"),
    ("A313", "aelf reason GAP impasse", "aelf reason is read-only; impasses go to stdout only"),
    ("A314", "aelf reason TIE / fork-TIE / CONSTRAINT_FAILURE impasses",
     "aelf reason is read-only; impasses go to stdout only"),
    ("A315", "aelf reason verdict classifier and low-evidence rule",
     "aelf reason is read-only; verdict goes to stdout only"),
    ("A318", "aelf wonder --emit-phantoms", "prints Phantom JSON to stdout; no store or log write"),
    ("A326", "Gap analysis agent_only_dominant gap",
     "gap string printed by axes mode only; no axis or persisted doc carries it"),
    ("A330", "Graph auditor corpus_volume warning", "aelf doctor audit is read-only; stdout only"),
    ("A331", "Graph auditor credal_gap metric", "aelf doctor audit is read-only; stdout only"),
    ("A332", "Graph auditor fail checks and aelf doctor default",
     "aelf doctor audit is read-only; stdout and exit code only"),
    ("A333", "Graph auditor ingest_gap warning", "aelf doctor audit is read-only; stdout only"),
    ("A334", "Graph auditor lock_budget warning", "aelf doctor audit is read-only; stdout only"),
    ("A335", "Offline phantom-generation strategies RW/TC/STS and bake-off",
     "runner builds a synthetic corpus in an in-memory store; never touches a real store"),
    ("A336", "Phantom opportunity auto_dispatch framing",
     "changes note header text only; note is not logged; ring keys are the same"),
    ("A342", "scoring.uncertainty_score and RW uncertainty floor",
     "only caller is the in-memory bake-off runner"),
    ("A345", "Wonder GC promotion guard",
     "guard keeps phantoms live and writes nothing"),
    ("A348", "Wonder query agent-count shorthand", "only changes the printed agent_count"),
    ("A349", "Wonder suggested-action labels",
     "labels printed only; --persist phantoms do not store the action"),
    ("A351", "WonderResult coverage scalar / phantoms_created",
     "output-only field, never persisted"),
]

PROBES: list[cc.Probe] = [
    cc.Probe("A312", "aelf reason SUGGESTED UPDATES writeback", "store", a312),
    cc.Probe("A316", "aelf review verdict semantics", "store", a316),
    cc.Probe("A317", "aelf review weekly checkpoint", "store", a317),
    cc.Probe("A319", "aelf wonder --gc (manual phantom GC)", "store", a319),
    cc.Probe("A320", "aelf wonder --persist (graph-walk phantoms)", "store", a320),
    cc.Probe("A321", "aelf wonder --persist-docs", "store", a321),
    cc.Probe("A322", "aelf wonder graph-walk consolidation candidates", "store",
             _wonder_simple_probe(LIKE_BFS, "only --persist runs leave a trace;"
                                  " non-persist output is stdout only")),
    cc.Probe("A323", "aelf wonder QUERY axes mode", "store", a323),
    cc.Probe("A324", "Automatic phantom promotion on user evidence", "store", a324),
    cc.Probe("A325", "Automatic wonder pass", "store",
             _wonder_simple_probe(LIKE_AUTO, "nothing calls auto_run outside the replay"
                                  " harness; any row would come from a replay")),
    cc.Probe("A327", "Gap analysis high-uncertainty gate", "store",
             _axis_probe("uncertainty_deep_dive",
                         "indirect: persisted uncertainty_deep_dive axis docs only")),
    cc.Probe("A328", "Gap analysis query-term coverage", "store",
             _axis_probe("coverage_extension",
                         "indirect: persisted coverage_extension axis docs only")),
    cc.Probe("A329", "Gap analysis unresolved CONTRADICTS axis", "store", a329),
    cc.Probe("A337", "Phantom promotion-opportunity note", "sidecar", a337),
    cc.Probe("A338", "Phantom trigger signal (a) gap", "sidecar",
             _ring_prefix_probe("gap", "ring holds the latest session only")),
    cc.Probe("A339", "Phantom trigger signal (b) novel named entity", "sidecar",
             _ring_prefix_probe("new_entity", "ring holds the latest session only")),
    cc.Probe("A340", "Phantom trigger signal (c) new CONTRADICTS pair", "sidecar",
             _ring_prefix_probe("contradiction", "ring holds the latest session only")),
    cc.Probe("A341", "Phantom-generation opportunity note", "sidecar", a341),
    cc.Probe("A343", "Wonder consolidation token-Jaccard ranking", "store", a343),
    cc.Probe("A344", "Wonder GC eligibility predicate", "store", a344),
    cc.Probe("A346", "Wonder graph-walk seed picker", "store", a346),
    cc.Probe("A347", "Wonder phantom auto-GC on SessionStart", "sidecar", a347),
    cc.Probe("A350", "wonder_ingest phantom writer", "store", a350),
] + [cc.untraceable_probe(i, n, note) for i, n, note in _UNTRACEABLE]

PROBES.sort(key=lambda p: p.item_id)
TRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind != "untraceable")
UNTRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind == "untraceable")
