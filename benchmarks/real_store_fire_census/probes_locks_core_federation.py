"""Census probes: subsystem locks-core-federation (A184-A216).

Every probe reads one store through a read-only connection (`ctx.db`) or
the sidecar files beside its `memory.db`, and returns integers and fixed
labels. A205 and A211 import pure aelfrice functions and constants; no
probe opens a store through aelfrice.

Write sites the probes key on (src/aelfrice):
  core_gate_batches / core_gate_labels   store.py create/accept/put/delete/reopen
  core_gate_background.jsonl             core_gate_background._append_log, one row per child run
  command_outcomes.jsonl                 hook._write_command_outcome_record (typed /aelf:lock)
  hook_audit.jsonl beliefs[].scope=user  hook._serialize_belief_for_audit (#1681)
  feed.jsonl belief.locked               cli._cmd_lock / _cmd_lock_promote_user (beside the repo store)
  ingest_log source_kind=cli_remember    cli._record_lock_ingest (aelf lock, repo or user store)
  beliefs.lock_expires_at, feedback_history source=lock:expire, schema_meta lock_expiry_sweep_at
  beliefs.scope global or shared:*       the rows search_peer_beliefs can return to a peer

Verification corrections folded in: A195 counts every child run the fixed
cap governed, A197 compares stored versions with the version the shipped
code computes, A205 reports whether a store can serve peer hits, A209
reports a store without core gate tables as untraceable, A211 bounds the
stderr-only `--for` suffix by eligible beliefs, and A216 counts lock
ingest rows in the user store as well as its active locks.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from . import core as cc

Probe = cc.Probe
Ctx = cc.Ctx
Result = cc.Result

BG_LOG = "core_gate_background.jsonl"
CMD_OUTCOMES = "command_outcomes.jsonl"
HOOK_AUDIT = "hook_audit.jsonl"
FEED = "feed.jsonl"
BG_OUTCOMES = ("accepted", "no_batch", "not_open", "stale_version", "stale_belief",
               "run_failed", "bad_reply", "spawn_failed", "refused", "store_error")
EPISODE_GAP_S = 3600          # models.CORROBORATION_EPISODE_GAP_SECONDS
MIN_EPISODES = 2              # models.CORROBORATION_MIN_EPISODES
CORE_MIN_CORR = 2             # hook._CORE_MIN_CORROBORATION
CORE_MIN_AB = 4               # hook._CORE_MIN_ALPHA_BETA
B_LIMIT = 2                   # core_gate.B_CORROBORATION_LIMIT
SELF_CHECK_MIN = 4            # core_gate.SELF_CHECK_MIN_BATCHES
BUDGET_NEAR_USD = 0.09        # RUNNER_BUDGET_USD is 0.10
TIMEOUT_NEAR_S = 295.0        # RUNNER_TIMEOUT_SECONDS is 300
BG_SHIP_DATE = "2026-10-22"   # 5.2.0: background labeling with the 0.10 USD cap (#1741)
GATE_SHIP_DATE = "2026-10-06"  # 5.1.0: core gate tables in the schema
PEER_V3_DATE = "2026-05-13"   # 3.0.0: scope column and valid_to
SUFFIX_SHIP_DATE = "2026-08-13"  # 4.3.0: Stop lock suggestion --for suffix (#1315)
VERSION_GLOB = "core-gate-[0-9a-f]*"

BatchRow = tuple[str, str, str, str, "str | None", str]


# --- helpers ---------------------------------------------------------------


def _has_gate(ctx: Ctx) -> bool:
    return ctx.table_exists("core_gate_batches") and ctx.table_exists("core_gate_labels")


def _num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _bg_rows(ctx: Ctx) -> list[dict[str, Any]]:
    return list(cc.iter_jsonl(ctx.sidecar(BG_LOG)))


def _bg_batch_ids(ctx: Ctx, outcome: str | None = None) -> set[str]:
    out: set[str] = set()
    for r in _bg_rows(ctx):
        bid = r.get("batch_id")
        if isinstance(bid, str) and (outcome is None or r.get("outcome") == outcome):
            out.add(bid)
    return out


def _batches(ctx: Ctx, where: str = "1=1") -> list[BatchRow]:
    """(batch_id, version, origin, created_at, accepted_at, items_json); in memory only."""
    rows = ctx.require_db().execute(
        "SELECT batch_id, classifier_version, origin, created_at, accepted_at, items_json "
        f"FROM core_gate_batches WHERE {where}")
    return [(str(a), str(b), str(c), str(d), None if e is None else str(e), str(f))
            for a, b, c, d, e, f in rows]


def _hashes(items_json: str) -> frozenset[str]:
    try:
        return frozenset(str(d["content_hash"]) for d in json.loads(items_json))
    except (ValueError, KeyError, TypeError):
        return frozenset()


def _owned_labels(ctx: Ctx) -> dict[str, int]:
    out: dict[str, int] = {}
    for bid, n in ctx.require_db().execute(
            "SELECT l.batch_id, COUNT(*) FROM core_gate_labels l JOIN core_gate_batches b "
            "ON b.batch_id = l.batch_id AND b.classifier_version = l.classifier_version "
            "GROUP BY l.batch_id"):
        out[str(bid)] = int(n)
    return out


def _subset_reruns(batches: list[BatchRow]) -> set[str]:
    """store.core_gate_subset_reruns over every doctor run at once."""
    runs: dict[tuple[str, str], dict[str, frozenset[str]]] = {}
    for bid, ver, origin, created, _acc, items in batches:
        if origin == "doctor":
            runs.setdefault((ver, created), {})[bid] = _hashes(items)
    out: set[str] = set()
    for members in runs.values():
        for bid, hs in members.items():
            if any(hs < other for o, other in members.items() if o != bid):
                out.add(bid)
    return out


def _parse_ts(v: Any) -> datetime | None:
    if not isinstance(v, str):
        return None
    t = v.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


_EPISODES_SQL = f"""
    SELECT belief_id,
           1 + SUM(CASE WHEN prev IS NOT NULL
                         AND ROUND((t - prev) * 86400.0, 3) >= {EPISODE_GAP_S}
                    THEN 1 ELSE 0 END) AS episodes
    FROM (
        SELECT belief_id, t, LAG(t) OVER (PARTITION BY belief_id ORDER BY t) AS prev
        FROM (
            SELECT b.id AS belief_id, julianday(b.created_at) AS t FROM beliefs b
            WHERE EXISTS (SELECT 1 FROM belief_corroborations c WHERE c.belief_id = b.id)
            UNION ALL
            SELECT belief_id, julianday(ingested_at) AS t FROM belief_corroborations
        ) WHERE t IS NOT NULL
    ) GROUP BY belief_id
"""


def _gate_exclusions(ctx: Ctx) -> dict[str, int]:
    """Active unlocked beliefs meeting the non-lock core rule that the gate
    drops under the newest label version (C, or B with >= 2 qualified
    corroborations). Mirrors store.gate_core_candidates + core_gate.admit."""
    if not _has_gate(ctx) or not ctx.table_exists("belief_corroborations"):
        return {}
    active = "b.valid_to IS NULL AND " if ctx.column_exists("beliefs", "valid_to") else ""
    sql = f"""
    WITH latest AS (
        SELECT classifier_version AS v FROM core_gate_labels
        GROUP BY classifier_version ORDER BY MAX(labeled_at) DESC LIMIT 1),
    ep AS ({_EPISODES_SQL}),
    corr AS (SELECT belief_id, COUNT(*) AS n FROM belief_corroborations GROUP BY belief_id),
    cand AS (
        SELECT b.content_hash AS h, COALESCE(corr.n, 0) AS n, COALESCE(ep.episodes, 0) AS e
        FROM beliefs b LEFT JOIN corr ON corr.belief_id = b.id
        LEFT JOIN ep ON ep.belief_id = b.id
        WHERE {active} b.lock_level = 'none' AND (
            (COALESCE(corr.n, 0) >= {CORE_MIN_CORR} AND COALESCE(ep.episodes, 0) >= {MIN_EPISODES})
            OR (b.alpha + b.beta >= {CORE_MIN_AB}
                AND b.alpha / (b.alpha + b.beta) >= 2.0 / 3.0)))
    SELECT
        SUM(l.label = 'C'),
        SUM(l.label = 'B' AND (CASE WHEN cand.e >= {MIN_EPISODES} THEN cand.n ELSE 0 END) >= {B_LIMIT}),
        SUM(l.label = 'B'),
        COUNT(l.label)
    FROM cand JOIN core_gate_labels l ON l.content_hash = cand.h
    WHERE l.classifier_version = (SELECT v FROM latest)
    """
    c_ex, b_ex, b_all, labeled = ctx.counts(sql)
    return {"c_excluded": c_ex, "b_excluded": b_ex, "b_labeled_candidates": b_all,
            "labeled_candidates": labeled}


def _no_table(note: str = "no core gate tables") -> Result:
    return cc.silent(0, note=note)


def _active_sql(ctx: Ctx, prefix: str = " AND ") -> str:
    return f"{prefix}valid_to IS NULL" if ctx.column_exists("beliefs", "valid_to") else ""


# --- core gate probes ------------------------------------------------------


def p_a184(ctx: Ctx) -> Result:
    """aelf core-gate accept: only this command accepts doctor batches; a
    session_end batch accepted with no background accepted row was too."""
    if not _has_gate(ctx):
        return _no_table()
    bg_ok = _bg_batch_ids(ctx, "accepted")
    doctor = ctx.count("SELECT COUNT(*) FROM core_gate_batches "
                       "WHERE origin='doctor' AND accepted_at IS NOT NULL")
    se = [b[0] for b in _batches(ctx, "origin='session_end' AND accepted_at IS NOT NULL")]
    se_cli = sum(1 for b in se if b not in bg_ok)
    return cc.from_count(doctor + se_cli, {"doctor_accepted": doctor,
                                           "session_end_accepted_no_bg_row": se_cli})


def p_a185(ctx: Ctx) -> Result:
    if not _has_gate(ctx):
        return _no_table()
    n, acc, runs, items = ctx.counts(
        "SELECT COUNT(*), SUM(accepted_at IS NOT NULL), COUNT(DISTINCT created_at), "
        "SUM(json_array_length(items_json)) FROM core_gate_batches WHERE origin='doctor'")
    return cc.from_count(n, {"batches": n, "accepted": acc, "emit_runs": runs, "items": items},
                         note="doctor origin batches; a rerun of a session_end batch also writes one")


def p_a186(ctx: Ctx) -> Result:
    """--rerun deletes the labels a batch owns; the batch stays accepted unless
    every belief is kept (then it is reopened and indistinguishable)."""
    if not _has_gate(ctx):
        return _no_table()
    batches = _batches(ctx)
    owned = _owned_labels(ctx)
    zero = [b for b in batches if b[4] is not None and owned.get(b[0], 0) == 0]
    subset = _subset_reruns(batches)
    return cc.from_count(len(zero), {
        "accepted_owning_no_labels": len(zero),
        "of_them_session_end": sum(1 for b in zero if b[2] == "session_end"),
        "subset_rerun_batches": len(subset)},
        note="whole batch reopen leaves no trace; a later overlapping relabel also empties a batch")


def p_a193(ctx: Ctx) -> Result:
    if not _has_gate(ctx):
        return _no_table()
    ex = _gate_exclusions(ctx)
    if not ex:
        return _no_table("no corroboration table")
    return cc.from_count(ex["b_excluded"], {"b_excluded": ex["b_excluded"],
                                            "b_labeled_candidates": ex["b_labeled_candidates"]},
                         note="newest label version stands in for the installed one")


def p_a194(ctx: Ctx) -> Result:
    rows = _bg_rows(ctx)
    tally = cc.jsonl_tally(ctx.sidecar(BG_LOG), lambda r: str(r.get("outcome")), BG_OUTCOMES)
    detail = dict(tally)
    if ctx.db is not None and _has_gate(ctx):
        bg_ok = _bg_batch_ids(ctx, "accepted")
        se = _batches(ctx, "origin='session_end'")
        detail["session_end_batches"] = len(se)
        detail["session_end_accepted_by_bg"] = sum(1 for b in se if b[4] is not None
                                                   and b[0] in bg_ok)
    return cc.from_count(len(rows), detail, note="one log row per detached child run")


def p_a195(ctx: Ctx) -> Result:
    """The cap is fixed runner argv plus a subprocess timeout on every child
    invocation, so every log row for an invoked child (numeric `seconds`, or
    error TimeoutExpired) is a run the cap governed. Detail keeps the
    binding counts."""
    rows = _bg_rows(ctx)
    invoked = budget = timeout = pre_ship = 0
    for r in rows:
        ts = r.get("ts")
        if isinstance(ts, str) and ts[:10] < BG_SHIP_DATE:
            pre_ship += 1
        secs = r.get("seconds")
        to = r.get("error") == "TimeoutExpired"
        if _num(secs) or to:
            invoked += 1
        cost = r.get("cost_usd")
        if _num(cost) and float(cost) >= BUDGET_NEAR_USD:  # pyright: ignore[reportArgumentType]
            budget += 1
        if to or (_num(secs) and float(secs) >= TIMEOUT_NEAR_S):  # pyright: ignore[reportArgumentType]
            timeout += 1
    detail = {"log_rows": len(rows), "invoked_runs": invoked, "cost_ge_0_09": budget,
              "timeout": timeout, "rows_before_ship": pre_ship}
    if invoked:
        return cc.fired(invoked, detail,
                        note="child runs under the fixed cap argv; cap bound in cost_ge_0_09 + timeout")
    if ctx.column_exists("core_gate_batches", "created_at"):
        detail["batches_since_ship"] = ctx.count(
            "SELECT COUNT(*) FROM core_gate_batches WHERE substr(created_at,1,10) >= ?",
            (BG_SHIP_DATE,))
    return cc.silent(0, detail, note="no background child run in this store")


def p_a196(ctx: Ctx) -> Result:
    if not _has_gate(ctx):
        return _no_table()
    ex = _gate_exclusions(ctx)
    if not ex:
        return _no_table("no corroboration table")
    versions = ctx.count("SELECT COUNT(DISTINCT classifier_version) FROM core_gate_labels")
    n = ex["c_excluded"] + ex["b_excluded"]
    return cc.from_count(n, {**ex, "label_versions": versions},
                         note="core candidates the gate drops now; newest label version stands in for installed")


def _days_since(ctx: Ctx, table: str, col: str, agg: str = "MAX") -> int:
    if not ctx.column_exists(table, col):
        return -1
    return ctx.count(
        f"SELECT CAST(julianday('now') - julianday({agg}({col})) AS INTEGER) FROM {table}")


def p_a197(ctx: Ctx) -> Result:
    """Rows keyed to a version the shipped code does not compute. The label
    cache is read under core_gate.CLASSIFIER_VERSION, so a version change
    after the last label write orphans every older row."""
    from aelfrice.core_gate import CLASSIFIER_VERSION  # noqa: PLC0415

    if not _has_gate(ctx):
        return _no_table()
    if CLASSIFIER_VERSION is None:
        return cc.untraceable("installed code computes no classifier version")
    cur = CLASSIFIER_VERSION
    d: dict[str, int] = {
        "labels": ctx.count("SELECT COUNT(*) FROM core_gate_labels"),
        "batches": ctx.count("SELECT COUNT(*) FROM core_gate_batches"),
        "versions": ctx.count(
            "SELECT COUNT(*) FROM (SELECT classifier_version FROM core_gate_labels "
            "UNION SELECT classifier_version FROM core_gate_batches)"),
        "cur_labels": ctx.count(
            "SELECT COUNT(*) FROM core_gate_labels WHERE classifier_version = ?", (cur,)),
        "cur_batches": ctx.count(
            "SELECT COUNT(*) FROM core_gate_batches WHERE classifier_version = ?", (cur,)),
        "bad_format": ctx.count(
            "SELECT (SELECT COUNT(*) FROM core_gate_labels WHERE length(classifier_version) != 26 "
            "OR classifier_version NOT GLOB ?) + (SELECT COUNT(*) FROM core_gate_batches "
            "WHERE length(classifier_version) != 26 OR classifier_version NOT GLOB ?)",
            (VERSION_GLOB, VERSION_GLOB)),
    }
    d["stale_labels"] = d["labels"] - d["cur_labels"]
    d["stale_batches"] = d["batches"] - d["cur_batches"]
    d["days_since_label"] = _days_since(ctx, "core_gate_labels", "labeled_at") if d["labels"] else -1
    d["days_since_first_label"] = (
        _days_since(ctx, "core_gate_labels", "labeled_at", "MIN") if d["labels"] else -1)
    d["days_since_belief"] = _days_since(ctx, "beliefs", "created_at")
    note = "rows keyed to a version the current code does not compute"
    if d["labels"] + d["batches"] == 0:
        note = "gate tables empty"
    return cc.from_count(d["stale_labels"] + d["stale_batches"], d, note)


def p_a198(ctx: Ctx) -> Result:
    """The self-check runs on an accept of a doctor batch once its run has
    >= 4 accepted non-subset batches owning labels: count those accepts."""
    if not _has_gate(ctx):
        return _no_table()
    batches = _batches(ctx)
    owned = _owned_labels(ctx)
    subset = _subset_reruns(batches)
    runs: dict[tuple[str, str], int] = {}
    for bid, ver, origin, created, acc, _items in batches:
        if origin == "doctor" and acc is not None and owned.get(bid, 0) > 0 and bid not in subset:
            runs[(ver, created)] = runs.get((ver, created), 0) + 1
    checks = sum(max(0, n - (SELF_CHECK_MIN - 1)) for n in runs.values())
    return cc.from_count(checks, {"runs_with_4plus": sum(1 for n in runs.values()
                                                         if n >= SELF_CHECK_MIN),
                                  "doctor_runs": len(runs)},
                         note="inferred from accepted batches per emit run; outlier lines go to stderr only")


def p_a199(ctx: Ctx) -> Result:
    """A session_end batch with no background log row is either a
    continuation-turn label or a background child killed before its log
    write, and the store can't tell them apart: untraceable when one exists,
    silent when every session_end batch has its background row."""
    if not _has_gate(ctx):
        return _no_table()
    logged = _bg_batch_ids(ctx)
    se = _batches(ctx, "origin='session_end'")
    cont = [b for b in se if b[0] not in logged]
    detail = {"session_end_batches": len(se), "session_end_no_bg_row": len(cont),
              "of_them_accepted": sum(1 for b in cont if b[4] is not None)}
    if cont:
        return cc.Result("untraceable", None, detail,
                         "ambiguous: a background child killed before its log write looks the same")
    return cc.silent(0, detail, "every session_end batch has a background log row")


def p_a204(ctx: Ctx) -> Result:
    if not _has_gate(ctx):
        return _no_table()
    detail = ctx.group_count(
        "SELECT origin, COUNT(*) FROM core_gate_batches WHERE accepted_at IS NULL GROUP BY 1",
        {"doctor", "session_end"})
    items = ctx.count("SELECT SUM(json_array_length(items_json)) FROM core_gate_batches "
                      "WHERE accepted_at IS NULL")
    n = sum(detail.values())
    return cc.from_count(n, {**{f"open_{k}": v for k, v in detail.items()},
                             "claimed_items": items},
                         note="open batches claim their hashes; includes batches still awaiting a reply")


# A store that serves peer FTS5 hits must hold active rows with scope
# 'global' or 'shared:*' (search_peer_beliefs filters on both). Search and
# reason output is never logged, so this counts the precondition: a store
# with none of those rows cannot have served a peer hit.
def p_a205(ctx: Ctx) -> Result:
    if not ctx.table_exists("beliefs"):
        return cc.untraceable("no beliefs table")
    has_scope = ctx.column_exists("beliefs", "scope")
    d: dict[str, int] = {"has_scope_col": int(has_scope)}
    if not ctx.column_exists("beliefs", "valid_to"):
        d["beliefs"] = ctx.count("SELECT COUNT(*) FROM beliefs")
        return cc.silent(0, d, note="pre v3 schema: the peer query skips this store")
    d["active"] = ctx.count("SELECT COUNT(*) FROM beliefs WHERE valid_to IS NULL")
    if has_scope:
        act = "SELECT COUNT(*) FROM beliefs WHERE valid_to IS NULL AND "
        d["active_project"] = ctx.count(act + "scope = 'project'")
        d["active_global"] = ctx.count(act + "scope = 'global'")
        d["active_shared"] = ctx.count(act + "scope LIKE 'shared:%'")
        d["active_other_scope"] = (d["active"] - d["active_project"] - d["active_global"]
                                   - d["active_shared"])
        d["any_global_or_shared"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE scope = 'global' OR scope LIKE 'shared:%'")
    if ctx.table_exists("feedback_history"):
        d["scope_flip_rows"] = ctx.count(
            "SELECT COUNT(*) FROM feedback_history WHERE source LIKE 'scope:%'")
    if ctx.column_exists("beliefs", "created_at"):
        d["rows_after_v3"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (PEER_V3_DATE,))
    d["deps_file_beside_store"] = int(ctx.sidecar("knowledge_deps.json").exists())
    visible = d.get("active_global", 0) + d.get("active_shared", 0)
    return cc.from_count(visible, d, note="count = active rows a peer search can return")


_CORE_SESSION = (
    "b.valid_to IS NULL AND b.lock_level != 'user' AND b.session_id IS NOT NULL "
    "AND ((b.alpha+b.beta>=4 AND b.alpha/(b.alpha+b.beta)>=2.0/3.0) "
    "OR (SELECT COUNT(*) FROM belief_corroborations bc WHERE bc.belief_id=b.id)>=2)")
_IN_BATCH = ("b.content_hash IN (SELECT json_extract(je.value,'$.content_hash') "
             "FROM core_gate_batches cb, json_each(cb.items_json) je)")
_LABELED = "EXISTS (SELECT 1 FROM core_gate_labels l WHERE l.content_hash=b.content_hash)"


def _core_session_detail(ctx: Ctx) -> dict[str, int]:
    """Core-rule session beliefs since 5.1.0 and 5.2.0: the expected candidates.
    The corroboration arm is corr >= 2, an upper bound (episodes not checked)."""
    if not (ctx.column_exists("beliefs", "session_id") and ctx.column_exists("beliefs", "lock_level")
            and ctx.column_exists("beliefs", "valid_to")
            and ctx.table_exists("belief_corroborations")):
        return {}
    since = (f"SELECT COUNT(*) FROM beliefs b WHERE {_CORE_SESSION} "
             "AND julianday(b.created_at)>=julianday(?)")
    return {
        "core_session_beliefs_since_510": ctx.count(since, (GATE_SHIP_DATE,)),
        "core_session_beliefs_since_520": ctx.count(since, (BG_SHIP_DATE,)),
        "since_520_unbatched_unlabeled": ctx.count(
            since + f" AND NOT {_IN_BATCH} AND NOT {_LABELED}", (BG_SHIP_DATE,)),
    }


def p_a209(ctx: Ctx) -> Result:
    """Only hook.py's Stop paths write origin='session_end' batches, and both
    batch list_core_gate_session_candidates output. The gate tables are in
    the schema since 5.1.0, so a store without them was never written by a
    build that has the feature: untraceable, not silent."""
    if not ctx.table_exists("beliefs"):
        return Result("untraceable", None, {"no_beliefs_table": 1},
                      "feature absent: no beliefs table")
    after510 = (ctx.count("SELECT COUNT(*) FROM beliefs WHERE julianday(created_at)>=julianday(?)",
                          (GATE_SHIP_DATE,))
                if ctx.column_exists("beliefs", "created_at") else 0)
    if not _has_gate(ctx):
        return Result("untraceable", None, {"beliefs_since_510": after510},
                      "feature absent: no core gate tables, store not written by 5.1.0+")
    d: dict[str, int] = {"beliefs_since_510": after510}
    n, items, sessions, acc = ctx.counts(
        "SELECT COUNT(*), SUM(json_array_length(items_json)), COUNT(DISTINCT session_id), "
        "SUM(accepted_at IS NOT NULL) FROM core_gate_batches WHERE origin='session_end'")
    d.update({"batches": n, "items": items, "sessions": sessions, "accepted": acc})
    d["doctor_batches"] = ctx.count("SELECT COUNT(*) FROM core_gate_batches WHERE origin='doctor'")
    d.update(_core_session_detail(ctx))
    d["bg_log_rows"] = cc.jsonl_count(ctx.sidecar(BG_LOG))
    if n > 0:
        return cc.fired(n, d)
    return cc.silent(0, d, "tables present; no session_end batch")


# --- lock probes -----------------------------------------------------------


def p_a192(ctx: Ctx) -> Result:
    """aelf lock fills ingest_log.session_id from $AELF_SESSION_ID when no
    --session-id is passed. Count lock rows that carry a session id; a lock
    with neither leaves NULL."""
    if not ctx.table_exists("ingest_log"):
        return cc.silent(0, note="no ingest_log table")
    n, null_sid = ctx.counts(
        "SELECT COUNT(*), SUM(session_id IS NULL) FROM ingest_log "
        "WHERE source_kind = 'cli_remember'")
    with_sid = n - null_sid
    return cc.from_count(with_sid, {"lock_ingests": n, "null_session": null_sid,
                                    "with_session": with_sid},
                         note="ambiguous: an explicit --session-id writes the same row")


_SUFFIX_KEYS = ("rows", "rows_post_ship", "directives", "directives_post", "window_spec",
                "window_spec_post", "expiring_locks", "dir_window_raw",
                "dir_window_raw_post", "dir_window_ambiguous")


def p_a211(ctx: Ctx) -> Result:
    """The `--for` suffix renders to Stop-hook stderr only, so a render has no
    trace. It can render for a belief only when hook._directive_window_spec
    of its decision text is not None, so every belief with a session_id is
    evaluated (locked and retired included). Silent when no post-ship belief
    is eligible; untraceable when some are (eligible, render unrecorded)."""
    from aelfrice.derivation import decision_text  # noqa: PLC0415
    from aelfrice.directive_detector import detect_directive  # noqa: PLC0415
    from aelfrice.hook import _directive_window_spec  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]
    from aelfrice.lock_expiry import (  # noqa: PLC0415
        extract_stated_window,
        stated_window_is_ambiguous,
    )

    d = {k: 0 for k in _SUFFIX_KEYS}
    if not ctx.column_exists("beliefs", "session_id"):
        return cc.silent(0, d, "no beliefs session_id column")
    if ctx.column_exists("beliefs", "lock_expires_at"):
        d["expiring_locks"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE lock_expires_at IS NOT NULL")
    created = "COALESCE(created_at,'')" if ctx.column_exists("beliefs", "created_at") else "''"
    for bid, content, ts in ctx.require_db().execute(
            f"SELECT id, content, {created} FROM beliefs "
            "WHERE session_id IS NOT NULL AND session_id != ''"):
        d["rows"] += 1
        post = int(str(ts) >= SUFFIX_SHIP_DATE)
        d["rows_post_ship"] += post
        text = decision_text(str(bid), str(content or ""))
        if not detect_directive(text):
            continue
        d["directives"] += 1
        d["directives_post"] += post
        if stated_window_is_ambiguous(text):
            d["dir_window_ambiguous"] += 1
        elif extract_stated_window(text) is not None:
            d["dir_window_raw"] += 1
            d["dir_window_raw_post"] += post
        if _directive_window_spec(text) is not None:
            d["window_spec"] += 1
            d["window_spec_post"] += post
    if d["window_spec_post"] == 0:
        return cc.silent(0, d, "no eligible belief")
    return Result("untraceable", None, d, "eligible but stderr only")


def p_a212(ctx: Ctx) -> Result:
    if not ctx.column_exists("beliefs", "lock_expires_at"):
        return cc.silent(0, note="no lock_expires_at column")
    expire_src = ("SELECT belief_id FROM feedback_history WHERE source = 'lock:expire'"
                  if ctx.table_exists("feedback_history") else "SELECT NULL WHERE 0")
    active = _active_sql(ctx)
    windowed, active_locked = ctx.counts(
        f"SELECT COUNT(*), SUM(lock_level = 'user' {active}) "
        "FROM beliefs WHERE lock_expires_at IS NOT NULL")
    expired = ctx.count(f"SELECT COUNT(*) FROM ({expire_src})")
    union = ctx.count(
        "SELECT COUNT(*) FROM (SELECT id FROM beliefs WHERE lock_expires_at IS NOT NULL "
        f"UNION {expire_src})")
    marker = ctx.count("SELECT COUNT(*) FROM schema_meta WHERE key = 'lock_expiry_sweep_at'") \
        if ctx.table_exists("schema_meta") else 0
    return cc.from_count(union, {"windowed": windowed, "windowed_active_locked": active_locked,
                                 "expire_rows": expired, "sweep_marker": marker})


def p_a213(ctx: Ctx) -> Result:
    paths = cc.jsonl_paths(ctx.sidecar(CMD_OUTCOMES))

    def is_lock(r: dict[str, Any]) -> bool:
        return r.get("hook") == "aelf_command" and r.get("command") == "lock"

    n = cc.jsonl_count(paths, is_lock)
    failed = cc.jsonl_count(paths, lambda r: is_lock(r)
                            and r.get("reason") in ("exception", "nonzero_exit"))
    user = cc.jsonl_count(paths, lambda r: is_lock(r) and r.get("scope") == "user")
    return cc.from_count(n, {"lock_outcomes": n, "failed": failed, "user_scope": user},
                         note="ledger rows; gap detection itself is doctor output only")


def _audit_belief_lists(ctx: Ctx) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for p in cc.jsonl_paths(ctx.sidecar(HOOK_AUDIT)):
        for r in cc.iter_jsonl(p):
            bs: Any = r.get("beliefs")
            if isinstance(bs, list):
                out.append((r, [dict(b) for b in bs if isinstance(b, dict)]))  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
    return out


def p_a214(ctx: Ctx) -> Result:
    rows = entries = 0
    by_hook: dict[str, int] = {}
    for r, bs in _audit_belief_lists(ctx):
        k = sum(1 for b in bs if b.get("scope") == "user")
        if k:
            rows += 1
            entries += k
            h = r.get("hook")
            lab = h if h in ("user_prompt_submit", "session_start") else "other"
            by_hook[str(lab)] = by_hook.get(str(lab), 0) + 1
    return cc.from_count(rows, {"rows": rows, "user_entries": entries, **by_hook},
                         note="audit rows carrying a user scope lock")


def _user_lock_times(ctx: Ctx) -> dict[str, datetime | None] | None:
    if ctx.layout == "user" or ctx.global_dir is None:
        return None
    f = ctx.global_dir / "user" / "memory.db"
    if not f.is_file():
        return None
    conn = cc.open_ro(f)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(beliefs)")}
        active = " AND valid_to IS NULL" if "valid_to" in cols else ""
        return {str(i): _parse_ts(t) for i, t in conn.execute(
            f"SELECT id, locked_at FROM beliefs WHERE lock_level = 'user'{active}")}
    finally:
        conn.close()


def p_a208(ctx: Ctx) -> Result:
    """Audit rows where a repository L0 entry shares its id with a user lock
    that existed by then and no user-scope entry of that id was injected:
    the merge dropped the user copy (exact-id arm of merge_lock_sets)."""
    try:
        users = _user_lock_times(ctx)
    except sqlite3.Error:
        return cc.silent(0, note="user store copy unreadable")
    if users is None:
        return cc.silent(0, note="no user store copy, or this is the user store")
    rows = drops = 0
    for r, bs in _audit_belief_lists(ctx):
        ts = _parse_ts(r.get("ts"))
        user_ids = {b.get("id") for b in bs if b.get("scope") == "user"}
        k = 0
        for b in bs:
            if b.get("scope") == "user" or not b.get("locked"):
                continue
            bid = b.get("id")
            if not isinstance(bid, str) or bid not in users or bid in user_ids:
                continue
            lt = users[bid]
            if ts is None or lt is None or lt.replace(microsecond=0) <= ts:
                k += 1
        if k:
            rows += 1
            drops += k
    return cc.from_count(rows, {"rows": rows, "dropped_user_locks": drops},
                         note="exact id arm only; near duplicate drops need aelfrice dedup and are not counted")


def _feed_rows(ctx: Ctx, pred: Callable[[dict[str, Any]], bool]) -> int:
    return cc.jsonl_count(ctx.sidecar(FEED), lambda r: r.get("event") == "belief.locked"
                          and bool(pred(r)))


def p_a190(ctx: Ctx) -> Result:
    n = _feed_rows(ctx, lambda r: r.get("promoted_from") not in (None, ""))
    return cc.from_count(n, {"promote_feed_rows": n})


def _n(ctx: Ctx, table: str, sql: str) -> int:
    return ctx.count(sql) if ctx.table_exists(table) else 0


def p_a216(ctx: Ctx) -> Result:
    """Every `aelf lock --user` writes a cli_remember ingest row in the user
    store, and `aelf unlock --user` a lock:unlock row there, so the user
    store counts lock ingest rows plus active locks; a user store whose
    locks were all unlocked later still fires. Repo stores count the feed
    rows of `aelf lock --user`."""
    if ctx.layout != "user":
        n = _feed_rows(ctx, lambda r: r.get("scope") == "user" and not r.get("promoted_from"))
        return cc.from_count(n, {"lock_user_feed_rows": n},
                             note="repo store: feed rows of aelf lock --user")
    if ctx.db is None or not ctx.table_exists("beliefs"):
        return cc.silent(0, note="no user store")
    locks = ctx.count(f"SELECT COUNT(*) FROM beliefs WHERE lock_level = 'user'{_active_sql(ctx)}")
    ingest = _n(ctx, "ingest_log",
                "SELECT COUNT(*) FROM ingest_log WHERE source_kind = 'cli_remember'")
    unlocks = _n(ctx, "feedback_history",
                 "SELECT COUNT(*) FROM feedback_history WHERE source = 'lock:unlock'")
    return cc.from_count(ingest + locks, {
        "user_store_active_locks": locks,
        "user_store_lock_ingest_rows": ingest,
        "user_store_unlock_audits": unlocks,
        "user_store_beliefs": ctx.count("SELECT COUNT(*) FROM beliefs"),
    }, note="user store: lock ingest rows + active locks")


def _safe(fn: Callable[[Ctx], Result]) -> Callable[[Ctx], Result]:
    """Run fn only when the context has a db; else silent with no count."""
    def wrapped(ctx: Ctx) -> Result:
        if ctx.db is None:
            return cc.silent(None, note="no db")
        return fn(ctx)
    wrapped.__name__ = fn.__name__
    return wrapped


U = cc.untraceable_probe

PROBES: list[cc.Probe] = [
    Probe("A184", "aelf core-gate accept", "store", _safe(p_a184)),
    Probe("A185", "doctor core-gate --emit", "store", _safe(p_a185)),
    Probe("A186", "doctor core-gate --rerun", "store", _safe(p_a186)),
    U("A187", "doctor core-gate coverage counts",
      "doctor display only; coverage is computed on read and never stored"),
    U("A188", "doctor repo locks duplicating user locks",
      "doctor display only; the list is computed on read and writes nothing"),
    U("A189", "aelf gate list", "queries GitHub issues and prints; no store or log write"),
    Probe("A190", "aelf lock --promote-user", "sidecar", p_a190),
    U("A191", "aelf locked merged listing",
      "read only listing; opens stores for read and writes no row or log"),
    Probe("A192", "AELF_SESSION_ID inference", "store", _safe(p_a192)),
    Probe("A193", "B-label corroboration limit", "store", _safe(p_a193)),
    Probe("A194", "background core-gate labeler", "sidecar", p_a194),
    Probe("A195", "background runner budget and timeout", "sidecar", p_a195),
    Probe("A196", "core admission gate in selection", "store", _safe(p_a196)),
    Probe("A197", "classifier version digest", "store", _safe(p_a197)),
    Probe("A198", "core-gate self-check", "store", _safe(p_a198)),
    Probe("A199", "session-end continuation turn", "store", _safe(p_a199)),
    U("A200", "doctor expiring-lock warning",
      "doctor display only; the sweep marker it reads is counted under A212"),
    U("A201", "fail-closed gate without classifier source",
      "fail closed path writes no batch or label by design; install state is not in a store"),
    U("A202", "foreign-belief mutation guard", "raises ForeignBeliefError; nothing is written"),
    U("A203", "knowledge_deps.json peer discovery",
      "discovery writes nothing; the knob file sits in the repo working tree outside the store"),
    Probe("A204", "open batch claims beliefs", "store", _safe(p_a204)),
    Probe("A205", "peer FTS5 hits in search and reason", "store", _safe(p_a205)),
    U("A206", "peer-aware BFS walk",
      "read only walk over peer handles; reason output is not logged"),
    U("A207", "read-time expiry filter for user locks",
      "filter drops rows in memory on read; no row or log records the drop"),
    Probe("A208", "repo-wins user lock suppression", "sidecar", p_a208),
    Probe("A209", "session-end candidate rule", "store", _safe(p_a209)),
    U("A210", "SessionStart lock-gap notice",
      "line printed to SessionStart stdout after the audit row is written; no log carries it"),
    Probe("A211", "Stop lock suggestion --for suffix", "store", _safe(p_a211)),
    Probe("A212", "time-boxed locks", "store", _safe(p_a212)),
    Probe("A213", "typed-lock outcome ledger", "sidecar", p_a213),
    Probe("A214", "user locks merged into L0", "sidecar", p_a214),
    U("A215", "user store read-only open timeout",
      "timeout fallback prints one stderr line; nothing is stored"),
    Probe("A216", "aelf lock --user store", "store", p_a216),
]

TRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind != "untraceable")
UNTRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind == "untraceable")
