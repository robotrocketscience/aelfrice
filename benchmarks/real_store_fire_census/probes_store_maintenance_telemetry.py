"""Census probes: store-maintenance-telemetry and benchmarks-ci items.

Every probe reads one store copy through `core.Ctx` read-only and returns
integers and fixed labels only. The comment above each probe names the
writer under src/aelfrice it keys on.

A few features apply to one kind of store. The runner marks each store
with `ctx.layout`: `repo` for a positional copy, `named` for a
`--named` copy of a `git:<name>` store, `legacy` for a `--legacy` copy of a
per-project store under `~/.aelfrice/projects`, and `global` or `user`
for the stores inside the `--global-dir` copy.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import core as cc

P = cc.Probe
U = cc.untraceable_probe

# Origins whose beliefs are written without an ingest_log row by paths
# other than migrate / store split: wonder phantoms (speculative),
# claude-memory reconcile and promote (user_validated), evidence promotion.
_UNLOGGED_OTHER_ORIGINS = frozenset({"speculative", "user_validated", "evidence_promoted"})
_ORIGINS = frozenset({
    "user_stated", "user_corrected", "user_validated", "user_transcript",
    "agent_inferred", "agent_remembered", "document_recent", "unknown",
    "speculative", "evidence_promoted",
})


# --- shared helpers --------------------------------------------------------


def _is_named(ctx: cc.Ctx) -> bool:
    return ctx.layout == "named"


def _is_legacy(ctx: cc.Ctx) -> bool:
    return ctx.layout == "legacy"


def _named_parent_db(ctx: cc.Ctx) -> Path | None:
    """The repository store a named store lives under, in the live layout
    `<git-common-dir>/aelfrice/stores/<name>/memory.db`. A copy of a whole
    `.git/aelfrice` folder keeps that layout."""
    cand = ctx.store_dir.parent.parent / "memory.db"
    if ctx.store_dir.parent.name == "stores" and cand.is_file():
        return cand
    return None


def _logged_belief_ids(ctx: cc.Ctx) -> set[str]:
    """Every belief id named by an ingest_log.derived_belief_ids array."""
    if not ctx.column_exists("ingest_log", "derived_belief_ids"):
        return set()
    out: set[str] = set()
    rows = ctx.require_db().execute(
        "SELECT j.value FROM ingest_log l, json_each(l.derived_belief_ids) j "
        "WHERE l.derived_belief_ids IS NOT NULL AND json_valid(l.derived_belief_ids)"
    )
    for (v,) in rows:
        if isinstance(v, str):
            out.add(v)
    return out


def _unlogged_beliefs(ctx: cc.Ctx) -> list[tuple[str, str]]:
    """(id, origin) for beliefs no ingest_log row derives."""
    logged = _logged_belief_ids(ctx)
    has_origin = ctx.column_exists("beliefs", "origin")
    sql = "SELECT id, origin FROM beliefs" if has_origin else "SELECT id, 'unknown' FROM beliefs"
    return [(str(i), str(o)) for i, o in ctx.require_db().execute(sql) if str(i) not in logged]


def _feedback_source_count(ctx: cc.Ctx, source: str) -> int:
    if not ctx.table_exists("feedback_history"):
        return 0
    return ctx.count("SELECT COUNT(*) FROM feedback_history WHERE source = ?", (source,))


def _meta(ctx: cc.Ctx, key: str) -> bool:
    if not ctx.table_exists("schema_meta"):
        return False
    return ctx.count("SELECT COUNT(*) FROM schema_meta WHERE key = ?", (key,)) > 0


# --- A268 [store] marker identity sidecar ----------------------------------
# db_paths.store_identity: a git:<name> marker store seeds `identity` beside
# stores/<name>/memory.db. A repo store seeds its own identity too, so only
# the named-store location tells the marker apart. Marker stores in plain
# folders outside `.git` are not counted.

def _a268(ctx: cc.Ctx) -> cc.Result:
    if not _is_named(ctx):
        return cc.silent(0, note="not a git-name store; marker folders outside .git not counted")
    has = 1 if ctx.sidecar("identity").is_file() else 0
    return cc.from_count(has, {"named_store": 1, "identity_sidecar": has})


# --- A269 clamp-ghosts -----------------------------------------------------
# clamp_ghosts.CLAMP_SOURCE: one feedback_history row per clamped belief.

def _a269(ctx: cc.Ctx) -> cc.Result:
    n = _feedback_source_count(ctx, "clamp_ghosts")
    d: dict[str, int] = {}
    if n:
        d["beliefs"] = ctx.count(
            "SELECT COUNT(DISTINCT belief_id) FROM feedback_history WHERE source='clamp_ghosts'")
    return cc.from_count(n, d)


# --- A270 doctor --classify-orphans -----------------------------------------
# cli._cmd_doctor_classify_orphans writes ~/.aelfrice/llm-classify-consented
# with scope stored_beliefs before the paid batch. classify_orphans itself
# rewrites type/origin via update_belief with no audit row, so the consent
# scope is the only trace; it proves consent, not a completed batch.

def _a270(ctx: cc.Ctx) -> cc.Result:
    p = ctx.global_file("llm-classify-consented")
    if not p.is_file():
        return cc.silent(0, {"sentinel": 0})
    try:
        data: Any = json.loads(p.read_text("utf-8", "replace"))
    except ValueError:
        return cc.silent(0, {"sentinel": 1, "malformed": 1})
    scopes: Any = data.get("scopes") if isinstance(data, dict) else None  # pyright: ignore[reportUnknownMemberType]
    stored = 1 if isinstance(scopes, list) and "stored_beliefs" in scopes else 0
    return cc.from_count(stored, {"sentinel": 1, "stored_beliefs_scope": stored},
                         note="consent scope only; batch completion leaves no row")


# --- A273 doctor --promote-retention ----------------------------------------
# doctor.promote_retention: retention_class='fact' plus feedback_history
# source='retention_promotion' (valence 0). Source string used nowhere else.

def _a273(ctx: cc.Ctx) -> cc.Result:
    n = _feedback_source_count(ctx, "retention_promotion")
    d: dict[str, int] = {}
    if n and ctx.column_exists("beliefs", "retention_class"):
        d["still_fact"] = ctx.count(
            "SELECT COUNT(DISTINCT b.id) FROM beliefs b JOIN feedback_history f "
            "ON f.belief_id=b.id WHERE f.source='retention_promotion' AND b.retention_class='fact'")
    return cc.from_count(n, d)


# --- A280 doctor telemetry stats -------------------------------------------
# `aelf doctor` writes nothing; it reads two ring buffers (cap 1000) the
# hooks write at db_path().parent/telemetry/: search_tool_hook.jsonl
# (hook_search_tool._append_telemetry, Bash branch only) and
# user_prompt_submit.jsonl (hook._write_telemetry, every UPS fire that emits
# a block). Doctor derives the path as <git-common-dir>/aelfrice/telemetry/
# (doctor._derive_telemetry_path), which equals the writer path for a
# repository store only. Named, legacy, global and user stores write
# beside their own db, which doctor never reads. Doctor itself writes no
# row when it prints these sections, so the probe is untraceable on every
# store; the detail counts the rows a doctor run would have read.

_TELEMETRY_CAP = 1000


def _int_field(r: dict[str, Any], key: str) -> int:
    v = r.get(key, 0)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _a280(ctx: cc.Ctx) -> cc.Result:
    tdir = ctx.store_dir / "telemetry"
    st = tdir / "search_tool_hook.jsonl"
    ups = tdir / "user_prompt_submit.jsonl"
    d: dict[str, int] = {"st_rows": cc.jsonl_count(st), "ups_rows": cc.jsonl_count(ups)}
    d["st_noise"] = cc.jsonl_count(
        st, lambda r: _int_field(r, "injected_l0") == 0 and _int_field(r, "injected_l1") == 0)
    d["st_at_cap"] = 1 if d["st_rows"] >= _TELEMETRY_CAP else 0
    d["ups_at_cap"] = 1 if d["ups_rows"] >= _TELEMETRY_CAP else 0
    is_repo = ctx.layout == "repo"
    d["doctor_reads"] = 1 if is_repo else 0
    if not is_repo:
        return cc.Result("untraceable", None, d, "doctor never reads this store telemetry")
    return cc.Result("untraceable", None, d,
                     "doctor only reads these rows; the hooks wrote them, not doctor")


# --- A281 doctor auto-migration of legacy per-project DBs ------------------
# doctor._check_legacy_schema_dbs flags ~/.aelfrice/projects/*/memory.db
# only when a beliefs table exists, has no `origin` column and holds rows.
# doctor._auto_migrate_legacy_dbs -> migrate.migrate_in_place renames the
# file to memory.db.pre-v1x.bak and writes a fresh modern-schema memory.db
# whose rows carry origin='unknown' and no ingest_log reference.
# Three cases per legacy store:
#   eligible      legacy schema with rows: doctor would migrate it, has not
#   empty_legacy  legacy schema, zero rows: doctor skips it by design
#   migrated      origin column present plus unlogged origin='unknown' rows
# Fired if migrated; silent otherwise (detail eligible = 1 when doctor would migrate).

def _a281(ctx: cc.Ctx) -> cc.Result:
    if not _is_legacy(ctx):
        return cc.silent(0, note="applies to legacy per-project stores only")
    d = {"eligible": 0, "empty_legacy": 0, "migrated": 0, "unlogged_unknown": 0,
         "rows": 0, "has_origin": 0}
    if ctx.db is None:
        return cc.silent(0, d, note="no db")
    if not ctx.table_exists("beliefs"):
        return cc.silent(0, d, note="no beliefs table; doctor skips")
    d["rows"] = ctx.count("SELECT COUNT(*) FROM beliefs")
    if not ctx.column_exists("beliefs", "origin"):
        if d["rows"] > 0:
            d["eligible"] = 1
            return cc.silent(0, d, note="legacy schema with rows; doctor would migrate")
        d["empty_legacy"] = 1
        return cc.silent(0, d, note="legacy schema, zero rows; doctor skips by design")
    d["has_origin"] = 1
    logged = _logged_belief_ids(ctx)
    unk = sum(1 for (i,) in ctx.require_db().execute(
        "SELECT id FROM beliefs WHERE origin='unknown'") if str(i) not in logged)
    d["unlogged_unknown"] = unk
    if unk > 0:
        d["migrated"] = 1
        return cc.fired(unk, d, note="modern schema with unlogged unknown-origin rows")
    return cc.silent(0, d, note="modern schema, no migration trace")


# --- A283 aelf migrate (legacy global -> per-project) ----------------------
# migrate.migrate: target.insert_belief with no ingest_log row. Ambiguous
# with a derivation-worker crash between insert and stamp.

def _a283(ctx: cc.Ctx) -> cc.Result:
    if ctx.is_global or _is_legacy(ctx) or _is_named(ctx):
        return cc.silent(0, note="applies to repository stores only")
    if ctx.db is None or not ctx.table_exists("beliefs"):
        return cc.silent(0, note="no db")
    rows = [(i, o) for i, o in _unlogged_beliefs(ctx) if o not in _UNLOGGED_OTHER_ORIGINS]
    detail: dict[str, int] = {}
    for _, o in rows:
        k = o if o in _ORIGINS else "other"
        detail[k] = detail.get(k, 0) + 1
    return cc.from_shared_count(len(rows), detail,
                                note="ambiguous: a worker crash before stamping looks the same")


# --- A284 aelf store split -------------------------------------------------
# store_split._write: migrate.copy_belief (insert_belief, no ingest_log row)
# into a marker store, keeping the source id. Count unlogged beliefs in a
# git:<name> store whose id is also in the repository store.

def _a284(ctx: cc.Ctx) -> cc.Result:
    if not _is_named(ctx):
        return cc.silent(0, note="applies to git-name stores; marker folders outside .git not counted")
    parent = _named_parent_db(ctx)
    if parent is None or ctx.db is None or not ctx.table_exists("beliefs"):
        return cc.silent(0, note="no repository store beside this named store")
    cand = {i for i, _ in _unlogged_beliefs(ctx)}
    pconn = cc.open_ro(parent)
    try:
        hit = 0
        for bid in cand:
            if pconn.execute("SELECT 1 FROM beliefs WHERE id = ?", (bid,)).fetchone():
                hit += 1
    finally:
        pconn.close()
    return cc.from_count(hit, {"unlogged": len(cand), "also_in_repo_store": hit})


# --- A285 feed log ---------------------------------------------------------
# feed_log.append: feed.jsonl rows {ts, event, ...} beside memory.db.
# Rotated feed.<ts>.jsonl.archive files are not read.

_FEED_EVENTS = frozenset({
    "belief.ingested", "belief.locked", "wonder.promoted", "feedback.applied",
    "feedback.locked_not_applied", "wonder.gc",
})


def _str_field(r: dict[str, Any], key: str) -> str | None:
    v = r.get(key)
    return v if isinstance(v, str) else None


def _a285(ctx: cc.Ctx) -> cc.Result:
    d = cc.jsonl_tally(ctx.sidecar("feed.jsonl"), lambda r: _str_field(r, "event"), _FEED_EVENTS)
    return cc.from_count(sum(d.values()), d)


# --- A287 content-hash dedup and UNIQUE(content_hash) ----------------------
# store._maybe_consolidate_content_hash_duplicates writes source_type
# 'consolidation_migration' rows; at runtime insert_or_corroborate turns a
# content-hash (or id) hit into a corroboration row. wonder_ingest rows are
# written on insert, not on a hit, so they are left out.

_HIT_SOURCES = ("transcript_ingest", "commit_ingest", "cli_remember", "mcp_remember",
                "claude_memory_mirror", "filesystem_ingest")


def _a287(ctx: cc.Ctx) -> cc.Result:
    if not ctx.column_exists("belief_corroborations", "source_type"):
        return cc.silent(0, note="no corroboration table")
    mig = ctx.count(
        "SELECT COUNT(*) FROM belief_corroborations WHERE source_type='consolidation_migration'")
    marks = ",".join("?" * len(_HIT_SOURCES))
    hits = ctx.count(
        f"SELECT COUNT(*) FROM belief_corroborations WHERE source_type IN ({marks})", _HIT_SOURCES)
    dup = ctx.count(
        "SELECT COUNT(*) FROM (SELECT 1 FROM beliefs GROUP BY content_hash HAVING COUNT(*)>1)")
    return cc.from_count(mig + hits, {"consolidation_migration": mig, "runtime_hit": hits,
                                      "duplicate_groups": dup})


# --- A289 federation version vectors ---------------------------------------
# store._bump_belief_version / edge / log version writers: one row per
# (row, scope) in belief_versions, edge_versions, log_versions.

def _a289(ctx: cc.Ctx) -> cc.Result:
    d: dict[str, int] = {}
    for t in ("belief_versions", "edge_versions", "log_versions"):
        d[t] = ctx.count(f"SELECT COUNT(*) FROM {t}") if ctx.table_exists(t) else 0
    if ctx.table_exists("belief_versions"):
        d["belief_counter_gt1"] = ctx.count(
            "SELECT COUNT(*) FROM belief_versions WHERE counter > 1")
    return cc.from_count(d["belief_versions"] + d["edge_versions"] + d["log_versions"], d)


# --- A292 guarded one-shot migrations --------------------------------------
# store._run_guarded_migration writes schema_meta 'migration_failed:<pass>'
# only when a pass raises, and deletes it when the pass later succeeds. A
# successful guarded run leaves no row of its own, so zero markers means no
# live failure and cannot show whether the guard ran: untraceable. Beliefs
# created on or after the 4.3.0 release that shipped the guard are reported
# so a reader can tell which stores a guarded build has opened.

GUARD_RELEASE = "2026-08-13"


def _a292(ctx: cc.Ctx) -> cc.Result:
    d = {"has_schema_meta": 0, "failure_markers": 0, "meta_keys": 0,
         "beliefs_post_guard": 0, "beliefs_total": 0}
    if ctx.db is None:
        return cc.Result("untraceable", None, d, "no db")
    if ctx.table_exists("schema_meta"):
        d["has_schema_meta"] = 1
        d["failure_markers"] = ctx.count(
            "SELECT COUNT(*) FROM schema_meta WHERE key LIKE 'migration_failed:%'")
        d["meta_keys"] = ctx.count("SELECT COUNT(*) FROM schema_meta")
    if ctx.table_exists("beliefs"):
        d["beliefs_total"] = ctx.count("SELECT COUNT(*) FROM beliefs")
        if ctx.column_exists("beliefs", "created_at"):
            d["beliefs_post_guard"] = ctx.count(
                "SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (GUARD_RELEASE,))
    if d["failure_markers"] > 0:
        return cc.fired(d["failure_markers"], d, note="live failure markers")
    return cc.Result("untraceable", None, d, "success path writes nothing distinguishable")


# --- A293 hook audit log ---------------------------------------------------
# hook_audit._append_audit: hook_audit.jsonl (+ .1) rows keyed by `hook`;
# a rotation opens the new file with hook='audit_rotation'.

_AUDIT_HOOKS = frozenset({
    "user_prompt_submit", "session_start", "sentiment_feedback", "transcript_logger",
    "sidecar_warm", "audit_rotation",
})


def _a293(ctx: cc.Ctx) -> cc.Result:
    paths = cc.jsonl_paths(ctx.sidecar("hook_audit.jsonl"))
    d = cc.jsonl_tally(paths, lambda r: _str_field(r, "hook") or "nohook",
                       _AUDIT_HOOKS | {"nohook"})
    d["rotated_slot"] = 1 if ctx.sidecar("hook_audit.jsonl.1").is_file() else 0
    return cc.from_count(sum(v for k, v in d.items() if k != "rotated_slot"), d)


# --- A294 classifier version and rule-set digest stamp ---------------------
# store.record_ingest(classifier_version=..., rule_set_hash=...) from the
# ingest, scanner, CLI, triple and worker paths (#1658). Rows from before
# the stamp, and the legacy synthesis, leave both NULL.

def _a294(ctx: cc.Ctx) -> cc.Result:
    if not ctx.column_exists("ingest_log", "classifier_version"):
        return cc.silent(0, note="no stamp columns")
    st, nul, vers = ctx.counts(
        "SELECT SUM(classifier_version IS NOT NULL AND rule_set_hash IS NOT NULL), "
        "SUM(classifier_version IS NULL), COUNT(DISTINCT classifier_version) FROM ingest_log")
    return cc.from_count(st, {"stamped": st, "unstamped": nul, "distinct_versions": vers})


# --- A295 legacy ingest_log synthesis --------------------------------------
# store._maybe_synthesize_legacy_log_rows: source_kind='legacy_unknown'.

def _a295(ctx: cc.Ctx) -> cc.Result:
    if not ctx.column_exists("ingest_log", "source_kind"):
        return cc.silent(0, note="no ingest_log")
    return cc.from_count(ctx.count(
        "SELECT COUNT(*) FROM ingest_log WHERE source_kind='legacy_unknown'"))


# --- A297 MCP surface cleanup report ---------------------------------------
# mcp_cleanup.maybe_clean_up_mcp writes ~/.aelfrice/mcp-surface-removed
# once a scan completes. --remove-mcp-config backups sit in host configs,
# outside any store or global dir.

def _a297(ctx: cc.Ctx) -> cc.Result:
    has = 1 if ctx.global_file("mcp-surface-removed").is_file() else 0
    return cc.from_count(has, {"sentinel": has})


# --- A298 open-time lock expiry sweep --------------------------------------
# store.sweep_expired_locks writes feedback_history source='lock:expire' per
# flip and schema_meta lock_expiry_sweep_at when it flipped any. It writes
# only when a row has lock_expires_at <= now and lock_level='user', so a
# store with no windowed lock gives the sweep nothing to do. Keys:
#   windowed       beliefs with lock_expires_at set (any lock_level)
#   windowed_user  windowed rows still lock_level='user'
#   due_user       windowed_user rows already past expiry (sweep owed)
#   flipped        windowed rows with lock_level='none' (sweep or unlock)
# Fired when expire rows or the marker exist; error when a due lock was
# never swept; untraceable when no windowed lock exists; silent otherwise.

def _a298_detail(ctx: cc.Ctx, now: str) -> dict[str, int]:
    d: dict[str, int] = {"has_col": 0, "windowed": 0, "windowed_user": 0,
                         "due_user": 0, "flipped": 0, "expire_rows": 0,
                         "sweep_marker": 0}
    if ctx.column_exists("beliefs", "lock_expires_at"):
        d["has_col"] = 1
        w, wu, du, fl = ctx.counts(
            "SELECT COUNT(*), SUM(lock_level='user'), "
            "SUM(lock_level='user' AND lock_expires_at <= ?), SUM(lock_level='none') "
            "FROM beliefs WHERE lock_expires_at IS NOT NULL", (now,))
        d.update(windowed=w, windowed_user=wu, due_user=du, flipped=fl)
    d["expire_rows"] = _feedback_source_count(ctx, "lock:expire")
    d["sweep_marker"] = 1 if _meta(ctx, "lock_expiry_sweep_at") else 0
    return d


def _a298(ctx: cc.Ctx) -> cc.Result:
    if not ctx.table_exists("beliefs"):
        return cc.untraceable("no beliefs table")
    d = _a298_detail(ctx, datetime.now(timezone.utc).isoformat())
    if d["expire_rows"] > 0 or d["sweep_marker"] > 0:
        return cc.fired(d["expire_rows"], d)
    if d["due_user"] > 0:
        return cc.Result("error", d["due_user"], d, "due windowed locks not swept")
    if d["windowed"] == 0:
        return cc.Result("untraceable", None, d, "no windowed lock; sweep has no input")
    return cc.silent(0, d, "windows exist, none due")


# --- A299 per-fire BM25F sidecar outcome audit field -----------------------
# hook._write_audit_record(sidecar_outcome=...) on hook_audit rows, only when
# index work ran; values fresh | incremental | full_rebuild.

def _a299(ctx: cc.Ctx) -> cc.Result:
    paths = cc.jsonl_paths(ctx.sidecar("hook_audit.jsonl"))
    d = cc.jsonl_tally(paths, lambda r: _str_field(r, "sidecar_outcome"),
                       {"fresh", "incremental", "full_rebuild"})
    return cc.from_count(sum(d.values()), d)


# --- A301 project_context stamping and backfill ----------------------------
# Insert stamps project_context_default (the store identity) on
# project-scope unlocked rows; the one-shot backfill stamps old rows and
# writes project_context_backfill_complete. migrate and store split also
# write the column, so this counts stamped rows of any origin.

def _a301(ctx: cc.Ctx) -> cc.Result:
    if not ctx.column_exists("beliefs", "project_context"):
        return cc.silent(0, note="no project_context column")
    st, empty = ctx.counts(
        "SELECT SUM(project_context IS NOT NULL AND project_context <> ''), "
        "SUM(project_context IS NULL OR project_context = '') FROM beliefs")
    marker = 1 if _meta(ctx, "project_context_backfill_complete") else 0
    return cc.from_count(st, {"stamped": st, "empty": empty, "backfill_marker": marker})


# --- A303 rebuild diagnostic log -------------------------------------------
# rebuild_log._append_rebuild_log_record: rebuild_logs/<session>.jsonl, one
# row per rebuild or UPS fire; UPS rows carry all-null scores; a size cap
# appends {"truncated": true}.

def _has_score(c: Any) -> bool:
    sc: Any = c.get("scores") if isinstance(c, dict) else None  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(sc, dict):
        return False
    return any(v is not None for v in sc.values())  # pyright: ignore[reportUnknownVariableType]


def _rebuild_kind(r: dict[str, Any]) -> str | None:
    if r.get("truncated") is True:
        return "truncated"
    cands: Any = r.get("candidates")
    if not isinstance(cands, list):
        return "other"
    scored = any(_has_score(c) for c in cands)  # pyright: ignore[reportUnknownVariableType]
    return "scored" if scored else "null_scores"


def _a303(ctx: cc.Ctx) -> cc.Result:
    d = ctx.sidecar("rebuild_logs")
    files = sorted(d.glob("*.jsonl")) if d.is_dir() else []
    tally = cc.jsonl_tally(files, _rebuild_kind, {"scored", "null_scores", "truncated"})
    rows = sum(v for k, v in tally.items() if k != "truncated")
    tally["files"] = len(files)
    return cc.from_count(rows, tally)


# --- A306 retired-content revival ------------------------------------------
# store.insert_or_corroborate is the only writer of feedback source
# 'reassert:revive' (models.FEEDBACK_SOURCE_REASSERT_REVIVE). It fires only
# when a cli_remember / mcp_remember ingest (aelf lock, aelf remember) hits a
# row with valid_to set; it clears valid_to and writes a corroboration row.
#   corr_on_retired  explicit corroborations on beliefs retired now
#   missed_post      of those, ingested after valid_to and on or after the
#                    4.3.0 release that shipped revival (a re-assertion the
#                    shipped code should have revived)
# Fired when revive > 0; silent otherwise; untraceable with no belief schema.

_REVIVE_RELEASE = "2026-08-13"
_EXPLICIT = ("cli_remember", "mcp_remember")


def _a306_corroborations(ctx: cc.Ctx, d: dict[str, int]) -> None:
    d["explicit_corr"] = ctx.count(
        "SELECT COUNT(*) FROM belief_corroborations WHERE source_type IN (?,?)", _EXPLICIT)
    d["explicit_corr_post"] = ctx.count(
        "SELECT COUNT(*) FROM belief_corroborations WHERE source_type IN (?,?) "
        "AND ingested_at >= ?", (*_EXPLICIT, _REVIVE_RELEASE))
    if not ctx.column_exists("beliefs", "valid_to"):
        return
    join = ("SELECT COUNT(*) FROM belief_corroborations c JOIN beliefs b ON b.id=c.belief_id "
            "WHERE b.valid_to IS NOT NULL AND c.source_type IN (?,?)")
    d["corr_on_retired"] = ctx.count(join, _EXPLICIT)
    d["missed_post"] = ctx.count(
        join + " AND c.ingested_at > b.valid_to AND c.ingested_at >= ?",
        (*_EXPLICIT, _REVIVE_RELEASE))


def _a306(ctx: cc.Ctx) -> cc.Result:
    if ctx.db is None or not ctx.table_exists("beliefs"):
        return cc.untraceable("no belief schema")
    d: dict[str, int] = {"revive": _feedback_source_count(ctx, "reassert:revive")}
    if ctx.column_exists("beliefs", "valid_to"):
        d["retired"] = ctx.count("SELECT COUNT(*) FROM beliefs WHERE valid_to IS NOT NULL")
    if ctx.table_exists("feedback_history"):
        d["user_retired"] = ctx.count(
            "SELECT COUNT(*) FROM feedback_history WHERE source IN "
            "('user_retired','user_retired_force')")
        d["user_restored"] = _feedback_source_count(ctx, "user_restored")
    if ctx.table_exists("belief_corroborations"):
        _a306_corroborations(ctx, d)
    if ctx.table_exists("ingest_log"):
        d["explicit_ingest_post"] = ctx.count(
            "SELECT COUNT(*) FROM ingest_log WHERE source_kind IN (?,?) AND ts >= ?",
            (*_EXPLICIT, _REVIVE_RELEASE))
    if ctx.column_exists("beliefs", "created_at"):
        d["beliefs_post"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (_REVIVE_RELEASE,))
    return cc.from_count(d["revive"], d)


# --- A307 session-delta telemetry writer -----------------------------------
# telemetry.emit_session_delta appends one v=1 row to ~/.aelfrice/telemetry.jsonl.

def _a307(ctx: cc.Ctx) -> cc.Result:
    d = cc.jsonl_split_count(ctx.global_file("telemetry.jsonl"), lambda r: "v" in r)
    return cc.from_count(sum(d.values()), d)


# --- A308 sessions table writer --------------------------------------------
# Only hook_commit_ingest._record_triples writes sessions (model
# 'commit-ingest'); MemoryStore.create_session has no caller.

def _a308(ctx: cc.Ctx) -> cc.Result:
    if not ctx.table_exists("sessions"):
        return cc.silent(0, note="no sessions table")
    d = ctx.group_count("SELECT model, COUNT(*) FROM sessions GROUP BY model", {"commit-ingest"})
    if ctx.column_exists("beliefs", "session_id"):
        d["belief_sessions_without_row"] = ctx.count(
            "SELECT COUNT(DISTINCT b.session_id) FROM beliefs b LEFT JOIN sessions s "
            "ON s.id = b.session_id WHERE b.session_id IS NOT NULL AND s.id IS NULL")
    n = sum(v for k, v in d.items() if k != "belief_sessions_without_row")
    return cc.from_count(n, d)


# --- A309 store write path settings ----------------------------------------
# MemoryStore open sets PRAGMA journal_mode=WAL (file header bytes 18-19 = 2)
# and seeds schema_meta store_generation.

def _a309(ctx: cc.Ctx) -> cc.Result:
    f = ctx.db_file
    if f is None or not f.is_file():
        return cc.silent(0, note="no db")
    with f.open("rb") as fh:
        hdr = fh.read(20)
    wal = 1 if len(hdr) >= 20 and hdr[18] == 2 and hdr[19] == 2 else 0
    gen = 1 if (ctx.db is not None and _meta(ctx, "store_generation")) else 0
    return cc.from_count(wal, {"wal_header": wal, "store_generation": gen})


PROBES: list[cc.Probe] = [
    U("A038", "academic benchmark canonical cut",
      "writes results JSON to an explicit out path in the repo; no store or log trace"),
    U("A039", "bench-gate null-model floors",
      "test-suite gate over lab corpora; no store or log trace"),
    P("A268", "store marker identity sidecar", "sidecar", cc.safe(_a268)),
    P("A269", "aelf clamp-ghosts", "store", cc.safe(_a269)),
    P("A270", "doctor classify-orphans consent", "global-log", cc.safe(_a270)),
    U("A271", "doctor gc-filesystem-corroboration",
      "delete-only pass with no audit row; residue cannot tell ran from never needed"),
    U("A272", "doctor gc-orphan-feedback",
      "delete-only pass with no audit row; residue cannot tell ran from never needed"),
    P("A273", "doctor promote-retention", "store", cc.safe(_a273)),
    U("A274", "doctor prune-dormant",
      "deletes whole legacy DB files after a prompt; nothing is written"),
    U("A275", "doctor repair-utc-created-at",
      "rewrites created_at in place and rechains spine edges; no marker or audit row"),
    U("A276", "doctor replay and derive-pending",
      "replay is read-only; derive-pending stamps ingest_log exactly as the inline worker does"),
    U("A277", "doctor retire-scheduled-prompts",
      "soft delete sets valid_to like aelf retire; no source label"),
    U("A278", "doctor dangling-edge count", "read-only report; writes nothing"),
    U("A279", "doctor hook and settings audit",
      "report is read-only; fix prune edits host settings outside any store"),
    P("A280", "doctor telemetry stats", "sidecar", cc.safe(_a280)),
    P("A281", "doctor legacy auto-migration", "store", cc.safe(_a281)),
    U("A282", "aelf introspect", "read-only view; writes nothing"),
    P("A283", "aelf migrate", "store", cc.safe(_a283)),
    P("A284", "aelf store split", "store", cc.safe(_a284)),
    P("A285", "belief-write feed log", "sidecar", cc.safe(_a285)),
    U("A286", "config discovery home warning",
      "notice printed on SessionStart stdout only; no log row"),
    P("A287", "content-hash dedup", "store", cc.safe(_a287)),
    U("A288", "corroboration source dedup",
      "INSERT OR IGNORE skip writes nothing; one-shot dedup deletes and stamps the marker on fresh stores too"),
    P("A289", "federation version vectors", "store", cc.safe(_a289)),
    U("A290", "filesystem re-read records no corroboration",
      "the skip returns without a write; absence is the only effect"),
    U("A291", "git common-dir lookup cache", "in-process memo; no durable trace"),
    P("A292", "guarded migrations failure markers", "store", cc.safe(_a292)),
    P("A293", "hook audit log", "sidecar", cc.safe(_a293)),
    P("A294", "ingest classifier stamp", "store", cc.safe(_a294)),
    P("A295", "legacy ingest_log synthesis", "store", cc.safe(_a295)),
    U("A296", "legacy one-shot backfills and column drops",
      "markers are stamped on fresh stores too and backfilled rows look like live rows"),
    P("A297", "MCP surface cleanup report", "global-log", cc.safe(_a297)),
    P("A298", "open-time lock expiry sweep", "store", cc.safe(_a298)),
    P("A299", "sidecar outcome audit field", "sidecar", cc.safe(_a299)),
    U("A300", "portable advisory file lock",
      "lock files predate the portable module and a byte copy drops mtimes"),
    P("A301", "project_context stamping", "store", cc.safe(_a301)),
    U("A302", "read-only store handle", "a read-only open writes nothing by design"),
    P("A303", "rebuild diagnostic log", "sidecar", cc.safe(_a303)),
    U("A304", "rebuilder threshold_fraction",
      "config value is parsed and never acts; no trace"),
    U("A305", "regime classifier", "hidden read-only report; writes nothing"),
    P("A306", "retired-content revival", "store", cc.safe(_a306)),
    P("A307", "session-delta telemetry", "global-log", cc.safe(_a307)),
    P("A308", "sessions table writer", "store", cc.safe(_a308)),
    P("A309", "store write path settings", "store", cc.safe(_a309)),
    U("A310", "write-log-authoritative gate",
      "env-gated check that raises instead of writing; no record of the flag"),
]

TRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind != "untraceable")
UNTRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind == "untraceable")
