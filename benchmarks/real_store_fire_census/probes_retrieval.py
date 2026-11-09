"""Census probes for the retrieval subsystem (inventory A217-A267).

Every probe returns integers and fixed labels only; no prompt, belief or
path text leaves a probe. Belief ids read from the audit are used in
memory for set membership and never reach a result.

Retrieval is a read path. Its per-lane counters (`retrieval.LaneTelemetry`)
live in process memory and are never persisted, and the hook audit labels
every non-locked hit 'L1' whatever lane produced it. What does persist:

- hook_audit.jsonl (store dir), written by hook._write_hook_audit_record:
  hook, ts, prompt_prefix (200 chars), rendered_block, n_beliefs, n_locked,
  beliefs[] (id, lane L0|L1, snippet), expansion_gate_reason,
  expansion_gate_skipped_bfs, order_policy, sidecar_outcome,
  prompt_shape_gate_skip.
- memory.db.bm25f (bm25.BM25IndexCache._write_sidecar): outer header
  AELFB25S with a generation, inner BM25Index.serialize header with
  anchor_weight, then k1, b, avgdl and k3 from inner version 3, and after
  the CSR arrays per_field / b_anchor.
- .hrr_struct_index/struct.npy + meta.npz (hrr_index.HRRStructIndex.save).
- meta_belief_signal_posteriors rows (store.update_meta_belief).
- belief_entities rows (store insert path and entity backfill).
- edges rows: TEMPORAL_NEXT (temporal spine writer), POTENTIALLY_STALE
  (`aelf doctor --detect-stale`), and the probed kinds the HRR expand lane
  walks.
- rebuild_logs/<session>.jsonl (rebuild_log._append_rebuild_log_record),
  where a rebuild_v14 row carries per-candidate scores and a decision.
- logs/hook-failures.log, when a host wrapper redirects hook stderr there.

Some lanes leave no direct mark but a necessary condition can be read: a
lane whose input is absent is `silent`, and one whose input is present but
whose fire is never recorded is `untraceable`. Items whose work leaves
none of these behind are untraceable probes.
"""
from __future__ import annotations

import ast
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import core as cc
from .core import Probe, Result

# --- fixed vocabularies copied from the code (tests pin them to it) --------

AUDIT_FILENAME = "hook_audit.jsonl"            # hook_audit.AUDIT_FILENAME
HOOK_UPS = "user_prompt_submit"                # hook.AUDIT_HOOK_USER_PROMPT_SUBMIT
HOOK_SESSION_START = "session_start"           # hook.AUDIT_HOOK_SESSION_START
SIDECAR_OUTCOMES = ("fresh", "incremental", "full_rebuild")  # sidecar_outcome
ORDER_POLICIES = ("lane", "score_desc", "locks_last")       # retrieval.ORDER_POLICIES
EDGE_TYPES = frozenset({                       # models.EDGE_TYPES
    "SUPPORTS", "CITES", "CONTRADICTS", "SUPERSEDES", "RELATES_TO",
    "DERIVED_FROM", "IMPLEMENTS", "TEMPORAL_NEXT", "TESTS", "RESOLVES",
})
ENTITY_KINDS = ("file_path", "url", "error_code", "version", "branch",
                "identifier", "noun_phrase")   # entity_extractor.KIND_*
META_KEYS = (                                  # retrieval.META_*_KEY
    "meta:retrieval.temporal_half_life_seconds",
    "meta:retrieval.bm25f_anchor_weight",
    "meta:retrieval.bfs_depth_budget",
)
BM25_SIDECAR_NAME = "memory.db.bm25f"          # db_path + bm25._SIDECAR_SUFFIX
SIDECAR_MAGIC = b"AELFB25S"                    # bm25._SIDECAR_MAGIC
SIDECAR_VERSION = 1                            # bm25._SIDECAR_VERSION
SERIALIZE_MAGIC = b"AELFBM25"                  # bm25._SERIALIZE_MAGIC
SERIALIZE_VERSION = 6                          # bm25._SERIALIZE_VERSION
K3_MIN_INNER_VERSION = 3                       # first serialize version with k3
HRR_DIRNAME = ".hrr_struct_index"              # hrr_index._PERSIST_DIRNAME
HRR_STRUCT = "struct.npy"
HRR_META = "meta.npz"
HRR_DEFAULT_DIM = 512                          # hrr.DEFAULT_DIM
HOOK_FAILURES_LOG = ("logs", "hook-failures.log")
# retrieval.warn_placeholder_flags message body (fixed text, no user text).
PLACEHOLDER_WARNING = "recognised but the corresponding lane has not yet shipped"
PREFIX_CAP = 200                               # hook.AUDIT_PROMPT_PREFIX_CAP
QUERY_ENTITY_CAP = 16                          # retrieval.DEFAULT_QUERY_ENTITY_CAP
LOCK_USER = "user"                             # models.LOCK_USER
TEMPORAL_NEXT = "TEMPORAL_NEXT"
POTENTIALLY_STALE = "POTENTIALLY_STALE"
# Edge kinds hrr_expand.expand_seeds probes (hrr_expand.hrr_expand_edge_types).
HRR_EXPAND_KINDS = ("CITES", "CONTRADICTS", "IMPLEMENTS", "SUPERSEDES", "SUPPORTS", "TESTS")

# Ship dates from the changelog, for features whose trace only exists after.
STRUCT_LANE_SHIP = "2026-05-09"   # 2.1.0: structural lane default on (#154)
BM25F_DEFAULT_ON = "2026-05-07"   # 1.7.0: BM25F anchors default on
SIDECAR_FIELD_SHIP = "2026-08-13"  # 4.3.0: audit sidecar_outcome field (#1407)
STALE_RERANK_SHIP = "2026-08-13"  # 4.3.0: edge-type rerank wired (#1207)
SPINE_LANE_SHIP = "2026-07-07"    # 4.0.0: spine lane on the live retrieve path

_HASH_N_RE = re.compile(r"#\d+")                # retrieval._HASH_N_LITERAL_RE
_STRUCT_MARKER_RE = re.compile(r"^([A-Z][A-Z_]*):(\S.*)$")  # hrr_index
_REF_LINE_RE = re.compile(r'(?m)^  ref [^\s:]+: "')         # lock_manifest_line, indented
_COVERAGE_RE = re.compile(r"(?m)^retrieved \d+ of \d+ matching beliefs for ")  # hook._coverage_line


# --- helpers ---------------------------------------------------------------


def _audit_paths(ctx: cc.Ctx) -> list[Path]:
    return cc.jsonl_paths(ctx.sidecar(AUDIT_FILENAME))


def _audit_rows(ctx: cc.Ctx) -> Iterator[dict[str, Any]]:
    for p in _audit_paths(ctx):
        yield from cc.iter_jsonl(p)


def _ups(row: dict[str, Any]) -> bool:
    return row.get("hook") == HOOK_UPS


def _injecting(row: dict[str, Any]) -> bool:
    return row.get("hook") in (HOOK_UPS, HOOK_SESSION_START)


def _int(v: Any) -> int:
    return v if type(v) is int else 0


def _l1_ids(row: dict[str, Any]) -> list[str]:
    """Injected non-locked belief ids of one audit row, in block order."""
    beliefs: Any = row.get("beliefs")
    if not isinstance(beliefs, list):
        return []
    out: list[str] = []
    for b in beliefs:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(b, dict) and b.get("lane") == "L1":  # pyright: ignore[reportUnknownMemberType]
            bid: Any = b.get("id")  # pyright: ignore[reportUnknownMemberType]
            if isinstance(bid, str):
                out.append(bid)
    return out


def _audit_result(ctx: cc.Ctx, matched: list[dict[str, Any]],
                  detail: dict[str, int], note: str = "") -> Result:
    """from_count over matched rows; on a global ctx add the attribution split."""
    if ctx.is_global:
        split = {"attributed": 0, "unattributed": 0}
        for r in matched:
            split[cc.row_attribution(r)] += 1
        detail = {**detail, **split}
    return cc.from_count(len(matched), detail, note)


def _no_audit(ctx: cc.Ctx) -> Result | None:
    if not _audit_paths(ctx):
        return cc.silent(0, {}, "no hook_audit file")
    return None


def _count_if(ctx: cc.Ctx, table: str, column: str, sql: str,
              params: tuple[Any, ...] = ()) -> int:
    """ctx.count, or 0 when the table or column is absent."""
    if not ctx.column_exists(table, column):
        return 0
    return ctx.count(sql, params)


def _parse_ts(s: object) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# --- BM25 sidecar header (struct, no numpy) --------------------------------


def _read_inner(f: Any, out: dict[str, float], full: bool) -> dict[str, float] | None:
    """The serialize blob after its magic: version, numeric header, per_field."""
    inner_ver, = struct.unpack("<I", f.read(4))
    aw, = struct.unpack("<i", f.read(4))
    out.update({"inner_version": inner_ver, "anchor_weight": aw})
    if inner_ver < K3_MIN_INNER_VERSION:
        out.update({"k1": -1.0, "b": -1.0, "k3": 0.0})
        return out
    k1, b, _avgdl, k3 = struct.unpack("<4d", f.read(32))
    n_docs, n_terms = struct.unpack("<2Q", f.read(16))
    out.update({"k1": k1, "b": b, "k3": k3, "n_docs": n_docs, "n_terms": n_terms})
    if not full:
        return out
    for _ in range(n_docs + n_terms):
        ln, = struct.unpack("<I", f.read(4))
        f.seek(ln, 1)
    f.seek(4 * n_docs + 4 * n_terms + 8 * (n_docs + 1), 1)
    nnz, = struct.unpack("<Q", f.read(8))
    f.seek(8 * nnz + 4 * nnz, 1)
    raw = f.read(9)
    if len(raw) < 9:
        return None
    out["per_field"] = raw[0]
    out["b_anchor"] = struct.unpack("<d", raw[1:9])[0]
    return out


def read_bm25_sidecar(path: Path, *, full: bool = False) -> dict[str, float] | None:
    """Numeric header fields of a BM25 sidecar, or None if absent or invalid.

    Opens the file for reading only. Never returns belief ids, vocabulary
    terms or the scope id: string sections are skipped by length. A blob
    older than inner version 3 has no k1, b or k3 (k3 reads as 0). `full`
    walks past the CSR arrays to read the per_field flag and b_anchor.
    """
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            if f.read(8) != SIDECAR_MAGIC:
                return None
            ver, = struct.unpack("<I", f.read(4))
            if ver != SIDECAR_VERSION:
                return None
            gen, = struct.unpack("<Q", f.read(8))
            scope_len, = struct.unpack("<I", f.read(4))
            f.seek(scope_len, 1)
            if f.read(8) != SERIALIZE_MAGIC:
                return None
            return _read_inner(f, {"generation": gen}, full)
    except (OSError, struct.error, ValueError):
        return None


def _sidecar(ctx: cc.Ctx, full: bool = False) -> dict[str, float] | None:
    return read_bm25_sidecar(ctx.sidecar(BM25_SIDECAR_NAME), full=full)


def read_npy_shape(path: Path) -> tuple[int, ...] | None:
    """Shape tuple from a .npy header (format 1.x/2.x/3.x), or None."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            if f.read(6) != b"\x93NUMPY":
                return None
            major = f.read(2)[0]
            hlen = struct.unpack("<H" if major == 1 else "<I",
                                 f.read(2 if major == 1 else 4))[0]
            header: Any = ast.literal_eval(f.read(hlen).decode("latin1"))
        shape: Any = header.get("shape") if isinstance(header, dict) else None  # pyright: ignore[reportUnknownMemberType]
        if isinstance(shape, tuple) and all(type(x) is int for x in shape):  # pyright: ignore[reportUnknownVariableType]
            return tuple(int(x) for x in shape)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    except (OSError, ValueError, SyntaxError, IndexError, struct.error):
        return None
    return None


# --- traceable probes ------------------------------------------------------


def p_a217(ctx: cc.Ctx) -> Result:
    if (r := _no_audit(ctx)) is not None:
        return r
    with_literal = 0
    matched: list[dict[str, Any]] = []
    for row in _audit_rows(ctx):
        if not _ups(row):
            continue
        literals = _HASH_N_RE.findall(str(row.get("prompt_prefix") or ""))
        if not literals:
            continue
        with_literal += 1
        beliefs: Any = row.get("beliefs") or []
        if any(isinstance(b, dict) and b.get("lane") == "L1"  # pyright: ignore[reportUnknownMemberType]
               and any(lit in str(b.get("snippet") or "") for lit in literals)  # pyright: ignore[reportUnknownMemberType]
               for b in beliefs):  # pyright: ignore[reportUnknownVariableType]
            matched.append(row)
    return _audit_result(ctx, matched, {"ups_rows_with_literal": with_literal},
                         "upper bound: audit lane L1 also covers L2.5 BFS spine hits")


def p_a219(ctx: cc.Ctx) -> Result:
    if (r := _no_audit(ctx)) is not None:
        return r
    d = {"evaluated": 0, "narrow": 0, "broad": 0, "override": 0}
    matched: list[dict[str, Any]] = []
    for row in _audit_rows(ctx):
        if not _ups(row):
            continue
        reason = row.get("expansion_gate_reason")
        if isinstance(reason, str):
            d["evaluated"] += 1
            if reason == "narrow":
                d["narrow"] += 1
            elif reason.startswith("broad:"):
                d["broad"] += 1
            else:
                d["override"] += 1
        if row.get("expansion_gate_skipped_bfs") is True:
            matched.append(row)
    return _audit_result(ctx, matched, d, "count = fires where the gate turned BFS off")


def p_a220(ctx: cc.Ctx) -> Result:
    if not ctx.table_exists("meta_belief_signal_posteriors"):
        return cc.silent(0, {}, "no meta_belief_signal_posteriors table")
    ph = ",".join("?" * len(META_KEYS))
    d = ctx.group_count(
        "SELECT meta_key, COUNT(*) FROM meta_belief_signal_posteriors "
        f"WHERE meta_key IN ({ph}) GROUP BY meta_key", META_KEYS, META_KEYS)
    installed = 0
    if ctx.table_exists("meta_beliefs"):
        installed = ctx.count(
            f"SELECT COUNT(*) FROM meta_beliefs WHERE key IN ({ph})", META_KEYS)
    return cc.from_count(sum(d.values()), {**d, "installed": installed},
                         "count = posterior rows a retrieval consumer updated")


def _store_generation(ctx: cc.Ctx) -> int:
    if not ctx.column_exists("schema_meta", "value"):
        return -1
    row = ctx.require_db().execute(
        "SELECT value FROM schema_meta WHERE key = 'store_generation'").fetchone()
    return int(row[0]) if row and str(row[0]).isdigit() else -1


def p_a222(ctx: cc.Ctx) -> Result:
    h = _sidecar(ctx)
    if h is None:
        return cc.silent(0, {"sidecar": 0})
    iv = int(h["inner_version"])
    on = 1 if iv >= K3_MIN_INNER_VERSION and h["k3"] > 0 else 0
    d = {"sidecar": 1, "inner_version": iv, "k3_on": on,
         "k3_int": int(round(h["k3"])) if on else 0,
         "k1_x100": int(h["k1"] * 100), "b_x100": int(h["b"] * 100),
         "gen_match": 1 if _store_generation(ctx) == int(h["generation"]) else 0}
    return cc.from_count(on, d, "sidecar header k3 > 0 (inner version 3 or later)")


def p_a223(ctx: cc.Ctx) -> Result:
    h = _sidecar(ctx)
    if h is None:
        return cc.silent(0, {"sidecar": 0})
    aw = int(h["anchor_weight"])
    return cc.fired(1, {"sidecar": 1, "anchor_weight": aw,
                        "w3": 1 if aw == 3 else 0},
                    "anchor_weight read from sidecar header")


def p_a224(ctx: cc.Ctx) -> Result:
    h = _sidecar(ctx)
    file_ok = 1 if h is not None else 0
    d: dict[str, int] = {o: 0 for o in SIDECAR_OUTCOMES}
    for row in _audit_rows(ctx):
        o = row.get("sidecar_outcome")
        if _ups(row) and o in d:
            d[o] += 1
    reuse = d["fresh"] + d["incremental"]
    return cc.from_count(reuse + file_ok, {**d, "sidecar_file": file_ok},
                         "count = valid sidecar file + fires that reused it")


def p_a230(ctx: cc.Ctx) -> Result:
    if not ctx.table_exists("belief_entities"):
        return cc.silent(0, {}, "no belief_entities table")
    d = ctx.group_count("SELECT kind, COUNT(*) FROM belief_entities GROUP BY kind",
                        ENTITY_KINDS)
    beliefs = ctx.count("SELECT COUNT(DISTINCT belief_id) FROM belief_entities")
    return cc.from_count(sum(d.values()), {**d, "beliefs_with_entities": beliefs})


def _ups_marker_rows(ctx: cc.Ctx) -> int:
    n = 0
    for row in _audit_rows(ctx):
        if _ups(row) and _is_marker(row.get("prompt_prefix")):
            n += 1
    return n


def p_a236(ctx: cc.Ctx) -> Result:
    """No production path writes struct.npy, so the HRR dim used is never recorded.

    The hook and CLI retrieve() calls build HRRStructIndex in memory at
    DEFAULT_DIM and never save it; make_hrr_struct_cache, the only
    persisting constructor, has no caller in the package, and
    HRRStructIndexCache.invalidate deletes struct.npy on any store write.
    """
    struct_npy = 1 if read_npy_shape(ctx.sidecar(HRR_DIRNAME, HRR_STRUCT)) is not None else 0
    edges = ctx.count("SELECT COUNT(*) FROM edges") if ctx.table_exists("edges") else 0
    return Result("untraceable", None,
                  {"struct_npy": struct_npy, "ups_marker_rows": _ups_marker_rows(ctx),
                   "edges": edges},
                  "dim is never persisted; production builds are in-memory only")


def p_a238(ctx: cc.Ctx) -> Result:
    s = 1 if read_npy_shape(ctx.sidecar(HRR_DIRNAME, HRR_STRUCT)) is not None else 0
    m = 1 if ctx.sidecar(HRR_DIRNAME, HRR_META).is_file() else 0
    return cc.from_count(s, {"struct_npy": s, "meta_npz": m})


def _is_marker(text: object) -> bool:
    m = _STRUCT_MARKER_RE.match(str(text or "").strip())
    return m is not None and m.group(1) in EDGE_TYPES and bool(m.group(2).strip())


def _a239_turns(ctx: cc.Ctx, d: dict[str, int]) -> None:
    for row in cc.iter_jsonl(ctx.sidecar("transcripts", "turns.jsonl")):
        if row.get("role") != "user":
            continue
        d["turn_user_rows"] += 1
        if _is_marker(row.get("text")):
            d["turn_marker_rows"] += 1


def p_a239(ctx: cc.Ctx) -> Result:
    """Structural lane: a marker prompt that returned beliefs with no gate reason.

    The structural early return in retrieval skips the expansion gate, so
    its audit row carries expansion_gate_skipped_bfs and no
    expansion_gate_reason. A store with no hook audit has no fire to read.
    """
    if not _audit_paths(ctx):
        return cc.untraceable("no hook_audit file")
    d = {"ups_rows": 0, "ups_marker_rows": 0, "marker_gate_skipped": 0,
         "signature_rows": 0, "turn_user_rows": 0, "turn_marker_rows": 0,
         "edge_rows": 0, "beliefs_since_ship": 0}
    if ctx.table_exists("edges"):
        d["edge_rows"] = ctx.count("SELECT COUNT(*) FROM edges")
    d["beliefs_since_ship"] = _count_if(
        ctx, "beliefs", "created_at",
        "SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (STRUCT_LANE_SHIP,))
    _a239_turns(ctx, d)
    for row in _audit_rows(ctx):
        if not _ups(row):
            continue
        d["ups_rows"] += 1
        if not _is_marker(row.get("prompt_prefix")):
            continue
        d["ups_marker_rows"] += 1
        if row.get("prompt_shape_gate_skip"):
            d["marker_gate_skipped"] += 1
        if (_int(row.get("n_beliefs")) > 0 and "expansion_gate_skipped_bfs" in row
                and "expansion_gate_reason" not in row):
            d["signature_rows"] += 1
    return cc.from_count(d["signature_rows"], d, "UPS structural signature rows")


def p_a240(ctx: cc.Ctx) -> Result:
    """HRR expand lane: silent without a probed-kind edge, else untraceable.

    retrieve_with_tiers counts expand_seeds hits only in LaneTelemetry and
    the audit copies no hrr_expand field. expand_seeds can return a
    neighbour only through a probed-kind edge, so a store with none of
    those edges cannot have fired.
    """
    if ctx.db is None:
        return cc.untraceable("no db")
    d: dict[str, int] = {"probed_edges": 0, "edges_typed": 0, "neighbors_table": 0,
                         "neighbors_rows": 0}
    for k in HRR_EXPAND_KINDS:
        d["e_" + k.lower()] = 0
    if ctx.column_exists("edges", "type"):
        d["edges_typed"] = 1
        ph = ",".join("?" * len(HRR_EXPAND_KINDS))
        kinds = [k.lower() for k in HRR_EXPAND_KINDS]
        g = ctx.group_count(
            f"SELECT lower(type), COUNT(*) FROM edges WHERE type IN ({ph}) GROUP BY type",
            kinds, HRR_EXPAND_KINDS)
        for k in kinds:
            d["e_" + k] = g.get(k, 0)
        d["probed_edges"] = sum(g.get(k, 0) for k in kinds)
    if ctx.table_exists("hrr_expand_neighbors"):
        d["neighbors_table"] = 1
        d["neighbors_rows"] = ctx.count("SELECT COUNT(*) FROM hrr_expand_neighbors")
    if d["probed_edges"] == 0:
        return cc.silent(0, d, "zero probed-kind edges; lane cannot return a candidate")
    return Result("untraceable", None, d,
                  "probed-kind edges present; fire leaves no durable trace")


def p_a241(ctx: cc.Ctx) -> Result:
    if (r := _no_audit(ctx)) is not None:
        return r
    d: dict[str, int] = {}
    matched: list[dict[str, Any]] = []
    for row in _audit_rows(ctx):
        if not _injecting(row) or "order_policy" not in row:
            continue
        pol = row.get("order_policy")
        label = pol if isinstance(pol, str) and pol in ORDER_POLICIES else "other"
        d[label] = d.get(label, 0) + 1
        if label != "lane":
            matched.append(row)
    return _audit_result(ctx, matched, d, "count = fires whose applied policy was not lane")


def p_a243(ctx: cc.Ctx) -> Result:
    if (r := _no_audit(ctx)) is not None:
        return r
    d = {"ups": 0, "session_start": 0, "locks_injected": 0}
    matched: list[dict[str, Any]] = []
    for row in _audit_rows(ctx):
        if not _injecting(row):
            continue
        n = _int(row.get("n_locked"))
        if n > 0:
            matched.append(row)
            d["ups" if _ups(row) else "session_start"] += 1
            d["locks_injected"] += n
    return _audit_result(ctx, matched, d)


def _a244_audit(ctx: cc.Ctx, d: dict[str, int]) -> None:
    for r in _audit_rows(ctx):
        if not _ups(r):
            continue
        d["ups"] += 1
        ts_raw = r.get("ts")
        ts = ts_raw if isinstance(ts_raw, str) else ""
        skip = bool(r.get("prompt_shape_gate_skip"))
        o = r.get("sidecar_outcome")
        if isinstance(o, str) and o in SIDECAR_OUTCOMES:
            d["warm_keyed" if skip else "lane_keyed"] += 1
            if not skip:
                d[o] += 1
        elif "expansion_gate_reason" in r and not skip:
            d["ran_unkeyed"] += 1
            if ts and ts < BM25F_DEFAULT_ON:
                d["ran_unkeyed_pre_1_7"] += 1
            if ts >= SIDECAR_FIELD_SHIP:
                d["ran_unkeyed_after_field_ship"] += 1


def p_a244(ctx: cc.Ctx) -> Result:
    """L1 BM25F lane: a keyed UPS row or a BM25F sidecar is a fire.

    hook_audit UPS rows carry sidecar_outcome when BM25IndexCache.get ran
    in the fire; a gate-skipped row with the key is the cadence warm, not
    the lane, so it is split out. The sidecar is reached only through
    retrieval.bm25f_cache_for_lane. A retrieving fire with no key that
    falls between the default-on flip and the field's ship date may have
    used the lane unrecorded: untraceable. A keyless retrieving fire after
    the field shipped did not use the lane.
    """
    d = {"ups": 0, "lane_keyed": 0, "warm_keyed": 0, "ran_unkeyed": 0,
         "ran_unkeyed_pre_1_7": 0, "ran_unkeyed_after_field_ship": 0,
         "sidecar": 0, "anchored_dst_active": 0, "beliefs": 0}
    d.update({o: 0 for o in SIDECAR_OUTCOMES})
    _a244_audit(ctx, d)
    sc = ctx.sidecar(BM25_SIDECAR_NAME)
    if sc.is_file():
        with sc.open("rb") as f:
            d["sidecar"] = 1 if f.read(8) == SIDECAR_MAGIC else 0
    if ctx.table_exists("beliefs"):
        d["beliefs"] = ctx.count("SELECT COUNT(*) FROM beliefs")
        if ctx.column_exists("edges", "anchor_text") and ctx.column_exists("beliefs", "valid_to"):
            d["anchored_dst_active"] = ctx.count(
                "SELECT COUNT(DISTINCT e.dst) FROM edges e JOIN beliefs b ON b.id=e.dst "
                "WHERE b.valid_to IS NULL AND e.anchor_text IS NOT NULL AND e.anchor_text<>''")
    if d["lane_keyed"] > 0 or d["sidecar"] == 1:
        return cc.fired(d["lane_keyed"], d, "lane_keyed rows or BM25F sidecar present")
    unrecorded = (d["ran_unkeyed"] - d["ran_unkeyed_pre_1_7"]
                  - d["ran_unkeyed_after_field_ship"])
    if unrecorded > 0:
        return Result("untraceable", None, d,
                      "retrieving fires predate the sidecar_outcome field and no sidecar")
    return cc.silent(0, d, "no retrieving fire that could have used the lane")


def _a245_pool(ctx: cc.Ctx, keys: list[str], ts: datetime) -> set[str]:
    """Non-locked beliefs sharing a query entity, created by the fire's second."""
    ph = ",".join("?" * len(keys))
    pool: set[str] = set()
    for bid, ca, lk in ctx.require_db().execute(
            "SELECT DISTINCT be.belief_id, b.created_at, b.lock_level "
            "FROM belief_entities be JOIN beliefs b ON b.id = be.belief_id "
            f"WHERE be.entity_lower IN ({ph})", keys):
        if lk == LOCK_USER:
            continue
        cts = _parse_ts(ca)
        # The audit ts has whole seconds; compare the belief at that precision.
        if cts is not None and cts.replace(microsecond=0) <= ts:
            pool.add(str(bid))
    return pool


def _a245_query(prefix: str) -> str:
    """The audited prompt prefix, with a cut-off last token dropped."""
    if len(prefix) < PREFIX_CAP:
        return prefix
    return prefix.rsplit(None, 1)[0] if len(prefix.split()) > 1 else ""


def _a245_row(ctx: cc.Ctx, row: dict[str, Any], d: dict[str, int]) -> None:
    from aelfrice.entity_extractor import extract_entities  # noqa: PLC0415
    ids = set(_l1_ids(row))
    if not isinstance(row.get("beliefs"), list):
        return
    d["rows_with_ids"] += 1
    d["injected_l1"] += len(ids)
    prefix = row.get("prompt_prefix")
    ts = _parse_ts(row.get("ts"))
    if not isinstance(prefix, str) or not prefix.strip() or ts is None:
        return
    if len(prefix) < PREFIX_CAP:
        d["rows_complete_prompt"] += 1
    q = _a245_query(prefix)
    ents = extract_entities(q, max_entities=QUERY_ENTITY_CAP) if q else []
    keys = list(dict.fromkeys(e.lower for e in ents if e.lower))[:512]
    if not keys:
        return
    d["rows_with_entities"] += 1
    pool = _a245_pool(ctx, keys, ts)
    if pool:
        d["rows_pool_nonempty"] += 1
    m = len(ids & pool)
    if m:
        d["rows_injected_match"] += 1
        d["injected_l1_matched"] += m


def p_a245(ctx: cc.Ctx) -> Result:
    """L2.5 entity lane, rebuilt per audited UPS fire.

    The audit labels every non-locked hit L1 and LaneTelemetry.l25 stays
    in process, so the lane is reconstructed: entities of the audited
    prompt (entity_extractor.extract_entities, as retrieval._l25_hits
    does), the non-locked beliefs sharing one created by the fire's time
    (store.lookup_entities), and the injected L1 ids in that pool. An
    injected pool member is a belief the lane could have supplied: fired.
    Audited fires with no such member: silent. No audited fire: untraceable.
    """
    d = {k: 0 for k in (
        "ups_rows", "rows_with_ids", "rows_complete_prompt", "rows_with_entities",
        "rows_pool_nonempty", "rows_injected_match", "injected_l1",
        "injected_l1_matched", "belief_entities_rows")}
    if ctx.table_exists("belief_entities") and ctx.column_exists("beliefs", "lock_level"):
        d["belief_entities_rows"] = ctx.count("SELECT COUNT(*) FROM belief_entities")
        for row in _audit_rows(ctx):
            if _ups(row):
                d["ups_rows"] += 1
                _a245_row(ctx, row, d)
    if d["rows_injected_match"] > 0:
        return cc.fired(d["rows_injected_match"], d, "upper bound: injected L1 ids in the entity pool")
    if d["ups_rows"] > 0:
        return cc.silent(0, d)
    return Result("untraceable", None, d, "no audited UPS fire to rebuild the lane from")


def p_a249(ctx: cc.Ctx) -> Result:
    if (r := _no_audit(ctx)) is not None:
        return r
    matched = [r for r in _audit_rows(ctx)
               if _ups(r) and isinstance(r.get("expansion_gate_reason"), str)]
    return _audit_result(ctx, matched, {},
                         "only the expansion gate fields of LaneTelemetry reach the audit")


def p_a250(ctx: cc.Ctx) -> Result:
    h = _sidecar(ctx, full=True)
    if h is None:
        return cc.silent(0, {"sidecar": 0})
    on = 1 if h.get("per_field") == 1 else 0
    return cc.from_count(on, {"sidecar": 1, "per_field": on},
                         "BM25F sidecar header per_field flag")


def p_a251(ctx: cc.Ctx) -> Result:
    p = ctx.global_file(*HOOK_FAILURES_LOG)
    if not p.is_file():
        return cc.silent(0, {"log": 0}, "no hook-failures log")
    n = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if PLACEHOLDER_WARNING in line:
                n += 1
    return cc.from_count(n, {"log": 1},
                         "stderr only; visible only when a host wrapper redirects hook stderr")


def p_a252(ctx: cc.Ctx) -> Result:
    h = _sidecar(ctx)
    if h is None:
        return cc.silent(0, {"sidecar": 0})
    v6 = 1 if int(h["inner_version"]) == SERIALIZE_VERSION else 0
    return cc.from_count(v6, {"sidecar": 1, "tokenizer_v6": v6},
                         "same sidecar evidence as A224; v6 marks the stemmed tokenisation")


def p_a254(ctx: cc.Ctx) -> Result:
    """POTENTIALLY_STALE penalty on BFS hops: silent with no marker edge.

    edge_rerank.apply_edge_type_rerank short-circuits to identity when the
    store holds no POTENTIALLY_STALE edge, and the only writer is the
    opt-in `aelf doctor --detect-stale`. With marker edges present the hop
    score is never recorded: untraceable.
    """
    if not ctx.table_exists("edges"):
        return cc.silent(0, {"stale_edges": 0}, "no edges table")
    if not ctx.column_exists("edges", "type"):
        n = _count_if(ctx, "edges", "edge_type",
                      "SELECT COUNT(*) FROM edges WHERE edge_type = ?", (POTENTIALLY_STALE,))
        return cc.silent(0, {"stale_edges": n, "legacy_schema": 1},
                         "pre-aelfrice edges schema; BFS cannot read it")
    stale = ctx.count("SELECT COUNT(*) FROM edges WHERE type = ?", (POTENTIALLY_STALE,))
    live = 0
    if ctx.table_exists("beliefs"):
        vt = "AND b.valid_to IS NULL" if ctx.column_exists("beliefs", "valid_to") else ""
        live = ctx.count(
            "SELECT COUNT(DISTINCT e.dst) FROM edges e JOIN beliefs b ON b.id = e.dst "
            f"WHERE e.type = ? {vt}", (POTENTIALLY_STALE,))
    after = _count_if(ctx, "beliefs", "created_at",
                      "SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (STALE_RERANK_SHIP,))
    skip = sum(1 for row in _audit_rows(ctx)
               if _ups(row) and row.get("expansion_gate_skipped_bfs") is True)
    detail = {"stale_edges": stale, "stale_targets_live": live,
              "beliefs_after_ship": after, "ups_bfs_gate_skip": skip}
    if stale == 0:
        return cc.silent(0, detail, "no POTENTIALLY_STALE rows; rerank is identity")
    return Result("untraceable", None, detail, "marker edges present; hop score not persisted")


def p_a255(ctx: cc.Ctx) -> Result:
    if (r := _no_audit(ctx)) is not None:
        return r
    lines = 0
    matched: list[dict[str, Any]] = []
    for row in _audit_rows(ctx):
        if not _injecting(row):
            continue
        n = len(_REF_LINE_RE.findall(str(row.get("rendered_block") or "")))
        if n:
            lines += n
            matched.append(row)
    ref_locks = 0
    if ctx.column_exists("beliefs", "lock_tier"):
        ref_locks = ctx.count("SELECT COUNT(*) FROM beliefs WHERE lock_level='user' "
                              "AND lock_tier='reference'")
    return _audit_result(ctx, matched, {"ref_lines": lines, "reference_locks": ref_locks})


def _tn_neighbours(ctx: cc.Ctx) -> dict[str, set[str]]:
    nb: dict[str, set[str]] = {}
    for s, t in ctx.require_db().execute(
            "SELECT src, dst FROM edges WHERE type = ?", (TEMPORAL_NEXT,)):
        nb.setdefault(str(s), set()).add(str(t))
        nb.setdefault(str(t), set()).add(str(s))
    return nb


def _a262_row(row: dict[str, Any], nb: dict[str, set[str]], d: dict[str, int]) -> None:
    ts = row.get("ts")
    ids = _l1_ids(row)
    if not isinstance(ts, str) or len(ids) < 2:
        return
    era = "post" if ts[:10] >= SPINE_LANE_SHIP else "pre"
    d["ups_" + era] += 1
    if era == "post" and row.get("order_policy", "lane") == "lane":
        d["lane_policy_post"] += 1
    seen: set[str] = set()
    any_adj = False
    for bid in ids:
        if seen & nb.get(bid, set()):
            any_adj = True
        seen.add(bid)
    if any_adj:
        d["adj_" + era] += 1
    if set(ids[:-1]) & nb.get(ids[-1], set()):
        d["tail_" + era] += 1


def p_a262(ctx: cc.Ctx) -> Result:
    """Temporal-spine lane, inferred from TEMPORAL_NEXT adjacency in the audit.

    retrieve_v2 appends spine hits after the L1 candidates and counts them
    only in LaneTelemetry.temporal_spine. A spine hit is a 1-hop
    TEMPORAL_NEXT neighbour of a packed L1 seed, so a UPS row holding an
    entry adjacent to an earlier entry is a candidate fire (upper bound:
    BM25 can also return two adjacent sentences). Rows before the lane
    reached the live path are the control.
    """
    if not ctx.column_exists("edges", "type"):
        return cc.untraceable("no typed edges table")
    d = {k: 0 for k in ("tn_edges", "beliefs_post_ship", "ups_pre", "ups_post",
                        "adj_pre", "adj_post", "tail_pre", "tail_post", "lane_policy_post")}
    d["tn_edges"] = ctx.count("SELECT COUNT(*) FROM edges WHERE type = ?", (TEMPORAL_NEXT,))
    d["beliefs_post_ship"] = _count_if(
        ctx, "beliefs", "created_at",
        "SELECT COUNT(*) FROM beliefs WHERE created_at >= ?", (SPINE_LANE_SHIP,))
    nb = _tn_neighbours(ctx) if d["tn_edges"] else {}
    for row in _audit_rows(ctx):
        if _ups(row):
            _a262_row(row, nb, d)
    if d["adj_post"] > 0:
        return cc.fired(d["adj_post"], d, "upper bound: TEMPORAL_NEXT neighbours in one block")
    if d["tn_edges"] > 0 and d["ups_post"] > 0:
        return cc.silent(0, d)
    return Result("untraceable", None, d, "no spine edges or no audited fire to infer from")


def _compressible_ids(ctx: cc.Ctx, d: dict[str, int]) -> set[str]:
    """Active non-locked snapshot or transient beliefs the compressor would shorten."""
    from aelfrice.compression import STRATEGY_VERBATIM, compress_for_retrieval  # noqa: PLC0415
    from aelfrice.models import Belief  # noqa: PLC0415
    if not (ctx.column_exists("beliefs", "retention_class")
            and ctx.column_exists("beliefs", "lock_level")):
        return set()
    where = "lock_level <> 'user' AND retention_class IN ('snapshot','transient')"
    if ctx.column_exists("beliefs", "valid_to"):
        where += " AND valid_to IS NULL"
    comp: set[str] = set()
    for bid, content, rc in ctx.require_db().execute(
            f"SELECT id, content, retention_class FROM beliefs WHERE {where}"):
        d["elig_" + str(rc)] += 1
        b = Belief(id=str(bid), content=str(content or ""), content_hash="", alpha=1.0,
                   beta=1.0, type="factual", lock_level="none", locked_at=None,
                   created_at="", last_retrieved_at=None, retention_class=str(rc))
        if compress_for_retrieval(b, locked=False).strategy != STRATEGY_VERBATIM:
            comp.add(str(bid))
    return comp


def _is_rebuild_v14_row(cands: list[Any]) -> bool:
    for x in cands:
        if isinstance(x, dict):
            scores: Any = x.get("scores")  # pyright: ignore[reportUnknownMemberType]
            if isinstance(scores, dict) and scores.get("final") is not None:  # pyright: ignore[reportUnknownMemberType]
                return True
    return False


def _a263_logs(ctx: cc.Ctx, comp: set[str], d: dict[str, int]) -> None:
    rdir = ctx.sidecar("rebuild_logs")
    if not rdir.is_dir():
        return
    for p in sorted(rdir.glob("*.jsonl")):
        for row in cc.iter_jsonl(p):
            cands: Any = row.get("candidates")
            if not isinstance(cands, list):
                continue
            d["rb_rows"] += 1
            if not _is_rebuild_v14_row(cands):  # pyright: ignore[reportUnknownArgumentType]
                continue
            d["rb14_rows"] += 1
            hit = False
            for x in cands:  # pyright: ignore[reportUnknownVariableType]
                if not isinstance(x, dict) or str(x.get("belief_id")) not in comp:  # pyright: ignore[reportUnknownMemberType]
                    continue
                d["rb14_cand_compressible"] += 1
                if x.get("decision") == "packed":  # pyright: ignore[reportUnknownMemberType]
                    d["rb14_packed_compressible"] += 1
                    hit = True
            d["rb14_rows_fired"] += int(hit)


def p_a263(ctx: cc.Ctx) -> Result:
    """Type-aware compression, seen through rebuild_v14 pack decisions.

    compress_for_retrieval shortens only non-locked snapshot (headline) and
    transient (stub) beliefs, and its only effect is pack cost. UPS and the
    search-tool lane bypass it. Of the surfaces that use it only
    rebuild_v14 leaves a per-belief record: rebuild_logs rows whose
    candidates carry a non-null scores.final. A rebuild_v14 row that packed
    a compressible belief charged it compressed cost: fired. A store with
    no compressible belief: silent. Otherwise untraceable.
    """
    d = {k: 0 for k in ("elig_snapshot", "elig_transient", "compressible",
                        "rb_rows", "rb14_rows", "rb14_cand_compressible",
                        "rb14_packed_compressible", "rb14_rows_fired")}
    comp = _compressible_ids(ctx, d)
    d["compressible"] = len(comp)
    _a263_logs(ctx, comp, d)
    if d["rb14_rows_fired"] > 0:
        return cc.fired(d["rb14_rows_fired"], d, "rebuild_v14 rows that packed a compressible belief")
    if d["compressible"] == 0:
        return cc.silent(0, d, "no compressible belief; cost is unchanged")
    return Result("untraceable", None, d, "compressible beliefs present; no rebuild_v14 pack record")


def p_a266(ctx: cc.Ctx) -> Result:
    if (r := _no_audit(ctx)) is not None:
        return r
    matched = [r for r in _audit_rows(ctx)
               if _ups(r) and _COVERAGE_RE.search(str(r.get("rendered_block") or ""))]
    return _audit_result(ctx, matched, {},
                         "coverage line = L1 candidates cut by the token budget")


# --- registry --------------------------------------------------------------

_U = cc.untraceable_probe
_S = cc.safe
_LANE = "in-process LaneTelemetry only; audit labels every non-locked hit L1"

PROBES: list[Probe] = [
    Probe("A217", "#N issue/PR literal boost", "sidecar", _S(p_a217)),
    _U("A218", "fan-effect ordering of L2.5", "reorders L2.5 in process; " + _LANE),
    Probe("A219", "adaptive expansion gate", "sidecar", _S(p_a219)),
    Probe("A220", "meta-belief consumers", "store", _S(p_a220)),
    _U("A221", "BFS multi-hop L3", "BFS hits leave no lane mark; " + _LANE),
    Probe("A222", "BM25 k3 saturation", "sidecar", _S(p_a222)),
    Probe("A223", "BM25F anchor weight", "sidecar", _S(p_a223)),
    Probe("A224", "BM25F persistent sidecar", "sidecar", _S(p_a224)),
    _U("A225", "max-coverage L1 pack", "pack choice not persisted; " + _LANE),
    _U("A226", "compression headline strategy",
       "compressed render feeds pack cost only; no output or log carries it"),
    _U("A227", "compression stub strategy",
       "stub render feeds pack cost only; no output or log carries it"),
    _U("A228", "env budget reach into PreToolUse lanes",
       "resolved budget and l1_limit are not recorded; search telemetry mixes L1 and L2.5"),
    _U("A229", "default budget split",
       "budget constants are not recorded per fire"),
    Probe("A230", "entity extraction writer", "store", _S(p_a230)),
    _U("A231", "entity-persist demotion", "score penalty only; " + _LANE),
    _U("A232", "FTS5 rarest-term MATCH", "query rewrite inside search_beliefs; nothing persisted"),
    _U("A233", "gamma posterior rerank", "score change only; " + _LANE),
    _U("A234", "heat-kernel rerank", "no production GraphEigenbasisCache constructor; no trace"),
    _U("A235", "touch-state rerank consumer",
       "consumer not built; belief_touches rows come from the writer"),
    Probe("A236", "HRR dim 512", "sidecar", _S(p_a236)),
    _U("A237", "HRR expand sim floor", "expand lane writes nothing; " + _LANE),
    Probe("A238", "HRR struct-index persistence", "sidecar", _S(p_a238)),
    Probe("A239", "HRR structural-query lane", "sidecar", _S(p_a239)),
    Probe("A240", "HRR vocabulary-bridge expansion", "store", _S(p_a240)),
    Probe("A241", "injection order policy", "sidecar", _S(p_a241)),
    _U("A242", "intentional clustering pack", "pack choice not persisted; " + _LANE),
    Probe("A243", "L0 locks injected", "sidecar", _S(p_a243)),
    Probe("A244", "L1 BM25F lane", "sidecar", _S(p_a244)),
    Probe("A245", "L2.5 entity-index lane", "sidecar", _S(p_a245)),
    _U("A246", "legacy 2000 budget downgrade", "resolved budget is not recorded"),
    _U("A247", "origin-priority tie-break", "tie order only; " + _LANE),
    _U("A248", "origin-tier rerank lane", "lane not shipped; no code path writes"),
    Probe("A249", "per-call lane telemetry", "sidecar", _S(p_a249)),
    Probe("A250", "per-field BM25F", "sidecar", _S(p_a250)),
    Probe("A251", "placeholder flag warning", "global-log", _S(p_a251)),
    Probe("A252", "Porter stemming tokenizer", "sidecar", _S(p_a252)),
    _U("A253", "posterior-weighted L1 rerank",
       "score change only; audit posterior_mean is not rerank evidence"),
    Probe("A254", "POTENTIALLY_STALE BFS penalty", "store", _S(p_a254)),
    Probe("A255", "reference-tier lock manifest", "sidecar", _S(p_a255)),
    _U("A256", "relevance floor under lock saturation", "budget split not recorded per fire"),
    _U("A257", "scoring.relevance dead scorer", "no call site"),
    _U("A258", "signed-Laplacian builder",
       "library only; no production caller and no persisted output"),
    _U("A259", "aelf spine verify", "read-only diagnostic; output goes to stdout only"),
    _U("A260", "supersession demote lane", "score change or drop only; " + _LANE),
    _U("A261", "temporal decay rerank", "no production caller passes temporal_sort"),
    Probe("A262", "temporal-spine lane", "sidecar", _S(p_a262)),
    Probe("A263", "type-aware compression", "sidecar", _S(p_a263)),
    _U("A264", "uri_baki adjusters", "code removed"),
    _U("A265", "utterance prior in L1", "score change only; " + _LANE),
    Probe("A266", "L1 limit and token budget trim", "sidecar", _S(p_a266)),
    _U("A267", "zeta posterior rerank", "score change only; " + _LANE),
]

TRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind != "untraceable")
UNTRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind == "untraceable")
