"""Census probes for the ingest-extraction subsystem (A132-A183).

Each probe reads one read-only `Ctx` and returns integers and fixed labels
only. Text read from a row (belief content, raw_text, turn text) is used
inside the probe for a predicate or an id check and never leaves it. A few
probes call pure aelfrice functions (sentence splitting, noise rules,
turn hashing) on that text; none of them opens a store.

Code facts the probes rely on (src/aelfrice on `main` at ad14ef545):
- ingest_log rows: ingest._ingest_turn (source_kind transcript, raw_meta
  call_site/role/turn_sha/holed/derived_from), scanner.scan_repo and
  classification.accept_classifications (filesystem, source_path doc:/ast:/
  git:commit:, raw_meta route_overrides or override_belief_type),
  triple_extractor.ingest_triples (git, call_site commit_ingest).
- derivation_worker stamps derived_belief_ids ('[]' when derive() returns no
  belief, which only the classifier path does) and derived_edge_ids.
- claude_memory_reconcile.ingest_memory_text writes beliefs directly with
  id sha256('claude_memory' NUL body)[:16] and no ingest_log row.
- transcript_logger writes transcripts/turns.jsonl, archive/turns-*.jsonl and
  .stop_flush_cursor; skipped duplicates go to hook_audit.jsonl beside the db.
- hook_commit_ingest._record_triples writes a sessions row with
  model='commit-ingest' for both the git post-commit hook (5.1.0) and the
  older PostToolUse path, so the two are told apart by date only.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import core as cc

P = cc.Probe
U = cc.untraceable_probe
Ctx = cc.Ctx
Result = cc.Result

EDGE_TYPES = frozenset({
    "SUPPORTS", "CITES", "CONTRADICTS", "SUPERSEDES", "RELATES_TO",
    "DERIVED_FROM", "IMPLEMENTS", "TEMPORAL_NEXT", "TESTS", "RESOLVES",
    "POTENTIALLY_STALE",
})
BELIEF_TYPES = frozenset({"factual", "correction", "preference", "requirement",
                          "speculative"})
CLASSIFIER_KINDS = ("transcript", "filesystem", "python_ast",
                    "feedback_loop_synthesis")
# The git post-commit hook (#1698) merged before this date and shipped in
# 5.1.0 on 2026-10-06. Before the cutoff only the PostToolUse path existed.
GIT_HOOK_CUTOFF = "2026-10-03"
GIT_HOOK_SHIP = "2026-10-06"
# #1874 (inline code and URLs kept in user sentences) merged on this date.
SPAN_KEEP_MERGE = "2026-11-07"
# The speaker-attribution gate shipped in 3.1.0; raw_meta.role from 3.3.0.
SPEAKER_GATE_SHIP = "2026-05-14"
ROLE_STAMP_SHIP = "2026-05-21"
# The temporal-spine writer is on by default since 4.0.0.
SPINE_DEFAULT_ON = "2026-07-07"


# --- helpers ---------------------------------------------------------------


def _missing(c: Ctx, *tables: str) -> Result | None:
    """silent when the db or a table is absent: the feature never wrote here."""
    if c.db is None:
        return cc.silent(0, note="no db")
    for t in tables:
        if not c.table_exists(t):
            return cc.silent(0, note="table absent: " + t)
    return None


def _rows(c: Ctx, sql: str, params: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    yield from c.require_db().execute(sql, params)


def _ids_from_log(c: Ctx, where: str) -> set[str]:
    """Belief ids stamped on ingest_log rows matching `where` (internal only)."""
    sql = ("SELECT DISTINCT j.value FROM ingest_log l, json_each(l.derived_belief_ids) j "
           "WHERE l.derived_belief_ids IS NOT NULL AND json_valid(l.derived_belief_ids) "
           "AND " + where)
    return {r[0] for r in _rows(c, sql) if isinstance(r[0], str)}


def _meta(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        v: Any = json.loads(raw)
    except ValueError:
        return {}
    return dict(v) if isinstance(v, dict) else {}  # pyright: ignore[reportUnknownArgumentType]


def _transcript_files(c: Ctx) -> list[Path]:
    tdir = c.sidecar("transcripts")
    out: list[Path] = []
    live = tdir / "turns.jsonl"
    if live.is_file():
        out.append(live)
    adir = tdir / "archive"
    if adir.is_dir():
        out.extend(sorted(p for p in adir.glob("turns-*.jsonl") if p.is_file()))
    return out


def _archive_files(c: Ctx) -> list[Path]:
    adir = c.sidecar("transcripts", "archive")
    if not adir.is_dir():
        return []
    return sorted(p for p in adir.glob("turns-*.jsonl") if p.is_file())


def _global_flag(c: Ctx, name: str) -> int:
    """1 when the global dir holds `name`; 0 when it does not or there is none."""
    if c.global_dir is None:
        return 0
    return 1 if c.global_file(name).is_file() else 0


# --- classifier tables copied from classification_core.py -------------------

_QUESTION_PREFIXES = ("what ", "how ", "why ", "when ", "where ", "can ", "does ",
                      "is there", "should ", "would ", "could ")
_FLOAT_LEADING_HEDGES = (
    "maybe ", "maybe,", "perhaps ", "perhaps,", "what if ", "how about ",
    "what about ", "i wonder ", "i wonder if", "wondering if ", "not sure ",
    "i guess ", "i suppose ", "or maybe ", "or perhaps ", "or should ",
    "or we could ", "or we should ", "should we ", "could we ", "shouldn't we ",
    "shouldnt we ",
)
_FLOAT_INTERNAL_HEDGES = (
    "maybe we ", "maybe there ", "maybe it would ", "we probably need ",
    "we could probably ", "we might want ", "we should probably ",
    "not sure if ", "not sure whether ",
)
_SENTENCE_BOUNDARY_RE = re.compile(r'''[.!?]['")\]]*\s''')


def _is_question(tl: str) -> bool:
    s = tl.strip()
    if not s.endswith("?"):
        return False
    if s.startswith(_QUESTION_PREFIXES):
        return True
    if "\n" in s:
        return False
    return _SENTENCE_BOUNDARY_RE.search(s[:-1]) is None


def _is_float(tl: str) -> bool:
    s = tl.strip()
    if not s:
        return False
    if s.startswith(_FLOAT_LEADING_HEDGES):
        return True
    return any(p in s for p in _FLOAT_INTERNAL_HEDGES)


def _dropped_rows(c: Ctx) -> dict[str, dict[str, int]]:
    """Classify ingest_log rows stamped '[]' (derive() returned no belief)."""
    out: dict[str, dict[str, int]] = {"question": {}, "float": {}, "empty": {}, "other": {}}
    sql = ("SELECT source_kind, source_path, raw_text, raw_meta FROM ingest_log "
           "WHERE derived_belief_ids = '[]'")
    for kind, spath, raw, meta_s in _rows(c, sql):
        meta = _meta(meta_s)
        text = raw if isinstance(raw, str) else ""
        if kind == "transcript" and spath == "transcript" and meta.get("role") == "user":
            holed = meta.get("holed")
            if isinstance(holed, str):
                text = holed
        tl = text.lower().strip()
        if not tl:
            bucket = "empty"
        elif _is_question(tl):
            bucket = "question"
        elif _is_float(tl):
            bucket = "float"
        else:
            bucket = "other"
        label = kind if kind in ("transcript", "filesystem") else "other_kind"
        out[bucket][label] = out[bucket].get(label, 0) + 1
    return out


# --- probes ----------------------------------------------------------------

_PRUNE_NEEDS = ("origin", "lock_level", "valid_to")


def _noise_now() -> Any:
    """doctor --prune-noise's own predicate, from aelfrice's pure functions."""
    from aelfrice.classification_core import classify_sentence
    from aelfrice.derivation import decision_text
    from aelfrice.models import ORIGIN_AGENT_INFERRED
    from aelfrice.noise_filter import is_stranded_capture_noise, is_transcript_noise

    def noise(bid: str, content: str) -> bool:
        text = decision_text(bid, content)
        if is_transcript_noise(text) or is_stranded_capture_noise(text):
            return True
        return not classify_sentence(text, ORIGIN_AGENT_INFERRED).persist
    return noise


def a133(c: Ctx) -> Result:
    """doctor --prune-noise soft-deletes active agent_inferred unlocked beliefs
    that the current noise filters reject, and writes no audit row.

    The count is retired agent_inferred unlocked beliefs with no retire or
    review audit row whose text the same filters still reject: the command's
    own selection, applied after the fact. A beliefs table without the
    columns the command reads is a pre-feature schema: silent.
    """
    if (r := _missing(c, "beliefs")) is not None:
        return r
    n_rows = c.count("SELECT COUNT(*) FROM beliefs")
    missing = [col for col in _PRUNE_NEEDS if not c.column_exists("beliefs", col)]
    if missing:
        return cc.silent(0, {"beliefs_rows": n_rows, "missing_cols": len(missing)},
                         "pre-feature schema: beliefs lacks a column prune-noise needs")
    sql = ("SELECT b.id, b.content FROM beliefs b WHERE b.origin='agent_inferred' "
           "AND b.lock_level='none' AND b.valid_to IS NOT NULL")
    if c.column_exists("feedback_history", "source"):
        sql += (" AND NOT EXISTS (SELECT 1 FROM feedback_history f WHERE f.belief_id=b.id "
                "AND (f.source LIKE 'user_retired%' OR f.source='review:remove'))")
    noise = _noise_now()
    retired = matched = 0
    for bid, content in _rows(c, sql):
        retired += 1
        if isinstance(bid, str) and isinstance(content, str) and noise(bid, content):
            matched += 1
    return cc.from_shared_count(matched, {"retired_unaudited": retired,
                                   "retired_noise_now": matched, "beliefs_rows": n_rows},
                         "ambiguous: another unaudited retire of text the filters reject "
                         "looks the same")


def a134(c: Ctx) -> Result:
    """aelf ingest-transcript: the transcript logger's PreCompact rotation and
    Stop flush spawn this command, so their rows are its rows. doctor
    --backfill-ingest calls ingest_jsonl with the same default label, and no
    column tells its rows apart, so a row is ambiguous."""
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    n, n_default, n_sess = c.counts(
        "SELECT COUNT(*), SUM(source_path='transcript'), COUNT(DISTINCT session_id) "
        "FROM ingest_log WHERE source_kind='transcript'")
    return cc.from_shared_count(n, {"transcript_rows": n, "default_label": n_default,
                                    "sessions": n_sess},
                                "doctor --backfill-ingest writes the same rows")


def a138(c: Ctx) -> Result:
    files = _transcript_files(c)
    full = cc.jsonl_count(files, lambda r: r.get("role") == "assistant"
                          and isinstance(r.get("text"), str) and r["text"].strip() != "")
    stub = cc.jsonl_count(files, lambda r: r.get("role") == "assistant"
                          and not (isinstance(r.get("text"), str) and r["text"].strip()))
    return cc.from_count(full, {"empty_stub": stub})


def a139(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    return cc.from_count(c.count(
        "SELECT COUNT(*) FROM ingest_log WHERE source_kind='filesystem' "
        "AND source_path LIKE 'ast:%'"))


def _ts(v: object) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if not isinstance(v, str):
        return None
    try:
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def _sentinel_text(c: Ctx) -> str | None:
    p = c.sidecar("claude-memory-reconciled")
    if not p.is_file():
        return None
    try:
        return p.read_text("utf-8", "replace")[:200]
    except OSError:
        return ""


def _sentinel_facts(c: Ctx) -> int | None:
    text = _sentinel_text(c)
    if text is None:
        return None
    m = re.search(r"reconciled (\d+) fact", text)
    return int(m.group(1)) if m else 0


def _sentinel_ts(c: Ctx) -> float | None:
    text = _sentinel_text(c)
    m = re.search(r" at (\d+)", text) if text else None
    return float(m.group(1)) if m else None


def a140(c: Ctx) -> Result:
    """Mirror-scheme belief ids, split by the reconcile sentinel's timestamp.

    origin and created_at are read only when the columns exist, so a
    pre-aelfrice schema reports 0 instead of raising.
    """
    if (r := _missing(c, "beliefs")) is not None:
        return r
    has_origin = c.column_exists("beliefs", "origin")
    has_created = c.column_exists("beliefs", "created_at")
    sel = "SELECT id, content, {}, {} FROM beliefs".format(
        "origin" if has_origin else "NULL", "created_at" if has_created else "NULL")
    sent_ts = _sentinel_ts(c)
    d = {"user_validated": 0, "agent_inferred": 0, "other_origin": 0,
         "after_sentinel": 0, "at_or_before_sentinel": 0, "untimed": 0,
         "beliefs_total": 0, "origin_col": int(has_origin),
         "sentinel": 0 if _sentinel_text(c) is None else 1}
    for bid, content, origin, created in _rows(c, sel):
        d["beliefs_total"] += 1
        if not isinstance(content, str) or not isinstance(bid, str):
            continue
        h = hashlib.sha256(("claude_memory\x00" + content).encode("utf-8")).hexdigest()
        if h[:16] != bid:
            continue
        key = origin if origin in ("user_validated", "agent_inferred") else "other_origin"
        d[key] += 1
        t = _ts(created)
        if t is None or sent_ts is None:
            d["untimed"] += 1
        elif t > sent_ts + 1.0:
            d["after_sentinel"] += 1
        else:
            d["at_or_before_sentinel"] += 1
    n = d["user_validated"] + d["agent_inferred"] + d["other_origin"]
    if c.column_exists("belief_corroborations", "source_type"):
        d["corroborations"] = c.count("SELECT COUNT(*) FROM belief_corroborations "
                                      "WHERE source_type='claude_memory_mirror'")
    return cc.from_count(n, d, "mirror-scheme ids; after_sentinel = hook writes after reconcile")


_COMMIT_CALL_SITE = "json_extract(raw_meta,'$.call_site')='commit_ingest'"


def _a141_log_detail(c: Ctx, d: dict[str, int]) -> int:
    """Commit ingest_log rows by era; returns ingest_log rows after ship."""
    if not c.table_exists("ingest_log"):
        return 0
    g = "SELECT COUNT(*) FROM ingest_log WHERE source_kind='git' AND json_valid(raw_meta)"
    d["log_commit_all"] = c.count(g + " AND " + _COMMIT_CALL_SITE)
    d["log_commit_hook_era"] = c.count(g + " AND " + _COMMIT_CALL_SITE + " AND ts >= ?",
                                       (GIT_HOOK_CUTOFF,))
    d["log_git_other"] = c.count(
        "SELECT COUNT(*) FROM ingest_log WHERE source_kind='git' AND NOT ("
        "json_valid(raw_meta) AND coalesce(" + _COMMIT_CALL_SITE + ",0))")
    active = c.count("SELECT COUNT(*) FROM ingest_log WHERE ts >= ?", (GIT_HOOK_SHIP,))
    d["log_rows_after_ship"] = active
    return active


def a141(c: Ctx) -> Result:
    """Hook-era commit-ingest sessions (on or after the cutoff).

    The legacy PostToolUse path writes the same sessions row, so sessions
    before the cutoff belong to A158. A silent store is split into one with
    no writes after 5.1.0 and an active one with no commit trace.
    """
    if (r := _missing(c, "sessions")) is not None:
        return r
    q = "SELECT COUNT(*) FROM sessions WHERE model='commit-ingest'"
    d: dict[str, int] = {
        "sessions_all": c.count(q),
        "sessions_pre_hook": c.count(q + " AND started_at < ?", (GIT_HOOK_CUTOFF,)),
        "sessions_after_ship": c.count(q + " AND started_at >= ?", (GIT_HOOK_SHIP,)),
    }
    hook = c.count(q + " AND started_at >= ?", (GIT_HOOK_CUTOFF,))
    active = _a141_log_detail(c, d)
    if c.table_exists("belief_corroborations"):
        b = "SELECT COUNT(*) FROM belief_corroborations WHERE source_type='commit_ingest'"
        d["corr_all"] = c.count(b)
        d["corr_hook_era"] = c.count(b + " AND ingested_at >= ?", (GIT_HOOK_CUTOFF,))
    d["other_sessions_after_ship"] = c.count(
        "SELECT COUNT(*) FROM sessions WHERE coalesce(model,'')<>'commit-ingest' "
        "AND started_at >= ?", (GIT_HOOK_SHIP,))
    active += d["other_sessions_after_ship"]
    if hook > 0:
        return cc.fired(hook, d, "hook-era sessions; legacy PostToolUse may linger "
                        "where setup was not rerun")
    if active == 0:
        return cc.silent(0, d, "store not written after 5.1.0 shipped the hook")
    return cc.silent(0, d, "active after ship, no hook-era commit row; "
                     "no-triple commits leave no trace")


def a143(c: Ctx) -> Result:
    """Per-folder commit routing (#1796) records no rule that picked the store.

    In a repo with no [store] marker every changed folder resolves to the
    repo store, the same store older code wrote, so a commit-ingest
    sessions row proves ingest ran, not that routing chose a marker store.
    Always untraceable; the detail carries the commit session counts.
    """
    d: dict[str, int] = {"named_store": int(c.layout == "named")}
    sd = c.sidecar("stores")
    d["named_store_dirs"] = sum(1 for p in sd.iterdir() if p.is_dir()) if sd.is_dir() else 0
    if c.table_exists("sessions"):
        d["commit_sessions"] = c.count(
            "SELECT COUNT(*) FROM sessions WHERE model='commit-ingest'")
    return cc.Result("untraceable", None, d, "routing leaves no rule trace")


def a144(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log", "edges")) is not None:
        return r
    n = c.count("SELECT COUNT(*) FROM ingest_log WHERE source_kind='git'")
    git_ids = _ids_from_log(c, "l.source_kind='git'")
    d: dict[str, int] = {"beliefs": len(git_ids)}
    for src, etype in _rows(c, "SELECT src, type FROM edges WHERE anchor_text IS NOT NULL"):
        if src in git_ids:
            lab = "edge_" + etype.lower() if etype in EDGE_TYPES else "edge_other"
            d[lab] = d.get(lab, 0) + 1
    return cc.from_count(n, d)


def a145(c: Ctx) -> Result:
    return cc.from_count(cc.jsonl_count(
        cc.jsonl_paths(c.sidecar("hook_audit.jsonl")),
        lambda r: r.get("hook") == "transcript_logger" and r.get("event") == "skipped_duplicate"))


def a146(c: Ctx) -> Result:
    if (r := _missing(c, "belief_corroborations")) is not None:
        return r
    n = c.count("SELECT COUNT(*) FROM belief_corroborations WHERE source_type='transcript_ingest'")
    d: dict[str, int] = {}
    if c.column_exists("belief_corroborations", "speaker"):
        d = c.group_count("SELECT speaker, COUNT(*) FROM belief_corroborations "
                          "WHERE source_type='transcript_ingest' GROUP BY 1",
                          {"user", "assistant"})
    return cc.from_count(n, d)


def a147(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log", "beliefs")) is not None:
        return r
    kinds = ",".join("'" + k + "'" for k in CLASSIFIER_KINDS)
    d = c.group_count(
        "SELECT b.type, COUNT(DISTINCT b.id) FROM ingest_log l, json_each(l.derived_belief_ids) j "
        "JOIN beliefs b ON b.id = j.value "
        "WHERE l.derived_belief_ids IS NOT NULL AND json_valid(l.derived_belief_ids) "
        "AND l.source_kind IN (" + kinds + ") AND (l.raw_meta IS NULL OR NOT json_valid(l.raw_meta) "
        "OR (json_extract(l.raw_meta,'$.override_belief_type') IS NULL "
        "AND json_extract(l.raw_meta,'$.route_overrides') IS NULL)) GROUP BY 1",
        BELIEF_TYPES)
    return cc.from_count(sum(d.values()), d)


def a148(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    n = c.count("SELECT COUNT(*) FROM ingest_log WHERE source_kind='filesystem' "
                "AND json_valid(raw_meta) AND json_extract(raw_meta,'$.route_overrides') IS NOT NULL")
    d: dict[str, int] = {}
    if c.table_exists("feedback_history"):
        d["fallback_rows"] = c.count("SELECT COUNT(*) FROM feedback_history "
                                     "WHERE source LIKE 'onboard.llm.fallback:%'")
    return cc.from_count(n, d)


def a149(c: Ctx) -> Result:
    if (r := _missing(c, "belief_documents")) is not None:
        return r
    d = c.group_count("SELECT anchor_type, COUNT(*) FROM belief_documents GROUP BY 1",
                      {"ingest", "manual", "derived"})
    return cc.from_count(sum(d.values()), d)


def a150(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    return cc.from_count(c.count(
        "SELECT COUNT(*) FROM ingest_log WHERE source_kind='filesystem' "
        "AND source_path LIKE 'doc:%'"))


def a151(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    return cc.from_count(c.count(
        "SELECT COUNT(*) FROM ingest_log WHERE source_kind='filesystem' "
        "AND source_path LIKE 'git:commit:%'"))


def a152(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    base = ("FROM ingest_log l WHERE l.source_kind='filesystem' "
            "AND l.source_path NOT LIKE 'git:commit:%' ")
    accept = 0
    if c.table_exists("onboard_sessions"):
        accept = c.count(
            "SELECT COUNT(*) " + base + "AND l.session_id IN (SELECT session_id FROM "
            "onboard_sessions) AND l.ts <> (SELECT o.completed_at FROM onboard_sessions o "
            "WHERE o.session_id = l.session_id)")
        scan_filter = "AND l.session_id NOT IN (SELECT session_id FROM onboard_sessions) "
    else:
        scan_filter = ""
    scan = c.count(
        "SELECT COUNT(*) " + base + scan_filter + "AND l.ts < (SELECT MAX(l2.ts) FROM "
        "ingest_log l2 WHERE l2.source_kind='filesystem' AND l2.session_id = l.session_id)")
    return cc.from_count(accept + scan, {"handshake": accept, "scan": scan},
                         "scan count is a lower bound: a session whose rows all carry "
                         "commit dates loses its newest")


def a154(c: Ctx) -> Result:
    if (r := _missing(c, "onboard_sessions")) is not None:
        return r
    d = c.group_count("SELECT state, COUNT(*) FROM onboard_sessions GROUP BY 1",
                      {"pending", "completed"})
    n = sum(d.values())
    if c.table_exists("ingest_log"):
        d["override_rows"] = c.count(
            "SELECT COUNT(*) FROM ingest_log WHERE json_valid(raw_meta) "
            "AND json_extract(raw_meta,'$.override_belief_type') IS NOT NULL")
    return cc.from_count(n, d)


_USER_TRANSCRIPT_ROW = (
    "source_kind='transcript' AND source_path='transcript' AND json_valid(raw_meta) "
    "AND json_extract(raw_meta,'$.role')='user'")


def _user_turns_since(c: Ctx, since: str) -> list[str]:
    """User turn texts logged at or after `since`, from the transcript logs."""
    out: list[str] = []
    for row in (r for p in _transcript_files(c) for r in cc.iter_jsonl(p)):
        text = row.get("text")
        if row.get("role") == "user" and str(row.get("ts", "")) >= since \
                and isinstance(text, str):
            out.append(text)
    return out


def _span_counts(texts: list[str], digests: list[object]) -> dict[str, int]:
    from aelfrice.classification_core import compute_rule_set_hash  # noqa: PLC0415
    from aelfrice.extraction import holed_form, sentence_forms  # noqa: PLC0415

    tree = compute_rule_set_hash()
    return {
        "span_in_stored": sum(1 for t in texts if holed_form(t) != t.strip()),
        "would_hole": sum(1 for t in texts
                          if any(f.whole != f.holed for f in sentence_forms(t))),
        "digest_match_tree": sum(1 for d in digests if d == tree),
        "digest_null": sum(1 for d in digests if d is None),
    }


def a156(c: Ctx) -> Result:
    """User sentences keep inline code and URLs (#1874): raw_meta.holed.

    fired when any transcript row carries `holed`; untraceable when no user
    row was written after the merge, since only those can show it; error
    when a stored post-merge row still holds a span without `holed`.
    """
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    rows = list(_rows(c, "SELECT raw_text, rule_set_hash, "
                         "json_extract(raw_meta,'$.holed') IS NOT NULL FROM ingest_log "
                         "WHERE " + _USER_TRANSCRIPT_ROW + " AND ts >= ?", (SPAN_KEEP_MERGE,)))
    holed_all = c.count(
        "SELECT COUNT(*) FROM ingest_log WHERE source_kind='transcript' "
        "AND json_valid(raw_meta) AND json_extract(raw_meta,'$.holed') IS NOT NULL")
    d: dict[str, int] = {"user_post": len(rows), "holed": holed_all}
    if not rows:
        return cc.Result("untraceable", None, d, "no user rows after the merge")
    d.update(_span_counts([str(r[0] or "") for r in rows], [r[1] for r in rows]))
    d.pop("would_hole")
    d["holed_post"] = sum(1 for r in rows if r[2])
    turns = _user_turns_since(c, SPAN_KEEP_MERGE)
    d["user_turns_post"] = len(turns)
    if turns:
        d["turns_would_hole"] = _span_counts(turns, [])["would_hole"]
    if holed_all > 0:
        return cc.fired(holed_all, d, "holed rows present")
    if d["span_in_stored"] > d["holed_post"]:
        return cc.Result("error", None, d, "span kept without holed key")
    if d.get("turns_would_hole", 0) > 0:
        return cc.silent(0, d, "post-merge turns held spans that no row kept")
    return cc.silent(0, d, "post-merge user rows hold no span")


def a157(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log", "edges")) is not None:
        return r
    tids = _ids_from_log(c, "l.source_kind='transcript'")
    logged: set[tuple[str, str, str]] = set()
    pre = 0
    for (s,) in _rows(c, "SELECT derived_edge_ids FROM ingest_log WHERE source_kind='transcript'"):
        if s is None:
            pre += 1
            continue
        try:
            v: Any = json.loads(s)
        except ValueError:
            continue
        if isinstance(v, list):
            for e in v:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(e, list) and len(e) == 3:  # pyright: ignore[reportUnknownArgumentType]
                    logged.add((str(e[0]), str(e[1]), str(e[2])))  # pyright: ignore[reportUnknownArgumentType]
    n = 0
    intra = 0
    for src, dst in _rows(c, "SELECT src, dst FROM edges WHERE type='DERIVED_FROM'"):
        if src not in tids or dst not in tids:
            continue
        if (src, dst, "DERIVED_FROM") in logged:
            intra += 1
        else:
            n += 1
    return cc.from_count(n, {"intra_turn_logged": intra, "pre1354_rows": pre},
                         "pre1354_rows > 0 means older intra-turn edges may count here")


def a158(c: Ctx) -> Result:
    """Commit-ingest sessions by era; the count is the pre-cutoff era only."""
    if (r := _missing(c, "sessions")) is not None:
        return r
    pre, win, post, all_ci = c.counts(
        "SELECT SUM(started_at < ?), SUM(started_at >= ? AND started_at < ?),"
        " SUM(started_at >= ?), COUNT(*) FROM sessions WHERE model='commit-ingest'",
        (GIT_HOOK_CUTOFF, GIT_HOOK_CUTOFF, GIT_HOOK_SHIP, GIT_HOOK_SHIP))
    bel_pre = 0
    if c.column_exists("beliefs", "session_id"):
        bel_pre = c.count(
            "SELECT COUNT(*) FROM beliefs b JOIN sessions s ON b.session_id=s.id "
            "WHERE s.model='commit-ingest' AND s.started_at < ?", (GIT_HOOK_CUTOFF,))
    detail = {
        "pre_cutoff": pre, "window": win, "post_510": post, "all": all_ci,
        "beliefs_pre_cutoff": bel_pre,
        "sessions_any_post_510": c.count("SELECT COUNT(*) FROM sessions WHERE started_at >= ?",
                                         (GIT_HOOK_SHIP,)),
        "sessions_total": c.count("SELECT COUNT(*) FROM sessions"),
    }
    return cc.from_count(pre, detail,
                         note="lower bound: sessions before 2026-10-03, when no git hook existed")


def a159(c: Ctx) -> Result:
    files = _transcript_files(c)
    n = cc.jsonl_count(files, lambda r: r.get("role") == "user")
    return cc.from_count(n, {"files": len(files)})


def _a161_scan(c: Ctx, d: dict[str, int]) -> None:
    from aelfrice.noise_filter import NoiseConfig, is_noise  # noqa: PLC0415

    cfg = NoiseConfig()
    for raw_text, raw_meta in _rows(c, "SELECT raw_text, raw_meta FROM ingest_log "
                                       "WHERE source_kind='filesystem'"):
        meta = _meta(raw_meta)
        if meta.get("call_site") != "filesystem_ingest":
            continue
        noisy = int(is_noise(raw_text or "", cfg))
        key = "emit_rows" if "override_belief_type" in meta else "scan_rows"
        d[key] += 1
        d[key + "_noise"] += noisy


def _a161_onboard(c: Ctx, d: dict[str, int]) -> None:
    from aelfrice.noise_filter import NoiseConfig, is_noise  # noqa: PLC0415

    cfg = NoiseConfig()
    for (cj,) in _rows(c, "SELECT candidates_json FROM onboard_sessions"):
        d["onboard_sessions"] += 1
        try:
            items: Any = json.loads(cj) if cj else []
        except (ValueError, TypeError):
            items = []
        for it in items if isinstance(items, list) else []:  # pyright: ignore[reportUnknownVariableType]
            text: Any = it.get("text") if isinstance(it, dict) else None  # pyright: ignore[reportUnknownMemberType]
            if isinstance(text, str):
                d["onboard_cands"] += 1
                d["onboard_cands_noise"] += int(is_noise(text, cfg))


def a161(c: Ctx) -> Result:
    """Onboard noise filter (noise_filter.is_noise in scan_repo).

    A dropped candidate leaves no row, so the probe cannot count a fire. It
    counts the falsifiable side instead: scan rows (filtered) and emitted
    handshake rows (never filtered) that the default rules call noise.
    untraceable when scan rows exist (the filter ran, drops are not
    recorded); silent when onboard never scanned this store.
    """
    if (r := _missing(c)) is not None:
        return r
    d = {k: 0 for k in ("scan_rows", "scan_rows_noise", "emit_rows", "emit_rows_noise",
                        "onboard_sessions", "onboard_cands", "onboard_cands_noise")}
    if c.table_exists("ingest_log"):
        _a161_scan(c, d)
    if c.table_exists("onboard_sessions"):
        _a161_onboard(c, d)
    if d["scan_rows"] > 0:
        return cc.Result("untraceable", None, d, "filter ran; drops not recorded")
    return cc.silent(0, d, "no scan_repo rows")


def a162(c: Ctx) -> Result:
    if (r := _missing(c, "onboard_rejections")) is not None:
        return r
    return cc.from_count(c.count("SELECT COUNT(*) FROM onboard_rejections"),
                         note="duplicate skip leaves no trace")


def a164(c: Ctx) -> Result:
    s = _sentinel_facts(c)
    if s is None:
        return cc.silent(0)
    return cc.fired(1, {"facts": s})


def a165(c: Ctx) -> Result:
    n = 0
    for p in _archive_files(c):
        if cc.jsonl_count(p, lambda r: r.get("event") == "compaction_start") > 0:
            n += 1
    return cc.from_count(n, {"archive_files": len(_archive_files(c))},
                         "detached ingest shows up under the ingest-transcript probe")


def a166(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    d = _dropped_rows(c)["question"]
    return cc.from_count(sum(d.values()), d)


def a167(c: Ctx) -> Result:
    if (r := _missing(c, "belief_support_evidence")) is not None:
        return r
    n, applied = c.counts(
        "SELECT COUNT(*), SUM(applied <> 0) FROM belief_support_evidence "
        "WHERE kind='reworded_restatement'")
    return cc.from_count(n, {"applied": applied})


def a169(c: Ctx) -> Result:
    """The reverted SUPPORTS and SUPERSEDES ingest writers (#999, #1002, #1005).

    Trace: unanchored SUPPORTS or SUPERSEDES edges, minus SUPERSEDES rows the
    resolver explains (a RESOLVES edge on the pair or a tiebreaker feedback
    row on the loser). The kept CONTRADICTS writer is reported in detail
    only. Other edge writers can leave the same unanchored shape, so
    leftover rows are ambiguous, not fired.
    """
    if (r := _missing(c, "edges")) is not None:
        return r
    if not c.column_exists("edges", "type"):
        return cc.silent(0, {"legacy_edge_rows": c.count("SELECT COUNT(*) FROM edges")},
                         note="pre-aelfrice edges schema")
    null_anchor = "e.anchor_text IS NULL" if c.column_exists("edges", "anchor_text") else "1=1"
    explained = ("EXISTS (SELECT 1 FROM edges r WHERE r.type='RESOLVES' "
                 "AND r.src=e.src AND r.dst=e.dst)")
    if c.table_exists("feedback_history"):
        explained = ("(" + explained + " OR EXISTS (SELECT 1 FROM feedback_history f "
                     "WHERE f.belief_id=e.dst AND f.source LIKE 'contradiction_tiebreaker%'))")
    base = "SELECT COUNT(*) FROM edges e WHERE " + null_anchor + " AND "
    d: dict[str, int] = {
        "supports_null_anchor": c.count(base + "e.type='SUPPORTS'"),
        "supersedes_null_anchor": c.count(base + "e.type='SUPERSEDES'"),
        "supersedes_resolver": c.count(base + "e.type='SUPERSEDES' AND " + explained),
        "contradicts_unanchored_w1": c.count(base + "e.type='CONTRADICTS' AND e.weight=1.0"),
    }
    unexplained = (d["supports_null_anchor"] + d["supersedes_null_anchor"]
                   - d["supersedes_resolver"])
    d["unexplained"] = unexplained
    if unexplained == 0:
        return cc.silent(0, d, note="no reverted-writer shaped edges")
    return cc.ambiguous(unexplained, d,
                        "other edge writers can leave unanchored typed edges with no resolver trace")


def a170(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    n, dropped = c.counts("SELECT COUNT(*), SUM(derived_belief_ids='[]') FROM ingest_log "
                          "WHERE source_kind='transcript'")
    return cc.from_count(n, {"no_belief": dropped})


def _a171_log(c: Ctx, d: dict[str, int]) -> set[str]:
    """ingest_log role counts into d; returns sessions with user rows."""
    tr = "source_kind='transcript'"
    d["tr_rows"] = c.count(f"SELECT COUNT(*) FROM ingest_log WHERE {tr}")
    d["tr_rows_post_gate"] = c.count(
        f"SELECT COUNT(*) FROM ingest_log WHERE {tr} AND ts >= ?", (SPEAKER_GATE_SHIP,))
    d["tr_rows_post_stamp"] = c.count(
        f"SELECT COUNT(*) FROM ingest_log WHERE {tr} AND ts >= ?", (ROLE_STAMP_SHIP,))
    g = c.group_count(
        f"SELECT json_extract(raw_meta,'$.role'), COUNT(*) FROM ingest_log "
        f"WHERE {tr} AND json_valid(raw_meta) GROUP BY 1", ("user", "assistant"))
    for k, v in g.items():
        d[f"role_{k}"] = v
    sids: set[str] = set()
    for (sid,) in _rows(c, f"SELECT DISTINCT session_id FROM ingest_log WHERE {tr} "
                           "AND session_id IS NOT NULL AND json_valid(raw_meta) "
                           "AND json_extract(raw_meta,'$.role')='user'"):
        if isinstance(sid, str) and sid:
            sids.add(sid)
    return sids


def a171(c: Ctx) -> Result:
    """Speaker-attribution gate (#785): ingest_jsonl skips assistant rows.

    The gate writes nothing. A violation is a transcript ingest_log row with
    raw_meta.role='assistant' or an assistant corroboration. Exposure is
    assistant lines in the transcript logs whose session also has ingested
    user rows: lines ingest read and dropped. fired on either (exposure is
    an inferred lower bound); untraceable otherwise.
    """
    if c.db is None:
        return cc.untraceable("no db")
    d: dict[str, int] = {}
    sids = _a171_log(c, d) if c.table_exists("ingest_log") else set()
    if c.column_exists("belief_corroborations", "speaker"):
        for k, v in c.group_count("SELECT speaker, COUNT(*) FROM belief_corroborations "
                                  "GROUP BY 1", ("user", "assistant")).items():
            d[f"speaker_{k}"] = v
    d["ingested_user_sessions"] = len(sids)
    files = _transcript_files(c)
    d["log_files"] = len(files)
    rows = [r for p in files for r in cc.iter_jsonl(p)]
    d["log_assistant_lines"] = sum(1 for r in rows if r.get("role") == "assistant")
    d["log_user_lines"] = sum(1 for r in rows if r.get("role") == "user")
    exposed = sum(1 for r in rows if r.get("role") == "assistant"
                  and isinstance(r.get("session_id"), str) and r["session_id"] in sids)
    d["assistant_lines_in_ingested_sessions"] = exposed
    violation = d.get("role_assistant", 0) + d.get("speaker_assistant", 0)
    d["violation"] = violation
    if violation > 0:
        return cc.Result("error", violation, d, "contract broken: assistant rows reached ingest_log")
    if exposed > 0:
        return cc.fired(exposed, d,
                        "inferred: assistant lines read by ingest, 0 assistant rows written")
    return cc.Result("untraceable", None, d, "no ingested session with assistant lines")


def a172(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    d = _dropped_rows(c)["float"]
    return cc.from_count(sum(d.values()), d)


def a173(c: Ctx) -> Result:
    p = c.sidecar("transcripts", ".stop_flush_cursor")
    if not p.is_file():
        return cc.silent(0)
    try:
        parts = p.read_text("utf-8", "replace")[:200].split()
        turns = int(parts[-1]) if parts else 0
    except (OSError, ValueError):
        turns = 0
    return cc.fired(1, {"turns_at_last_flush": turns})


def a174(c: Ctx) -> Result:
    if (r := _missing(c, "ingest_log")) is not None:
        return r
    return cc.from_count(c.count(
        "SELECT COUNT(*) FROM ingest_log WHERE source_kind='transcript' "
        "AND json_valid(raw_meta) AND json_extract(raw_meta,'$.derived_from') IS NOT NULL"))


_TRANSCRIPT_IDS_CTE = (
    "WITH t(id) AS (SELECT DISTINCT j.value FROM ingest_log l, "
    "json_each(l.derived_belief_ids) j WHERE l.derived_belief_ids IS NOT NULL "
    "AND json_valid(l.derived_belief_ids) AND l.source_kind='transcript') ")
# Writer-eligible: a transcript belief with an earlier belief in its session.
_SPINE_ELIGIBLE = (
    _TRANSCRIPT_IDS_CTE + ", e AS (SELECT b1.id, b1.created_at FROM beliefs b1 "
    "JOIN t ON t.id=b1.id WHERE b1.session_id IS NOT NULL AND EXISTS (SELECT 1 "
    "FROM beliefs b2 WHERE b2.session_id=b1.session_id AND (b2.created_at<b1.created_at "
    "OR (b2.created_at=b1.created_at AND b2.rowid<b1.rowid)))) ")
_SPINE_EDGE = "x.type='TEMPORAL_NEXT' AND x.anchor_text IS NULL AND ABS(x.weight-0.8)<1e-9"


def _spine_schema_ok(c: Ctx) -> bool:
    return (c.column_exists("edges", "type") and c.column_exists("edges", "anchor_text")
            and c.table_exists("ingest_log") and c.column_exists("beliefs", "session_id")
            and c.column_exists("beliefs", "created_at"))


def a175(c: Ctx) -> Result:
    """Temporal-spine TEMPORAL_NEXT writer at ingest.

    Count: spine-shaped edges (weight 0.8, no anchor) out of transcript
    beliefs. `aelf spine backfill` writes the same edge, and it leaves the
    sentinel only on some paths, so a count above 0 is ambiguous. silent
    splits into no eligible belief and eligible beliefs with no edge; a
    schema with no typed edges or ingest_log predates the writer and is
    untraceable.
    """
    if c.db is None:
        return cc.untraceable("no db")
    if not _spine_schema_ok(c):
        return cc.untraceable("legacy schema: no typed edges or ingest_log; predates writer")
    covered = "EXISTS (SELECT 1 FROM edges x WHERE x.src=e.id AND " + _SPINE_EDGE + ")"
    d: dict[str, int] = {
        "transcript_beliefs": c.count(_TRANSCRIPT_IDS_CTE + "SELECT COUNT(*) FROM t"),
        "eligible": c.count(_SPINE_ELIGIBLE + "SELECT COUNT(*) FROM e"),
        "eligible_since_v4": c.count(_SPINE_ELIGIBLE + "SELECT COUNT(*) FROM e "
                                     "WHERE created_at>=?", (SPINE_DEFAULT_ON,)),
        "eligible_covered": c.count(_SPINE_ELIGIBLE + "SELECT COUNT(*) FROM e WHERE " + covered),
        "backfill_sentinel": _global_flag(c, "spine-backfilled"),
    }
    n = c.count(_TRANSCRIPT_IDS_CTE + "SELECT COUNT(*) FROM edges x JOIN t ON t.id=x.src "
                "WHERE " + _SPINE_EDGE)
    d["spine_non_transcript"] = c.count(
        _TRANSCRIPT_IDS_CTE + "SELECT COUNT(*) FROM edges x WHERE " + _SPINE_EDGE
        + " AND x.src NOT IN (SELECT id FROM t)")
    if n > 0:
        return cc.ambiguous(n, d, "aelf spine backfill writes the same edge, sentinel or not")
    if d["eligible"] == 0:
        return cc.silent(0, d, "no writer-eligible belief")
    return cc.silent(0, d, "eligible beliefs carry no spine edge")


def a176(c: Ctx) -> Result:
    files = _transcript_files(c)
    return cc.from_count(len(files), {"archive_files": len(_archive_files(c))})


_A177_KEYS = ("tx_rows", "tx_rows_sha", "log_turns", "user_records", "joined_turns",
              "joined_noise_turns", "noise_dropped", "noise_anchored", "noise_logged",
              "leak_rows", "leak_rows_sha")


def _a177_log_turns(c: Ctx, d: dict[str, int]) -> dict[str, tuple[set[str], list[str]]]:
    """turn_sha -> (logged sentence texts, derived_from anchor texts)."""
    from aelfrice.derivation import META_DERIVED_FROM, META_HOLED, META_TURN_SHA  # noqa: PLC0415
    from aelfrice.noise_filter import is_transcript_noise  # noqa: PLC0415

    turns: dict[str, tuple[set[str], list[str]]] = {}
    for raw_text, raw_meta in _rows(c, "SELECT raw_text, raw_meta FROM ingest_log "
                                       "WHERE source_kind='transcript'"):
        d["tx_rows"] += 1
        meta = _meta(raw_meta)
        h = meta.get(META_HOLED)
        holed = h if isinstance(h, str) else None
        text = raw_text if isinstance(raw_text, str) else ""
        noisy = is_transcript_noise(holed or text)
        d["leak_rows"] += int(noisy)
        sha = meta.get(META_TURN_SHA)
        if not isinstance(sha, str):
            continue
        d["tx_rows_sha"] += 1
        d["leak_rows_sha"] += int(noisy)
        logged, anchors = turns.setdefault(sha, (set(), []))
        logged.update(t for t in (text, holed) if t)
        df = meta.get(META_DERIVED_FROM)
        anchor: Any = df.get("anchor_text") if isinstance(df, dict) else None  # pyright: ignore[reportUnknownMemberType]
        if isinstance(anchor, str):
            anchors.append(anchor)
    return turns


def _a177_join(c: Ctx, d: dict[str, int], turns: dict[str, tuple[set[str], list[str]]]) -> None:
    from aelfrice.extraction import sentence_forms  # noqa: PLC0415
    from aelfrice.ingest import (  # noqa: PLC0415
        _normalize_jsonl_turn,  # pyright: ignore[reportPrivateUsage]
        _user_speech,  # pyright: ignore[reportPrivateUsage]
        turn_sha,
    )
    from aelfrice.noise_filter import is_transcript_noise  # noqa: PLC0415

    seen: set[str] = set()
    for obj in (r for p in _transcript_files(c) for r in cc.iter_jsonl(p)):
        norm = _normalize_jsonl_turn(obj)
        if not norm or norm.get("role") != "user" or not norm.get("text"):
            continue
        speech = _user_speech(str(norm["text"]))
        if not speech:
            continue
        d["user_records"] += 1
        sha = turn_sha(speech)
        if sha not in turns or sha in seen:
            continue
        seen.add(sha)
        logged, anchors = turns[sha]
        blob = " | ".join(anchors)
        noise = [f for f in sentence_forms(speech) if is_transcript_noise(f.holed)]
        d["joined_noise_turns"] += int(bool(noise))
        for f in noise:
            if f.whole in logged or f.holed in logged:
                d["noise_logged"] += 1
            elif f.whole in blob or f.holed in blob:
                d["noise_anchored"] += 1
            else:
                d["noise_dropped"] += 1
    d["joined_turns"] = len(seen)


def a177(c: Ctx) -> Result:
    """Transcript sentence noise filter (sentence_partition.partition_sentences).

    A dropped noise sentence writes no row, but its turn does: since 5.1.0
    (#1602) every sentence row carries raw_meta.turn_sha. The probe rebuilds
    each user turn's speech and turn_sha from the transcript logs, joins to
    the logged rows, and re-splits the speech. A noise sentence missing from
    its turn's rows and from every derived_from anchor was dropped at ingest.
    fired on a drop; silent when turns join and none dropped; untraceable
    when no turn joins.
    """
    if (r := _missing(c)) is not None:
        return r
    d = {k: 0 for k in _A177_KEYS}
    turns = _a177_log_turns(c, d) if c.table_exists("ingest_log") else {}
    d["log_turns"] = len(turns)
    _a177_join(c, d, turns)
    if d["noise_dropped"] > 0:
        return cc.fired(d["noise_dropped"], d)
    if d["joined_turns"] > 0:
        return cc.silent(0, d, "joined turns carry no dropped noise sentence")
    return cc.Result("untraceable", None, d, "no transcript turn joins ingest_log by turn_sha")


def a178(c: Ctx) -> Result:
    if (r := _missing(c, "belief_documents")) is not None:
        return r
    n = c.count("SELECT COUNT(*) FROM belief_documents WHERE anchor_type='ingest' "
                "AND position_hint IS NOT NULL")
    return cc.from_count(n, {"transcript_label": c.count(
        "SELECT COUNT(*) FROM belief_documents WHERE doc_uri LIKE 'file:transcript#%'")})


def _near(a: str, v: float) -> str:
    return f"ABS({a} - {v!r}) < 1e-9"


_DEFLATED = (
    f"((type IN ('requirement','correction') AND {_near('alpha', 1.8)} AND {_near('beta', 0.5)})"
    f" OR (type='preference' AND {_near('alpha', 1.4)} AND {_near('beta', 1.0)})"
    f" OR (type='factual' AND {_near('alpha', 0.6)} AND {_near('beta', 1.0)}))")
_FULL_PRIOR = (
    f"((type IN ('requirement','correction') AND {_near('alpha', 9.0)} AND {_near('beta', 0.5)})"
    f" OR (type='preference' AND {_near('alpha', 7.0)} AND {_near('beta', 1.0)})"
    f" OR (type='factual' AND {_near('alpha', 3.0)} AND {_near('beta', 1.0)}))")
_PRIOR_COLUMNS = ("type", "origin", "alpha", "beta")


def a179(c: Ctx) -> Result:
    """Type priors and non-user deflation; a predecessor schema is silent."""
    if (r := _missing(c, "beliefs")) is not None:
        return r
    total = c.count("SELECT COUNT(*) FROM beliefs")
    if not all(c.column_exists("beliefs", col) for col in _PRIOR_COLUMNS):
        return cc.silent(0, {"beliefs_total": total},
                         "predecessor schema: no type, origin, beta columns")
    d = c.count("SELECT COUNT(*) FROM beliefs WHERE origin='agent_inferred' AND " + _DEFLATED)
    u = c.count("SELECT COUNT(*) FROM beliefs WHERE origin='user_transcript' AND " + _FULL_PRIOR)
    return cc.from_count(d + u, {
        "deflated": d, "undeflated": u, "beliefs_total": total,
        "agent_inferred": c.count("SELECT COUNT(*) FROM beliefs WHERE origin='agent_inferred'"),
        "user_transcript": c.count("SELECT COUNT(*) FROM beliefs WHERE origin='user_transcript'"),
    }, "lower bound: feedback moves alpha and beta off the prior")


def a180(c: Ctx) -> Result:
    rdir = c.sidecar("rebuild_logs")
    files = sorted(p for p in rdir.glob("*.jsonl") if p.is_file()) if rdir.is_dir() else []

    def triple_added(r: dict[str, Any]) -> bool:
        inp: Any = r.get("input")
        if not isinstance(inp, dict):
            return False
        q: Any = inp.get("extracted_query")  # pyright: ignore[reportUnknownMemberType]
        ents: Any = inp.get("extracted_entities")  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(q, str) or not q or not isinstance(ents, list) or not ents:
            return False
        return q != " ".join(str(e) for e in ents)  # pyright: ignore[reportUnknownVariableType]

    def has_query(r: dict[str, Any]) -> bool:
        inp: Any = r.get("input")
        q: Any = inp.get("extracted_query") if isinstance(inp, dict) else None  # pyright: ignore[reportUnknownMemberType]
        return isinstance(q, str) and bool(q)

    n = cc.jsonl_count(files, triple_added)
    return cc.from_count(n, {"rows_with_query": cc.jsonl_count(files, has_query)},
                         "counts rows where triple phrases extend the entity list")


_USER_PRIOR_TYPES = ("factual", "preference", "requirement", "correction")


def a182(c: Ctx) -> Result:
    """user_transcript origin; a beliefs table with no origin column is silent."""
    if (r := _missing(c, "beliefs")) is not None:
        return r
    nb = c.count("SELECT COUNT(*) FROM beliefs")
    if not c.column_exists("beliefs", "origin") or not c.column_exists("beliefs", "type"):
        return cc.silent(0, {"beliefs": nb}, "no origin column")
    d = {"ut_" + k: v for k, v in c.group_count(
        "SELECT type, COUNT(*) FROM beliefs WHERE origin='user_transcript' GROUP BY 1",
        _USER_PRIOR_TYPES).items()}
    n = sum(d.values())
    d["ut_undeflated"] = c.count(
        "SELECT COUNT(*) FROM beliefs WHERE origin='user_transcript' AND alpha >= 3.0")
    d["beliefs"] = nb
    el = der = 0
    if c.table_exists("ingest_log"):
        el, der = c.counts(
            "SELECT COUNT(*), SUM(derived_belief_ids IS NOT NULL AND derived_belief_ids != '[]') "
            "FROM ingest_log WHERE " + _USER_TRANSCRIPT_ROW)
    d["eligible_rows"] = el
    d["eligible_derived"] = der
    note = "eligible rows but no user_transcript" if (n == 0 and der > 0) else ""
    return cc.from_count(n, d, note)


PROBES: list[cc.Probe] = [
    U("A132", "aelf audit-claude-memory",
      "read-only command; compare_slots writes nothing"),
    P("A133", "aelf doctor --prune-noise", "store", a133),
    P("A134", "aelf ingest-transcript", "store", a134),
    U("A135", "aelf onboard --check pre-scan",
      "check_onboard_candidates writes no session, belief or log row"),
    U("A136", "aelf scan-derivation n-gram gate",
      "handler opens no store and writes no file"),
    U("A137", "ANTHROPIC_BASE_URL gateway",
      "the SDK reads the env var; aelfrice records no endpoint"),
    P("A138", "Assistant-reply extraction for Stop", "sidecar", a138),
    P("A139", "AST docstring extractor", "store", a139),
    P("A140", "Claude-memory write-through mirror", "store", a140),
    P("A141", "Commit ingest via git post-commit hook", "store", a141),
    U("A142", "Commit session keying and skips",
      "keying writes the same sessions row as the legacy path; revert, rebase and "
      "self-reread skips write nothing"),
    P("A143", "Commit store routing per changed folder", "store", a143),
    P("A144", "Commit triple extractor and belief derivation", "store", a144),
    P("A145", "Consecutive-duplicate turn guard", "sidecar", a145),
    P("A146", "Corroboration on re-assertion", "store", a146),
    P("A147", "Deterministic sentence classifier type", "store", a147),
    P("A148", "Direct-API LLM onboard classifier", "store", a148),
    P("A149", "Document linker", "store", a149),
    P("A150", "Document scanner paragraph extractor", "store", a150),
    P("A151", "Git log commit-subject extractor", "store", a151),
    P("A152", "Git-recency created_at", "store", a152),
    U("A153", "Headless session capture skip",
      "a skipped headless record or turn writes nothing"),
    P("A154", "Host-model onboard handshake", "store", a154),
    U("A155", "INEDIBLE basename opt-out",
      "an INEDIBLE file is skipped before any read or write"),
    P("A156", "Inline code and URLs kept with credential scrub", "store", a156),
    P("A157", "Inter-turn DERIVED_FROM chain", "store", a157),
    P("A158", "Legacy PostToolUse commit ingest", "store", a158),
    P("A159", "Live transcript logger", "sidecar", a159),
    U("A160", "Logger prompt gate",
      "a gated prompt is not appended and no skip row is written"),
    P("A161", "Onboard noise filter", "store", a161),
    P("A162", "Onboard rejection ledger", "store", a162),
    U("A163", "Onboard skip dirs", "skipped directories are never read or logged"),
    P("A164", "One-shot claude-memory reconcile", "sidecar", a164),
    P("A165", "PreCompact rotation", "sidecar", a165),
    P("A166", "Question drop", "store", a166),
    P("A167", "Reworded-restatement SUPPORTS log", "store", a167),
    U("A168", "Scheduled and meta prompt skip",
      "a scheduled or meta prompt is not appended and no skip row is written"),
    P("A169", "Semantic edge-writer substrate", "store", a169),
    P("A170", "Sentence extraction", "store", a170),
    P("A171", "Speaker-attribution gate", "store", a171),
    P("A172", "Speculative-float drop", "store", a172),
    P("A173", "Stop-cadence ingest flush", "sidecar", a173),
    P("A174", "Sub-floor fragment demotion", "store", a174),
    P("A175", "Temporal-spine TEMPORAL_NEXT writer", "store", a175),
    P("A176", "Transcript log location", "sidecar", a176),
    P("A177", "Transcript sentence noise filter", "store", a177),
    P("A178", "Turn-anchored transcript doc anchors", "store", a178),
    P("A179", "Type priors and non-user deflation", "store", a179),
    P("A180", "Read-path triple extraction (rebuild_log)", "sidecar", a180),
    U("A181", "User-speech extraction from host records",
      "cut text never reaches ingest_log; the remainder is indistinguishable"),
    P("A182", "user_transcript origin", "store", a182),
    U("A183", "UTF-8 stream reconfiguration", "stream setup writes nothing"),
]

TRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind != "untraceable")
UNTRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind == "untraceable")
# Store probes that report untraceable, not silent, on an empty store: no
# row they could join or count exists there.
UNTRACEABLE_WHEN_EMPTY: frozenset[str] = frozenset({"A143", "A156", "A171", "A177"})
