"""Census probes for the hook-injection subsystem (A077-A131).

Every literal copied from aelfrice is checked against aelfrice's own constant
by `tests/test_real_store_fire_census_hook_injection.py`. Only A123 imports
aelfrice, and only pure functions (`decision_text`, `detect_directive`).

Sources these probes read (all written by aelfrice):
    hook_audit.jsonl (+.1)   hook.py _write_hook_audit_record: rows hook in
                             {user_prompt_submit, session_start} carry the
                             post-ceiling rendered_block; sidecar_warm.py
                             writes hook=sidecar_warm rows.
    command_outcomes.jsonl   hook.py _write_command_outcome_record (typed lock).
    rebuild_logs/<sid>.jsonl rebuild_log.record_user_prompt_submit_log (UPS rows:
                             one synthetic turn, every score None) and
                             context_rebuilder.rebuild_v14 (rebuilder rows).
    cadence_resume_cache.json, cadence_shadow/<sid>.jsonl   hook.py cadence.
    session_first_prompt.json, session_exclusions.json,
    session_injection_ledger.json, session_injected_ids.json,
    telemetry/*.jsonl        hook-side files.
    transcripts/turns.jsonl (+ archive/)  transcript_logger compaction markers.
    store tables feedback_history (source='hook'), injection_events,
    belief_touches, exploration_events, categories, core_gate_batches.

Output is integers and fixed labels only.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import core as cc

P = cc.Probe

# --- literals copied from aelfrice (checked by the test file) --------------

AUDIT_FILENAME = "hook_audit.jsonl"                     # hook_audit.AUDIT_FILENAME
COMMAND_OUTCOMES_FILENAME = "command_outcomes.jsonl"    # hook_audit
COMMAND_OUTCOME_HOOK = "aelf_command"                   # hook_audit
HOOK_UPS = "user_prompt_submit"                         # hook.AUDIT_HOOK_USER_PROMPT_SUBMIT
HOOK_SS = "session_start"                               # hook.AUDIT_HOOK_SESSION_START
HOOK_WARM = "sidecar_warm"                              # sidecar_warm.WARM_AUDIT_HOOK
REBUILD_LOG_DIRNAME = "rebuild_logs"                    # rebuild_log
CADENCE_RESUME_CACHE = "cadence_resume_cache.json"      # hook._CADENCE_RESUME_CACHE_FILENAME
CADENCE_SHADOW_DIRNAME = "cadence_shadow"               # cadence.CADENCE_SHADOW_DIRNAME
SESSION_STATE_FILENAME = "session_first_prompt.json"    # hook.SESSION_STATE_FILENAME
EXCLUSIONS_FILENAME = "session_exclusions.json"         # session_exclusions
LEDGER_FILENAME = "session_injection_ledger.json"       # injection_ledger
RING_FILENAME = "session_injected_ids.json"             # session_ring.SESSION_RING_FILENAME
RECAP_LAST_TS_FILENAME = "sessionstart_last.txt"        # hook._RECAP_LAST_TS_FILENAME
FEED_FILENAME = "feed.jsonl"                            # feed_log.FEED_FILENAME
UPS_TELEMETRY = ("telemetry", "user_prompt_submit.jsonl")   # hook._telemetry_path_for_db
SEARCH_TELEMETRY = ("telemetry", "search_tool_hook.jsonl")  # hook_search_tool
TRANSCRIPTS_DIRNAME = "transcripts"                     # transcript_logger.TRANSCRIPTS_SUBDIR
TURNS_FILENAME = "turns.jsonl"                          # transcript_logger.TURNS_FILENAME

OPEN_TAG = "<aelfrice-memory>"
CLOSE_TAG = "</aelfrice-memory>"
BASELINE_OPEN_TAG = "<aelfrice-baseline>"
SESSION_START_OPEN = "<session-start>"
SESSION_START_CLOSE = "</session-start>"
CORE_OPEN = "<core>"
CORE_CLOSE = "</core>"
RECENT_WORK_OPEN = "<recent-work>"
RESUME_OPEN = "<cadence-resume"
CATEGORY_FOCUS_OPEN = "<category-focus>"
FRAMING_HEAD = "The memory store contents below are in two trust tiers."
SPECULATIVE_SENTENCE_HEAD = ' Items marked speculative="1" are machine-synthesised'
TRUNCATION_MARKER = " […truncated]"                     # hook._cap_belief_content
PROVENANCE_SECTIONS = ("user-locked", "observed", "inferred")  # provenance_render
COVERAGE_RE = re.compile(r"retrieved \d+ of \d+ matching beliefs for \"")
LOCK_OVERFLOW_RE = re.compile(
    r"\naelfrice: \d+ user lock\(s\) did not fit this hook's output limit")
SEEN_LINE_RE = re.compile(r"\n  seen [^\s:]+: \"")       # retrieval.seen_manifest_line
CORE_LINE_RE = re.compile(r'^<belief id="[^"]*" corr="')  # hook._core_belief_line
NONLOCK_ELEM_RE = re.compile(r'<belief id="[^"]*" lock="none"[^>]*>.*?</belief>', re.DOTALL)

HOOK_PAYLOAD_CHAR_LIMIT = 9_500                         # hook_payload
SESSION_START_CORE_BUDGET = 1500                        # hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET
CORE_NEAR_CAP_TOKENS = 1275                             # 85 percent of the budget
HOOK_TOKEN_BUDGET = 1500                                # hook.DEFAULT_HOOK_TOKEN_BUDGET
HOOK_BLOCK_TOKEN_CEILING = 6000                         # hook.HOOK_BLOCK_TOKEN_CEILING
PAYLOAD_TOKENS = HOOK_PAYLOAD_CHAR_LIMIT // 4
MAX_CONV_AWARE_WEIGHT = 8                               # hook.MAX_CONV_AWARE_WEIGHT
CADENCE_RESUME_TTL_S = 3600                             # hook._CADENCE_RESUME_TTL_SECONDS
RECAP_THRESHOLD = 3                                     # hook._DEFAULT_RECAP_THRESHOLD

FEEDBACK_SOURCE_HOOK = "hook"                           # hook_search.HOOK_FEEDBACK_SOURCE
LOCK_INGEST_KINDS = ("cli_remember", "mcp_remember")    # models.INGEST_SOURCE_*_REMEMBER
REVIEW_LOCK_SOURCE = "review:lock"                      # review.py

POLICY_P1 = "p1_every_k_turns"
POLICY_P2 = "p2_ctx_threshold"
POLICY_P3V = "p3_velocity"
POLICY_P3S = "p3_substantive"
POLICIES = (POLICY_P1, POLICY_P2, POLICY_P3V, POLICY_P3S)
DEFAULT_P2_WATERMARK = 300_000                          # int(0.50 * 600000)
DEFAULT_P3_VELOCITY_THRESHOLD = 3000
DEFAULT_P3_SUBSTANTIVE_WINDOW = 10
DEFAULT_P3_SUBSTANTIVE_THRESHOLD = "0.600"
REBUILDER_TURN_WINDOW = 50                              # rebuild_log default turn_window_n

GATE_LABELS = ("system-tag", "trivial:short", "trivial:ack", "trivial:token-count")
WARM_LABELS = ("fresh", "incremental", "full_rebuild")
COMMAND_REASONS = ("ok", "empty_argument", "over_cap", "leading_dash",
                   "invalid_text", "exception", "nonzero_exit")
BASH_COMMANDS = ("grep", "egrep", "fgrep", "rg", "ripgrep", "ack", "find", "fd", "fdfind")

# Release days the corrected probes date rows against (CHANGELOG).
PAYLOAD_CAP_DAY = "2026-10-06"      # 5.1.0: 9,500-char hook output cap (#1639)
CEILING_ERA_START = "2026-09-13"    # 6000-token ceiling (#1551) was the only trim limit
REL_4_3_0 = "2026-08-13"            # scored_query recorded from here (#1405)
SPEC_FRAMING_SHIP = "2026-08-14"    # day after 4.3.0, speculative framing (#1171)
REL_5_0_0 = "2026-09-18"            # stack-r1-r3 back to opt-in (#1501); token unit (#1526)
AFTER_5_1_0 = "2026-10-07"          # day after 5.1.0, which removed the reference sweeper
SHIP_1031 = "2026-06-30"            # 3.8.0: SessionStart(compact) carries the rebuild block
COMPACT_WINDOW_S = 30
_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


# --- helpers ---------------------------------------------------------------


def _audit_paths(ctx: cc.Ctx) -> list[Path]:
    return cc.jsonl_paths(ctx.sidecar(AUDIT_FILENAME))


def _audit_rows(ctx: cc.Ctx, hooks: Iterable[str]) -> Iterator[dict[str, Any]]:
    want = set(hooks)
    for p in _audit_paths(ctx):
        for r in cc.iter_jsonl(p):
            if r.get("hook") in want:
                yield r


def _block(r: dict[str, Any]) -> str:
    v = r.get("rendered_block")
    return v if isinstance(v, str) else ""


def _blocks(ctx: cc.Ctx, hooks: Iterable[str] = (HOOK_UPS,)) -> Iterator[str]:
    for r in _audit_rows(ctx, hooks):
        yield _block(r)


def _section(text: str, open_tag: str, close_tag: str) -> str | None:
    i = text.find(open_tag)
    if i < 0:
        return None
    j = text.find(close_tag, i + len(open_tag))
    if j < 0:
        return text[i + len(open_tag):]
    return text[i + len(open_tag):j]


def _core_lines(block: str) -> list[str]:
    ss = _section(block, SESSION_START_OPEN, SESSION_START_CLOSE)
    if ss is None:
        return []
    core = _section(ss, CORE_OPEN, CORE_CLOSE)
    if core is None:
        return []
    return [ln for ln in core.split("\n") if CORE_LINE_RE.match(ln)]


def _rebuild_rows(ctx: cc.Ctx) -> Iterator[dict[str, Any]]:
    d = ctx.sidecar(REBUILD_LOG_DIRNAME)
    if not d.is_dir():
        return
    for p in sorted(d.glob("*.jsonl")):
        for r in cc.iter_jsonl(p):
            if r.get("truncated") is True:
                continue
            if isinstance(r.get("input"), dict):
                yield r


def _cands(r: dict[str, Any]) -> list[dict[str, Any]]:
    c: Any = r.get("candidates")
    if not isinstance(c, list):
        return []
    return [dict(x) for x in c if isinstance(x, dict)]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


def _is_ups_rebuild(r: dict[str, Any]) -> bool:
    """UPS rows: exactly one synthetic turn and every candidate score None."""
    if r["input"].get("n_recent_turns") != 1:
        return False
    for c in _cands(r):
        s: Any = c.get("scores")
        if isinstance(s, dict) and any(v is not None for v in s.values()):  # pyright: ignore[reportUnknownVariableType]
            return False
    return True


def _rebuilder_rows(ctx: cc.Ctx) -> Iterator[dict[str, Any]]:
    for r in _rebuild_rows(ctx):
        if not _is_ups_rebuild(r):
            yield r


def _ups_rebuild_rows(ctx: cc.Ctx) -> Iterator[dict[str, Any]]:
    for r in _rebuild_rows(ctx):
        if _is_ups_rebuild(r):
            yield r


def _pack(r: dict[str, Any], key: str) -> int:
    ps: Any = r.get("pack_summary")
    v: Any = ps.get(key) if isinstance(ps, dict) else None  # pyright: ignore[reportUnknownMemberType]
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return None


def _cache_policy(ctx: cc.Ctx) -> str | None:
    obj = _read_json(ctx.sidecar(CADENCE_RESUME_CACHE))
    if not isinstance(obj, dict):
        return None
    pol: Any = obj.get("policy")  # pyright: ignore[reportUnknownMemberType]
    return pol if pol in POLICIES else ("other" if pol is not None else None)


def _shadow_rows(ctx: cc.Ctx) -> Iterator[dict[str, Any]]:
    d = ctx.sidecar(CADENCE_SHADOW_DIRNAME)
    if not d.is_dir():
        return
    for p in sorted(d.glob("*.jsonl")):
        yield from cc.iter_jsonl(p)


def _shadow_reason(row: dict[str, Any], policy: str) -> str:
    sh: Any = row.get("shadow")
    ent: Any = sh.get(policy) if isinstance(sh, dict) else None  # pyright: ignore[reportUnknownMemberType]
    v: Any = ent.get("reason") if isinstance(ent, dict) else None  # pyright: ignore[reportUnknownMemberType]
    return v if isinstance(v, str) else ""


def _cadence_fire(ctx: cc.Ctx, policy: str) -> cc.Result:
    """Stop-side fire of `policy`: the resume cache (last fire only) plus
    shadow rows whose selected policy fired."""
    cache = 1 if _cache_policy(ctx) == policy else 0
    shadow = sum(1 for r in _shadow_rows(ctx)
                 if r.get("selected") == policy and r.get("fired") is True)
    return cc.from_count(cache + shadow, {"resume_cache": cache, "shadow_fired_rows": shadow},
                         "cache keeps the last fire only; UPS-side fires leave no policy field")


def _need(ctx: cc.Ctx, *tables: str) -> bool:
    return all(ctx.table_exists(t) for t in tables)


def _no_table(note: str = "table absent") -> cc.Result:
    return cc.silent(0, {}, note)


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _row_ids(r: dict[str, Any]) -> list[str]:
    """Belief ids an audit row lists under `beliefs`."""
    raw: Any = r.get("beliefs")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for b in raw:  # pyright: ignore[reportUnknownVariableType]
        bid: Any = b.get("id") if isinstance(b, dict) else None  # pyright: ignore[reportUnknownMemberType]
        if isinstance(bid, str):
            out.append(bid)
    return out


def _parse_ts(v: Any) -> datetime | None:
    if not isinstance(v, str):
        return None
    try:
        return datetime.strptime(v, _TS_FMT)
    except ValueError:
        return None


def _ts_key(v: Any) -> int | None:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", v) if isinstance(v, str) else None
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    days = (y * 372 + mo * 31 + d)  # monotone; only small gaps are compared
    return ((days * 24 + h) * 60 + mi) * 60 + s


# --- probe functions -------------------------------------------------------


def a077(ctx: cc.Ctx) -> cc.Result:
    """<core> lines in first-prompt UPS audit rows.

    A store with no first-prompt row (<session-start> sub-block) in its audit
    window had no chance to fire, so it is untraceable rather than silent.
    """
    if not _audit_paths(ctx):
        return cc.untraceable("no hook audit log in this store")
    d = {"ups_rows": 0, "session_start_rows": 0, "core_rows": 0, "core_beliefs": 0,
         "post_cap_session_start_rows": 0, "post_cap_core_rows": 0}
    for r in _audit_rows(ctx, (HOOK_UPS,)):
        d["ups_rows"] += 1
        b = _block(r)
        if SESSION_START_OPEN not in b:
            continue
        d["session_start_rows"] += 1
        post = int(_str(r.get("ts"))[:10] >= PAYLOAD_CAP_DAY)
        d["post_cap_session_start_rows"] += post
        lines = _core_lines(b)
        if lines:
            d["core_rows"] += 1
            d["core_beliefs"] += len(lines)
            d["post_cap_core_rows"] += post
    if d["session_start_rows"] == 0:
        return cc.untraceable("no first-prompt row in the audit window")
    return cc.from_count(d["core_rows"], d)


def a078(ctx: cc.Ctx) -> cc.Result:
    rows = near = over = 0
    for b in _blocks(ctx):
        lines = _core_lines(b)
        if not lines:
            continue
        rows += 1
        cost = sum((len(ln) + 1 + 3) // 4 for ln in lines)
        if cost > SESSION_START_CORE_BUDGET:
            over += 1
        elif cost >= CORE_NEAR_CAP_TOKENS:
            near += 1
    return cc.from_count(near, {"core_rows": rows, "near_cap_rows": near, "over_cap_rows": over},
                         "binding inferred from packed cost within 15 pct of 1500; candidates not logged")


def a079(ctx: cc.Ctx) -> cc.Result:
    rows = commits = linked = 0
    for b in _blocks(ctx):
        rw = _section(b, RECENT_WORK_OPEN, "</recent-work>")
        if rw is None:
            continue
        rows += 1
        commits += 1 if '<commit sha="' in rw else 0
        linked += 1 if "<linked-issues>" in rw else 0
    return cc.from_count(rows, {"rows_with_commits": commits, "rows_with_linked_issues": linked})


def a081(ctx: cc.Ctx) -> cc.Result:
    if not _need(ctx, "beliefs", "ingest_log", "feedback_history"):
        return _no_table()
    marks = ",".join("?" * len(LOCK_INGEST_KINDS))
    n = ctx.count(
        "SELECT COUNT(*) FROM beliefs b WHERE b.lock_level='user' AND b.origin='user_stated'"
        " AND b.session_id IS NOT NULL AND b.session_id<>'' AND b.locked_at IS NOT NULL"
        " AND b.locked_at<>b.created_at"
        f" AND NOT EXISTS (SELECT 1 FROM ingest_log l WHERE l.source_kind IN ({marks})"
        "   AND instr(l.derived_belief_ids, '\"' || b.id || '\"') > 0)"
        " AND NOT EXISTS (SELECT 1 FROM feedback_history f WHERE f.belief_id=b.id AND f.source=?)",
        (*LOCK_INGEST_KINDS, REVIEW_LOCK_SOURCE))
    corr = ctx.count(
        "SELECT COUNT(*) FROM beliefs b WHERE b.lock_level='user' AND b.origin='user_stated'"
        " AND b.type='correction' AND b.session_id IS NOT NULL AND b.locked_at<>b.created_at"
        f" AND NOT EXISTS (SELECT 1 FROM ingest_log l WHERE l.source_kind IN ({marks})"
        "   AND instr(l.derived_belief_ids, '\"' || b.id || '\"') > 0)",
        LOCK_INGEST_KINDS)
    return cc.from_count(n, {"type_correction": corr},
                         "inferred: session lock upgraded after birth with no lock ingest row; "
                         "pre-ingest_log locks also match")


def _telemetry_pre_trim(ctx: cc.Ctx) -> dict[str, list[tuple[str, int]]]:
    """timestamp -> [(query, hits before the trim)] from UPS telemetry rows."""
    tel: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for p in cc.jsonl_paths(ctx.sidecar(*UPS_TELEMETRY)):
        for r in cc.iter_jsonl(p):
            ts, q = r.get("timestamp"), r.get("query")
            nr, nu = r.get("n_returned"), r.get("n_unique_content_hashes")
            if isinstance(ts, str) and isinstance(q, str) and type(nr) is int:
                pre = min(nr, nu) if type(nu) is int else nr
                tel[ts].append((q, pre))
    return tel


def _ts_minus_one(ts: str) -> str:
    try:
        return (datetime.strptime(ts, _TS_FMT) - timedelta(seconds=1)).strftime(_TS_FMT)
    except ValueError:
        return ""


def _match_pre_trim(tel: dict[str, list[tuple[str, int]]], ts: str, prefix: str) -> int | None:
    for key in (ts, _ts_minus_one(ts)):
        for q, pre in tel.get(key, ()):
            if q.startswith(prefix):
                return pre
    return None


def a082(ctx: cc.Ctx) -> cc.Result:
    """Assembled-block token ceiling (6000).

    UPS telemetry records hits before the trim and the UPS audit row records
    n_beliefs after it; nothing between the two writes drops a hit except
    the trim. A joined fire with fewer emitted than returned is a trim. The
    6000 constant can bind only before 5.1.0 or on a block over the payload
    bound in tokens. The join keys on timestamp (or one second earlier) and
    on the audit prompt prefix being a prefix of the telemetry query; no text
    leaves the probe.
    """
    tel = _telemetry_pre_trim(ctx)
    d = {"ups_rows": 0, "joined": 0, "trim_fires": 0,
         "trim_fires_era_6000_only": 0, "trim_fires_over_payload_tokens": 0,
         "trim_fires_post_1639": 0, "rows_5801_6000_since_1551": 0,
         "rows_over_6000_since_1551": 0, "tel_rows": sum(len(v) for v in tel.values())}
    for r in _audit_rows(ctx, (HOOK_UPS, HOOK_SS)):
        ts, tok = r.get("ts"), r.get("tokens")
        if not isinstance(ts, str) or type(tok) is not int:
            continue
        day = ts[:10]
        if day >= CEILING_ERA_START:
            if 5800 < tok <= HOOK_BLOCK_TOKEN_CEILING:
                d["rows_5801_6000_since_1551"] += 1
            elif tok > HOOK_BLOCK_TOKEN_CEILING:
                d["rows_over_6000_since_1551"] += 1
        if r.get("hook") != HOOK_UPS or r.get("prompt_shape_gate_skip"):
            continue
        d["ups_rows"] += 1
        pp, nb = r.get("prompt_prefix"), r.get("n_beliefs")
        if not isinstance(pp, str) or type(nb) is not int:
            continue
        pre = _match_pre_trim(tel, ts, pp)
        if pre is None:
            continue
        d["joined"] += 1
        if nb < pre:
            d["trim_fires"] += 1
            d["trim_fires_era_6000_only"] += int(CEILING_ERA_START <= day < PAYLOAD_CAP_DAY)
            d["trim_fires_over_payload_tokens"] += int(tok > PAYLOAD_TOKENS)
            d["trim_fires_post_1639"] += int(day >= PAYLOAD_CAP_DAY)
    if d["ups_rows"] == 0:
        return cc.silent(0, d, "no ups audit rows")
    ceiling = d["trim_fires_era_6000_only"] + d["trim_fires_over_payload_tokens"]
    return cc.from_count(ceiling, d, "count is trims the 6000 ceiling could have bound")


def a083(ctx: cc.Ctx) -> cc.Result:
    rows = sum(1 for b in _blocks(ctx) if CATEGORY_FOCUS_OPEN in b)
    detail: dict[str, int] = {}
    if ctx.table_exists("categories"):
        detail["categories"] = ctx.count("SELECT COUNT(*) FROM categories")
    if ctx.table_exists("belief_categories"):
        detail["memberships"] = ctx.count("SELECT COUNT(*) FROM belief_categories")
    return cc.from_count(rows, detail)


def a084(ctx: cc.Ctx) -> cc.Result:
    rows = list(_audit_rows(ctx, (HOOK_WARM,)))
    detail: dict[str, int] = {}
    for r in rows:
        o = r.get("sidecar_outcome")
        k = o if o in WARM_LABELS else ("null" if o is None else "other")
        detail[k] = detail.get(k, 0) + 1
    return cc.from_count(len(rows), detail)


def a085(ctx: cc.Ctx) -> cc.Result:
    return _cadence_fire(ctx, POLICY_P1)


_WATERMARK_RE = re.compile(r"watermark=(\d+)")


def a086(ctx: cc.Ctx) -> cc.Result:
    rows = total = 0
    for r in _shadow_rows(ctx):
        m = _WATERMARK_RE.search(_shadow_reason(r, POLICY_P2))
        if m is None:
            continue
        total += 1
        if int(m.group(1)) != DEFAULT_P2_WATERMARK:
            rows += 1
    return cc.from_count(rows, {"rows_with_watermark": total},
                         "shadow rows only; non-default P2 watermark, env and TOML not told apart")


def a087(ctx: cc.Ctx) -> cc.Result:
    return _cadence_fire(ctx, POLICY_P2)


def a088(ctx: cc.Ctx) -> cc.Result:
    return _cadence_fire(ctx, POLICY_P3S)


def a089(ctx: cc.Ctx) -> cc.Result:
    return _cadence_fire(ctx, POLICY_P3V)


_VEL_THR_RE = re.compile(r"threshold=(\d+)$")
_SUB_THR_RE = re.compile(r"threshold=([0-9.]+)")
_SUB_WIN_RE = re.compile(r"\((\d+)/(\d+)\)$|p3_substantive_window=(-?\d+)")


def a090(ctx: cc.Ctx) -> cc.Result:
    vel = sub = 0
    for r in _shadow_rows(ctx):
        mv = _VEL_THR_RE.search(_shadow_reason(r, POLICY_P3V))
        if mv is not None and int(mv.group(1)) != DEFAULT_P3_VELOCITY_THRESHOLD:
            vel += 1
        rs = _shadow_reason(r, POLICY_P3S)
        ms = _SUB_THR_RE.search(rs)
        mw = _SUB_WIN_RE.search(rs)
        win = None
        if mw is not None:
            win = int(mw.group(2) or mw.group(3))
        if (ms is not None and ms.group(1) != DEFAULT_P3_SUBSTANTIVE_THRESHOLD) or (
                win is not None and win != DEFAULT_P3_SUBSTANTIVE_WINDOW):
            sub += 1
    return cc.from_count(vel + sub, {"velocity_rows": vel, "substantive_rows": sub},
                         "shadow rows only; non-default P3 knobs, env and TOML not told apart; "
                         "substantive window visible only on fired rows")


def _resume_cache(ctx: cc.Ctx) -> tuple[int, dict[str, int], datetime | None]:
    p = ctx.sidecar(CADENCE_RESUME_CACHE)
    if not p.is_file():
        return 0, {"cache_file": 0, "cache_malformed": 0}, None
    detail = {"cache_file": 1, "cache_malformed": 0}
    obj = _read_json(p)
    if not isinstance(obj, dict):
        detail["cache_malformed"] = 1
        return 1, detail, None
    pol: Any = obj.get("policy")  # pyright: ignore[reportUnknownMemberType]
    detail["cache_policy_" + (pol if pol in POLICIES else "other")] = 1
    return 1, detail, _parse_ts(obj.get("ts"))  # pyright: ignore[reportUnknownMemberType]


def a091(ctx: cc.Ctx) -> cc.Result:
    """Cadence resume cache plus the <cadence-resume> recap on a later first prompt.

    The cache file's only writer runs after a live cadence fire, so its
    presence proves the Stop side fired at least once. Recap rows are a lower
    bound: the reader needs a new session within the TTL, and the payload
    ceiling can shed the recap whole.
    """
    cache, detail, cache_ts = _resume_cache(ctx)
    recap = ups = 0
    first: dict[str, datetime] = {}
    for r in _audit_rows(ctx, (HOOK_UPS,)):
        ups += 1
        if (RESUME_OPEN + " ") in _block(r):
            recap += 1
        t, sid = _parse_ts(r.get("ts")), r.get("session_id")
        if t is not None and isinstance(sid, str) and (sid not in first or t < first[sid]):
            first[sid] = t
    in_ttl = 0
    if cache_ts is not None:
        in_ttl = sum(1 for t in first.values()
                     if 0 < (t - cache_ts).total_seconds() <= CADENCE_RESUME_TTL_S)
    detail.update({"ups_rows": ups, "recap_rows": recap,
                   "sessions_first_prompt_in_ttl_of_last_write": in_ttl})
    return cc.from_count(cache + recap, detail,
                         "count = cache file + recap rows; recap is a lower bound")


def a092(ctx: cc.Ctx) -> cc.Result:
    n = 0
    fired = 0
    for r in _shadow_rows(ctx):
        n += 1
        fired += 1 if r.get("fired") is True else 0
    return cc.from_count(n, {"fired_rows": fired})


def a093(ctx: cc.Ctx) -> cc.Result:
    rows = dropped = 0
    for r in _ups_rebuild_rows(ctx):
        k = sum(1 for c in _cands(r) if _str(c.get("reason")).startswith(
            "content_hash_collision_with:"))
        if k:
            rows += 1
            dropped += k
    return cc.from_count(rows, {"dropped_duplicates": dropped},
                         "UPS rebuild rows; exploration displacements carry a bare reason and are excluded")


def composed_end(s: str, reps: int) -> int | None:
    """End offset of [P]*reps joined by spaces at the start of s, else None.

    The conversation-aware query is `" ".join([prompt] * weight + turns)`.
    The scan starts at offset 1 so a prompt that opens with a space is seen.
    """
    n = len(s)
    k = s.find(" ", 1)
    while 0 < k and reps * k + (reps - 1) <= n:
        head = s[:k]
        ok = all(s[i * (k + 1) - 1] == " " and s[i * (k + 1):i * (k + 1) + k] == head
                 for i in range(1, reps))
        end = reps * k + (reps - 1)
        if ok and (end == n or s[end] == " "):
            return end
        k = s.find(" ", k + 1)
    return None


def _composed_weight(sq: str) -> tuple[int, int] | None:
    for w in [3] + [x for x in range(2, MAX_CONV_AWARE_WEIGHT + 1) if x != 3]:
        e = composed_end(sq, w)
        if e is not None:
            return w, e
    return None


def a095(ctx: cc.Ctx) -> cc.Result:
    """Conversation-aware UPS query, read from UPS rebuild rows' scored_query.

    The field is forward-only (4.3.0), so a store whose UPS rows all lack it
    is untraceable rather than silent. Prompt weights 2 to 8 are detected.
    """
    d = {"ups_rows": 0, "ups_rows_no_scored_query": 0, "ups_rows_scored": 0,
         "composed_w3": 0, "composed_other_weight": 0, "composed_with_turns": 0,
         "prompt_only": 0, "prompt_only_first_row_of_session": 0}
    logdir = ctx.sidecar(REBUILD_LOG_DIRNAME)
    for p in sorted(logdir.glob("*.jsonl")) if logdir.is_dir() else []:
        first = True
        for r in cc.iter_jsonl(p):
            if r.get("truncated") is True or not isinstance(r.get("input"), dict):
                continue
            if not _is_ups_rebuild(r):
                continue
            d["ups_rows"] += 1
            sq = r["input"].get("scored_query")
            if not isinstance(sq, str):
                d["ups_rows_no_scored_query"] += 1
                first = False
                continue
            d["ups_rows_scored"] += 1
            hit = _composed_weight(sq)
            if hit is None:
                d["prompt_only"] += 1
                d["prompt_only_first_row_of_session"] += int(first)
            else:
                d["composed_w3" if hit[0] == 3 else "composed_other_weight"] += 1
                d["composed_with_turns"] += int(hit[1] < len(sq))
            first = False
    if d["ups_rows_scored"] == 0:
        return cc.Result("untraceable", None, d,
                         "no UPS row carries scored_query; field is forward-only from 4.3.0")
    return cc.from_count(d["composed_w3"] + d["composed_other_weight"], d,
                         "composed query detected at prompt weight 2 to 8")


def a096(ctx: cc.Ctx) -> cc.Result:
    rows = sum(1 for b in _blocks(ctx) if COVERAGE_RE.search(b))
    return cc.from_count(rows)


def a097(ctx: cc.Ctx) -> cc.Result:
    if not _need(ctx, "exploration_events"):
        return _no_table()
    n = ctx.count("SELECT COUNT(*) FROM exploration_events")
    detail: dict[str, int] = {}
    if ctx.table_exists("schema_meta"):
        detail["fire_idx"] = ctx.count(
            "SELECT CAST(value AS INTEGER) FROM schema_meta WHERE key='exploration_fire_idx'")
    return cc.from_count(n, detail)


def a098(ctx: cc.Ctx) -> cc.Result:
    if not _need(ctx, "feedback_history"):
        return _no_table()
    n, d = ctx.counts("SELECT COUNT(*), COUNT(DISTINCT belief_id) FROM feedback_history"
                      " WHERE source=?", (FEEDBACK_SOURCE_HOOK,))
    return cc.from_count(n, {"distinct_beliefs": d},
                         "source hook is shared by UPS and agent-context exposures")


def a099(ctx: cc.Ctx) -> cc.Result:
    rows = locks = 0
    for b in _blocks(ctx):
        ss = _section(b, SESSION_START_OPEN, SESSION_START_CLOSE)
        if ss is None:
            continue
        lk = _section(ss, "<locked>", "</locked>") or ""
        k = lk.count('<belief id="')
        if k:
            rows += 1
            locks += k
    return cc.from_count(rows, {"locked_elements": locks})


def a100(ctx: cc.Ctx) -> cc.Result:
    if not _need(ctx, "belief_touches"):
        return _no_table()
    n, s, mx = ctx.counts("SELECT COUNT(*), COUNT(DISTINCT session_id), MAX(last_fire_idx)"
                          " FROM belief_touches")
    return cc.from_count(n, {"sessions": s, "max_fire_idx": mx})


def _budget_tokens(block: str) -> int | None:
    """Packed cost of the lock="none" retrieval lines outside <session-start>."""
    close = block.rfind(CLOSE_TAG)
    env = block[block.find(OPEN_TAG): close if close >= 0 else len(block)]
    ss = env.find(SESSION_START_OPEN)
    if ss >= 0:
        se = env.find(SESSION_START_CLOSE, ss)
        env = env[:ss] + (env[se + len(SESSION_START_CLOSE):] if se >= 0 else "")
    elems = NONLOCK_ELEM_RE.findall(env)
    if not elems:
        return None
    return sum(math.ceil((len(e) + 1) / 4) for e in elems)


def _coverage_split(block: str) -> tuple[int, int]:
    """Coverage-line matches after the envelope close, and inside it."""
    close = block.rfind(CLOSE_TAG)
    after = inside = 0
    for m in COVERAGE_RE.finditer(block):
        if close >= 0 and m.start() > close:
            after += 1
        else:
            inside += 1
    return after, inside


def a101(ctx: cc.Ctx) -> cc.Result:
    """UPS injection token budget (1500), applied on every UPS retrieval.

    Fired counts UPS blocks that packed a retrieval line; the blocks whose
    packed cost exceeds the budget are in detail, split at 5.0.0 (token unit
    change). Untraceable when the store has no UPS audit rows.
    """
    d = {k: 0 for k in ("ups_rows", "ups_blocks", "cov_after_close", "cov_inside",
                        "budget_rows", "over_budget_pre500", "over_budget_post500",
                        "rows_post500", "max_nonlock_tokens")}
    for r in _audit_rows(ctx, (HOOK_UPS,)):
        d["ups_rows"] += 1
        post = _str(r.get("ts"))[:10] >= REL_5_0_0
        d["rows_post500"] += int(post)
        b = _block(r)
        if OPEN_TAG not in b:
            continue
        d["ups_blocks"] += 1
        after, inside = _coverage_split(b)
        d["cov_after_close"] += after
        d["cov_inside"] += inside
        tok = _budget_tokens(b)
        if tok is None:
            continue
        d["budget_rows"] += 1
        d["max_nonlock_tokens"] = max(d["max_nonlock_tokens"], tok)
        if tok > HOOK_TOKEN_BUDGET:
            d["over_budget_post500" if post else "over_budget_pre500"] += 1
    if d["ups_rows"] == 0:
        return cc.Result("untraceable", None, d, "no UPS audit rows in this store")
    if d["budget_rows"] > 0:
        return cc.fired(d["budget_rows"], d, "budget applied on every packed UPS block")
    return cc.silent(0, d, "UPS rows but no packed retrieval line")


def a102(ctx: cc.Ctx) -> cc.Result:
    """injection_events writer, with the tri-state `referenced` column split.

    NULL is unscored, 0 scored and not referenced, 1 referenced. A store
    whose UPS fires carried a block but which has no rows is an error
    (writer missed).
    """
    if ctx.db is None:
        return cc.silent(None, note="no db")
    block_fires = sum(1 for r in _audit_rows(ctx, (HOOK_UPS,)) if _block(r))
    if not ctx.table_exists("injection_events"):
        if block_fires:
            return cc.Result("error", 0, {"ups_block_fires": block_fires}, "table absent")
        return cc.silent(0, {"ups_block_fires": 0}, "table absent")
    n, n_post, pos, neg, unscored, scored_post, cons = ctx.counts(
        "SELECT COUNT(*),"
        " SUM(substr(injected_at, 1, 10) >= ?),"
        " SUM(referenced = 1), SUM(referenced = 0), SUM(referenced IS NULL),"
        " SUM(substr(referenced_at, 1, 10) >= ?),"
        " SUM(active_consumers <> '[]') FROM injection_events",
        (AFTER_5_1_0, AFTER_5_1_0))
    detail = {
        "rows_post_510": n_post, "ref_pos": pos, "ref_neg": neg,
        "ref_unscored": unscored, "scored_post_510": scored_post,
        "with_consumers": cons, "ups_block_fires": block_fires,
    }
    if n == 0 and block_fires:
        return cc.Result("error", 0, detail, "writer missed")
    return cc.from_count(n, detail)


def a103(ctx: cc.Ctx) -> cc.Result:
    rows = elems = 0
    for b in _blocks(ctx):
        k = b.count(TRUNCATION_MARKER + "</belief>")
        if k:
            rows += 1
            elems += k
    return cc.from_count(rows, {"truncated_elements": elems})


def _compaction_markers(ctx: cc.Ctx) -> dict[str, list[tuple[int, str]]]:
    tdir = ctx.sidecar(TRANSCRIPTS_DIRNAME)
    files = [tdir / TURNS_FILENAME] if (tdir / TURNS_FILENAME).is_file() else []
    if (tdir / "archive").is_dir():
        files += sorted((tdir / "archive").glob("*.jsonl"))
    out: dict[str, list[tuple[int, str]]] = {"compaction_start": [], "compaction_complete": []}
    for p in files:
        for r in cc.iter_jsonl(p):
            ev = r.get("event")
            if ev in out:
                ts = _str(r.get("ts"))
                t = _ts_key(ts)
                if t is not None:
                    out[ev].append((t, ts[:10]))
    return out


def _compact_audit_times(ctx: cc.Ctx) -> dict[str, list[int]]:
    compact_ss: dict[str, list[int]] = {}
    for r in _audit_rows(ctx, (HOOK_SS,)):
        if r.get("source") == "compact":
            sid, t = r.get("session_id"), _ts_key(r.get("ts"))
            if isinstance(sid, str) and t is not None:
                compact_ss.setdefault(sid, []).append(t)
    return compact_ss


def a105(ctx: cc.Ctx) -> cc.Result:
    """Post-compaction rebuild block (SessionStart source=compact).

    A fire is a rebuilder row within 30 s of a transcript-logger
    compaction_complete marker, excluding rows next to a compaction_start
    marker (the older PreCompact path), unioned with rows within 30 s after
    a compact SessionStart audit row. That audit row exists only when locks
    rendered, so it alone hides lock-less fires.
    """
    mk = _compaction_markers(ctx)
    starts = [t for t, _ in mk["compaction_start"]]
    rows: list[tuple[int, Any]] = []
    for r in _rebuilder_rows(ctx):
        t = _ts_key(r.get("ts"))
        if t is not None:
            rows.append((t, r.get("session_id")))
    pre_hook = {i for i, (t, _) in enumerate(rows)
                if any(-5 <= t - s <= COMPACT_WINDOW_S for s in starts)}
    fired_rows: set[int] = set()
    completes_post = misses_post = 0
    for c, day in mk["compaction_complete"]:
        hit = [i for i, (t, _) in enumerate(rows)
               if i not in pre_hook and i not in fired_rows and abs(t - c) <= COMPACT_WINDOW_S]
        if hit:
            fired_rows.add(hit[0])
        if day >= SHIP_1031:
            completes_post += 1
            misses_post += 0 if hit else 1
    marker_fires = len(fired_rows)
    compact_ss = _compact_audit_times(ctx)
    audit_only = 0
    for i, (t, sid) in enumerate(rows):
        if i in fired_rows or not isinstance(sid, str):
            continue
        if any(0 <= t - c <= COMPACT_WINDOW_S for c in compact_ss.get(sid, [])):
            fired_rows.add(i)
            audit_only += 1
    detail = {
        "rebuilder_rows": len(rows),
        "compaction_starts": len(starts),
        "compaction_completes": len(mk["compaction_complete"]),
        "completes_post_1031": completes_post,
        "misses_post_1031": misses_post,
        "precompact_path_rows": len(pre_hook),
        "marker_matched_fires": marker_fires,
        "audit_only_fires": audit_only,
        "compact_ss_audit_rows": sum(len(v) for v in compact_ss.values()),
    }
    note = "rebuild row within 30 s of a compaction_complete marker or compact SessionStart audit row"
    if not fired_rows and completes_post == 0 and not compact_ss:
        note = "no post-1031 compaction observed"
    return cc.from_count(len(fired_rows), detail, note)


def a106(ctx: cc.Ctx) -> cc.Result:
    path = ctx.sidecar(*SEARCH_TELEMETRY)
    tally = cc.jsonl_tally(path, lambda r: r.get("command") if isinstance(r.get("command"), str)
                           else "null", BASH_COMMANDS)
    n = sum(tally.values())
    empty = cc.jsonl_count(path, lambda r: _int(r.get("injected_l1")) + _int(r.get("injected_l0")) == 0)
    return cc.from_count(n, {**tally, "empty_hit_rows": empty})


def _ring(ctx: cc.Ctx) -> dict[str, Any] | None:
    obj = _read_json(ctx.sidecar(RING_FILENAME))
    return dict(obj) if isinstance(obj, dict) else None  # pyright: ignore[reportUnknownArgumentType]


def _ring_session_audit(ctx: cc.Ctx, sid: str) -> tuple[int, set[str], str | None, str | None]:
    """UPS rows for sid, ids they list, sid's first ts, and the first ts of other sessions."""
    ups_ids: set[str] = set()
    ups_rows = 0
    s_min: str | None = None
    other_min: str | None = None
    for r in _audit_rows(ctx, (HOOK_UPS, HOOK_SS)):
        ts = _str(r.get("ts")) or None
        if r.get("session_id") == sid:
            if ts and (s_min is None or ts < s_min):
                s_min = ts
            if r.get("hook") == HOOK_UPS:
                ups_rows += 1
                ups_ids.update(_row_ids(r))
        elif ts and (other_min is None or ts < other_min):
            other_min = ts
    return ups_rows, ups_ids, s_min, other_min


def _bash_rows_for(ctx: cc.Ctx, sid: str) -> tuple[int, int]:
    rows = inj = 0
    for r in cc.iter_jsonl(ctx.sidecar(*SEARCH_TELEMETRY)):
        if r.get("session_id") != sid:
            continue
        rows += 1
        inj += int(_int(r.get("injected_l1")) + _int(r.get("injected_l0")) > 0)
    return rows, inj


def a107(ctx: cc.Ctx) -> cc.Result:
    """PreToolUse search-tool hook (Grep, Glob, WebSearch, WebFetch).

    The session ring has two writers: UPS (ids its audit row also lists) and
    the search-tool hook. A non-locked ring id for the ring's session that no
    UPS audit row lists came from the search hook; with no Bash-lane
    telemetry injection for that session, it came from the Grep, Glob or Web
    lane. A fire whose hits were already in the ring leaves no id, so no
    orphan id is untraceable, never silent.
    """
    ring = _ring(ctx)
    if ring is None:
        return cc.untraceable("no ring file")
    sid: Any = ring.get("session_id")
    entries: Any = ring.get("ring")
    if not isinstance(sid, str) or not sid or not isinstance(entries, list):
        return cc.untraceable("ring has no session")
    ents: list[dict[str, Any]] = [
        dict(e) for e in entries  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if isinstance(e, dict) and isinstance(e.get("id"), str)]  # pyright: ignore[reportUnknownMemberType]
    unlocked = [e for e in ents if not e.get("locked")]
    ups_rows, ups_ids, s_min, other_min = _ring_session_audit(ctx, sid)
    rotated = int(ctx.sidecar(AUDIT_FILENAME + ".1").is_file())
    history_ok = int(not rotated or (s_min is not None and other_min is not None
                                     and other_min < s_min))
    bash_rows, bash_inj = _bash_rows_for(ctx, sid)
    orphans = [e for e in unlocked if e["id"] not in ups_ids]
    orphan_fires = len({e.get("fire_idx") for e in orphans if isinstance(e.get("fire_idx"), int)})
    detail = {
        "ring_ids": len(ents), "ring_unlocked": len(unlocked),
        "orphan_ids": len(orphans), "orphan_fires": orphan_fires,
        "ups_rows": ups_rows, "bash_tel_rows": bash_rows, "bash_tel_inj": bash_inj,
        "audit_rotated": rotated, "history_ok": history_ok,
    }
    if ups_rows == 0:
        return cc.Result("untraceable", None, detail, "no UPS audit rows for ring session")
    if not orphans:
        return cc.Result("untraceable", None, detail, "no orphan ring ids")
    if bash_inj > 0:
        return cc.ambiguous(max(orphan_fires, 1), detail,
                            "the Bash lane also injected for this session")
    if not history_ok:
        return cc.ambiguous(max(orphan_fires, 1), detail,
                            "UPS rows for this session rotated out of the audit log")
    return cc.fired(max(orphan_fires, 1), detail, "orphan ring ids lower bound")


def _gate_label(r: dict[str, Any]) -> str | None:
    v = r.get("prompt_shape_gate_skip")
    if not isinstance(v, str):
        return None
    for lab in GATE_LABELS:
        if v == lab or v.startswith(lab + ":") or v.startswith(lab):
            return lab
    return "other"


def a109(ctx: cc.Ctx) -> cc.Result:
    detail: dict[str, int] = {}
    for r in _audit_rows(ctx, (HOOK_UPS,)):
        lab = _gate_label(r)
        if lab is not None:
            detail[lab] = detail.get(lab, 0) + 1
    return cc.from_count(sum(detail.values()), detail)


def a110(ctx: cc.Ctx) -> cc.Result:
    detail: dict[str, int] = {}
    rows = 0
    for b in _blocks(ctx, (HOOK_UPS, HOOK_SS)):
        hit = False
        for s in PROVENANCE_SECTIONS:
            if f"<{s}><!-- " in b:
                detail[s] = detail.get(s, 0) + 1
                hit = True
        rows += 1 if hit else 0
    return cc.from_count(rows, detail)


def a111(ctx: cc.Ctx) -> cc.Result:
    rows = sess = l1 = 0
    for r in _rebuilder_rows(ctx):
        k_s = k_l = 0
        for c in _cands(r):
            reason = _str(c.get("reason"))
            k_s += reason.startswith("below_floor_session:")
            k_l += reason.startswith("below_floor_l1:")
        if k_s or k_l:
            rows += 1
        sess += k_s
        l1 += k_l
    return cc.from_count(rows, {"session_floored": sess, "l1_floored": l1})


def a112(ctx: cc.Ctx) -> cc.Result:
    rows = dropped = at_window = total = 0
    for r in _rebuilder_rows(ctx):
        total += 1
        k = _pack(r, "n_dropped_by_budget")
        if k:
            rows += 1
            dropped += k
        if _int(r["input"].get("n_recent_turns")) >= REBUILDER_TURN_WINDOW:
            at_window += 1
    return cc.from_count(rows, {"rebuilder_rows": total, "dropped_by_budget": dropped,
                                "rows_at_turn_window": at_window})


_WORD = re.compile(r"\w+", re.UNICODE)


def _stack_shaped(sq: str, eq: str) -> bool:
    """All-lowercase terms drawn from the extracted query's words (transform_query output)."""
    if sq != sq.lower():
        return False
    allowed = {m.group(0).lower() for m in _WORD.finditer(eq)}
    return all(t in allowed for t in sq.split())


def _a113_bucket(day: str | None) -> str:
    if day is None:
        return "diff_no_ts"
    if day < REL_5_0_0:
        return "diff_pre_500"
    if day == REL_5_0_0:
        return "diff_release_day"
    return "diff_post_500"


def a113(ctx: cc.Ctx) -> cc.Result:
    """Rebuilder query_strategy stack-r1-r3, opt-in since 5.0.0.

    The stack was the default from 3.0 to 5.0.0, so a scored query that
    differs from the extracted query counts as an opt-in fire only when it
    is dated after the 5.0.0 release day.
    """
    d = {"rows": 0, "diff": 0, "diff_pre_500": 0, "diff_post_500": 0,
         "diff_no_ts": 0, "diff_release_day": 0, "diff_stack_shaped": 0, "diff_other_shape": 0,
         "rows_pre_430": 0, "rows_post_500": 0}
    for r in _rebuilder_rows(ctx):
        sq, eq = r["input"].get("scored_query"), r["input"].get("extracted_query")
        if not isinstance(sq, str) or not isinstance(eq, str):
            continue
        d["rows"] += 1
        ts = _str(r.get("ts"))
        day = ts[:10] if len(ts) >= 10 else None
        d["rows_pre_430"] += int(day is not None and day < REL_4_3_0)
        d["rows_post_500"] += int(day is not None and day >= REL_5_0_0)
        if sq == eq:
            continue
        d["diff"] += 1
        d[_a113_bucket(day)] += 1
        d["diff_stack_shaped" if _stack_shaped(sq, eq) else "diff_other_shape"] += 1
    note = "fired = diff rows dated after the 5.0.0 release day, when stack-r1-r3 is opt-in"
    if d["diff_post_500"] > 0:
        return cc.fired(d["diff_post_500"], d, note)
    return cc.silent(0, d, note)


def a117(ctx: cc.Ctx) -> cc.Result:
    obj = _read_json(ctx.sidecar(EXCLUSIONS_FILENAME))
    if not isinstance(obj, dict):
        return cc.silent(0)
    pats: Any = obj.get("patterns")  # pyright: ignore[reportUnknownMemberType]
    n = len([p for p in pats if isinstance(p, str) and p]) if isinstance(pats, list) else 0  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    st = _read_json(ctx.sidecar(SESSION_STATE_FILENAME))
    cur: Any = st.get("session_id") if isinstance(st, dict) else None  # pyright: ignore[reportUnknownMemberType]
    match = 1 if isinstance(cur, str) and cur == obj.get("session_id") else 0  # pyright: ignore[reportUnknownMemberType]
    return cc.from_count(n, {"bound_to_current_session": match})


def a118(ctx: cc.Ctx) -> cc.Result:
    obj = _read_json(ctx.sidecar(SESSION_STATE_FILENAME))
    if not isinstance(obj, dict):
        return cc.silent(0)
    ids: Any = obj.get("session_ids")  # pyright: ignore[reportUnknownMemberType]
    n = len([s for s in ids if isinstance(s, str) and s]) if isinstance(ids, list) else 0  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    note = "fired when the window holds 2 or more session ids"
    if n >= 2:
        return cc.fired(n, {"window_ids": n}, note)
    return cc.silent(0, {"window_ids": n}, note)


def a119(ctx: cc.Ctx) -> cc.Result:
    detail: dict[str, int] = {}
    n = 0
    for r in _audit_rows(ctx, (HOOK_SS,)):
        if BASELINE_OPEN_TAG not in _block(r):
            continue
        n += 1
        src = r.get("source")
        lab = src if src in ("startup", "resume", "clear", "compact") else (
            "none" if src is None else "other")
        detail[lab] = detail.get(lab, 0) + 1
    return cc.from_count(n, detail)


_RECAP_EVENTS = ("belief.locked", "belief.ingested", "wonder.promoted", "feedback.applied")


def _recap_gaps(ss: list[str], feed: list[str]) -> dict[str, int]:
    first = sum(1 for t in feed if t <= ss[0])
    hits = events = max_gap = 0
    for prev, cur in zip(ss, ss[1:]):
        n = sum(1 for t in feed if prev < t <= cur)
        events += n
        max_gap = max(max_gap, n)
        hits += int(n >= RECAP_THRESHOLD)
    return {
        "first_gap_events": first, "first_gap_ge3": int(first >= RECAP_THRESHOLD),
        "inner_gaps": len(ss) - 1, "inner_gap_events": events,
        "inner_gaps_ge3": hits, "max_inner_gap": max_gap,
        "events_after_last_ss": sum(1 for t in feed if t > ss[-1]),
    }


def a120(ctx: cc.Ctx) -> cc.Result:
    """SessionStart belief-write recap line: untraceable.

    The line goes to stdout only. `sessionstart_last.txt` is rewritten on
    every SessionStart while the recap is enabled, printed or not, and the
    recap can be off (AELFRICE_SESSIONSTART_RECAP=0) while feed rows still
    accrue, so no durable row shows that a line printed. The detail replays
    the decision as an upper bound: feed rows of the recap events between
    audited SessionStart fires, and the gaps that reached the threshold.
    """
    ss = sorted(_str(r.get("ts")) for r in _audit_rows(ctx, (HOOK_SS,))
                if isinstance(r.get("ts"), str))
    feed: list[str] = []
    for p in cc.jsonl_paths(ctx.sidecar(FEED_FILENAME)):
        for r in cc.iter_jsonl(p):
            if r.get("event") in _RECAP_EVENTS and isinstance(r.get("ts"), str):
                feed.append(r["ts"])
    feed.sort()
    detail = {"ss_audit_rows": len(ss), "recap_events": len(feed),
              "last_ts_file": int(ctx.sidecar(RECAP_LAST_TS_FILENAME).is_file())}
    if ss:
        detail.update(_recap_gaps(ss, feed))
    else:
        detail["events_no_ss"] = len(feed)
    return cc.Result("untraceable", None, detail,
                     "prints only; detail replays an upper bound from audit and feed rows")


def _phantom_ids(ctx: cc.Ctx, detail: dict[str, int]) -> set[str]:
    rows = ctx.require_db().execute(
        "SELECT id FROM beliefs WHERE origin='speculative'").fetchall()
    ids = {str(r[0]) for r in rows}
    detail["phantoms_all"] = len(ids)
    if ctx.column_exists("beliefs", "valid_to"):
        detail["live_phantoms"] = ctx.count(
            "SELECT COUNT(*) FROM beliefs WHERE origin='speculative' AND valid_to IS NULL")
    return ids


def a121(ctx: cc.Ctx) -> cc.Result:
    """Speculative marker and framing sentence.

    A pre-v1 schema with no beliefs.origin column has no phantoms and the
    hook never reads it. A UPS or SessionStart row that injected a phantom
    id must carry the framing sentence once the feature shipped; a row
    that does not is an error.
    """
    if ctx.db is None:
        return cc.silent(None, note="no db")
    has_origin = ctx.column_exists("beliefs", "origin")
    detail: dict[str, int] = {}
    phantom_ids = _phantom_ids(ctx, detail) if has_origin else set()
    marker = elems = prow = prow_nos = prow_pre = 0
    for r in _audit_rows(ctx, (HOOK_UPS, HOOK_SS)):
        b = _block(r)
        has_sentence = SPECULATIVE_SENTENCE_HEAD in b
        if has_sentence:
            marker += 1
            elems += b.count(' speculative="1">') + b.count(' origin="speculative"')
        if phantom_ids and any(i in phantom_ids for i in _row_ids(r)):
            prow += 1
            if not has_sentence:
                late = _str(r.get("ts"))[:10] >= SPEC_FRAMING_SHIP
                prow_nos += int(late)
                prow_pre += int(not late)
    detail.update({"spec_elements": elems, "phantom_rows": prow,
                   "phantom_rows_no_sentence": prow_nos, "phantom_rows_pre_ship": prow_pre})
    if not has_origin:
        detail["beliefs_rows"] = (ctx.count("SELECT COUNT(*) FROM beliefs")
                                  if ctx.table_exists("beliefs") else 0)
        return cc.Result("fired" if marker else "silent", marker, detail,
                         "pre-v1 schema, no origin column; hook never reads this store")
    if prow_nos:
        return cc.Result("error", marker, detail, "phantom injected without sentence")
    return cc.from_count(marker, detail)


def _ts19(v: Any) -> str:
    return v[:19] if isinstance(v, str) else ""


# Per-session candidate record: (created, valid_to, user-locked, locked_at, correction-class).
_Cand = tuple[str, str, bool, str, bool]


def _stop_candidates(ctx: cc.Ctx) -> dict[str, list[_Cand]]:
    """Lock candidates by session, as hook.stop's _collect_lock_candidates selects them."""
    from aelfrice.derivation import decision_text  # noqa: PLC0415
    from aelfrice.directive_detector import detect_directive  # noqa: PLC0415
    from aelfrice.models import (  # noqa: PLC0415
        BELIEF_CORRECTION,
        LOCK_USER,
        ORIGIN_AGENT_INFERRED,
        ORIGIN_AGENT_REMEMBERED,
    )

    agent = {ORIGIN_AGENT_INFERRED, ORIGIN_AGENT_REMEMBERED}
    vt = "valid_to" if ctx.column_exists("beliefs", "valid_to") else "NULL"
    rows = ctx.require_db().execute(
        f"SELECT id, content, type, origin, lock_level, locked_at, created_at, {vt},"
        " session_id FROM beliefs WHERE session_id IS NOT NULL AND session_id <> ''"
    ).fetchall()
    by_sess: dict[str, list[_Cand]] = {}
    for bid, content, typ, origin, lock, locked_at, created, valid_to, sid in rows:
        cls = typ == BELIEF_CORRECTION or origin in agent
        if not cls:
            try:
                if not detect_directive(decision_text(str(bid), str(content or ""))):
                    continue
            except Exception:  # noqa: BLE001
                continue
        by_sess.setdefault(str(sid), []).append(
            (_ts19(created), _ts19(valid_to), lock == LOCK_USER, _ts19(locked_at), cls))
    return by_sess


def _candidates_at(recs: list[_Cand], t: str) -> list[bool]:
    """Correction-class flags of the candidates live at time t."""
    return [cls for (c, v, lu, la, cls) in recs
            if c and c <= t and (not v or v > t) and (not lu or (la and la > t))]


def a123(ctx: cc.Ctx) -> cc.Result:
    """Stop-hook lock-candidate listing, reconstructed from session_end batches.

    The listing goes to stderr only, but the same stop() call may record a
    core_gate_batches row with origin session_end. Such a row proves stop()
    evaluated the candidate list for that session at that time; the
    candidates live then tell whether the listing printed.
    """
    if ctx.db is None or not ctx.table_exists("beliefs"):
        return cc.untraceable("no beliefs table")
    if ctx.count("SELECT COUNT(*) FROM beliefs") == 0:
        return cc.silent(0, {"beliefs": 0}, "empty store")
    for col in ("session_id", "origin", "locked_at"):
        if not ctx.column_exists("beliefs", col):
            return cc.untraceable("pre-582 schema")
    by_sess = _stop_candidates(ctx)
    would_fire = sum(1 for recs in by_sess.values()
                     if any(not lu and not v for (_, v, lu, _, _) in recs))
    detail = {"would_fire_sessions": would_fire, "sessions_any_candidate": len(by_sess)}
    if not ctx.table_exists("core_gate_batches"):
        return cc.untraceable("no core_gate_batches table")
    batches = [(str(s), _ts19(t)) for s, t in ctx.require_db().execute(
        "SELECT session_id, created_at FROM core_gate_batches"
        " WHERE origin = 'session_end' AND session_id IS NOT NULL").fetchall()]
    detail["stop_runs_proven"] = len(batches)
    detail["stop_sessions_proven"] = len({s for s, _ in batches})
    if not batches:
        return cc.Result("untraceable", None, detail, "no session_end batch; listing is stderr only")
    hit_b = hit_dir_b = cand_sum = 0
    hit_sess: set[str] = set()
    for sid, t in batches:
        live = _candidates_at(by_sess.get(sid, []), t)
        cand_sum += len(live)
        if live:
            hit_b += 1
            hit_sess.add(sid)
        hit_dir_b += int(any(not cls for cls in live))
    detail.update({"runs_with_candidates": hit_b,
                   "runs_with_directive_only_candidates": hit_dir_b,
                   "sessions_with_candidates_at_run": len(hit_sess),
                   "candidates_at_runs_sum": cand_sum})
    if hit_b:
        return cc.fired(hit_b, detail,
                        "inferred from session_end batch rows written by the same stop() call")
    return cc.silent(0, detail, "stop() ran but no candidate existed at run time")


def a124(ctx: cc.Ctx) -> cc.Result:
    if not _need(ctx, "feedback_history", "injection_events"):
        return _no_table()
    first = ctx.count("SELECT COUNT(*) FROM injection_events")
    if first == 0:
        return cc.silent(0, {}, "no injection_events rows to tell UPS exposures apart")
    n, rows = ctx.counts(
        "WITH b AS (SELECT created_at, COUNT(*) n FROM feedback_history f WHERE f.source=?"
        "  AND julianday(f.created_at) >= (SELECT MIN(julianday(injected_at)) FROM injection_events)"
        "    - 2.0 / 86400.0"
        "  AND NOT EXISTS (SELECT 1 FROM injection_events e WHERE e.belief_id=f.belief_id"
        "    AND abs(julianday(e.injected_at) - julianday(f.created_at)) * 86400.0 <= 5.0)"
        "  GROUP BY created_at)"
        " SELECT COUNT(*), COALESCE(SUM(n), 0) FROM b", (FEEDBACK_SOURCE_HOOK,))
    return cc.from_count(n, {"exposure_rows": rows},
                         "inferred: hook exposure batches with no injection_events row within 5 s; "
                         "UPS fires without a session id also land here")


def a125(ctx: cc.Ctx) -> cc.Result:
    ledger = 1 if ctx.sidecar(LEDGER_FILENAME).is_file() else 0
    rows = sum(1 for b in _blocks(ctx) if SESSION_START_OPEN not in b and SEEN_LINE_RE.search(b))
    st = "fired" if (rows or ledger) else "silent"
    return cc.Result(st, rows, {"ledger_file": ledger},
                     "seen pointers outside a first-prompt block; the ledger file exists only when on")


def a127(ctx: cc.Ctx) -> cc.Result:
    detail = {"ups": 0, "session_start": 0}
    for r in _audit_rows(ctx, (HOOK_UPS, HOOK_SS)):
        if FRAMING_HEAD in _block(r):
            detail["ups" if r.get("hook") == HOOK_UPS else "session_start"] += 1
    return cc.from_count(sum(detail.values()), detail,
                         "agent-context blocks carry it too but are not logged")


def a128(ctx: cc.Ctx) -> cc.Result:
    path = cc.jsonl_paths(ctx.sidecar(COMMAND_OUTCOMES_FILENAME))
    tally = cc.jsonl_tally(path, lambda r: r.get("reason") if r.get("hook") == COMMAND_OUTCOME_HOOK
                           else None, COMMAND_REASONS)
    return cc.from_count(sum(tally.values()), tally,
                         "lock outcomes only; typed confirm, promote, scope-out are not logged")


def a129(ctx: cc.Ctx) -> cc.Result:
    emitted = suppressed = empty = 0
    for r in _audit_rows(ctx, (HOOK_UPS,)):
        b = _block(r)
        if OPEN_TAG in b:
            emitted += 1
        elif _int(r.get("n_beliefs")) > 0:
            suppressed += 1
        else:
            empty += 1
    return cc.from_count(emitted, {"emitted": emitted, "suppressed_with_hits": suppressed,
                                   "empty": empty})


def a130(ctx: cc.Ctx) -> cc.Result:
    return cc.from_count(cc.jsonl_count(ctx.sidecar(*UPS_TELEMETRY)))


def a131(ctx: cc.Ctx) -> cc.Result:
    rows = over = 0
    for b in _blocks(ctx, (HOOK_UPS, HOOK_SS)):
        if LOCK_OVERFLOW_RE.search(b):
            rows += 1
        if len(b) > HOOK_PAYLOAD_CHAR_LIMIT:
            over += 1
    return cc.from_count(rows, {"rows_over_limit": over},
                         "fires when the bound cut a user lock; belief and frame drops go to stderr only")


def _u(item: str, name: str, note: str) -> cc.Probe:
    return cc.untraceable_probe(item, name, note)


def _p(item: str, name: str, kind: cc.Kind, fn: Callable[[cc.Ctx], cc.Result]) -> cc.Probe:
    return P(item, name, kind, fn)


PROBES: list[cc.Probe] = [
    _p("A077", "core section and selection rule", "sidecar", a077),
    _p("A078", "core token budget 1500", "sidecar", a078),
    _p("A079", "recent-work git sub-block", "sidecar", a079),
    _u("A080", "aelf tail viewer", "read-only viewer of hook_audit; writes nothing"),
    _p("A081", "AELF_AUTOLOCK_CORRECTIONS", "store", a081),
    _p("A082", "block token ceiling 6000", "sidecar", a082),
    _p("A083", "belief categories focus", "sidecar", a083),
    _p("A084", "BM25F sidecar warm child", "sidecar", a084),
    _p("A085", "cadence P1 every K", "sidecar", a085),
    _p("A086", "cadence P2 env overrides", "sidecar", a086),
    _p("A087", "cadence P2 threshold plus boundary", "sidecar", a087),
    _p("A088", "cadence P3 substantive", "sidecar", a088),
    _p("A089", "cadence P3 velocity", "sidecar", a089),
    _p("A090", "cadence P3 env overrides", "sidecar", a090),
    _p("A091", "cadence resume recap", "sidecar", a091),
    _p("A092", "cadence shadow log", "sidecar", a092),
    _p("A093", "collapse_duplicate_hashes", "sidecar", a093),
    _u("A094", "rebuilder dynamic trigger",
       "parked: PreCompact prints a stderr note and returns; nothing is written"),
    _p("A095", "conversation-aware UPS query", "sidecar", a095),
    _p("A096", "coverage line", "sidecar", a096),
    _p("A097", "exploration slot", "store", a097),
    _p("A098", "exposure record source hook", "store", a098),
    _p("A099", "first-prompt locked sub-block", "sidecar", a099),
    _p("A100", "belief_touches writer", "store", a100),
    _p("A101", "injection token budgets", "sidecar", a101),
    _p("A102", "injection_events writer", "store", a102),
    _p("A103", "per-belief render cap 1200", "sidecar", a103),
    _u("A104", "working-state projector",
       "context_rebuilder.main has no production caller and its block goes to stdout only"),
    _p("A105", "post-compaction rebuild block", "sidecar", a105),
    _p("A106", "PreToolUse Bash search hook", "sidecar", a106),
    _p("A107", "PreToolUse search-tool hook", "sidecar", a107),
    _u("A108", "project-context filter",
       "drops hits in memory; no log field or row records the filter or its env value"),
    _p("A109", "prompt-shape gate", "sidecar", a109),
    _p("A110", "provenance-grouped render", "sidecar", a110),
    _p("A111", "rebuild relevance floors", "sidecar", a111),
    _p("A112", "rebuilder budget and turn window", "sidecar", a112),
    _p("A113", "rebuilder query rewrite stack-r1-r3", "sidecar", a113),
    _u("A114", "render-cost estimator",
       "packer per-line costs are not logged; audit tokens are a whole-block estimate"),
    _u("A115", "injection ledger kill experiment", "design doc only; no code writes anything"),
    _u("A116", "session ring dedup for search fires",
       "suppression count goes to hook output only; ring entries do not name the writer"),
    _p("A117", "session scope-out exclusions", "sidecar", a117),
    _p("A118", "session-first-prompt window", "sidecar", a118),
    _p("A119", "SessionStart baseline block", "sidecar", a119),
    _p("A120", "SessionStart recap line", "sidecar", a120),
    _p("A121", "speculative marker and framing", "sidecar", a121),
    _u("A122", "statusline count badges", "statusline reads counts and writes nothing"),
    _p("A123", "Stop lock-candidate listing", "store", a123),
    _p("A124", "agent worker-context injection", "store", a124),
    _p("A125", "turn-differential rendering", "sidecar", a125),
    _u("A126", "two-tier headline rendering", "not built; only a benchmark bound exists"),
    _p("A127", "two-tier trust framing header", "sidecar", a127),
    _p("A128", "typed aelf command executor", "sidecar", a128),
    _p("A129", "UPS memory block and off-switch", "sidecar", a129),
    _p("A130", "UPS telemetry ring", "sidecar", a130),
    _p("A131", "whole-fire payload bound", "sidecar", a131),
]

TRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind != "untraceable")
UNTRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind == "untraceable")
