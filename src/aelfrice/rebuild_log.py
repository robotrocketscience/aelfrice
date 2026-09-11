"""#1527 — the rebuilder config and the phase-1a rebuild_log, off the
retrieval import graph.

`hook.py`'s prompt-shape gate exists so a trivial or system-generated prompt
skips retrieval. It did skip retrieval, and then imported the whole retrieval
subtree anyway: the skip branch calls `_emit_user_prompt_submit_rebuild_log`
unconditionally, and that function reached into `aelfrice.context_rebuilder`
for `_rebuild_log_dir_for_db`, `load_rebuilder_config` and
`record_user_prompt_submit_log` — pulling `retrieval`, `triple_extractor`,
`scoring`, `clustering`, `bfs_multihop`, `correction`, `derivation*`,
`doc_linker*`, `exploration`, `compression` and `np_pattern` behind them.

The deferral cannot be gated on config, because deciding whether the log is
enabled *is* `load_rebuilder_config`, which lived in the module being avoided.
That circularity is why this is a module extraction and not a lazy import —
the same remedy #1407 applied to `sidecar_outcome`, and for the same reason:
every hook fire is a fresh process, and roughly a third of `UserPromptSubmit`
fires are refused by the shape gate and never retrieve.

Import discipline for this module, which is the whole point of it:

* stdlib, plus `aelfrice.config_discovery` (stdlib-only), `aelfrice.models`
  (a dataclass module) and `aelfrice.query_understanding` at module scope.
  #1527 asks for a stdlib-only leaf and this is a deliberate deviation from
  that word: `RebuilderConfig.query_strategy`'s default and
  `load_rebuilder_config`'s validation both bind `query_understanding`, so
  deferring it would need a `None` sentinel in a frozen dataclass field
  default. It buys nothing on the path this module exists to protect.
  `import aelfrice.hook` loads 18 `aelfrice` modules, and only 5 of them --
  `query_understanding` and its four submodules -- are here because of that
  binding; `store`, `meta_beliefs` and `ulid` arrive through it but the hook
  imports `aelfrice.store` eagerly regardless. None of the five is on the
  retrieval path. Importing this module alone loads 12, against the 28
  `aelfrice.context_rebuilder` loaded before the extraction.
* `entity_extractor` and `triple_extractor` are imported **inside**
  `_extracted_entities_for_log` / `_query_for_recent_turns`. Both run only
  once a record is actually being built, which is strictly below
  `record_user_prompt_submit_log`'s early returns — so a gate-skipped fire,
  which passes an empty candidate list, never reaches them.
* Nothing here may import `aelfrice.context_rebuilder`, `aelfrice.retrieval`
  or `aelfrice.hook_search`, at module scope or otherwise. `aelfrice.store`
  is not imported here either, but it is *in* this module's closure anyway,
  because `query_understanding.store_cache` binds `MemoryStore` at its own
  module scope -- so treat the rule as "no direct import", not as a claim
  about the closure. `tests/test_hook_import_cost_1351.py` pins the module
  set a fire actually loads, which is the assertion that matters.

`aelfrice.context_rebuilder` re-exports every name below, so existing
`from aelfrice.context_rebuilder import ...` callers are unaffected.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, IO, cast

from aelfrice.config_discovery import discover_config
from aelfrice.models import LOCK_USER
from aelfrice.query_understanding import (
    DEFAULT_STRATEGY as DEFAULT_QUERY_STRATEGY,
)
from aelfrice.query_understanding import VALID_STRATEGIES

if TYPE_CHECKING:
    from aelfrice.models import Belief

__all__ = [
    "DEFAULT_FLOOR_L1",
    "DEFAULT_FLOOR_SESSION",
    "DEFAULT_QUERY_ENTITY_CAP",
    "DEFAULT_QUERY_STRATEGY",
    "DEFAULT_REBUILDER_TOKEN_BUDGET",
    "DEFAULT_REBUILD_LOG_ENABLED",
    "DEFAULT_THRESHOLD_FRACTION",
    "DEFAULT_TRIGGER_MODE",
    "DEFAULT_TURN_WINDOW_N",
    "MIN_QUERY_TOKEN_LENGTH",
    "QUERY_STRATEGY_KEY",
    "REBUILDER_SECTION",
    "REBUILD_FLOOR_L1_KEY",
    "REBUILD_FLOOR_SECTION",
    "REBUILD_FLOOR_SESSION_KEY",
    "REBUILD_LOG_DIRNAME",
    "REBUILD_LOG_ENABLED_KEY",
    "REBUILD_LOG_ENV",
    "REBUILD_LOG_MAX_BYTES",
    "REBUILD_LOG_SECTION",
    "RebuilderConfig",
    "RecentTurn",
    "THRESHOLD_FRACTION_KEY",
    "TOKEN_BUDGET_KEY",
    "TRIGGER_MODE_DYNAMIC",
    "TRIGGER_MODE_KEY",
    "TRIGGER_MODE_MANUAL",
    "TRIGGER_MODE_THRESHOLD",
    "TURN_WINDOW_KEY",
    "VALID_TRIGGER_MODES",
    "load_rebuilder_config",
    "record_user_prompt_submit_log",
    # Underscore-prefixed but genuinely cross-module: `context_rebuilder`
    # calls all of these, and `hook.py` calls `_rebuild_log_dir_for_db` from
    # four places. Named here so a checker does not read them as dead code
    # inside this file, which is where none of their callers live.
    "_append_rebuild_log_record",
    "_belief_lock_level_for_log",
    "_build_rebuild_log_record",
    "_empty_scores",
    "_extracted_entities_for_log",
    "_query_for_recent_turns",
    "_query_tokens",
    "_rebuild_log_dir_for_db",
    "_rebuild_log_disabled_via_env",
    "_recent_turns_hash",
]

# --- Query-construction constants -----------------------------------------

MIN_QUERY_TOKEN_LENGTH: Final[int] = 4


# --- v1.4.0 defaults ------------------------------------------------------

DEFAULT_TURN_WINDOW_N: Final[int] = 50
"""How many recent turns the v1.4 hook consults. Configurable via
`[rebuilder] turn_window_n` in `.aelfrice.toml`."""

DEFAULT_REBUILDER_TOKEN_BUDGET: Final[int] = 4000
"""Total token budget for the v1.4 rebuild block. Configurable via
`[rebuilder] token_budget` in `.aelfrice.toml`. Larger than the v1.2
default of 2000 to accommodate L0 + session-scoped + L2.5 + L1 in a
single block."""

DEFAULT_QUERY_ENTITY_CAP: Final[int] = 32
"""Cap on entities extracted from the recent-turn window. Past the
cap, additional entities are dropped on the floor. Sized so a 50-turn
window with diverse subject matter still fits without pathological
slowdown in the regex pass."""

REBUILDER_SECTION: Final[str] = "rebuilder"
TURN_WINDOW_KEY: Final[str] = "turn_window_n"
TOKEN_BUDGET_KEY: Final[str] = "token_budget"
TRIGGER_MODE_KEY: Final[str] = "trigger_mode"
THRESHOLD_FRACTION_KEY: Final[str] = "threshold_fraction"
QUERY_STRATEGY_KEY: Final[str] = "query_strategy"

# --- Rebuild diagnostic log (#288 phase-1a) ------------------------------

REBUILD_LOG_SECTION: Final[str] = "rebuild_log"
REBUILD_LOG_ENABLED_KEY: Final[str] = "enabled"
REBUILD_LOG_ENV: Final[str] = "AELFRICE_REBUILD_LOG"
REBUILD_LOG_DIRNAME: Final[str] = "rebuild_logs"
REBUILD_LOG_MAX_BYTES: Final[int] = 5 * 1024 * 1024
"""Per-session rebuild_log file size cap. On reach, append a final
`{"truncated": true, ...}` row and stop writing further records to
that file."""
DEFAULT_REBUILD_LOG_ENABLED: Final[bool] = True
"""Rebuild diagnostic log is default-on. Opt out via the
`AELFRICE_REBUILD_LOG=0` env var or `[rebuild_log] enabled = false`
in `.aelfrice.toml`."""

# --- Relevance floor (#289 / #364) ---------------------------------------

REBUILD_FLOOR_SECTION: Final[str] = "rebuild_floor"
REBUILD_FLOOR_SESSION_KEY: Final[str] = "session"
REBUILD_FLOOR_L1_KEY: Final[str] = "l1"

DEFAULT_FLOOR_SESSION: Final[float] = 0.10
"""Soft floor for session-scoped (L2) hits. Beliefs from the current
session have a high prior of relevance; reject only on near-zero
composite scores. Placeholder per `docs/design/historical/relevance_floor.md` §4 — the
production value lands in a follow-up after #288 phase-1b
calibration. Operator-tunable via `[rebuild_floor] session`."""

DEFAULT_FLOOR_L1: Final[float] = 0.40
"""Hard floor for L1 BM25 / L2.5 entity hits. Most candidates that
fall below this composite score are off-topic for the recent-turn
query. Placeholder per `docs/design/historical/relevance_floor.md` §4. Operator-tunable
via `[rebuild_floor] l1`."""



# --- v1.4.0 trigger-mode constants (issue #141) ---------------------------

TRIGGER_MODE_MANUAL: Final[str] = "manual"
TRIGGER_MODE_THRESHOLD: Final[str] = "threshold"
TRIGGER_MODE_DYNAMIC: Final[str] = "dynamic"

VALID_TRIGGER_MODES: Final[tuple[str, ...]] = (
    TRIGGER_MODE_MANUAL,
    TRIGGER_MODE_THRESHOLD,
    TRIGGER_MODE_DYNAMIC,
)
"""Allowed values for `[rebuilder] trigger_mode` in `.aelfrice.toml`.

`manual`:    PreCompact hook never fires the rebuild block; only
             explicit invocations (`aelf rebuild` / `/aelf:rebuild`)
             produce output. Default at v1.4.0.
`threshold`: the rebuild block fires on the SessionStart hook when
             `source == "compact"` fires -- i.e. immediately after
             the harness finishes a compaction (since #1031, a
             PreCompact hook can no longer inject the block itself).
             The harness's own decision to compact is the trigger;
             `threshold_fraction` documents the calibrated operating
             point.
`dynamic`:   Heuristic-driven trigger. Parked at v1.4.0 -- see
             `docs/design/context_rebuilder.md § Dynamic mode (parked)`.
             Setting this raises a clear error in the hook path.
"""

DEFAULT_TRIGGER_MODE: Final[str] = TRIGGER_MODE_THRESHOLD
"""Ship-default trigger mode.

Threshold (#746). v1.4.0 shipped with `manual` as a conservative
default pending production telemetry; the eval-harness commit-3
bench (#592) cleared on 2026-05-13 (hot-start 100% at
`trigger_threshold <= 0.6`, cold-start 75% across t in {0.5..0.8}),
ratified by the Cohen's-kappa multi-run gate (#687) the same day.
With both gates closed the rebuilder now fires by default; the
calibrated operating point is set by `DEFAULT_THRESHOLD_FRACTION`.

Opt out: `[rebuilder] trigger_mode = "manual"` in `.aelfrice.toml`.
"""

DEFAULT_THRESHOLD_FRACTION: Final[float] = 0.6
"""Calibrated default fraction for `trigger_mode = "threshold"`.

Sourced from the eval-harness calibration in
`benchmarks/context-rebuilder/calibration_v1_4_0.json` (run on the
bundled synthetic fixture sweeping 0.5/0.6/0.7/0.8/0.9). 0.6
maximizes the **token-efficient** continuation-fidelity proxy
(fidelity / token_budget_ratio) within the documented token-cost
band, with ties broken on lowest threshold (earlier firing catches
drift sooner). See `docs/design/context_rebuilder.md § Threshold
calibration` for the full sweep table and rationale.

This value is fixture-bound -- a v1.5.x re-calibration on a
captured corpus may move it. Production users opting into
`trigger_mode = "threshold"` should re-run calibration on a
representative session and override via
`[rebuilder] threshold_fraction = X` in `.aelfrice.toml`.
"""



@dataclass(frozen=True)
class RecentTurn:
    """One normalized turn fed to rebuild().

    Adapters convert wire-format transcript records into this shape.
    `session_id` is populated when the wire format carries it (the
    canonical aelfrice turns.jsonl schema does; the Claude-Code
    fallback adapter typically does not). The latest turn's
    `session_id` drives session-scoped retrieval in the v1.4 rebuild
    path; legacy callers may pass None.
    """
    role: str  # "user" or "assistant"
    text: str
    session_id: str | None = None
    ts: str | None = None
    """RFC3339/ISO-8601 turn timestamp. Populated by
    `read_recent_turns_aelfrice` when the JSONL line carries `ts`;
    `None` for the Claude-Code-transcript adapter (which lives in a
    different schema) and for legacy callers. Used by the v1.5
    working-state projector (#587) to bound `git log --since=<ts>`."""


@dataclass(frozen=True)
class RebuilderConfig:
    """Resolved `[rebuilder]` section of `.aelfrice.toml`.

    All fields default to the v1.4 module-level defaults; any may be
    overridden in a project-local `.aelfrice.toml`. Malformed values
    fall back to the default with a stderr trace, matching the
    `noise_filter`/`retrieval` config-resolution convention.

    v1.4 (issue #141) adds two trigger-mode fields:

    * `trigger_mode`  -- one of `manual`, `threshold`, `dynamic`.
                         Default `threshold` since v3.1 (#746; was
                         `manual` at v1.4). `dynamic` is parked and
                         raises in the hook path.
    * `threshold_fraction` -- float in (0.0, 1.0]; default 0.6 from
                              calibration. Documents the operating
                              point at which threshold-mode is tuned;
                              the actual gate is the harness's own
                              PreCompact firing.
    """
    turn_window_n: int = DEFAULT_TURN_WINDOW_N
    token_budget: int = DEFAULT_REBUILDER_TOKEN_BUDGET
    trigger_mode: str = DEFAULT_TRIGGER_MODE
    threshold_fraction: float = DEFAULT_THRESHOLD_FRACTION
    rebuild_log_enabled: bool = DEFAULT_REBUILD_LOG_ENABLED
    floor_session: float = DEFAULT_FLOOR_SESSION
    """v1.7 (#289 / #364) placeholder — operator-tunable via
    `[rebuild_floor] session` in .aelfrice.toml. Calibration lands
    in a follow-up after #288 phase-1b."""
    floor_l1: float = DEFAULT_FLOOR_L1
    """v1.7 (#289 / #364) placeholder — operator-tunable via
    `[rebuild_floor] l1` in .aelfrice.toml. Calibration lands
    in a follow-up after #288 phase-1b."""
    query_strategy: str = DEFAULT_QUERY_STRATEGY
    """v1.7 (#291 PR-2) selector for the R1+R3 query-understanding
    stack. Default `legacy-bm25`, the v1.4-byte-identical raw-query
    path; `stack-r1-r3` runs entity expansion + per-store IDF
    clipping and was the default from v3.0 (#718) until #1501
    reverted it. Operator-tunable via `[rebuilder] query_strategy`
    in .aelfrice.toml."""


def load_rebuilder_config(start: Path | None = None) -> RebuilderConfig:
    """Walk up from `start` looking for `.aelfrice.toml`.

    Returns the resolved `[rebuilder]` config. Missing file / missing
    section / malformed TOML / wrong-typed values all degrade to
    defaults with a stderr trace; never raises.
    """
    serr: IO[str] = sys.stderr
    # Shared discovery (#1304): inside a `config_discovery_scope`
    # N readers cost one walk instead of N. Semantics unchanged —
    # the loop this replaces already stopped at the first
    # `.aelfrice.toml` it found and never continued past it.
    candidate = discover_config(start)
    if candidate is not None:
        try:
            raw = candidate.read_bytes()
        except OSError as exc:
            print(
                f"aelfrice rebuilder: cannot read {candidate}: {exc}",
                file=serr,
            )
            return RebuilderConfig()
        try:
            parsed: dict[str, Any] = tomllib.loads(
                raw.decode("utf-8", errors="replace"),
            )
        except tomllib.TOMLDecodeError as exc:
            print(
                f"aelfrice rebuilder: malformed TOML in {candidate}: {exc}",
                file=serr,
            )
            return RebuilderConfig()
        section_obj: Any = parsed.get(REBUILDER_SECTION, {})
        if not isinstance(section_obj, dict):
            return RebuilderConfig()
        section = cast(dict[str, Any], section_obj)
        n_obj: Any = section.get(TURN_WINDOW_KEY, DEFAULT_TURN_WINDOW_N)
        b_obj: Any = section.get(
            TOKEN_BUDGET_KEY, DEFAULT_REBUILDER_TOKEN_BUDGET,
        )
        mode_obj: Any = section.get(
            TRIGGER_MODE_KEY, DEFAULT_TRIGGER_MODE,
        )
        frac_obj: Any = section.get(
            THRESHOLD_FRACTION_KEY, DEFAULT_THRESHOLD_FRACTION,
        )
        if isinstance(n_obj, bool) or not isinstance(n_obj, int) or n_obj <= 0:
            print(
                f"aelfrice rebuilder: ignoring [{REBUILDER_SECTION}] "
                f"{TURN_WINDOW_KEY} in {candidate} "
                f"(expected positive int)",
                file=serr,
            )
            n_resolved = DEFAULT_TURN_WINDOW_N
        else:
            n_resolved = n_obj
        if isinstance(b_obj, bool) or not isinstance(b_obj, int) or b_obj <= 0:
            print(
                f"aelfrice rebuilder: ignoring [{REBUILDER_SECTION}] "
                f"{TOKEN_BUDGET_KEY} in {candidate} "
                f"(expected positive int)",
                file=serr,
            )
            b_resolved = DEFAULT_REBUILDER_TOKEN_BUDGET
        else:
            b_resolved = b_obj
        if (
            not isinstance(mode_obj, str)
            or mode_obj not in VALID_TRIGGER_MODES
        ):
            print(
                f"aelfrice rebuilder: ignoring [{REBUILDER_SECTION}] "
                f"{TRIGGER_MODE_KEY} in {candidate} "
                f"(expected one of {VALID_TRIGGER_MODES})",
                file=serr,
            )
            mode_resolved = DEFAULT_TRIGGER_MODE
        else:
            mode_resolved = mode_obj
        if (
            isinstance(frac_obj, bool)
            or not isinstance(frac_obj, (int, float))
            or not (0.0 < float(frac_obj) <= 1.0)
        ):
            print(
                f"aelfrice rebuilder: ignoring [{REBUILDER_SECTION}] "
                f"{THRESHOLD_FRACTION_KEY} in {candidate} "
                f"(expected float in (0.0, 1.0])",
                file=serr,
            )
            frac_resolved = DEFAULT_THRESHOLD_FRACTION
        else:
            frac_resolved = float(frac_obj)
        log_section_obj: Any = parsed.get(REBUILD_LOG_SECTION, {})
        log_enabled_resolved: bool = DEFAULT_REBUILD_LOG_ENABLED
        if isinstance(log_section_obj, dict):
            log_section = cast(dict[str, Any], log_section_obj)
            log_enabled_obj: Any = log_section.get(
                REBUILD_LOG_ENABLED_KEY, DEFAULT_REBUILD_LOG_ENABLED,
            )
            if isinstance(log_enabled_obj, bool):
                log_enabled_resolved = log_enabled_obj
            else:
                print(
                    f"aelfrice rebuilder: ignoring "
                    f"[{REBUILD_LOG_SECTION}] "
                    f"{REBUILD_LOG_ENABLED_KEY} in {candidate} "
                    f"(expected bool)",
                    file=serr,
                )
        floor_section_obj: Any = parsed.get(REBUILD_FLOOR_SECTION, {})
        floor_session_resolved: float = DEFAULT_FLOOR_SESSION
        floor_l1_resolved: float = DEFAULT_FLOOR_L1
        if isinstance(floor_section_obj, dict):
            floor_section = cast(dict[str, Any], floor_section_obj)
            fs_obj: Any = floor_section.get(
                REBUILD_FLOOR_SESSION_KEY, DEFAULT_FLOOR_SESSION,
            )
            fl_obj: Any = floor_section.get(
                REBUILD_FLOOR_L1_KEY, DEFAULT_FLOOR_L1,
            )
            if (
                isinstance(fs_obj, bool)
                or not isinstance(fs_obj, (int, float))
                or float(fs_obj) < 0.0
            ):
                print(
                    f"aelfrice rebuilder: ignoring "
                    f"[{REBUILD_FLOOR_SECTION}] "
                    f"{REBUILD_FLOOR_SESSION_KEY} in {candidate} "
                    f"(expected non-negative number)",
                    file=serr,
                )
            else:
                floor_session_resolved = float(fs_obj)
            if (
                isinstance(fl_obj, bool)
                or not isinstance(fl_obj, (int, float))
                or float(fl_obj) < 0.0
            ):
                print(
                    f"aelfrice rebuilder: ignoring "
                    f"[{REBUILD_FLOOR_SECTION}] "
                    f"{REBUILD_FLOOR_L1_KEY} in {candidate} "
                    f"(expected non-negative number)",
                    file=serr,
                )
            else:
                floor_l1_resolved = float(fl_obj)
        qs_obj: Any = section.get(
            QUERY_STRATEGY_KEY, DEFAULT_QUERY_STRATEGY,
        )
        if (
            not isinstance(qs_obj, str)
            or qs_obj not in VALID_STRATEGIES
        ):
            print(
                f"aelfrice rebuilder: ignoring [{REBUILDER_SECTION}] "
                f"{QUERY_STRATEGY_KEY} in {candidate} "
                f"(expected one of {sorted(VALID_STRATEGIES)})",
                file=serr,
            )
            qs_resolved = DEFAULT_QUERY_STRATEGY
        else:
            qs_resolved = qs_obj
        return RebuilderConfig(
            turn_window_n=n_resolved,
            token_budget=b_resolved,
            trigger_mode=mode_resolved,
            threshold_fraction=frac_resolved,
            rebuild_log_enabled=log_enabled_resolved,
            floor_session=floor_session_resolved,
            floor_l1=floor_l1_resolved,
            query_strategy=qs_resolved,
        )
    return RebuilderConfig()


def _query_tokens(recent_turns: list[RecentTurn]) -> list[str]:
    """Whitespace-split each turn's text, keep tokens >= MIN_QUERY_TOKEN_LENGTH.

    Order is preserved (earliest mention first). Duplicate tokens are
    dropped on first appearance.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in recent_turns:
        if not t.text:
            continue
        for raw in t.text.split():
            tok = raw.strip()
            if len(tok) < MIN_QUERY_TOKEN_LENGTH:
                continue
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tok)
    return out


def _query_for_recent_turns(recent_turns: list[RecentTurn]) -> str:
    """Build a `retrieve()` query from recent-turn text.

    Strategy: extract entities (file paths, identifiers, error codes,
    etc.) and triple subjects/objects from the concatenated turns,
    deduplicate (case-folded), and join with spaces. The downstream
    `retrieve()` path runs L2.5 entity lookup on this string and L1
    BM25 over its tokens; both benefit from a high-signal query.

    Empty / non-extractable input returns "" -- `retrieve()` then
    returns L0 only, which is the correct degenerate case.
    """
    if not recent_turns:
        return ""
    # Concatenate to one string for batch extraction.
    full_text = "\n".join(t.text for t in recent_turns if t.text)
    if not full_text.strip():
        return ""

    # #1527: deferred, and below the two early returns above. Both
    # extractors are only reachable once there is text to extract from, and
    # a gate-skipped fire never gets here at all -- it stops at
    # `record_user_prompt_submit_log`'s empty-candidate guard. Importing
    # either at module scope would put `np_pattern` back on the import graph
    # of every skipped fire.
    from aelfrice.entity_extractor import extract_entities  # noqa: PLC0415
    from aelfrice.triple_extractor import extract_triples  # noqa: PLC0415

    seen: set[str] = set()
    parts: list[str] = []

    for ent in extract_entities(
        full_text, max_entities=DEFAULT_QUERY_ENTITY_CAP,
    ):
        key = ent.lower
        if key in seen:
            continue
        seen.add(key)
        parts.append(ent.raw)

    # Triples add subject + object phrases that aren't already
    # captured by entity extraction (entity extractor doesn't see
    # full noun phrases reliably outside its NP fallback pattern).
    for tr in extract_triples(full_text):
        for phrase in (tr.subject, tr.object):
            if not phrase:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(phrase)

    if not parts:
        # Fallback: short non-stopword tokens. Same shape as the
        # legacy `_query_tokens` filter; gives `retrieve()` something
        # to match against on prose-only turns where neither the
        # entity nor triple extractor finds structured signal.
        for tok in _query_tokens(recent_turns):
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(tok)

    return " ".join(parts)


def _rebuild_log_disabled_via_env() -> bool:
    """Honour `AELFRICE_REBUILD_LOG=0` opt-out. Any other value -> not
    disabled. Missing var -> not disabled (default-on)."""
    val = os.environ.get(REBUILD_LOG_ENV)
    if val is None:
        return False
    return val.strip() == "0"


def _rebuild_log_dir_for_db(db_path_val: Path) -> Path:
    """Derive the rebuild_log directory from the brain-graph DB path.

    The DB lives at `<git-common-dir>/aelfrice/memory.db`; the per-
    session JSONL files live at
    `<git-common-dir>/aelfrice/rebuild_logs/<session_id>.jsonl`.
    """
    return db_path_val.parent / REBUILD_LOG_DIRNAME


def _recent_turns_hash(recent_turns: list[RecentTurn]) -> str:
    """SHA-256 of the concatenated recent-turn `text` fields, no
    separator. Spec § Layer 1."""
    h = hashlib.sha256()
    for t in recent_turns:
        h.update(t.text.encode("utf-8"))
    return h.hexdigest()


def _extracted_entities_for_log(
    recent_turns: list[RecentTurn],
) -> list[str]:
    """Return the extracted entities surfaced into the retrieval query,
    deduplicated case-insensitively. Mirrors `_query_for_recent_turns`
    so the log records the actual entities the rebuilder saw."""
    if not recent_turns:
        return []
    full_text = "\n".join(t.text for t in recent_turns if t.text)
    if not full_text.strip():
        return []
    # #1527: deferred for the same reason as in `_query_for_recent_turns`.
    from aelfrice.entity_extractor import extract_entities  # noqa: PLC0415

    seen: set[str] = set()
    out: list[str] = []
    for ent in extract_entities(
        full_text, max_entities=DEFAULT_QUERY_ENTITY_CAP,
    ):
        key = ent.lower
        if key in seen:
            continue
        seen.add(key)
        out.append(ent.raw)
    return out


def _belief_lock_level_for_log(b: Belief) -> str:
    """Map the internal lock_level enum to the spec's "lock_level"
    field. Spec uses "user" for L0 locks and "none" otherwise; the
    internal model already uses these strings, so this is a passthrough
    that also normalises any unexpected value to "none"."""
    return b.lock_level if b.lock_level == LOCK_USER else "none"


def _empty_scores() -> dict[str, float | None]:
    """Per-belief score block. None where the field is not computed
    in the rebuilder's current code path; the floor / reranker / final
    composite land in the #289-#291 follow-ups under the #286 redesign
    tree. The log shape is locked here so phase-1b operator data is
    forward-compatible with phase-2 fixes."""
    return {
        "bm25": None,
        "posterior_mean": None,
        "reranker": None,
        "final": None,
    }


def _build_rebuild_log_record(
    recent_turns: list[RecentTurn],
    session_id: str | None,
    candidates: list[dict[str, object]],
    pack_summary: dict[str, int],
    scored_query: str | None = None,
) -> dict[str, object]:
    """Build one Layer-1 rebuild-log row.

    `scored_query` is the string `retrieve()` was **actually handed**, passed
    in by the caller rather than recomputed here (#1405). Recomputing is what
    made the row unusable: `extracted_query` below is
    `_query_for_recent_turns(...)`, and neither production path scores that.
    `rebuild_v14` scores `transform_query()` of it, and the
    `user_prompt_submit` path scores a conversation-aware composition built in
    `hook.py` that this module never sees. So the recorded query matched
    neither caller, and every replay of it measured a population production
    does not issue.

    Both are kept. `extracted_query` stays because it is what makes the
    transform's effect auditable; `scored_query` is what a replay must use.
    **Forward-only**: rows written before this carry no `scored_query`, and a
    consumer must treat its absence as "unknown", never as "no transform was
    applied".
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    extracted_query = _query_for_recent_turns(recent_turns)
    return {
        "ts": ts,
        "session_id": session_id,
        "input": {
            "recent_turns_hash": _recent_turns_hash(recent_turns),
            "n_recent_turns": len(recent_turns),
            "extracted_query": extracted_query,
            # The string retrieve() received. Absent on rows predating
            # #1405; absent is "unknown", not "same as extracted_query".
            "scored_query": scored_query,
            "extracted_entities": _extracted_entities_for_log(recent_turns),
            # Intent classification is not part of the v1.4 rebuild
            # path; null until #291 query-understanding lands.
            "extracted_intent": None,
        },
        "candidates": candidates,
        "pack_summary": pack_summary,
    }


def _append_rebuild_log_record(
    log_path: Path,
    record: dict[str, object],
    *,
    stderr: IO[str] | None = None,
) -> None:
    """Append one JSON record to the per-session rebuild_log JSONL.

    Fail-soft: any I/O error traces one line to stderr and never
    raises. Enforces the 5 MB per-session cap by appending a final
    `{"truncated": true, ...}` row when the next record would exceed
    the cap, then refusing further writes to that file.
    """
    serr = stderr if stderr is not None else sys.stderr
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            record, separators=(",", ":"), ensure_ascii=False,
        ) + "\n"
        encoded = line.encode("utf-8")
        try:
            current_size = log_path.stat().st_size
        except FileNotFoundError:
            current_size = 0
        if current_size >= REBUILD_LOG_MAX_BYTES:
            # Already at/past cap; nothing to write. The truncated
            # marker was emitted on the run that crossed the cap.
            return
        if current_size + len(encoded) > REBUILD_LOG_MAX_BYTES:
            marker = json.dumps(
                {
                    "truncated": True,
                    "ts": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                    ),
                    "reason": "size_cap",
                    "cap_bytes": REBUILD_LOG_MAX_BYTES,
                },
                separators=(",", ":"),
                ensure_ascii=False,
            ) + "\n"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(marker)
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        print(
            f"aelfrice: rebuild_log write failed (non-fatal): {exc}",
            file=serr,
        )


def record_user_prompt_submit_log(
    *,
    prompt: str,
    session_id: str | None,
    hits_pre_dedup: list[Belief],
    hits_post_dedup: list[Belief],
    log_path: Path | None,
    scored_query: str | None = None,
    enabled: bool = DEFAULT_REBUILD_LOG_ENABLED,
    stderr: IO[str] | None = None,
) -> None:
    """Emit one rebuild_log row for a UserPromptSubmit retrieval.

    Phase-1a wired the per-rebuild log only into ``rebuild_v14`` —
    fired by ``PreCompact`` and rare. The high-frequency retrieval
    path is ``user_prompt_submit``, which calls
    ``hook_search.search_for_prompt`` directly. Without this hook,
    an operator-week of normal use produces no rebuild_log rows and
    phase-1b cannot accumulate data.

    Schema is the same Layer-1 record the spec ratifies in
    ``docs/design/historical/rebuild_eval_harness.md``: synthesise a single
    ``RecentTurn`` from the prompt so the existing
    ``_build_rebuild_log_record`` machinery (hash, extracted_query,
    extracted_entities) applies unchanged. Candidates are the
    pre-dedup hit list; pre-dedup hits that survive content-hash
    dedup are ``packed``, the rest are ``dropped`` with reason
    ``content_hash_collision_with:<surviving_belief_id>``. Score
    fields are ``None`` per ``_empty_scores`` — the BM25 / posterior
    decomposition is not exposed at this call site, and locking the
    schema in phase-1a means phase-2 ranker work fills the same
    fields without a log-format migration.

    No-op when ``enabled`` is False, when the env opt-out is set, or
    when ``log_path`` is None / ``hits_pre_dedup`` is empty (mirrors
    ``rebuild_v14``: no candidate set, no row).
    """
    if not enabled:
        return
    if _rebuild_log_disabled_via_env():
        return
    if log_path is None:
        return
    if not hits_pre_dedup:
        return
    surviving_ids: set[str] = {b.id for b in hits_post_dedup}
    survivor_by_hash: dict[str, str] = {}
    for b in hits_post_dedup:
        survivor_by_hash.setdefault(
            hashlib.sha1(b.content.encode("utf-8")).hexdigest(), b.id,
        )
    candidates: list[dict[str, object]] = []
    n_dropped_by_dedup = 0
    for rank, b in enumerate(hits_pre_dedup, start=1):
        if b.id in surviving_ids:
            decision = "packed"
            reason: str | None = None
        else:
            decision = "dropped"
            digest = hashlib.sha1(b.content.encode("utf-8")).hexdigest()
            survivor = survivor_by_hash.get(digest)
            reason = (
                f"content_hash_collision_with:{survivor}"
                if survivor
                else "content_hash_collision"
            )
            n_dropped_by_dedup += 1
        candidates.append({
            "belief_id": b.id,
            "rank": rank,
            "scores": _empty_scores(),
            "lock_level": _belief_lock_level_for_log(b),
            "decision": decision,
            "reason": reason,
        })
    pack_summary: dict[str, int] = {
        "n_candidates": len(hits_pre_dedup),
        "n_packed": len(hits_post_dedup),
        # The UPS path has no visibility into floor / budget drops:
        # ranking happens inside `retrieve()` and only the surviving
        # set crosses the function boundary. Holding these at zero
        # keeps the on-disk schema stable; phase-2 wiring will fill
        # them when the ranker exposes its drop reasons.
        "n_dropped_by_floor": 0,
        "n_dropped_by_dedup": n_dropped_by_dedup,
        "n_dropped_by_budget": 0,
        "total_chars_packed": sum(len(b.content) for b in hits_post_dedup),
    }
    synthetic_turn = RecentTurn(
        role="user", text=prompt, session_id=session_id,
    )
    record = _build_rebuild_log_record(
        recent_turns=[synthetic_turn],
        session_id=session_id,
        candidates=candidates,
        pack_summary=pack_summary,
        scored_query=scored_query,
    )
    _append_rebuild_log_record(log_path, record, stderr=stderr)
