"""Context-rebuilder: post-compaction retrieval-curated context block.

When the harness compacts context, it fires a SessionStart
hook with `source == "compact"` immediately afterward. (Per #1031, a
PreCompact hook cannot inject `additionalContext` and is neutered --
see `aelfrice.hook.pre_compact`.) This module supplies the retrieval
half of that path: it reads the most recent N turns from a transcript
log, runs aelfrice retrieval against those turns to surface load-
bearing beliefs, and returns a single XML-tag-delimited context block
that `aelfrice.hook.session_start()` writes to stdout when it fires
post-compaction, which the harness then injects above the next prompt.

v1.4.0 (closes #139) replaces the v1.2.0a0 alpha's per-token union
retrieval workaround with the v1.3 `retrieve()` codepath (L0 + L1 +
L2.5 in one call). It also wires:

  * **Session-scoped retrieval.** When the latest transcript turn
    carries a `session_id`, beliefs whose `session_id` matches are
    pulled from the store and ranked above L1 BM25 hits (but below
    L0 locked beliefs and L2.5 entity hits).
  * **Triple/entity-extracted query construction.** The query fed
    to `retrieve()` is built from entities and triples extracted
    from the recent turns -- no more flat whitespace concatenation.
  * **Configurable budgets via `.aelfrice.toml`.** The
    `[rebuilder] turn_window_n` and `[rebuilder] token_budget`
    keys override the defaults of 50 and 4000 respectively.
  * **Raw stdout, no JSON envelope.** Since #1031 the harness rejects
    `additionalContext` emitted from a PreCompact hook, so the block
    is written as raw text to stdout by `aelfrice.hook.session_start()`
    on `source == "compact"` instead.

Augment-mode only at v1.4.0. Both the harness's compaction summary
and the rebuilder's block land in the new context. Suppress mode
(replacing the harness compaction entirely) is parked for v2.x per
[ROADMAP.md](../docs/concepts/ROADMAP.md).

The pure rebuild() function is decoupled from the I/O paths so the
eval harness can drive it directly with a pre-loaded list of recent
turns. Two adapters convert different on-disk transcript formats
into that list: read_recent_turns_aelfrice() for the canonical
turns.jsonl format described in docs/design/transcript_ingest.md, and
read_recent_turns_claude_transcript() for Claude Code's internal
transcript format.

Output schema:

    <aelfrice-rebuild>
      <recent-turns>
        <turn role="user">...</turn>
        <turn role="assistant">...</turn>
      </recent-turns>
      <retrieved-beliefs budget_used="N/M">
        <belief id="..." locked="true">...</belief>
        <belief id="..." session_scoped="true">...</belief>
        <belief id="..." locked="false">...</belief>
      </retrieved-beliefs>
      <continue/>
    </aelfrice-rebuild>

The <continue/> marker is the stable signal the model interprets as
"resume the prior task using the above context, do not greet, do not
summarize."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, IO, cast

from aelfrice.stream_encoding import read_payload_text

from aelfrice.models import LOCK_USER, Belief
from aelfrice.query_understanding import transform_query
from aelfrice.compression import compress_for_retrieval
from aelfrice.retrieval import (
    _lock_topic,
    is_reference_lock,
    resolve_use_type_aware_compression,
    retrieve,
)
from aelfrice.scoring import posterior_mean
from aelfrice.store import MemoryStore

# Re-exported, not re-implemented (#1527). The rebuilder config and the
# phase-1a rebuild_log moved into `aelfrice.rebuild_log`, a module that imports
# no part of the retrieval subtree, so `hook.py`'s prompt-shape gate can
# resolve them on a skipped fire without loading the very thing it is skipping.
# Every name kept its old spelling and behaviour; this block exists so
# `from aelfrice.context_rebuilder import ...` keeps working for the CLI, the
# benchmarks and the existing tests. Written in the redundant-alias form,
# which is how a type checker is told an import is a deliberate re-export
# rather than a dead one.
from aelfrice.rebuild_log import (
    DEFAULT_FLOOR_L1 as DEFAULT_FLOOR_L1,
    DEFAULT_FLOOR_SESSION as DEFAULT_FLOOR_SESSION,
    DEFAULT_QUERY_ENTITY_CAP as DEFAULT_QUERY_ENTITY_CAP,
    DEFAULT_QUERY_STRATEGY as DEFAULT_QUERY_STRATEGY,
    DEFAULT_REBUILD_LOG_ENABLED as DEFAULT_REBUILD_LOG_ENABLED,
    DEFAULT_REBUILDER_TOKEN_BUDGET as DEFAULT_REBUILDER_TOKEN_BUDGET,
    DEFAULT_THRESHOLD_FRACTION as DEFAULT_THRESHOLD_FRACTION,
    DEFAULT_TRIGGER_MODE as DEFAULT_TRIGGER_MODE,
    DEFAULT_TURN_WINDOW_N as DEFAULT_TURN_WINDOW_N,
    MIN_QUERY_TOKEN_LENGTH as MIN_QUERY_TOKEN_LENGTH,
    QUERY_STRATEGY_KEY as QUERY_STRATEGY_KEY,
    REBUILD_FLOOR_L1_KEY as REBUILD_FLOOR_L1_KEY,
    REBUILD_FLOOR_SECTION as REBUILD_FLOOR_SECTION,
    REBUILD_FLOOR_SESSION_KEY as REBUILD_FLOOR_SESSION_KEY,
    REBUILD_LOG_DIRNAME as REBUILD_LOG_DIRNAME,
    REBUILD_LOG_ENABLED_KEY as REBUILD_LOG_ENABLED_KEY,
    REBUILD_LOG_ENV as REBUILD_LOG_ENV,
    REBUILD_LOG_MAX_BYTES as REBUILD_LOG_MAX_BYTES,
    REBUILD_LOG_SECTION as REBUILD_LOG_SECTION,
    REBUILDER_SECTION as REBUILDER_SECTION,
    THRESHOLD_FRACTION_KEY as THRESHOLD_FRACTION_KEY,
    TOKEN_BUDGET_KEY as TOKEN_BUDGET_KEY,
    TRIGGER_MODE_DYNAMIC as TRIGGER_MODE_DYNAMIC,
    TRIGGER_MODE_KEY as TRIGGER_MODE_KEY,
    TRIGGER_MODE_MANUAL as TRIGGER_MODE_MANUAL,
    TRIGGER_MODE_THRESHOLD as TRIGGER_MODE_THRESHOLD,
    TURN_WINDOW_KEY as TURN_WINDOW_KEY,
    VALID_TRIGGER_MODES as VALID_TRIGGER_MODES,
    RebuilderConfig as RebuilderConfig,
    RecentTurn as RecentTurn,
    load_rebuilder_config as load_rebuilder_config,
    record_user_prompt_submit_log as record_user_prompt_submit_log,
)

# The private half. `rebuild_v14` and the floor path below still call these,
# so they are imports this module uses, not just re-exports -- but a leading
# underscore across a module boundary is what `reportPrivateUsage` exists to
# flag, and it cannot tell a leaked internal from a subsystem deliberately
# split across two files. Suppressed per name so a genuinely new private
# import still has to justify itself.
from aelfrice.rebuild_log import (
    _append_rebuild_log_record as _append_rebuild_log_record,  # pyright: ignore[reportPrivateUsage]
    _belief_lock_level_for_log as _belief_lock_level_for_log,  # pyright: ignore[reportPrivateUsage]
    _build_rebuild_log_record as _build_rebuild_log_record,  # pyright: ignore[reportPrivateUsage]
    _empty_scores as _empty_scores,  # pyright: ignore[reportPrivateUsage]
    _query_for_recent_turns as _query_for_recent_turns,  # pyright: ignore[reportPrivateUsage]
    _query_tokens as _query_tokens,  # pyright: ignore[reportPrivateUsage]
    _rebuild_log_dir_for_db as _rebuild_log_dir_for_db,  # pyright: ignore[reportPrivateUsage]
    _rebuild_log_disabled_via_env as _rebuild_log_disabled_via_env,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from aelfrice.working_state import WorkingState

# --- Legacy v1.2.0a0 constants (preserved for backwards compatibility) ----

DEFAULT_N_RECENT_TURNS: Final[int] = 10
"""Legacy default kept for v1.2.0a0 callers. v1.4.0 callers should
use `DEFAULT_TURN_WINDOW_N` (50) instead. The hook entry point in
`aelfrice.hook` reads the v1.4 default by default."""

DEFAULT_TOKEN_BUDGET: Final[int] = 2000
"""Legacy default kept for v1.2.0a0 callers of the plain `rebuild()`
function (exercised only by the unit-test suite). `aelf rebuild
--budget` and the v1.4 hook path both resolve to
`DEFAULT_REBUILDER_TOKEN_BUDGET` (4000) via `rebuild_v14()`."""

DEFAULT_PER_TOKEN_LIMIT: Final[int] = 20

# --- Relevance floor (#289 / #364) ---------------------------------------

FLOOR_SCORED_QUERY_LIMIT: Final[int] = 200
"""Cap on `search_beliefs_scored()` rows pulled to derive bm25_raw
for the floor's composite score. Generous enough to cover the
retrieve()-returned candidate set in normal operation; bounded so a
pathological store cannot turn the floor into an O(N) scan."""

# --- Format constants -----------------------------------------------------

_CHARS_PER_TOKEN: Final[float] = 4.0
MAX_TURN_TEXT_CHARS: Final[int] = 500
"""Per-turn text truncation in the rebuild block.

Recent turns are decoration, not load-bearing -- the retrieved beliefs
carry the durable session state. Cap each turn's quoted text so the
block size stays bounded even when one turn is enormous (e.g., a
pasted log)."""

OPEN_TAG: Final[str] = "<aelfrice-rebuild>"
CLOSE_TAG: Final[str] = "</aelfrice-rebuild>"

_AELFRICE_LOG_RELPATH: Final[Path] = Path(".git") / "aelfrice" / "transcripts" / "turns.jsonl"

# --- Hook envelope --------------------------------------------------------

HOOK_EVENT_NAME: Final[str] = "PreCompact"
"""Value Claude Code expects in `hookSpecificOutput.hookEventName`
when a PreCompact hook emits its `additionalContext` payload."""




# --- Public rebuild API ---------------------------------------------------


def rebuild(
    recent_turns: list[RecentTurn],
    store: MemoryStore,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> str:
    """Build the rebuild context block. Pure function.

    Legacy v1.2.0a0 entry point. Retained for the direct-caller
    unit-test suite (`tests/test_context_rebuilder.py`); the `aelf
    rebuild` CLI and the eval harness have both moved to
    `rebuild_v14()`. Uses the legacy per-token union retrieval. New
    callers (the v1.4 hook) should use `rebuild_v14()` for the
    L0+L1+L2.5 path.

    Given a list of recent turns and an open MemoryStore, retrieve the
    most relevant beliefs and emit the formatted XML block.
    Deterministic given the same inputs and store contents -- the
    eval harness depends on this.

    Empty recent_turns: retrieval returns L0 locked only, the block is
    still well-formed (no <recent-turns> section).
    """
    hits = _retrieve_for_rebuild(
        store, recent_turns, token_budget=token_budget
    )
    return _format_block(
        recent_turns, hits, set(), token_budget=token_budget,
    )


def rebuild_v14(
    recent_turns: list[RecentTurn],
    store: MemoryStore,
    *,
    token_budget: int = DEFAULT_REBUILDER_TOKEN_BUDGET,
    rebuild_log_path: Path | None = None,
    rebuild_log_enabled: bool = DEFAULT_REBUILD_LOG_ENABLED,
    session_id_for_log: str | None = None,
    floor_session: float = 0.0,
    floor_l1: float = 0.0,
    query_strategy: str = DEFAULT_QUERY_STRATEGY,
    working_state: "WorkingState | None" = None,
    use_type_aware_compression: bool | None = None,
) -> str:
    """v1.4 rebuild: L0 + session-scoped + L2.5/L1 via `retrieve()`.

    Replaces the v1.2.0a0 per-token union with one `retrieve()` call
    that handles L0 locked beliefs, L2.5 entity-index, and L1 BM25
    in one pass (and inherits posterior-weighted ranking when #146
    lands). Session-scoping injects beliefs whose `session_id`
    matches the latest turn's session above the L2.5/L1 tail.

    Pack order:
      1. L0 locked (full, never trimmed).
      2. Session-scoped beliefs whose `session_id` matches the
         latest recent_turn's `session_id` (if any).
      3. L2.5 + L1 hits from `retrieve()`, in retrieve()'s native
         order, deduplicated against L0 + session-scoped.

    Pure function. Deterministic given the same inputs and store
    contents -- the regression test for issue #139 relies on this.
    Empty recent_turns: returns L0 only, block is still well-formed.

    v1.7 (#289 / #364): per-lane relevance floor. L0 locked always
    packs (no floor). Session-scoped (L2) packs above `floor_session`.
    L1 / L2.5 packs above `floor_l1`. When all hits are floored out
    AND no locks exist the function returns the empty string —
    `rebuild_v14`'s "I don't know — say nothing" path. See
    `docs/design/historical/relevance_floor.md` for the composite-score formula and
    the rationale for the split.

    The function-signature defaults are `floor_session=0.0,
    floor_l1=0.0` — backwards-compatible no-floor behavior for
    direct callers. The production hook path threads the
    placeholder values (`DEFAULT_FLOOR_SESSION`,
    `DEFAULT_FLOOR_L1`) from `RebuilderConfig`, so operators get
    the floor end-to-end while existing tests and ad-hoc callers
    are unaffected.

    v1.7 (#291 PR-2): `query_strategy` selects the query rewriter.
    Default `legacy-bm25` — byte-identical to the v1.4 path, the raw
    query passed through. `stack-r1-r3` runs the R1 entity-expansion
    + R3 per-store IDF-clip stack from `aelfrice.query_understanding`;
    it was the default from v3.0 (#718) and was reverted in #1501.
    See `query_understanding.strategy.DEFAULT_STRATEGY` for why.
    """
    # #798: resolve compression flag once, thread through pack accounting
    # so rebuild_v14's trim matches retrieve()'s trim under the same flag.
    compress_on: bool = resolve_use_type_aware_compression(
        use_type_aware_compression,
    )

    locked: list[Belief] = store.list_locked_beliefs()
    locked_ids: set[str] = {b.id for b in locked}

    sid = _latest_session_id(recent_turns)

    raw_query = _query_for_recent_turns(recent_turns)
    query = transform_query(raw_query, store, query_strategy)

    # retrieve() returns L0 + L2.5 + L1 in that order. We already
    # have L0 from list_locked_beliefs(); we'll rebuild the
    # composite ourselves so we can interleave session-scoped
    # beliefs in the right slot.
    retrieved: list[Belief] = retrieve(
        store,
        query,
        token_budget=token_budget,
        use_type_aware_compression=compress_on,
        # #1016-B: this injection path renders reference locks as a bounded
        # topic, so budget them at manifest size too (byte-identical until
        # a lock is demoted to reference).
        manifest_reference_locks=True,
    )
    # Drop L0 from retrieved (we'll prepend our own copy).
    non_locked_hits: list[Belief] = [
        b for b in retrieved if b.id not in locked_ids
    ]

    session_hits: list[Belief] = _session_scoped_hits(
        store, sid, exclude_ids=locked_ids,
    )
    session_ids: set[str] = {b.id for b in session_hits}

    # v1.7 (#289 / #364): per-lane composite-score floor. Pull bm25_raw
    # for every candidate via one extra FTS5 call; off-FTS5 candidates
    # (entity-only, BFS, session-only) score with bm25_normalized=1.0
    # so the floor decision rests on posterior_mean alone for them.
    bm25_lookup: dict[str, float] = _bm25_lookup_for_query(store, query)

    def _score_for(b: Belief) -> tuple[float, float | None]:
        bm25_raw = bm25_lookup.get(b.id)
        return (
            floor_composite_score(bm25_raw, b.alpha, b.beta),
            bm25_raw,
        )

    # Pack, accounting tokens. L0 always survives.
    # Output-level content_hash dedup (#281): different belief_ids can
    # share a content_hash (re-ingest before #219, multi-source ingest,
    # or any future dedup gap). Without this, the rebuild block can
    # surface 10+ identical lines. Locked wins (it's prepended whole);
    # subsequent tiers skip any hash already packed.
    used: int = sum(
        _estimate_belief_tokens(b, compress_on=compress_on) for b in locked
    )
    out: list[Belief] = list(locked)
    seen_hashes: set[str] = {b.content_hash for b in locked}
    hash_to_packed_id: dict[str, str] = {b.content_hash: b.id for b in locked}

    # #288 phase-1a: track per-candidate decision metadata so the
    # rebuild_log records every belief the rebuilder considered, not
    # just those it packed. Order: L0, then session-scoped, then L2.5/L1.
    log_candidates: list[dict[str, object]] = []
    n_dropped_by_dedup = 0
    n_dropped_by_budget = 0
    n_dropped_by_floor = 0
    budget_exceeded_session = False
    budget_exceeded_non_locked = False

    for b in locked:
        log_candidates.append(
            {
                "belief_id": b.id,
                "rank": len(log_candidates) + 1,
                "scores": _empty_scores(),
                "lock_level": _belief_lock_level_for_log(b),
                "decision": "packed",
                "reason": None,
            }
        )

    for b in session_hits:
        decision: str
        reason: str | None
        composite, bm25_raw = _score_for(b)
        if b.content_hash in seen_hashes:
            decision = "dropped"
            reason = (
                f"content_hash_collision_with:{hash_to_packed_id[b.content_hash]}"
            )
            n_dropped_by_dedup += 1
        elif composite < floor_session:
            decision = "dropped"
            reason = (
                f"below_floor_session:{composite:.4f}<{floor_session:.4f}"
            )
            n_dropped_by_floor += 1
        elif budget_exceeded_session:
            decision = "dropped"
            reason = "budget_exceeded"
            n_dropped_by_budget += 1
        else:
            cost = _estimate_belief_tokens(b, compress_on=compress_on)
            if used + cost > token_budget:
                budget_exceeded_session = True
                decision = "dropped"
                reason = "budget_exceeded"
                n_dropped_by_budget += 1
            else:
                out.append(b)
                used += cost
                seen_hashes.add(b.content_hash)
                hash_to_packed_id[b.content_hash] = b.id
                decision = "packed"
                reason = None
        log_candidates.append(
            {
                "belief_id": b.id,
                "rank": len(log_candidates) + 1,
                "scores": {
                    "bm25": bm25_raw,
                    "posterior_mean": posterior_mean(b.alpha, b.beta),
                    "reranker": None,
                    "final": composite,
                },
                "lock_level": _belief_lock_level_for_log(b),
                "decision": decision,
                "reason": reason,
            }
        )

    for b in non_locked_hits:
        if b.id in session_ids:
            continue  # already surfaced above as a session candidate
        decision2: str
        reason2: str | None
        composite2, bm25_raw2 = _score_for(b)
        if b.content_hash in seen_hashes:
            decision2 = "dropped"
            reason2 = (
                f"content_hash_collision_with:{hash_to_packed_id[b.content_hash]}"
            )
            n_dropped_by_dedup += 1
        elif composite2 < floor_l1:
            decision2 = "dropped"
            reason2 = f"below_floor_l1:{composite2:.4f}<{floor_l1:.4f}"
            n_dropped_by_floor += 1
        elif budget_exceeded_non_locked:
            decision2 = "dropped"
            reason2 = "budget_exceeded"
            n_dropped_by_budget += 1
        else:
            cost = _estimate_belief_tokens(b, compress_on=compress_on)
            if used + cost > token_budget:
                budget_exceeded_non_locked = True
                decision2 = "dropped"
                reason2 = "budget_exceeded"
                n_dropped_by_budget += 1
            else:
                out.append(b)
                used += cost
                seen_hashes.add(b.content_hash)
                hash_to_packed_id[b.content_hash] = b.id
                decision2 = "packed"
                reason2 = None
        log_candidates.append(
            {
                "belief_id": b.id,
                "rank": len(log_candidates) + 1,
                "scores": {
                    "bm25": bm25_raw2,
                    "posterior_mean": posterior_mean(b.alpha, b.beta),
                    "reranker": None,
                    "final": composite2,
                },
                "lock_level": _belief_lock_level_for_log(b),
                "decision": decision2,
                "reason": reason2,
            }
        )

    if (
        rebuild_log_enabled
        and not _rebuild_log_disabled_via_env()
        and rebuild_log_path is not None
        and log_candidates
    ):
        n_packed = sum(
            1 for c in log_candidates if c["decision"] == "packed"
        )
        total_chars_packed = sum(len(b.content) for b in out)
        pack_summary: dict[str, int] = {
            "n_candidates": len(log_candidates),
            "n_packed": n_packed,
            "n_dropped_by_floor": n_dropped_by_floor,
            "n_dropped_by_dedup": n_dropped_by_dedup,
            "n_dropped_by_budget": n_dropped_by_budget,
            "total_chars_packed": total_chars_packed,
        }
        record = _build_rebuild_log_record(
            recent_turns=recent_turns,
            session_id=session_id_for_log,
            candidates=log_candidates,
            pack_summary=pack_summary,
            # The post-transform string handed to retrieve() above, not a
            # re-derivation (#1405).
            scored_query=query,
        )
        _append_rebuild_log_record(rebuild_log_path, record)

    # v1.7 (#289 / #364) silent path: when no candidate cleared any
    # lane (no L0 locks, no above-floor session/L1), return "" so
    # downstream callers do not inject an empty memory block. The
    # PreCompact hook's `if body:` guard already drops the envelope
    # on falsy output (hook.py L966), so empty input -> no
    # additionalContext written.
    #
    # v1.5 (#587): the working-state sub-block is its own load-bearing
    # signal (state-of-work, not retrieval). When there are no belief
    # hits but working-state has content, still emit — the rebuild
    # block becomes "no beliefs to surface, here's where you were."
    has_working_state = working_state is not None and not working_state.is_empty()
    if not out and not has_working_state:
        return ""

    return _format_block(
        recent_turns,
        out,
        session_ids,
        token_budget=token_budget,
        working_state=working_state,
    )


# --- Hook envelope helpers ------------------------------------------------


def emit_pre_compact_envelope(block: str) -> str:
    """Wrap a rebuild block in the JSON envelope the pre-#1031
    PreCompact contract used.

    Retained for the legacy `main()` entry point and tests only: the
    harness rejects `additionalContext` emitted from a PreCompact
    hook, so this envelope is not part of the live
    SessionStart(source == "compact") injection path, which writes
    the raw block to stdout instead. Identical wire shape to the
    v1.2.x search-tool hook (`hook_search_tool.py`).
    """
    payload: dict[str, object] = {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT_NAME,
            "additionalContext": block,
        }
    }
    return json.dumps(payload)


# --- Config resolution ----------------------------------------------------




# --- Internal: legacy retrieval path (v1.2.0a0) ---------------------------


def _retrieve_for_rebuild(
    store: MemoryStore,
    recent_turns: list[RecentTurn],
    *,
    token_budget: int,
) -> list[Belief]:
    """Per-token union retrieval over recent-turn text.

    Legacy v1.2.0a0 path, reachable only via `rebuild()` (now
    unit-test-only); the `aelf rebuild` CLI moved to the
    `retrieve()`-based `rebuild_v14()` path.

    L0 locked beliefs always come first and are never trimmed.
    """
    locked: list[Belief] = store.list_locked_beliefs()
    locked_ids: set[str] = {b.id for b in locked}

    tokens = _query_tokens(recent_turns)
    l1: list[Belief] = []
    seen_ids: set[str] = set(locked_ids)
    for tok in tokens:
        for b in store.search_beliefs(tok, limit=DEFAULT_PER_TOKEN_LIMIT):
            if b.id in seen_ids:
                continue
            seen_ids.add(b.id)
            l1.append(b)

    used: int = sum(_estimate_belief_tokens(b) for b in locked)
    out: list[Belief] = list(locked)
    for b in l1:
        cost: int = _estimate_belief_tokens(b)
        if used + cost > token_budget:
            break
        out.append(b)
        used += cost
    return out




# --- Internal: v1.4 query construction -----------------------------------




def _latest_session_id(recent_turns: list[RecentTurn]) -> str | None:
    """Return the most recent turn's `session_id`, or None.

    Walks from the tail because (a) the latest session is the live
    one and (b) earlier turns in a long-running window may carry
    stale session ids if the harness ever rotates a session in the
    middle of a transcript.
    """
    for t in reversed(recent_turns):
        if t.session_id and t.session_id.strip():
            return t.session_id
    return None


def _session_scoped_hits(
    store: MemoryStore,
    session_id: str | None,
    *,
    exclude_ids: set[str],
) -> list[Belief]:
    """Return beliefs tagged with `session_id`, dedup against L0.

    Uses the public `MemoryStore` connection directly because there
    is no v1.x public accessor for "list beliefs by session_id."
    Read-only; returns at most a generous bound (1000) so a runaway
    session doesn't blow the rebuild block's budget on session
    membership alone -- the budget cap in the caller still trims.

    Empty `session_id` -> []. No matching rows -> [].
    """
    if not session_id or not session_id.strip():
        return []
    cur = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        """
        SELECT * FROM beliefs
        WHERE session_id = ?
        ORDER BY created_at DESC, id ASC
        LIMIT 1000
        """,
        (session_id,),
    )
    out: list[Belief] = []
    # Reuse the store's row-to-belief converter so the dataclass
    # shape stays in sync with whatever schema migrations land.
    from aelfrice.store import _row_to_belief  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]
    for row in cur.fetchall():
        b = _row_to_belief(row)  # pyright: ignore[reportPrivateUsage]
        if b.id in exclude_ids:
            continue
        out.append(b)
    return out


# --- Rebuild diagnostic log (#288 phase-1a) -------------------------------




def floor_composite_score(
    bm25_raw: float | None, alpha: float, beta: float,
) -> float:
    """Composite floor score: `bm25_normalized * (0.5 + 0.5 * posterior_mean)`.

    `bm25_raw` is SQLite's signed FTS5 score (smaller = better, ≤ 0
    in normal cases). It is converted to `bm25_normalized` ∈ [0, 1]
    via `min(-bm25_raw, 1.0) / 1.0` style clamping. `None` means the
    candidate is not in the FTS5 hit set (e.g. L2.5 entity-only or
    L3 BFS expansion); the floor gives such candidates the
    benefit-of-doubt `bm25_normalized = 1.0` so the decision rests
    on `posterior_mean`. Locked beliefs bypass the floor entirely;
    callers should not call this for them.

    Composite formula and clamp behavior pinned by
    `docs/design/historical/relevance_floor.md` §1. Coefficient 0.5/0.5 is the
    placeholder split; tuning lands post-#288.
    """
    if bm25_raw is None:
        bm25_normalized = 1.0
    else:
        # SQLite FTS5 returns a non-positive score; flip sign and
        # clamp to [0, 1] against a unit baseline. Above-baseline
        # raw scores saturate at 1.0 — the floor is about whether
        # there is a signal at all, not how strong it is.
        signed = -float(bm25_raw)
        if signed <= 0.0:
            bm25_normalized = 0.0
        elif signed >= 1.0:
            bm25_normalized = 1.0
        else:
            bm25_normalized = signed
    pm = posterior_mean(alpha, beta)
    return bm25_normalized * (0.5 + 0.5 * pm)


def _bm25_lookup_for_query(
    store: MemoryStore, query: str,
) -> dict[str, float]:
    """Return `{belief_id: bm25_raw}` for the query.

    Used to score floor candidates against the same BM25 raw values
    SQLite would return inside `retrieve()`. Off-FTS5 candidates
    (entity index, BFS expansion, session-scoped) are absent from
    the dict — callers should treat absence as "no FTS5 signal" and
    pass `None` to `floor_composite_score`.

    Empty / whitespace-only query: returns `{}`.
    """
    if not query or not query.strip():
        return {}
    try:
        scored = store.search_beliefs_scored(
            query, limit=FLOOR_SCORED_QUERY_LIMIT,
        )
    except Exception:  # pyright: ignore[reportBroadException]
        # Floor must never block rebuild on a transient store error.
        return {}
    return {b.id: bm25_raw for b, bm25_raw in scored}




# --- Format helpers --------------------------------------------------------


def _estimate_belief_tokens(b: Belief, *, compress_on: bool = False) -> int:
    # Verbatim cost by default; compressed render cost when compress_on=True.
    # Mirrors retrieval._cost so rebuild_v14's pack accounting matches
    # retrieve()'s pack accounting under the same flag — closes #798.
    # Locks always render verbatim (compress_for_retrieval honors locked=True).
    if not b.content:
        return 0
    # #1016-B: a reference-tier lock renders as its bounded topic (see
    # _format_block), so cost it at topic size — not full content — or it
    # would over-budget the pack and crowd out later hits.
    if is_reference_lock(b):
        return int(
            (len(_lock_topic(b.content)) + _CHARS_PER_TOKEN - 1)
            // _CHARS_PER_TOKEN
        )
    if compress_on:
        cb = compress_for_retrieval(b, locked=(b.lock_level == LOCK_USER))
        return cb.rendered_tokens
    return int((len(b.content) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _format_block(
    recent_turns: list[RecentTurn],
    hits: list[Belief],
    session_ids: set[str],
    *,
    token_budget: int,
    working_state: "WorkingState | None" = None,
) -> str:
    lines: list[str] = [OPEN_TAG]
    if recent_turns:
        lines.append("  <recent-turns>")
        for t in recent_turns:
            text = _normalize_turn_text(t.text)
            role = _xml_attr_value(t.role)
            lines.append(f'    <turn role="{role}">{_xml_escape(text)}</turn>')
        lines.append("  </recent-turns>")
    if working_state is not None and not working_state.is_empty():
        lines.extend(_format_working_state(working_state))
    if hits:
        # #1016-B: a reference-tier lock renders its bounded topic, not
        # full content (full text on demand via `aelf locked`), so it
        # cannot bloat the rebuild block. Frozen/other beliefs unchanged.
        rendered: list[tuple[Belief, str, bool]] = [
            (b, _lock_topic(b.content), True)
            if is_reference_lock(b)
            else (b, _normalize_turn_text(b.content), False)
            for b in hits
        ]
        used_chars = sum(len(text) for _, text, _ in rendered)
        lines.append(
            f'  <retrieved-beliefs budget_used="{used_chars}/{token_budget * 4}">'
        )
        for b, content, is_ref in rendered:
            attrs: list[str] = [f'id="{_xml_attr_value(b.id)}"']
            is_locked = b.lock_level == LOCK_USER
            attrs.append(f'locked="{"true" if is_locked else "false"}"')
            if not is_locked and b.id in session_ids:
                attrs.append('session_scoped="true"')
            if is_ref:
                attrs.append('tier="reference"')
            lines.append(
                f'    <belief {" ".join(attrs)}>{_xml_escape(content)}</belief>'
            )
        lines.append("  </retrieved-beliefs>")
    lines.append("  <continue/>")
    lines.append(CLOSE_TAG)
    lines.append("")
    return "\n".join(lines)


def _format_working_state(ws: "WorkingState") -> list[str]:
    """Render a WorkingState into the rebuild-block sub-block (#587).

    Empty fields are individually omitted so the sub-block stays terse.
    Caller (`_format_block`) gates on `ws.is_empty()` for the all-empty
    case.
    """
    out: list[str] = ["  <working-state>"]
    if ws.branch:
        out.append(f"    <branch>{_xml_escape(ws.branch)}</branch>")
    if ws.status_porcelain:
        out.append("    <git-status>")
        for line in ws.status_porcelain:
            out.append(f"      <line>{_xml_escape(line)}</line>")
        out.append("    </git-status>")
    if ws.recent_log:
        out.append("    <recent-commits>")
        for line in ws.recent_log:
            out.append(f"      <commit>{_xml_escape(line)}</commit>")
        out.append("    </recent-commits>")
    if ws.recent_user_prompts:
        out.append("    <recent-user-prompts>")
        for prompt in ws.recent_user_prompts:
            text = _normalize_turn_text(prompt)
            out.append(f"      <prompt>{_xml_escape(text)}</prompt>")
        out.append("    </recent-user-prompts>")
    if ws.session_commits:
        out.append("    <session-commits>")
        for line in ws.session_commits:
            out.append(f"      <commit>{_xml_escape(line)}</commit>")
        out.append("    </session-commits>")
    out.append("  </working-state>")
    return out


def _normalize_turn_text(text: str) -> str:
    """Collapse whitespace and truncate to MAX_TURN_TEXT_CHARS."""
    collapsed = " ".join(text.split())
    if len(collapsed) > MAX_TURN_TEXT_CHARS:
        return collapsed[:MAX_TURN_TEXT_CHARS] + "..."
    return collapsed


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_attr_value(s: str) -> str:
    return _xml_escape(s).replace('"', "&quot;")


# --- Transcript adapters --------------------------------------------------


def read_recent_turns_aelfrice(path: Path, n: int) -> list[RecentTurn]:
    """Read tail of an aelfrice turns.jsonl into RecentTurn records.

    Schema per docs/design/transcript_ingest.md: each line is a JSON object
    with at least {"role": str, "text": str}. The optional
    `session_id` and `ts` fields are plumbed through to
    `RecentTurn.session_id` and `RecentTurn.ts` so v1.4's session-
    scoped retrieval and v1.5's working-state projector (#587) have
    per-turn signals. Other fields (turn_id, context) are ignored.

    Robust: malformed lines are skipped, missing files return empty.
    """
    if not path.exists():
        return []
    out: list[RecentTurn] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue
        rd = cast(dict[str, object], r)
        role = rd.get("role")
        body = rd.get("text")
        if not isinstance(role, str) or role not in ("user", "assistant"):
            continue
        if not isinstance(body, str) or not body.strip():
            continue
        sid_obj = rd.get("session_id")
        sid = sid_obj if isinstance(sid_obj, str) and sid_obj else None
        ts_obj = rd.get("ts")
        ts = ts_obj if isinstance(ts_obj, str) and ts_obj else None
        out.append(RecentTurn(role=role, text=body, session_id=sid, ts=ts))
    return out[-n:] if n > 0 else []


def read_recent_turns_claude_transcript(
    path: Path, n: int
) -> list[RecentTurn]:
    """Adapter for Claude Code's internal transcript format.

    Fallback used when the canonical aelfrice turns.jsonl does not
    exist (typical pre-transcript_ingest setup). Reads Claude Code's
    per-session JSONL at ~/.claude/projects/<hash>/<session>.jsonl.
    Schema is internal-harness JSON and subject to change between
    Claude Code releases; this adapter is best-effort and fails
    closed (returns [] on shape mismatch).

    `session_id` on the returned `RecentTurn` is set from the
    record's `sessionId` field when present (Claude Code's wire
    name); v1.4 session-scoped retrieval falls back to None when
    the record doesn't carry it.

    Records of interest:
      {"type": "user", "message": {"role": "user", "content": "..."}}
      {"type": "assistant", "message": {"role": "assistant",
        "content": [{"type": "text", "text": "..."}, ...]}}

    Tool-call records, sub-agent records, and meta records are
    skipped.
    """
    if not path.exists():
        return []
    out: list[RecentTurn] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue
        rd = cast(dict[str, object], r)
        record_type = rd.get("type")
        if record_type not in ("user", "assistant"):
            continue
        msg = rd.get("message")
        if not isinstance(msg, dict):
            continue
        msg_typed = cast(dict[str, object], msg)
        role = msg_typed.get("role")
        if not isinstance(role, str) or role not in ("user", "assistant"):
            continue
        content = msg_typed.get("content")
        body = _extract_text_from_claude_content(content)
        if not body:
            continue
        sid_obj = rd.get("sessionId")
        sid = sid_obj if isinstance(sid_obj, str) and sid_obj else None
        out.append(RecentTurn(role=role, text=body, session_id=sid))
    return out[-n:] if n > 0 else []


def _extract_text_from_claude_content(content: object) -> str:
    """Pull the human-readable text out of a Claude message content field.

    Claude's content is either a bare string (user prompts) or a list
    of content blocks (assistant turns and structured user messages).
    We concatenate every "text"-typed block; non-text blocks (tool
    use, tool result, images) are ignored.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in cast(list[object], content):
        if not isinstance(block, dict):
            continue
        bd = cast(dict[str, object], block)
        if bd.get("type") != "text":
            continue
        t = bd.get("text")
        if isinstance(t, str) and t.strip():
            parts.append(t.strip())
    return "\n".join(parts).strip()


def find_aelfrice_log(cwd: Path) -> Path | None:
    """Walk upward from cwd to find a .git/ root, return its turns.jsonl path.

    Returns the path even if the file does not exist -- callers use
    Path.exists() to decide. Returns None if no .git/ is found in any
    ancestor (cwd is outside a git repo).
    """
    cur = cwd.resolve()
    while True:
        if (cur / ".git").exists():
            return cur / _AELFRICE_LOG_RELPATH
        if cur.parent == cur:
            return None
        cur = cur.parent


# --- v1.4 hook entry point -----------------------------------------------


def main(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Legacy v1.4 PreCompact entry point -- NOT the live hook.

    Retained for tests/benchmarks parity only. Since #1031 the harness
    rejects `additionalContext` emitted from a PreCompact hook, so
    production wiring runs `aelfrice.hook.pre_compact()` (emits
    nothing) and `aelfrice.hook.session_start()` (`source ==
    "compact"`) instead. This function still reads the harness's
    PreCompact JSON payload from stdin, locates a transcript log
    (canonical aelfrice turns.jsonl preferred, harness-internal
    transcript as fallback), runs the v1.4 rebuild against it, wraps
    the block in the (harness-rejected) `additionalContext` envelope,
    and writes the JSON to stdout, matching the pre-#1031 contract.

    Hook contract: never block, never raise, never propagate. Every
    failure mode (empty payload, malformed JSON, missing transcript,
    missing store, internal exception) returns exit 0 with no
    `additionalContext` written. Internal exceptions write a stack
    trace to stderr (the bash hook wrapper appends stderr to
    `~/.aelfrice/logs/hook-failures.log`).
    """
    from aelfrice.db_paths import db_path

    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    try:
        raw = read_payload_text(sin, serr) or ""
        payload = _parse_payload(raw)
        if payload is None:
            return 0

        cwd_obj = payload.get("cwd")
        cwd_str = cwd_obj if isinstance(cwd_obj, str) and cwd_obj else None
        cwd = Path(cwd_str) if cwd_str else Path.cwd()

        config = load_rebuilder_config(cwd)

        # v1.4 trigger-mode gating (issue #141). Manual + dynamic
        # short-circuit before any retrieval or transcript work.
        # See `aelfrice.hook.pre_compact` for the same gate; both
        # entry points must agree.
        if config.trigger_mode == TRIGGER_MODE_MANUAL:
            return 0
        if config.trigger_mode == TRIGGER_MODE_DYNAMIC:
            print(
                "aelfrice rebuilder: trigger_mode='dynamic' is parked "
                "at v1.4, ships v1.5; falling back to no-op. See "
                "docs/design/context_rebuilder.md § Dynamic mode (parked v1.5).",
                file=serr,
            )
            return 0

        recent = _read_recent_for_pre_compact(payload, config.turn_window_n)
        if not recent:
            # Empty transcript: emit nothing per the issue's edge-
            # case acceptance criterion.
            _maybe_emit_locked_only(sout, cwd, config.token_budget)
            return 0

        p = db_path()
        if str(p) != ":memory:" and not p.exists():
            # Missing store: emit nothing (issue acceptance criterion).
            return 0

        store = MemoryStore(str(p))
        # #288 phase-1a: route the rebuild_log alongside the brain-
        # graph DB. Disabled when the store is in-memory (no on-disk
        # location to write to) or when the operator opted out.
        sid_for_log = _latest_session_id(recent)
        log_path: Path | None = None
        if str(p) != ":memory:" and sid_for_log:
            log_path = (
                _rebuild_log_dir_for_db(p) / f"{sid_for_log}.jsonl"
            )
        # v1.5 (#587): post-compact hot-start. Project working-state from
        # cwd + recent turns; rebuild_v14 surfaces it under a
        # <working-state> sub-block alongside the retrieved beliefs.
        # Best-effort: any projector failure returns an empty
        # WorkingState that the formatter omits.
        # Lazy import to break the working_state ↔ context_rebuilder
        # cyclic-import that CodeQL `py/unsafe-cyclic-import` flagged.
        from aelfrice.working_state import WorkingState, project_working_state

        try:
            working_state = project_working_state(cwd, recent)
        except Exception:  # noqa: BLE001 -- hook contract: never raise
            working_state = WorkingState()
        try:
            block = rebuild_v14(
                recent,
                store,
                token_budget=config.token_budget,
                rebuild_log_path=log_path,
                rebuild_log_enabled=config.rebuild_log_enabled,
                session_id_for_log=sid_for_log,
                working_state=working_state,
            )
        finally:
            store.close()
        sout.write(emit_pre_compact_envelope(block))
    except Exception:  # non-blocking: surface but do not fail
        import traceback  # noqa: PLC0415
        traceback.print_exc(file=serr)
    return 0


def _parse_payload(raw: str) -> dict[str, object] | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return cast(dict[str, object], payload)


def _read_recent_for_pre_compact(
    payload: dict[str, object], n_recent_turns: int
) -> list[RecentTurn]:
    """Locate a transcript and read its tail.

    Resolution order:
      1. <payload.cwd>/.git/aelfrice/transcripts/turns.jsonl -- the
         canonical aelfrice log written by the per-turn
         UserPromptSubmit/Stop hooks.
      2. <payload.transcript_path> -- Claude Code's internal per-
         session transcript JSONL. Fallback used when the canonical
         log is absent.
      3. Empty list -- both sources missing or unreadable.
    """
    cwd_obj = payload.get("cwd")
    if isinstance(cwd_obj, str) and cwd_obj.strip():
        try:
            cwd = Path(cwd_obj)
            log_path = find_aelfrice_log(cwd)
        except OSError:
            log_path = None
        if log_path is not None and log_path.exists():
            return read_recent_turns_aelfrice(log_path, n=n_recent_turns)
    tp_obj = payload.get("transcript_path")
    if isinstance(tp_obj, str) and tp_obj.strip():
        tp = Path(tp_obj)
        if tp.exists():
            return read_recent_turns_claude_transcript(
                tp, n=n_recent_turns,
            )
    return []


def _maybe_emit_locked_only(
    stdout: IO[str], cwd: Path, token_budget: int,
) -> None:
    """Empty-transcript edge case: emit nothing.

    Per issue #139 acceptance criterion, an empty transcript exits 0
    without `additionalContext`. The store may still hold L0 locked
    beliefs, but the SessionStart hook handles that channel; the
    rebuilder's job is to surface state from the *current* session
    tail. With no tail to point at, the right answer is silence.
    """
    _ = stdout, cwd, token_budget  # kept for future suppress-mode
