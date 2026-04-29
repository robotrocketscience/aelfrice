"""Context-rebuilder: PreCompact-driven retrieval-curated context block.

When Claude Code's context window approaches its compaction threshold,
the harness fires a PreCompact hook. This module is the script-side
half of that hook: it reads the most recent N turns from a transcript
log, runs aelfrice retrieval against those turns to surface load-
bearing beliefs, and emits a single XML-tag-delimited context block
that Claude Code injects above the next prompt.

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
  * **`additionalContext` JSON envelope.** The hook emits the
    Claude Code PreCompact harness contract (a JSON object with
    `hookSpecificOutput.additionalContext`) instead of writing the
    raw block to stdout.

Augment-mode only at v1.4.0. Both the harness's compaction summary
and the rebuilder's block land in the new context. Suppress mode
(replacing the harness compaction entirely) is parked for v2.x per
[ROADMAP.md](../docs/ROADMAP.md).

The pure rebuild() function is decoupled from the I/O paths so the
eval harness can drive it directly with a pre-loaded list of recent
turns. Two adapters convert different on-disk transcript formats
into that list: read_recent_turns_aelfrice() for the canonical
turns.jsonl format described in docs/transcript_ingest.md, and
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
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, IO, cast

from aelfrice.entity_extractor import extract_entities
from aelfrice.models import LOCK_USER, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore
from aelfrice.triple_extractor import extract_triples

# --- Legacy v1.2.0a0 constants (preserved for backwards compatibility) ----

DEFAULT_N_RECENT_TURNS: Final[int] = 10
"""Legacy default kept for v1.2.0a0 callers. v1.4.0 callers should
use `DEFAULT_TURN_WINDOW_N` (50) instead. The hook entry point in
`aelfrice.hook` reads the v1.4 default by default."""

DEFAULT_TOKEN_BUDGET: Final[int] = 2000
"""Legacy default kept for v1.2.0a0 callers (`rebuild()` and
`aelf rebuild --budget`). The v1.4 hook path uses
`DEFAULT_REBUILDER_TOKEN_BUDGET` (4000)."""

DEFAULT_PER_TOKEN_LIMIT: Final[int] = 20
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

CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
REBUILDER_SECTION: Final[str] = "rebuilder"
TURN_WINDOW_KEY: Final[str] = "turn_window_n"
TOKEN_BUDGET_KEY: Final[str] = "token_budget"
TRIGGER_MODE_KEY: Final[str] = "trigger_mode"
THRESHOLD_FRACTION_KEY: Final[str] = "threshold_fraction"

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
`threshold`: PreCompact hook fires when called by Claude Code's
             harness; the harness's own threshold gating is the
             trigger. The `threshold_fraction` documents the
             calibrated operating point.
`dynamic`:   Heuristic-driven trigger. Parked at v1.4.0 -- see
             `docs/context_rebuilder.md § Dynamic mode (parked)`.
             Setting this raises a clear error in the hook path.
"""

DEFAULT_TRIGGER_MODE: Final[str] = TRIGGER_MODE_MANUAL
"""Ship-default trigger mode at v1.4.0.

Manual is the default until production telemetry confirms the
calibrated threshold. The threshold-mode default value is set by
`benchmarks/context_rebuilder/calibration_v1_4_0.json`; the user
opts in via `[rebuilder] trigger_mode = "threshold"`.
"""

DEFAULT_THRESHOLD_FRACTION: Final[float] = 0.6
"""Calibrated default fraction for `trigger_mode = "threshold"`.

Sourced from the eval-harness calibration in
`benchmarks/context-rebuilder/calibration_v1_4_0.json` (run on the
bundled synthetic fixture sweeping 0.5/0.6/0.7/0.8/0.9). 0.6
maximizes the **token-efficient** continuation-fidelity proxy
(fidelity / token_budget_ratio) within the documented token-cost
band, with ties broken on lowest threshold (earlier firing catches
drift sooner). See `docs/context_rebuilder.md § Threshold
calibration` for the full sweep table and rationale.

This value is fixture-bound -- a v1.5.x re-calibration on a
captured corpus may move it. Production users opting into
`trigger_mode = "threshold"` should re-run calibration on a
representative session and override via
`[rebuilder] threshold_fraction = X` in `.aelfrice.toml`.
"""

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


# --- Public rebuild API ---------------------------------------------------


def rebuild(
    recent_turns: list[RecentTurn],
    store: MemoryStore,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> str:
    """Build the rebuild context block. Pure function.

    Legacy v1.2.0a0 entry point. Preserved for backwards compatibility
    with the `aelf rebuild` CLI and the eval harness. Uses the legacy
    per-token union retrieval. New callers (the v1.4 hook) should use
    `rebuild_v14()` for the L0+L1+L2.5 path.

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
    """
    locked: list[Belief] = store.list_locked_beliefs()
    locked_ids: set[str] = {b.id for b in locked}

    sid = _latest_session_id(recent_turns)

    query = _query_for_recent_turns(recent_turns)

    # retrieve() returns L0 + L2.5 + L1 in that order. We already
    # have L0 from list_locked_beliefs(); we'll rebuild the
    # composite ourselves so we can interleave session-scoped
    # beliefs in the right slot.
    retrieved: list[Belief] = retrieve(
        store, query, token_budget=token_budget,
    )
    # Drop L0 from retrieved (we'll prepend our own copy).
    non_locked_hits: list[Belief] = [
        b for b in retrieved if b.id not in locked_ids
    ]

    session_hits: list[Belief] = _session_scoped_hits(
        store, sid, exclude_ids=locked_ids,
    )
    session_ids: set[str] = {b.id for b in session_hits}

    # Pack, accounting tokens. L0 always survives.
    # Output-level content_hash dedup (#281): different belief_ids can
    # share a content_hash (re-ingest before #219, multi-source ingest,
    # or any future dedup gap). Without this, the rebuild block can
    # surface 10+ identical lines. Locked wins (it's prepended whole);
    # subsequent tiers skip any hash already packed.
    used: int = sum(_estimate_belief_tokens(b) for b in locked)
    out: list[Belief] = list(locked)
    seen_hashes: set[str] = {b.content_hash for b in locked}

    for b in session_hits:
        if b.content_hash in seen_hashes:
            continue
        cost = _estimate_belief_tokens(b)
        if used + cost > token_budget:
            break
        out.append(b)
        used += cost
        seen_hashes.add(b.content_hash)

    for b in non_locked_hits:
        if b.id in session_ids:
            continue  # already surfaced above
        if b.content_hash in seen_hashes:
            continue
        cost = _estimate_belief_tokens(b)
        if used + cost > token_budget:
            break
        out.append(b)
        used += cost
        seen_hashes.add(b.content_hash)

    return _format_block(
        recent_turns, out, session_ids, token_budget=token_budget,
    )


# --- Hook envelope helpers ------------------------------------------------


def emit_pre_compact_envelope(block: str) -> str:
    """Wrap a rebuild block in the Claude Code PreCompact JSON envelope.

    The harness expects `additionalContext` under
    `hookSpecificOutput`. Identical wire shape to the v1.2.x search-
    tool hook (`hook_search_tool.py`).
    """
    payload: dict[str, object] = {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT_NAME,
            "additionalContext": block,
        }
    }
    return json.dumps(payload)


# --- Config resolution ----------------------------------------------------


@dataclass(frozen=True)
class RebuilderConfig:
    """Resolved `[rebuilder]` section of `.aelfrice.toml`.

    All fields default to the v1.4 module-level defaults; any may be
    overridden in a project-local `.aelfrice.toml`. Malformed values
    fall back to the default with a stderr trace, matching the
    `noise_filter`/`retrieval` config-resolution convention.

    v1.4 (issue #141) adds two trigger-mode fields:

    * `trigger_mode`  -- one of `manual`, `threshold`, `dynamic`.
                         Default `manual`. `dynamic` is parked at
                         v1.4 and raises in the hook path.
    * `threshold_fraction` -- float in (0.0, 1.0]; default 0.7 from
                              calibration. Documents the operating
                              point at which threshold-mode is tuned;
                              the actual gate is the harness's own
                              PreCompact firing.
    """
    turn_window_n: int = DEFAULT_TURN_WINDOW_N
    token_budget: int = DEFAULT_REBUILDER_TOKEN_BUDGET
    trigger_mode: str = DEFAULT_TRIGGER_MODE
    threshold_fraction: float = DEFAULT_THRESHOLD_FRACTION


def load_rebuilder_config(start: Path | None = None) -> RebuilderConfig:
    """Walk up from `start` looking for `.aelfrice.toml`.

    Returns the resolved `[rebuilder]` config. Missing file / missing
    section / malformed TOML / wrong-typed values all degrade to
    defaults with a stderr trace; never raises.
    """
    serr: IO[str] = sys.stderr
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
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
            return RebuilderConfig(
                turn_window_n=n_resolved,
                token_budget=b_resolved,
                trigger_mode=mode_resolved,
                threshold_fraction=frac_resolved,
            )
        if current.parent == current:
            break
        current = current.parent
    return RebuilderConfig()


# --- Internal: legacy retrieval path (v1.2.0a0) ---------------------------


def _retrieve_for_rebuild(
    store: MemoryStore,
    recent_turns: list[RecentTurn],
    *,
    token_budget: int,
) -> list[Belief]:
    """Per-token union retrieval over recent-turn text.

    Legacy v1.2.0a0 path; preserved so the existing `aelf rebuild`
    CLI behaves byte-identical to the alpha. v1.4 callers use the
    `retrieve()`-based path in `rebuild_v14`.

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


# --- Internal: v1.4 query construction -----------------------------------


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


# --- Format helpers --------------------------------------------------------


def _estimate_belief_tokens(b: Belief) -> int:
    if not b.content:
        return 0
    return int((len(b.content) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _format_block(
    recent_turns: list[RecentTurn],
    hits: list[Belief],
    session_ids: set[str],
    *,
    token_budget: int,
) -> str:
    lines: list[str] = [OPEN_TAG]
    if recent_turns:
        lines.append("  <recent-turns>")
        for t in recent_turns:
            text = _normalize_turn_text(t.text)
            role = _xml_attr_value(t.role)
            lines.append(f'    <turn role="{role}">{_xml_escape(text)}</turn>')
        lines.append("  </recent-turns>")
    if hits:
        used_chars = sum(len(b.content) for b in hits)
        lines.append(
            f'  <retrieved-beliefs budget_used="{used_chars}/{token_budget * 4}">'
        )
        for b in hits:
            attrs: list[str] = [f'id="{_xml_attr_value(b.id)}"']
            is_locked = b.lock_level == LOCK_USER
            attrs.append(f'locked="{"true" if is_locked else "false"}"')
            if not is_locked and b.id in session_ids:
                attrs.append('session_scoped="true"')
            content = _normalize_turn_text(b.content)
            lines.append(
                f'    <belief {" ".join(attrs)}>{_xml_escape(content)}</belief>'
            )
        lines.append("  </retrieved-beliefs>")
    lines.append("  <continue/>")
    lines.append(CLOSE_TAG)
    lines.append("")
    return "\n".join(lines)


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

    Schema per docs/transcript_ingest.md: each line is a JSON object
    with at least {"role": str, "text": str}. The optional
    `session_id` field is plumbed through to `RecentTurn.session_id`
    so v1.4's session-scoped retrieval has a per-turn signal. Other
    fields (turn_id, ts, context) are ignored.

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
        out.append(RecentTurn(role=role, text=body, session_id=sid))
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
    """PreCompact hook entry point for v1.4.

    Reads the Claude Code PreCompact JSON payload from stdin, locates
    a transcript log (canonical aelfrice turns.jsonl preferred,
    Claude Code internal transcript as fallback), runs the v1.4
    rebuild against it, wraps the block in the harness's expected
    `additionalContext` envelope, and writes the JSON to stdout.

    Hook contract: never block, never raise, never propagate. Every
    failure mode (empty payload, malformed JSON, missing transcript,
    missing store, internal exception) returns exit 0 with no
    `additionalContext` written. Internal exceptions write a stack
    trace to stderr (the bash hook wrapper appends stderr to
    `~/.aelfrice/logs/hook-failures.log`).
    """
    # Imported here to avoid circular import at module load
    # (cli imports context_rebuilder for `aelf rebuild`).
    from aelfrice.cli import db_path  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]

    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    try:
        raw = sin.read()
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
                "docs/context_rebuilder.md § Dynamic mode (parked v1.5).",
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
        try:
            block = rebuild_v14(
                recent, store, token_budget=config.token_budget,
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
