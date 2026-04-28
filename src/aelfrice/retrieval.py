"""Three-layer retrieval: L0 locked beliefs, L2.5 entity-index, L1 FTS5 BM25.

Token-budgeted output (default 2400 tokens at v1.3.0, ~4 chars/token
estimate). L0 beliefs always present in the output above any non-locked
result and never trimmed by the budget — locks are user-asserted ground
truth and must survive retrieval.

L2.5 (v1.3.0) is a deterministic entity-index lookup that runs between
L0 and L1. It extracts entities from the query using
`aelfrice.entity_extractor.extract_entities`, looks them up in the
`belief_entities` table, ranks by entity-overlap count (tie-break:
belief_id ASC), and feeds a `DEFAULT_L25_TOKEN_SUBBUDGET`-sized slice
into the output ahead of L1. L1 fills the remaining budget.

Default-on at v1.3.0 via the config flag `entity_index_enabled` in
`[retrieval]` of `.aelfrice.toml`. Two off-switches:

  - `AELFRICE_ENTITY_INDEX=0` env var (emergency disable; matches the
    v1.2.x `AELFRICE_SEARCH_TOOL=0` convention).
  - `entity_index_enabled=False` kwarg on `retrieve()` /
    `retrieve_v2()`.

When the flag is off — for any reason — `retrieve()` reproduces the
v1.2 byte-identical L0 + L1 path with the v1.0 default budget of 2000
tokens. This is the spec's "default-off byte-identical fallback"
acceptance criterion.

NO HRR, NO BFS multi-hop in v1.3.0. Those land in subsequent v1.3.x
PRs once the L2.5 chain-valid lift is verified on MAB.

A `RetrievalCache` wrapper provides bounded LRU memoization. Cache
invalidation is wired through the store's callback registry, which
fires on every belief / edge / entity-row mutation (the entity rows
mutate inside `insert_belief` / `update_belief` / `delete_belief`,
so the existing callback semantics already cover them).
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Final
import sys
import tomllib

from aelfrice.entity_extractor import extract_entities
from aelfrice.models import LOCK_NONE, Belief
from aelfrice.store import MemoryStore

# v1.0 / v1.2 baseline. Used by the disabled-flag fallback so the
# byte-identical regression test sees the same budget the v1.2 caller
# would have seen.
LEGACY_TOKEN_BUDGET: Final[int] = 2000

# v1.3.0 expanded default. L2.5 fills against a sub-budget of 400 and
# L1 fills against the remaining 2000 — preserves the v1.0 L1
# behaviour byte-for-byte on queries where L2.5 returns nothing.
DEFAULT_TOKEN_BUDGET: Final[int] = 2400

_CHARS_PER_TOKEN: Final[float] = 4.0
DEFAULT_L1_LIMIT: Final[int] = 50

# v1.3.0 entity-index defaults (docs/entity_index.md § Budget split).
DEFAULT_L25_LIMIT: Final[int] = 20
DEFAULT_L25_TOKEN_SUBBUDGET: Final[int] = 400
DEFAULT_QUERY_ENTITY_CAP: Final[int] = 16

DEFAULT_CACHE_CAPACITY: Final[int] = 256

# Section / key names in `.aelfrice.toml`. Public so consumers can
# reference them in their own config.
CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
RETRIEVAL_SECTION: Final[str] = "retrieval"
ENTITY_INDEX_FLAG: Final[str] = "entity_index_enabled"

# Env var override. Set to "0", "false", or "no" to force-disable
# the index. Unset / any other value falls through to the TOML
# config (which defaults to True at v1.3.0). Same convention as the
# v1.2.x `AELFRICE_SEARCH_TOOL=0` off-switch.
ENV_ENTITY_INDEX: Final[str] = "AELFRICE_ENTITY_INDEX"
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})

_CANONICALIZE_PUNCT: Final[re.Pattern[str]] = re.compile(r"[^\w\s]")


def canonicalize_query(query: str) -> str:
    """Return a deterministic key for cache lookup.

    Lowercase, replace punctuation with whitespace, split on whitespace,
    sort tokens, rejoin with single spaces. Two queries that differ only
    in word order or punctuation map to the same key — correct for FTS5
    BM25, which is bag-of-words.
    """
    cleaned = _CANONICALIZE_PUNCT.sub(" ", query.lower()).strip()
    tokens = sorted(cleaned.split())
    return " ".join(tokens)


@dataclass(frozen=True)
class RetrievalResult:
    """Wrapper object for retrieve_v2 callers (academic-suite adapters).

    Public v1.0.x retrieve() returns list[Belief] directly. Lab v2.0.0
    adapters expect `result.beliefs` plus auxiliary diagnostics fields
    that aren't yet computed in public — those are placeholders here so
    adapter code that reads them does not crash.

    `entity_hits` (v1.3.0) exposes the L2.5 belief ids surfaced by the
    last call. The benchmark adapter consumes it for the L0/L1/L2.5
    counts surface; default `[]` for backwards-compat with adapters
    that only inspect `beliefs`.
    """

    beliefs: list[Belief]
    hrr_expansions: list[str] = field(default_factory=lambda: [])
    bfs_chains: list[list[str]] = field(default_factory=lambda: [])
    entity_hits: list[str] = field(default_factory=lambda: [])
    locked_ids: list[str] = field(default_factory=lambda: [])
    l1_ids: list[str] = field(default_factory=lambda: [])


def _estimate_tokens(text: str) -> int:
    """Cheap char-based token estimate. Conservative (rounds up)."""
    if not text:
        return 0
    return int((len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _belief_tokens(b: Belief) -> int:
    return _estimate_tokens(b.content)


# --- Config flag resolution ----------------------------------------------


def _env_disabled() -> bool:
    """Return True if AELFRICE_ENTITY_INDEX is set to a falsy value."""
    raw = os.environ.get(ENV_ENTITY_INDEX)
    if raw is None:
        return False
    return raw.strip().lower() in _ENV_FALSY


def _read_toml_flag(start: Path | None = None) -> bool | None:
    """Walk up from `start` looking for a `.aelfrice.toml` with
    `[retrieval] entity_index_enabled`. Returns the boolean value
    when found, or None when no file / no key.

    Tolerant: a malformed TOML or wrong-typed value returns None
    (let the default win) and traces to stderr without raising.
    Mirrors `noise_filter.NoiseConfig.discover` semantics.
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
                    f"aelfrice retrieval: cannot read {candidate}: {exc}",
                    file=serr,
                )
                return None
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    raw.decode("utf-8", errors="replace"),
                )
            except tomllib.TOMLDecodeError as exc:
                print(
                    f"aelfrice retrieval: malformed TOML in {candidate}: {exc}",
                    file=serr,
                )
                return None
            section_obj: Any = parsed.get(RETRIEVAL_SECTION, {})
            if not isinstance(section_obj, dict):
                return None
            if ENTITY_INDEX_FLAG not in section_obj:  # type: ignore[operator]
                return None
            value: Any = section_obj[ENTITY_INDEX_FLAG]  # type: ignore[index]
            if isinstance(value, bool):
                return value
            print(
                f"aelfrice retrieval: ignoring [{RETRIEVAL_SECTION}] "
                f"{ENTITY_INDEX_FLAG} in {candidate} (expected bool)",
                file=serr,
            )
            return None
        if current.parent == current:
            break
        current = current.parent
    return None


def is_entity_index_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the entity-index flag.

    Precedence (first decisive wins):
      1. AELFRICE_ENTITY_INDEX=0 (env override).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] entity_index_enabled` in `.aelfrice.toml`.
      4. Default: True (v1.3.0 default-on).
    """
    if _env_disabled():
        return False
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag(start)
    if toml_value is not None:
        return toml_value
    return True


# --- Retrieval -----------------------------------------------------------


def _l25_hits(
    store: MemoryStore,
    query: str,
    *,
    locked_ids: set[str],
    l25_limit: int,
    l25_token_subbudget: int,
    query_entity_cap: int,
) -> list[Belief]:
    """Run L2.5: query-side extraction, entity lookup, materialise
    beliefs, dedupe vs L0, trim to `l25_token_subbudget`.

    Returns at most `l25_limit` beliefs whose summed token estimate
    is at or below `l25_token_subbudget`. The trim is from the
    tail (lowest-overlap matches drop first).

    A `l25_token_subbudget <= 0` short-circuits to []. The outer
    `retrieve()` enforces that the L2.5 sub-budget never exceeds
    the remaining `token_budget`, so passing 0 is the correct
    expression of "no L2.5 budget left".
    """
    if l25_token_subbudget <= 0:
        return []
    q_entities = extract_entities(query, max_entities=query_entity_cap)
    if not q_entities:
        return []
    keys = [e.lower for e in q_entities]
    hits = store.lookup_entities(keys, limit=l25_limit)
    out: list[Belief] = []
    used = 0
    for bid, _overlap in hits:
        if bid in locked_ids:
            continue
        b = store.get_belief(bid)
        if b is None:
            # Race: belief was deleted between lookup and fetch.
            # Skip; the index will be cleaned up by the next mutation
            # cycle (delete_belief cascades to belief_entities).
            continue
        cost = _belief_tokens(b)
        if used + cost > l25_token_subbudget:
            break
        out.append(b)
        used += cost
    return out


def retrieve(
    store: MemoryStore,
    query: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    l1_limit: int = DEFAULT_L1_LIMIT,
    *,
    entity_index_enabled: bool | None = None,
    l25_limit: int = DEFAULT_L25_LIMIT,
    l25_token_subbudget: int = DEFAULT_L25_TOKEN_SUBBUDGET,
    query_entity_cap: int = DEFAULT_QUERY_ENTITY_CAP,
) -> list[Belief]:
    """Return L0 locked + L2.5 entity hits + L1 FTS5 BM25 results.

    Output is token-budgeted: L1 results are trimmed from the tail
    until the estimated total token count is at or below
    `token_budget`. L0 beliefs are never trimmed.

    L2.5 (v1.3.0): entity-index lookup. Default-on; gated by
    `is_entity_index_enabled()` (env override → kwarg → TOML →
    default True). When disabled the path collapses to v1.2's L0+L1
    behaviour byte-for-byte, with the legacy budget if the caller
    didn't pass an explicit one.

    Empty / whitespace-only query: returns L0 only (no L2.5, no L1).

    Dedupe: L1 hits whose id appears in L0 or L2.5 are dropped
    before budget accounting. L2.5 hits whose id appears in L0 are
    likewise dropped.
    """
    enabled = is_entity_index_enabled(entity_index_enabled)

    locked: list[Belief] = store.list_locked_beliefs()
    locked_ids: set[str] = {b.id for b in locked}

    # Backwards-compat: when the flag is off and the caller didn't
    # pass an explicit budget, snap back to the v1.2 default of
    # 2000. The byte-identical regression test relies on this.
    effective_budget = (
        token_budget
        if (enabled or token_budget != DEFAULT_TOKEN_BUDGET)
        else LEGACY_TOKEN_BUDGET
    )

    # L2.5 is bounded by both its own sub-budget AND the outer
    # remaining-after-L0 budget. This preserves the v1.0 invariant
    # that callers passing a tight `token_budget` never get more
    # tokens back than they asked for, while still letting the
    # default 2400-budget caller see the full 400-token L2.5 slice.
    locked_used: int = sum(_belief_tokens(b) for b in locked)
    l25_room: int = max(0, effective_budget - locked_used)
    effective_l25_subbudget: int = min(l25_token_subbudget, l25_room)

    l25: list[Belief]
    if enabled and query.strip():
        l25 = _l25_hits(
            store,
            query,
            locked_ids=locked_ids,
            l25_limit=l25_limit,
            l25_token_subbudget=effective_l25_subbudget,
            query_entity_cap=query_entity_cap,
        )
    else:
        l25 = []
    l25_ids: set[str] = {b.id for b in l25}

    l1: list[Belief] = []
    if query.strip():
        raw_l1: list[Belief] = store.search_beliefs(query, limit=l1_limit)
        l1 = [
            b for b in raw_l1
            if b.id not in locked_ids and b.id not in l25_ids
        ]

    # Token accounting. L0 always survives. L2.5 lands above L1 in
    # the output. L1 trims from the tail.
    used: int = locked_used + sum(_belief_tokens(b) for b in l25)
    out: list[Belief] = list(locked) + list(l25)
    for b in l1:
        cost: int = _belief_tokens(b)
        if used + cost > effective_budget:
            break
        out.append(b)
        used += cost
    return out


def retrieve_with_tiers(
    store: MemoryStore,
    query: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    l1_limit: int = DEFAULT_L1_LIMIT,
    *,
    entity_index_enabled: bool | None = None,
    l25_limit: int = DEFAULT_L25_LIMIT,
    l25_token_subbudget: int = DEFAULT_L25_TOKEN_SUBBUDGET,
    query_entity_cap: int = DEFAULT_QUERY_ENTITY_CAP,
) -> tuple[list[Belief], list[str], list[str], list[str]]:
    """Same logic as `retrieve()` but returns the per-tier id lists
    alongside the merged output.

    Used by the v1.3.0 benchmark adapter to surface L0 / L1 / L2.5
    counts in the per-question JSON without making a second call.
    Returns (merged_output, locked_ids, l25_ids, l1_ids).
    """
    enabled = is_entity_index_enabled(entity_index_enabled)

    locked: list[Belief] = store.list_locked_beliefs()
    locked_ids_list: list[str] = [b.id for b in locked]
    locked_ids: set[str] = set(locked_ids_list)

    effective_budget = (
        token_budget
        if (enabled or token_budget != DEFAULT_TOKEN_BUDGET)
        else LEGACY_TOKEN_BUDGET
    )
    locked_used: int = sum(_belief_tokens(b) for b in locked)
    l25_room: int = max(0, effective_budget - locked_used)
    effective_l25_subbudget: int = min(l25_token_subbudget, l25_room)

    if enabled and query.strip():
        l25 = _l25_hits(
            store,
            query,
            locked_ids=locked_ids,
            l25_limit=l25_limit,
            l25_token_subbudget=effective_l25_subbudget,
            query_entity_cap=query_entity_cap,
        )
    else:
        l25 = []
    l25_ids_list: list[str] = [b.id for b in l25]
    l25_ids: set[str] = set(l25_ids_list)

    l1: list[Belief] = []
    if query.strip():
        raw_l1: list[Belief] = store.search_beliefs(query, limit=l1_limit)
        l1 = [
            b for b in raw_l1
            if b.id not in locked_ids and b.id not in l25_ids
        ]

    used: int = locked_used + sum(_belief_tokens(b) for b in l25)
    out: list[Belief] = list(locked) + list(l25)
    l1_ids_list: list[str] = []
    for b in l1:
        cost: int = _belief_tokens(b)
        if used + cost > effective_budget:
            break
        out.append(b)
        used += cost
        l1_ids_list.append(b.id)
    return out, locked_ids_list, l25_ids_list, l1_ids_list


def retrieve_v2(
    store: MemoryStore,
    query: str,
    budget: int = DEFAULT_TOKEN_BUDGET,
    include_locked: bool = True,
    use_hrr: bool = False,  # noqa: ARG001
    use_bfs: bool = False,  # noqa: ARG001
    use_entity_index: bool | None = None,
    l1_limit: int = DEFAULT_L1_LIMIT,
) -> RetrievalResult:
    """Lab-compatible retrieval wrapper for academic-suite adapters.

    Wraps the public `retrieve()` in the signature lab v2.0.0 adapters
    expect:

    - `budget` (lab kwarg) maps to `token_budget` (public kwarg).
    - `include_locked=False` filters out lock_level != LOCK_NONE post-retrieval
      (public always returns L0 first; this wrapper drops them on demand).
    - `use_hrr` and `use_bfs` are accepted but no-op at v1.3.0 — the HRR
      vocabulary bridge and BFS multi-hop chaining have not yet
      ported. Callers can pass them for forward-compat without conditionals.
    - `use_entity_index` (v1.3.0) maps to `retrieve()`'s
      `entity_index_enabled` kwarg. None falls through to the default
      (env / TOML / True). The v1.3.0 benchmark adapter sets it
      explicitly.
    - Returns a `RetrievalResult` wrapper so adapters can read
      `result.beliefs` (and stub diagnostics fields, plus the new
      v1.3 `entity_hits`).
    """
    out, locked_ids_list, l25_ids_list, l1_ids_list = retrieve_with_tiers(
        store, query,
        token_budget=budget,
        entity_index_enabled=use_entity_index,
        l1_limit=l1_limit,
    )
    if include_locked:
        beliefs = out
    else:
        beliefs = [b for b in out if b.lock_level == LOCK_NONE]
    return RetrievalResult(
        beliefs=beliefs,
        entity_hits=l25_ids_list,
        locked_ids=locked_ids_list,
        l1_ids=l1_ids_list,
    )


class RetrievalCache:
    """Bounded LRU cache wrapping `retrieve()` for an attached store.

    Subscribes to the store's invalidation callback registry on
    construction, so any belief / edge / entity-row mutation wipes
    the cache. Per-instance: two `RetrievalCache` objects pointing
    at different stores never share state.

    Cache key includes the entity-index flag (v1.3.0). Two queries
    that differ only in `entity_index_enabled` are distinct entries.
    """

    def __init__(
        self,
        store: MemoryStore,
        capacity: int = DEFAULT_CACHE_CAPACITY,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._store = store
        self._capacity = capacity
        self._entries: OrderedDict[
            tuple[str, int, int, bool | None], list[Belief]
        ] = OrderedDict()
        store.add_invalidation_callback(self.invalidate)

    def retrieve(
        self,
        query: str,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        l1_limit: int = DEFAULT_L1_LIMIT,
        *,
        entity_index_enabled: bool | None = None,
    ) -> list[Belief]:
        """Cached `retrieve()`. Identical contract to the free function."""
        key = (
            canonicalize_query(query),
            token_budget,
            l1_limit,
            entity_index_enabled,
        )
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return list(cached)
        result = retrieve(
            self._store, query,
            token_budget=token_budget, l1_limit=l1_limit,
            entity_index_enabled=entity_index_enabled,
        )
        self._entries[key] = list(result)
        if len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return result

    def invalidate(self) -> None:
        """Drop every cached entry. Wired to the store's mutation hook."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
