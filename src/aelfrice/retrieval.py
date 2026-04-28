"""Four-layer retrieval: L0 locked beliefs, L2.5 entity-index, L1 FTS5
BM25, L3 BFS multi-hop graph traversal.

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

L3 (v1.3.0) is `aelfrice.bfs_multihop.expand_bfs` — edge-type-weighted
BFS over outbound edges from the L0+L2.5+L1 seed set. Bounded depth /
fanout / total-budget; multiplicative path-score over the table in
`bfs_multihop.BFS_EDGE_WEIGHTS`. Default-OFF at v1.3.0; opt in via the
`bfs_enabled` flag in `[retrieval]` of `.aelfrice.toml`, the
`AELFRICE_BFS=1` env var, or an explicit kwarg. See
`docs/bfs_multihop.md` for the spec.

Default-on at v1.3.0 via the config flag `entity_index_enabled` in
`[retrieval]` of `.aelfrice.toml`. Two off-switches:

  - `AELFRICE_ENTITY_INDEX=0` env var (emergency disable; matches the
    v1.2.x `AELFRICE_SEARCH_TOOL=0` convention).
  - `entity_index_enabled=False` kwarg on `retrieve()` /
    `retrieve_v2()`.

When BOTH flags are off — for any reason — `retrieve()` reproduces the
v1.2 byte-identical L0 + L1 path with the v1.0 default budget of 2000
tokens. When only `bfs_enabled` is off (the v1.3.0 default) and L2.5
is on (also default), `retrieve()` is byte-identical to the entity-
index-enabled v1.3.0 baseline. Both invariants are guarded by
regression tests.

NO HRR in v1.3.0. That lands at v2.0.0.

A `RetrievalCache` wrapper provides bounded LRU memoization. Cache
invalidation is wired through the store's callback registry, which
fires on every belief / edge / entity-row mutation (the entity rows
mutate inside `insert_belief` / `update_belief` / `delete_belief`,
so the existing callback semantics already cover them). The v1.0.1
wipe-on-write policy on edge mutators (`insert_edge`, `update_edge`,
`delete_edge`) is exactly what makes the v1.3 BFS cache correctness
zero-effort — see docs/bfs_multihop.md § Cache invalidation.
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

from aelfrice.bfs_multihop import (
    DEFAULT_MAX_DEPTH as BFS_DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_PATH_SCORE as BFS_DEFAULT_MIN_PATH_SCORE,
    DEFAULT_NODES_PER_HOP as BFS_DEFAULT_NODES_PER_HOP,
    DEFAULT_TOTAL_BUDGET_NODES as BFS_DEFAULT_TOTAL_BUDGET_NODES,
    expand_bfs,
)
from aelfrice.entity_extractor import extract_entities
from aelfrice.models import LOCK_NONE, Belief
from aelfrice.scoring import (
    DEFAULT_POSTERIOR_WEIGHT,
    partial_bayesian_score,
)
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
BFS_FLAG: Final[str] = "bfs_enabled"
POSTERIOR_WEIGHT_FLAG: Final[str] = "posterior_weight"

# Env var override. Set to "0", "false", or "no" to force-disable
# the index. Unset / any other value falls through to the TOML
# config (which defaults to True at v1.3.0). Same convention as the
# v1.2.x `AELFRICE_SEARCH_TOOL=0` off-switch.
ENV_ENTITY_INDEX: Final[str] = "AELFRICE_ENTITY_INDEX"
# BFS env override. Symmetric to ENV_ENTITY_INDEX but with default
# OFF at v1.3.0 — set to "1", "true", "yes", "on" to opt in. The
# default-off contract means the env-var omission is the same as
# the explicit-off case.
ENV_BFS: Final[str] = "AELFRICE_BFS"
# v1.3.0 posterior-weight env override. Float-typed; "0.0" is the
# only value that fully disables (collapsing to BM25-only ordering).
# Empty / non-numeric values fall through to the next precedence
# layer (kwarg → TOML → DEFAULT_POSTERIOR_WEIGHT) and trace to
# stderr. Same shape as `_read_toml_flag_for` tolerance.
ENV_POSTERIOR_WEIGHT: Final[str] = "AELFRICE_POSTERIOR_WEIGHT"
# Number of decimal places used to round `posterior_weight` before
# inclusion in the cache key. Two callers passing weights that
# differ by less than this granularity collapse to the same key.
POSTERIOR_WEIGHT_KEY_PRECISION: Final[int] = 4
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

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


def _env_bfs_override() -> bool | None:
    """Return True/False if AELFRICE_BFS is set to a recognised
    truthy/falsy value, else None.

    Symmetric to `_env_disabled` but tri-state because the BFS flag
    ships default-OFF at v1.3.0 — an unset env var is "fall through
    to the next precedence layer", not "force off". The config-flag
    semantics for BFS are: env > kwarg > TOML > False.
    """
    raw = os.environ.get(ENV_BFS)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _read_toml_flag_for(
    key: str,
    start: Path | None = None,
) -> bool | None:
    """Walk up from `start` looking for a `.aelfrice.toml` with
    `[retrieval] <key>`. Returns the boolean value when found, or
    None when no file / no key.

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
            if key not in section_obj:  # type: ignore[operator]
                return None
            value: Any = section_obj[key]  # type: ignore[index]
            if isinstance(value, bool):
                return value
            print(
                f"aelfrice retrieval: ignoring [{RETRIEVAL_SECTION}] "
                f"{key} in {candidate} (expected bool)",
                file=serr,
            )
            return None
        if current.parent == current:
            break
        current = current.parent
    return None


def _read_toml_float_for(
    key: str,
    start: Path | None = None,
) -> float | None:
    """Walk up from `start` looking for a `.aelfrice.toml` with
    `[retrieval] <key>` typed as int or float. Returns the float
    value when found, or None when no file / no key.

    Tolerant: a malformed TOML or wrong-typed value returns None
    and traces to stderr without raising. Mirrors
    `_read_toml_flag_for` semantics but accepts numeric types.
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
            if key not in section_obj:  # type: ignore[operator]
                return None
            value: Any = section_obj[key]  # type: ignore[index]
            # bool is a subclass of int -- reject it explicitly so
            # `posterior_weight = true` reads as malformed rather
            # than silently coercing to 1.0.
            if isinstance(value, bool):
                print(
                    f"aelfrice retrieval: ignoring [{RETRIEVAL_SECTION}] "
                    f"{key} in {candidate} (expected number, got bool)",
                    file=serr,
                )
                return None
            if isinstance(value, (int, float)):
                return float(value)
            print(
                f"aelfrice retrieval: ignoring [{RETRIEVAL_SECTION}] "
                f"{key} in {candidate} (expected number)",
                file=serr,
            )
            return None
        if current.parent == current:
            break
        current = current.parent
    return None


def _env_posterior_weight() -> float | None:
    """Return the AELFRICE_POSTERIOR_WEIGHT env value as a float,
    or None when unset / non-numeric.

    Non-numeric values trace to stderr and fall through (same
    fail-soft contract as the TOML readers).
    """
    raw = os.environ.get(ENV_POSTERIOR_WEIGHT)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        print(
            f"aelfrice retrieval: ignoring {ENV_POSTERIOR_WEIGHT}={raw!r} "
            f"(expected float)",
            file=sys.stderr,
        )
        return None


def resolve_posterior_weight(
    explicit: float | None = None,
    *,
    start: Path | None = None,
) -> float:
    """Resolve the posterior weight per v1.3 precedence:

      1. AELFRICE_POSTERIOR_WEIGHT env var (float, including 0.0).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] posterior_weight` in `.aelfrice.toml`.
      4. Default: DEFAULT_POSTERIOR_WEIGHT (0.5 at v1.3.0).

    A weight of `0.0` is treated as "BM25-only" (the byte-identical-
    with-v1.0.x ordering case); negative weights are clamped to
    0.0 since the spec defines the contract for weight ≥ 0 only.
    """
    env = _env_posterior_weight()
    if env is not None:
        weight = env
    elif explicit is not None:
        weight = float(explicit)
    else:
        toml_value = _read_toml_float_for(POSTERIOR_WEIGHT_FLAG, start)
        weight = float(toml_value) if toml_value is not None else (
            DEFAULT_POSTERIOR_WEIGHT
        )
    if weight < 0.0:
        return 0.0
    return weight


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
    toml_value = _read_toml_flag_for(ENTITY_INDEX_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def is_bfs_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the BFS multi-hop flag.

    Precedence (first decisive wins):
      1. AELFRICE_BFS env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] bfs_enabled` in `.aelfrice.toml`.
      4. Default: False (v1.3.0 default-OFF).

    The default-off contract is part of the v1.3.0 acceptance
    criteria: a fresh install must not change retrieval output
    against the v1.2 baseline.
    """
    env = _env_bfs_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(BFS_FLAG, start)
    if toml_value is not None:
        return toml_value
    return False


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


def _l1_hits(
    store: MemoryStore,
    query: str,
    *,
    l1_limit: int,
    posterior_weight: float,
) -> list[Belief]:
    """Run L1: FTS5 BM25 search, optionally reranked by partial-
    Bayesian score.

    `posterior_weight = 0.0` short-circuits to the v1.0.x path —
    `store.search_beliefs(query, limit)` returns rows already
    ordered by `bm25(beliefs_fts)` ascending, and we discard the
    score. This guarantees byte-identical ordering with the v1.0
    ranker.

    `posterior_weight > 0` swaps in the scored variant and re-
    sorts by `partial_bayesian_score(...)` descending. The
    underlying SQL ORDER BY keeps the BM25 prefilter deterministic
    in the truncation case (rare-but-possible at small `l1_limit`).
    Tie-break on belief id ASC so result lists are reproducible.
    """
    if posterior_weight == 0.0:
        return store.search_beliefs(query, limit=l1_limit)
    scored = store.search_beliefs_scored(query, limit=l1_limit)
    if not scored:
        return []
    keyed: list[tuple[float, str, Belief]] = []
    for b, bm25_raw in scored:
        s = partial_bayesian_score(
            bm25_raw, b.alpha, b.beta, posterior_weight,
        )
        keyed.append((s, b.id, b))
    # Higher score = more relevant. Tie-break on id ASC for
    # determinism (matches the convention in bfs_multihop and L2.5).
    keyed.sort(key=lambda x: (-x[0], x[1]))
    return [b for _, _, b in keyed]


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
    bfs_enabled: bool | None = None,
    bfs_max_depth: int = BFS_DEFAULT_MAX_DEPTH,
    bfs_nodes_per_hop: int = BFS_DEFAULT_NODES_PER_HOP,
    bfs_total_budget_nodes: int = BFS_DEFAULT_TOTAL_BUDGET_NODES,
    bfs_min_path_score: float = BFS_DEFAULT_MIN_PATH_SCORE,
    posterior_weight: float | None = None,
) -> list[Belief]:
    """Return L0 locked + L2.5 entity + L1 BM25 + L3 BFS expansions.

    Output is token-budgeted: results are trimmed from the tail
    until the estimated total token count is at or below
    `token_budget`. L0 beliefs are never trimmed.

    L2.5 (v1.3.0): entity-index lookup. Default-on; gated by
    `is_entity_index_enabled()` (env override → kwarg → TOML →
    default True). When disabled the path collapses to v1.2's L0+L1
    behaviour byte-for-byte, with the legacy budget if the caller
    didn't pass an explicit one.

    L3 (v1.3.0): BFS multi-hop expansion. Default-OFF; gated by
    `is_bfs_enabled()` (env override → kwarg → TOML → default
    False). Seeds are the L0+L2.5+L1 set that survived the prior
    tiers' filtering. Expansions are appended in score-descending
    order until the shared token budget is exhausted. When
    disabled, output is byte-identical to the L0+L2.5+L1 path.

    `posterior_weight` (v1.3.0): float ≥ 0. Combines the L1 BM25
    score with the Beta-Bernoulli posterior_mean log-additively:
    `score = log(-bm25) + posterior_weight * log(posterior_mean)`.
    `0.0` collapses to v1.0.x BM25-only ordering (byte-identical
    regression-tested). Default `0.5` per docs/bayesian_ranking.md
    § Defaults; resolved via `resolve_posterior_weight()` (env →
    kwarg → TOML → 0.5). L0 locks bypass the score entirely; L2.5
    and L3 are unaffected.

    Empty / whitespace-only query: returns L0 only (no L2.5, L1, or
    L3).

    Dedupe: L1 hits whose id appears in L0 or L2.5 are dropped
    before budget accounting. L2.5 hits whose id appears in L0 are
    likewise dropped. L3 expansions whose id appears in any prior
    tier are dropped (the visited-set in `expand_bfs` prevents
    seeds from being re-surfaced; we additionally guard against
    overlap with L1 hits the seeds didn't include).
    """
    enabled = is_entity_index_enabled(entity_index_enabled)
    bfs_on = is_bfs_enabled(bfs_enabled)
    weight = resolve_posterior_weight(posterior_weight)

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
        raw_l1: list[Belief] = _l1_hits(
            store, query,
            l1_limit=l1_limit, posterior_weight=weight,
        )
        l1 = [
            b for b in raw_l1
            if b.id not in locked_ids and b.id not in l25_ids
        ]

    # Token accounting. L0 always survives. L2.5 lands above L1 in
    # the output. L1 trims from the tail.
    used: int = locked_used + sum(_belief_tokens(b) for b in l25)
    out: list[Belief] = list(locked) + list(l25)
    l1_packed: list[Belief] = []
    for b in l1:
        cost: int = _belief_tokens(b)
        if used + cost > effective_budget:
            break
        out.append(b)
        l1_packed.append(b)
        used += cost

    # L3 BFS expansion. Default-off — `bfs_on` False short-circuits
    # before any graph work, preserving byte-identical output.
    if bfs_on and query.strip():
        seeds: list[Belief] = list(locked) + list(l25) + list(l1_packed)
        if seeds:
            hops = expand_bfs(
                seeds,
                store,
                max_depth=bfs_max_depth,
                nodes_per_hop=bfs_nodes_per_hop,
                total_budget=bfs_total_budget_nodes,
                min_path_score=bfs_min_path_score,
            )
            seen_ids: set[str] = (
                locked_ids | l25_ids | {b.id for b in l1_packed}
            )
            for hop in hops:
                if hop.belief.id in seen_ids:
                    continue
                cost = _belief_tokens(hop.belief)
                if used + cost > effective_budget:
                    break
                out.append(hop.belief)
                seen_ids.add(hop.belief.id)
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
    bfs_enabled: bool | None = None,
    bfs_max_depth: int = BFS_DEFAULT_MAX_DEPTH,
    bfs_nodes_per_hop: int = BFS_DEFAULT_NODES_PER_HOP,
    bfs_total_budget_nodes: int = BFS_DEFAULT_TOTAL_BUDGET_NODES,
    bfs_min_path_score: float = BFS_DEFAULT_MIN_PATH_SCORE,
    posterior_weight: float | None = None,
) -> tuple[
    list[Belief], list[str], list[str], list[str], list[list[str]],
]:
    """Same logic as `retrieve()` but returns the per-tier id lists
    alongside the merged output.

    Used by the v1.3.0 benchmark adapter to surface L0 / L1 / L2.5
    counts in the per-question JSON without making a second call.
    Returns
    `(merged_output, locked_ids, l25_ids, l1_ids, bfs_chains)`.
    `bfs_chains[i]` is the edge-type path that reached the i-th
    L3-tier expansion belief in `merged_output` (empty list when
    BFS is off / produced nothing / bfs hits collide with prior
    tiers).
    """
    enabled = is_entity_index_enabled(entity_index_enabled)
    bfs_on = is_bfs_enabled(bfs_enabled)
    weight = resolve_posterior_weight(posterior_weight)

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
        raw_l1: list[Belief] = _l1_hits(
            store, query,
            l1_limit=l1_limit, posterior_weight=weight,
        )
        l1 = [
            b for b in raw_l1
            if b.id not in locked_ids and b.id not in l25_ids
        ]

    used: int = locked_used + sum(_belief_tokens(b) for b in l25)
    out: list[Belief] = list(locked) + list(l25)
    l1_ids_list: list[str] = []
    l1_packed: list[Belief] = []
    for b in l1:
        cost: int = _belief_tokens(b)
        if used + cost > effective_budget:
            break
        out.append(b)
        l1_packed.append(b)
        used += cost
        l1_ids_list.append(b.id)

    bfs_chains: list[list[str]] = []
    if bfs_on and query.strip():
        seeds: list[Belief] = list(locked) + list(l25) + list(l1_packed)
        if seeds:
            hops = expand_bfs(
                seeds,
                store,
                max_depth=bfs_max_depth,
                nodes_per_hop=bfs_nodes_per_hop,
                total_budget=bfs_total_budget_nodes,
                min_path_score=bfs_min_path_score,
            )
            seen_ids: set[str] = (
                locked_ids | l25_ids | set(l1_ids_list)
            )
            for hop in hops:
                if hop.belief.id in seen_ids:
                    continue
                cost = _belief_tokens(hop.belief)
                if used + cost > effective_budget:
                    break
                out.append(hop.belief)
                bfs_chains.append(list(hop.path))
                seen_ids.add(hop.belief.id)
                used += cost
    return out, locked_ids_list, l25_ids_list, l1_ids_list, bfs_chains


def retrieve_v2(
    store: MemoryStore,
    query: str,
    budget: int = DEFAULT_TOKEN_BUDGET,
    include_locked: bool = True,
    use_hrr: bool = False,  # noqa: ARG001
    use_bfs: bool | None = None,
    use_entity_index: bool | None = None,
    l1_limit: int = DEFAULT_L1_LIMIT,
    bfs_max_depth: int = BFS_DEFAULT_MAX_DEPTH,
    bfs_nodes_per_hop: int = BFS_DEFAULT_NODES_PER_HOP,
    bfs_total_budget_nodes: int = BFS_DEFAULT_TOTAL_BUDGET_NODES,
    bfs_min_path_score: float = BFS_DEFAULT_MIN_PATH_SCORE,
    posterior_weight: float | None = None,
) -> RetrievalResult:
    """Lab-compatible retrieval wrapper for academic-suite adapters.

    Wraps the public `retrieve()` in the signature lab v2.0.0 adapters
    expect:

    - `budget` (lab kwarg) maps to `token_budget` (public kwarg).
    - `include_locked=False` filters out lock_level != LOCK_NONE post-retrieval
      (public always returns L0 first; this wrapper drops them on demand).
    - `use_hrr` is accepted but no-op at v1.3.0 — the HRR vocabulary
      bridge has not yet ported. Callers can pass it for forward-compat
      without conditionals.
    - `use_bfs` (v1.3.0) maps to `retrieve()`'s `bfs_enabled` kwarg.
      None falls through to the default-OFF resolution (env / TOML
      / False at v1.3.0). Setting it True opts a single retrieve_v2
      call into BFS regardless of process-wide config.
    - `use_entity_index` (v1.3.0) maps to `retrieve()`'s
      `entity_index_enabled` kwarg. None falls through to the default
      (env / TOML / True). The v1.3.0 benchmark adapter sets it
      explicitly.
    - `bfs_max_depth`, `bfs_nodes_per_hop`, `bfs_total_budget_nodes`,
      `bfs_min_path_score` — pass-through tuning knobs for the L3
      tier. Defaults match `bfs_multihop.DEFAULT_*`.
    - Returns a `RetrievalResult` wrapper so adapters can read
      `result.beliefs` (and stub diagnostics fields, plus the new
      v1.3 `entity_hits` and `bfs_chains`).
    """
    (
        out,
        locked_ids_list,
        l25_ids_list,
        l1_ids_list,
        bfs_chains,
    ) = retrieve_with_tiers(
        store, query,
        token_budget=budget,
        entity_index_enabled=use_entity_index,
        l1_limit=l1_limit,
        bfs_enabled=use_bfs,
        bfs_max_depth=bfs_max_depth,
        bfs_nodes_per_hop=bfs_nodes_per_hop,
        bfs_total_budget_nodes=bfs_total_budget_nodes,
        bfs_min_path_score=bfs_min_path_score,
        posterior_weight=posterior_weight,
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
        bfs_chains=bfs_chains,
    )


class RetrievalCache:
    """Bounded LRU cache wrapping `retrieve()` for an attached store.

    Subscribes to the store's invalidation callback registry on
    construction, so any belief / edge / entity-row mutation wipes
    the cache. Per-instance: two `RetrievalCache` objects pointing
    at different stores never share state.

    Cache key includes the entity-index flag (v1.3.0 default-on),
    the BFS flag (v1.3.0 default-off), and `posterior_weight`
    (v1.3.0 default 0.5, rounded to `POSTERIOR_WEIGHT_KEY_PRECISION`
    decimals so floating-point jitter does not fragment the cache).
    Two queries that differ in any of these are distinct entries.
    BFS knobs (`bfs_max_depth` etc.) are NOT in the key — per
    docs/bfs_multihop.md § Cache invalidation, callers that toggle
    them per call would defeat the cache anyway.

    The `posterior_weight` cache-key extension is a structural fix
    against cross-caller collisions per docs/bayesian_ranking.md §
    "Cache invalidation". Posterior-write staleness is handled by
    the existing store-mutation callback (apply_feedback ->
    update_belief -> _fire_invalidation -> cache wipe).
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
            tuple[
                str, int, int, bool | None, bool | None, float | None,
            ],
            list[Belief],
        ] = OrderedDict()
        store.add_invalidation_callback(self.invalidate)

    def retrieve(
        self,
        query: str,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        l1_limit: int = DEFAULT_L1_LIMIT,
        *,
        entity_index_enabled: bool | None = None,
        bfs_enabled: bool | None = None,
        posterior_weight: float | None = None,
    ) -> list[Belief]:
        """Cached `retrieve()`. Identical contract to the free function.

        Cache key keeps `posterior_weight` in its caller-supplied
        form (None or a float) — `None` is its own bucket and
        deferred env / TOML resolution happens once on the miss
        path. Resolving on every hit would walk Path.cwd().resolve()
        each time and blow the AC2 cache-hit latency budget.
        """
        if posterior_weight is None:
            key_weight: float | None = None
        else:
            key_weight = round(
                float(posterior_weight),
                POSTERIOR_WEIGHT_KEY_PRECISION,
            )
        key = (
            canonicalize_query(query),
            token_budget,
            l1_limit,
            entity_index_enabled,
            bfs_enabled,
            key_weight,
        )
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return list(cached)
        result = retrieve(
            self._store, query,
            token_budget=token_budget, l1_limit=l1_limit,
            entity_index_enabled=entity_index_enabled,
            bfs_enabled=bfs_enabled,
            posterior_weight=posterior_weight,
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
