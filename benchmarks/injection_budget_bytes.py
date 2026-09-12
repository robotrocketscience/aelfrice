"""#1526 — what the corrected per-belief accounting does to each injection lane.

#1526 changed what the injection packers charge: from the belief's *content* to
the *line the renderer actually emits*. Every budget constant is unchanged, so
the effect of the change is a change in what each budget buys. This module
measures that effect, per lane, and is the producer for every #1526 figure the
repo publishes.

## What it measures

Seven lanes, each at its own shipped `(budget, l1_limit)` and through its own
renderer:

* `ups` — the per-turn `<aelfrice-memory>` block (`hook.DEFAULT_HOOK_TOKEN_BUDGET`).
* `core` — the first-prompt `<core>` section
  (`hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET`).
* `session_start` — the `<aelfrice-baseline>` block
  (`hook.DEFAULT_SESSION_START_TOKEN_BUDGET`).
* `search_tool` / `search_tool_bash` — the PreToolUse Grep|Glob and Bash lanes
  (`hook_search_tool.INJECTED_TOKEN_BUDGET` / `BASH_INJECTED_TOKEN_BUDGET`).
* `agent_context` — the Agent/Task worker-context lane
  (`hook_agent_context.INJECTED_TOKEN_BUDGET`).
* `cli_search` — `aelf search` (`retrieval.DEFAULT_TOKEN_BUDGET`).

The *before* arm rebinds the pre-#1526 cost functions; the *after* arm runs the
shipped ones. **Both arms run at the same, unchanged budget constant**, so the
difference between them is the accounting change and nothing else. Both render
through the shipped renderer and the counts are of rendered bytes.

## Saturation: what is suppressed, and what that suppression means

A budget that ends no pack is not evidence about that budget. This module
checks that by re-running an arm with **every budget that can end its pack**
raised by `SATURATION_PROBE_FACTOR` — `token_budget` *and*
`l25_token_subbudget`, because the L2.5 sub-pack has its own cap and a pack
ended by that cap does not move when `token_budget` alone is raised.

Round 2 of #1526 found exactly that error, and it reproduces on the shipped
tree. On `ups` at 92 content characters, against a store of 300 beliefs, the
pack returns 58 hits at `token_budget=6000` and 58 at 1,500,000 with the
sub-budget left at its default 400 — and 60 at `token_budget=6000` with the
sub-budget at 1600. A probe that varied `token_budget` alone read that as a
budget binding on nothing, and suppressed the cell.

Each arm records which cap ended it as `before_binds_on` / `after_binds_on`,
and **a byte count is printed either way**. An arm that ended on the candidate
pool is weak evidence about the budget; its bytes are still the bytes the model
receives, and the earlier version of this module suppressed whole cells whose
deviation ran to 53%.

Only one case is genuinely uninformative: **both** arms ending on the pool,
where the two arms return the same candidates and their equality is an equality
of pools. Those cells are marked `pool_equality: true`. Their bytes are still
reported.

`session_start` is `pool_equality` at every length, and that is the finding
rather than a gap in the measurement. `hook.session_start` retrieves with an
**empty query**, and in `retrieve_with_tiers` every relevance lane is gated on
`query.strip()`; only L0 contributes, and L0 is never trimmed by the budget
(#379). `DEFAULT_SESSION_START_TOKEN_BUDGET` therefore cannot bind on the lane
it governs, at any value. This module renders that lane with the empty query it
actually issues rather than with `QUERY`, so the reader sees it.

## The effect is length-dependent, and this module reports the whole curve

What the correction removes from a pack is the ratio of a rendered line to its
content: largest where beliefs are shortest, shrinking towards parity as they
grow. On the `search_tool` lanes it changes sign, because those lanes truncate
to `PER_LINE_CHAR_CAP` and past that cap the old accounting charged *more* than
the lane emits. So there is no single multiplier for this change, and any
figure taken from it is only readable beside the belief length it was measured
at. `--curve` prints the whole grid.

Usage:

    uv run python benchmarks/injection_budget_bytes.py
    uv run python benchmarks/injection_budget_bytes.py --curve
    uv run python benchmarks/injection_budget_bytes.py --emit-figures
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))


# Beliefs in the synthetic store. 300 is well past every lane's L1 cap, so the
# candidate pool is never the first thing that ends a pack.
STORE_BELIEFS = 300

# Every `LOCK_EVERY`-th belief is a user lock, and every `SPECULATIVE_EVERY`-th
# is speculative-origin. Both arms this change rewrote — `lock_injection_tokens`
# and the `speculative="1"` attribute width — are dead code against a corpus of
# plain agent-inferred beliefs, so the corpus carries both. The lock share is
# small on purpose: L0 is never trimmed, so a store that is mostly locks spends
# the whole budget before the pack loop is reached and measures nothing.
LOCK_EVERY = 50
SPECULATIVE_EVERY = 5

# Content lengths the curve is reported at, in characters. 92 is the committed
# `tests/corpus/replay_soak` corpus's median belief length. 200 is
# `hook_search_tool.PER_LINE_CHAR_CAP`, where the truncation arm starts to
# dominate; 300 is past it.
CORPUS_MEDIAN_CHARS = 92
LENGTH_GRID: tuple[int, ...] = (40, 92, 150, 200, 300)

# The saturation probe re-runs an arm with every pack-ending budget raised by
# this factor. Both `token_budget` and `l25_token_subbudget` are raised: the
# L2.5 sub-pack has its own cap, and raising `token_budget` alone leaves a pack
# that ended on that cap unchanged, which reads as "the budget does not bind".
SATURATION_PROBE_FACTOR = 4

# A query broad enough that the L1 candidate cap, not the query, decides how
# many beliefs reach the pack. Fixed text so the measurement is reproducible.
QUERY = (
    "belief retrieval store commit scanner memory python file git hook "
    "MemoryStore EntityIndex"
)

# Vocabulary the synthetic beliefs are drawn from. Every word appears in QUERY
# or is a near neighbour of one, so BM25 ranks the whole store and the L1 cap
# is what bounds the candidate set.
_VOCAB: tuple[str, ...] = tuple(
    (
        "belief retrieval store commit scanner memory python file git hook "
        "budget render packer token session prompt injection lane corpus "
        "index entity cluster lock manifest baseline worker context"
    ).split()
)

# Identifier-shaped tokens, one per belief, cycled. Without them the entity
# index extracts nothing from a bag of lowercase words, L2.5 returns empty and
# the candidate pool collapses to `l1_limit` alone -- which on the two
# PreToolUse lanes (l1_limit 10 and 5) is small enough that no budget can bind
# on it. Measured: at 92 content chars the Grep|Glob lane's pool is 10 beliefs
# without these and 18 with them. A production store carries identifiers; a
# corpus that does not would silently measure the L1 cap instead of the budget.
_ENTITIES: tuple[str, ...] = tuple(
    (
        "MemoryStore RetrievalLane BeliefPacker HookSearchTool ContextRebuilder "
        "src/aelfrice/retrieval.py tests/test_hook.py BudgetTuner LaneSpec "
        "PerLineCap TokenBudget SessionRing ReplaySoak EntityIndex"
    ).split()
)


def _synthetic_store(path: Path, content_chars: int, *, seed: int = 1526) -> Any:
    """A store of `STORE_BELIEFS` beliefs, each `content_chars` long.

    Seeded, so the same length always produces the same store. Every belief
    carries alpha=5/beta=1: that clears both `<core>` gates (alpha+beta >= 4,
    mean 0.833 >= 2/3) and renders a five-character `posterior` attribute,
    which is the common case for that attribute's width. A freshly derived
    belief carries alpha=beta=1 and would render an empty `<core>`.

    Every `LOCK_EVERY`-th belief is a user lock and every
    `SPECULATIVE_EVERY`-th is speculative-origin, so the two per-belief arms
    #1526 rewrote for those cases (`retrieval.lock_injection_tokens` and the
    `speculative="1"` attribute in `render_cost.belief_line_chars`) are
    exercised by the published figures rather than by the unit tests alone.
    """
    from aelfrice.models import (
        LOCK_NONE,
        LOCK_USER,
        ORIGIN_SPECULATIVE,
        Belief,
    )
    from aelfrice.store import MemoryStore

    rng = random.Random(seed)
    store = MemoryStore(str(path))
    for i in range(STORE_BELIEFS):
        # A per-belief unique token first: `beliefs.content_hash` is UNIQUE,
        # and at 40 characters a shared prefix plus a small vocabulary
        # collides. Then one identifier, so L2.5 has something to index.
        words: list[str] = [f"b{i:04d}", _ENTITIES[i % len(_ENTITIES)]]
        while len(" ".join(words)) < content_chars:
            words.append(rng.choice(_VOCAB))
        content = " ".join(words)[:content_chars]
        bid = hashlib.sha256(f"{i}:{seed}".encode()).hexdigest()[:16]
        locked = i % LOCK_EVERY == 0
        speculative = not locked and i % SPECULATIVE_EVERY == 0
        store.insert_belief(
            Belief(
                id=bid,
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                alpha=5.0,
                beta=1.0,
                type="factual",
                lock_level=LOCK_USER if locked else LOCK_NONE,
                locked_at="2026-01-01T00:00:00Z" if locked else None,
                created_at="2026-01-01T00:00:00Z",
                last_retrieved_at=None,
                origin=(
                    "user_asserted"
                    if locked
                    else ORIGIN_SPECULATIVE
                    if speculative
                    else "agent_inferred"
                ),
            )
        )
    return store


def corpus_shape(store: Any) -> dict[str, int]:
    """Locked and speculative-origin counts, read back off the store.

    Emitted with the figures so a reader can check that the two arms
    #1526 rewrote for those cases were populated, rather than trusting
    `_synthetic_store`'s arithmetic.
    """
    from aelfrice.models import LOCK_USER, ORIGIN_SPECULATIVE

    locked = 0
    speculative = 0
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is None:
            continue
        if b.lock_level == LOCK_USER:
            locked += 1
        if b.origin == ORIGIN_SPECULATIVE:
            speculative += 1
    return {"locked": locked, "speculative": speculative}


def _legacy_belief_tokens(b: Any) -> int:
    """`retrieval._belief_tokens`'s pre-#1526 body.

    Transcribed from `github/main`: `return _estimate_tokens(b.content)`,
    where `_estimate_tokens` is `ceil(len(text) / 4)` and returns 0 on empty
    text.
    """
    if not b.content:
        return 0
    return int((len(b.content) + 3) // 4)


def _legacy_lock_injection_tokens(b: Any, *, manifest_reference_locks: bool) -> int:
    """`retrieval.lock_injection_tokens`'s pre-#1526 body, from `github/main`."""
    from aelfrice.retrieval import is_reference_lock, lock_manifest_line

    if manifest_reference_locks and is_reference_lock(b):
        text = lock_manifest_line(b)
        return int((len(text) + 3) // 4) if text else 0
    return _legacy_belief_tokens(b)


def _legacy_core_cost(b: Any) -> int:
    """`<core>`'s pre-#1526 per-belief cost.

    Transcribed from `hook._build_session_start_subblock` on `github/main`:
    `cost = max(1, len(b.content) // _CORE_CHARS_PER_TOKEN)`.
    """
    return max(1, len(b.content) // 4)


@contextlib.contextmanager
def _legacy_accounting() -> Iterator[None]:
    """Rebind every cost function #1526 changed to its pre-#1526 body.

    Four names, enumerated rather than summarised, because a before arm that
    misses one measures a hybrid: `retrieval._belief_tokens` (the uncompressed
    pack cost, and the L2.5 sub-pack's), `retrieval._render_wrapper_tokens`
    (what the compressed arm of `retrieve_with_tiers._cost` adds -- pre-#1526
    it added nothing), `retrieval.lock_injection_tokens` (the L0 arm), and
    `clustering._belief_tokens` (the cluster and max-coverage packs' default
    `cost_fn`).

    The packers read these off the module global rather than closing over
    them, so rebinding reaches them. `_render_wrapper_tokens` exists as a
    module-level name for exactly this reason: compression resolves ON by
    default, so the compressed arm is the production one, and with its
    wrapper addition inlined in the closure there was nothing for a before
    arm to rebind.

    The Grep|Glob lanes' fifth change is not here: `_belief_line_cost` is
    passed in as `belief_cost_fn`, so their before arm passes None instead.
    """
    from aelfrice import clustering, retrieval

    saved = (
        retrieval._belief_tokens,
        retrieval._render_wrapper_tokens,
        retrieval.lock_injection_tokens,
        clustering._belief_tokens,
    )
    try:
        retrieval._belief_tokens = _legacy_belief_tokens
        retrieval._render_wrapper_tokens = lambda b: 0
        retrieval.lock_injection_tokens = _legacy_lock_injection_tokens
        clustering._belief_tokens = _legacy_belief_tokens
        yield
    finally:
        (
            retrieval._belief_tokens,
            retrieval._render_wrapper_tokens,
            retrieval.lock_injection_tokens,
            clustering._belief_tokens,
        ) = saved


# --- lanes ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Arm:
    """One rendered block: how many beliefs reached it and how big it is."""

    n_items: int
    n_bytes: int


def _core_candidates(store: Any) -> list[Any]:
    """Every store belief in `<core>`'s order: posterior descending, id ascending."""
    out: list[Any] = []
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is not None:
            out.append(b)
    out.sort(key=lambda b: (-(b.alpha / (b.alpha + b.beta)), b.id))
    return out


def _render_core(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    """`<core>` packs directly, with no retrieval, so `sub` is unused here."""
    del sub
    from aelfrice import hook

    candidates = _core_candidates(store)
    cost_fn = _legacy_core_cost if legacy else None
    packed = hook._pack_core_candidates(candidates, budget, cost_fn)
    return Arm(len(packed), sum(len(hook._core_belief_line(b)) + 1 for b in packed))


def _render_ups(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    del legacy
    from aelfrice import hook, retrieval

    hits = retrieval.retrieve(
        store, QUERY, token_budget=budget, l25_token_subbudget=sub,
        manifest_reference_locks=True,
    )
    return Arm(len(hits), len(hook._format_hits(hits)))


def _render_session_start(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    """The SessionStart baseline block, retrieved the way the lane retrieves it.

    The empty query is the point. `hook._retrieve_baseline_with_block` calls
    `retrieve(store, "", ...)`, and every relevance lane in
    `retrieve_with_tiers` is gated on `query.strip()`, so only L0 reaches the
    block and L0 is never trimmed (#379). Rendering this lane with `QUERY`
    would measure a lane that does not exist.
    """
    del legacy
    from aelfrice import hook, retrieval

    hits = retrieval.retrieve(
        store, "", token_budget=budget, l25_token_subbudget=sub,
        manifest_reference_locks=True,
    )
    return Arm(len(hits), len(hook._format_baseline_hits(hits)))


def _render_agent_context(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    del legacy
    from aelfrice import hook_agent_context, retrieval

    hits = retrieval.retrieve(
        store,
        QUERY,
        token_budget=budget,
        l25_token_subbudget=sub,
        l1_limit=hook_agent_context.INJECTED_L1_LIMIT,
        manifest_reference_locks=True,
    )
    return Arm(len(hits), len(hook_agent_context._build_block(list(hits))))


def _render_search_tool(
    store: Any, budget: int, sub: int, *, legacy: bool, bash: bool = False,
) -> Arm:
    from aelfrice import hook_search_tool, retrieval

    l1_limit = (
        hook_search_tool.BASH_INJECTED_L1_LIMIT
        if bash
        else hook_search_tool.INJECTED_L1_LIMIT
    )
    locked_ids = {
        b.id for b in store.list_locked_beliefs()
    }
    hits = retrieval.retrieve(
        store,
        QUERY,
        token_budget=budget,
        l25_token_subbudget=sub,
        l1_limit=l1_limit,
        manifest_reference_locks=True,
        # Pre-#1526 this lane had no cost function of its own and was charged
        # the `<belief …>` element it does not emit.
        belief_cost_fn=None if legacy else hook_search_tool._belief_line_cost,
    )
    block = hook_search_tool._format_results(
        QUERY,
        list(hits),
        locked_ids,
        bash_source=("rg", QUERY) if bash else None,
    )
    return Arm(len(hits), len(block))


def _render_cli_search(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    """`aelf search`'s own renderer: `{prefix} {id}: {content}` per line.

    `_cmd_search` prints an 8-character lock prefix, a space, the full belief
    id, `": "`, the content and a newline. Reproduced here rather than called
    because `_cmd_search` opens the ambient store and writes to a stream; the
    line shape is the whole of what it emits per hit.
    """
    del legacy
    from aelfrice import retrieval
    from aelfrice.models import LOCK_USER

    hits = retrieval.retrieve(
        store, QUERY, token_budget=budget, l25_token_subbudget=sub,
    )
    n_bytes = 0
    for h in hits:
        prefix = "[locked]" if h.lock_level == LOCK_USER else "        "
        n_bytes += len(f"{prefix} {h.id}: {h.content}") + 1
    return Arm(len(hits), n_bytes)


_RENDERERS: dict[str, Callable[..., Arm]] = {
    "ups": _render_ups,
    "core": _render_core,
    "session_start": _render_session_start,
    "search_tool": _render_search_tool,
    "search_tool_bash": lambda s, b, sub, *, legacy: _render_search_tool(
        s, b, sub, legacy=legacy, bash=True,
    ),
    "agent_context": _render_agent_context,
    "cli_search": _render_cli_search,
}

LANES: tuple[str, ...] = (
    "ups",
    "core",
    "session_start",
    "search_tool",
    "search_tool_bash",
    "agent_context",
    "cli_search",
)


def _subbudget() -> int:
    from aelfrice.retrieval import DEFAULT_L25_TOKEN_SUBBUDGET

    return DEFAULT_L25_TOKEN_SUBBUDGET


def _measure(lane: str, store: Any, budget: int, *, legacy: bool) -> tuple[Arm, str]:
    """Render one arm, and name which budget ended its pack.

    Returns `(arm, binds_on)`, where `binds_on` is one of:

    * `token_budget` — raising `token_budget` alone moves the bytes.
    * `l25_subbudget` — raising `l25_token_subbudget` alone moves them; the
      pack ended on the L2.5 sub-cap and the lane's own budget bound nothing.
    * `both` — neither alone moves them but raising both does. The two caps
      are binding together.
    * `pool` — raising both moves nothing. The pack ended on the candidate
      pool, and this arm is not evidence about any budget.

    **Every budget that can end the pack is varied, which is the round-2
    correction.** The earlier version raised only `token_budget` while
    `l25_token_subbudget` stayed at the module default, so a pack the L2.5
    sub-cap had ended read as a budget that does not bind, and the cell was
    suppressed. Measured on `ups` at 92 content characters, against a store of
    300 beliefs: 58 hits at `token_budget=6000` and 58 at 1,500,000 with the
    sub-budget at its default 400, but 60 at `token_budget=6000` with the
    sub-budget at 1600.

    The arm is returned in every case. A `pool` arm is weak evidence about
    the budget; its bytes are still the bytes the model receives, and
    suppressing them is how the earlier version hid a +53% deviation.
    """
    render = _RENDERERS[lane]
    sub = _subbudget()
    f = SATURATION_PROBE_FACTOR
    ctx = _legacy_accounting if legacy else contextlib.nullcontext
    with ctx():
        arm = render(store, budget, sub, legacy=legacy)
        wide_budget = render(store, budget * f, sub, legacy=legacy)
        wide_sub = render(store, budget, sub * f, legacy=legacy)
        wide_both = render(store, budget * f, sub * f, legacy=legacy)
    if wide_budget.n_bytes != arm.n_bytes:
        return (arm, "token_budget")
    if wide_sub.n_bytes != arm.n_bytes:
        return (arm, "l25_subbudget")
    if wide_both.n_bytes != arm.n_bytes:
        return (arm, "both")
    return (arm, "pool")


# Content lengths the untuned rebuild block is reported at. Shorter than the
# full grid because this is a "did it move" figure, not a tuning.
REBUILD_LENGTHS: tuple[int, ...] = (92, 150, 300)


def _rebuild_block_bytes(stores: dict[int, Any]) -> dict[str, Any]:
    """The PreCompact rebuild block's emitted bytes, before and after.

    `context_rebuilder._estimate_belief_tokens` is left on the old currency by
    #1526 on purpose, because `DEFAULT_REBUILDER_TOKEN_BUDGET` is reachable
    from `.aelfrice.toml` as `[rebuilder] token_budget` and re-denominating it
    would silently reinterpret values users have already written.

    Measured anyway, because #1526 round 1 justified leaving it with "correcting
    it would move this block's bytes", and whether that is true is a question
    with an answer rather than a premise. The answer is: only where this
    block's own budget binds. `rebuild_v14` takes its non-locked candidates
    from `retrieve()`, whose pack cost #1526 changed, so the hits reaching this
    formatter *can* be a different set -- but at 92 and 150 content characters
    they are the same set and the emitted bytes are identical (9,011 -> 9,011
    and 12,178 -> 12,178), because `DEFAULT_REBUILDER_TOKEN_BUDGET` is nowhere
    near binding there. At 300 it binds and the bytes move (19,293 -> 16,421).

    So the round-1 justification is not so much wrong as unstated: it holds at
    one of the three lengths this function measures and not at the other two.
    That is the reason to produce these numbers rather than assert them, and
    it is why the byte figures quoted in `context_rebuilder._format_block`
    carry `derived:` markers back to the keys below. An earlier revision of
    this docstring said the bytes "are not" constant, flatly; that was written
    before the cost functions settled and is what this paragraph replaces.
    """
    from aelfrice.context_rebuilder import RecentTurn, rebuild_v14
    from aelfrice.rebuild_log import DEFAULT_REBUILDER_TOKEN_BUDGET

    turns = [RecentTurn(role="user", text=QUERY, session_id="s1")]
    out: dict[str, Any] = {}
    for chars in REBUILD_LENGTHS:
        store = stores[chars]
        with _legacy_accounting():
            before = len(
                rebuild_v14(
                    turns, store,
                    token_budget=DEFAULT_REBUILDER_TOKEN_BUDGET,
                    rebuild_log_enabled=False,
                )
            )
        after = len(
            rebuild_v14(
                turns, store,
                token_budget=DEFAULT_REBUILDER_TOKEN_BUDGET,
                rebuild_log_enabled=False,
            )
        )
        out[f"rebuild_bytes_before_{chars}"] = before
        out[f"rebuild_bytes_after_{chars}"] = after
    return out


def shipped_budget(lane: str) -> int:
    """The budget constant this lane ships, read off the module that holds it."""
    from aelfrice import hook, hook_agent_context, hook_search_tool, retrieval

    return {
        "ups": hook.DEFAULT_HOOK_TOKEN_BUDGET,
        "core": hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET,
        "session_start": hook.DEFAULT_SESSION_START_TOKEN_BUDGET,
        "search_tool": hook_search_tool.INJECTED_TOKEN_BUDGET,
        "search_tool_bash": hook_search_tool.BASH_INJECTED_TOKEN_BUDGET,
        "agent_context": hook_agent_context.INJECTED_TOKEN_BUDGET,
        "cli_search": retrieval.DEFAULT_TOKEN_BUDGET,
    }[lane]


def figures(*, lengths: tuple[int, ...] = LENGTH_GRID) -> dict[str, Any]:
    """Re-derive every #1526 figure this repo publishes.

    `lengths` narrows the content-length grid. The full grid is what the
    CHANGELOG publishes; a caller that only needs the headline cell passes a
    shorter one, which is how `tests/test_render_cost_1526.py` keeps its
    acceptance run inside the CI per-test timeout.
    """
    from aelfrice import hook
    from aelfrice.render_cost import BELIEF_LINE_WRAPPER_CHARS
    from aelfrice.retrieval import resolve_use_type_aware_compression

    values: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Hermetic: `retrieve()` resolves ~22 `[retrieval]` flags by walking
        # up from the process cwd, and this repo sits under a home directory
        # that has an `.aelfrice.toml`. A tempdir has no ancestor config, so
        # the run reads the shipped defaults and nobody's local file.
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            values["compression"] = resolve_use_type_aware_compression()
            values["belief_line_wrapper_chars"] = BELIEF_LINE_WRAPPER_CHARS
            values["framing_header_chars"] = len(hook._FRAMING_HEADER)
            values["store_beliefs"] = STORE_BELIEFS
            values["lengths"] = list(lengths)
            values["l25_token_subbudget"] = _subbudget()

            stores = {
                chars: _synthetic_store(tmp / f"s{chars}.db", chars)
                for chars in lengths
            }
            try:
                values["corpus_shape"] = corpus_shape(stores[lengths[0]])
                for lane in LANES:
                    values.update(_lane_figures(lane, stores, lengths))
                if set(REBUILD_LENGTHS) <= set(lengths):
                    values.update(_rebuild_block_bytes(stores))
            finally:
                for store in stores.values():
                    store.close()
        finally:
            os.chdir(cwd)
    return values


def _lane_figures(
    lane: str, stores: dict[int, Any], lengths: tuple[int, ...],
) -> dict[str, Any]:
    """One lane's full curve, plus the headline cell at the corpus median.

    The headline cell is the grid length nearest `CORPUS_MEDIAN_CHARS` that is
    present, chosen by proximity and not by how flattering the number is. There
    is no tuning here and therefore no tuning length: the budget is the same in
    both arms, so every cell is a measurement of the same change.
    """
    budget = shipped_budget(lane)
    curve = _curve(lane, stores, budget, lengths)
    headline_chars = min(lengths, key=lambda c: (abs(c - CORPUS_MEDIAN_CHARS), c))
    headline = curve[str(headline_chars)]
    return {
        f"{lane}_budget": budget,
        f"{lane}_headline_chars": headline_chars,
        f"{lane}_bytes_before": headline["before"],
        f"{lane}_bytes_after": headline["after"],
        f"{lane}_items_before": headline["before_items"],
        f"{lane}_items_after": headline["after_items"],
        f"{lane}_pct": headline["pct"],
        f"{lane}_pool_equality": headline["pool_equality"],
        f"{lane}_curve": curve,
    }


def _curve(
    lane: str, stores: dict[int, Any], budget: int, lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Emitted bytes before and after, at every length in the grid.

    Every cell carries a byte count. `before_binds_on` / `after_binds_on`
    name which budget ended that arm's pack (see `_measure`); `pool_equality`
    is true only when both arms read `pool`, which is the one case where the
    two numbers say nothing about any budget because they are the same
    candidate pool rendered twice.
    """
    rows: dict[str, Any] = {}
    for chars in lengths:
        store = stores[chars]
        row: dict[str, Any] = {}
        for name, legacy in (("before", True), ("after", False)):
            arm, binds_on = _measure(lane, store, budget, legacy=legacy)
            row[name] = arm.n_bytes
            row[f"{name}_items"] = arm.n_items
            row[f"{name}_binds_on"] = binds_on
        row["pool_equality"] = (
            row["before_binds_on"] == "pool" and row["after_binds_on"] == "pool"
        )
        b, a = row["before"], row["after"]
        row["pct"] = round(100.0 * (a - b) / b, 1) if b else None
        rows[str(chars)] = row
    return rows


def _flag(row: dict[str, Any]) -> str:
    """The one-line legend for a cell, saying what is actually true of it."""
    if row["pool_equality"]:
        return (
            "  [both arms ended on the candidate pool: a pool equality, "
            "not budget evidence]"
        )
    marks = [
        f"{name} ended on {row[f'{name}_binds_on']}"
        for name in ("before", "after")
        if row[f"{name}_binds_on"] != "token_budget"
    ]
    return f"  [{'; '.join(marks)}]" if marks else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--emit-figures",
        action="store_true",
        help="emit a flat JSON object of key -> value on stdout and nothing else",
    )
    ap.add_argument(
        "--curve",
        action="store_true",
        help="print the per-length curve for each lane as well as the summary",
    )
    args = ap.parse_args(argv)

    values = figures()
    if args.emit_figures:
        json.dump(values, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    flat = {
        k: v
        for k, v in values.items()
        if not k.endswith("_curve") and not isinstance(v, (dict, list))
    }
    width = max(len(k) for k in flat)
    for key in sorted(flat):
        print(f"{key:<{width}}  {flat[key]}")
    print(f"corpus_shape  {values['corpus_shape']}")
    print()
    for lane in LANES:
        before = values[f"{lane}_bytes_before"]
        after = values[f"{lane}_bytes_after"]
        chars = values[f"{lane}_headline_chars"]
        row = values[f"{lane}_curve"][str(chars)]
        print(
            f"{lane}: budget {values[f'{lane}_budget']} unchanged, emitted bytes "
            f"{before} -> {after} ({values[f'{lane}_pct']:+.1f}%) at {chars} "
            f"content chars{_flag(row)}"
        )
    if args.curve:
        print()
        for lane in LANES:
            print(f"--- {lane} (budget {values[f'{lane}_budget']}, unchanged) ---")
            for chars, row in values[f"{lane}_curve"].items():
                pct = "" if row["pct"] is None else f"  ({row['pct']:+.1f}%)"
                print(
                    f"  {chars:>4} chars: {row['before']:>8} -> "
                    f"{row['after']:>8}{pct}"
                    f"  items {row['before_items']} -> {row['after_items']}"
                    f"{_flag(row)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
