"""Exploration slots — surfacing beliefs the ranker never surfaces (#1176).

A belief that starts underranked is never retrieved, therefore never
referenced, therefore never acquires evidence, therefore stays underranked.
Nothing in the retrieval path breaks that loop, and the loop is most of the
store: of 44,586 active beliefs measured on the live store, **37,489 (84.1%)
have never received a `feedback_history` row or an `injection_events` row**,
and only **1,352 (3.0%) have ever been injected into a context at all**. The
92,685 feedback rows land on 7,450 beliefs — the feedback is concentrated on
the beliefs that were already winning.

The fix is one slot in every K prompts, drawn from the beliefs the ranker
never reaches.

**The draw is uniform, and that is a measured decision rather than a
simplification.** The proposal specified Efraimidis-Spirakis A-Res weighted
reservoir sampling keyed on `scoring.uncertainty_score`, so that beliefs the
system is genuinely unsure about are explored first. Two measurements killed
that:

  * `uncertainty_score` is Beta *differential* entropy, which on ``[0, 1]`` is
    ``<= 0`` and exactly ``0`` for ``Beta(1, 1)``. A-Res requires strictly
    positive weights: ``key = u ** (1 / w)`` raises ``ZeroDivisionError`` on
    the 39 ``Beta(1, 1)`` beliefs in the pool, and every other key comes out
    ``> 1`` — not a reservoir at all.
  * After either natural sign repair the weighting is indistinguishable from
    uniform on this corpus. Total-variation distance from uniform is **0.0586**
    for an affine ``H - H_min`` shift and **0.0890** for ``exp(H)``, because
    two entropy values cover 88.1% of the pool and four cover 98.2%. The two
    dominant classes draw at 0.99x and 1.15x their pool share.

So the weighting buys nothing here, and uniform is the honest model of what
the weighted version would have done. If the posterior ever becomes
informative — it is currently a two-valued function of ``(type x origin)``
assigned at ``derive()``, not a learned quantity — a weighted key drops in
behind the same seeded stream with no other change.

Determinism is the whole contract. The draw is seeded from logged state only
(`scope_id`, the monotonic `fire_idx`, and the query), never from a clock or
`random`, so the same store and counter reproduce the same draw. `random` is
deliberately unused: its Mersenne state and float conversion are not a stable
contract across CPython versions, and replay has to hold across upgrades.
"""
from __future__ import annotations

import hashlib
from typing import Final, Iterable, Iterator

__all__ = [
    "DEFAULT_EXPLORATION_CADENCE",
    "DEFAULT_EXPLORATION_SLOTS",
    "derive_seed",
    "draw_uniform",
    "should_explore",
    "splitmix64_stream",
]

_MASK64: Final[int] = (1 << 64) - 1
_GOLDEN_GAMMA: Final[int] = 0x9E3779B97F4A7C15
_MIX_A: Final[int] = 0xBF58476D1CE4E5B9
_MIX_B: Final[int] = 0x94D049BB133111EB

# Explore on one prompt in twenty, one slot at a time. K matches the shipped
# P1 cadence predicate; M is 1 because the cost of an exploration slot is a
# ranked belief that does not get shown, and one slot is the smallest
# intervention that can still produce a signal.
DEFAULT_EXPLORATION_CADENCE: Final[int] = 20
DEFAULT_EXPLORATION_SLOTS: Final[int] = 1


def splitmix64_stream(seed: int) -> Iterator[int]:
    """Yield the SplitMix64 sequence for `seed`, one 64-bit word at a time.

    Vigna's reference `splitmix64.c` (public domain, used as the seeding
    generator for xoshiro/xoroshiro). Chosen over `random` because the whole
    point is a bit-exact contract that survives a CPython upgrade: this is
    twelve lines of integer arithmetic with no library surface to drift.

    Verified against the published vectors for seed 0 in
    `tests/test_exploration_1176.py`.
    """
    state = seed & _MASK64
    while True:
        state = (state + _GOLDEN_GAMMA) & _MASK64
        z = state
        z = ((z ^ (z >> 30)) * _MIX_A) & _MASK64
        z = ((z ^ (z >> 27)) * _MIX_B) & _MASK64
        yield z ^ (z >> 31)


def derive_seed(scope_id: str, fire_idx: int, query: str) -> int:
    """Seed the draw from logged state only.

    Every input is either persisted or the caller's own argument: `scope_id`
    lives in `schema_meta`, `fire_idx` is the monotonic hot-path counter
    (never wall-clock), and `query` is the prompt. So the seed — and therefore
    the whole draw — is a function of the write log and the code, which is
    what makes an exploration slot replayable rather than a coin flip.

    The three fields are joined with a `\\x1f` separator so that no pair of
    distinct triples can collide by concatenation (`("ab", 1)` and `("a", "b1")`
    would otherwise hash alike).
    """
    payload = "\x1f".join((scope_id, str(fire_idx), query)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def should_explore(
    fire_idx: int, *, cadence: int = DEFAULT_EXPLORATION_CADENCE
) -> bool:
    """True on every `cadence`-th fire. `cadence <= 0` disables exploration.

    Guarding `cadence <= 0` here rather than at the call site means a
    misconfigured cadence degrades to "never explore" instead of raising
    `ZeroDivisionError` inside a retrieval.
    """
    if cadence <= 0:
        return False
    return fire_idx % cadence == 0


def _bounded(words: Iterator[int], bound: int) -> int:
    """Uniform integer in ``[0, bound)`` from the 64-bit `words` stream.

    Rejection-sampled rather than `next(words) % bound`. The modulo shortcut
    is biased toward small values whenever `bound` does not divide 2**64, and
    while the bias is negligible at these sizes, an exploration mechanism
    whose entire justification is "the ranker is systematically skewed"
    should not itself introduce a systematic skew. Rejection is unbiased and
    still deterministic: the number of words consumed is a function of the
    stream, so replay reproduces it.
    """
    if bound <= 0:
        raise ValueError(f"bound must be positive, got {bound}")
    limit = (1 << 64) - ((1 << 64) % bound)
    for word in words:
        if word < limit:
            return word % bound
    raise RuntimeError("splitmix64_stream terminated")  # pragma: no cover


def draw_uniform(
    candidate_ids: Iterable[str], *, seed: int, count: int
) -> list[str]:
    """Draw up to `count` distinct ids uniformly at random, without replacement.

    Returns them in draw order — the caller wants to know which slot each one
    filled, and re-sorting would throw that away.

    `candidate_ids` is sorted ASC before drawing, so the result depends on the
    *set* of candidates and not on the order the caller happened to collect
    them in. That matters because the pool arrives from a SQL query whose row
    order is not contractual; without this, a plan change in SQLite would
    silently change which belief gets explored.

    Duplicate ids are collapsed. A `count` at or above the pool size returns
    the whole pool, still in draw order, which keeps the small-pool case from
    being a special case at the call site.
    """
    pool = sorted(set(candidate_ids))
    if count <= 0 or not pool:
        return []
    words = splitmix64_stream(seed)
    take = min(count, len(pool))
    # Partial Fisher-Yates: swap each chosen element to the front so the
    # remaining pool stays contiguous. O(take) swaps rather than a full
    # shuffle, and identical to what a full shuffle's first `take` entries
    # would be, so widening M later does not change the earlier slots.
    for i in range(take):
        j = i + _bounded(words, len(pool) - i)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:take]
