"""Exploration-slot draw (#1176 proposal 5).

The module's whole value is that a draw is reproducible from logged state, so
the tests are mostly determinism contracts. Each one is written so that it
fails if the specific property is removed, not merely if the code crashes.
"""
from __future__ import annotations

import collections
import itertools

import pytest

from aelfrice.exploration import (
    DEFAULT_EXPLORATION_CADENCE,
    derive_seed,
    draw_uniform,
    should_explore,
    splitmix64_stream,
)

# Vigna's reference `splitmix64.c` for seed 0. These are the published
# vectors, not values captured from this implementation — they are what makes
# the generator a contract rather than whatever the code happens to do.
_SPLITMIX64_SEED0: tuple[int, ...] = (
    0xE220A8397B1DCDAF,
    0x6E789E6AA1B965F4,
    0x06C45D188009454F,
    0xF88BB8A8724C81EC,
    0x1B39896A51A8749B,
)


def test_splitmix64_matches_the_published_reference_vectors() -> None:
    got = tuple(itertools.islice(splitmix64_stream(0), len(_SPLITMIX64_SEED0)))
    assert got == _SPLITMIX64_SEED0


def test_splitmix64_stays_in_64_bits() -> None:
    """A missing mask would still pass the vector test for a few words."""
    for word in itertools.islice(splitmix64_stream(0xDEADBEEFCAFEF00D), 500):
        assert 0 <= word < (1 << 64)


def test_splitmix64_is_a_pure_function_of_the_seed() -> None:
    a = list(itertools.islice(splitmix64_stream(12345), 20))
    b = list(itertools.islice(splitmix64_stream(12345), 20))
    assert a == b
    assert a != list(itertools.islice(splitmix64_stream(12346), 20))


# --- seed derivation -------------------------------------------------------


def test_derive_seed_is_reproducible_and_input_sensitive() -> None:
    base = derive_seed("scope", 40, "how do i configure the pack budget")
    assert base == derive_seed("scope", 40, "how do i configure the pack budget")
    assert base != derive_seed("scope", 60, "how do i configure the pack budget")
    assert base != derive_seed("other", 40, "how do i configure the pack budget")
    assert base != derive_seed("scope", 40, "something else entirely")


def test_derive_seed_separates_its_fields() -> None:
    """Concatenation-collision guard.

    `("a", 11, "q")` and `("a1", 1, "q")` both render as `a11q` once the
    fields are concatenated without a separator, so they would seed the same
    draw. The `\\x1f` join is what prevents that.

    The colliding pair has to be chosen deliberately: verified by mutation
    that removing the separator makes exactly this assertion fail, whereas an
    arbitrary pair like `("ab", 1)` vs `("a", 11)` still differs
    (`ab1q` vs `a11q`) and would let the mutation through.
    """
    assert derive_seed("a", 11, "q") != derive_seed("a1", 1, "q")


def test_derive_seed_fits_64_bits() -> None:
    assert 0 <= derive_seed("scope", 7, "q") < (1 << 64)


# --- cadence ---------------------------------------------------------------


def test_should_explore_fires_on_every_kth_prompt() -> None:
    fires = [i for i in range(100) if should_explore(i, cadence=20)]
    assert fires == [0, 20, 40, 60, 80]


def test_should_explore_default_cadence_is_one_in_twenty() -> None:
    n = sum(should_explore(i) for i in range(1000))
    assert n == 1000 // DEFAULT_EXPLORATION_CADENCE


@pytest.mark.parametrize("cadence", [0, -1, -20])
def test_non_positive_cadence_disables_rather_than_raising(cadence: int) -> None:
    """`fire_idx % 0` is a ZeroDivisionError inside a retrieval call."""
    assert [i for i in range(50) if should_explore(i, cadence=cadence)] == []


# --- the draw --------------------------------------------------------------


def _pool(n: int) -> list[str]:
    return [f"b{i:04d}" for i in range(n)]


def test_draw_is_reproducible_for_the_same_seed() -> None:
    got = draw_uniform(_pool(200), seed=99, count=3)
    assert got == draw_uniform(_pool(200), seed=99, count=3)
    assert len(got) == 3


def test_draw_changes_with_the_seed() -> None:
    """Guards against a 'draw' that just returns the first `count` ids."""
    seen = {tuple(draw_uniform(_pool(200), seed=s, count=3)) for s in range(25)}
    assert len(seen) > 1


def test_draw_does_not_depend_on_candidate_order() -> None:
    """The pool arrives from SQL, whose row order is not contractual.

    Fails if the internal `sorted()` is dropped — which is exactly the change
    that would make an SQLite plan change silently pick a different belief.
    """
    pool = _pool(60)
    assert draw_uniform(pool, seed=5, count=4) == draw_uniform(
        list(reversed(pool)), seed=5, count=4
    )


def test_draw_is_without_replacement() -> None:
    got = draw_uniform(_pool(30), seed=3, count=30)
    assert len(got) == len(set(got)) == 30


def test_draw_collapses_duplicate_ids() -> None:
    assert draw_uniform(["a", "a", "a", "b"], seed=1, count=4) == draw_uniform(
        ["a", "b"], seed=1, count=4
    )


def test_draw_returns_the_whole_pool_when_count_exceeds_it() -> None:
    got = draw_uniform(_pool(4), seed=7, count=99)
    assert sorted(got) == _pool(4)


@pytest.mark.parametrize("count", [0, -1])
def test_non_positive_count_draws_nothing(count: int) -> None:
    assert draw_uniform(_pool(10), seed=1, count=count) == []


def test_empty_pool_draws_nothing() -> None:
    assert draw_uniform([], seed=1, count=3) == []


def test_widening_the_slot_count_keeps_the_earlier_slots() -> None:
    """Prefix stability, claimed by `draw_uniform`'s docstring.

    Partial Fisher-Yates gives it; a "shuffle the whole pool then take
    `count`" implementation does not, because the number of random words
    consumed would depend on the pool size rather than on `take`. Raising M
    from 1 to 2 must add a slot, not resample slot 1.
    """
    pool = _pool(50)
    one = draw_uniform(pool, seed=11, count=1)
    assert draw_uniform(pool, seed=11, count=3)[:1] == one


def test_the_draw_is_actually_uniform() -> None:
    """The property the A-Res weighting was dropped in favour of.

    Every belief in a 40-id pool should be drawn about equally often across
    4,000 seeds. A biased `_bounded` (naive `% bound`) or an off-by-one in the
    Fisher-Yates bound shows up here as a lopsided histogram. Bounds are wide
    enough not to flake: this is deterministic, so they only need to hold for
    these exact seeds.
    """
    pool = _pool(40)
    counts = collections.Counter()
    trials = 4000
    for s in range(trials):
        counts.update(draw_uniform(pool, seed=s, count=1))
    assert len(counts) == len(pool), "some belief was never drawn"
    expected = trials / len(pool)
    assert max(counts.values()) < expected * 1.5
    assert min(counts.values()) > expected * 0.5


def test_bounded_rejects_the_biased_tail_of_the_word_range() -> None:
    """Pins the rejection sampling in `_bounded`.

    This reaches for a private symbol on purpose. The statistical test above
    cannot see the modulo bias: at a pool size of 40 against a 2**64 word
    range the skew is on the order of 1e-18, so `word % bound` passes every
    black-box test in this file. Verified by mutation — replacing the
    rejection limit with `1 << 64` leaves all the other tests green.

    Driving `_bounded` with a hand-made word stream makes the difference
    exact rather than statistical. For `bound = 3`, `2**64 % 3 == 1`, so the
    single word `2**64 - 1` is the biased tail and must be skipped:

      * rejection  -> skips it, consumes the next word, returns `5 % 3 == 2`
      * naive `%`  -> returns `(2**64 - 1) % 3 == 0`
    """
    from aelfrice.exploration import _bounded

    assert (1 << 64) % 3 == 1, "arithmetic premise of this test"
    assert ((1 << 64) - 1) % 3 == 0, "the two paths must disagree"
    assert _bounded(iter([(1 << 64) - 1, 5]), 3) == 2


def test_bounded_rejects_a_non_positive_bound() -> None:
    from aelfrice.exploration import _bounded

    with pytest.raises(ValueError):
        _bounded(splitmix64_stream(0), 0)


def test_every_pool_member_is_reachable_in_a_later_slot_too() -> None:
    """Slot 2 must not be drawn from a truncated pool."""
    pool = _pool(20)
    second = {draw_uniform(pool, seed=s, count=2)[1] for s in range(2000)}
    assert second == set(pool)
