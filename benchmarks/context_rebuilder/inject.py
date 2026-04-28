"""Midpoint-clear injection for the context-rebuilder eval harness.

Forces a synthetic context-clear at a configurable point in a
replay. The injection is a contract between `__main__` /
`replay.run()` and the test suite: when a `ClearInjection` is
passed in, the replay walks turns 0..clear_at-1 normally,
substitutes the rebuild block at `clear_at`, then continues.

Scaffolding only -- the injected "rebuild" is a fixed-overhead
synthetic block. Real rebuilder integration (calling
`aelfrice.context_rebuilder.rebuild()`) lands when #138 wires the
fidelity scorer in.

The injection logic deliberately lives in its own module so the
test suite can import it without pulling in the replay loader.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClearInjection:
    """Configuration for a single midpoint-clear injection.

    `clear_at` is the 0-based content-turn index at which the
    synthetic clear fires. Must be non-negative; values >= the
    fixture's turn count silently skip the injection (the replay
    treats it as "no clear was ever reached").

    The intentional asymmetry: a too-large `clear_at` is a
    well-formed config that simply never triggers, while a
    negative one is a programming error and raises.
    """
    clear_at: int

    def __post_init__(self) -> None:
        if self.clear_at < 0:
            raise ValueError(
                f"ClearInjection.clear_at must be >= 0, got {self.clear_at}"
            )


def midpoint_clear(n_turns: int) -> ClearInjection:
    """Build a ClearInjection that fires at the floor-midpoint turn.

    Convenience for the common case where the test wants "clear
    halfway through the fixture". For a fixture with `n_turns`,
    fires at `n_turns // 2`. For `n_turns < 2`, falls back to
    `clear_at=0` (which is itself a valid scaffolding case --
    the replay just records a clear at the very first turn).
    """
    if n_turns < 0:
        raise ValueError(f"n_turns must be >= 0, got {n_turns}")
    return ClearInjection(clear_at=max(0, n_turns // 2))
