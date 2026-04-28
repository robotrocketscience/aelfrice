"""Token-cost + hook-latency measurement primitives.

Two scaffolding metrics for the context-rebuilder eval harness:

  1. **Token-budget delta.** `rebuild_block_tokens / full_replay_tokens`
     in the headline metric framing; this module exposes the per-turn
     signed delta used to build that ratio.
  2. **Hook latency.** Wall-clock from "PreCompact hook fires" to
     "rebuild block emitted", in milliseconds. Measured via
     `time.monotonic()` so it is monotonic by construction.

Token estimation uses the same chars-per-token heuristic as
`aelfrice.context_rebuilder` (4 chars/token) so the harness's
estimates and the rebuilder's internal estimates stay aligned --
a real tokenizer (tiktoken, sentencepiece) would land alongside
the fidelity scorer in #138.
"""
from __future__ import annotations

import time
from typing import Final

#: Chars-per-token estimate. Mirrors `aelfrice.context_rebuilder._CHARS_PER_TOKEN`.
#: When the v1.4.x fidelity scorer ships, both call sites move to a
#: real tokenizer at the same time -- keep the constants in sync until
#: then.
CHARS_PER_TOKEN: Final[float] = 4.0


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate for replay measurement.

    Ceiling division of character count by `CHARS_PER_TOKEN`. Empty
    string returns 0. Negative-length input is impossible (Python
    `str` is non-negative-length by construction); no guard needed.
    """
    if not text:
        return 0
    return int((len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def token_budget_delta(*, full: int, rebuilt: int) -> int:
    """Signed cumulative-token delta at one turn.

    `rebuilt - full`. Negative means the rebuilder saved tokens at
    this turn; positive means it didn't (or the rebuild incurred
    extra overhead). Pre-clear, both sides equal the same running
    total and the delta is 0.

    Returned as `int`, not `float`, because both inputs are integer
    token counts; the headline ratio in #136 is a separate
    presentation-layer concern.
    """
    return rebuilt - full


def hook_latency_ms(start_monotonic: float) -> float:
    """Wall-clock since `start_monotonic`, in milliseconds.

    Caller supplies `time.monotonic()` from before the hook fired.
    Return value is `(now - start) * 1000.0`, floored at 0.0 to
    paper over clock jitter on systems where `time.monotonic()`
    is technically only non-decreasing within a process and not
    across forks. The floor keeps `hook_latency_ms >= 0` always
    true, which is the test contract.
    """
    elapsed = time.monotonic() - start_monotonic
    if elapsed < 0.0:
        return 0.0
    return elapsed * 1000.0
