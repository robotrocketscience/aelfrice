"""Hot-path touch state — v1 storage substrate (#748 / #816).

Pure helpers + constants for the per-(belief, session) touch-state
sidecar table (`belief_touches`). The schema and the store-side write /
read APIs live in :mod:`aelfrice.store`; this module owns the
fire-window predicate, the event-kind bitmask values, and the default
window-K module constant.

DESIGN.md v1 (`6b40538`, lab `experiments/hot-path/DESIGN.md`) is the
source of truth for the scope this module ships against:

- **Boolean decay only.** No exponential `tau`. H2 refuted at R2c/R2d
  on the lab corpus — exponential and boolean produce indistinguishable
  rerank orderings at the pre-registered gate. The decay shape here is
  a window check against a monotonic counter, not a sum over touches
  with a decay kernel.
- **INJECTION-only events.** H4 refuted at R4/R4b/R4c/R4d/R4e/R5 — the
  top-K Jaccard between INJECTION-only and INJECTION + retrieve_hit is
  1.000 across the lab corpus. `retrieve_hit`, `bfs_visit`, and
  `user_action` are out for v1; the bitmask reserves the bits.
- **No rerank consumer.** Touch state is written but not read at
  rerank time in v1. The H3 fidelity test (R3 / R7c) gates the
  consumer flip. v1 ships the substrate; the multiplier helper here
  has no production caller yet.

Locked decisions honored:

- **Determinism** (`c06f8d575fad71fb`, #605). ``current_fire_idx`` and
  ``last_fire_idx`` are monotonic integers, never wall-clock. Same
  query + same store + same fire_idx → same predicate verdict across
  replays.
- **Federation** (`d0c5ecdebb3f0f4d`, #661). Touch state is per-DB;
  foreign federated beliefs come in cold every read. The store table's
  composite PK ``(belief_id, session_id)`` is the structural guarantee
  — no cross-scope rows exist by construction.
- **PHILOSOPHY narrow surface** (#605). Pure stdlib. No ML, no
  embeddings, no LLM. The predicate is one integer comparison.
"""

from __future__ import annotations

from typing import Final

DEFAULT_TOUCH_WINDOW_K: Final[int] = 50
"""Default K for the "touched in last K fires" predicate.

Sourced from R2c's per-session corpus scale (`HYPOTHESES.md §H2`,
canonical cell). Promotion to a `meta:retrieval.hot_window_K` knob is
an explicit follow-up if H3 lands and motivates tuning — DESIGN.md v1
locks the constant per the ratified "non-decisions" section: move it
only by re-measurement, not by config knob.
"""

# Event-kind bitmask values. Only ``INJECTION`` (bit 0) is populated in
# v1; bits 1-3 are schema-reserved for v2 expansion if a future
# campaign re-opens H4 under different telemetry.
TOUCH_EVENT_KIND_INJECTION: Final[int] = 1 << 0
"""Bit 0. Set on the UPS + PreToolUse output-side injection path."""

TOUCH_EVENT_KIND_RETRIEVE_HIT: Final[int] = 1 << 1
"""Bit 1. Reserved. v1 does not write this bit; H4 refuted at R4."""

TOUCH_EVENT_KIND_BFS_VISIT: Final[int] = 1 << 2
"""Bit 2. Reserved. v1 does not write this bit."""

TOUCH_EVENT_KIND_USER_ACTION: Final[int] = 1 << 3
"""Bit 3. Reserved. v1 does not write this bit (H4a deferred)."""


def is_hot(
    *,
    last_fire_idx: int,
    current_fire_idx: int,
    window_k: int = DEFAULT_TOUCH_WINDOW_K,
) -> bool:
    """True iff ``last_fire_idx`` falls inside the last ``window_k`` fires.

    Boolean decay shape from DESIGN.md v1 §"Decay shape — H2 REFUTED
    for exponential":

        is_hot(b) iff touch.last_fire_idx >= current_fire_idx - window_k

    No wall-clock; ``fire_idx`` is a monotonic per-session counter.
    Negative ``last_fire_idx`` (sentinel for "never touched in this
    session") returns False at any positive ``window_k`` because the
    comparison ``-1 >= current_fire_idx - window_k`` is false whenever
    ``current_fire_idx > window_k - 1`` — which always holds at the
    point the predicate is queried (current_fire_idx is the *next* fire
    about to write, never zero on a populated touch row).
    """
    if window_k <= 0:
        raise ValueError(f"window_k must be > 0; got {window_k!r}")
    if current_fire_idx < 0:
        raise ValueError(
            f"current_fire_idx must be >= 0; got {current_fire_idx!r}"
        )
    return last_fire_idx >= current_fire_idx - window_k


__all__ = [
    "DEFAULT_TOUCH_WINDOW_K",
    "TOUCH_EVENT_KIND_BFS_VISIT",
    "TOUCH_EVENT_KIND_INJECTION",
    "TOUCH_EVENT_KIND_RETRIEVE_HIT",
    "TOUCH_EVENT_KIND_USER_ACTION",
    "is_hot",
]
