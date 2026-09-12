"""#1526 — the render scaffolding the injection budgets used to spend for free.

The budgets on the UserPromptSubmit path were denominated in belief-content
characters, but what reaches the model is XML. A packed belief is not emitted
as its content: it is emitted as a `<belief …>` element on its own line, and
that wrapper does not shrink when the content does. A packer that charges
`len(b.content)` therefore spends a currency it does not emit, and the block
overruns its cap by a factor that depends entirely on how short the store's
beliefs are.

**The overrun is a ratio of wrapper to content, so it is not a constant and
must never be quoted as one.** It is largest on the shortest beliefs and shrinks
towards parity as content grows. Whatever a store's overrun is today, it moves
when the store's median belief length moves, so a figure derived from it is only
readable beside the length it was measured at.
`benchmarks/injection_budget_bytes.py` is the producer for the #1526 byte
figures; it measures them against the committed `tests/corpus/replay_soak`
corpus and emits the corpus's median belief length alongside them.

This module holds the wrapper widths, in one place, for the packers that have
to charge them. It deliberately does not hold the renderers — those live beside
the blocks they emit (`hook._split_belief_lines`, `hook._core_belief_line`,
`context_rebuilder._format_block`). A copy of a width can drift from the
renderer it describes, so each wrapper width here is pinned against live
renderer output by `tests/test_render_cost_1526.py`; that test, not this
docstring, is what keeps them true.

Kept free of any `aelfrice` runtime import so `hook`, `retrieval` and
`clustering` can all read it without a cycle and without pulling a heavy
subtree onto the hook's module-load path (#1527).
"""
from __future__ import annotations

from typing import Final

CHARS_PER_TOKEN: Final[float] = 4.0
"""The char-per-token estimator the packers here were already using.

Spelled as a float to match `retrieval._CHARS_PER_TOKEN`; the integer
spelling `hook._CORE_CHARS_PER_TOKEN` divides to the same value.
"""

# #1526 — the `<belief>` element and its newline: 52 characters, itemised
# in the docstring below and pinned against the live renderer by
# tests/test_render_cost_1526.py.
# <!-- derived: benchmarks/injection_budget_bytes.py#belief_line_wrapper_chars = 52 -->
BELIEF_LINE_WRAPPER_CHARS: Final[int] = 52
"""What one `<belief>` line costs beyond its own content.

`hook._split_belief_lines` emits `<belief id="{id}" lock="{user|none}">`
+ content + `</belief>`, and the caller joins the lines with a newline:

    12  `<belief id="`
    16  the belief id -- sha256 hex truncated to 16 (`derivation._belief_id`)
     8  `" lock="`
     4  `user` or `none`, the only two values `_split_belief_lines` emits
     2  `">`
     9  `</belief>`
     1  the newline the join contributes
    --
    52

This is a floor, not a universal. A belief that carries the `speculative`
attribute costs `SPECULATIVE_ATTR_CHARS` more, angle brackets in content are
escaped to entities at render time (`hook._escape_for_hook_block`), and a
belief id of some other width would move the 16. The first is charged
explicitly by the callers; the other two remain uncharged and are the residual
this issue does not close.
"""

SPECULATIVE_ATTR_CHARS: Final[int] = 16
"""` speculative="1"` — the #1171 provenance marker, when present."""

MANIFEST_LINE_WRAPPER_CHARS: Final[int] = 3
"""What one locks-manifest line costs beyond the manifest text itself.

`hook._split_belief_lines` prefixes each manifest entry with two spaces and
the join contributes a newline. The entry text itself comes from
`retrieval.lock_manifest_line` / `seen_manifest_line` and is already charged
by the caller.
"""


def chars_to_tokens(chars: int) -> int:
    """Round a character count up to whole tokens. Never negative."""
    if chars <= 0:
        return 0
    return int((chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def belief_line_chars(content_len: int, *, speculative: bool = False) -> int:
    """Characters one rendered `<belief>` line costs, newline included."""
    total = content_len + BELIEF_LINE_WRAPPER_CHARS
    if speculative:
        total += SPECULATIVE_ATTR_CHARS
    return total
