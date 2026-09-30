"""The whole-output bound every aelfrice hook fire shares (#1639).

A leaf module on purpose. The PreToolUse search hook and the
UserPromptSubmit and SessionStart hooks all charge their output against
this bound and name the locks it cuts with the same line. The search
hook importing them from `aelfrice.hook` closed an import cycle (CodeQL
py/cyclic-import), so both hooks import from here. This is not a
latency change: a search-hook fire loads `aelfrice.hook` through other
imports either way, measured on 2026-09-30.
"""
from __future__ import annotations

from typing import Final

HOOK_PAYLOAD_CHAR_LIMIT: Final[int] = 9_500
"""Characters one hook fire may write to stdout in total (#1639).

The host inlines at most 10,000 characters of a hook's output. Anything
longer is saved to a file and replaced in the model's context by a
2,000-character preview and the file's path
(https://code.claude.com/docs/en/hooks.md). A block over that limit is
therefore not a large injection but a small one: the model reads the
first 2,000 characters and nothing after them unless it opens the file.
Measured on transcripts before this constant existed, 41% of aelfrice
hook outputs were saved rather than inlined, and of the saved ones that
carried user locks, 46% lost at least one lock from the preview.

The limit is 9,500, not 10,000, by operator ruling: 5% headroom for the
host's framing. It is counted in characters with `len()`, as the host's
documentation states its limit, not in estimated tokens. Whether the host
counts code points or UTF-16 units is not documented; the headroom covers
the difference for text that is mostly outside the astral planes.

This is the payload-wide bound #1560 ruled out. #1639 reopened that
ruling for this one change, because the host's cap applies to a fire's
whole stdout and a set of per-block bounds cannot guarantee a sum.
`HOOK_BLOCK_TOKEN_CEILING` and every per-block bound still hold; this
constant sits outside them. It does not add a shed order: the memory
envelope gets whatever room the fire's other blocks leave, and
`enforce_block_ceiling` trims to that room in the order it already uses.
The one new behaviour is what happens when the user locks alone do not
fit, described on `_choose_locks_for_room`.
"""

LOCK_POINTER_ID_CAP: Final[int] = 20
"""Most lock ids the #1639 overflow line names before it says "and N more".

Every omitted id is named when there are this many or fewer. Past it, the
line names the first ones and counts the rest, because on a store of
hundreds of locks the full list alone would fill the limit and leave no
room for a single lock to be shown.
"""

LOCK_POINTER_ID_CHARS: Final[int] = 40
"""Longest id the overflow line prints whole (#1639). Store ids are 16
characters; one longer than this is shown cut, so no id can make the line
itself outgrow the room it is priced into."""


def lock_overflow_line(omitted: list[str]) -> str:
    """The #1639 line that names user locks the payload bound cut."""
    named = [
        b if len(b) <= LOCK_POINTER_ID_CHARS
        else b[:LOCK_POINTER_ID_CHARS - 1] + "…"
        for b in omitted[:LOCK_POINTER_ID_CAP]
    ]
    more = len(omitted) - len(named)
    ids = ", ".join(named) + (f", and {more} more" if more else "")
    return (
        f"\naelfrice: {len(omitted)} user lock(s) did not fit this hook's "
        f"output limit and are not shown in full here: {ids}. Run "
        f"`aelf locked` to read "
        "every lock.\n"
    )
