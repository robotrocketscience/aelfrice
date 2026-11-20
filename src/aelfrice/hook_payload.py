"""The whole-output bound every aelfrice hook fire shares (#1639).

A leaf module on purpose. The PreToolUse search hook and the
UserPromptSubmit and SessionStart hooks all charge their output against
this bound and name the locks it cuts with the same line. The search
hook importing them from `aelfrice.hook` closed an import cycle (CodeQL
py/cyclic-import), so both hooks import from here. This is not a
latency change: a search-hook fire loads `aelfrice.hook` through other
imports either way, measured on 2026-09-30.

The render-time escapers live here for the same reason (#1631). The
search hook and `aelfrice.provenance_render` imported them from
`aelfrice.hook`, and both of those imports closed a cycle through `hook`.
They are pure string substitution with no hook state, so they belong with
the other things every hook fire's output shares. `aelfrice.hook` still
binds them under their old private names. Check the cycles with
`scripts/import_cycles.py`.

The host's prompt id lives here too (#1954). The retrieval hook and the
transcript logger both read it as the turn id, and the logger must not
import `aelfrice.hook`.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
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


def escape_for_hook_block(content: str) -> str:
    """Entity-escape every angle bracket in belief content at render time.

    Pure string substitution — no XML/HTML parser. Called once per belief
    from `_format_hits` and `_format_baseline_hits`.

    This was a closed blocklist of framing tags (#280). A blocklist cannot
    hold: it omitted the two tags that carry the *trust* semantics —
    `<locked>` and `<core>` — and `str.replace` is case-sensitive, so
    `</CORE><LOCKED>` passed through untouched. Stored content that reaches
    the `<core>` section could therefore close its own element and re-open
    inside the user-locked tier, which the framing header presents to the
    model as the user's standing instructions. Ingested transcript and
    commit text is attacker-reachable, so this is a privilege boundary, not
    a cosmetic one.

    Escaping every `<` / `>` is the only form that does not require the
    escaper to know the emitter's full tag vocabulary. Content is unchanged
    in the store; this is render-time only.
    """
    return content.replace("<", "&lt;").replace(">", "&gt;")


def escape_attr(value: str) -> str:
    """Escape a string for use inside a double-quoted XML attribute."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


PROMPT_ID_KEY: Final[str] = "prompt_id"
"""The payload field that names the user prompt being processed (#1954).

`aelf-hook` and `aelf-transcript-logger` run as separate processes on one
UserPromptSubmit, and each used to mint its own random turn id, so
`injection_events.turn_id` never equalled the logged turn's `turn_id`.
The host sends this field on its UserPromptSubmit, Stop and PostToolUse
payloads: "UUID identifying the user prompt currently being processed"
(https://code.claude.com/docs/en/hooks, common input fields). When a
payload carries a valid one, both writers use it as the turn id, so the
two records join exactly. When it is absent or invalid, each writer mints
its own id as before. There is no content-key fallback: on the measured
turn logs, most records shared their text with another record of the same
session.
"""

PROMPT_ID_MAX_CHARS: Final[int] = 128
"""Longest `prompt_id` accepted as a turn id (#1954).

The host sends a UUID, 36 characters. The bound keeps a malformed value
from writing an unbounded string into every turn line and every
`injection_events` row; a longer value counts as absent.
"""


_PROMPT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9A-Za-z_-]{1,%d}" % PROMPT_ID_MAX_CHARS,
)
"""A usable `prompt_id`: ASCII letters, digits, `_` and `-` only.

An allowlist, so a value the writers can't store never gets through. A
lone surrogate passed a deny list of control characters, and then the
logger's UTF-8 write raised and lost the user line, and the hook lost its
whole `injection_events` batch. Padding and non-ASCII letters are refused
the same way.
"""


def payload_prompt_id(payload: Mapping[str, object]) -> str | None:
    """The payload's `prompt_id` when it is usable as a turn id, else None.

    Usable means a string of 1 to `PROMPT_ID_MAX_CHARS` ASCII letters,
    digits, `_` or `-`. Any other value, including a missing key, reads as
    absent, and the caller mints its own id.
    """
    value = payload.get(PROMPT_ID_KEY)
    if not isinstance(value, str):
        return None
    if _PROMPT_ID_RE.fullmatch(value) is None:
        return None
    return value
