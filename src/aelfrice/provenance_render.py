"""Trust-tier grouping and evidence attributes for the injected block (#1326).

`#1177` proposal `18`. The per-turn belief line carries `id`, `lock` and
(since #1171) `speculative`. Everything else the store knows about how far
to trust a belief — where it came from, how much evidence stands behind it,
how many times it has been re-asserted — is on the `Belief` object already
in hand at render time and is thrown away.

That matters because the ranking cannot use the one number that varies most.
`mu = 0.6 at n = 2` is byte-identical to `mu = 0.6 at n = 200` at every
scoring site, and the spread is not a cross-store curiosity: one live pack
measured for #1326 carried **25 distinct `n` values from 1.6 to 363.2 across
74 hits**, inside a single turn's block. A ranker has to collapse that to one
scalar. A model shown the number can weigh it against the question being
asked, which is the cheapest place to recover the signal.

**Default off.** `AELFRICE_PROVENANCE_RENDER` (env, decisive) then
`[hook] provenance_render` in `.aelfrice.toml`, then off. With the flag off
`hook._split_belief_lines` takes its original path and the block is
byte-identical, which is asserted rather than assumed — the framing header
is validated wording (rule-compliance 0/3 -> 5/5) and this must not perturb
it by accident.

## The section rule is total, deliberately

The proposal as filed named `<observed>` = `{user_transcript, commit, file}`
and `<inferred>` = `{agent_inferred, speculative}` and called that "a total
function of `lock_level` and `origin`". Measured against the live store it
strands **6,396 active beliefs — 14.3%** in no section at all (6,391
`unknown` plus 5 unlocked `user_stated`), and `commit` / `file` are not
origin values that exist. A renderer written to that spec drops 14.3% of the
block silently, which is the worst available failure mode: fewer beliefs, no
error.

So `SECTION_BY_ORIGIN` below enumerates every `models.ORIGIN_*` constant, and
`section_for` falls back to `DEFAULT_SECTION` for anything unrecognised. The
fallback direction is **`<inferred>`**, not `<observed>`: an origin nobody
has classified is one whose trustworthiness nobody has established, and the
safe error is to under-trust it. `test_every_declared_origin_has_a_section`
enumerates `models.ORIGINS` rather than a literal list, so adding a new
`ORIGIN_*` without classifying it fails the suite instead of silently
landing in the default.
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final, cast

from aelfrice.models import (
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_AGENT_REMEMBERED,
    ORIGIN_DOCUMENT_RECENT,
    ORIGIN_SPECULATIVE,
    ORIGIN_UNKNOWN,
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_STATED,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
)

if TYPE_CHECKING:
    from aelfrice.models import Belief

CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
SECTION: Final[str] = "hook"
PROVENANCE_RENDER_KEY: Final[str] = "provenance_render"
ENV_PROVENANCE_RENDER: Final[str] = "AELFRICE_PROVENANCE_RENDER"

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})

SECTION_LOCKED: Final[str] = "user-locked"
SECTION_OBSERVED: Final[str] = "observed"
SECTION_INFERRED: Final[str] = "inferred"

DEFAULT_SECTION: Final[str] = SECTION_INFERRED
"""Where an origin nobody has classified goes.

`<inferred>` rather than `<observed>`: the sections differ in how much the
framing text tells the model to trust them, so the fallback has to err
toward the tier that says "check this", not the one that says "this came
from the repository".
"""

SECTION_BY_ORIGIN: Final[dict[str, str]] = {
    # A person said it, typed it, or corrected it.
    ORIGIN_USER_STATED: SECTION_OBSERVED,
    ORIGIN_USER_CORRECTED: SECTION_OBSERVED,
    ORIGIN_USER_VALIDATED: SECTION_OBSERVED,
    ORIGIN_USER_TRANSCRIPT: SECTION_OBSERVED,
    # Read out of the repository rather than asserted by anyone, but still
    # a record of something that exists rather than a guess about it.
    ORIGIN_DOCUMENT_RECENT: SECTION_OBSERVED,
    # The agent wrote these. `agent_remembered` is an agent calling
    # `aelf remember`, which is agent authorship however deliberate, so it
    # sits with `agent_inferred` rather than with the user tiers.
    ORIGIN_AGENT_INFERRED: SECTION_INFERRED,
    ORIGIN_AGENT_REMEMBERED: SECTION_INFERRED,
    ORIGIN_SPECULATIVE: SECTION_INFERRED,
    # 14.3% of the live store. Named explicitly rather than left to the
    # fallback so the census above stays checkable against this table.
    ORIGIN_UNKNOWN: SECTION_INFERRED,
}

SECTION_ORDER: Final[tuple[str, ...]] = (
    SECTION_LOCKED,
    SECTION_OBSERVED,
    SECTION_INFERRED,
)
"""Render order. Locked first because the framing header addresses it first
and #1274 measured position 1 as a lock 100% of the time already."""

SECTION_FRAMING: Final[dict[str, str]] = {
    SECTION_LOCKED: (
        "standing instructions the user locked as ground truth; verify any "
        "factual claim against the actual project before relying on it."
    ),
    SECTION_OBSERVED: (
        "recorded from what the user said or what the repository contains. "
        "n is how much evidence stands behind each one and mu is its "
        "posterior — weigh a claim at n=2 far less than one at n=200."
    ),
    SECTION_INFERRED: (
        "the memory system's own hypotheses, not assertions by anyone. "
        "Treat them as questions to check against the project, never as "
        "fact, however high their mu."
    ),
}


def _env_override() -> bool | None:
    """True/False if the env var is set to a recognised value, else None."""
    raw = os.environ.get(ENV_PROVENANCE_RENDER)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_TRUTHY:
        return True
    if norm in _ENV_FALSY:
        return False
    return None


def _read_toml(start: Path | None = None) -> bool | None:
    """Read `[hook] provenance_render` from the nearest `.aelfrice.toml`.

    None on missing file / section / key / malformed TOML / non-bool value;
    never raises. A render flag must not be able to break the hook.
    """
    serr: IO[str] = sys.stderr
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    candidate.read_bytes().decode("utf-8", errors="replace"),
                )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(
                    f"aelfrice provenance_render: cannot read {candidate}: "
                    f"{exc}",
                    file=serr,
                )
                return None
            section_obj: Any = parsed.get(SECTION, {})
            if not isinstance(section_obj, dict):
                return None
            val: Any = cast(dict[str, Any], section_obj).get(
                PROVENANCE_RENDER_KEY
            )
            return val if isinstance(val, bool) else None
        if current.parent == current:
            break
        current = current.parent
    return None


def is_provenance_render_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the render flag. Precedence: env > explicit > TOML > off.

    Default-off is load-bearing: with it off the injected block must be
    byte-identical to the pre-#1326 output, including the validated framing
    header. Turning it on changes every belief line, so it is an opt-in.
    """
    env = _env_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml(start)
    if toml_value is not None:
        return toml_value
    return False


def section_for(belief: "Belief") -> str:
    """The section a belief renders in. Total over every input.

    `lock_level` is consulted first and wins outright: a locked belief is a
    standing instruction whatever its origin, which is what the framing
    header already tells the model. Origin decides the rest, and anything
    unrecognised falls to `DEFAULT_SECTION`.
    """
    if belief.lock_level == LOCK_USER:
        return SECTION_LOCKED
    return SECTION_BY_ORIGIN.get(belief.origin, DEFAULT_SECTION)


def evidence_attrs(belief: "Belief") -> str:
    """The evidence attributes for one non-locked belief line.

    Every value is already on the `Belief` in hand — measured on a live
    pack, `origin`, `alpha`, `beta` and `corroboration_count` were populated
    on 74/74 retrieved hits — so this adds no query and no store read.

    `mu` is rounded to 3 decimals and `n` to 1, matching the rounding the
    session-start `<core>` line has used since #1016. `origin` is emitted
    from the belief rather than from the section so a reader can tell
    `agent_inferred` from `speculative` inside `<inferred>`; it is a store
    value, so it is attribute-escaped like any other.
    """
    from aelfrice.hook import _escape_attr  # noqa: PLC0415 - render-time only

    alpha = belief.alpha or 0.0
    beta = belief.beta or 0.0
    n = alpha + beta
    mu = (alpha / n) if n else 0.0
    seen = belief.corroboration_count or 0
    return (
        f' origin="{_escape_attr(str(belief.origin))}"'
        f' n="{n:.1f}" mu="{mu:.3f}" seen="{seen}"'
    )


__all__ = [
    "DEFAULT_SECTION",
    "ENV_PROVENANCE_RENDER",
    "SECTION_BY_ORIGIN",
    "SECTION_FRAMING",
    "SECTION_INFERRED",
    "SECTION_LOCKED",
    "SECTION_OBSERVED",
    "SECTION_ORDER",
    "evidence_attrs",
    "is_provenance_render_enabled",
    "section_for",
]
