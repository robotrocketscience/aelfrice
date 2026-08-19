"""#1382 — the per-session injection ledger behind turn-differential rendering.

Records which beliefs have already been rendered **verbatim** into the context
of the current session epoch, so a later turn can emit a one-line reference
instead of the identical block again.

**Opt-in.** `AELFRICE_TURN_DIFFERENTIAL=1` turns it on; see
`is_turn_differential_enabled` for why the default is off.

## No prize figure is quoted here

An earlier version of this docstring published "32% of injected block tokens
as-is, 43% against the post-#1344 baseline". Those numbers were measured on a
different corpus, nothing in this tree re-derives them, and the second is
contradicted by direct measurement: a `seen` entry opens the
`<aelfrice-locks-manifest>` wrapper on a block that had none, a fixed ~237
characters, so a single-belief block must carry more than ~310 characters of
content before the change saves anything at all. Recorded median content
length is 86. The saving is real for long or many-belief blocks and negative
for the modal small one.

## The epoch boundary

A boundary is any event after which text already injected can no longer be
assumed present in the window. Two fire it:

- **SessionStart**, which re-emits the baseline into a fresh or compacted
  window. The reset runs *before* the store read, because that read can raise
  or be killed at the hook timeout, and a boundary skipped on failure leaves
  the previous epoch live under an unchanged `session_id`.
- **PreCompact**, before every early return in that hook. **This hook is
  opt-in** (`aelf setup --rebuilder`) and is absent from the default manifest,
  so on a default install SessionStart is the only reset.

A `session_id` change is a boundary too: reading a ledger written under a
different session yields the empty set. A boundary that carries *no*
`session_id` calls `invalidate`, because returning early would leave the
previous epoch live under an id the next fire may match.

## Fail-soft direction, and its one exception

Every read failure returns "nothing rendered yet", so the block is emitted
verbatim. A missing, unreadable, malformed or foreign ledger costs redundant
tokens.

**This is not the same as "the mechanism can never suppress".** That stronger
claim was made for the 2026-08-11 default-ON ruling and is false: a boundary
that does not fire — a store read that raises, a host that compacts without
firing either hook, `aelf setup --no-session-start` — leaves stale ids live,
and the beliefs they name render as one-line stubs the model was never shown.
The boundaries above close the cases that were reachable in this codebase.
They do not turn the guarantee into a property of the design, which is the
second reason the default is off.

Storage is a separate file from `session_first_prompt.json` on purpose. That
file already serves two consumers with different update rules (the first-fire
window and `aelf scope-out`'s active-session resolution), and adding a third
with a third rule is how the #1346 regression happened.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

LEDGER_FILENAME: Final[str] = "session_injection_ledger.json"

TURN_DIFFERENTIAL_ENV_VAR: Final[str] = "AELFRICE_TURN_DIFFERENTIAL"
"""Set to a truthy value (1/true/yes/on) to enable the turn-differential; the
default is off. A falsy value (0/false/no/off) pins the pre-#1382 behaviour
explicitly, which also survives a future flip of the default."""

_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _env_override() -> bool | None:
    """Env value as a tristate. `None` means unset, so a caller's explicit
    argument still wins — returning False for "unset" is the bug that makes a
    resolver silently un-overridable."""
    raw = os.environ.get(TURN_DIFFERENTIAL_ENV_VAR)
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    return None


def is_turn_differential_enabled(explicit: bool | None = None) -> bool:
    """Resolve the turn-differential flag. Precedence: env > explicit > off.

    Default-**off**, ratified 2026-08-19. The 2026-08-11 ruling shipped this
    default-on on one argument — that the failure direction is one-way, so a
    broken ledger can only cost redundant tokens. Two measurements retired it:

    1. **The one-way claim was false.** `begin_epoch` sat inside a
       `if body:` guard in `session_start`, so a baseline that rendered
       nothing (a store with no locked beliefs) never opened an epoch. The
       pre-compaction ledger survived under the same `session_id` and later
       turns emitted `seen <id>` for text the window no longer held. That is
       fixed — the reset is unconditional and PreCompact resets too — but the
       argument was load-bearing for the default, not just for the code.

    2. **The prize is negative at the median.** A `seen` entry forces the
       `<aelfrice-locks-manifest>` wrapper open on blocks that emitted none,
       a fixed ~237 characters. Break-even is roughly 310 characters for a
       single-belief block; the recorded candidate distribution has p50 = 86.
       Small blocks of short beliefs — the modal case — get bigger.

    So it stays behind the flag until the wrapper cost is addressed and the
    saving is measured on a real store rather than asserted.
    """
    env = _env_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    return False


def ledger_path() -> Path | None:
    """Sibling of memory.db under `<git-common-dir>/aelfrice/`, or None for an
    in-memory store (tests with no real path), so callers gate on None."""
    try:
        from aelfrice.db_paths import db_path  # noqa: PLC0415

        p = db_path()
    except Exception:
        return None
    if str(p) == ":memory:":
        return None
    return p.parent / LEDGER_FILENAME


def read_rendered(
    session_id: str | None, *, path: Path | None = None
) -> frozenset[str]:
    """Belief ids already rendered verbatim in this session's epoch.

    Empty when the ledger is missing, unreadable, malformed, or was written
    under a different `session_id` — all of which mean "render verbatim", the
    behaviour that predates this file.
    """
    if not session_id:
        return frozenset()
    p = path if path is not None else ledger_path()
    if p is None or not p.exists():
        return frozenset()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # ValueError, not JSONDecodeError: non-UTF-8 bytes raise
        # UnicodeDecodeError out of read_text, a sibling of JSONDecodeError
        # under ValueError rather than of OSError.
        return frozenset()
    if not isinstance(data, dict):
        return frozenset()
    if data.get("session_id") != session_id:
        return frozenset()  # a different epoch's ledger is not ours to trust
    raw = data.get("rendered")
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(s for s in raw if isinstance(s, str) and s)


def _write(session_id: str, ids: frozenset[str], path: Path | None) -> None:
    p = path if path is not None else ledger_path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                # Sorted so the file is byte-stable across runs with the same
                # content — an unstable ordering makes every turn look like a
                # change to anything diffing or checksumming it.
                {"session_id": session_id, "rendered": sorted(ids)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, p)
    except OSError:
        # Non-blocking hook contract: an unwritable ledger costs redundant
        # verbatim injections next turn, which is the safe direction.
        return


def begin_epoch(
    session_id: str | None, ids: frozenset[str], *, path: Path | None = None
) -> None:
    """Reset the ledger to exactly `ids` — called on a SessionStart fire.

    Replaces rather than unions: the previous epoch's verbatim text is not in
    this context window, so carrying its ids forward would suppress content the
    model has never seen.
    """
    if not session_id:
        return
    _write(session_id, frozenset(ids), path)


def invalidate(*, path: Path | None = None) -> bool:
    """Drop the ledger entirely — no epoch can be scoped, so suppress nothing.

    Called at an epoch boundary whose `session_id` is unknown (an absent id in
    the payload, or a payload that failed to parse). `begin_epoch` returns
    early on a falsy id, which would leave the PREVIOUS epoch's file in place
    under an id the next fire may well match — and that is an under-injection
    path, not a fail-soft one. Removing the file makes the next `read_rendered`
    return the empty set, i.e. render everything verbatim.

    **Returns whether the ledger is now gone**, unlike the other writers here,
    which return None because their failure mode is benign (a redundant
    verbatim injection). This one is not: a stale file that cannot be removed
    goes on suppressing content, so a caller that reported "epoch reset" on a
    failed unlink would be asserting the opposite of what happened. The hook
    contract forbids raising, so the signal is a return value and the caller
    surfaces it on stderr.
    """
    p = path if path is not None else ledger_path()
    if p is None:
        return True  # nothing to remove: no on-disk ledger can suppress
    try:
        p.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def record_rendered(
    session_id: str | None, ids: frozenset[str], *, path: Path | None = None
) -> None:
    """Union `ids` into the current epoch's ledger — called after a turn renders.

    A `session_id` that does not match the stored one starts a fresh epoch
    rather than merging into a foreign one.
    """
    if not session_id or not ids:
        return
    _write(session_id, read_rendered(session_id, path=path) | frozenset(ids), path)


__all__ = [
    "LEDGER_FILENAME",
    "invalidate",
    "TURN_DIFFERENTIAL_ENV_VAR",
    "begin_epoch",
    "is_turn_differential_enabled",
    "ledger_path",
    "read_rendered",
    "record_rendered",
]
