"""Stop-hook cadence policy — periodic-checkpoint rebuild firing (#749).

The host harness fires PreCompact when its own context-window
threshold is crossed. Empirical post-#746 data (6 fires across 2
sessions over 6 days at trigger_mode='threshold') showed that
maintainer-typical workloads `/clear` faster than the harness compacts,
so PreCompact-only ("P0" in #749 body terminology) misses most of the
state-recovery opportunities.

This module implements **P1 every-K-turns**: a deterministic, monotonic-
counter-keyed cadence policy that fires a rebuilder pass from the Stop
hook every K turn boundaries. The rebuilder's side effects (rebuild_log
entry, touch-state refresh) accumulate at the chosen cadence instead of
at whatever rate the harness chooses to compact.

P1 ships **default-OFF** behind a `[cadence] enabled` opt-in flag,
mirroring the bench-gate / ship-or-defer pattern used by the #480 meta-
belief consumers (B-F) and the #769 type-aware-compression flag. The
campaign to evaluate P1 against P2/P3 (token-budget watermark / turn-
density-aware) is the #749 tracker's broader scope; this module ships
the P1 mechanism only.

Determinism (#605): the firing predicate is a pure function of
``(fire_idx, k)`` — no wall-clock, no random sampling. A replay with
the same fire_idx sequence produces the same fire decisions.

Discretion (`ab96e9d3501b1c14`): cadence reads no content from
``~/.claude/``; the rebuilder pass it triggers operates on the same
public surfaces PreCompact already drives.

Resolver shape: ``resolve_*`` functions follow the env > kwarg > TOML >
default precedence used by every other ``.aelfrice.toml``-backed knob
in the project. Future #480 meta-belief integration (``meta:hook.
cadence_k``) slots in between env and kwarg without changing call
sites — see ``resolve_cadence_k`` for the placement.
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
CADENCE_SECTION: Final[str] = "cadence"

ENABLED_KEY: Final[str] = "enabled"
POLICY_KEY: Final[str] = "policy"
K_KEY: Final[str] = "k"

ENV_CADENCE_ENABLED: Final[str] = "AELFRICE_CADENCE_ENABLED"
ENV_CADENCE_POLICY: Final[str] = "AELFRICE_CADENCE_POLICY"
ENV_CADENCE_K: Final[str] = "AELFRICE_CADENCE_K"

POLICY_OFF: Final[str] = "off"
POLICY_P1_EVERY_K_TURNS: Final[str] = "p1_every_k_turns"

_VALID_POLICIES: Final[frozenset[str]] = frozenset({
    POLICY_OFF,
    POLICY_P1_EVERY_K_TURNS,
})

DEFAULT_ENABLED: Final[bool] = False
DEFAULT_POLICY: Final[str] = POLICY_OFF
DEFAULT_K: Final[int] = 15
"""K=15 turns is the body-suggested literature-typical starting point per
the #749 §"Pre-registered policies to evaluate" P1 entry. Operator
override via `[cadence] k` once empirical evidence narrows the band."""

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class CadenceConfig:
    """Resolved `[cadence]` section of `.aelfrice.toml`.

    Three fields. ``enabled`` gates the whole feature: when False, no
    cadence fire ever happens regardless of policy/k. ``policy`` selects
    the firing pattern; only ``p1_every_k_turns`` is implemented at
    v3.x. ``k`` parameterises P1 (turns between fires).

    Defaults are off / off / 15. The K default ships even when enabled
    and policy aren't set, so a half-configured TOML is well-defined.
    """
    enabled: bool = DEFAULT_ENABLED
    policy: str = DEFAULT_POLICY
    k: int = DEFAULT_K


def _env_bool(name: str) -> bool | None:
    """Parse env truthy/falsy; return None on unset / unparseable."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if not norm:
        return None
    if norm in _ENV_TRUTHY:
        return True
    if norm in _ENV_FALSY:
        return False
    return None


def _env_positive_int(name: str) -> int | None:
    """Parse env positive-int; return None on unset / non-int / non-positive."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    norm = raw.strip()
    if not norm:
        return None
    try:
        val = int(norm)
    except ValueError:
        return None
    if val <= 0:
        return None
    return val


def _env_policy() -> str | None:
    """Parse env policy string; return None on unset / unknown policy."""
    raw = os.environ.get(ENV_CADENCE_POLICY)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _VALID_POLICIES:
        return norm
    return None


def load_cadence_config(start: Path | None = None) -> CadenceConfig:
    """Walk up from ``start`` looking for ``.aelfrice.toml``.

    Returns the resolved ``[cadence]`` section. Missing file / missing
    section / malformed TOML / wrong-typed values all degrade to
    defaults; never raises. Wrong-typed values trace to stderr,
    matching the ``RebuilderConfig`` / ``UserPromptSubmitConfig``
    fail-soft contract.
    """
    serr: IO[str] = sys.stderr
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            try:
                raw = candidate.read_bytes()
            except OSError as exc:
                print(
                    f"aelfrice cadence: cannot read {candidate}: {exc}",
                    file=serr,
                )
                return CadenceConfig()
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    raw.decode("utf-8", errors="replace"),
                )
            except tomllib.TOMLDecodeError as exc:
                print(
                    f"aelfrice cadence: malformed TOML in {candidate}: {exc}",
                    file=serr,
                )
                return CadenceConfig()
            section_obj: Any = parsed.get(CADENCE_SECTION, {})
            if not isinstance(section_obj, dict):
                return CadenceConfig()
            section = section_obj
            enabled = _read_bool(section, ENABLED_KEY, DEFAULT_ENABLED, candidate, serr)
            policy = _read_policy(section, candidate, serr)
            k = _read_k(section, candidate, serr)
            return CadenceConfig(enabled=enabled, policy=policy, k=k)
        if current.parent == current:
            break
        current = current.parent
    return CadenceConfig()


def _read_bool(
    section: dict[str, Any],
    key: str,
    default: bool,
    candidate: Path,
    serr: IO[str],
) -> bool:
    if key not in section:
        return default
    value = section[key]
    if isinstance(value, bool):
        return value
    print(
        f"aelfrice cadence: ignoring [{CADENCE_SECTION}] {key} "
        f"in {candidate} (expected bool)",
        file=serr,
    )
    return default


def _read_policy(
    section: dict[str, Any],
    candidate: Path,
    serr: IO[str],
) -> str:
    if POLICY_KEY not in section:
        return DEFAULT_POLICY
    value = section[POLICY_KEY]
    if isinstance(value, str) and value in _VALID_POLICIES:
        return value
    print(
        f"aelfrice cadence: ignoring [{CADENCE_SECTION}] {POLICY_KEY} "
        f"in {candidate} (expected one of {sorted(_VALID_POLICIES)})",
        file=serr,
    )
    return DEFAULT_POLICY


def _read_k(
    section: dict[str, Any],
    candidate: Path,
    serr: IO[str],
) -> int:
    if K_KEY not in section:
        return DEFAULT_K
    value = section[K_KEY]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        print(
            f"aelfrice cadence: ignoring [{CADENCE_SECTION}] {K_KEY} "
            f"in {candidate} (expected positive int)",
            file=serr,
        )
        return DEFAULT_K
    return value


def resolve_cadence_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the cadence enabled flag.

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_ENABLED`` env var.
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] enabled`` in ``.aelfrice.toml``.
      4. Default: ``False`` — ships default-OFF per #749 §"Why this is
         parked" reasoning. Operator opts in once subjective experience
         or empirical evidence (P0 fire-rate insufficient) justifies it.

    The env tier exists so an operator can flip cadence on for a
    single shell without committing a TOML change. The kwarg tier is
    for callers that want to inject behaviour in tests / replay.
    """
    env = _env_bool(ENV_CADENCE_ENABLED)
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    return load_cadence_config(start).enabled


def resolve_cadence_policy(
    explicit: str | None = None,
    *,
    start: Path | None = None,
) -> str:
    """Resolve the cadence policy string.

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_POLICY`` env var.
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] policy`` in ``.aelfrice.toml``.
      4. Default: ``"off"`` — even with ``enabled=True``, an unset
         policy means no fire ever happens.

    Currently recognised policies: ``off``, ``p1_every_k_turns``.
    P2 (token-budget watermark) and P3 (turn-density-aware) per #749
    are unimplemented; the policy string is forward-compatible so
    those land as new enum values without changing this resolver.
    """
    env = _env_policy()
    if env is not None:
        return env
    if explicit is not None and explicit in _VALID_POLICIES:
        return explicit
    return load_cadence_config(start).policy


def resolve_cadence_k(
    explicit: int | None = None,
    *,
    start: Path | None = None,
) -> int:
    """Resolve the cadence-K (turns between fires for P1).

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_K`` env var (positive int).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] k`` in ``.aelfrice.toml``.
      4. Default: ``15`` — body-suggested literature-typical starting
         point per #749 §"Pre-registered policies to evaluate".

    Future #480 meta-belief integration: a ``meta:hook.cadence_k``
    read can slot in between env and ``explicit`` — read the meta-
    belief, decode via the log-linear bounded scheme already used by
    #756, and pass the decoded value as the new tier-2 input. No
    call-site changes required at integration time.
    """
    env = _env_positive_int(ENV_CADENCE_K)
    if env is not None:
        return env
    if explicit is not None and explicit > 0:
        return explicit
    return load_cadence_config(start).k


def should_fire(fire_idx: int, config: CadenceConfig) -> bool:
    """Return True iff cadence should fire at this ``fire_idx``.

    Pure function of ``(fire_idx, config)``. Determinism (#605): a
    replay with the same fire_idx sequence and same config produces
    the same fire decisions. No wall-clock, no I/O.

    Conditions:
      * ``config.enabled`` is True.
      * ``config.policy`` is ``p1_every_k_turns``.
      * ``config.k`` is positive.
      * ``fire_idx`` is positive (the zeroth turn has no prior state
        worth checkpointing — also avoids ``0 % k == 0`` firing at
        session start, which would be a spurious cold-start fire).
      * ``fire_idx % config.k == 0`` — i.e. ``fire_idx`` is a multiple
        of ``k``. With k=15: fires at 15, 30, 45, ...

    Returns False on any condition unmet. Never raises.
    """
    if not config.enabled:
        return False
    if config.policy != POLICY_P1_EVERY_K_TURNS:
        return False
    if config.k <= 0:
        return False
    if fire_idx <= 0:
        return False
    return fire_idx % config.k == 0
