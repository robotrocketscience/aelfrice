"""Stop-hook cadence policy — periodic-checkpoint rebuild firing (#749).

The host harness fires PreCompact when its own context-window
threshold is crossed. Empirical post-#746 data (6 fires across 2
sessions over 6 days at trigger_mode='threshold') showed that
maintainer-typical workloads `/clear` faster than the harness compacts,
so PreCompact-only ("P0" in #749 body terminology) misses most of the
state-recovery opportunities.

This module implements two of the policies pre-registered by #749:

* **P1 every-K-turns** (#749 §"Pre-registered policies to evaluate"):
  a deterministic, monotonic-counter-keyed cadence policy that fires
  a rebuilder pass from the Stop hook every K turn boundaries.
* **P2 ctx-threshold + phase-boundary** (#871, refines #749 P2):
  fires when the transcript byte-count exceeds a configurable
  fraction of a configured byte-window AND the most-recent user
  prompt looks like a task-boundary signal. Composite predicate;
  both conditions must hold.

The rebuilder's side effects (rebuild_log entry, touch-state refresh)
accumulate at the chosen cadence instead of at whatever rate the
harness chooses to compact.

P1 ships **default-OFF** behind a `[cadence] enabled` opt-in flag.
P2 inherits the same flag; selecting between policies is per-project
via `[cadence] policy`.

Determinism (#605): all firing predicates are pure functions of
their inputs — no wall-clock, no random sampling. P1 replays from
``(fire_idx, k)``; P2 replays from ``(transcript_bytes, last_user_prompt,
config)``. The transcript byte-count is reproducible from filesystem
state; replay with the same transcript file produces the same fire
decision.

Discretion (`ab96e9d3501b1c14`): cadence reads no content from
``~/.claude/``; the rebuilder pass it triggers operates on the same
public surfaces PreCompact already drives. The phase-boundary
allowlist is a closed set of generic acknowledgment tokens with no
host-product references.

Resolver shape: ``resolve_*`` functions follow the env > kwarg > TOML >
default precedence used by every other ``.aelfrice.toml``-backed knob
in the project.
"""
from __future__ import annotations

import json
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
CTX_THRESHOLD_KEY: Final[str] = "ctx_threshold"
CTX_BYTE_WINDOW_KEY: Final[str] = "ctx_byte_window"
SHADOW_MODE_ENABLED_KEY: Final[str] = "shadow_mode_enabled"
CADENCE_SHADOW_DIRNAME: Final[str] = "cadence_shadow"
P3_VELOCITY_THRESHOLD_KEY: Final[str] = "p3_velocity_threshold"
P3_SUBSTANTIVE_WINDOW_KEY: Final[str] = "p3_substantive_window"
P3_SUBSTANTIVE_THRESHOLD_KEY: Final[str] = "p3_substantive_threshold"

ENV_CADENCE_ENABLED: Final[str] = "AELFRICE_CADENCE_ENABLED"
ENV_CADENCE_POLICY: Final[str] = "AELFRICE_CADENCE_POLICY"
ENV_CADENCE_K: Final[str] = "AELFRICE_CADENCE_K"
ENV_CADENCE_CTX_THRESHOLD: Final[str] = "AELFRICE_CADENCE_CTX_THRESHOLD"
ENV_CADENCE_CTX_BYTE_WINDOW: Final[str] = "AELFRICE_CADENCE_CTX_BYTE_WINDOW"
ENV_CADENCE_SHADOW_MODE_ENABLED: Final[str] = "AELFRICE_CADENCE_SHADOW_MODE_ENABLED"
ENV_CADENCE_P3_VELOCITY_THRESHOLD: Final[str] = (
    "AELFRICE_CADENCE_P3_VELOCITY_THRESHOLD"
)
ENV_CADENCE_P3_SUBSTANTIVE_WINDOW: Final[str] = (
    "AELFRICE_CADENCE_P3_SUBSTANTIVE_WINDOW"
)
ENV_CADENCE_P3_SUBSTANTIVE_THRESHOLD: Final[str] = (
    "AELFRICE_CADENCE_P3_SUBSTANTIVE_THRESHOLD"
)

POLICY_OFF: Final[str] = "off"
POLICY_P1_EVERY_K_TURNS: Final[str] = "p1_every_k_turns"
POLICY_P2_CTX_THRESHOLD: Final[str] = "p2_ctx_threshold"
POLICY_P3_VELOCITY: Final[str] = "p3_velocity"
POLICY_P3_SUBSTANTIVE: Final[str] = "p3_substantive"

_VALID_POLICIES: Final[frozenset[str]] = frozenset({
    POLICY_OFF,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
    POLICY_P3_VELOCITY,
    POLICY_P3_SUBSTANTIVE,
})

DEFAULT_ENABLED: Final[bool] = False
DEFAULT_POLICY: Final[str] = POLICY_OFF
DEFAULT_K: Final[int] = 15
"""K=15 turns is the body-suggested literature-typical starting point per
the #749 §"Pre-registered policies to evaluate" P1 entry. Operator
override via `[cadence] k` once empirical evidence narrows the band."""

DEFAULT_CTX_THRESHOLD: Final[float] = 0.50
"""Aligns with #749 P2's "flush at 50% / 80%" hypothesis — the lower
bound. P2 first fires when transcript bytes cross 50% of the window."""

DEFAULT_CTX_BYTE_WINDOW: Final[int] = 600_000
"""~200K tokens × ~3 bytes/token ≈ 600K bytes. English transcripts
average ~3.5 bytes/token, code-heavy ~4-5; the operator-side ctx%
reading is ground truth and this default gets re-tuned after
empirical use. Configurable per-project via TOML."""

DEFAULT_SHADOW_MODE_ENABLED: Final[bool] = False
"""Shadow-evaluation mode (#875): when True, every Stop-hook tick
evaluates would_fire_* for every implemented policy and logs a row
to .git/aelfrice/cadence_shadow/<session_id>.jsonl. The selected
policy still drives live firing; the log is for offline scoring of
non-selected policy decisions on identical workload. Ships default-
OFF: most operators do not need the comparison data, and the per-
tick predicate evaluations + JSONL write are a non-trivial extra
I/O cost."""

DEFAULT_P3_VELOCITY_THRESHOLD: Final[int] = 3000
"""Bytes-per-turn floor that triggers p3_velocity (#876 axis 1).
3000 bytes/turn ≈ ~1000 tokens/turn (English) or ~750 tokens/turn
(code-heavy) — the rough boundary between exploratory back-and-forth
and dense working session. Placeholder per the issue body; the
operator tunes via `[cadence] p3_velocity_threshold = N` after
empirical use, same posture as K=15 for P1."""

DEFAULT_P3_SUBSTANTIVE_WINDOW: Final[int] = 10
"""Number of last-N turns over which p3_substantive counts
substantive turns. 10-turn window mirrors P1's K=15 starting point
at a tighter horizon — the substantive predicate is denser than
P1's pure counter so a smaller window is more sensitive."""

DEFAULT_P3_SUBSTANTIVE_THRESHOLD: Final[float] = 0.6
"""Substantive-ratio floor (0.0-1.0). p3_substantive fires when
substantive_count / p3_substantive_window >= threshold. 0.6 ≈
6-of-10 turns must be substantive (non-boundary). Same placeholder
posture as p3_velocity_threshold."""

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})

# Phase-boundary detector: closed allowlist of generic acknowledgment
# / transition tokens. Calibration starts conservative — false negatives
# (missing a real boundary) are recoverable by the operator clearing
# manually; false positives (firing mid-task) are not, so the list
# stays short and unambiguous.
_PHASE_BOUNDARY_PHRASES: Final[frozenset[str]] = frozenset({
    # Pure acknowledgments
    "done",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "ok thanks",
    "okay thanks",
    "ok cool",
    "okay cool",
    "ok great",
    "okay great",
    "great",
    "awesome",
    "perfect",
    "nice",
    "got it",
    "sounds good",
    "good",
    "good to go",
    "all good",
    "looks good",
    # Transition signals (exact match only; longer forms via prefix)
    "next",
    "next task",
    "next step",
    "move on",
    "ship it",
    "merge it",
})

# Prefix-match boundary phrases. The user's prompt is a boundary if
# its normalized form *starts with* one of these (with the trailing
# space included to enforce a word boundary).
_PHASE_BOUNDARY_PREFIXES: Final[tuple[str, ...]] = (
    "switch to ",
    "lets switch to ",
    "let us switch to ",
    "lets move on ",
    "let us move on ",
    "now lets ",
    "now let us ",
    "moving on to ",
    "lets do ",
    "let us do ",
    "next lets ",
    "next let us ",
)

_PHASE_BOUNDARY_MAX_LEN: Final[int] = 80
"""Boundary signals are short. A long user message — even one that
begins with "ok" — almost always contains substantive work after the
ack token, so treating it as a boundary would over-fire."""


@dataclass(frozen=True)
class CadenceConfig:
    """Resolved `[cadence]` section of `.aelfrice.toml`.

    Six fields. ``enabled`` gates the whole feature: when False, no
    cadence fire ever happens regardless of policy. ``policy`` selects
    the firing pattern. ``shadow_mode_enabled`` is an independent
    opt-in for #875 shadow-evaluation logging.

    P1-specific: ``k`` parameterises the every-K-turns interval.
    P2-specific: ``ctx_threshold`` (fraction of window) and
    ``ctx_byte_window`` (absolute byte count) parameterise the
    ctx-threshold predicate. The phase-boundary half of P2 is not
    configurable here — its allowlist lives at module scope.

    Defaults are off / off / 15 / 0.50 / 600000 / off. Each field
    has a default that makes a half-configured TOML well-defined.
    """
    enabled: bool = DEFAULT_ENABLED
    policy: str = DEFAULT_POLICY
    k: int = DEFAULT_K
    ctx_threshold: float = DEFAULT_CTX_THRESHOLD
    ctx_byte_window: int = DEFAULT_CTX_BYTE_WINDOW
    shadow_mode_enabled: bool = DEFAULT_SHADOW_MODE_ENABLED
    p3_velocity_threshold: int = DEFAULT_P3_VELOCITY_THRESHOLD
    p3_substantive_window: int = DEFAULT_P3_SUBSTANTIVE_WINDOW
    p3_substantive_threshold: float = DEFAULT_P3_SUBSTANTIVE_THRESHOLD


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


def _env_unit_float(name: str) -> float | None:
    """Parse env float in (0, 1]; return None on unset / out-of-range."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    norm = raw.strip()
    if not norm:
        return None
    try:
        val = float(norm)
    except ValueError:
        return None
    if val <= 0 or val > 1:
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
                print(
                    f"aelfrice cadence: ignoring [{CADENCE_SECTION}] "
                    f"in {candidate} (expected table)",
                    file=serr,
                )
                return CadenceConfig()
            section = section_obj
            enabled = _read_bool(section, ENABLED_KEY, DEFAULT_ENABLED, candidate, serr)
            policy = _read_policy(section, candidate, serr)
            k = _read_k(section, candidate, serr)
            ctx_threshold = _read_unit_float(
                section, CTX_THRESHOLD_KEY, DEFAULT_CTX_THRESHOLD, candidate, serr,
            )
            ctx_byte_window = _read_positive_int(
                section, CTX_BYTE_WINDOW_KEY, DEFAULT_CTX_BYTE_WINDOW,
                candidate, serr,
            )
            shadow_mode_enabled = _read_bool(
                section, SHADOW_MODE_ENABLED_KEY,
                DEFAULT_SHADOW_MODE_ENABLED, candidate, serr,
            )
            p3_velocity_threshold = _read_positive_int(
                section, P3_VELOCITY_THRESHOLD_KEY,
                DEFAULT_P3_VELOCITY_THRESHOLD, candidate, serr,
            )
            p3_substantive_window = _read_positive_int(
                section, P3_SUBSTANTIVE_WINDOW_KEY,
                DEFAULT_P3_SUBSTANTIVE_WINDOW, candidate, serr,
            )
            p3_substantive_threshold = _read_unit_float(
                section, P3_SUBSTANTIVE_THRESHOLD_KEY,
                DEFAULT_P3_SUBSTANTIVE_THRESHOLD, candidate, serr,
            )
            return CadenceConfig(
                enabled=enabled,
                policy=policy,
                k=k,
                ctx_threshold=ctx_threshold,
                ctx_byte_window=ctx_byte_window,
                shadow_mode_enabled=shadow_mode_enabled,
                p3_velocity_threshold=p3_velocity_threshold,
                p3_substantive_window=p3_substantive_window,
                p3_substantive_threshold=p3_substantive_threshold,
            )
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


def _read_unit_float(
    section: dict[str, Any],
    key: str,
    default: float,
    candidate: Path,
    serr: IO[str],
) -> float:
    """Read a float field in (0, 1]; fall back to default on mismatch."""
    if key not in section:
        return default
    value = section[key]
    if isinstance(value, bool):
        # bool is a subclass of int — guard explicitly.
        pass
    elif isinstance(value, (int, float)):
        as_float = float(value)
        if 0 < as_float <= 1:
            return as_float
    print(
        f"aelfrice cadence: ignoring [{CADENCE_SECTION}] {key} "
        f"in {candidate} (expected float in (0, 1])",
        file=serr,
    )
    return default


def _read_positive_int(
    section: dict[str, Any],
    key: str,
    default: int,
    candidate: Path,
    serr: IO[str],
) -> int:
    """Read a positive-int field; fall back to default on mismatch."""
    if key not in section:
        return default
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        print(
            f"aelfrice cadence: ignoring [{CADENCE_SECTION}] {key} "
            f"in {candidate} (expected positive int)",
            file=serr,
        )
        return default
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

    Currently recognised policies: ``off``, ``p1_every_k_turns``,
    ``p2_ctx_threshold``. P3 (turn-density-aware) per #749 is
    unimplemented; the policy string is forward-compatible.
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
      4. Default: ``15``.
    """
    env = _env_positive_int(ENV_CADENCE_K)
    if env is not None:
        return env
    if explicit is not None and explicit > 0:
        return explicit
    return load_cadence_config(start).k


def resolve_cadence_ctx_threshold(
    explicit: float | None = None,
    *,
    start: Path | None = None,
) -> float:
    """Resolve the P2 ctx-threshold (fraction of ctx_byte_window).

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_CTX_THRESHOLD`` env var (float in (0, 1]).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] ctx_threshold`` in ``.aelfrice.toml``.
      4. Default: ``0.50``.
    """
    env = _env_unit_float(ENV_CADENCE_CTX_THRESHOLD)
    if env is not None:
        return env
    if explicit is not None and 0 < explicit <= 1:
        return explicit
    return load_cadence_config(start).ctx_threshold


def resolve_cadence_ctx_byte_window(
    explicit: int | None = None,
    *,
    start: Path | None = None,
) -> int:
    """Resolve the P2 ctx_byte_window (absolute byte budget).

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_CTX_BYTE_WINDOW`` env var (positive int).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] ctx_byte_window`` in ``.aelfrice.toml``.
      4. Default: ``600000``.
    """
    env = _env_positive_int(ENV_CADENCE_CTX_BYTE_WINDOW)
    if env is not None:
        return env
    if explicit is not None and explicit > 0:
        return explicit
    return load_cadence_config(start).ctx_byte_window


def resolve_cadence_shadow_mode_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the cadence shadow-mode opt-in flag (#875).

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_SHADOW_MODE_ENABLED`` env var.
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] shadow_mode_enabled`` in ``.aelfrice.toml``.
      4. Default: ``False`` — shadow logging is opt-in.

    When True, the Stop-hook computes :func:`would_fire_p1` /
    :func:`would_fire_p2` for every implemented policy on each tick
    and writes one row to ``.git/aelfrice/cadence_shadow/<sid>.jsonl``.
    Selected policy still drives live firing; non-selected decisions
    are log-only. Designed to unblock the P1 vs P2 vs (eventual) P3
    comparison that the #749 campaign was opened to produce, without
    requiring longitudinal flip-and-rebake with workload-drift
    confounds.
    """
    env = _env_bool(ENV_CADENCE_SHADOW_MODE_ENABLED)
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    return load_cadence_config(start).shadow_mode_enabled


def resolve_cadence_p3_velocity_threshold(
    explicit: int | None = None,
    *,
    start: Path | None = None,
) -> int:
    """Resolve the P3 velocity threshold (bytes/turn) — #876 axis 1.

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_P3_VELOCITY_THRESHOLD`` env var (positive int).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] p3_velocity_threshold`` in ``.aelfrice.toml``.
      4. Default: ``3000`` bytes/turn.
    """
    env = _env_positive_int(ENV_CADENCE_P3_VELOCITY_THRESHOLD)
    if env is not None:
        return env
    if explicit is not None and explicit > 0:
        return explicit
    return load_cadence_config(start).p3_velocity_threshold


def resolve_cadence_p3_substantive_window(
    explicit: int | None = None,
    *,
    start: Path | None = None,
) -> int:
    """Resolve the P3 substantive-turn window (turns) — #876 axis 1.

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_P3_SUBSTANTIVE_WINDOW`` env var (positive int).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] p3_substantive_window`` in ``.aelfrice.toml``.
      4. Default: ``10`` turns.
    """
    env = _env_positive_int(ENV_CADENCE_P3_SUBSTANTIVE_WINDOW)
    if env is not None:
        return env
    if explicit is not None and explicit > 0:
        return explicit
    return load_cadence_config(start).p3_substantive_window


def resolve_cadence_p3_substantive_threshold(
    explicit: float | None = None,
    *,
    start: Path | None = None,
) -> float:
    """Resolve the P3 substantive-ratio threshold (0.0-1.0) — #876 axis 1.

    Precedence (first decisive wins):
      1. ``AELFRICE_CADENCE_P3_SUBSTANTIVE_THRESHOLD`` env var (unit float).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[cadence] p3_substantive_threshold`` in ``.aelfrice.toml``.
      4. Default: ``0.6``.
    """
    env = _env_unit_float(ENV_CADENCE_P3_SUBSTANTIVE_THRESHOLD)
    if env is not None:
        return env
    if explicit is not None and 0.0 <= explicit <= 1.0:
        return explicit
    return load_cadence_config(start).p3_substantive_threshold


def is_substantive_turn(prompt: str | None) -> bool:
    """Classify a user prompt as substantive (vs phase-boundary).

    Pure function, inverse of :func:`is_phase_boundary_signal`. A turn
    is "substantive" when the user prompt does NOT match the closed
    phase-boundary allowlist used by P2. Empty or None prompts return
    False (no signal to count).

    Used by P3 substantive-window counting (#876 axis 1, option C).
    The classifier reuses P2's allowlist by inversion so the same
    closed-set maintenance burden covers both predicates.
    """
    if not prompt or not prompt.strip():
        return False
    return not is_phase_boundary_signal(prompt)


def would_fire_p1(
    *, fire_idx: int, config: CadenceConfig
) -> tuple[bool, str]:
    """P1 policy-agnostic predicate — every-K-turns.

    Pure function of ``(fire_idx, config)``. Determinism (#605): a
    replay with the same inputs reproduces the same ``(bool, reason)``.
    No wall-clock, no I/O.

    Unlike :func:`should_fire`, this does **not** check
    ``config.policy``. It answers "would P1 fire here, evaluated as
    if it were the selected policy" — the predicate used by shadow-
    evaluation mode (#875) to log non-selected policy decisions.

    Conditions for True:
      * ``config.enabled`` is True.
      * ``config.k`` is positive.
      * ``fire_idx`` is positive.
      * ``fire_idx % config.k == 0``.

    Returns ``(False, reason)`` on any condition unmet, where
    ``reason`` is a short human-readable diagnostic string suitable
    for logging.
    """
    if not config.enabled:
        return (False, "cadence disabled")
    if config.k <= 0:
        return (False, f"k={config.k} not positive")
    if fire_idx <= 0:
        return (False, f"fire_idx={fire_idx} not positive")
    rem = fire_idx % config.k
    if rem != 0:
        return (False, f"fire_idx={fire_idx} mod k={config.k} = {rem}")
    return (True, f"fire_idx={fire_idx} mod k={config.k} = 0")


def should_fire(fire_idx: int, config: CadenceConfig) -> bool:
    """P1 firing predicate — every-K-turns, policy-gated.

    Pure function of ``(fire_idx, config)``. Determinism (#605): a
    replay with the same fire_idx sequence and same config produces
    the same fire decisions. No wall-clock, no I/O.

    Conditions:
      * ``config.policy`` is ``p1_every_k_turns``.
      * :func:`would_fire_p1` returns True for ``(fire_idx, config)``.

    Returns False on any condition unmet. Never raises.

    For P2 firing decisions, callers must use :func:`should_fire_p2`
    — the inputs and predicate differ. For shadow-evaluation (non-
    selected policy logging), see :func:`would_fire_p1`.
    """
    if config.policy != POLICY_P1_EVERY_K_TURNS:
        return False
    fired, _ = would_fire_p1(fire_idx=fire_idx, config=config)
    return fired


def _normalize_for_boundary(text: str) -> str:
    """Lowercase + strip non-alphanumeric + collapse whitespace.

    Boundary matching is whitespace- and punctuation-insensitive, so
    "Ok, thanks!" normalizes the same as "ok thanks". Apostrophes
    are *removed* (not replaced with space) so "let's" → "lets"
    matches the allowlist's apostrophe-free form; other punctuation
    becomes whitespace so word boundaries are preserved
    ("ok,thanks" → "ok thanks").
    """
    norm = text.strip().lower()
    out_chars: list[str] = []
    for c in norm:
        if c.isalnum() or c.isspace():
            out_chars.append(c)
        elif c == "'":
            # Drop apostrophes entirely to fold contractions.
            continue
        else:
            out_chars.append(" ")
    return " ".join("".join(out_chars).split())


def is_phase_boundary_signal(prompt: str | None) -> bool:
    """Return True iff ``prompt`` looks like a task-boundary signal.

    Pure function. Determinism (#605): same prompt → same verdict,
    no I/O, no clock.

    Matches via:
      * Exact normalized match against :data:`_PHASE_BOUNDARY_PHRASES`.
      * Prefix match against :data:`_PHASE_BOUNDARY_PREFIXES`.

    Empty / None / oversized (>80 normalized chars) prompts return
    False. The length cap is deliberate — long prompts beginning with
    "ok" almost always contain substantive work after the ack token.
    """
    if not prompt:
        return False
    norm = _normalize_for_boundary(prompt)
    if not norm:
        return False
    if len(norm) > _PHASE_BOUNDARY_MAX_LEN:
        return False
    if norm in _PHASE_BOUNDARY_PHRASES:
        return True
    for prefix in _PHASE_BOUNDARY_PREFIXES:
        if norm.startswith(prefix):
            return True
    return False


def estimate_transcript_bytes(
    transcript_path: Path | str | os.PathLike[str] | None,
) -> int:
    """Return transcript file size in bytes, 0 on missing/unreadable.

    Accepts ``Path``, ``str``, or any ``os.PathLike[str]`` (the hook
    surface receives the transcript path as a JSON string and converts
    to ``Path`` upstream, but other callers — tests, replay — may pass
    either form). Returns 0 on ``None``, missing file, or any I/O
    error. Never raises.

    Used by P2 (ctx-threshold) as the byte-count input. Deterministic
    per #605: same on-disk file → same byte count → same fire
    decision. The byte-count is an approximation of token usage; the
    operator-side ctx% reading from the host harness statusline is
    ground truth.
    """
    if transcript_path is None:
        return 0
    try:
        path = (
            transcript_path
            if isinstance(transcript_path, Path)
            else Path(transcript_path)
        )
    except TypeError:
        return 0
    try:
        if not path.exists():
            return 0
        return path.stat().st_size
    except OSError:
        return 0


def read_last_user_prompt(transcript_path: Path | None) -> str | None:
    """Return the content of the most-recent user-role line in the
    transcript JSONL, or None if not findable.

    Fail-soft: any I/O or parse error returns None. The transcript
    file format is the host harness's per-session jsonl with one JSON
    object per line. Schema (best-effort): each line is a dict with
    a ``message`` sub-dict (or the fields inlined at top level)
    carrying ``role`` and ``content``. ``content`` may be a string
    or a list of ``{type, text}`` blocks.

    Reads the tail of the file (last 64KB) to avoid loading huge
    transcripts into memory.
    """
    if transcript_path is None:
        return None
    try:
        path = transcript_path if isinstance(transcript_path, Path) else Path(transcript_path)
    except TypeError:
        return None
    try:
        if not path.exists():
            return None
        size = path.stat().st_size
    except OSError:
        return None
    try:
        with path.open("rb") as f:
            if size > 65536:
                f.seek(-65536, os.SEEK_END)
                _ = f.readline()  # discard partial first line
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    last_user: str | None = None
    for raw_line in tail.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        msg_obj = obj.get("message")
        msg: dict[str, Any] = msg_obj if isinstance(msg_obj, dict) else obj
        if msg.get("role") != "user":
            continue
        content_obj: Any = msg.get("content")
        if isinstance(content_obj, str):
            last_user = content_obj
        elif isinstance(content_obj, list):
            texts: list[str] = []
            for blk in content_obj:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") != "text":
                    continue
                txt = blk.get("text")
                if isinstance(txt, str):
                    texts.append(txt)
            if texts:
                last_user = "\n".join(texts)
    return last_user


def would_fire_p2(
    *,
    transcript_path: Path | None,
    last_user_prompt: str | None,
    config: CadenceConfig,
) -> tuple[bool, str]:
    """P2 policy-agnostic predicate — ctx-threshold AND phase-boundary.

    Pure function of ``(transcript bytes, last_user_prompt, config)``.
    Determinism (#605): replay with the same on-disk transcript file
    and same prompt input reproduces the same ``(bool, reason)``. No
    wall-clock, no random sampling.

    Unlike :func:`should_fire_p2`, this does **not** check
    ``config.policy``. Shadow-evaluation mode (#875) calls this to
    log what P2 would have decided when another policy was selected.

    Conditions for True (all must hold):
      * ``config.enabled`` is True.
      * ``config.ctx_byte_window`` is positive.
      * ``config.ctx_threshold`` is in ``(0, 1]``.
      * Transcript file size (in bytes) is >=
        ``ctx_threshold * ctx_byte_window``.
      * ``last_user_prompt`` passes :func:`is_phase_boundary_signal`.

    Returns ``(False, reason)`` on any condition unmet, where
    ``reason`` is a short human-readable diagnostic string.
    """
    if not config.enabled:
        return (False, "cadence disabled")
    if config.ctx_byte_window <= 0:
        return (False, f"ctx_byte_window={config.ctx_byte_window} not positive")
    if config.ctx_threshold <= 0 or config.ctx_threshold > 1:
        return (
            False,
            f"ctx_threshold={config.ctx_threshold} out of (0, 1]",
        )
    bytes_used = estimate_transcript_bytes(transcript_path)
    # Floor watermark at 1 byte so pathological configs (e.g.
    # ctx_byte_window=1, ctx_threshold=0.5 -> int(0.5)=0) don't
    # turn the ctx half of the predicate into a no-op gate. With
    # the floor, a literally empty transcript (0 bytes) still
    # cannot trip the threshold, no matter how small the window.
    watermark = max(1, int(config.ctx_threshold * config.ctx_byte_window))
    if bytes_used < watermark:
        return (
            False,
            f"transcript bytes={bytes_used} < watermark={watermark}",
        )
    if not is_phase_boundary_signal(last_user_prompt):
        return (False, "last_user_prompt not a phase-boundary signal")
    return (
        True,
        f"transcript bytes={bytes_used} >= watermark={watermark}, phase-boundary",
    )


def should_fire_p2(
    *,
    transcript_path: Path | None,
    last_user_prompt: str | None,
    config: CadenceConfig,
) -> bool:
    """P2 firing predicate — ctx-threshold AND phase-boundary,
    policy-gated.

    Pure function of ``(transcript bytes, last_user_prompt, config)``.
    Determinism (#605): replay with the same on-disk transcript file
    and same prompt input produces the same fire decision. No
    wall-clock, no random sampling.

    Conditions (all must hold):
      * ``config.policy`` is ``p2_ctx_threshold``.
      * :func:`would_fire_p2` returns True for the same inputs.

    Returns False on any condition unmet. Never raises.

    For shadow-evaluation (non-selected policy logging), see
    :func:`would_fire_p2`.
    """
    if config.policy != POLICY_P2_CTX_THRESHOLD:
        return False
    fired, _ = would_fire_p2(
        transcript_path=transcript_path,
        last_user_prompt=last_user_prompt,
        config=config,
    )
    return fired


def would_fire_p3_velocity(
    *,
    bytes_at_last_fire: int,
    transcript_bytes: int,
    turns_since_last_fire: int,
    config: CadenceConfig,
) -> tuple[bool, str]:
    """P3-velocity policy-agnostic predicate — bytes/turn density.

    Pure function over the four inputs. Determinism (#605): same
    inputs reproduce same ``(bool, reason)``. No wall-clock, no I/O.

    Density = ``(transcript_bytes - bytes_at_last_fire) / turns_since_last_fire``.
    Fires when density exceeds ``config.p3_velocity_threshold``.

    Unlike :func:`should_fire_p3_velocity`, this does NOT check
    ``config.policy``. Answers "would p3_velocity fire here" — the
    predicate the shadow-evaluation log (#875) writes for non-selected
    policy decisions.

    Conditions for True:
      * ``config.enabled`` is True.
      * ``turns_since_last_fire`` is positive (no division-by-zero).
      * ``transcript_bytes >= bytes_at_last_fire`` (monotonic transcript).
      * ``config.p3_velocity_threshold`` is positive.
      * computed density >= threshold.
    """
    if not config.enabled:
        return (False, "cadence disabled")
    if turns_since_last_fire <= 0:
        return (False, f"turns_since_last_fire={turns_since_last_fire} not positive")
    if transcript_bytes < bytes_at_last_fire:
        return (
            False,
            f"transcript_bytes={transcript_bytes} below "
            f"bytes_at_last_fire={bytes_at_last_fire}",
        )
    threshold = config.p3_velocity_threshold
    if threshold <= 0:
        return (False, f"p3_velocity_threshold={threshold} not positive")
    delta_bytes = transcript_bytes - bytes_at_last_fire
    velocity = delta_bytes / turns_since_last_fire
    if velocity < threshold:
        return (
            False,
            f"velocity={velocity:.1f} bytes/turn below threshold={threshold}",
        )
    return (
        True,
        f"velocity={velocity:.1f} bytes/turn >= threshold={threshold}",
    )


def should_fire_p3_velocity(
    *,
    bytes_at_last_fire: int,
    transcript_bytes: int,
    turns_since_last_fire: int,
    config: CadenceConfig,
) -> bool:
    """P3-velocity firing predicate — policy-gated.

    Pure function. Determinism (#605): same inputs → same decision.

    Conditions:
      * ``config.policy`` is ``p3_velocity``.
      * :func:`would_fire_p3_velocity` returns True.

    Never raises. For shadow-evaluation, see :func:`would_fire_p3_velocity`.
    """
    if config.policy != POLICY_P3_VELOCITY:
        return False
    fired, _ = would_fire_p3_velocity(
        bytes_at_last_fire=bytes_at_last_fire,
        transcript_bytes=transcript_bytes,
        turns_since_last_fire=turns_since_last_fire,
        config=config,
    )
    return fired


def would_fire_p3_substantive(
    *,
    substantive_count: int,
    config: CadenceConfig,
) -> tuple[bool, str]:
    """P3-substantive policy-agnostic predicate — substantive-ratio.

    Pure function over ``(substantive_count, config)``. Determinism
    (#605): same inputs reproduce same decision.

    Predicate: ``substantive_count / config.p3_substantive_window >=
    config.p3_substantive_threshold``. The caller computes
    ``substantive_count`` by classifying the last N prompts via
    :func:`is_substantive_turn` and summing the True count; the
    session-ring maintenance of that rolling window is separate
    (lands in the next PR of the #876 stack).

    Conditions for True:
      * ``config.enabled`` is True.
      * ``config.p3_substantive_window`` is positive.
      * ``0.0 <= substantive_count <= window`` (sanity).
      * ratio >= ``config.p3_substantive_threshold``.

    Threshold is treated as a unit-float; values outside [0, 1] are
    rejected as misconfigured (return False with a reason).
    """
    if not config.enabled:
        return (False, "cadence disabled")
    window = config.p3_substantive_window
    if window <= 0:
        return (False, f"p3_substantive_window={window} not positive")
    if substantive_count < 0 or substantive_count > window:
        return (
            False,
            f"substantive_count={substantive_count} outside [0, {window}]",
        )
    threshold = config.p3_substantive_threshold
    if not 0.0 <= threshold <= 1.0:
        return (
            False,
            f"p3_substantive_threshold={threshold} outside [0.0, 1.0]",
        )
    ratio = substantive_count / window
    if ratio < threshold:
        return (
            False,
            f"ratio={ratio:.3f} below threshold={threshold:.3f}",
        )
    return (
        True,
        f"ratio={ratio:.3f} >= threshold={threshold:.3f} "
        f"({substantive_count}/{window})",
    )


def should_fire_p3_substantive(
    *,
    substantive_count: int,
    config: CadenceConfig,
) -> bool:
    """P3-substantive firing predicate — policy-gated.

    Pure function. Determinism (#605): same inputs → same decision.

    Conditions:
      * ``config.policy`` is ``p3_substantive``.
      * :func:`would_fire_p3_substantive` returns True.

    Never raises. For shadow-evaluation, see
    :func:`would_fire_p3_substantive`.
    """
    if config.policy != POLICY_P3_SUBSTANTIVE:
        return False
    fired, _ = would_fire_p3_substantive(
        substantive_count=substantive_count,
        config=config,
    )
    return fired


# --- Shadow-evaluation log (#875) ----------------------------------------


def shadow_log_path(
    *,
    project_aelfrice_dir: Path,
    session_id: str,
) -> Path:
    """Resolve the per-session shadow-log path.

    Layout: ``<project_aelfrice_dir>/cadence_shadow/<session_id>.jsonl``.

    ``project_aelfrice_dir`` is the project's ``.git/aelfrice/`` (the
    parent of ``rebuild_logs/``). Caller derives it from
    ``_rebuild_log_dir_for_db(db_path).parent`` to keep the layout
    aligned with existing per-session artifacts (rebuild_logs,
    cadence_resume_cache).

    Pure path computation — does not touch disk. Caller decides
    whether to mkdir / write.
    """
    return project_aelfrice_dir / CADENCE_SHADOW_DIRNAME / f"{session_id}.jsonl"


def format_shadow_row(
    *,
    session_id: str,
    selected_policy: str,
    fired: bool,
    shadow: dict[str, dict[str, Any]],
    now: str,
) -> str:
    """Format one shadow-log row as a JSON line (trailing newline).

    Pure function. Determinism (#605): same inputs reproduce the same
    JSON string, byte-for-byte. ``now`` is supplied by the caller as
    an ISO-8601 UTC string — keeping wall-clock outside the formatter
    preserves replay-ability in tests and scoring.

    ``shadow`` maps each policy name (e.g. ``"p1_every_k_turns"``) to
    a dict with at minimum ``{"would_fire": bool, "reason": str}``.
    Extra keys are passed through, so future policies can carry
    additional state (e.g. P3 density metrics) without a schema bump.

    The row schema is intentionally flat and deterministic — no
    auto-generated UUIDs, no random ordering. ``json.dumps`` is
    called with ``sort_keys=False``: the caller controls field order
    via the order they pass ``shadow`` (Python 3.7+ dict insertion
    order).
    """
    row: dict[str, Any] = {
        "ts": now,
        "session_id": session_id,
        "selected": selected_policy,
        "fired": fired,
        "shadow": shadow,
    }
    return json.dumps(row, ensure_ascii=False) + "\n"


def append_shadow_row(
    *,
    log_path: Path,
    row_line: str,
) -> None:
    """Append one pre-formatted JSON line to the shadow log.

    Creates the parent directory if missing. Opens in append mode
    with ``encoding="utf-8"``. Caller is responsible for the trailing
    newline on ``row_line`` (``format_shadow_row`` includes one).

    Never raises: any ``OSError`` is swallowed silently. The caller
    is the Stop hook hot-path; a missing shadow row is recoverable
    (the next tick writes a fresh one) but a propagating exception
    would tear down the hook, which is not. Fail-soft matches the
    rebuild_log emit contract.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(row_line)
    except OSError:
        # Fail-soft: hot-path stays alive even if disk is full /
        # permissions wrong / parent unwritable. Diagnostic is the
        # absence of a row for this tick, which the operator can
        # detect post-hoc.
        return
