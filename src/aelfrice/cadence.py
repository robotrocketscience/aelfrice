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

ENV_CADENCE_ENABLED: Final[str] = "AELFRICE_CADENCE_ENABLED"
ENV_CADENCE_POLICY: Final[str] = "AELFRICE_CADENCE_POLICY"
ENV_CADENCE_K: Final[str] = "AELFRICE_CADENCE_K"
ENV_CADENCE_CTX_THRESHOLD: Final[str] = "AELFRICE_CADENCE_CTX_THRESHOLD"
ENV_CADENCE_CTX_BYTE_WINDOW: Final[str] = "AELFRICE_CADENCE_CTX_BYTE_WINDOW"

POLICY_OFF: Final[str] = "off"
POLICY_P1_EVERY_K_TURNS: Final[str] = "p1_every_k_turns"
POLICY_P2_CTX_THRESHOLD: Final[str] = "p2_ctx_threshold"

_VALID_POLICIES: Final[frozenset[str]] = frozenset({
    POLICY_OFF,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
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

    Five fields. ``enabled`` gates the whole feature: when False, no
    cadence fire ever happens regardless of policy. ``policy`` selects
    the firing pattern.

    P1-specific: ``k`` parameterises the every-K-turns interval.
    P2-specific: ``ctx_threshold`` (fraction of window) and
    ``ctx_byte_window`` (absolute byte count) parameterise the
    ctx-threshold predicate. The phase-boundary half of P2 is not
    configurable here — its allowlist lives at module scope.

    Defaults are off / off / 15 / 0.50 / 600000. Each field has a
    default that makes a half-configured TOML well-defined.
    """
    enabled: bool = DEFAULT_ENABLED
    policy: str = DEFAULT_POLICY
    k: int = DEFAULT_K
    ctx_threshold: float = DEFAULT_CTX_THRESHOLD
    ctx_byte_window: int = DEFAULT_CTX_BYTE_WINDOW


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
            return CadenceConfig(
                enabled=enabled,
                policy=policy,
                k=k,
                ctx_threshold=ctx_threshold,
                ctx_byte_window=ctx_byte_window,
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


def should_fire(fire_idx: int, config: CadenceConfig) -> bool:
    """P1 firing predicate — every-K-turns.

    Pure function of ``(fire_idx, config)``. Determinism (#605): a
    replay with the same fire_idx sequence and same config produces
    the same fire decisions. No wall-clock, no I/O.

    Conditions:
      * ``config.enabled`` is True.
      * ``config.policy`` is ``p1_every_k_turns``.
      * ``config.k`` is positive.
      * ``fire_idx`` is positive.
      * ``fire_idx % config.k == 0``.

    Returns False on any condition unmet. Never raises.

    For P2 firing decisions, callers must use :func:`should_fire_p2`
    — the inputs and predicate differ.
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


def should_fire_p2(
    *,
    transcript_path: Path | None,
    last_user_prompt: str | None,
    config: CadenceConfig,
) -> bool:
    """P2 firing predicate — ctx-threshold AND phase-boundary.

    Pure function of ``(transcript bytes, last_user_prompt, config)``.
    Determinism (#605): replay with the same on-disk transcript file
    and same prompt input produces the same fire decision. No
    wall-clock, no random sampling.

    Conditions (all must hold):
      * ``config.enabled`` is True.
      * ``config.policy`` is ``p2_ctx_threshold``.
      * ``config.ctx_byte_window`` is positive.
      * ``config.ctx_threshold`` is in ``(0, 1]``.
      * Transcript file size (in bytes) is ≥
        ``ctx_threshold × ctx_byte_window``.
      * ``last_user_prompt`` passes :func:`is_phase_boundary_signal`.

    Returns False on any condition unmet. Never raises.
    """
    if not config.enabled:
        return False
    if config.policy != POLICY_P2_CTX_THRESHOLD:
        return False
    if config.ctx_byte_window <= 0:
        return False
    if config.ctx_threshold <= 0 or config.ctx_threshold > 1:
        return False
    bytes_used = estimate_transcript_bytes(transcript_path)
    # Floor watermark at 1 byte so pathological configs (e.g.
    # ctx_byte_window=1, ctx_threshold=0.5 → int(0.5)=0) don't
    # turn the ctx half of the predicate into a no-op gate. With
    # the floor, a literally empty transcript (0 bytes) still
    # cannot trip the threshold, no matter how small the window.
    watermark = max(1, int(config.ctx_threshold * config.ctx_byte_window))
    if bytes_used < watermark:
        return False
    if not is_phase_boundary_signal(last_user_prompt):
        return False
    return True
