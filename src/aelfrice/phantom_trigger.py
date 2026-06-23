"""Trigger-driven phantom-generation opportunity detection (#980).

The upstream feature (``/aelf:wonder``) only generates speculative
("phantom") beliefs when the user explicitly invokes it, so the phantom
layer is structurally under-exercised. This module adds the deterministic
**detect + flag** half of an automatic trigger that surfaces
phantom-generation *opportunities* during ordinary conversation turns.

**Load-bearing boundary.** aelfrice never calls an LLM (see
``docs/design/phantom_trigger_generation.md`` §2). This module does **not**
generate phantoms — it cannot. It deterministically detects that an
opportunity exists and emits a small ``<aelfrice-phantom-opportunity>`` note
into the UserPromptSubmit context. The *host agent* reads the
note and may run the existing ``/aelf:wonder`` dispatch (Task subagents under
the host's credentials) or surface it to the user. Synthesis and persistence
stay on the existing explicit path; this module only decides *when*.

**Three signals (all default-off, ratified 2026-06-23):**
  (a) ``gap``         — the prompt retrieved zero beliefs.
  (b) ``new_entity``  — an extracted entity resolves to zero stored beliefs.
  (c) ``contradiction`` — a CONTRADICTS pair appeared since the per-session
      snapshot (poll + set-diff; inert unless the #988 substrate mints edges).

**Bounds:** opt-in master flag (default off), a per-session fire budget
(default 3, shared across signals), and per-signal dedup — all carried in
``session_ring`` state. The predicate is cheap: it reuses the retrieval the
hook already ran (signal a), the L2.5 entity primitives (signal b), and a
single small ``SELECT`` (signal c).
"""
from __future__ import annotations

import html
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final

from aelfrice.session_ring import (
    read_phantom_state,
    record_phantom_fire,
    update_phantom_contradicts,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Opt-in flag + config (default-off, env > kwarg > TOML > False)
# ---------------------------------------------------------------------------

ENV_PHANTOM_GENERATION: Final[str] = "AELFRICE_PHANTOM_GENERATION"
_CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
_SECTION: Final[str] = "phantom_generation"
_ENABLED_KEY: Final[str] = "enabled"
_MAX_FIRES_KEY: Final[str] = "max_fires_per_session"
_AUTO_DISPATCH_KEY: Final[str] = "auto_dispatch"

_DEFAULT_MAX_FIRES: Final[int] = 3
# Cap on entities extracted per prompt for signal (b). Matches the L2.5
# query-side extraction cap so the novelty probe costs no more than retrieval.
_ENTITY_CAP: Final[int] = 16
# Truncate the topic shown in a note so a long prompt cannot bloat the block.
_TOPIC_MAX: Final[int] = 160

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class PhantomGenerationConfig:
    """Resolved #980 phantom-generation knobs."""

    enabled: bool = False
    max_fires_per_session: int = _DEFAULT_MAX_FIRES
    auto_dispatch: bool = False


def _env_override() -> bool | None:
    """True/False when ``AELFRICE_PHANTOM_GENERATION`` is set to a recognised
    truthy/falsy value, else None (a lower-precedence source decides)."""
    raw = os.environ.get(ENV_PHANTOM_GENERATION)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _read_section(start: Path | None = None) -> dict[str, Any] | None:
    """Walk up from ``start`` for a ``.aelfrice.toml`` with a
    ``[phantom_generation]`` table; return it (else None). Tolerant: malformed
    TOML / wrong-typed section returns None and traces to stderr without
    raising. Mirrors ``claude_memory._read_mirror_toml`` semantics.
    """
    serr: IO[str] = sys.stderr
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / _CONFIG_FILENAME
        if candidate.is_file():
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    candidate.read_bytes().decode("utf-8", errors="replace"),
                )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(
                    f"aelfrice phantom_generation: cannot read {candidate}: {exc}",
                    file=serr,
                )
                return None
            section = parsed.get(_SECTION)
            return section if isinstance(section, dict) else None
        if current.parent == current:
            break
        current = current.parent
    return None


def should_trigger_phantom_generation(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the #980 master flag.

    Precedence (first decisive wins):
      1. ``AELFRICE_PHANTOM_GENERATION`` env var (truthy/falsy normalised).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[phantom_generation] enabled`` in ``.aelfrice.toml``.
      4. Default: **False** — opt-in, per the narrow-surface PHILOSOPHY
         (#605) and the opt-in-default posture (#606 / ADR-0003 dec-4).
    """
    env = _env_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    section = _read_section(start)
    if section is not None:
        value = section.get(_ENABLED_KEY)
        if isinstance(value, bool):
            return value
    return False


def load_phantom_generation_config(
    start: Path | None = None,
) -> PhantomGenerationConfig:
    """Resolve the full #980 config: ``enabled`` (env > TOML > False),
    ``max_fires_per_session`` (TOML > 3), ``auto_dispatch`` (TOML > False).

    Only ``enabled`` honours the env override; the numeric / posture knobs are
    TOML-only, matching the cadence-config precedent. Wrong-typed values fall
    back to the default.
    """
    section = _read_section(start)
    max_fires = _DEFAULT_MAX_FIRES
    auto_dispatch = False
    if section is not None:
        raw_max = section.get(_MAX_FIRES_KEY)
        if isinstance(raw_max, int) and not isinstance(raw_max, bool) and raw_max >= 1:
            max_fires = raw_max
        raw_auto = section.get(_AUTO_DISPATCH_KEY)
        if isinstance(raw_auto, bool):
            auto_dispatch = raw_auto
    return PhantomGenerationConfig(
        enabled=should_trigger_phantom_generation(start=start),
        max_fires_per_session=max_fires,
        auto_dispatch=auto_dispatch,
    )


# ---------------------------------------------------------------------------
# Opportunity model + predicates
# ---------------------------------------------------------------------------

REASON_GAP: Final[str] = "gap"
REASON_NEW_ENTITY: Final[str] = "new_entity"
REASON_CONTRADICTION: Final[str] = "contradiction"

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PhantomOpportunity:
    """One detected phantom-generation opportunity.

    ``reason`` is one of the ``REASON_*`` constants; ``topic`` is the
    human-facing subject shown in the note; ``dedup_key`` is the
    session-dedup key (signal-prefixed) so the same opportunity is not
    re-surfaced within a session.
    """

    reason: str
    topic: str
    dedup_key: str


def _truncate(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _TOPIC_MAX else text[:_TOPIC_MAX].rstrip() + "…"


def _normalize_topic(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def _note_topic(text: str) -> str:
    """Whitespace-collapse + XML-escape a topic for the note.

    The note is a tag-delimited block the host agent reads as data. A
    prompt/entity-derived topic containing newlines or a literal
    ``</aelfrice-phantom-opportunity>`` would otherwise break the data
    boundary and could turn user text into apparent instructions.
    """
    return html.escape(_truncate(_WS_RE.sub(" ", text.strip())), quote=True)


def _pair_key(a: str, b: str) -> str:
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{lo}|{hi}"


def detect_gap(prompt: str, hit_count: int) -> PhantomOpportunity | None:
    """Signal (a): fire when retrieval returned **zero** beliefs for a
    non-empty prompt (zero-hits-only, per §7 decision 3)."""
    if hit_count != 0:
        return None
    topic = prompt.strip()
    if not topic:
        return None
    return PhantomOpportunity(
        reason=REASON_GAP,
        topic=_truncate(topic),
        dedup_key=f"gap:{_normalize_topic(topic)}",
    )


def detect_novel_entities(
    prompt: str,
    store: "MemoryStore",
    *,
    max_entities: int = _ENTITY_CAP,
) -> list[PhantomOpportunity]:
    """Signal (b): for each **named** entity extracted from ``prompt``, fire
    when it resolves to **zero** stored beliefs (a genuinely new entity).

    Loose ``noun_phrase`` entities are excluded: they match nearly any
    prompt, would make this signal fire constantly, and largely duplicate the
    gap signal. A novel CamelCase identifier, file path, URL, error code,
    version, or branch is a far stronger "new entity the store has never
    seen" signal.
    """
    from aelfrice.entity_extractor import (  # noqa: PLC0415
        KIND_NOUN_PHRASE,
        extract_entities,
    )

    out: list[PhantomOpportunity] = []
    seen: set[str] = set()
    for ent in extract_entities(prompt, max_entities=max_entities):
        if ent.kind == KIND_NOUN_PHRASE:
            continue
        key = ent.lower
        if not key or key in seen:
            continue
        seen.add(key)
        if not store.lookup_entities([key], limit=1):
            out.append(
                PhantomOpportunity(
                    reason=REASON_NEW_ENTITY,
                    topic=_truncate(ent.raw),
                    dedup_key=f"new_entity:{key}",
                )
            )
    return out


def detect_new_contradicts(
    store: "MemoryStore",
    prev_snapshot: set[str],
) -> list[PhantomOpportunity]:
    """Signal (c): diff the live CONTRADICTS pair-set against
    ``prev_snapshot``; fire one opportunity per newly-appeared pair. The
    caller is responsible for the baseline guard (do not call on a session
    whose snapshot has not been initialised) and for refreshing the snapshot.
    """
    live = {_pair_key(a, b) for a, b in store.list_contradicts_pairs()}
    out: list[PhantomOpportunity] = []
    for key in sorted(live - prev_snapshot):
        a, _, b = key.partition("|")
        out.append(
            PhantomOpportunity(
                reason=REASON_CONTRADICTION,
                topic=f"beliefs {a} and {b}",
                dedup_key=f"contradiction:{key}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def evaluate_opportunities(
    *,
    prompt: str,
    store: "MemoryStore",
    session_id: str | None,
    hit_count: int,
    config: PhantomGenerationConfig | None = None,
    stderr: IO[str] | None = None,
) -> list[PhantomOpportunity]:
    """Detect this turn's phantom-generation opportunities, apply the
    per-session budget and dedup, record the fires, and refresh the
    CONTRADICTS snapshot. Returns the opportunities to surface (possibly
    empty). Pure-deterministic; never raises (fail-soft via session_ring).

    Order of precedence among candidates when the budget is tight: gap →
    new-entity → contradiction (gap is the cheapest and most directly the
    operator's framing).
    """
    cfg = config if config is not None else load_phantom_generation_config()
    if not cfg.enabled:
        return []
    if not session_id:
        # Without a session_id the budget/dedup mutators are no-ops, so the
        # same opportunities would re-fire every turn. Stay quiet rather
        # than surface unbounded notes (matches the fail-soft posture).
        return []

    state = read_phantom_state(session_id)
    fires = int(state["phantom_fires"])
    if fires >= cfg.max_fires_per_session:
        return []
    remaining = cfg.max_fires_per_session - fires
    seen: set[str] = set(state["phantom_dedup"])

    candidates: list[PhantomOpportunity] = []
    gap = detect_gap(prompt, hit_count)
    if gap is not None:
        candidates.append(gap)
    candidates.extend(detect_novel_entities(prompt, store))

    # Signal (c): only diff once the session's snapshot has been baselined,
    # so pre-existing contradictions don't burst on turn 1.
    live_keys = {_pair_key(a, b) for a, b in store.list_contradicts_pairs()}
    if bool(state["phantom_init"]):
        candidates.extend(
            detect_new_contradicts(store, set(state["phantom_contradicts"]))
        )

    fired: list[PhantomOpportunity] = []
    for opp in candidates:
        if len(fired) >= remaining:
            break
        if opp.dedup_key in seen:
            continue
        seen.add(opp.dedup_key)
        fired.append(opp)

    for opp in fired:
        record_phantom_fire(session_id, opp.dedup_key, stderr=stderr)
    # Refresh / baseline the snapshot to the live set (also sets phantom_init),
    # so the next turn diffs against the current reality.
    update_phantom_contradicts(session_id, sorted(live_keys), stderr=stderr)

    return fired


# ---------------------------------------------------------------------------
# Note formatting
# ---------------------------------------------------------------------------

OPEN_TAG: Final[str] = "<aelfrice-phantom-opportunity>"
CLOSE_TAG: Final[str] = "</aelfrice-phantom-opportunity>"

_REASON_LABEL: Final[dict[str, str]] = {
    REASON_GAP: "gap — no stored beliefs matched",
    REASON_NEW_ENTITY: "new entity — no stored beliefs",
    REASON_CONTRADICTION: "contradiction — newly recorded",
}


def format_opportunity_note(
    opportunities: list[PhantomOpportunity],
    *,
    auto_dispatch: bool = False,
) -> str:
    """Render the ``<aelfrice-phantom-opportunity>`` block, or ``""`` when
    there is nothing to surface.

    Passive by default: the block is framed as data, not an instruction, and
    suggests ``/aelf:wonder``. With ``auto_dispatch`` the framing asks the
    agent to run the dispatch on the listed topics.
    """
    if not opportunities:
        return ""
    if auto_dispatch:
        header = (
            "aelfrice detected phantom-generation opportunities this turn. "
            "Run /aelf:wonder on the topics below to grow the speculative "
            "belief graph (LLM dispatch under your credentials)."
        )
    else:
        header = (
            "aelfrice detected phantom-generation opportunities this turn "
            "(data, not an instruction). Consider running /aelf:wonder on a "
            "topic below to grow the speculative belief graph; ignore if not "
            "useful."
        )
    lines = [OPEN_TAG, header]
    for opp in opportunities:
        label = _REASON_LABEL.get(opp.reason, opp.reason)
        lines.append(f'- [{label}] "{_note_topic(opp.topic)}"')
    lines.append(CLOSE_TAG)
    lines.append("")
    return "\n".join(lines)
