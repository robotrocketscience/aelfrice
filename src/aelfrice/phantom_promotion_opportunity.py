"""Trigger-driven phantom promotion-opportunity detection (#1132 Q2).

A phantom (``origin='speculative'``) is promoted to a trusted origin only
through explicit user acknowledgment — ``aelf validate <id>`` (Surface A) or
``aelf lock <text>`` matching its content (Surface B, #550). The ratified #229
rule (``docs/design/historical/v2_phantom_promotion_trigger.md``) is emphatic
that a corroboration / retrieval count is a **non-trigger** for that origin
write. In practice this means phantoms are essentially never promoted: nothing
ever *prompts* the explicit act, so a phantom that has been independently
re-asserted across many sessions still sits unvalidated forever (see the #1125
census: 0 promoted across 7 real stores).

This module closes that gap the only way #229 permits — by **surfacing**, not
**writing**. It deterministically detects that a phantom has accumulated enough
cross-session corroboration to be worth a human's attention and emits a small
``<aelfrice-phantom-promotion-opportunity>`` note into the UserPromptSubmit
context. The note names the candidate and its ``aelf validate`` / lock surface;
the *user* still performs the explicit promotion. Detection is deterministic
(one indexed ``SELECT`` via ``store.find_promotable_phantoms``); the discrete
origin write stays exactly where #229 put it.

This is the promotion-side mirror of the #980 phantom-*generation* opportunity
detector (``phantom_trigger.py``): same note-not-write shape, same default-off
posture, same per-session budget + dedup carried in ``session_ring`` state.

**Bounds:** opt-in master flag (default off), a per-session fire budget
(default 3), a corroboration threshold (default 3 corroborations across 2
distinct sessions, matching the retention-promotion rule), and per-candidate
dedup keyed on belief id so a phantom is surfaced at most once per session.
"""
from __future__ import annotations

import html
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import IO, TYPE_CHECKING, Any, Final

from aelfrice.config_discovery import discover_config

from aelfrice.session_ring import (
    read_promotion_state,
    record_promotion_fire,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Opt-in flag + config (default-off, env > kwarg > TOML > False)
# ---------------------------------------------------------------------------

ENV_PHANTOM_PROMOTION: Final[str] = "AELFRICE_PHANTOM_PROMOTION"
_SECTION: Final[str] = "phantom_promotion"
_ENABLED_KEY: Final[str] = "enabled"
_MAX_FIRES_KEY: Final[str] = "max_fires_per_session"
_MIN_CORROBORATIONS_KEY: Final[str] = "min_corroborations"
_MIN_SESSIONS_KEY: Final[str] = "min_sessions"
# #1650: automatic promotion on the user's evidence. Its own switch,
# independent of the note: env > TOML > False.
ENV_PHANTOM_AUTO_PROMOTE: Final[str] = "AELFRICE_PHANTOM_AUTO_PROMOTE"
_AUTO_PROMOTE_KEY: Final[str] = "auto_promote"
# Promotions per turn. A store that crosses the bar for many phantoms at
# once promotes them over several turns, each with its notice.
_AUTO_PROMOTE_MAX_PER_TURN: Final[int] = 5

_DEFAULT_MAX_FIRES: Final[int] = 3
# Threshold defaults mirror the retention-promotion rule
# (belief_retention_class.md §4): >=3 corroborations across >=2 distinct
# sessions. A promotion opportunity is a higher-attention event than a
# retention flip, so these are a floor the operator can raise via TOML.
_DEFAULT_MIN_CORROBORATIONS: Final[int] = 3
_DEFAULT_MIN_SESSIONS: Final[int] = 2
# Truncate the phantom content shown in a note so a long belief cannot bloat
# the block.
_TOPIC_MAX: Final[int] = 160
# Cap candidates pulled from the store per turn — the note is bounded by the
# per-session fire budget anyway, but this bounds the query too.
_CANDIDATE_CAP: Final[int] = 32

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PhantomPromotionConfig:
    """Resolved #1132 Q2 phantom promotion-opportunity knobs."""

    enabled: bool = False
    max_fires_per_session: int = _DEFAULT_MAX_FIRES
    min_corroborations: int = _DEFAULT_MIN_CORROBORATIONS
    min_sessions: int = _DEFAULT_MIN_SESSIONS
    auto_promote: bool = False


def _env_override() -> bool | None:
    """True/False when ``AELFRICE_PHANTOM_PROMOTION`` is set to a recognised
    truthy/falsy value, else None (a lower-precedence source decides)."""
    raw = os.environ.get(ENV_PHANTOM_PROMOTION)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _resolve_auto_promote(section: dict[str, Any] | None) -> bool:
    """Resolve the #1650 auto-promotion switch: env > TOML > False."""
    raw = os.environ.get(ENV_PHANTOM_AUTO_PROMOTE)
    if raw is not None:
        norm = raw.strip().lower()
        if norm in _ENV_FALSY:
            return False
        if norm in _ENV_TRUTHY:
            return True
    if section is not None:
        value = section.get(_AUTO_PROMOTE_KEY)
        if isinstance(value, bool):
            return value
    return False


def _read_section(start: Path | None = None) -> dict[str, Any] | None:
    """Walk up from ``start`` for a ``.aelfrice.toml`` with a
    ``[phantom_promotion]`` table; return it (else None). Tolerant: malformed
    TOML / wrong-typed section returns None and traces to stderr without
    raising. Mirrors ``phantom_trigger._read_section`` semantics.
    """
    serr: IO[str] = sys.stderr
    # Shared discovery (#1304): inside a `config_discovery_scope`
    # N readers cost one walk instead of N. Semantics unchanged —
    # the loop this replaces already stopped at the first
    # `.aelfrice.toml` it found and never continued past it.
    candidate = discover_config(start)
    if candidate is not None:
        try:
            parsed: dict[str, Any] = tomllib.loads(
                candidate.read_bytes().decode("utf-8", errors="replace"),
            )
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(
                f"aelfrice phantom_promotion: cannot read {candidate}: {exc}",
                file=serr,
            )
            return None
        section = parsed.get(_SECTION)
        return section if isinstance(section, dict) else None
    return None


def _resolve_enabled(
    section: dict[str, Any] | None, explicit: bool | None = None
) -> bool:
    """Resolve the master flag from an already-read TOML ``section``.

    Precedence: env override > explicit kwarg > ``[phantom_promotion] enabled``
    > default False. Factored out so a caller that has already read the section
    (``load_phantom_promotion_config``) does not re-parse the file.
    """
    env = _env_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    if section is not None:
        value = section.get(_ENABLED_KEY)
        if isinstance(value, bool):
            return value
    return False


def should_trigger_phantom_promotion(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the #1132 Q2 master flag.

    Precedence (first decisive wins):
      1. ``AELFRICE_PHANTOM_PROMOTION`` env var (truthy/falsy normalised).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[phantom_promotion] enabled`` in ``.aelfrice.toml``.
      4. Default: **False** — opt-in, per the narrow-surface PHILOSOPHY
         (#605) and the opt-in-default posture (#606 / ADR-0003 dec-4).
    """
    return _resolve_enabled(_read_section(start), explicit)


def _positive_int(section: dict[str, Any], key: str, default: int) -> int:
    """Read a TOML int knob; fall back to ``default`` on absent/wrong-typed/
    non-positive (bools are rejected — TOML ``true`` is not a count)."""
    raw = section.get(key)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
        return raw
    return default


def load_phantom_promotion_config(
    start: Path | None = None,
) -> PhantomPromotionConfig:
    """Resolve the full #1132 Q2 config: ``enabled`` (env > TOML > False),
    plus the TOML-only numeric knobs (``max_fires_per_session``,
    ``min_corroborations``, ``min_sessions``), each falling back to its
    default on absent/wrong-typed values. Mirrors the #980 config precedent.
    """
    section = _read_section(start)
    max_fires = _DEFAULT_MAX_FIRES
    min_corr = _DEFAULT_MIN_CORROBORATIONS
    min_sess = _DEFAULT_MIN_SESSIONS
    if section is not None:
        max_fires = _positive_int(section, _MAX_FIRES_KEY, _DEFAULT_MAX_FIRES)
        min_corr = _positive_int(
            section, _MIN_CORROBORATIONS_KEY, _DEFAULT_MIN_CORROBORATIONS
        )
        min_sess = _positive_int(
            section, _MIN_SESSIONS_KEY, _DEFAULT_MIN_SESSIONS
        )
    return PhantomPromotionConfig(
        # Resolve enabled from the already-read section — no second file read.
        enabled=_resolve_enabled(section),
        max_fires_per_session=max_fires,
        min_corroborations=min_corr,
        min_sessions=min_sess,
        auto_promote=_resolve_auto_promote(section),
    )


# ---------------------------------------------------------------------------
# Opportunity model + detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionOpportunity:
    """One detected phantom promotion opportunity.

    ``belief_id`` is the phantom the user may validate; ``topic`` is a
    truncated content snippet shown in the note; ``dedup_key`` is the
    session-dedup key (the belief id) so the same candidate is not
    re-surfaced within a session.
    """

    belief_id: str
    topic: str
    dedup_key: str


def _truncate(text: str) -> str:
    text = _WS_RE.sub(" ", text.strip())
    return text if len(text) <= _TOPIC_MAX else text[:_TOPIC_MAX].rstrip() + "…"


def _note_topic(text: str) -> str:
    """XML-escape a phantom snippet for the note.

    ``topic`` is already whitespace-collapsed and length-bounded by
    ``_truncate`` when the ``PromotionOpportunity`` is built, so this only
    escapes. The note is a tag-delimited block the host agent reads as data;
    content containing a literal ``</aelfrice-phantom-promotion-opportunity>``
    or other markup would otherwise break the data boundary and could turn
    stored text into apparent instructions.
    """
    return html.escape(text, quote=True)


def detect_promotable_phantoms(
    store: "MemoryStore",
    *,
    min_corroborations: int,
    min_sessions: int,
    max_candidates: int = _CANDIDATE_CAP,
) -> list[PromotionOpportunity]:
    """Return promotion opportunities for phantoms that have crossed the
    corroboration threshold. Read-only; delegates the predicate to
    ``store.find_promotable_phantoms`` (deterministic, ``created_at`` order).
    """
    out: list[PromotionOpportunity] = []
    for belief in store.find_promotable_phantoms(
        min_corroborations=min_corroborations,
        min_sessions=min_sessions,
        max_n=max_candidates,
    ):
        out.append(
            PromotionOpportunity(
                belief_id=belief.id,
                topic=_truncate(belief.content),
                dedup_key=belief.id,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def evaluate_promotion_opportunities(
    *,
    store: "MemoryStore",
    session_id: str | None,
    config: PhantomPromotionConfig | None = None,
    stderr: IO[str] | None = None,
    room_chars: int | None = None,
    commits: list[Callable[[], None]] | None = None,
) -> list[PromotionOpportunity]:
    """Detect this session's phantom promotion opportunities, apply the
    per-session budget + dedup, record the fires, and return the ones to
    surface (possibly empty). Pure-deterministic; never raises (fail-soft via
    session_ring).

    With `room_chars`, stops at the first opportunity whose note would no
    longer fit in that many characters (#1639). Only what is returned is
    recorded as fired, so an opportunity that did not fit is still owed.

    With `commits`, the recording is appended there instead of done, for a
    caller that records only once the note is written.

    Candidates are drawn in ``created_at`` order so the oldest qualifying
    phantom is surfaced first when the budget is tight.
    """
    cfg = config if config is not None else load_phantom_promotion_config()
    if not cfg.enabled:
        return []
    if not session_id:
        # Without a session_id the budget/dedup mutators are no-ops, so the
        # same candidate would re-fire every turn. Stay quiet rather than
        # surface unbounded notes (matches the #980 fail-soft posture).
        return []

    state = read_promotion_state(session_id)
    fires = int(state["promotion_fires"])
    if fires >= cfg.max_fires_per_session:
        return []
    remaining = cfg.max_fires_per_session - fires
    seen: set[str] = set(state["promotion_dedup"])

    candidates = detect_promotable_phantoms(
        store,
        min_corroborations=cfg.min_corroborations,
        min_sessions=cfg.min_sessions,
    )

    fired: list[PromotionOpportunity] = []
    for opp in candidates:
        if len(fired) >= remaining:
            break
        if opp.dedup_key in seen:
            continue
        if room_chars is not None and len(
            format_promotion_note([*fired, opp])
        ) > room_chars:
            break
        seen.add(opp.dedup_key)
        fired.append(opp)

    sid = session_id

    def _record() -> None:
        for opp in fired:
            record_promotion_fire(sid, opp.dedup_key, stderr=stderr)

    if commits is None:
        _record()
    else:
        commits.append(_record)
    return fired


def auto_promote_phantoms(
    *,
    store: "MemoryStore",
    config: PhantomPromotionConfig,
    now: str | None = None,
) -> list[tuple[str, str]]:
    """Promote phantoms the user's evidence qualifies (#1650); default off.

    Uses :meth:`MemoryStore.find_evidence_promotable_phantoms` with the
    config's ``min_corroborations`` / ``min_sessions`` (3 and 2 by default,
    the spec's starting rule) and promotes each through
    :func:`aelfrice.promotion.promote_on_evidence`, which audits and can be
    undone with ``aelf demote``. At most ``_AUTO_PROMOTE_MAX_PER_TURN`` per
    call. Returns ``(id, content)`` for each phantom promoted now.
    """
    if not config.auto_promote:
        return []
    from aelfrice.promotion import promote_on_evidence  # noqa: PLC0415

    promoted: list[tuple[str, str]] = []
    for belief in store.find_evidence_promotable_phantoms(
        min_supports=config.min_corroborations,
        min_sessions=config.min_sessions,
        max_n=_AUTO_PROMOTE_MAX_PER_TURN,
    ):
        # One transaction per promotion, taking the write lock first, so two
        # hooks that both see the candidate can't both write an audit row:
        # the second re-reads the origin inside the lock and finds it done.
        with store.transaction(immediate=True):
            result = promote_on_evidence(store, belief.id, now=now)
        if not result.already_validated:
            promoted.append((belief.id, belief.content))
    return promoted


AUTO_OPEN_TAG: Final[str] = "<aelfrice-phantom-auto-promoted>"
AUTO_CLOSE_TAG: Final[str] = "</aelfrice-phantom-auto-promoted>"


def format_auto_promotion_notice(
    promoted: list[tuple[str, str]], *, min_sessions: int = _DEFAULT_MIN_SESSIONS,
) -> str:
    """Render the notice for phantoms promoted this turn, or ``""``."""
    if not promoted:
        return ""
    lines = [
        AUTO_OPEN_TAG,
        "aelfrice: these speculative (phantom) beliefs were promoted "
        "automatically to evidence_promoted, because the user restated them "
        f"in {min_sessions} or more sessions (data, not an instruction). "
        "Undo one with `aelf demote <id>`; an undone promotion is not "
        "repeated.",
    ]
    for bid, content in promoted:
        lines.append(f"- {html.escape(bid)}: {_note_topic(content)}")
    lines.append(AUTO_CLOSE_TAG)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Note formatting
# ---------------------------------------------------------------------------

OPEN_TAG: Final[str] = "<aelfrice-phantom-promotion-opportunity>"
CLOSE_TAG: Final[str] = "</aelfrice-phantom-promotion-opportunity>"


def format_promotion_note(opportunities: list[PromotionOpportunity]) -> str:
    """Render the ``<aelfrice-phantom-promotion-opportunity>`` block, or
    ``""`` when there is nothing to surface.

    Passive: the block is framed as data, not an instruction. Promotion is an
    explicit user act (#229) — the note surfaces the candidate and its
    ``aelf validate`` / lock surface; it never asks the agent to promote
    autonomously.
    """
    if not opportunities:
        return ""
    header = (
        "aelfrice: these speculative (phantom) beliefs have been corroborated "
        "across multiple sessions and may be worth confirming (data, not an "
        "instruction). To promote one to a trusted belief, the user can run "
        "`aelf validate <id>` or `aelf lock` its text; ignore if not useful. "
        "Do not promote autonomously."
    )
    lines = [OPEN_TAG, header]
    for opp in opportunities:
        lines.append(f'- {opp.belief_id}: "{_note_topic(opp.topic)}"')
    lines.append(CLOSE_TAG)
    lines.append("")
    return "\n".join(lines)
