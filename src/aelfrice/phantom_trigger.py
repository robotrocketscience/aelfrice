"""Deterministic phantom-opportunity trigger for normal turns (#980 item 3).

The #980 audit found phantoms are structurally under-generated: they only
appear when the user explicitly types ``/aelf:wonder`` (74 phantoms across
all stores in months). The operator's headline ask is a *trigger* that
flags a phantom-generation opportunity during ordinary conversation turns.

Per the locked PHILOSOPHY decision (#605) and the determinism property,
aelfrice owns only the **deterministic flag**; the actual phantom synthesis
is an LLM dispatch under the host agent's credentials (the same boundary as
``/aelf:wonder``), never in aelfrice's retrieval path. This module is that
deterministic half:

* :func:`detect_phantom_opportunity` — a pure function over the turn's
  prompt and retrieval-hit count. Returns a :class:`PhantomOpportunity`
  when the turn is a genuine knowledge gap (a real query that retrieved
  fewer beliefs than a floor), else ``None``. No I/O, no LLM, no clock.

* :func:`register_opportunity` — bounds the flag: per-session dedup on the
  normalized-query key plus a per-session rate cap, backed by a small
  sidecar JSON file. Keeps the trigger a *trickle*, not a spammer.

The host hook (``user_prompt_submit``) consumes both behind a default-off
env flag and surfaces the opportunity; it never generates the phantom
itself. See ``docs/feature-phantom-trigger-generation.md``.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

# Tuning defaults. All deterministic; the hook may override via env.
DEFAULT_MIN_HITS: int = 1
"""Gap iff the turn retrieved fewer than this many beliefs (default: 0 hits)."""

DEFAULT_MIN_QUERY_TOKENS: int = 3
"""A query thinner than this many meaningful tokens can't anchor a phantom."""

DEFAULT_MAX_PER_SESSION: int = 3
"""At most this many opportunities are flagged per session (rate cap)."""

_KEYS_CAP: int = 64
"""Retain at most this many recent dedup keys per session sidecar."""

_MIN_TOKEN_LEN: int = 3
"""Tokens shorter than this are dropped before keying / counting."""

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PhantomOpportunity:
    """A deterministic flag that this turn is a phantom-generation gap.

    Pure data. ``dedup_key`` is a stable hash of the normalized query
    tokens — two prompts that normalize to the same token set share a key
    and so dedup against each other within a session.
    """

    query: str
    n_hits: int
    reason: str  # "no_hits" | "below_floor"
    dedup_key: str


def normalize_query(prompt: str) -> tuple[str, ...]:
    """Lowercase, tokenize on ``[a-z0-9]+``, drop short tokens, sort-unique.

    Deterministic and order-independent so ``"about session ids"`` and
    ``"session id about"`` key identically. No stopword list — short-token
    pruning alone keeps the key stable without a non-deterministic lexicon.
    """
    toks = {
        t
        for t in _TOKEN_RE.findall(prompt.lower())
        if len(t) >= _MIN_TOKEN_LEN
    }
    return tuple(sorted(toks))


def dedup_key_for(prompt: str) -> str:
    """Stable 16-hex-char key over the normalized query tokens."""
    toks = normalize_query(prompt)
    digest = hashlib.sha256("\x1f".join(toks).encode("utf-8")).hexdigest()
    return digest[:16]


def detect_phantom_opportunity(
    prompt: str,
    n_hits: int,
    *,
    gate_skipped: bool,
    min_hits: int = DEFAULT_MIN_HITS,
    min_query_tokens: int = DEFAULT_MIN_QUERY_TOKENS,
) -> PhantomOpportunity | None:
    """Return a :class:`PhantomOpportunity` iff this turn is a real gap.

    A gap requires ALL of:

    * ``gate_skipped`` is False — trivial acks and system envelopes that
      the prompt-shape gate (#674) skips are not knowledge gaps; they need
      no belief, so they must never trigger generation.
    * the normalized query carries ≥ ``min_query_tokens`` meaningful
      tokens — too thin a query can't anchor a useful phantom.
    * the turn retrieved fewer than ``min_hits`` beliefs — the query
      seeded nothing (or nothing above the floor).

    Pure: no I/O, no clock, no randomness. ``reason`` is ``"no_hits"``
    when nothing was retrieved, else ``"below_floor"``.
    """
    if gate_skipped:
        return None
    toks = normalize_query(prompt)
    if len(toks) < min_query_tokens:
        return None
    if n_hits >= min_hits:
        return None
    reason = "no_hits" if n_hits <= 0 else "below_floor"
    return PhantomOpportunity(
        query=prompt.strip(),
        n_hits=n_hits,
        reason=reason,
        dedup_key=dedup_key_for(prompt),
    )


def _state_path(state_dir: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id or "nosession")[:64]
    return state_dir / f"phantom_trigger_{safe}.json"


def _load_state(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"keys": [], "count": 0}
    if not isinstance(data, dict):
        return {"keys": [], "count": 0}
    keys = data.get("keys")
    count = data.get("count")
    return {
        "keys": keys if isinstance(keys, list) else [],
        "count": count if isinstance(count, int) else 0,
    }


def register_opportunity(
    state_dir: Path | None,
    session_id: str,
    opportunity: PhantomOpportunity,
    *,
    max_per_session: int = DEFAULT_MAX_PER_SESSION,
) -> bool:
    """Decide whether to surface ``opportunity``, bounding the trigger.

    Returns True (and records the flag) only when, within this session:

    * the opportunity's ``dedup_key`` has not already been flagged, AND
    * fewer than ``max_per_session`` opportunities have been flagged.

    State lives in a per-session sidecar JSON next to the store. Fail-soft:
    when ``state_dir`` is None (in-memory store) or any file op fails, the
    bound can't be enforced, so the function returns False rather than risk
    unbounded emission — the trigger errs quiet, never spammy.
    """
    if state_dir is None:
        return False
    try:
        path = _state_path(state_dir, session_id)
        state = _load_state(path)
        keys = list(state["keys"])  # type: ignore[arg-type]
        count = int(state["count"])  # type: ignore[arg-type]
        if opportunity.dedup_key in keys:
            return False
        if count >= max_per_session:
            return False
        keys.append(opportunity.dedup_key)
        if len(keys) > _KEYS_CAP:
            keys = keys[-_KEYS_CAP:]
        new_state = {"keys": keys, "count": count + 1}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(new_state), encoding="utf-8")
        return True
    except OSError:
        return False


__all__ = [
    "DEFAULT_MAX_PER_SESSION",
    "DEFAULT_MIN_HITS",
    "DEFAULT_MIN_QUERY_TOKENS",
    "PhantomOpportunity",
    "dedup_key_for",
    "detect_phantom_opportunity",
    "normalize_query",
    "register_opportunity",
]
