"""Post-rank score adjusters (research module — issue #153).

Three independently-toggleable effects applied after the textual-lane
ranker has produced (belief, score) pairs and before the top-K cut:

1. ``apply_locked_floor`` — Uri's protection. A belief with
   ``lock_level != LOCK_NONE`` gets a score floor: ``score = max(score,
   floor)``. Floor only; never demote.
2. ``apply_supersession_demote`` — Baki's undermining. A belief whose
   ``id`` appears in the ``superseded_ids`` set has its score
   multiplied by ``factor`` (default 0.5). The set is the union of
   targets of incoming ``SUPERSEDES`` edges, computed by the caller.
3. ``apply_recency_decay`` — Aelfrice time-tilt. Score multiplied by
   ``exp(-lam * age_days)`` for configurable ``lam`` (default
   ``1/180`` ≈ 180-day half-life).

Each function consumes and returns a fresh score list parallel to the
input ``beliefs`` list. They never mutate ``Belief`` objects. The
production wiring sequence (if the retest is positive) is intended to
be:

    scores = apply_supersession_demote(beliefs, scores, superseded_ids)
    scores = apply_recency_decay(beliefs, scores, now=now)
    scores = apply_locked_floor(beliefs, scores)

— demote and decay are multiplicative and order-independent between
themselves; the locked floor is applied last so a relevant locked
belief cannot be evicted by either effect.

This module ships only the pure-function primitives. Issue #153 is a
research issue: the deliverable is the benchmark result table at
``benchmarks/uri_baki_retest/``, not production wiring. If the retest
is positive, integration into ``retrieval.py`` lands in a follow-up
issue under the retrieval pipeline tracker (#154).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Final

from aelfrice.models import LOCK_NONE, Belief

DEFAULT_LOCKED_FLOOR: Final[float] = 0.0
"""Default floor when caller does not override.

A floor of ``0.0`` is *not* the right operational default — it does
nothing on the natural score scale (BM25-derived log-scores are
typically negative for non-matches and modest-positive for matches).
The benchmark harness sweeps the floor; production callers must
choose a value calibrated to the score distribution of the live
ranker (a quantile of the matched-document score distribution is the
intended construction)."""

DEFAULT_SUPERSESSION_FACTOR: Final[float] = 0.5
"""Default multiplicative demote for a superseded belief."""

DEFAULT_RECENCY_LAMBDA: Final[float] = 1.0 / 180.0
"""Default decay rate (1/days). 1/180 ≈ 180-day half-life
(``log(2) / lambda`` ≈ 125 days at this lambda — close enough; the
issue spec calls out 180-day half-life as the intent and the
benchmark sweeps ``lam`` anyway)."""


def apply_locked_floor(
    beliefs: Sequence[Belief],
    scores: Sequence[float],
    *,
    floor: float = DEFAULT_LOCKED_FLOOR,
) -> list[float]:
    """Raise the score of every locked belief to at least ``floor``.

    Floor only. A locked belief whose ranker-assigned score already
    exceeds ``floor`` is unchanged. Non-locked beliefs are unchanged
    regardless.
    """
    if len(beliefs) != len(scores):
        msg = (
            f"beliefs / scores length mismatch: "
            f"{len(beliefs)} vs {len(scores)}"
        )
        raise ValueError(msg)
    out = list(scores)
    for i, b in enumerate(beliefs):
        if b.lock_level != LOCK_NONE and out[i] < floor:
            out[i] = floor
    return out


def apply_supersession_demote(
    beliefs: Sequence[Belief],
    scores: Sequence[float],
    superseded_ids: Iterable[str],
    *,
    factor: float = DEFAULT_SUPERSESSION_FACTOR,
) -> list[float]:
    """Multiply the score of every superseded belief by ``factor``.

    ``superseded_ids`` is the set of belief ids targeted by an
    incoming ``SUPERSEDES`` edge — i.e., the belief that has been
    superseded. The caller computes the set; this function does not
    touch the edge store.

    ``factor`` of ``1.0`` is a no-op; ``0.0`` zeros superseded
    beliefs out entirely. Default ``0.5`` matches the issue spec.
    """
    if len(beliefs) != len(scores):
        msg = (
            f"beliefs / scores length mismatch: "
            f"{len(beliefs)} vs {len(scores)}"
        )
        raise ValueError(msg)
    sup = set(superseded_ids)
    if not sup:
        return list(scores)
    return [
        s * factor if b.id in sup else s
        for b, s in zip(beliefs, scores, strict=True)
    ]


def apply_recency_decay(
    beliefs: Sequence[Belief],
    scores: Sequence[float],
    *,
    now: datetime | None = None,
    lam: float = DEFAULT_RECENCY_LAMBDA,
) -> list[float]:
    """Multiply each score by ``exp(-lam * age_days)``.

    ``age_days`` is computed from ``Belief.created_at`` (ISO-8601
    UTC string) and ``now`` (default: ``datetime.now(timezone.utc)``).
    A belief with an unparseable or future ``created_at`` is treated
    as ``age_days = 0`` (no decay) — the function is a research
    primitive, not a sanitiser; bad timestamps are an upstream bug.

    ``lam = 0`` short-circuits to a no-op.
    """
    if len(beliefs) != len(scores):
        msg = (
            f"beliefs / scores length mismatch: "
            f"{len(beliefs)} vs {len(scores)}"
        )
        raise ValueError(msg)
    if lam == 0.0:
        return list(scores)
    ref = now if now is not None else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    out: list[float] = []
    for b, s in zip(beliefs, scores, strict=True):
        age_days = _age_days(b.created_at, ref)
        if age_days <= 0.0:
            out.append(s)
        else:
            out.append(s * math.exp(-lam * age_days))
    return out


def _age_days(created_at: str, ref: datetime) -> float:
    """Days between ``created_at`` (ISO-8601 string) and ``ref``.

    Returns 0.0 if ``created_at`` is unparseable or in the future;
    callers treat 0 as "no decay applied".
    """
    try:
        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = ref - ts
    days = delta.total_seconds() / 86400.0
    return days if days > 0.0 else 0.0
