"""Type-specific half-lives match the spec exactly.

Spec (carried from the previous codebase's CHANGELOG and confirmed by R&D):
    factual     336h   (14 days)
    requirement 720h   (30 days)
    preference  2016h  (12 weeks)
    correction  4032h  (24 weeks)

If any of these values drifts, decay behavior in production silently changes
and downstream property tests (test_decay_required, test_lock_floor_sharp)
no longer assert what they appear to assert. Lock the values in here.
"""
from __future__ import annotations

from aelfrice.scoring import TYPE_HALF_LIFE_SECONDS, type_half_life

_HOUR_SECONDS = 3600.0


def test_factual_half_life_336_hours() -> None:
    assert TYPE_HALF_LIFE_SECONDS["factual"] == 336.0 * _HOUR_SECONDS


def test_requirement_half_life_720_hours() -> None:
    assert TYPE_HALF_LIFE_SECONDS["requirement"] == 720.0 * _HOUR_SECONDS


def test_preference_half_life_2016_hours() -> None:
    assert TYPE_HALF_LIFE_SECONDS["preference"] == 2016.0 * _HOUR_SECONDS


def test_correction_half_life_4032_hours() -> None:
    assert TYPE_HALF_LIFE_SECONDS["correction"] == 4032.0 * _HOUR_SECONDS


def test_only_four_belief_types_have_half_lives() -> None:
    """No drift via accidental extra entries."""
    assert set(TYPE_HALF_LIFE_SECONDS.keys()) == {
        "factual",
        "requirement",
        "preference",
        "correction",
    }


def test_type_half_life_lookup_returns_factual_for_unknown_type() -> None:
    """Unknown types fall back to factual (most aggressive decay)."""
    assert type_half_life("nonexistent") == TYPE_HALF_LIFE_SECONDS["factual"]


def test_type_half_life_lookup_returns_each_known_value() -> None:
    for t, hl in TYPE_HALF_LIFE_SECONDS.items():
        assert type_half_life(t) == hl
