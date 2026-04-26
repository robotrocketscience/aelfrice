"""Without decay, evidence mass runs unbounded.

Property: 100 mixed-sign feedback events on a baseline (1,1) belief.
- Without decay: final (alpha+beta) >= 2 * baseline (= 4).
- With decay (factual half-life): mass stays bounded under that threshold.
"""
from __future__ import annotations

from aelfrice.scoring import TYPE_HALF_LIFE_SECONDS, decay


def test_decay_required() -> None:
    baseline_mass = 2.0  # alpha=1, beta=1

    # --- Scenario A: no decay ---
    a_no, b_no = 1.0, 1.0
    for i in range(100):
        if i % 2 == 0:
            a_no += 1.0
        else:
            b_no += 1.0
    final_no_decay = a_no + b_no
    assert final_no_decay >= 2.0 * baseline_mass, (
        f"without decay, expected mass >= {2.0 * baseline_mass}, got {final_no_decay}"
    )

    # --- Scenario B: with decay between every event ---
    hl = TYPE_HALF_LIFE_SECONDS["factual"]
    # Step ~1 half-life per event so decay is meaningful within 100 events.
    step = hl
    a_dec, b_dec = 1.0, 1.0
    for i in range(100):
        a_dec, b_dec = decay(a_dec, b_dec, age_seconds=step, half_life_seconds=hl)
        if i % 2 == 0:
            a_dec += 1.0
        else:
            b_dec += 1.0
    final_with_decay = a_dec + b_dec
    # Decay pulls toward prior mass (1.0); single-event addition keeps it bounded.
    assert final_with_decay < 2.0 * baseline_mass, (
        f"with decay, expected mass < {2.0 * baseline_mass}, got {final_with_decay}"
    )
