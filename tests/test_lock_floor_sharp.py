"""R&D round-1 E2: lock-floor is a sharp step, not a gradient.

- Below floor (lock_level=user): decay() returns input unchanged regardless of age.
- Above floor (lock_level=none): mass decreases as ~1 - 0.5^(age/half_life)
  -- linear in that quantity by construction (allow epsilon tolerance).
"""
from __future__ import annotations

from aelfrice.scoring import TYPE_HALF_LIFE_SECONDS, decay


def test_lock_floor_sharp() -> None:
    hl = TYPE_HALF_LIFE_SECONDS["factual"]

    # --- Below floor: zero work even for extreme age ---
    locked = decay(10.0, 5.0, age_seconds=10_000.0 * hl, half_life_seconds=hl,
                   lock_level="user")
    assert locked == (10.0, 5.0), f"locked decay leaked: {locked}"

    # --- Above floor: mass shrinks predictably ---
    a0, b0 = 10.0, 5.0
    initial_mass = a0 + b0
    prior_mass = 1.0  # 0.5 + 0.5
    excess = initial_mass - prior_mass

    for k in range(11):  # age = 0, 1*hl, ... 10*hl
        age = k * hl
        a, b = decay(a0, b0, age_seconds=age, half_life_seconds=hl,
                     lock_level="none")
        observed_mass = a + b
        factor = 0.5 ** (age / hl)
        # Mass model: prior + excess * factor
        expected_mass = prior_mass + excess * factor
        assert abs(observed_mass - expected_mass) < 1e-9, (
            f"age={age}: expected {expected_mass}, got {observed_mass}"
        )
        # Mass must be monotonically non-increasing in age and bounded below by prior.
        assert observed_mass <= initial_mass + 1e-9
        assert observed_mass >= prior_mass - 1e-9
