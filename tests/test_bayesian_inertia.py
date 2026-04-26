"""R&D round-1 E1b: high evidence-mass beliefs are inertial.

Property: Spearman rho < -0.5 between (alpha+beta) and per-event
|delta posterior_mean| under a single +1 alpha update.

Spearman is hand-rolled (stdlib only) -- scipy is banned per MVP_SCOPE.
"""
from __future__ import annotations

from aelfrice.scoring import posterior_mean


def _rank(values: list[float]) -> list[float]:
    """Average-rank assignment (handles ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        # ranks i..j (inclusive) get average rank (1-indexed)
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def test_bayesian_inertia() -> None:
    masses: list[float] = []
    deltas: list[float] = []
    # 50 beliefs with (alpha+beta) ranging from 1 to ~200.
    # Hold posterior mean ~0.5 (alpha == beta) so that the only varying
    # factor is the evidence mass.
    for k in range(50):
        total = 1.0 + k * 4.0  # 1, 5, 9, ... ~197
        alpha = total / 2.0
        beta = total / 2.0
        before = posterior_mean(alpha, beta)
        after = posterior_mean(alpha + 1.0, beta)
        masses.append(total)
        deltas.append(abs(after - before))

    rho = _spearman(masses, deltas)
    assert rho < -0.5, f"expected Spearman rho < -0.5, got {rho:.4f}"
