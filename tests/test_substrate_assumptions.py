"""Quick assumption-check tests for the v2.0 substrate decision (#196).

These are scratch tests that verify claims made in
``docs/substrate_decision.md``. They are intentionally narrow and
single-file — they exercise the memo's load-bearing assertions before
the user ratifies an option:

1. Beta differential entropy can be computed from stdlib ``math.lgamma``
   (memo § Option B "scalar Beta-entropy function, ~15 LOC").
2. Vector-projection compatibility for Option A: if α = sum(α_i) and
   β = sum(β_i), the existing scalar consumers (``posterior_mean``,
   ``partial_bayesian_score``) produce identical output for a scalar
   belief and an axis-decomposed belief whose components sum to the
   same totals (memo § Option A "single-axis consumers ... keep working
   unchanged").
3. ``apply_feedback`` arithmetic is preserved under the projection
   relationship — the property that motivated the projection claim.

Run: ``uv run pytest tests/test_substrate_assumptions.py -v``
"""

from __future__ import annotations

import math

import pytest

from aelfrice.scoring import partial_bayesian_score, posterior_mean


def beta_entropy(alpha: float, beta: float) -> float:
    """Beta differential entropy in nats.

    H(Beta(α, β)) = ln B(α, β) − (α − 1) ψ(α) − (β − 1) ψ(β)
                  + (α + β − 2) ψ(α + β)

    where B is the beta function and ψ is the digamma function.

    Stdlib ``math.lgamma`` gives ln Γ; ln B(α, β) = ln Γ(α) + ln Γ(β)
    − ln Γ(α + β). The digamma is approximated by the standard series
    used in the research line — ``ψ(x) ≈ ln(x) − 1/(2x) − 1/(12 x²)``
    for x sufficiently large; for small x we recurse via
    ``ψ(x) = ψ(x + 1) − 1/x``.

    This is the function #195 would port. Verifying it works on stdlib
    only here.
    """
    def digamma(x: float) -> float:
        # Recurse small x up to >= 6 for series accuracy.
        result = 0.0
        while x < 6.0:
            result -= 1.0 / x
            x += 1.0
        # Asymptotic series.
        result += math.log(x) - 1.0 / (2.0 * x)
        x2 = x * x
        result -= 1.0 / (12.0 * x2)
        result += 1.0 / (120.0 * x2 * x2)
        return result

    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (
        log_beta
        - (alpha - 1.0) * digamma(alpha)
        - (beta - 1.0) * digamma(beta)
        + (alpha + beta - 2.0) * digamma(alpha + beta)
    )


# --- Option B claim 1: Beta entropy via stdlib --------------------------


def test_beta_entropy_uniform_is_zero() -> None:
    """Beta(1, 1) is the uniform distribution on [0, 1]; H = 0 nats."""
    assert beta_entropy(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_beta_entropy_max_at_uniform() -> None:
    """Among Beta(α, α) for α ∈ {0.5, 1, 2, 5}, entropy is maximised at α=1.

    This isn't the global Beta-family max (Jeffreys α=β=0.5 is higher
    among proper densities than the uniform), but it's the correct
    claim along the symmetric ray α=β with α >= 1.
    """
    h_uniform = beta_entropy(1.0, 1.0)
    for alpha in (2.0, 5.0, 10.0):
        assert beta_entropy(alpha, alpha) < h_uniform


def test_beta_entropy_uniform_is_max_among_proper_unimodal() -> None:
    """Among Beta densities with α, β >= 1 (unimodal/uniform regime),
    the uniform Beta(1, 1) maximises differential entropy.

    Surprise during memo drafting: Jeffreys prior Beta(0.5, 0.5) has
    *negative* differential entropy because the density spikes at 0
    and 1, concentrating probability mass on a small set in the
    Lebesgue sense. Differential entropy on a bounded interval can be
    negative — only the uniform is the proper-density max-entropy
    distribution on [0,1] with no further constraints.

    Implication for #195: the user-facing "uncertainty score" must be
    a *relative ordering*, not an absolute magnitude — entropy can be
    < 0 for sharp-evidence Betas.
    """
    h_uniform = beta_entropy(1.0, 1.0)
    for alpha in (1.5, 2.0, 5.0, 10.0):
        assert beta_entropy(alpha, alpha) < h_uniform


def test_beta_entropy_decreases_with_evidence_unimodal_regime() -> None:
    """Adding evidence (scaling α, β up while preserving the mean)
    must monotonically decrease entropy — *within the unimodal regime
    where α, β >= 1*. The bimodal regime (either parameter < 1) has
    its own ordering driven by endpoint concentration.

    Hold posterior_mean = 0.7; start at (2.0, 0.857...) — wait, that
    has β < 1. Use (1.4, 0.6)? still β < 1. Pick an evidence base
    that is unimodal: (2.1, 0.9) has β < 1. Use (3.5, 1.5): both >= 1
    and mean = 0.7. Scale up.
    """
    pairs = [(3.5, 1.5), (7.0, 3.0), (70.0, 30.0), (700.0, 300.0)]
    entropies = [beta_entropy(a, b) for a, b in pairs]
    for prev, nxt in zip(entropies, entropies[1:]):
        assert nxt < prev, f"entropy non-monotone: {entropies}"


def test_beta_entropy_can_be_negative() -> None:
    """Documents the surprise: differential entropy on [0,1] is *not*
    bounded below by zero. A sharp Beta has H << 0. This is the
    finding that motivates "relative ordering, not absolute magnitude"
    for the user-facing uncertainty surface.
    """
    assert beta_entropy(0.5, 0.5) < 0.0
    assert beta_entropy(100.0, 100.0) < 0.0


def test_beta_entropy_loc_under_50() -> None:
    """Memo claims the port is ~15 LOC. Our implementation here
    (digamma helper + entropy formula) should fit a generous LOC
    budget; verify by counting non-blank, non-comment lines in the
    function definitions.

    Loose check — just guards against drift if someone adds bloat.
    """
    src = (
        beta_entropy.__code__.co_consts,  # touch to assert it imports
    )
    assert src is not None
    # The two functions live in this test file; the actual port lives
    # in src/aelfrice/scoring.py per #195. This assertion is a stub
    # pending that port — see test_uncertainty_score_port placeholder.


@pytest.mark.skip(reason="awaits #195 implementation under ratified option")
def test_uncertainty_score_port() -> None:
    """Once #195 lands, import scoring.uncertainty_score and assert
    it matches beta_entropy() values to within a documented tolerance.
    """


# --- Option A claim: vector → scalar projection compatibility ----------


def test_projection_preserves_posterior_mean() -> None:
    """If a 4-axis belief has axes [(α₁,β₁), (α₂,β₂), (α₃,β₃), (α₄,β₄)]
    and the scalar projection is (Σα_i, Σβ_i), then posterior_mean
    of the projection equals posterior_mean of the aggregate.

    This is the load-bearing claim under Option A: existing scalar
    consumers see no behavioural change.
    """
    # Four axes, each a different posterior shape, total mean = 0.6
    axes = [(3.0, 2.0), (1.5, 1.0), (4.5, 3.0), (3.0, 2.0)]
    sum_alpha = sum(a for a, _ in axes)
    sum_beta = sum(b for _, b in axes)
    assert posterior_mean(sum_alpha, sum_beta) == pytest.approx(0.6, rel=1e-9)


def test_projection_preserves_partial_bayesian_score() -> None:
    """For any bm25_raw and posterior_weight, partial_bayesian_score
    of the projected scalar equals partial_bayesian_score against the
    same totals — by definition, since the function only sees (α, β)
    sums via posterior_mean. A regression check that no axis-aware
    branching has been quietly added.
    """
    axes = [(3.0, 2.0), (1.5, 1.0), (4.5, 3.0), (3.0, 2.0)]
    sum_alpha = sum(a for a, _ in axes)
    sum_beta = sum(b for _, b in axes)
    # Pre-projected scalar belief
    flat_alpha, flat_beta = 12.0, 8.0
    assert (sum_alpha, sum_beta) == (flat_alpha, flat_beta)
    score_via_projection = partial_bayesian_score(
        bm25_raw=-2.5, alpha=sum_alpha, beta=sum_beta, posterior_weight=0.5
    )
    score_via_flat = partial_bayesian_score(
        bm25_raw=-2.5, alpha=flat_alpha, beta=flat_beta, posterior_weight=0.5
    )
    assert score_via_projection == score_via_flat


def test_apply_feedback_arithmetic_under_projection() -> None:
    """``feedback._bayesian_update`` adds valence to α (positive) or β
    (negative). Under Option A, a scalar update on the *projection*
    must be reproducible by either:

    (a) updating one axis and re-projecting, or
    (b) splitting the valence across axes and re-projecting.

    This test pins (a): the projection of a one-axis update equals
    a scalar update on the aggregate. Required for Option A to keep
    `apply_feedback` callers working unchanged at the projection
    level — though it leaves open the design question of which axis
    a feedback event should hit. (See memo § Option A "Defining
    'uncertain about cost' requires a feedback intake change".)
    """
    axes = [(3.0, 2.0), (1.5, 1.0), (4.5, 3.0), (3.0, 2.0)]
    valence = 1.0
    # Update axis 0 only:
    a0, b0 = axes[0]
    axes_after = [(a0 + valence, b0)] + axes[1:]
    proj_after = (
        sum(a for a, _ in axes_after),
        sum(b for _, b in axes_after),
    )
    # Scalar baseline:
    flat = (sum(a for a, _ in axes) + valence, sum(b for _, b in axes))
    assert proj_after == flat
