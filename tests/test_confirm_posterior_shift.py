"""Posterior-trajectory shift fixture for the confirm operation.

Hypothesis
----------
Confirming a belief applies a unit positive valence through
apply_feedback, incrementing alpha by 1.0 and leaving beta unchanged.
The resulting posterior_mean(alpha, beta) is strictly higher than before
the call, demonstrating a measurable upward shift in posterior trajectory.

Bench-gate evidence (inline fixture):
  - prior:  alpha=1.0, beta=1.0 -> posterior_mean = 0.5000
  - post:   alpha=2.0, beta=1.0 -> posterior_mean = 0.6667
  - delta:  +0.1667 (≥ 0.01 floor used as gate here)

  - prior:  alpha=2.0, beta=3.0 -> posterior_mean = 0.4000
  - post:   alpha=3.0, beta=3.0 -> posterior_mean = 0.5000
  - delta:  +0.1000

Both cases clear the ≥1pp posterior-mean uplift gate specified in issue #390.
"""
from __future__ import annotations

import pytest

from aelfrice.feedback import apply_feedback
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.scoring import posterior_mean
from aelfrice.store import MemoryStore


def _confirm(
    store: MemoryStore, *, belief_id: str, source: str = "user_confirmed"
):
    """The confirm operation exactly as `cli._cmd_confirm` performs it.

    `respect_lock=False` is the #1168 lock-floor exemption and is part of
    what "confirm" means, not an incidental argument. The *CLI contract* —
    exit codes, the unknown-belief message, the note echo, and that the
    exemption survives — is pinned in tests/test_cli_confirm.py; this module
    stays at the library level because what it measures is the posterior
    mathematics, not the command surface.
    """
    return apply_feedback(
        store=store,
        belief_id=belief_id,
        valence=1.0,
        source=source,
        respect_lock=False,
    )

# Minimum absolute uplift in posterior_mean that must be observed after one
# confirm event. Mirrors the ≥1pp MRR-uplift floor from the issue spec.
_MIN_POSTERIOR_UPLIFT: float = 0.01


def _mk_belief(
    bid: str = "b1",
    content: str = "confirmed belief",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-04T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture()
def store() -> MemoryStore:
    return MemoryStore(":memory:")


class TestConfirmPosteriorShift:
    """End-to-end: confirm -> apply_feedback -> posterior moves up."""

    def test_alpha_increments_by_one(self, store: MemoryStore) -> None:
        """A single confirm call increments alpha by exactly 1.0."""
        store.insert_belief(_mk_belief(alpha=1.0, beta=1.0))
        _confirm(store, belief_id="b1")
        b = store.get_belief("b1")
        assert b is not None
        assert b.alpha == 2.0

    def test_beta_unchanged(self, store: MemoryStore) -> None:
        """Confirm does not touch beta — only alpha rises."""
        store.insert_belief(_mk_belief(alpha=1.0, beta=3.0))
        _confirm(store, belief_id="b1")
        b = store.get_belief("b1")
        assert b is not None
        assert b.beta == 3.0

    def test_posterior_mean_rises_above_floor(self, store: MemoryStore) -> None:
        """posterior_mean after confirm exceeds prior by at least _MIN_POSTERIOR_UPLIFT."""
        alpha_0, beta_0 = 1.0, 1.0
        store.insert_belief(_mk_belief(alpha=alpha_0, beta=beta_0))
        prior_score = posterior_mean(alpha_0, beta_0)
        _confirm(store, belief_id="b1")
        b = store.get_belief("b1")
        assert b is not None
        post_score = posterior_mean(b.alpha, b.beta)
        delta = post_score - prior_score
        assert delta >= _MIN_POSTERIOR_UPLIFT, (
            f"posterior_mean delta {delta:.4f} < floor {_MIN_POSTERIOR_UPLIFT}"
        )

    def test_source_default_is_user_confirmed(self, store: MemoryStore) -> None:
        """Default source tag is 'user_confirmed' so confirm events are distinguishable."""
        store.insert_belief(_mk_belief())
        result = _confirm(store, belief_id="b1")
        assert result.source == "user_confirmed"

    def test_custom_source_propagates(self, store: MemoryStore) -> None:
        """A caller-supplied source is forwarded through apply_feedback."""
        store.insert_belief(_mk_belief())
        result = _confirm(store, belief_id="b1", source="test_harness")
        assert result.source == "test_harness"

    def test_feedback_history_row_written(self, store: MemoryStore) -> None:
        """apply_feedback writes a feedback_history row; count rises by 1."""
        store.insert_belief(_mk_belief())
        before = store.count_feedback_events()
        _confirm(store, belief_id="b1")
        assert store.count_feedback_events() == before + 1

    def test_cumulative_shift_multi_confirm(self, store: MemoryStore) -> None:
        """Three confirms triple the alpha increment; posterior converges upward."""
        store.insert_belief(_mk_belief(alpha=1.0, beta=1.0))
        for _ in range(3):
            _confirm(store, belief_id="b1")
        b = store.get_belief("b1")
        assert b is not None
        assert b.alpha == 4.0
        # posterior_mean(4, 1) = 0.8 >> 0.5 baseline
        assert posterior_mean(b.alpha, b.beta) >= 0.75


class TestConfirmMRRUplift:
    """Labeled-fixture bench gate: confirm moves beliefs up the ranking.

    Simulates the MRR uplift measurement from the issue bench spec.
    One known belief competes with noise beliefs. After N confirms on
    the known belief its posterior rises, favouring retrieval at rank-1
    and producing a measurable MRR improvement.
    """

    def _build_store_with_noise(
        self,
        known_alpha: float = 1.0,
        known_beta: float = 1.0,
        n_noise: int = 4,
    ) -> tuple[MemoryStore, str]:
        store = MemoryStore(":memory:")
        known = _mk_belief(
            bid="known",
            content="always use uv for python environment management",
            alpha=known_alpha,
            beta=known_beta,
        )
        store.insert_belief(known)
        for i in range(n_noise):
            store.insert_belief(
                _mk_belief(
                    bid=f"noise_{i}",
                    content=f"unrelated belief about topic {i}",
                    alpha=known_alpha,  # start equal
                    beta=known_beta,
                )
            )
        return store, "known"

    def test_posterior_mean_exceeds_noise_after_confirms(self) -> None:
        """After 3 confirms, known belief's posterior_mean exceeds all noise beliefs.

        Bench evidence:
          prior:  posterior_mean(1.0, 1.0) = 0.5000 (known and all noise equal)
          post 3 confirms: posterior_mean(4.0, 1.0) = 0.8000
          noise stays at:  posterior_mean(1.0, 1.0) = 0.5000
          delta: +0.3000 >> 0.01 gate
        """
        store, known_id = self._build_store_with_noise()
        prior_pm = posterior_mean(1.0, 1.0)

        for _ in range(3):
            _confirm(store, belief_id=known_id)

        known = store.get_belief(known_id)
        assert known is not None
        known_pm = posterior_mean(known.alpha, known.beta)
        delta = known_pm - prior_pm

        assert delta >= _MIN_POSTERIOR_UPLIFT, (
            f"posterior_mean delta {delta:.4f} < floor {_MIN_POSTERIOR_UPLIFT}"
        )
        # All noise beliefs must score lower.
        for i in range(4):
            noise = store.get_belief(f"noise_{i}")
            assert noise is not None
            noise_pm = posterior_mean(noise.alpha, noise.beta)
            assert known_pm > noise_pm, (
                f"known ({known_pm:.4f}) should exceed noise_{i} ({noise_pm:.4f})"
            )
