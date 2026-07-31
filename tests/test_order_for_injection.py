"""Injection-block ordering policy (#1274, proposal 14 on #1177).

The permutation is the whole product surface here, so every test asserts a
*distinguishing* order — that policy A puts the beliefs somewhere policy B
does not — rather than merely that the call returned the right number of
hits. A test that only counts would pass against the identity permutation
for all three policies, which is exactly the bug worth catching.
"""

from __future__ import annotations

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import (
    ORDER_POLICIES,
    ORDER_POLICY_LANE,
    ORDER_POLICY_LOCKS_LAST,
    ORDER_POLICY_SCORE_DESC,
    order_for_injection,
    resolve_order_policy,
)


def _mk(bid: str, lock_level: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=f"content of {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-07-31T00:00:00Z" if lock_level == LOCK_USER else None,
        created_at="2026-07-31T00:00:00Z",
        last_retrieved_at=None,
    )


def _ids(hits: list[Belief]) -> list[str]:
    return [b.id for b in hits]


# --- lane (the default) ---------------------------------------------------


def test_lane_is_the_identity_permutation() -> None:
    hits = [_mk("a", LOCK_USER), _mk("b"), _mk("c")]
    assert _ids(order_for_injection(hits, ORDER_POLICY_LANE)) == ["a", "b", "c"]


def test_lane_returns_a_copy_not_the_input_list() -> None:
    # The caller mutating the rendered list must not corrupt the retrieval
    # result it was derived from.
    hits = [_mk("a"), _mk("b")]
    out = order_for_injection(hits, ORDER_POLICY_LANE)
    out.clear()
    assert _ids(hits) == ["a", "b"]


def test_default_resolver_is_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin the ambient env/TOML off: a live config in the developer's tree
    # would otherwise decide this test's answer.
    monkeypatch.delenv("AELFRICE_ORDER_POLICY", raising=False)
    monkeypatch.setattr(
        "aelfrice.retrieval._read_toml_str_for", lambda *a, **k: None
    )
    assert resolve_order_policy() == ORDER_POLICY_LANE


# --- locks_last -----------------------------------------------------------


def test_locks_last_moves_the_locked_tier_to_the_end() -> None:
    hits = [_mk("L1", LOCK_USER), _mk("n1"), _mk("L2", LOCK_USER), _mk("n2")]
    out = _ids(order_for_injection(hits, ORDER_POLICY_LOCKS_LAST))
    # Distinguishing: under `lane` the locks lead; here they trail.
    assert out == ["n1", "n2", "L1", "L2"]


def test_locks_last_is_stable_within_each_tier() -> None:
    hits = [_mk("n1"), _mk("n2"), _mk("n3"), _mk("L1", LOCK_USER)]
    out = _ids(order_for_injection(hits, ORDER_POLICY_LOCKS_LAST))
    assert out == ["n1", "n2", "n3", "L1"]


def test_locks_last_on_an_all_locked_block_is_the_identity() -> None:
    # 27.6% of real blocks are 100% locks (#1274 pre-flight); this policy is
    # a no-op on them, and that must not be mistaken for a bug.
    hits = [_mk("L1", LOCK_USER), _mk("L2", LOCK_USER)]
    assert _ids(order_for_injection(hits, ORDER_POLICY_LOCKS_LAST)) == ["L1", "L2"]


# --- score_desc -----------------------------------------------------------


def test_score_desc_sorts_non_locks_by_descending_score() -> None:
    hits = [_mk("n1"), _mk("n2"), _mk("n3")]
    # Scores are log-domain and negative; -0.1 is the best.
    scores = {"n1": -3.0, "n2": -0.1, "n3": -1.5}
    out = _ids(order_for_injection(hits, ORDER_POLICY_SCORE_DESC, scores=scores))
    assert out == ["n2", "n3", "n1"]


def test_score_desc_keeps_the_locked_tier_first() -> None:
    hits = [_mk("n1"), _mk("L1", LOCK_USER), _mk("n2")]
    scores = {"n1": -5.0, "n2": -0.5}
    out = _ids(order_for_injection(hits, ORDER_POLICY_SCORE_DESC, scores=scores))
    assert out == ["L1", "n2", "n1"]


def test_score_desc_sorts_an_unscored_hit_last_not_first() -> None:
    # Scores are negative, so a missing id defaulting to 0.0 would sort it
    # to the TOP — the opposite of intended. It must default to -inf.
    hits = [_mk("scored"), _mk("missing")]
    out = _ids(
        order_for_injection(
            hits, ORDER_POLICY_SCORE_DESC, scores={"scored": -9.9}
        )
    )
    assert out == ["scored", "missing"]


def test_score_desc_ties_break_on_original_index() -> None:
    hits = [_mk("first"), _mk("second")]
    scores = {"first": -1.0, "second": -1.0}
    out = _ids(order_for_injection(hits, ORDER_POLICY_SCORE_DESC, scores=scores))
    assert out == ["first", "second"]


def test_score_desc_without_scores_degrades_to_lane_and_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A silent downgrade to a proxy signal is the #1271 failure mode: an
    # explicit setting quietly replaced by a different one.
    hits = [_mk("a"), _mk("b")]
    out = _ids(order_for_injection(hits, ORDER_POLICY_SCORE_DESC))
    assert out == ["a", "b"]
    assert "score_desc" in capsys.readouterr().err


# --- invariants shared by every policy ------------------------------------


@pytest.mark.parametrize("policy", ORDER_POLICIES)
def test_every_policy_is_a_permutation_dropping_nothing(policy: str) -> None:
    hits = [_mk("L1", LOCK_USER), _mk("n1"), _mk("n2"), _mk("L2", LOCK_USER)]
    scores = {"n1": -1.0, "n2": -2.0}
    out = order_for_injection(hits, policy, scores=scores)
    assert sorted(_ids(out)) == sorted(_ids(hits))
    assert len(out) == len(hits)


def test_unknown_policy_falls_back_to_identity() -> None:
    hits = [_mk("a"), _mk("b")]
    assert _ids(order_for_injection(hits, "no-such-policy")) == ["a", "b"]


def test_resolver_rejects_an_unknown_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AELFRICE_ORDER_POLICY", "u_shaped")
    monkeypatch.setattr(
        "aelfrice.retrieval._read_toml_str_for", lambda *a, **k: None
    )
    # `u_shaped` is proposed but not implemented; it must not silently
    # resolve to something else that looks like it worked.
    assert resolve_order_policy() == ORDER_POLICY_LANE
    assert "u_shaped" in capsys.readouterr().err
