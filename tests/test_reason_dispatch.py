"""Pure-derivation tests for R3 dispatch + suggested-updates helpers.

Covers :func:`aelfrice.reason.dispatch_policy` and
:func:`aelfrice.reason.suggested_updates`. Both are pure functions of
``(verdict, impasses[, hops])`` returned from R1's :func:`classify`,
so the tests pin the verdict + impasses directly and assert the
mapping byte-for-byte.
"""
from __future__ import annotations

import pytest

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_RELATES_TO,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.reason import (
    DispatchItem,
    Impasse,
    ImpasseKind,
    SubagentRole,
    SuggestedUpdate,
    Verdict,
    dispatch_policy,
    suggested_updates,
)


def _mk(bid: str, *, alpha: float = 1.0, beta: float = 1.0) -> Belief:
    return Belief(
        id=bid,
        content=bid,
        content_hash=f"h-{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-11T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _hop(b: Belief, depth: int = 1) -> ScoredHop:
    return ScoredHop(belief=b, score=0.8, depth=depth, path=[EDGE_RELATES_TO])


def test_sufficient_dispatches_nothing() -> None:
    assert dispatch_policy(Verdict.SUFFICIENT, []) == []


def test_sufficient_with_phantom_impasse_still_dispatches_nothing() -> None:
    # Verdict beats impasses for the dispatch gate.
    assert dispatch_policy(
        Verdict.SUFFICIENT,
        [Impasse(kind=ImpasseKind.GAP, belief_ids=("x",), note="leaf")],
    ) == []


def test_contradictory_emits_fork_resolver_per_tie() -> None:
    impasses = [
        Impasse(kind=ImpasseKind.TIE, belief_ids=("a", "b"), note="tied"),
        Impasse(kind=ImpasseKind.TIE, belief_ids=("c", "d"), note="tied"),
    ]
    items = dispatch_policy(Verdict.CONTRADICTORY, impasses)
    assert items == [
        DispatchItem(
            role=SubagentRole.FORK_RESOLVER, belief_ids=("a", "b"), note="tied"
        ),
        DispatchItem(
            role=SubagentRole.FORK_RESOLVER, belief_ids=("c", "d"), note="tied"
        ),
    ]


def test_partial_with_constraint_failure_emits_verifier() -> None:
    impasses = [
        Impasse(
            kind=ImpasseKind.CONSTRAINT_FAILURE,
            belief_ids=("locked",),
            note="blocked",
        )
    ]
    items = dispatch_policy(Verdict.PARTIAL, impasses)
    assert items == [
        DispatchItem(
            role=SubagentRole.VERIFIER,
            belief_ids=("locked",),
            note="blocked",
        )
    ]


def test_uncertain_with_gaps_emits_gap_filler() -> None:
    impasses = [
        Impasse(kind=ImpasseKind.GAP, belief_ids=("g1",), note="leaf"),
        Impasse(kind=ImpasseKind.GAP, belief_ids=("g2",), note="leaf"),
    ]
    items = dispatch_policy(Verdict.UNCERTAIN, impasses)
    assert [i.role for i in items] == [
        SubagentRole.GAP_FILLER,
        SubagentRole.GAP_FILLER,
    ]
    assert [i.belief_ids for i in items] == [("g1",), ("g2",)]


def test_insufficient_with_no_change_emits_gap_filler() -> None:
    # R1 emits INSUFFICIENT only when NO_CHANGE fires; dispatch maps
    # NO_CHANGE → Gap-filler so the host agent fans out seed-recovery.
    impasses = [
        Impasse(
            kind=ImpasseKind.NO_CHANGE,
            belief_ids=("s1", "s2"),
            note="no expansions",
        )
    ]
    items = dispatch_policy(Verdict.INSUFFICIENT, impasses)
    assert items == [
        DispatchItem(
            role=SubagentRole.GAP_FILLER,
            belief_ids=("s1", "s2"),
            note="no expansions",
        )
    ]


def test_dispatch_order_matches_impasse_order() -> None:
    # Mixed-kind impasses preserve input order in the output.
    impasses = [
        Impasse(kind=ImpasseKind.GAP, belief_ids=("g",), note="leaf"),
        Impasse(kind=ImpasseKind.TIE, belief_ids=("a", "b"), note="tied"),
        Impasse(
            kind=ImpasseKind.CONSTRAINT_FAILURE,
            belief_ids=("c",),
            note="blocked",
        ),
    ]
    items = dispatch_policy(Verdict.CONTRADICTORY, impasses)
    assert [i.role for i in items] == [
        SubagentRole.GAP_FILLER,
        SubagentRole.FORK_RESOLVER,
        SubagentRole.VERIFIER,
    ]


def test_sufficient_with_confident_hops_emits_plus_one_each() -> None:
    h1 = _hop(_mk("h1", alpha=4.0, beta=2.0))
    h2 = _hop(_mk("h2", alpha=3.0, beta=3.0))
    rows = suggested_updates(Verdict.SUFFICIENT, [], [h1, h2])
    assert rows == [
        SuggestedUpdate(
            belief_id="h1",
            direction="+1",
            note="confident hop on the answer chain",
        ),
        SuggestedUpdate(
            belief_id="h2",
            direction="+1",
            note="confident hop on the answer chain",
        ),
    ]


def test_low_evidence_hops_do_not_get_plus_one() -> None:
    # alpha + beta = 2 < CONFIDENT_TRIALS_MIN → not confident, omit.
    h_low = _hop(_mk("h_low"))
    h_ok = _hop(_mk("h_ok", alpha=4.0, beta=1.0))
    rows = suggested_updates(Verdict.SUFFICIENT, [], [h_low, h_ok])
    assert [r.belief_id for r in rows] == ["h_ok"]


def test_insufficient_emits_no_plus_one_even_with_confident_hops() -> None:
    # INSUFFICIENT means the walk failed; no chain to vote up.
    h = _hop(_mk("h", alpha=10.0, beta=1.0))
    impasses = [
        Impasse(
            kind=ImpasseKind.NO_CHANGE,
            belief_ids=("s",),
            note="no expansions",
        )
    ]
    rows = suggested_updates(Verdict.INSUFFICIENT, impasses, [h])
    assert all(r.direction != "+1" for r in rows)


def test_impasse_belief_gets_question_mark() -> None:
    impasses = [
        Impasse(kind=ImpasseKind.GAP, belief_ids=("leaf",), note="leaf")
    ]
    rows = suggested_updates(Verdict.UNCERTAIN, impasses, [])
    assert rows == [
        SuggestedUpdate(belief_id="leaf", direction="?", note="leaf")
    ]


def test_belief_on_chain_and_impasse_resolves_to_question_mark() -> None:
    # A confident hop that is also an impasse locus must take ``?``,
    # not ``+1`` — impasse evidence wins so the caller knows to look.
    same = _mk("dup", alpha=10.0, beta=1.0)
    rows = suggested_updates(
        Verdict.PARTIAL,
        [Impasse(kind=ImpasseKind.GAP, belief_ids=("dup",), note="leaf")],
        [_hop(same)],
    )
    assert rows == [
        SuggestedUpdate(belief_id="dup", direction="?", note="leaf")
    ]


def test_suggested_updates_deduplicates_across_impasses() -> None:
    # Same belief in two impasses gets one ``?`` row.
    imp = [
        Impasse(kind=ImpasseKind.GAP, belief_ids=("x",), note="first"),
        Impasse(kind=ImpasseKind.TIE, belief_ids=("x", "y"), note="second"),
    ]
    rows = suggested_updates(Verdict.CONTRADICTORY, imp, [])
    assert [(r.belief_id, r.direction) for r in rows] == [
        ("x", "?"),
        ("y", "?"),
    ]
    # First-occurrence note wins.
    assert rows[0].note == "first"


def test_never_emits_minus_one_in_r3_minimal() -> None:
    # R2's fork-path data lands later; until then -1 is unreachable.
    impasses = [
        Impasse(kind=ImpasseKind.TIE, belief_ids=("a", "b"), note="tied"),
        Impasse(kind=ImpasseKind.GAP, belief_ids=("c",), note="leaf"),
    ]
    h = _hop(_mk("d", alpha=5.0, beta=1.0))
    for v in Verdict:
        rows = suggested_updates(v, impasses, [h])
        assert all(r.direction != "-1" for r in rows)
