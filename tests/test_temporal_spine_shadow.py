"""Pure-logic unit tests for the #1064 G2 shadow-eval harness.

The full shadow eval needs a real hook-ingested backfilled store and a real
turn log (private, machine-specific) and is run on-demand — same posture as
``temporal_spine_latency.py``. What CI *can* pin is the harness's pure logic:
the chain-length aggregation, the lane-aggregate arithmetic, and the
query-loader's role filter + order-preserving dedup. Those carry the report's
correctness; the retrieval loop is a thin driver over already-tested
``retrieve_v2`` + ``RankInvarianceAccumulator``.
"""
from __future__ import annotations

import json

from benchmarks.temporal_spine_shadow import (
    LaneAggregates,
    summarize_chain_lengths,
    load_user_queries,
)


def test_summarize_chain_lengths_basic() -> None:
    # 5 sessions: lengths 1, 2, 3, 4, 10  → 20 beliefs total.
    summary = summarize_chain_lengths([3, 1, 10, 2, 4], no_session_beliefs=7)
    assert summary.n_sessions == 5
    assert summary.n_beliefs_in_chains == 20
    assert summary.no_session_beliefs == 7
    assert summary.singletons == 1          # the length-1 session
    assert summary.min_len == 1
    assert summary.max_len == 10
    assert summary.median_len == 3
    # histogram buckets are disjoint and cover every session exactly once.
    assert sum(summary.histogram.values()) == summary.n_sessions
    assert summary.histogram["1"] == 1
    assert summary.histogram["2"] == 1
    assert summary.histogram["3-4"] == 2   # lengths 3 and 4
    assert summary.histogram["10-24"] == 1
    # belief-share is a fraction of the 20 chained beliefs.
    assert abs(summary.histogram_belief_share["10-24"] - 10 / 20) < 1e-9
    assert abs(sum(summary.histogram_belief_share.values()) - 1.0) < 1e-9


def test_summarize_chain_lengths_empty() -> None:
    summary = summarize_chain_lengths([], no_session_beliefs=0)
    assert summary.n_sessions == 0
    assert summary.n_beliefs_in_chains == 0
    assert summary.histogram == {}
    assert summary.max_len == 0


def test_summarize_chain_lengths_heavy_tail() -> None:
    # One mega-chain dominates the belief mass but is a single session —
    # the production shape open question 1 warns about.
    summary = summarize_chain_lengths([1] * 99 + [9000])
    assert summary.n_sessions == 100
    assert summary.singletons == 99
    assert summary.max_len == 9000
    # 99 short sessions but the lone mega-chain holds ~99% of beliefs.
    assert summary.histogram_belief_share["1000+"] > 0.98
    assert summary.histogram["1000+"] == 1


def test_lane_aggregates_arithmetic() -> None:
    lane = LaneAggregates(
        n_queries=10, lane_fired=8,
        spine_candidates=100, spine_survivors=30,
    )
    assert lane.fire_rate == 0.8
    assert lane.trim_loss == 70
    assert lane.trim_rate == 0.7
    assert lane.mean_survivors_per_query == 3.0


def test_lane_aggregates_zero_division_safe() -> None:
    lane = LaneAggregates()
    assert lane.fire_rate == 0.0
    assert lane.trim_rate == 0.0
    assert lane.mean_survivors_per_query == 0.0


def test_load_user_queries_filters_role_and_dedups(tmp_path) -> None:
    log = tmp_path / "turns.jsonl"
    rows = [
        {"role": "user", "text": "first question"},
        {"role": "assistant", "text": "an answer — must be excluded"},
        {"role": "user", "text": "second question"},
        {"role": "user", "text": "first question"},   # duplicate → dropped
        {"role": "user", "text": "   "},               # blank → dropped
        {"role": None, "text": "no role → dropped"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    queries = load_user_queries(str(log))
    # order-preserving, deduped, user-only, non-blank.
    assert queries == ["first question", "second question"]


def test_load_user_queries_skips_malformed_lines(tmp_path) -> None:
    log = tmp_path / "turns.jsonl"
    log.write_text(
        '{"role": "user", "text": "kept"}\n'
        "not json at all\n"
        "\n"
        '{"role": "user", "text": "also kept"}\n'
    )
    assert load_user_queries(str(log)) == ["kept", "also kept"]
