"""Tests for the offline cadence replay bench (#876).

The harness replays a synthetic fixture through all four would_fire
predicates and aggregates via cadence_score, producing the same
comparison a live shadow bake would — deterministically, no live data.
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.cadence import (
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
    POLICY_P3_SUBSTANTIVE,
    POLICY_P3_VELOCITY,
)
from aelfrice.cadence_score import compute_summary
from benchmarks.cadence_replay import main, replay_fixture

_FIXTURE = {
    "config": {
        "k": 15,
        "ctx_threshold": 0.8,
        "ctx_byte_window": 100000,
        "p3_velocity_threshold": 3000,
        "p3_substantive_window": 5,
        "p3_substantive_threshold": 0.6,
    },
    "selected": "p1_every_k_turns",
    "ticks": [
        # p1 fires (15 % 15 == 0); p3_substantive fires (5/5 >= 0.6).
        {
            "fire_idx": 15,
            "bytes_at_last_fire": 0,
            "fire_idx_at_last_fire": 0,
            "transcript_bytes": 10000,
            "last_prompt": "implement the parser module",
            "classifications": [True, True, True, True, True],
        },
        # p2 fires (>= watermark + "next task" boundary); p3_velocity fires.
        {
            "fire_idx": 7,
            "bytes_at_last_fire": 0,
            "fire_idx_at_last_fire": 0,
            "transcript_bytes": 120000,
            "last_prompt": "next task",
            "classifications": [False, False, False, False, False],
        },
        # nothing fires.
        {
            "fire_idx": 3,
            "bytes_at_last_fire": 0,
            "fire_idx_at_last_fire": 0,
            "transcript_bytes": 5000,
            "last_prompt": "thinking about the design",
            "classifications": [True, False, False, False, False],
        },
    ],
}


def test_replay_fixture_row_count_and_shape() -> None:
    rows = replay_fixture(_FIXTURE)
    assert len(rows) == 3
    for row in rows:
        assert set(row["shadow"]) == {
            POLICY_P1_EVERY_K_TURNS,
            POLICY_P2_CTX_THRESHOLD,
            POLICY_P3_VELOCITY,
            POLICY_P3_SUBSTANTIVE,
        }


def test_replay_per_tick_decisions() -> None:
    rows = replay_fixture(_FIXTURE)
    t0, t1, t2 = rows

    # tick 0: p1 + p3_substantive
    assert t0["shadow"][POLICY_P1_EVERY_K_TURNS]["would_fire"] is True
    assert t0["shadow"][POLICY_P2_CTX_THRESHOLD]["would_fire"] is False
    assert t0["shadow"][POLICY_P3_VELOCITY]["would_fire"] is False
    assert t0["shadow"][POLICY_P3_SUBSTANTIVE]["would_fire"] is True

    # tick 1: p2 + p3_velocity
    assert t1["shadow"][POLICY_P1_EVERY_K_TURNS]["would_fire"] is False
    assert t1["shadow"][POLICY_P2_CTX_THRESHOLD]["would_fire"] is True
    assert t1["shadow"][POLICY_P3_VELOCITY]["would_fire"] is True
    assert t1["shadow"][POLICY_P3_SUBSTANTIVE]["would_fire"] is False

    # tick 2: nothing
    assert all(
        t2["shadow"][p]["would_fire"] is False
        for p in t2["shadow"]
    )


def test_replay_selected_fired_tracks_selected_policy() -> None:
    rows = replay_fixture(_FIXTURE)
    # selected = p1; only tick 0 fires for p1
    fired = [r["fired"] for r in rows]
    assert fired == [True, False, False]


def test_replay_summary_per_policy_rates() -> None:
    summary = compute_summary(replay_fixture(_FIXTURE))
    # each policy fires on exactly one of the 3 ticks
    for policy in (
        POLICY_P1_EVERY_K_TURNS,
        POLICY_P2_CTX_THRESHOLD,
        POLICY_P3_VELOCITY,
        POLICY_P3_SUBSTANTIVE,
    ):
        assert summary.per_policy_total[policy] == 3
        assert summary.per_policy_fire_count[policy] == 1
    # 6 unordered pairs across 4 policies
    assert len(summary.pairwise_agreement) == 6


def test_replay_is_deterministic() -> None:
    """Same fixture → byte-identical rows (#605)."""
    a = replay_fixture(_FIXTURE)
    b = replay_fixture(_FIXTURE)
    # strip the would_fire/reason — reasons embed byte counts which are
    # stable across runs too, so the whole structure should match.
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_replay_cli_reads_sample_fixture(capsys) -> None:  # type: ignore[no-untyped-def]
    fixture = (
        Path(__file__).resolve().parents[1]
        / "benchmarks" / "fixtures" / "cadence_replay_sample.json"
    )
    rc = main(["--fixture", str(fixture)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "per-policy would_fire rate:" in out
    assert "pairwise policy agreement" in out


def test_replay_cli_json_output(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    fixture = tmp_path / "fx.json"
    fixture.write_text(json.dumps(_FIXTURE))
    out_path = tmp_path / "summary.json"
    rc = main(["--fixture", str(fixture), "--json", "--output", str(out_path)])
    assert rc == 0
    parsed = json.loads(out_path.read_text())
    assert parsed["per_policy_fire_count"][POLICY_P3_VELOCITY] == 1
    assert "pairwise_agreement" in parsed


def test_replay_cli_missing_fixture_returns_2(tmp_path: Path) -> None:
    rc = main(["--fixture", str(tmp_path / "nope.json")])
    assert rc == 2
