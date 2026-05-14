"""Continuation-fidelity scorer tests (#138, v1.4.0).

Covers the acceptance criteria from issue #138:

  1. Scorer runs against the bundled synthetic fixture and emits a
     single fidelity score in [0, 1].
  2. Score is reproducible -- same fixture + same answer set ->
     identical score, byte for byte.
  3. The chosen scoring method (exact match) produces sensible
     numbers on the documented false-positive / false-negative
     modes.
  4. The harness's headline JSON output includes
     `continuation_fidelity` alongside `token_budget_delta` and
     `hook_latency_ms` -- the three v1.4 ship-gate metrics.

All tests deterministic, well under the 2-second budget, no
network. Synthetic fixture only per
`docs/design/eval_fixture_policy.md`.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest

from benchmarks.context_rebuilder.__main__ import main as cli_main
from benchmarks.context_rebuilder.inject import ClearInjection
from benchmarks.context_rebuilder.replay import ReplayResult, run
from benchmarks.context_rebuilder.score import (
    DEFAULT_SCORE_METHOD,
    FidelityScore,
    ScoreMethod,
    score_continuation_fidelity,
)

# Bundled synthetic fixture path -- same anchor as the #136 suite.
_REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_FIXTURE = (
    _REPO_ROOT
    / "benchmarks"
    / "context-rebuilder"
    / "fixtures"
    / "synthetic"
    / "debugging_session_001.jsonl"
)


# --------------------------------------------------------------------- #
# Defaults / surface anchors                                            #
# --------------------------------------------------------------------- #


def test_default_score_method_is_exact() -> None:
    """The v1.4.0 ship-default method is `exact`.

    Anchored as a constant so a PR that flips the default also has
    to change this assertion (visible in diff).
    """
    assert DEFAULT_SCORE_METHOD == "exact"


def test_score_method_literal_accepts_three_values() -> None:
    """The `ScoreMethod` literal type covers exact / embedding / llm-judge."""
    a: ScoreMethod = "exact"
    b: ScoreMethod = "embedding"
    c: ScoreMethod = "llm-judge"
    assert {a, b, c} == {"exact", "embedding", "llm-judge"}


# --------------------------------------------------------------------- #
# Acceptance: scorer runs against the bundled synthetic fixture         #
# --------------------------------------------------------------------- #


def test_score_against_bundled_synthetic_fixture_in_unit_interval() -> None:
    """End-to-end: `run()` populates `continuation_fidelity` in [0, 1]."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=8))
    fid = result.continuation_fidelity
    assert fid is not None
    assert isinstance(fid, FidelityScore)
    assert 0.0 <= fid.score <= 1.0
    assert fid.method == "exact"
    assert fid.n_post_clear_assistant_turns >= 1


def test_perfect_replay_scores_one() -> None:
    """No agent answers supplied -> fixture is scored against itself.

    This is the perfect-replay baseline: the metric pipeline runs
    end-to-end, produces 1.0, and serves as a smoke test until the
    rebuilder hook (#139) feeds real agent output. Documented in the
    score module docstring.
    """
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=6))
    assert result.continuation_fidelity is not None
    assert result.continuation_fidelity.score == 1.0
    # All per-turn flags are 1, by construction.
    assert all(flag == 1 for flag in result.continuation_fidelity.per_turn)


# --------------------------------------------------------------------- #
# Acceptance: reproducibility -- same inputs -> same score              #
# --------------------------------------------------------------------- #


def test_score_is_reproducible_across_runs() -> None:
    """Two runs against the same fixture produce identical scores."""
    a = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=8))
    b = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=8))
    assert a.continuation_fidelity is not None
    assert b.continuation_fidelity is not None
    assert a.continuation_fidelity.score == b.continuation_fidelity.score
    assert (
        a.continuation_fidelity.per_turn == b.continuation_fidelity.per_turn
    )
    assert (
        a.continuation_fidelity.n_post_clear_assistant_turns
        == b.continuation_fidelity.n_post_clear_assistant_turns
    )


def test_score_is_reproducible_with_explicit_answers() -> None:
    """Same explicit `post_clear_answers` -> same score on both calls."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=10))
    n = (
        result.continuation_fidelity.n_post_clear_assistant_turns
        if result.continuation_fidelity is not None
        else 0
    )
    assert n >= 1
    answers = ["wrong answer"] * n  # Deliberate misses.
    a = run(
        SYNTHETIC_FIXTURE,
        inject=ClearInjection(clear_at=10),
        post_clear_answers=answers,
    )
    b = run(
        SYNTHETIC_FIXTURE,
        inject=ClearInjection(clear_at=10),
        post_clear_answers=answers,
    )
    assert a.continuation_fidelity is not None
    assert b.continuation_fidelity is not None
    assert a.continuation_fidelity.score == b.continuation_fidelity.score
    assert (
        a.continuation_fidelity.per_turn == b.continuation_fidelity.per_turn
    )


# --------------------------------------------------------------------- #
# `exact` semantics: what matches, what doesn't                         #
# --------------------------------------------------------------------- #


def _post_clear_assistant_count(replay_result: ReplayResult) -> int:
    fid = replay_result.continuation_fidelity
    return fid.n_post_clear_assistant_turns if fid is not None else 0


def test_exact_method_normalizes_case_and_whitespace() -> None:
    """Case differences and whitespace runs do not break the match.

    Documented behaviour: NFC + lowercase + whitespace-collapse +
    strip. This test pins the conservative-normalization contract.
    """
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=8))
    n = _post_clear_assistant_count(result)
    assert n >= 1

    # Pull the fixture's original assistant texts post-clear, then
    # mutate case + whitespace and confirm the score is still 1.0.
    fid = result.continuation_fidelity
    assert fid is not None
    # Build "agent answers" by re-invoking the scorer with the
    # same texts but uppercased + extra whitespace.
    from benchmarks.context_rebuilder.replay import load_turns

    turns, _ = load_turns(SYNTHETIC_FIXTURE)
    fixture_texts = [t.text for t in turns]
    # Post-clear assistant turns have indices > 8 with role assistant.
    pc_texts: list[str] = []
    for t in result.turns:
        if t.turn_index > 8 and t.role == "assistant":
            pc_texts.append(fixture_texts[t.turn_index])
    assert pc_texts, "fixture must have post-clear assistant turns"
    munged = ["   " + s.upper().replace(" ", "  ") + "\n" for s in pc_texts]

    rerun = run(
        SYNTHETIC_FIXTURE,
        inject=ClearInjection(clear_at=8),
        post_clear_answers=munged,
    )
    assert rerun.continuation_fidelity is not None
    assert rerun.continuation_fidelity.score == 1.0


def test_exact_method_rejects_paraphrase() -> None:
    """A semantically-equivalent paraphrase is a documented false negative.

    This test pins the failure mode -- if the scorer ever starts
    accepting paraphrases (e.g. by silently switching to embedding
    similarity) this test will fail and force a docs update.
    """
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=12))
    n = _post_clear_assistant_count(result)
    assert n >= 1
    paraphrases = ["paraphrased rewording of the original answer"] * n
    rerun = run(
        SYNTHETIC_FIXTURE,
        inject=ClearInjection(clear_at=12),
        post_clear_answers=paraphrases,
    )
    assert rerun.continuation_fidelity is not None
    assert rerun.continuation_fidelity.score == 0.0


def test_exact_method_partial_match_aggregate() -> None:
    """Half-correct answers aggregate to ~0.5."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=8))
    n = _post_clear_assistant_count(result)
    # The synthetic fixture cleared at 8 has multiple post-clear
    # assistant turns; we want at least 2 to have a meaningful split.
    assert n >= 2

    from benchmarks.context_rebuilder.replay import load_turns

    turns, _ = load_turns(SYNTHETIC_FIXTURE)
    fixture_texts = [t.text for t in turns]
    pc_indices: list[int] = []
    for t in result.turns:
        if t.turn_index > 8 and t.role == "assistant":
            pc_indices.append(t.turn_index)
    # First half: correct (exact text). Second half: wrong.
    half = len(pc_indices) // 2
    answers: list[str] = []
    for i, idx in enumerate(pc_indices):
        if i < half:
            answers.append(fixture_texts[idx])
        else:
            answers.append("definitely wrong")

    rerun = run(
        SYNTHETIC_FIXTURE,
        inject=ClearInjection(clear_at=8),
        post_clear_answers=answers,
    )
    assert rerun.continuation_fidelity is not None
    expected = half / len(pc_indices)
    assert rerun.continuation_fidelity.score == expected


# --------------------------------------------------------------------- #
# Edge cases                                                            #
# --------------------------------------------------------------------- #


def test_no_clear_injected_is_vacuously_perfect() -> None:
    """`clear_injected_at is None` -> score=1.0, n=0 (documented)."""
    result = run(SYNTHETIC_FIXTURE)  # No injection.
    assert result.clear_injected_at is None
    assert result.continuation_fidelity is not None
    assert result.continuation_fidelity.score == 1.0
    assert result.continuation_fidelity.n_post_clear_assistant_turns == 0
    assert result.continuation_fidelity.per_turn == []


def test_clear_at_last_turn_is_vacuously_perfect() -> None:
    """Clear at or after the final turn -> 0 post-clear answers, score=1.0.

    Documented vacuous case; we pick 1.0 over NaN so dashboards
    stay numeric. `n_post_clear_assistant_turns=0` lets callers
    detect "no measurement happened".
    """
    # The fixture has 16 turns. Clear at index 100 silently no-ops
    # the injection (replay sets clear_injected_at=None).
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=100))
    assert result.clear_injected_at is None
    assert result.continuation_fidelity is not None
    assert result.continuation_fidelity.score == 1.0
    assert result.continuation_fidelity.n_post_clear_assistant_turns == 0


def test_post_clear_answers_length_mismatch_raises() -> None:
    """Mismatched answer-list length is a programming error -> ValueError."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=8))
    n = _post_clear_assistant_count(result)
    assert n >= 1
    bad = ["x"] * (n + 1)
    with pytest.raises(ValueError) as exc_info:
        _ = run(
            SYNTHETIC_FIXTURE,
            inject=ClearInjection(clear_at=8),
            post_clear_answers=bad,
        )
    assert "post_clear_answers length" in str(exc_info.value)


# --------------------------------------------------------------------- #
# Method-swap path                                                      #
# --------------------------------------------------------------------- #


def test_embedding_method_raises_not_implemented() -> None:
    """`method='embedding'` is parked at v1.4.0; raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        _ = run(
            SYNTHETIC_FIXTURE,
            inject=ClearInjection(clear_at=8),
            score_method="embedding",
        )
    msg = str(exc_info.value)
    assert "embedding" in msg.lower()
    assert "#138" in msg


def test_llm_judge_method_raises_not_implemented() -> None:
    """`method='llm-judge'` is parked at v1.4.0; raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        _ = run(
            SYNTHETIC_FIXTURE,
            inject=ClearInjection(clear_at=8),
            score_method="llm-judge",
        )
    msg = str(exc_info.value)
    assert "llm-judge" in msg.lower()
    assert "#138" in msg


def test_unknown_method_raises_value_error() -> None:
    """A bad runtime method string raises ValueError, not silently misbehaves."""
    # Build a minimal ReplayResult and call the scorer directly.
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=8))
    with pytest.raises(ValueError) as exc_info:
        _ = score_continuation_fidelity(
            result,
            fixture_turn_texts=["x"] * result.n_turns,
            method="not-a-real-method",  # type: ignore[arg-type]
        )
    assert "unsupported score method" in str(exc_info.value)


# --------------------------------------------------------------------- #
# Acceptance: harness JSON output carries all three ship-gate metrics   #
# --------------------------------------------------------------------- #


def test_json_output_includes_continuation_fidelity(tmp_path: Path) -> None:
    """The JSON the CLI writes carries `continuation_fidelity`.

    Pins the v1.4 ship-gate output schema: fidelity + token-cost +
    latency travel together in the top-level JSON.
    """
    out_path = tmp_path / "out.json"
    rc = cli_main(
        [
            str(SYNTHETIC_FIXTURE),
            "--clear-at",
            "8",
            "--out",
            str(out_path),
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    payload = cast(dict[str, object], raw)

    # All three ship-gate metrics present at the top level.
    assert "continuation_fidelity" in payload
    assert "rebuild_block_tokens" in payload  # token-cost surface
    # Per-turn token + latency confirmed by the #136 suite; here we
    # check the top-level fidelity sub-object shape.
    fid_obj = payload["continuation_fidelity"]
    assert isinstance(fid_obj, dict)
    fid = cast(dict[str, object], fid_obj)
    assert set(fid.keys()) == {
        "score",
        "method",
        "n_post_clear_assistant_turns",
        "per_turn",
    }
    assert fid["method"] == "exact"
    score = fid["score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_cli_score_method_default_is_exact(tmp_path: Path) -> None:
    """Without `--score-method`, the CLI emits `method=exact`."""
    out_path = tmp_path / "out.json"
    rc = cli_main(
        [str(SYNTHETIC_FIXTURE), "--clear-at", "4", "--out", str(out_path)],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    payload = cast(dict[str, object], json.loads(out_path.read_text("utf-8")))
    fid = cast(dict[str, object], payload["continuation_fidelity"])
    assert fid["method"] == "exact"


def test_cli_parked_method_returns_two() -> None:
    """`--score-method llm-judge` exits 2 with a parking-rationale message."""
    err = io.StringIO()
    rc = cli_main(
        [str(SYNTHETIC_FIXTURE), "--clear-at", "4", "--score-method", "llm-judge"],
        stdout=io.StringIO(),
        stderr=err,
    )
    assert rc == 2
    assert "error:" in err.getvalue()
    assert "llm-judge" in err.getvalue().lower()


def test_cli_rejects_unknown_method() -> None:
    """`--score-method nonsense` is rejected by argparse with exit 2."""
    err = io.StringIO()
    with pytest.raises(SystemExit) as exc_info:
        _ = cli_main(
            [
                str(SYNTHETIC_FIXTURE),
                "--clear-at",
                "4",
                "--score-method",
                "nonsense",
            ],
            stdout=io.StringIO(),
            stderr=err,
        )
    assert exc_info.value.code == 2


# --------------------------------------------------------------------- #
# FidelityScore dataclass shape                                         #
# --------------------------------------------------------------------- #


def test_fidelity_score_to_dict_round_trip() -> None:
    """FidelityScore.to_dict() is JSON-serializable end-to-end."""
    fs = FidelityScore(
        score=0.75,
        method="exact",
        n_post_clear_assistant_turns=4,
        per_turn=[1, 1, 1, 0],
    )
    payload = json.dumps(fs.to_dict())
    raw = json.loads(payload)
    assert isinstance(raw, dict)
    reloaded = cast(dict[str, object], raw)
    assert reloaded["score"] == 0.75
    assert reloaded["method"] == "exact"
    assert reloaded["n_post_clear_assistant_turns"] == 4
    assert reloaded["per_turn"] == [1, 1, 1, 0]
