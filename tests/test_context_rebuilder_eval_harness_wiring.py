"""Wire-up tests for benchmarks/context-rebuilder/eval_harness.py (#592).

The hyphenated parent directory blocks normal `import benchmarks.context-rebuilder`,
so the module is loaded by file path via importlib. Each test boots a fresh
in-memory store and asserts the harness's integration points behave.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HARNESS_PATH = (
    _REPO_ROOT / "benchmarks" / "context-rebuilder" / "eval_harness.py"
)
_HARNESS_MOD_NAME = "eval_harness_under_test"


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        _HARNESS_MOD_NAME, _HARNESS_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find
    # the module via sys.modules[mod.__module__].
    sys.modules[_HARNESS_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    return _load_harness()


def _write_jsonl(path: Path, turns: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")


def _make_turn(idx: int, role: str, text: str) -> dict:
    return {
        "schema_version": 1,
        "ts": f"2026-04-27T15:{40 + idx:02d}:00.000Z",
        "role": role,
        "text": text,
        "session_id": "test-session-001",
        "turn_id": f"t{idx:04d}",
    }


@pytest.fixture()
def transcript_path(tmp_path: Path) -> Path:
    """Six-turn transcript: alternating user/assistant."""
    p = tmp_path / "session.jsonl"
    turns = [
        _make_turn(0, "user", "How does the rebuilder pack L0 hits?"),
        _make_turn(1, "assistant", "L0 locked beliefs are packed first."),
        _make_turn(2, "user", "What about session-scoped retrieval?"),
        _make_turn(3, "assistant", "Session-scoped runs above L2.5 and L1."),
        _make_turn(4, "user", "Does the floor apply to L0?"),
        _make_turn(5, "assistant", "L0 always packs without floor."),
    ]
    _write_jsonl(p, turns)
    return p


# --------------------------------------------------------------------- #
# replay_to_fork                                                        #
# --------------------------------------------------------------------- #


def test_replay_to_fork_returns_memory_store(
    harness: ModuleType, transcript_path: Path
) -> None:
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=4, eval_turns=(4, 5),
    )
    store = harness.replay_to_fork(case)
    try:
        assert store is not None
        ids = list(store.list_belief_ids())
        assert len(ids) > 0
    finally:
        store.close()


def test_replay_to_fork_respects_fork_boundary(
    harness: ModuleType, transcript_path: Path
) -> None:
    """Forking at turn 2 should produce strictly fewer beliefs than
    forking at turn 6 — the harness only sees pre-fork content."""
    case_early = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(),
    )
    case_late = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=6, eval_turns=(),
    )
    s_early = harness.replay_to_fork(case_early)
    s_late = harness.replay_to_fork(case_late)
    try:
        n_early = len(list(s_early.list_belief_ids()))
        n_late = len(list(s_late.list_belief_ids()))
        assert n_early < n_late
    finally:
        s_early.close()
        s_late.close()


def test_replay_to_fork_zero_fork_turn_returns_empty_store(
    harness: ModuleType, transcript_path: Path
) -> None:
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=0, eval_turns=(),
    )
    store = harness.replay_to_fork(case)
    try:
        assert list(store.list_belief_ids()) == []
    finally:
        store.close()


# --------------------------------------------------------------------- #
# run_rebuilder                                                         #
# --------------------------------------------------------------------- #


def test_run_rebuilder_returns_block_and_positive_latency(
    harness: ModuleType, transcript_path: Path
) -> None:
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=4, eval_turns=(4, 5),
    )
    store = harness.replay_to_fork(case)
    try:
        block, latency_ms = harness.run_rebuilder(
            store, case, trigger_threshold=0.0, token_budget=2000,
        )
        assert isinstance(block, str)
        assert latency_ms >= 0.0
    finally:
        store.close()


def test_run_rebuilder_passes_threshold_as_floor(
    harness: ModuleType, transcript_path: Path
) -> None:
    """At a high enough floor, no L1 / session beliefs survive.
    Without locked beliefs in the store, the block is empty per
    rebuild_v14's silent-path contract."""
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=6, eval_turns=(),
    )
    store = harness.replay_to_fork(case)
    try:
        block_low, _ = harness.run_rebuilder(
            store, case, trigger_threshold=0.0, token_budget=2000,
        )
        block_hi, _ = harness.run_rebuilder(
            store, case, trigger_threshold=10.0, token_budget=2000,
        )
        # High floor zeroes out non-locked content. Low floor lets some
        # beliefs through (or matches; tolerate equality if the store
        # had no above-floor candidates either way).
        assert len(block_hi) <= len(block_low)
    finally:
        store.close()


def test_run_rebuilder_empty_store_returns_empty_block(
    harness: ModuleType, transcript_path: Path
) -> None:
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=0, eval_turns=(),
    )
    store = harness.replay_to_fork(case)
    try:
        block, latency_ms = harness.run_rebuilder(
            store, case, trigger_threshold=0.0, token_budget=2000,
        )
        # No beliefs ingested + no recent_turns + no locks → empty.
        assert block == ""
        assert latency_ms >= 0.0
    finally:
        store.close()


# --------------------------------------------------------------------- #
# measure_token_cost                                                    #
# --------------------------------------------------------------------- #


def test_measure_token_cost_zero_pre_clear_returns_zero(
    harness: ModuleType, transcript_path: Path
) -> None:
    """fork_turn=0 → no pre-clear text → ratio is 0.0, not divide-by-zero."""
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=0, eval_turns=(),
    )
    assert harness.measure_token_cost("anything", case) == 0.0


def test_measure_token_cost_smaller_rebuilt_means_smaller_ratio(
    harness: ModuleType, transcript_path: Path
) -> None:
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=4, eval_turns=(),
    )
    big_block = "x" * 1000
    small_block = "x" * 10
    big = harness.measure_token_cost(big_block, case)
    small = harness.measure_token_cost(small_block, case)
    assert small < big
    assert small >= 0.0


def test_measure_token_cost_uses_shared_estimator(
    harness: ModuleType, transcript_path: Path
) -> None:
    """The harness ratio must match estimate_tokens(rebuilt) /
    estimate_tokens(pre_clear) so the constant stays in sync with
    the rebuilder's own bookkeeping."""
    from benchmarks.context_rebuilder.measure import estimate_tokens

    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=4, eval_turns=(),
    )
    rebuilt = "context block of arbitrary length"
    pre_clear = harness._pre_clear_text(case)
    expected = estimate_tokens(rebuilt) / estimate_tokens(pre_clear)
    assert harness.measure_token_cost(rebuilt, case) == pytest.approx(expected)


# --------------------------------------------------------------------- #
# replay_post_fork                                                      #
# --------------------------------------------------------------------- #


def test_replay_post_fork_returns_one_placeholder_per_eval_turn(
    harness: ModuleType, transcript_path: Path
) -> None:
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(2, 4),
    )
    results = harness.replay_post_fork("rebuilt-block", case)
    assert len(results) == 2
    assert {r["turn_idx"] for r in results} == {2, 4}
    assert all(not r["matched"] for r in results)
    assert all(r["reason"] == harness.REPLAY_PENDING_REASON for r in results)


def test_replay_post_fork_pulls_expected_from_transcript(
    harness: ModuleType, transcript_path: Path
) -> None:
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(2,),
    )
    results = harness.replay_post_fork("", case)
    assert results[0]["expected"] == "What about session-scoped retrieval?"
    assert results[0]["actual"] == ""


def test_replay_post_fork_empty_eval_turns_returns_empty_list(
    harness: ModuleType, transcript_path: Path
) -> None:
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(),
    )
    assert harness.replay_post_fork("rebuilt", case) == []


def test_replay_post_fork_score_fidelity_returns_zero(
    harness: ModuleType, transcript_path: Path
) -> None:
    """Stubbed replay → score_fidelity returns 0.0, not a crash."""
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(2, 4),
    )
    results = harness.replay_post_fork("", case)
    assert harness.score_fidelity(results) == 0.0


# --------------------------------------------------------------------- #
# replay_post_fork host-agent eval-replay (#600)                        #
# --------------------------------------------------------------------- #


def test_replay_post_fork_writes_requests_jsonl(
    harness: ModuleType, transcript_path: Path, tmp_path: Path
) -> None:
    """With `run_dir`, `replay_post_fork` writes one request row per
    eval_turn to `<run_dir>/replay_requests.jsonl`."""
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(2, 4),
    )
    run_dir = tmp_path / "run"
    harness.replay_post_fork("REBUILT_BLOCK", case, run_dir=run_dir)
    req_path = run_dir / harness.REPLAY_REQUESTS_FILENAME
    assert req_path.is_file()
    rows = [json.loads(line) for line in req_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    by_idx = {r["turn_idx"]: r for r in rows}
    assert set(by_idx) == {2, 4}
    # Each row carries the rebuilt block plus the user prompt at-or-before
    # the eval turn.
    assert by_idx[2]["rebuilt_block"] == "REBUILT_BLOCK"
    assert by_idx[2]["user_turn"] == "What about session-scoped retrieval?"
    assert by_idx[2]["expected"] == "What about session-scoped retrieval?"
    # eval_turn=4 is itself a user turn → user_turn == expected.
    assert by_idx[4]["user_turn"] == "Does the floor apply to L0?"


def test_replay_post_fork_pending_reason_when_no_responses(
    harness: ModuleType, transcript_path: Path, tmp_path: Path
) -> None:
    """run_dir set but no response file → rows hold `pending_replay`,
    distinct from the bare-stub `needs_replay_client`."""
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(2, 4),
    )
    run_dir = tmp_path / "run"
    rows = harness.replay_post_fork("rebuilt", case, run_dir=run_dir)
    assert all(not r["matched"] for r in rows)
    assert all(r["reason"] == harness.PENDING_REPLAY_REASON for r in rows)


def test_replay_pending_round_trip(
    harness: ModuleType, transcript_path: Path, tmp_path: Path
) -> None:
    """End-to-end: run replay_post_fork once to emit requests, hand-author
    a `replay_responses.jsonl` (no subagent), re-invoke replay_post_fork,
    and assert score_fidelity reports the substring-match verdict.

    Acceptance bullet from #600. No model client involved — the round
    trip is the contract; the dispatcher is operator-driven.
    """
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(2, 4),
    )
    run_dir = tmp_path / "run"

    # First pass: requests get written, responses absent → all pending.
    first_pass = harness.replay_post_fork("rebuilt", case, run_dir=run_dir)
    assert (run_dir / harness.REPLAY_REQUESTS_FILENAME).is_file()
    assert all(r["reason"] == harness.PENDING_REPLAY_REASON for r in first_pass)
    assert harness.score_fidelity(first_pass) == 0.0

    # Hand-author replay_responses.jsonl: turn 2 substring-matches the
    # expected text, turn 4 does not (drops to needs_llm_judge).
    # Expected for idx=2 is the literal text at line 2:
    #   "What about session-scoped retrieval?"
    responses_path = run_dir / harness.REPLAY_RESPONSES_FILENAME
    responses_path.write_text(
        json.dumps({
            "turn_idx": 2,
            "actual": "Earlier we discussed: WHAT ABOUT SESSION-SCOPED RETRIEVAL? Yes.",
        }) + "\n" +
        json.dumps({
            "turn_idx": 4,
            "actual": "no idea what the floor does, sorry",
        }) + "\n",
        encoding="utf-8",
    )

    # Second pass: same args, but now responses join in.
    second_pass = harness.replay_post_fork("rebuilt", case, run_dir=run_dir)
    by_idx = {r["turn_idx"]: r for r in second_pass}
    assert by_idx[2]["matched"] is True
    assert by_idx[2]["reason"] == ""
    assert by_idx[2]["actual"].startswith("Earlier we discussed")
    assert by_idx[4]["matched"] is False
    assert by_idx[4]["reason"] == harness.NEEDS_LLM_JUDGE_REASON
    assert by_idx[4]["actual"].startswith("no idea")

    # score_fidelity reports the substring-match-half verdict.
    assert harness.score_fidelity(second_pass) == 0.5


def test_replay_pending_partial_response_keeps_others_pending(
    harness: ModuleType, transcript_path: Path, tmp_path: Path
) -> None:
    """Acceptance: `missing rows stay pending_replay`. A response file
    that only covers some eval_turns must not promote the others."""
    case = harness.TranscriptCase(
        path=transcript_path, task_type="debug",
        fork_turn=2, eval_turns=(2, 4),
    )
    run_dir = tmp_path / "run"
    harness.replay_post_fork("rebuilt", case, run_dir=run_dir)
    (run_dir / harness.REPLAY_RESPONSES_FILENAME).write_text(
        json.dumps({
            "turn_idx": 2,
            # Expected at idx=2 is "What about session-scoped retrieval?";
            # actual contains that as a substring → matched=True.
            "actual": "Reply: what about session-scoped retrieval? — covered.",
        }) + "\n",
        encoding="utf-8",
    )
    rows = harness.replay_post_fork("rebuilt", case, run_dir=run_dir)
    by_idx = {r["turn_idx"]: r for r in rows}
    assert by_idx[2]["matched"] is True
    assert by_idx[4]["matched"] is False
    assert by_idx[4]["reason"] == harness.PENDING_REPLAY_REASON
    assert by_idx[4]["actual"] == ""


def test_run_one_threads_run_dir_to_replay_post_fork(
    harness: ModuleType, tmp_path: Path
) -> None:
    """`run_one(run_dir=...)` produces a per-(case, config) subdirectory
    under the base run_dir with a `replay_requests.jsonl` inside it."""
    corpus_dir = (
        _REPO_ROOT / "benchmarks" / "context-rebuilder" / "fixtures" / "synthetic"
    )
    cases = harness.load_corpus(corpus_dir)
    case = cases[0]
    base = tmp_path / "sweep_run"
    harness.run_one(
        case, trigger_threshold=0.5, token_budget=1000, run_dir=base,
    )
    expected_subdir = (
        base / f"{case.path.stem}__t0.5__b1000"
    )
    assert (expected_subdir / harness.REPLAY_REQUESTS_FILENAME).is_file()


# --------------------------------------------------------------------- #
# End-to-end: threshold-sweep against the bundled synthetic fixture     #
# --------------------------------------------------------------------- #


def test_threshold_sweep_runs_end_to_end_on_synthetic_fixture(
    harness: ModuleType, tmp_path: Path
) -> None:
    """Smoke test: load the bundled synthetic fixture, run a 3-threshold
    sweep, and verify the JSON output schema. Until the model client
    lands all fidelity scores will be 0.0 — the sweep still produces
    valid latency + token-cost numbers."""
    corpus_dir = (
        _REPO_ROOT / "benchmarks" / "context-rebuilder" / "fixtures" / "synthetic"
    )
    cases = harness.load_corpus(corpus_dir)
    assert len(cases) >= 1, "synthetic fixture meta.json must resolve"

    result = harness.sweep_thresholds(
        cases, thresholds=(0.0, 0.5), token_budget=2000,
    )
    assert result.mode == "threshold-sweep"
    assert len(result.runs) == len(cases) * 2
    for r in result.runs:
        assert r.fidelity == 0.0  # stub returns 0 until model client
        assert r.rebuild_latency_ms >= 0.0
        assert r.token_cost_ratio >= 0.0
        assert r.fork_turn > 0
    # summary structure: keyed by task_type, then per-threshold metrics
    assert "debug" in result.summary
    debug_summary = result.summary["debug"]
    assert "threshold=0.0" in debug_summary
    assert "median_fidelity" in debug_summary["threshold=0.0"]
    assert "p99_latency_ms" in debug_summary["threshold=0.0"]


def test_run_one_returns_runresult_without_crashing(
    harness: ModuleType, tmp_path: Path
) -> None:
    """`run_one` is the per-case entry point used by both sweep modes;
    it must thread store + rebuild + replay + score without raising
    even with the replay stub in place."""
    corpus_dir = (
        _REPO_ROOT / "benchmarks" / "context-rebuilder" / "fixtures" / "synthetic"
    )
    cases = harness.load_corpus(corpus_dir)
    case = cases[0]
    result = harness.run_one(case, trigger_threshold=0.0, token_budget=2000)
    assert isinstance(result, harness.RunResult)
    assert result.task_type == case.task_type
    assert result.config == {"trigger_threshold": 0.0, "token_budget": 2000}
    assert result.n_eval_turns == len(case.eval_turns)
    # All eval_turns are unmatched stubs, so failures == eval_turns
    assert len(result.failures) == result.n_eval_turns
    assert all(
        f["reason"] == harness.REPLAY_PENDING_REASON
        for f in result.failures
    )
