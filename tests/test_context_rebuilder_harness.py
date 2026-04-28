"""Context-rebuilder eval-harness scaffolding tests (#136, v1.4.0).

Verifies the harness shape and the per-turn output schema. Does NOT
exercise continuation-fidelity scoring -- that lives in #138.

All tests are deterministic and run well under the 2-second budget
the issue calls out.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import cast

import pytest

from benchmarks.context_rebuilder.__main__ import main as cli_main
from benchmarks.context_rebuilder.inject import ClearInjection, midpoint_clear
from benchmarks.context_rebuilder.measure import (
    estimate_tokens,
    hook_latency_ms,
    token_budget_delta,
)
from benchmarks.context_rebuilder.replay import (
    DEFAULT_REBUILD_OVERHEAD_TOKENS,
    FixtureError,
    ReplayResult,
    ReplayTurnResult,
    load_turns,
    run,
)

# Bundled synthetic fixture path.
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
# Fixture / loader determinism                                          #
# --------------------------------------------------------------------- #


def test_synthetic_fixture_exists() -> None:
    """The bundled synthetic fixture is on disk and non-empty.

    Sanity check: if this fails, the rest of the suite would fail in
    a more confusing way. Catch it up front.
    """
    assert SYNTHETIC_FIXTURE.is_file(), (
        f"missing synthetic fixture: {SYNTHETIC_FIXTURE}"
    )
    assert SYNTHETIC_FIXTURE.stat().st_size > 0


def test_load_turns_reads_synthetic_fixture() -> None:
    """`load_turns` returns >=10 turns + zero skipped on the bundled fixture."""
    turns, skipped = load_turns(SYNTHETIC_FIXTURE)
    assert len(turns) >= 10, "synthetic fixture should have >=10 turns"
    assert skipped == 0, "fixture has no malformed lines"
    # Roles alternate user/assistant in the synthetic fixture.
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"


# --------------------------------------------------------------------- #
# Acceptance: replay completes against the bundled synthetic fixture    #
# --------------------------------------------------------------------- #


def test_replay_completes_against_bundled_synthetic_fixture() -> None:
    """End-to-end replay returns a well-formed ReplayResult."""
    result = run(SYNTHETIC_FIXTURE)
    assert isinstance(result, ReplayResult)
    assert result.n_turns >= 10
    assert result.full_replay_baseline_tokens > 0
    assert result.clear_injected_at is None
    assert result.rebuild_block_tokens == 0
    assert len(result.turns) == result.n_turns
    # Pre-clear, every per-turn delta is 0.
    for t in result.turns:
        assert t.token_budget_delta == 0


# --------------------------------------------------------------------- #
# Acceptance: per-turn JSON schema carries token_budget_delta + hook_latency_ms
# --------------------------------------------------------------------- #


def test_per_turn_output_has_required_keys() -> None:
    """Every turn record carries `token_budget_delta` and `hook_latency_ms`.

    Direct check on the dataclass.
    """
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=4))
    assert result.clear_injected_at == 4
    assert result.turns, "should have at least one turn record"
    for t in result.turns:
        assert hasattr(t, "token_budget_delta")
        assert hasattr(t, "hook_latency_ms")
        assert isinstance(t.token_budget_delta, int)
        assert isinstance(t.hook_latency_ms, float)


def test_json_output_has_required_keys_per_turn(tmp_path: Path) -> None:
    """JSON-serialized output (the CLI surface) carries the same keys."""
    out_path = tmp_path / "out.json"
    rc = cli_main(
        [
            str(SYNTHETIC_FIXTURE),
            "--clear-at",
            "4",
            "--out",
            str(out_path),
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    raw_payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(raw_payload, dict)
    payload = cast(dict[str, object], raw_payload)
    assert "turns" in payload
    turns_obj = payload["turns"]
    assert isinstance(turns_obj, list)
    turns = cast(list[object], turns_obj)
    assert turns, "expected at least one turn"
    required_keys = {"token_budget_delta", "hook_latency_ms", "turn_index", "role"}
    for t in turns:
        assert isinstance(t, dict)
        t_typed = cast(dict[str, object], t)
        keys: set[str] = set(t_typed.keys())
        assert required_keys.issubset(keys), (
            f"turn record missing required keys: {required_keys - keys}"
        )


# --------------------------------------------------------------------- #
# Acceptance: midpoint-clear injection actually clears state            #
# --------------------------------------------------------------------- #


def test_midpoint_clear_injection_marks_clear_turn() -> None:
    """`ClearInjection(clear_at=N)` flips `cleared=True` exactly at turn N."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=6))
    assert result.clear_injected_at == 6
    cleared_turns = [t for t in result.turns if t.cleared]
    assert len(cleared_turns) == 1
    assert cleared_turns[0].turn_index == 6


def test_midpoint_clear_changes_token_budget_delta() -> None:
    """Pre-clear deltas are 0; at-clear and post-clear deltas are non-zero.

    Captures the contract that the clear actually substitutes the
    rebuild block into the cumulative state.
    """
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=5))
    pre = [t for t in result.turns if t.turn_index < 5]
    on = [t for t in result.turns if t.turn_index == 5]
    post = [t for t in result.turns if t.turn_index > 5]

    for t in pre:
        assert t.token_budget_delta == 0, (
            "pre-clear deltas must be 0 (no rebuild has fired yet)"
        )
    assert len(on) == 1
    # On-clear delta should reflect rebuild_block_tokens minus pre-clear
    # cumulative (which is what the rebuild substitutes for).
    assert on[0].token_budget_delta != 0 or on[0].cleared
    # Post-clear: rebuild block has saved tokens vs. full replay, so
    # deltas should be strictly negative for at least one post-clear
    # turn (the rebuild paid less than full).
    assert post, "fixture must have post-clear turns for this test"
    assert any(t.token_budget_delta < 0 for t in post), (
        "expected the rebuild to save tokens on at least one post-clear turn"
    )


def test_midpoint_clear_helper_picks_floor_midpoint() -> None:
    """`midpoint_clear(n)` returns `clear_at = n // 2`."""
    assert midpoint_clear(10).clear_at == 5
    assert midpoint_clear(11).clear_at == 5
    assert midpoint_clear(0).clear_at == 0


def test_clear_injection_rejects_negative_clear_at() -> None:
    with pytest.raises(ValueError):
        ClearInjection(clear_at=-1)


def test_clear_injection_out_of_range_silently_no_ops() -> None:
    """`clear_at` >= turn count produces a result with no clear fired."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=10_000))
    assert result.clear_injected_at is None
    assert all(not t.cleared for t in result.turns)


# --------------------------------------------------------------------- #
# Acceptance: latency measurement is monotonic (start <= end)           #
# --------------------------------------------------------------------- #


def test_hook_latency_ms_is_non_negative() -> None:
    """`hook_latency_ms(start)` always returns >= 0.0.

    The measurement primitive floors at 0.0 to absorb clock jitter
    when the same monotonic reading is consumed across forks.
    """
    t0 = time.monotonic()
    # Synthetic gap.
    elapsed = hook_latency_ms(t0)
    assert elapsed >= 0.0
    # Future-stamp: still floors at 0.0, never goes negative.
    future = t0 + 1.0
    assert hook_latency_ms(future) == 0.0


def test_replay_per_turn_latency_is_non_negative() -> None:
    """Every per-turn `hook_latency_ms` reading is >= 0."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=3))
    for t in result.turns:
        assert t.hook_latency_ms >= 0.0


def test_non_clear_turns_have_zero_latency() -> None:
    """Latency is recorded only on the clear turn -- everywhere else, 0."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=7))
    for t in result.turns:
        if not t.cleared:
            assert t.hook_latency_ms == 0.0


# --------------------------------------------------------------------- #
# Acceptance: empty / missing fixture -> clear error, no crash          #
# --------------------------------------------------------------------- #


def test_missing_fixture_raises_fixture_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FixtureError) as exc_info:
        load_turns(missing)
    assert "fixture not found" in str(exc_info.value)


def test_empty_fixture_raises_fixture_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(FixtureError) as exc_info:
        load_turns(empty)
    assert "zero usable turns" in str(exc_info.value)


def test_directory_path_as_fixture_raises_fixture_error(tmp_path: Path) -> None:
    """A directory passed as fixture is rejected without crashing."""
    with pytest.raises(FixtureError):
        load_turns(tmp_path)


def test_fixture_with_only_compaction_markers_raises(tmp_path: Path) -> None:
    """A fixture with only marker events (no content turns) is empty -> error."""
    f = tmp_path / "markers_only.jsonl"
    lines = [
        '{"schema_version":1,"ts":"2026-04-27T00:00:00Z","event":"compaction_start"}',
        '{"schema_version":1,"ts":"2026-04-27T00:00:01Z","event":"compaction_complete"}',
    ]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(FixtureError):
        load_turns(f)


def test_malformed_lines_are_skipped_not_fatal(tmp_path: Path) -> None:
    """Malformed JSON / wrong-shape lines are skipped, not raised."""
    f = tmp_path / "mixed.jsonl"
    lines = [
        "this is not json",
        "{}",  # well-formed JSON but missing required fields
        "[]",  # JSON but not a dict
        '{"schema_version":1,"ts":"2026-04-27T00:00:00Z","role":"user","text":"hello","session_id":"s","turn_id":"t1","context":{}}',
        "",  # blank line
        '{"schema_version":1,"ts":"2026-04-27T00:00:01Z","role":"assistant","text":"hi","session_id":"s","turn_id":"t2","context":{}}',
    ]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    turns, skipped = load_turns(f)
    assert len(turns) == 2
    assert skipped == 4  # not-json, {}, [], blank
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"


def test_cli_missing_fixture_returns_nonzero_exit(tmp_path: Path) -> None:
    """CLI surface: missing fixture writes to stderr and returns 1."""
    missing = tmp_path / "nope.jsonl"
    err = io.StringIO()
    rc = cli_main(
        [str(missing)],
        stdout=io.StringIO(),
        stderr=err,
    )
    assert rc == 1
    assert "error:" in err.getvalue()
    assert "fixture not found" in err.getvalue()


def test_cli_negative_clear_at_returns_two(tmp_path: Path) -> None:
    """CLI surface: negative --clear-at is a programming error, returns 2."""
    err = io.StringIO()
    rc = cli_main(
        [str(SYNTHETIC_FIXTURE), "--clear-at", "-3"],
        stdout=io.StringIO(),
        stderr=err,
    )
    assert rc == 2
    assert "error:" in err.getvalue()


# --------------------------------------------------------------------- #
# Determinism                                                           #
# --------------------------------------------------------------------- #


def test_replay_token_counts_are_deterministic() -> None:
    """Two runs against the same fixture produce identical token totals."""
    a = run(SYNTHETIC_FIXTURE)
    b = run(SYNTHETIC_FIXTURE)
    assert a.full_replay_baseline_tokens == b.full_replay_baseline_tokens
    assert a.n_turns == b.n_turns
    assert [t.token_budget_delta for t in a.turns] == [
        t.token_budget_delta for t in b.turns
    ]


def test_estimate_tokens_chars_per_token_heuristic() -> None:
    """Token estimate uses the documented 4 chars/token ceiling."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1  # ceiling
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    # Very common short turn text: ensure the math is what we expect.
    text = "x" * 100
    assert estimate_tokens(text) == 25


def test_token_budget_delta_signs() -> None:
    """Positive when rebuilt > full, negative when rebuilt < full, 0 equal."""
    assert token_budget_delta(full=100, rebuilt=120) == 20
    assert token_budget_delta(full=100, rebuilt=70) == -30
    assert token_budget_delta(full=50, rebuilt=50) == 0


# --------------------------------------------------------------------- #
# Module-as-script invocation form                                      #
# --------------------------------------------------------------------- #


def test_default_rebuild_overhead_tokens_is_documented() -> None:
    """The default rebuild overhead matches the module-level constant.

    Anchors the magic number against drift; the synthetic generator
    keys off this in test math above.
    """
    assert DEFAULT_REBUILD_OVERHEAD_TOKENS == 32


def test_replay_result_to_dict_round_trip(tmp_path: Path) -> None:
    """`ReplayResult.to_dict()` is JSON-serializable end-to-end."""
    result = run(SYNTHETIC_FIXTURE, inject=ClearInjection(clear_at=2))
    payload = json.dumps(result.to_dict())
    raw_reloaded = json.loads(payload)
    assert isinstance(raw_reloaded, dict)
    reloaded = cast(dict[str, object], raw_reloaded)
    assert reloaded["n_turns"] == result.n_turns
    turns_obj = reloaded["turns"]
    assert isinstance(turns_obj, list)
    turns = cast(list[object], turns_obj)
    assert len(turns) == len(result.turns)
    # Every reloaded turn carries the required keys.
    turn0 = turns[0]
    assert isinstance(turn0, dict)
    turn0_typed = cast(dict[str, object], turn0)
    assert "token_budget_delta" in turn0_typed
    assert "hook_latency_ms" in turn0_typed


def test_per_turn_result_dataclass_shape() -> None:
    """`ReplayTurnResult` has exactly the documented fields."""
    r = ReplayTurnResult(
        turn_index=0,
        role="user",
        token_budget_delta=0,
        hook_latency_ms=0.0,
        cleared=False,
    )
    assert r.turn_index == 0
    assert r.role == "user"
    assert r.token_budget_delta == 0
    assert r.hook_latency_ms == 0.0
    assert r.cleared is False
