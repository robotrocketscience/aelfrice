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
