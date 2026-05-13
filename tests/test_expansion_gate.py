"""Unit + integration tests for the adaptive expansion-gate (#741).

Covers four layers:

1. Resolver precedence — env-force > env-no-gate > TOML > heuristics.
2. Heuristic gates — length, structural markers, question-form prefix.
3. Integration with :func:`aelfrice.retrieval.retrieve` — broad queries
   short-circuit the BFS lane; narrow queries (with markers) keep BFS.
4. Telemetry — :class:`LaneTelemetry` exposes the gate reason and
   skipped-BFS flag for downstream surfaces (``aelf doctor``, ``aelf tail``).

PHILOSOPHY (#605) check: every heuristic in
``src/aelfrice/expansion_gate.py`` is deterministic stdlib-only — no
embeddings, no model calls. The tests below pin the verdict for each
distinct surface so any drift toward non-deterministic gating fails CI.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from aelfrice.expansion_gate import (
    BROAD_PROMPT_TOKEN_THRESHOLD,
    ENV_FORCE_EXPANSION,
    ENV_NO_EXPANSION_GATE,
    ExpansionDecision,
    should_run_expansion,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import (
    LaneTelemetry,
    last_lane_telemetry,
    retrieve,
)
from aelfrice.store import MemoryStore


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_gate_env() -> Iterator[None]:
    """Ensure neither env override is set during a test unless the
    test sets it explicitly.
    """
    saved = {
        k: os.environ.pop(k, None)
        for k in (ENV_FORCE_EXPANSION, ENV_NO_EXPANSION_GATE)
    }
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-05-13T00:00:00Z",
        last_retrieved_at=None,
    )


# --- Heuristic gates ------------------------------------------------------


def test_narrow_query_with_issue_ref_runs_expansion() -> None:
    d = should_run_expansion("can we fix #741")
    assert d.run_bfs is True
    assert d.reason == "narrow"


def test_narrow_query_with_file_path_runs_expansion() -> None:
    d = should_run_expansion("update src/aelfrice/retrieval.py")
    assert d.run_bfs is True


def test_narrow_query_with_snake_case_identifier_runs_expansion() -> None:
    d = should_run_expansion("look at should_run_expansion")
    assert d.run_bfs is True


def test_narrow_query_with_camel_case_identifier_runs_expansion() -> None:
    d = should_run_expansion("debug retrieveWithTiers behavior")
    assert d.run_bfs is True


def test_narrow_query_with_edge_type_runs_expansion() -> None:
    # No question-form prefix — only the edge-type marker. The
    # structural-marker gate satisfies "has markers" so all three
    # broad signals are silent.
    d = should_run_expansion("trace SUPPORTS chain from belief root")
    assert d.run_bfs is True


def test_question_form_prefix_skips_expansion() -> None:
    d = should_run_expansion("what is the meaning of life")
    assert d.run_bfs is False
    assert "question-form" in d.reason


def test_no_markers_skips_expansion() -> None:
    d = should_run_expansion("hello there")
    assert d.run_bfs is False
    assert "no-markers" in d.reason


def test_long_prompt_without_markers_skips_expansion() -> None:
    body = " ".join(["word"] * (BROAD_PROMPT_TOKEN_THRESHOLD + 5))
    d = should_run_expansion(body)
    assert d.run_bfs is False
    assert "long" in d.reason


def test_question_word_inside_word_does_not_trip_gate() -> None:
    # `whatever` must not match the `what` prefix gate. The query has
    # no markers though, so it still skips on "no-markers" alone — we
    # only check the question-form sub-reason here.
    d = should_run_expansion("whatever happens next")
    assert "question-form" not in d.reason


def test_empty_query_returns_narrow_passthrough() -> None:
    d = should_run_expansion("")
    assert d == ExpansionDecision(
        run_bfs=True,
        run_hrr_structural=True,
        reason="empty-query",
    )


def test_whitespace_query_returns_narrow_passthrough() -> None:
    d = should_run_expansion("   \t  \n  ")
    assert d.run_bfs is True
    assert d.reason == "empty-query"


def test_decision_dataclass_is_frozen() -> None:
    d = should_run_expansion("what is X")
    with pytest.raises(Exception):
        d.run_bfs = True  # type: ignore[misc]


# --- Resolver precedence --------------------------------------------------


def test_env_force_expansion_overrides_broad_signals() -> None:
    os.environ[ENV_FORCE_EXPANSION] = "1"
    d = should_run_expansion("what is happiness")
    assert d.run_bfs is True
    assert d.reason == "env-force-expansion"


def test_env_no_gate_disables_gate_entirely() -> None:
    os.environ[ENV_NO_EXPANSION_GATE] = "1"
    d = should_run_expansion("how does this work")
    assert d.run_bfs is True
    assert d.reason == "env-no-gate"


def test_env_force_beats_env_no_gate() -> None:
    # Both set → force wins (precedence #1).
    os.environ[ENV_FORCE_EXPANSION] = "1"
    os.environ[ENV_NO_EXPANSION_GATE] = "1"
    d = should_run_expansion("what is X")
    assert d.reason == "env-force-expansion"


def test_env_force_falsy_falls_through_to_heuristics() -> None:
    os.environ[ENV_FORCE_EXPANSION] = "0"
    d = should_run_expansion("how does this work")
    assert d.run_bfs is False  # gate still fires


def test_env_garbage_value_falls_through() -> None:
    os.environ[ENV_FORCE_EXPANSION] = "maybe"
    d = should_run_expansion("how does this work")
    assert d.run_bfs is False


def test_toml_disable_overrides_heuristics(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\nexpansion_gate_enabled = false\n"
    )
    d = should_run_expansion("what is X", start=tmp_path)
    assert d.run_bfs is True
    assert d.reason == "toml-disabled"


def test_toml_enable_explicit_runs_heuristics(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\nexpansion_gate_enabled = true\n"
    )
    d = should_run_expansion("what is X", start=tmp_path)
    # TOML=True is the default; heuristics still fire.
    assert d.run_bfs is False
    assert "question-form" in d.reason


def test_toml_absent_runs_heuristics(tmp_path: Path) -> None:
    # No TOML file → fall through to heuristics.
    d = should_run_expansion("how does this work", start=tmp_path)
    assert d.run_bfs is False


def test_malformed_toml_fails_soft(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text("not = valid = toml = at all\n")
    d = should_run_expansion("what is X", start=tmp_path)
    # Malformed TOML must not crash; gate still runs heuristics.
    assert d.run_bfs is False


# --- Integration with retrieve() -----------------------------------------


def test_retrieve_broad_query_skips_bfs_even_when_enabled() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("L1", "user-pinned cats fact",
                        lock_level=LOCK_USER, locked_at="2026-05-13T00:00:00Z"))
    s.insert_belief(_mk("F1", "rainbow elephants graze quietly"))

    retrieve(s, query="what is the answer", bfs_enabled=True)
    tel = last_lane_telemetry()
    assert tel.expansion_gate_skipped_bfs is True
    assert "broad" in tel.expansion_gate_reason


def test_retrieve_narrow_query_keeps_bfs_when_enabled() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "RELATES_TO marker stays narrow"))

    retrieve(s, query="show beliefs that SUPPORTS this thread", bfs_enabled=True)
    tel = last_lane_telemetry()
    assert tel.expansion_gate_skipped_bfs is False
    assert tel.expansion_gate_reason == "narrow"


def test_retrieve_gate_noop_when_bfs_disabled() -> None:
    # bfs_enabled=False explicitly → gate's verdict is irrelevant for
    # the BFS lane (nothing to suppress). Telemetry should report
    # skipped_bfs=False (no actual suppression happened).
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "some content"))

    retrieve(s, query="what is X", bfs_enabled=False)
    tel = last_lane_telemetry()
    assert tel.expansion_gate_skipped_bfs is False


def test_retrieve_force_env_keeps_bfs_on_broad_query() -> None:
    os.environ[ENV_FORCE_EXPANSION] = "1"
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "some content"))

    retrieve(s, query="what is X", bfs_enabled=True)
    tel = last_lane_telemetry()
    assert tel.expansion_gate_skipped_bfs is False
    assert tel.expansion_gate_reason == "env-force-expansion"


def test_retrieve_empty_query_does_not_set_skipped_flag() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("L1", "locked",
                        lock_level=LOCK_USER, locked_at="2026-05-13T00:00:00Z"))

    retrieve(s, query="", bfs_enabled=True)
    tel = last_lane_telemetry()
    # Empty query short-circuits before BFS would run anyway; gate
    # reports "empty-query" and no suppression.
    assert tel.expansion_gate_skipped_bfs is False
    assert tel.expansion_gate_reason == "empty-query"


# --- LaneTelemetry contract ----------------------------------------------


def test_lane_telemetry_defaults_preserve_backcompat() -> None:
    # A LaneTelemetry built with no #741 kwargs must still construct
    # cleanly. This guards callers that pin against the pre-#741
    # field set.
    tel = LaneTelemetry()
    assert tel.expansion_gate_reason == ""
    assert tel.expansion_gate_skipped_bfs is False
