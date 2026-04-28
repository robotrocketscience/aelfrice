"""Tests for the v1.5.0 #154 composition-tracker surface.

Covers:

- Placeholder flag warner emits one stderr line per recognised
  flag (`use_signed_laplacian`, `use_heat_kernel`,
  `use_posterior_ranking`, `use_hrr_structural`) and is silent
  when none are set.
- Warner is idempotent within a process — second call produces
  no new warnings unless the seen-set is reset.
- LaneTelemetry surface populates from `retrieve()` and reflects
  the BM25F lane swap when `use_bm25f_anchors=True`.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import (
    PLACEHOLDER_FLAGS,
    LaneTelemetry,
    _reset_placeholder_warnings,
    last_lane_telemetry,
    retrieve,
    warn_placeholder_flags,
)
from aelfrice.store import MemoryStore


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def test_placeholder_flag_set_emits_warning(tmp_path: Path) -> None:
    """`use_signed_laplacian = true` produces a stderr warning that
    names the flag and references #154."""
    _reset_placeholder_warnings()
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nuse_signed_laplacian = true\n")
    buf = StringIO()
    with patch("sys.stderr", buf):
        warned = warn_placeholder_flags(start=tmp_path)
    assert warned == ["use_signed_laplacian"]
    out = buf.getvalue()
    assert "use_signed_laplacian" in out
    assert "#154" in out


def test_placeholder_flag_warner_idempotent_within_process(
    tmp_path: Path,
) -> None:
    """Second call to `warn_placeholder_flags` is silent for an
    already-warned flag."""
    _reset_placeholder_warnings()
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nuse_heat_kernel = true\n")
    buf1 = StringIO()
    with patch("sys.stderr", buf1):
        warn_placeholder_flags(start=tmp_path)
    buf2 = StringIO()
    with patch("sys.stderr", buf2):
        warn_placeholder_flags(start=tmp_path)
    assert buf2.getvalue() == ""


def test_placeholder_flag_no_warning_when_unset(tmp_path: Path) -> None:
    """Empty TOML produces no warnings."""
    _reset_placeholder_warnings()
    (tmp_path / ".aelfrice.toml").write_text("[retrieval]\n")
    buf = StringIO()
    with patch("sys.stderr", buf):
        warned = warn_placeholder_flags(start=tmp_path)
    assert warned == []
    assert buf.getvalue() == ""


def test_placeholder_flag_set_to_false_no_warning(tmp_path: Path) -> None:
    """`use_X = false` does not trigger a warning. Only True does."""
    _reset_placeholder_warnings()
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\nuse_signed_laplacian = false\n"
        "use_heat_kernel = false\nuse_posterior_ranking = false\n"
        "use_hrr_structural = false\n",
    )
    buf = StringIO()
    with patch("sys.stderr", buf):
        warned = warn_placeholder_flags(start=tmp_path)
    assert warned == []


def test_placeholder_flags_constant_lists_unwired_lanes() -> None:
    """`PLACEHOLDER_FLAGS` shrinks as each placeholder lane wires up.

    `use_heat_kernel` (#150) and `use_hrr_structural` (#152) ship
    the wiring, so each flag exits the placeholder set when its
    lane lands. Updating this test in lockstep with each move is
    the intentional drift guard.
    """
    assert set(PLACEHOLDER_FLAGS) == {
        "use_signed_laplacian",
        "use_posterior_ranking",
    }


def test_lane_telemetry_records_fts5_lane_by_default() -> None:
    """A retrieve() call with no opt-in flag populates LaneTelemetry
    with bm25f_used=False."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "alpha beta"))
    s.insert_belief(_mk("b2", "gamma delta"))
    retrieve(s, "alpha")
    t = last_lane_telemetry()
    assert isinstance(t, LaneTelemetry)
    assert t.bm25f_used is False
    # b1 matches "alpha"; b2 does not.
    assert t.l1 == 1
    assert t.locked == 0


def test_lane_telemetry_records_bm25f_lane_when_opted_in() -> None:
    """Setting use_bm25f_anchors=True flips the bm25f_used flag."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "alpha beta"))
    retrieve(s, "alpha", use_bm25f_anchors=True)
    t = last_lane_telemetry()
    assert t.bm25f_used is True


def test_lane_telemetry_posterior_weight_passthrough() -> None:
    """LaneTelemetry.posterior_weight reflects the resolved weight."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "alpha"))
    retrieve(s, "alpha", posterior_weight=0.0)
    assert last_lane_telemetry().posterior_weight == 0.0
    retrieve(s, "alpha", posterior_weight=0.7)
    assert last_lane_telemetry().posterior_weight == 0.7
