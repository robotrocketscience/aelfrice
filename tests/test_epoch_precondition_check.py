"""Pairing-rule tests for scripts/epoch_precondition_check.py (#1252).

The verdict of the epoch precondition check rests entirely on how
boundaries are paired with fire markers. A pairing bug does not crash —
it silently reports a rate, which is the failure mode the check exists
to prevent. Each test below distinguishes the specified rule from the
plausible wrong ones: pairing by aggregate count, letting one marker
satisfy several boundaries, or scoring the trailing boundary.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "epoch_precondition_check.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_epoch_check", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_epoch_check"] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _boundary(trigger: str = "manual", ts: str = "2026-07-01T00:00:00Z") -> str:
    return json.dumps(
        {
            "type": "system",
            "subtype": "compact_boundary",
            "timestamp": ts,
            "compactMetadata": {"trigger": trigger},
        }
    )


def _marker(ts: str = "2026-07-01T00:00:01Z") -> str:
    return json.dumps(
        {"type": "user", "timestamp": ts, "content": "SessionStart:compact"}
    )


def _noise(ts: str = "2026-07-01T00:00:02Z") -> str:
    return json.dumps({"type": "assistant", "timestamp": ts, "content": "hi"})


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _scan(tmp_path: Path, lines: list[str], since: str | None = None):
    tally = mod.Tally()
    mod.scan_file(_write(tmp_path, "s.jsonl", lines), since, tally)
    return tally


def test_boundary_followed_by_marker_is_fired(tmp_path: Path) -> None:
    tally = _scan(tmp_path, [_boundary(), _marker(), _boundary(), _marker()])
    scoreable = [b for b in tally.boundaries if not b.trailing]
    assert len(tally.boundaries) == 2
    assert all(b.fired for b in tally.boundaries)
    # Second boundary is trailing only if nothing followed it; a marker
    # did, so both are scoreable.
    assert len(scoreable) == 2


def test_marker_before_boundary_does_not_count(tmp_path: Path) -> None:
    """Order matters. A marker that precedes the reset cannot be the
    fire *for* that reset; counting it would make any file with equal
    totals look perfect."""
    tally = _scan(tmp_path, [_marker(), _boundary()])
    assert tally.markers_unpaired == 1
    assert tally.boundaries[0].fired is False


def test_one_marker_cannot_satisfy_two_boundaries(tmp_path: Path) -> None:
    """The distinguishing case against count-based pairing: two
    boundaries and one marker. A totals comparison would report 50%
    either way, but the rule must attribute the fire to the *older*
    boundary and leave the newer one unfired."""
    tally = _scan(tmp_path, [_boundary(), _boundary(), _marker()])
    first, second = tally.boundaries
    assert first.fired is True
    assert second.fired is False


def test_last_unfired_boundary_is_trailing_not_a_failure(tmp_path: Path) -> None:
    """A session ending after a reset is truncation, not unreliability."""
    tally = _scan(tmp_path, [_boundary(), _marker(), _boundary(), _noise()])
    assert tally.boundaries[0].fired is True
    assert tally.boundaries[1].trailing is True
    scoreable = [b for b in tally.boundaries if not b.trailing]
    assert len(scoreable) == 1


def test_only_the_last_pending_boundary_is_trailing(tmp_path: Path) -> None:
    """An earlier unfired boundary had a whole subsequent stretch of
    session in which to fire and did not — it stays a failure. Marking
    every pending boundary trailing would erase real misses."""
    tally = _scan(tmp_path, [_boundary(), _boundary(), _noise()])
    first, second = tally.boundaries
    assert first.trailing is False and first.fired is False
    assert second.trailing is True


def test_trigger_is_read_from_compact_metadata(tmp_path: Path) -> None:
    tally = _scan(tmp_path, [_boundary(trigger="auto"), _marker()])
    assert tally.boundaries[0].trigger == "auto"


def test_missing_metadata_is_unknown_not_a_crash(tmp_path: Path) -> None:
    line = json.dumps({"subtype": "compact_boundary", "timestamp": "t"})
    tally = _scan(tmp_path, [line])
    assert tally.boundaries[0].trigger == "unknown"


def test_since_window_excludes_older_records(tmp_path: Path) -> None:
    tally = _scan(
        tmp_path,
        [_boundary(ts="2026-06-01T00:00:00Z"), _boundary(ts="2026-08-01T00:00:00Z")],
        since="2026-07-01T00:00:00Z",
    )
    assert len(tally.boundaries) == 1


def test_file_without_boundary_counts_as_session_without(tmp_path: Path) -> None:
    """The never-compacted population is the headline number in the
    report, so it must not be silently folded into the denominator."""
    tally = _scan(tmp_path, [_noise(), _noise()])
    assert tally.sessions_without_boundary == 1
    assert tally.sessions_with_boundary == 0


def test_malformed_json_is_skipped_not_fatal(tmp_path: Path) -> None:
    tally = _scan(tmp_path, ["{not json", _boundary(), _marker()])
    assert len(tally.boundaries) == 1
    assert tally.boundaries[0].fired is True


@pytest.mark.parametrize(
    ("fired", "unfired", "expected"),
    [
        (98, 2, "CLEARS"),
        (89, 11, "KILLS"),
        (95, 5, "GREY"),
        (19, 0, "NO VERDICT"),
    ],
)
def test_verdict_thresholds(fired: int, unfired: int, expected: str) -> None:
    """Pins the pre-registered rule so a later edit to the thresholds
    is a visible test change rather than a quiet re-scoring."""
    rate = mod._rate(fired, unfired)
    assert mod.verdict(rate, fired + unfired).startswith(expected)


def test_underpowered_pass_is_not_a_pass() -> None:
    """100% on 19 boundaries must not read as CLEARS."""
    assert mod.verdict(1.0, 19).startswith("NO VERDICT")
    assert mod.verdict(1.0, 20) == "CLEARS"
