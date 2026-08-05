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
    """A genuine host-written hook-result record."""
    return json.dumps(
        {
            "type": "attachment",
            "timestamp": ts,
            "attachment": {
                "type": "hook_success",
                "content": "SessionStart:compact hook success",
            },
        }
    )


def _authored_marker(ts: str = "2026-07-01T00:00:01Z") -> str:
    """The same literal as ordinary message text, not a hook result.

    This is what a conversation *about* the marker writes into its own
    transcript.
    """
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


# Per-arm power is its own rung (#1360). Tests below that exercise a
# different rung supply counts that satisfy it, so a failure names the
# rung it is actually about.
_POWERED: dict[str, int] = {"manual": 40, "auto": 40}


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
    assert mod.verdict(
        rate, fired + unfired, None, _POWERED
    ).startswith(expected)


def test_underpowered_pass_is_not_a_pass() -> None:
    """100% on 19 boundaries must not read as CLEARS."""
    assert mod.verdict(1.0, 19, None, _POWERED).startswith("NO VERDICT")
    assert mod.verdict(1.0, 20, None, _POWERED) == "CLEARS"


# --- the divergence guard and the contamination guard ------------------


def test_divergence_over_the_limit_withdraws_the_pooled_verdict() -> None:
    """The rule that actually decided the real run.

    Its siblings (power floor, thresholds) are pinned above; this one
    was the only decision rule with no test, and it is the one that
    turned a 98.6% pooled rate into no verdict. Without the guard
    inside `verdict()` this returns CLEARS.
    """
    v = mod.verdict(0.986, 73, {"manual": 1.0, "auto": 0.0}, _POWERED)
    assert v.startswith("NO VERDICT")
    assert "diverge" in v


def test_divergence_within_the_limit_still_clears() -> None:
    """Negative control: the guard must not swallow every verdict."""
    assert (
        mod.verdict(0.99, 73, {"manual": 1.0, "auto": 0.95}, _POWERED)
        == "CLEARS"
    )


def test_a_single_trigger_cannot_diverge() -> None:
    """One trigger is not agreement between two.

    The spread is genuinely undefined here, so `trigger_divergence`
    abstains -- that part was always right. What was wrong (#1360) is
    what happened next: abstention let the other rungs decide, and the
    other rungs had nothing to say about an arm that was never observed,
    so a single-trigger corpus reached CLEARS. The per-arm power rung is
    what catches it; divergence is not, and should not be, that guard.
    """
    assert mod.trigger_divergence({"manual": 1.0}) is None
    assert mod.trigger_divergence(None) is None
    v = mod.verdict(0.99, 73, {"manual": 1.0}, {"manual": 73})
    assert v.startswith("NO VERDICT")
    assert "auto: n=0" in v


# --- #1360: an unobserved arm must score, not vanish -------------------


def test_an_unobserved_arm_does_not_clear() -> None:
    """The real-corpus failure, reduced.

    63 manual boundaries at 100%, zero auto: the pooled floor is
    satisfied by the one arm carrying the whole corpus, and divergence
    has no second arm to measure against. Before the per-arm rung this
    printed `VERDICT: CLEARS` on a rule whose whole purpose is the auto
    arm.
    """
    v = mod.verdict(1.0, 63, {"manual": 1.0}, {"manual": 63})
    assert v.startswith("NO VERDICT")
    assert "auto: n=0" in v


def test_the_rule_is_monotonic_in_evidence() -> None:
    """Observing more must never make the verdict strictly worse.

    This is the sharp form of #1360. With zero auto boundaries the old
    rule said CLEARS; adding a single failing auto boundary flipped it to
    NO VERDICT. A decision rule that pays you to stop looking is not a
    decision rule, so both must now be NO VERDICT -- and for the same
    reason, that the auto arm is too thin to score either way.
    """
    without = mod.verdict(1.0, 63, {"manual": 1.0}, {"manual": 63})
    with_one = mod.verdict(
        63 / 64, 64, {"manual": 1.0, "auto": 0.0}, {"manual": 63, "auto": 1}
    )
    assert without.startswith("NO VERDICT")
    assert with_one.startswith("NO VERDICT")
    # The precise regression: these must not straddle the CLEARS boundary.
    assert ("CLEARS" == without) is ("CLEARS" == with_one)


def test_a_powered_auto_arm_can_still_clear() -> None:
    """Negative control: the rung must not swallow every verdict.

    Without this the fix is indistinguishable from hard-wiring NO
    VERDICT, and the check would be useless in the state it exists to
    report on.
    """
    assert (
        mod.verdict(
            0.99, 80, {"manual": 1.0, "auto": 0.98}, {"manual": 40, "auto": 40}
        )
        == "CLEARS"
    )


def test_unknown_arm_power_is_not_treated_as_fine() -> None:
    """`None` counts means the caller could not say, not that it is ok.

    Defaulting the unknown case to "powered" is the same shape as the
    bug: an absence that reads as an affirmation.
    """
    assert mod.underpowered_arms(None) == list(mod.EXPECTED_TRIGGERS)
    assert mod.verdict(1.0, 100, {"manual": 1.0}).startswith("NO VERDICT")


def test_divergence_is_checked_before_the_clears_rung() -> None:
    """Order matters: a diverging run must never print CLEARS first.

    A rate comfortably over the bar plus a divergence must resolve to
    NO VERDICT, not to CLEARS with a footnote.
    """
    assert mod.verdict(
        1.0, 100, {"manual": 1.0, "auto": 0.0}, _POWERED
    ).startswith("NO VERDICT")


def test_underpowered_still_wins_over_divergence() -> None:
    """Too little data is the more fundamental objection of the two."""
    assert mod.verdict(1.0, 5, {"manual": 1.0, "auto": 0.0}, _POWERED) == (
        "NO VERDICT (underpowered)"
    )


def test_authored_marker_text_does_not_count_as_a_firing(
    tmp_path: Path,
) -> None:
    """Writing about the marker must not manufacture one.

    A substring match on the raw line counted a conversation discussing
    `SessionStart:compact` as a fire. Measured on the development
    corpus: 72 genuine `hook_success` attachments against 15 authored
    mentions. Here the authored mention follows a real boundary, which
    is the case that would have inflated the numerator.
    """
    tally = _scan(tmp_path, [_boundary(), _authored_marker(), _noise()])
    assert tally.markers_seen == 0
    assert tally.boundaries[0].fired is False


def test_a_genuine_marker_is_still_counted(tmp_path: Path) -> None:
    """Negative control for the tightening: the real record still fires."""
    tally = _scan(tmp_path, [_boundary(), _marker(), _noise()])
    assert tally.markers_seen == 1
    assert tally.boundaries[0].fired is True


def test_a_hook_success_for_another_event_does_not_count(
    tmp_path: Path,
) -> None:
    """The attachment kind is necessary but not sufficient.

    `SessionStart:clear` is a hook result too, and it is not a
    compaction.
    """
    other = json.dumps(
        {
            "type": "attachment",
            "timestamp": "2026-07-01T00:00:01Z",
            "attachment": {
                "type": "hook_success",
                "content": "SessionStart:clear hook success",
            },
        }
    )
    tally = _scan(tmp_path, [_boundary(), other, _noise()])
    assert tally.markers_seen == 0
    assert tally.boundaries[0].fired is False
