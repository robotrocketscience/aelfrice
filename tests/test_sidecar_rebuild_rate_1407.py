"""#1407 — `benchmarks/sidecar_rebuild_rate.py` denominators.

The script's own docstring forbids counting a fire that did no index work as a
cache hit, because that drives the measured rebuild rate toward zero as a
function of how old the log is. Keeping an *unmeasured* row in a denominator it
can never enter the numerator of is the same arithmetic, one step removed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "sidecar_rebuild_rate.py"


def _row(
    ts: str, *, outcome: str | None = None, gate_skip: str | None = None
) -> str:
    rec: dict[str, object] = {"hook": "user_prompt_submit", "ts": ts}
    if outcome is not None:
        rec["sidecar_outcome"] = outcome
    if gate_skip is not None:
        rec["prompt_shape_gate_skip"] = gate_skip
    return json.dumps(rec)


def _run(*logs: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), *(str(p) for p in logs)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.timeout(90)
def test_unmeasured_rows_are_out_of_every_denominator(tmp_path: Path) -> None:
    """The reviewer's executed case, pinned.

    50 pre-#1407 rows, 60 scored rows of which 10 are `full_rebuild`. The true
    measured rate is 10/60 = 16.67%. Before the fix the script printed
    10/110 = 9.09% on both the all-fires and the retrieval-fires denominators —
    the 50 unmeasured rows dragging it down exactly as if each had been scored
    as not-a-rebuild.

    The `9.09%` assertion is what makes this distinguishing: a test that only
    asserted `16.67%` appears would also pass on a script that printed both.
    """
    log = tmp_path / "hook_audit.jsonl"
    lines = [_row(f"2026-08-01T00:{i:02d}:00Z") for i in range(50)]
    lines += [
        _row(f"2026-08-05T00:{i:02d}:00Z", outcome="full_rebuild") for i in range(10)
    ]
    lines += [_row(f"2026-08-05T01:{i:02d}:00Z", outcome="fresh") for i in range(50)]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = _run(log)

    assert "10/60 = 16.67%" in out, out
    assert "9.09%" not in out, (
        "the 50 unmeasured rows are still in a denominator; that is the same "
        f"arithmetic as scoring each of them not-a-rebuild.\n{out}"
    )
    assert "10/110" not in out, out


@pytest.mark.timeout(90)
def test_measured_zeros_stay_in_the_all_fires_denominator(tmp_path: Path) -> None:
    """The other half of the fix, and the one an over-eager version breaks.

    Gate-skipped and no-index-work rows must NOT be subtracted. Those fires
    happened and built nothing — dropping them would inflate the rate #1380 is
    priced on, which is per-fire. Here: 10 scored (1 rebuild) plus 10
    gate-skipped and no unmeasured rows at all, so the all-fires denominator
    must be the full 20 while the retrieval denominator is 10.
    """
    log = tmp_path / "hook_audit.jsonl"
    lines = [_row("2026-08-05T00:00:00Z", outcome="full_rebuild")]
    lines += [
        _row(f"2026-08-05T00:{i:02d}:00Z", outcome="fresh") for i in range(1, 10)
    ]
    lines += [
        _row(f"2026-08-05T02:{i:02d}:00Z", gate_skip="trivial:short")
        for i in range(10)
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = _run(log)

    assert "0 unmeasured rows" in out, out
    assert "1/20 = 5.00%" in out, (
        "the gate-skipped rows left the all-fires denominator; they are "
        f"measured zeros and a per-fire cold_rate needs them.\n{out}"
    )
    assert "1/10 = 10.00%" in out, out


@pytest.mark.timeout(90)
def test_a_gate_skipped_row_carrying_an_outcome_is_scored(tmp_path: Path) -> None:
    """The cadence case the hook fix makes real.

    A fire refused by the shape gate can still have paid a rebuild inside the
    cadence dispatch, which runs above the gate. Such a row carries both
    `prompt_shape_gate_skip` and `sidecar_outcome`, and must be scored rather
    than swept into the gate-skipped bucket — which is where the expensive
    fires would otherwise hide.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        _row(
            "2026-08-05T00:00:00Z",
            outcome="full_rebuild",
            gate_skip="trivial:short",
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(log)

    assert "NO MEASUREMENT YET" not in out, out
    assert "fires with an outcome (scored) 1" in out, out
    assert "no key: gate-skipped           0" in out, out
    assert "1/1 = 100.00%" in out, out


@pytest.mark.timeout(90)
@pytest.mark.parametrize("suffix", ["", ".1"])
def test_the_no_measurement_early_return_still_holds(
    tmp_path: Path, suffix: str
) -> None:
    """Guard the guard: with nothing scored the script must refuse to print a
    rate at all, or every assertion above is about a code path the live log
    never reaches."""
    log = tmp_path / ("hook_audit.jsonl" + suffix)
    log.write_text(
        "\n".join(_row(f"2026-08-05T00:{i:02d}:00Z") for i in range(5)) + "\n",
        encoding="utf-8",
    )

    out = _run(log)

    assert "NO MEASUREMENT YET" in out, out
    assert "FULL-REBUILD RATE" not in out, out


# ---------------------------------------------------------------------------
# #1528: the script must state the window it actually covered
# ---------------------------------------------------------------------------

def _marker(generation: int, discarded: int) -> str:
    return json.dumps(
        {
            "hook": "audit_rotation",
            "ts": "2026-08-05T00:00:00Z",
            "generation": generation,
            "discarded_generations": discarded,
            "rotated_from": {
                "generation": generation - 1,
                "records": 999,
                "first_ts": "2026-08-01T00:00:00Z",
                "last_ts": "2026-08-04T23:59:59Z",
            },
        }
    )


@pytest.mark.timeout(90)
def test_output_names_the_window_and_flags_truncation(tmp_path: Path) -> None:
    """A truncated input must be visible in the output beside the rate.

    Before #1528 a benchmark reading `hook_audit.jsonl*` could not tell a
    short history from one whose older generations had been destroyed by
    single-slot rotation, so a long-horizon rate silently became a rate
    over the recent tail.
    """
    live = tmp_path / "hook_audit.jsonl"
    rotated = tmp_path / "hook_audit.jsonl.1"
    rotated.write_text(
        _marker(2, 0)
        + "\n"
        + "\n".join(
            _row(f"2026-08-06T00:{i:02d}:00Z", outcome="fresh")
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    live.write_text(
        _marker(3, 1)
        + "\n"
        + "\n".join(
            _row(f"2026-08-07T00:{i:02d}:00Z", outcome="full_rebuild")
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(live, rotated)

    assert "WINDOW TRUNCATED" in out, out
    assert "1 rotation generation(s) discarded" in out, out
    assert "window first row               2026-08-06T00:00:00Z" in out, out
    assert "window last row                2026-08-07T00:04:00Z" in out, out
    assert "rotated .1 present             True" in out, out
    # The marker rows are bookkeeping about the files, not fires in them.
    assert "user_prompt_submit fires       10" in out, out
    assert "non-UPS rows (ignored)         0" in out, out


@pytest.mark.timeout(90)
def test_unrotated_log_is_reported_complete_not_truncated(
    tmp_path: Path,
) -> None:
    """The truncation line must stay quiet when nothing was discarded.

    A warning that fires on every log is not a signal. This is the
    distinguishing half of the test above.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            _row(f"2026-08-05T00:{i:02d}:00Z", outcome="fresh")
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(log)

    assert "WINDOW TRUNCATED" not in out, out
    assert "window complete                no rotation has occurred" in out, out
    assert "rotated .1 present             False" in out, out


@pytest.mark.timeout(90)
def test_pre_1528_rotated_pair_is_reported_unknown(tmp_path: Path) -> None:
    """A `.1` with no marker cannot support a completeness claim.

    Logs already rotated in the wild have no generation stamp. One
    rollover discards nothing and a second discards the first archive;
    the files cannot say which, so the honest output is UNKNOWN — not
    "complete", which would be the exact unearned claim #1528 is about.
    """
    live = tmp_path / "hook_audit.jsonl"
    rotated = tmp_path / "hook_audit.jsonl.1"
    for path, hour in ((rotated, 0), (live, 1)):
        path.write_text(
            "\n".join(
                _row(f"2026-08-05T{hour:02d}:{i:02d}:00Z", outcome="fresh")
                for i in range(3)
            )
            + "\n",
            encoding="utf-8",
        )

    out = _run(live, rotated)

    assert "WINDOW UNKNOWN" in out, out
    assert "window complete" not in out, out
    assert "WINDOW TRUNCATED" not in out, out
    # And the legacy files still parse into the ordinary counts.
    assert "user_prompt_submit fires       6" in out, out


@pytest.mark.timeout(90)
def test_a_first_rotation_pair_is_reported_complete(tmp_path: Path) -> None:
    """Generation 2 beside a `.1`: a complete history, and it must say so.

    The first rollover fills an empty slot and destroys nothing, so this
    is the one rotated shape that has earned the word "complete". It is
    also the branch that separates "rotated" from "truncated" -- without
    it the truncation warning could be firing on the mere presence of a
    `.1` and every test above would still pass.
    """
    live = tmp_path / "hook_audit.jsonl"
    rotated = tmp_path / "hook_audit.jsonl.1"
    # The `.1` is the retired generation 1: it never had a predecessor to
    # stamp it, so it legitimately carries no marker.
    rotated.write_text(
        "\n".join(
            _row(f"2026-08-06T00:{i:02d}:00Z", outcome="fresh")
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    live.write_text(
        _marker(2, 0)
        + "\n"
        + "\n".join(
            _row(f"2026-08-07T00:{i:02d}:00Z", outcome="fresh")
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(live, rotated)

    assert "window complete                nothing discarded by rotation" in out, out
    assert "WINDOW TRUNCATED" not in out, out
    assert "WINDOW UNKNOWN" not in out, out
    assert "rotated .1 present             True" in out, out
    assert "rotation generation            2" in out, out


@pytest.mark.timeout(90)
def test_a_marked_live_file_alone_never_says_no_rotation_occurred(
    tmp_path: Path,
) -> None:
    """The script must not contradict its own generation line.

    Passing an explicit path is the documented usage in the script's
    docstring, so handing it the live half of a rotated pair is ordinary.
    `rotated .1 present` is then False -- the `.1` was not passed -- and
    gating the completeness wording on THAT printed "no rotation has
    occurred" directly under "rotation generation 2". A marked live file
    has provably rotated; only the generation can decide that sentence.
    """
    live = tmp_path / "hook_audit.jsonl"
    live.write_text(
        _marker(2, 0)
        + "\n"
        + "\n".join(
            _row(f"2026-08-07T00:{i:02d}:00Z", outcome="fresh")
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(live)

    assert "rotated .1 present             False" in out, out
    assert "rotation generation            2" in out, out
    assert "no rotation has occurred" not in out, out
    assert "window complete                nothing discarded by rotation" in out, out


@pytest.mark.timeout(90)
def test_the_archive_alone_is_unknown_not_complete(tmp_path: Path) -> None:
    """A `.1` handed in without its live file cannot claim completeness.

    Each file records what had been lost when it was CREATED, so an
    archive is blind to the loss caused by the rotation that archived it.
    The `.1` of a twice-rotated pair carries `generation 2, discarded 0`
    and, read alone, used to print "nothing discarded by rotation" -- for
    a file whose successor discarded exactly one generation.
    """
    rotated = tmp_path / "hook_audit.jsonl.1"
    rotated.write_text(
        _marker(2, 0)
        + "\n"
        + "\n".join(
            _row(f"2026-08-06T00:{i:02d}:00Z", outcome="fresh")
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(rotated)

    assert "WINDOW UNKNOWN" in out, out
    assert "window complete" not in out, out


@pytest.mark.timeout(90)
def test_a_bound_is_printed_as_a_bound_not_a_count(tmp_path: Path) -> None:
    """A pre-#1528 log that has since rolled over reports "at least N".

    This is the write-side half of #1528's load-bearing case, seen
    through the output a human reads. The rotation that destroyed an
    unmarked archive knows one generation died and cannot know how many
    died before it, so the count is a floor. Printing it bare would let a
    reader treat "1" as the whole loss when the real figure is unbounded.
    """
    live = tmp_path / "hook_audit.jsonl"
    rotated = tmp_path / "hook_audit.jsonl.1"
    marker = json.loads(_marker(3, 1))
    marker["discarded_unknown"] = True
    rotated.write_text(
        "\n".join(
            _row(f"2026-08-06T00:{i:02d}:00Z", outcome="fresh")
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    live.write_text(
        json.dumps(marker)
        + "\n"
        + "\n".join(
            _row(f"2026-08-07T00:{i:02d}:00Z", outcome="fresh")
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(live, rotated)

    assert "WINDOW TRUNCATED               at least 1 " in out, out
    assert "rotation generation            3 (at least)" in out, out
    assert "is a floor" in out, out
