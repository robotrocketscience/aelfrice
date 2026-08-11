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


def _run(log: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), str(log)],
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
