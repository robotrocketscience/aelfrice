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
    ts: str,
    *,
    outcome: str | None = None,
    gate_skip: str | None = None,
    session_id: str | None = None,
) -> str:
    rec: dict[str, object] = {"hook": "user_prompt_submit", "ts": ts}
    if outcome is not None:
        rec["sidecar_outcome"] = outcome
    if gate_skip is not None:
        rec["prompt_shape_gate_skip"] = gate_skip
    if session_id is not None:
        rec["session_id"] = session_id
    return json.dumps(rec)


def _run(*logs: Path, window: tuple[str | None, str | None] = (None, None)) -> str:
    since, until = window
    flags: list[str] = []
    if since is not None:
        flags += ["--since", since]
    if until is not None:
        flags += ["--until", until]
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), *flags, *(str(p) for p in logs)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _bucket_lines(out: str) -> tuple[str, str]:
    """Return the (session-FIRST, LATER) report lines, or fail loudly.

    Reading the whole stdout with `in` would let the pooled rate printed
    higher up satisfy a bucket assertion, which is precisely the collapse
    #1513's AC3 exists to forbid. Every bucketing assertion below is made
    against these two lines alone.
    """
    first = [ln for ln in out.splitlines() if "session-FIRST fires" in ln]
    later = [ln for ln in out.splitlines() if "LATER" in ln and "fires" in ln]
    assert len(first) == 1, f"expected one session-FIRST line\n{out}"
    assert len(later) == 1, f"expected one LATER line\n{out}"
    return first[0], later[0]


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


# ---- #1513: the session-first vs later split ---------------------------


@pytest.mark.timeout(90)
def test_the_session_position_split_cannot_collapse_into_one_number(
    tmp_path: Path,
) -> None:
    """#1513 AC3. The two buckets must be reported separately.

    Three sessions, each opening with a `full_rebuild` and then running nine
    `fresh` fires. The truth is 3/3 = 100.0% on session-FIRST against
    0/27 = 0.0% on LATER — a complete separation. Pooled, the same rows read
    3/30 = 10.00%, which is the number that hides a session-first tail.

    The distinguishing assertion is the negative one: a script that pooled
    the buckets would print `10.0%` on both bucket lines and still satisfy a
    test that merely looked for `100.0%` somewhere in stdout.
    """
    log = tmp_path / "hook_audit.jsonl"
    lines: list[str] = []
    for s in range(3):
        lines.append(
            _row(
                f"2026-08-1{s}T00:00:00Z",
                outcome="full_rebuild",
                session_id=f"sess-{s}",
            )
        )
        lines += [
            _row(
                f"2026-08-1{s}T00:{i:02d}:00Z",
                outcome="fresh",
                session_id=f"sess-{s}",
            )
            for i in range(1, 10)
        ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = _run(log)
    first_line, later_line = _bucket_lines(out)

    assert "3/3 = 100.0%" in first_line, out
    assert "0/27 = 0.0%" in later_line, out
    assert "10.0%" not in first_line, (
        "the session-first bucket is reporting the POOLED rate; that is the "
        f"collapse AC3 forbids.\n{out}"
    )
    assert "10.0%" not in later_line, out
    # The pooled rate is still reported — it is a different question, not a
    # replacement. Losing it would be its own regression.
    assert "3/30 = 10.00%" in out, out


@pytest.mark.timeout(90)
def test_a_row_with_no_session_id_lands_in_neither_bucket(
    tmp_path: Path,
) -> None:
    """A fire with no position must not be swept into LATER.

    Sweeping it there drags the session-first rate toward zero exactly the
    way folding an unmeasured row into a denominator does, one axis over.
    Here one session (1 rebuild + 1 fresh) plus two session-less rebuilds:
    the buckets must stay 1/1 and 0/1, and the two loose rows must be named.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            [
                _row(
                    "2026-08-05T00:00:00Z",
                    outcome="full_rebuild",
                    session_id="sess-a",
                ),
                _row(
                    "2026-08-05T00:01:00Z", outcome="fresh", session_id="sess-a"
                ),
                _row("2026-08-05T00:02:00Z", outcome="full_rebuild"),
                _row("2026-08-05T00:03:00Z", outcome="full_rebuild"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(log)
    first_line, later_line = _bucket_lines(out)

    assert "1/1 = 100.0%" in first_line, out
    assert "0/1 = 0.0%" in later_line, (
        "the two session-less rebuilds were folded into LATER\n" + out
    )
    assert "no session_id            2" in out, out


@pytest.mark.timeout(90)
def test_position_follows_the_timestamp_not_the_file_order(
    tmp_path: Path,
) -> None:
    """Rotated logs are globbed newest-first; position is not read that way.

    `_default_logs` sorts by name, so `hook_audit.jsonl` (the live file)
    comes before `hook_audit.jsonl.1` (the older rotation). Walking rows in
    that order would mark the session's LAST fire as its first. Here the
    session's opening `full_rebuild` is in the rotated file and its later
    `fresh` fire is in the live one: if position followed file order the
    buckets would invert to 0/1 and 1/1.
    """
    live = tmp_path / "hook_audit.jsonl"
    rotated = tmp_path / "hook_audit.jsonl.1"
    live.write_text(
        _row("2026-08-05T09:00:00Z", outcome="fresh", session_id="sess-a")
        + "\n",
        encoding="utf-8",
    )
    rotated.write_text(
        _row(
            "2026-08-05T08:00:00Z",
            outcome="full_rebuild",
            session_id="sess-a",
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(live, rotated)
    first_line, later_line = _bucket_lines(out)

    assert "1/1 = 100.0%" in first_line, (
        "the rebuild was read as a LATER fire; position followed the glob "
        f"order rather than the timestamp.\n{out}"
    )
    assert "0/1 = 0.0%" in later_line, out


# ---- #1513 round 2: one log set, two windows, one comparison ------------
#
# The first statement of #1513's AC1 target compared a post-fix rate taken
# on one day against a baseline taken weeks earlier over a log set that had
# grown and rotated in between, so the two numbers were never about the same
# population and their difference was not the fix. `--since` / `--until` cut
# both windows out of one log set in one pair of invocations, which is what
# makes them comparable. These tests pin that the cut partitions the input
# and that it reaches the session-position buckets, not just the totals.


@pytest.mark.timeout(90)
def test_the_two_windows_partition_the_input_with_no_row_in_both(
    tmp_path: Path,
) -> None:
    """Half-open, so `--until T` and `--since T` lose nothing and share nothing.

    A fire exactly on the boundary must land in the later window and only
    there. If the bound were inclusive on both sides, a baseline run and a
    post-fix run over the same log set would double-count it, and the
    target's two halves would overlap by however many fires sat on the
    boundary second.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            [
                _row("2026-09-01T00:00:00Z", outcome="full_rebuild", session_id="a"),
                _row("2026-09-01T23:59:59Z", outcome="fresh", session_id="a"),
                # Exactly the boundary: belongs to the --since side alone.
                _row("2026-09-02T00:00:00Z", outcome="full_rebuild", session_id="b"),
                _row("2026-09-02T00:00:01Z", outcome="fresh", session_id="b"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    boundary = "2026-09-02T00:00:00Z"

    before = _run(log, window=(None, boundary))
    after = _run(log, window=(boundary, None))

    assert "fires with an outcome (scored) 2" in before, before
    assert "fires with an outcome (scored) 2" in after, after
    assert "UPS fires outside the window   2" in before, before
    assert "UPS fires outside the window   2" in after, after

    # The whole log carries 4 scored fires, so 2 + 2 is the partition: no
    # row is in both windows and none was dropped by either.
    whole = _run(log)
    assert "fires with an outcome (scored) 4" in whole, whole


@pytest.mark.timeout(90)
def test_the_window_reaches_the_session_position_buckets(tmp_path: Path) -> None:
    """The cut has to move the two numbers the #1513 target is read on.

    A filter that only trimmed the totals would leave the session-FIRST and
    LATER lines computed over the whole log, so a baseline run and a
    post-fix run would print the same two rates and the target would be
    unfalsifiable. Here the early window holds a session-first rebuild and
    the late window a session-first fresh; a filter that never reached the
    buckets would print the pre-fix 100.0% on the post-fix run too.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            [
                _row("2026-09-01T00:00:00Z", outcome="full_rebuild", session_id="old"),
                _row("2026-09-01T00:00:01Z", outcome="fresh", session_id="old"),
                _row("2026-09-03T00:00:00Z", outcome="fresh", session_id="new"),
                _row("2026-09-03T00:00:01Z", outcome="fresh", session_id="new"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    boundary = "2026-09-02T00:00:00Z"

    first_before, later_before = _bucket_lines(_run(log, window=(None, boundary)))
    first_after, later_after = _bucket_lines(_run(log, window=(boundary, None)))

    assert "1/1 = 100.0%" in first_before, first_before
    assert "0/1 = 0.0%" in later_before, later_before
    assert "0/1 = 0.0%" in first_after, first_after
    assert "0/1 = 0.0%" in later_after, later_after
    assert "100.0%" not in first_after, (
        "the window did not reach the session-position buckets; the "
        f"post-fix run still carries the pre-fix rebuild\n{first_after}"
    )


@pytest.mark.timeout(90)
def test_an_undated_fire_is_out_of_any_window_that_was_asked_for(
    tmp_path: Path,
) -> None:
    """A row with no `ts` has no position on the time axis.

    Keeping it inside a window puts a fire of unknown date into a rate the
    caller asked to be about a specific period — the same unearned
    attribution the script refuses for unmeasured rows, one axis over. With
    no window asked for it is still scored, because then no claim about a
    period is being made.
    """
    log = tmp_path / "hook_audit.jsonl"
    undated = json.dumps(
        {
            "hook": "user_prompt_submit",
            "sidecar_outcome": "full_rebuild",
            "session_id": "u",
        }
    )
    log.write_text(
        "\n".join(
            [undated, _row("2026-09-03T00:00:00Z", outcome="fresh", session_id="d")]
        )
        + "\n",
        encoding="utf-8",
    )

    windowed = _run(log, window=("2026-09-01T00:00:00Z", None))
    assert "fires with an outcome (scored) 1" in windowed, windowed
    assert "UPS fires outside the window   1" in windowed, windowed

    unwindowed = _run(log)
    assert "fires with an outcome (scored) 2" in unwindowed, unwindowed


# ---- #1513 round 4: the session buckets have to partition too ---------
#
# `--since` / `--until` partition the ROWS. Until this pass they did not
# partition the SESSIONS: a session that began before the boundary and ran
# past it had its first in-window fire promoted to session-FIRST in the
# `--since` run, so the same session was counted as session-FIRST on both
# sides. Over the machine-wide logs at 2026-09-01T00:00:00Z that printed 35
# sessions for `--until`, 6 for `--since`, and 40 for the whole set.
#
# It is not a bookkeeping nit. The promoted fire is a mid-session prompt no
# `SessionStart` warm inside the window could have spared, and LATER fires
# carry the lower rebuild rate, so each pseudo-first pulls the `--since`
# session-FIRST rate down — toward satisfying a target stated as that rate
# falling.


def _session_line(out: str, label: str) -> str:
    hits = [ln for ln in out.splitlines() if label in ln]
    assert len(hits) == 1, f"expected one {label!r} line\n{out}"
    return hits[0]


@pytest.mark.timeout(90)
def test_the_two_windows_partition_the_sessions_not_only_the_rows(
    tmp_path: Path,
) -> None:
    """A straddling session is counted as beginning on exactly one side.

    Session `straddle` starts before the boundary and fires again after it;
    `early` and `late` sit wholly on one side each. The whole-set run sees
    3 sessions, so `--until` + `--since` has to be 3 as well. Before this
    pass `--since` promoted `straddle`'s post-boundary fire to
    session-FIRST and the two sides summed to 4.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            [
                _row("2026-09-01T00:00:00Z", outcome="fresh", session_id="early"),
                _row(
                    "2026-09-01T00:00:01Z", outcome="fresh", session_id="straddle"
                ),
                _row(
                    "2026-09-03T00:00:00Z",
                    outcome="full_rebuild",
                    session_id="straddle",
                ),
                _row("2026-09-03T00:00:01Z", outcome="fresh", session_id="late"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    boundary = "2026-09-02T00:00:00Z"

    before = _run(log, window=(None, boundary))
    after = _run(log, window=(boundary, None))
    whole = _run(log)

    assert "sessions beginning in this window   2" in before, before
    assert "sessions beginning in this window   1" in after, after
    assert "sessions with a scored fire   3" in whole, whole

    # The rows still partition, which is what makes the session count the
    # thing being fixed here rather than a side effect of dropping rows.
    assert "fires with an outcome (scored) 2" in before, before
    assert "fires with an outcome (scored) 2" in after, after
    assert "fires with an outcome (scored) 4" in whole, whole


@pytest.mark.timeout(90)
def test_a_straddling_sessions_in_window_fires_are_all_later(
    tmp_path: Path,
) -> None:
    """And the promoted fire does not land in the session-FIRST rate.

    `straddle`'s only post-boundary fire is a `full_rebuild`. Counting it
    as session-FIRST puts 1/1 = 100.0% on the `--since` session-FIRST line;
    counting it as LATER — which is what it is, the session having begun
    before the window — leaves session-FIRST holding only `late`'s `fresh`.
    The two readings differ on both lines, so this is distinguishing.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            [
                _row(
                    "2026-09-01T00:00:00Z", outcome="fresh", session_id="straddle"
                ),
                _row(
                    "2026-09-03T00:00:00Z",
                    outcome="full_rebuild",
                    session_id="straddle",
                ),
                _row("2026-09-03T00:00:01Z", outcome="fresh", session_id="late"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = _run(log, window=("2026-09-02T00:00:00Z", None))
    first_line, later_line = _bucket_lines(out)

    assert "0/1 = 0.0%" in first_line, first_line
    assert "1/1 = 100.0%" in later_line, later_line
    assert "100.0%" not in first_line, (
        "the straddling session's post-boundary rebuild was scored as a "
        f"session-FIRST fire\n{first_line}"
    )


@pytest.mark.timeout(90)
def test_the_window_reports_how_many_sessions_it_carried_in(
    tmp_path: Path,
) -> None:
    """The count is printed, so a LATER-heavy window is visible.

    A `--since` run whose fires mostly belong to sessions that began
    earlier is a run with few first prompts in it, and its session-FIRST
    rate rests on a small sample. The number is reported rather than left
    for the reader to derive by diffing session counts across two runs.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            [
                _row("2026-09-01T00:00:00Z", outcome="fresh", session_id="a"),
                _row("2026-09-03T00:00:00Z", outcome="fresh", session_id="a"),
                _row("2026-09-01T00:00:01Z", outcome="fresh", session_id="b"),
                _row("2026-09-03T00:00:01Z", outcome="fresh", session_id="b"),
                _row("2026-09-03T00:00:02Z", outcome="fresh", session_id="c"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    after = _run(log, window=("2026-09-02T00:00:00Z", None))
    assert "sessions carried in from earlier    2" in after, after
    assert "sessions beginning in this window   1" in after, after

    # With no window nothing can be carried in, and the line that would
    # report it is not printed at all.
    whole = _run(log)
    assert "carried in from earlier" not in whole, whole
    assert "sessions with a scored fire   3" in whole, whole


@pytest.mark.timeout(90)
def test_an_out_of_window_fire_is_counted_in_no_bucket(tmp_path: Path) -> None:
    """Carrying the row for positioning must not also score it.

    The out-of-window rows exist in `scored_rows` only so a session's
    position can be decided over the whole input. If one leaked into a
    counter, the `--until` and `--since` runs would double-count it and the
    row partition the previous tests pin would break — which is why this
    asserts the bucket totals and not only the headline `scored` count.
    """
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            [
                _row(
                    "2026-09-01T00:00:00Z",
                    outcome="full_rebuild",
                    session_id="a",
                ),
                _row("2026-09-03T00:00:00Z", outcome="fresh", session_id="a"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    after = _run(log, window=("2026-09-02T00:00:00Z", None))
    first_line, later_line = _bucket_lines(after)

    assert "fresh 0  incremental 0  full_rebuild 0" in first_line, first_line
    assert "no scored fires in this bucket" in first_line, first_line
    assert "fresh 1  incremental 0  full_rebuild 0" in later_line, later_line
    assert "0/1 = 0.0%" in later_line, later_line
    assert "fires with an outcome (scored) 1" in after, after
    assert "UPS fires outside the window   1" in after, after


@pytest.mark.timeout(90)
def test_an_out_of_window_undated_fire_is_not_counted_as_unattributed(
    tmp_path: Path,
) -> None:
    """A row with no `session_id` and no `ts` positions nothing.

    It is carried like any other out-of-window scored row, so it has to be
    kept out of the `no session_id` count as well — that count is a count
    of in-window fires, and inflating it would report fires the window does
    not contain.
    """
    undated = json.dumps(
        {"hook": "user_prompt_submit", "sidecar_outcome": "full_rebuild"}
    )
    log = tmp_path / "hook_audit.jsonl"
    log.write_text(
        "\n".join(
            [undated, _row("2026-09-03T00:00:00Z", outcome="fresh", session_id="a")]
        )
        + "\n",
        encoding="utf-8",
    )

    after = _run(log, window=("2026-09-02T00:00:00Z", None))
    assert "no session_id" not in after, after
    assert "fires with an outcome (scored) 1" in after, after

    # With no window it IS in range, and then it is reported.
    whole = _run(log)
    assert "no session_id            1" in whole, whole
