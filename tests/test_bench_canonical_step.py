"""The nightly band check must run on a partial bench, and fail closed (CI-01).

`aelf bench all --canonical` exits 2 when an adapter's dataset is
missing. `bench-canonical.yml` ran it under `set -euo pipefail` and
wrote `out` to GITHUB_OUTPUT only afterwards, so exit 2 ended the step
first. The band check was gated on `out`, so it was skipped, and the
final step fired only on a band-check `failure`. A skipped band check
is not a failure, so the job reported success on 40 consecutive nights
without comparing a band.

`scripts/bench_canonical_step.py` now holds the step logic. The unit
tests drive it against fake bench commands. The workflow-text tests pin
the wiring, read as text the way `test_mutation_workflow_1457.py` does,
since PyYAML is not in the dev group.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import bench_canonical_step as step  # noqa: E402

_WORKFLOW = _REPO / ".github" / "workflows" / "bench-canonical.yml"

_BENCH = "Run aelf bench all --canonical"
_BANDCHECK = "Band-check vs canonical"
_FINAL = "Fail job unless the band check passed"


def _fake_bench(rc: int, *, write: bool) -> list[str]:
    """A bench stand-in that may write its `--out` file, then exits `rc`."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        f"if {write!r}:\n"
        "    Path(sys.argv[1]).write_text('{}\\n')\n"
        f"sys.exit({rc})\n"
    )
    return [sys.executable, "-c", code]


def _outputs(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@pytest.fixture()
def gh_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    return target


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("rc", "write", "expected"),
    [
        (0, True, 0),
        (2, True, 0),
        (1, True, 1),
        (0, False, 1),
        (2, False, 1),
        (1, False, 1),
        (3, True, 1),
    ],
)
def test_run_accepts_only_a_finished_bench_that_wrote_json(
    tmp_path: Path, gh_output: Path, rc: int, write: bool, expected: int,
) -> None:
    out = tmp_path / "nightly.json"
    cmd = [*_fake_bench(rc, write=write), str(out)]
    assert step.main(["run", "--out", str(out), "--", *cmd]) == expected
    # `rc` is recorded whatever it was, so the final step can judge it.
    assert _outputs(gh_output) == [f"out={out}", f"rc={rc}"]


@pytest.mark.timeout(60)
def test_run_writes_out_before_the_bench_starts(tmp_path: Path, gh_output: Path) -> None:
    """The defect itself: `out` was written after a command that could exit 2."""
    out = tmp_path / "nightly.json"
    probe = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "seen = Path(os.environ['GITHUB_OUTPUT']).read_text()\n"
        "Path(sys.argv[1]).write_text('{}\\n')\n"
        "sys.exit(0 if 'out=' in seen else 7)\n"
    )
    assert step.main(["run", "--out", str(out), "--", sys.executable, "-c", probe, str(out)]) == 0
    assert _outputs(gh_output) == [f"out={out}", "rc=0"]


@pytest.mark.timeout(60)
def test_run_dry_run_neither_runs_nor_writes(tmp_path: Path, gh_output: Path) -> None:
    out = tmp_path / "nightly.json"
    cmd = [*_fake_bench(0, write=True), str(out)]
    assert step.main(["run", "--out", str(out), "--dry-run", "--", *cmd]) == 0
    assert not out.exists()
    assert not gh_output.exists()


def test_run_without_a_command_fails(tmp_path: Path, gh_output: Path) -> None:
    assert step.main(["run", "--out", str(tmp_path / "x.json"), "--"]) == 1
    assert not gh_output.exists()


@pytest.mark.parametrize(
    ("rc", "bandcheck", "expected"),
    [
        ("0", "success", 0),
        ("2", "success", 0),
        ("1", "success", 1),
        ("", "success", 1),
        ("137", "success", 1),
        ("0", "failure", 1),
        ("2", "failure", 1),
        ("0", "skipped", 1),
        ("2", "skipped", 1),
        ("0", "cancelled", 1),
        ("0", "", 1),
        ("1", "skipped", 1),
    ],
)
def test_verdict_passes_only_a_finished_run_with_a_passing_band_check(
    rc: str, bandcheck: str, expected: int,
) -> None:
    assert step.main(["verdict", "--rc", rc, "--bandcheck", bandcheck]) == expected


def test_verdict_names_every_reason(capsys: pytest.CaptureFixture[str]) -> None:
    assert step.main(["verdict", "--rc", "1", "--bandcheck", "skipped"]) == 1
    errors = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("::error::")
    ]
    assert len(errors) == 2, errors


def test_verdict_dry_run_reports_without_failing(capsys: pytest.CaptureFixture[str]) -> None:
    assert step.main(["verdict", "--rc", "1", "--bandcheck", "failure", "--dry-run"]) == 0
    assert "::error::" in capsys.readouterr().out


# --- workflow wiring -------------------------------------------------------


def _step_block(name: str) -> list[str]:
    """Every line of the step titled `name`, continuations joined."""
    lines = _WORKFLOW.read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
    marker = f"- name: {name}"
    starts = [i for i, line in enumerate(lines) if line.strip() == marker]
    assert len(starts) == 1, f"expected exactly one step named {name!r}"
    start = starts[0]
    column = len(lines[start]) - len(lines[start].lstrip(" "))
    block: list[str] = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip(" ")) <= column:
            break
        block.append(line)
    return block


def _key(block: list[str], key: str) -> str:
    """The value of a step-level `key:`, or '' when the step has none."""
    for line in block:
        match = re.match(rf"\s*{re.escape(key)}:\s*(.*)$", line)
        if match and not line.strip().startswith("#"):
            return match.group(1).strip()
    return ""


def _code(block: list[str]) -> list[str]:
    return [line.strip() for line in block if line.strip() and not line.strip().startswith("#")]


def test_the_bench_command_runs_inside_the_step_script() -> None:
    """`out` is written by the script before the bench, not by a line after it."""
    code = _code(_step_block(_BENCH))
    bench_lines = [line for line in code if "aelf bench all" in line]
    assert len(bench_lines) == 1, bench_lines
    line = bench_lines[0]
    assert "bench_canonical_step.py run" in line, line
    assert line.index("bench_canonical_step.py run") < line.index(" -- ") < line.index("aelf bench all")
    assert not any("GITHUB_OUTPUT" in line for line in code), (
        "the bench step writes GITHUB_OUTPUT itself again; a write after the "
        "bench command is the defect CI-01 removed"
    )


def test_no_step_relies_on_continue_on_error() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert not re.search(r"^\s*continue-on-error:", text, re.MULTILINE)


def test_band_check_is_gated_on_the_json_file_not_a_bench_output() -> None:
    block = _step_block(_BANDCHECK)
    condition = _key(block, "if")
    assert "!cancelled()" in condition, condition
    assert "steps.bench" not in condition, (
        f"band-check's if: reads a bench step output again: {condition!r}"
    )
    assert any(re.search(r"test -s \"\$\{OUT\}\"", line) for line in _code(block))


def test_the_final_step_covers_a_skipped_band_check_and_a_bad_rc() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    names = re.findall(r"^\s*- name: (.+)$", text, re.MULTILINE)
    assert names[-1] == _FINAL, f"the verdict step must be last; found {names[-1]!r}"
    block = _step_block(_FINAL)
    condition = _key(block, "if")
    assert condition == "${{ !cancelled() }}", (
        "the final step must run whatever the band check did; a condition on "
        f"its outcome skips exactly the case CI-01 is about: {condition!r}"
    )
    assert _key(block, "BENCH_RC") == "${{ steps.bench.outputs.rc }}"
    assert _key(block, "BANDCHECK_OUTCOME") == "${{ steps.bandcheck.outcome }}"
    code = " ".join(_code(block))
    assert "bench_canonical_step.py verdict" in code
    assert '--rc "${BENCH_RC}"' in code
    assert '--bandcheck "${BANDCHECK_OUTCOME}"' in code


def test_exit_code_two_counts_as_a_finished_run() -> None:
    assert step.FINISHED_RCS == frozenset({0, 2})
    badge_if = _key(_step_block("Compute badge text"), "if")
    assert "steps.bench.outputs.rc == '2'" in badge_if, badge_if


def test_the_nightly_json_is_committed_after_a_failed_band_check() -> None:
    block = _step_block("Commit + push to bench-canonical-results")
    assert _key(block, "if") == "${{ !cancelled() }}"
