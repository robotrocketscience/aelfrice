"""The per-PR mutation report says when a run checked nothing (#1746 AC2).

When the clean run inside `mutants/` fails, mutmut stops before executing any
mutant and every one stays `not checked`. The Report step exits 0 by design,
so the green tick hid runs that tested nothing. These tests run the step's own
body with bash against fixed reports, like the #1709 tests.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tests.test_mutation_report_count_1709 import _BASH, _run_step

_REPO = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.skipif(_BASH is None, reason="needs bash"),
    pytest.mark.timeout(60),  # spawns bash and grep (#1307)
]

_WARNING = "::warning title=Mutation run checked nothing::"


def _lines(*statuses: str) -> str:
    return "".join(
        f"    pkg.mod.x_f__mutmut_{i}: {status}\n" for i, status in enumerate(statuses, 1)
    )


def test_all_not_checked_warns_in_the_log_and_the_summary(tmp_path: Path) -> None:
    proc, summary = _run_step("Report", _lines(*["not checked"] * 4), tmp_path, "C.UTF-8")
    assert proc.returncode == 0, proc.stderr
    assert (
        f"{_WARNING}all 4 mutants are 'not checked', so this run tested nothing."
        in proc.stdout
    ), proc.stdout
    assert "> [!WARNING]\n> all 4 mutants are 'not checked'" in summary, summary
    assert summary.index("[!WARNING]") < summary.index("| mutants |"), summary


def test_no_mutants_at_all_warns(tmp_path: Path) -> None:
    proc, summary = _run_step("Report", "", tmp_path, "C.UTF-8")
    assert proc.returncode == 0, proc.stderr
    assert f"{_WARNING}mutmut listed no mutants at all" in proc.stdout, proc.stdout
    assert "> mutmut listed no mutants at all" in summary, summary


@pytest.mark.parametrize("statuses", [
    ("killed", "not checked", "not checked"),
    ("survived", "not checked"),
    ("killed", "survived"),
    ("timeout", "not checked"),
])
def test_a_run_that_checked_something_does_not_warn(
    tmp_path: Path, statuses: tuple[str, ...],
) -> None:
    proc, summary = _run_step("Report", _lines(*statuses), tmp_path, "C.UTF-8")
    assert proc.returncode == 0, proc.stderr
    assert "::warning" not in proc.stdout, proc.stdout
    assert "[!WARNING]" not in summary, summary


def _deselected() -> list[str]:
    config = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    args = config["tool"]["mutmut"].get("pytest_add_cli_args", [])
    return [args[i + 1] for i, a in enumerate(args) if a == "--deselect"]


def test_each_mutation_run_deselect_names_a_real_test() -> None:
    """AC1: a renamed test would turn its `--deselect` into a silent no-op,
    and the clean run inside `mutants/` would fail on it again."""
    targets = _deselected()
    assert targets, "the mutation run deselects nothing"
    for target in targets:
        path, sep, func = target.partition("::")
        assert sep and func, target
        source = (_REPO / path).read_text(encoding="utf-8")
        assert re.search(rf"^def {re.escape(func)}\(", source, re.M), target


def test_each_deselect_states_its_reason_in_the_config() -> None:
    """AC1 asks for the reason to be stated where the test is excluded."""
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    block = text[text.index("# #1746: tests that can't pass inside `mutants/`"):
                 text.index("pytest_add_cli_args = [")]
    for target in _deselected():
        module = target.split("::")[0].removeprefix("tests/").removesuffix(".py")
        assert module in block, target


def test_the_mutation_run_deselects_exactly_these_tests() -> None:
    """Dropping a `--deselect`, or swapping it for another flag, keeps every
    other guard here green, and the next mutation run checks nothing again.
    Update this list together with the reasons in `pyproject.toml`."""
    config = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["mutmut"]["pytest_add_cli_args"] == [
        "-m",
        "not source_scan",
        "--ignore",
        "tests/e2e",
        "--deselect",
        "tests/test_conflict_markers_1491.py::test_no_tracked_file_carries_a_marker",
        "--deselect",
        "tests/test_docs_cross_file_anchors_1511.py::test_the_scan_is_not_vacuous",
        "--deselect",
        "tests/test_block_ceiling_hermetic_home_1716.py::"
        "test_a_home_config_does_not_move_the_session_start_figures",
        *(
            arg
            for case in (
                "test_module_raises",
                "worker_path_passes",
                "allowlisted_module_passes",
                "benchmark_seed_corpus_passes",
                "simulator_populate_store_passes",
                "migrate_passes",
            )
            for arg in ("--deselect", f"tests/test_insert_belief_gate.py::test_gate_on_{case}")
        ),
    ]
