"""The weekly mutation run is planned into shards that each finish (#1747).

The weekly job ran mutmut over the whole tree in one job and hit its time
limit during mutant generation every week, with no mutant ever run.
`scripts/mutation_shards.py` cuts the tree into shards, scopes one shard's
checkout, and merges the shards' reports; `mutation.yml` runs it as a matrix.
These tests pin the planner, the scoping, the merge, and the workflow wiring.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "mutation_shards.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "mutation.yml"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("mutation_shards_1747", str(_SCRIPT))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mutation_shards_1747"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sh() -> Any:
    return _load()


# --- cutting a file into pieces --------------------------------------------


def test_a_small_file_is_one_whole_piece(sh: Any) -> None:
    pieces = sh.file_pieces("src/aelfrice/a.py", {"f": 3, "g": 4}, {}, 10)
    assert pieces == [sh.Piece("src/aelfrice/a.py", None, 7)]


def test_a_large_file_splits_into_consecutive_runs_under_the_cap(sh: Any) -> None:
    counts = {"a": 4, "b": 3, "c": 5, "d": 2, "e": 1}
    pieces = sh.file_pieces("src/aelfrice/a.py", counts, {}, 8)
    assert [p.units for p in pieces] == [("a", "b"), ("c", "d", "e")]
    assert [p.mutants for p in pieces] == [7, 8]


def test_a_unit_over_the_cap_is_a_piece_by_itself(sh: Any) -> None:
    pieces = sh.file_pieces("src/aelfrice/a.py", {"a": 2, "big": 50, "c": 2}, {}, 10)
    assert [p.units for p in pieces] == [("a",), ("big",), ("c",)]


def test_units_with_no_mutants_are_left_out(sh: Any) -> None:
    pieces = sh.file_pieces("src/aelfrice/a.py", {"a": 6, "z": 0, "b": 6}, {}, 6)
    assert [p.units for p in pieces] == [("a",), ("b",)]
    assert sh.file_pieces("src/aelfrice/a.py", {"z": 0}, {}, 6) == []


def test_an_excluded_unit_is_never_in_a_piece(sh: Any) -> None:
    """Even under the cap, the units are listed so `apply` can pragma the rest."""
    pieces = sh.file_pieces(
        "src/aelfrice/cli.py", {"main": 3, "build_parser": 99}, {"build_parser": "r"}, 1000,
    )
    assert pieces == [sh.Piece("src/aelfrice/cli.py", ("main",), 3)]


def test_a_missing_excluded_unit_stops_the_plan(sh: Any) -> None:
    """A renamed `build_parser` would otherwise put its mutants back unseen."""
    with pytest.raises(sh.PlanError, match="build_parser"):
        sh.file_pieces("src/aelfrice/cli.py", {"main": 3}, {"build_parser": "r"}, 1000)


# --- assigning pieces to shards -------------------------------------------


def test_assign_takes_the_largest_piece_first_to_the_lightest_shard(sh: Any) -> None:
    pieces = [sh.Piece(f"src/aelfrice/m{n}.py", None, n) for n in (5, 4, 3, 3, 2, 1)]
    shards = sh.assign(pieces, 2)
    assert [s.mutants for s in shards] == [9, 9]
    assert sorted(p.mutants for s in shards for p in s.pieces) == [1, 2, 3, 3, 4, 5]


def test_assign_is_deterministic_and_breaks_ties_by_index(sh: Any) -> None:
    pieces = [sh.Piece(f"src/aelfrice/m{n}.py", None, 1) for n in range(3)]
    first = sh.assign(pieces, 3)
    assert [s.pieces[0].path for s in first] == [
        "src/aelfrice/m0.py", "src/aelfrice/m1.py", "src/aelfrice/m2.py",
    ]
    again = sh.assign(list(reversed(pieces)), 3)
    assert [s.files() for s in again] == [s.files() for s in first]


def test_pieces_of_one_file_in_one_shard_merge_in_source_order(sh: Any) -> None:
    shard = sh.Shard(0, [
        sh.Piece("src/aelfrice/a.py", ("c", "d"), 2),
        sh.Piece("src/aelfrice/a.py", ("a", "b"), 2),
    ])
    assert shard.files() == {"src/aelfrice/a.py": ["c", "d", "a", "b"]}


# --- planning a tree --------------------------------------------------------


def _tree(root: Path) -> Path:
    pkg = root / "src" / "aelfrice"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "cli.py").write_text(
        "def build_parser():\n    return 1\n\n\ndef main():\n    return 2\n",
        encoding="utf-8",
    )
    (pkg / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (pkg / "sub" / "b.py").write_text("def g():\n    return 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n[tool.mutmut]\nsource_paths = ["src/aelfrice"]\n',
        encoding="utf-8",
    )
    return root


def _units_counter(path: str, source: str) -> dict[str, int]:
    """One mutant per line of each unit: deterministic, and no mutmut."""
    tree = ast.parse(source)
    out: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert node.end_lineno is not None
            out[node.name] = node.end_lineno - node.lineno + 1
    return out


def test_plan_covers_subpackages_and_drops_the_excluded_unit(sh: Any, tmp_path: Path) -> None:
    plan = sh.plan_tree(_tree(tmp_path), 2, 100, _units_counter)
    files = {p: u for s in plan["shards"] for p, u in s["files"].items()}
    assert set(files) == {
        "src/aelfrice/a.py", "src/aelfrice/cli.py", "src/aelfrice/sub/b.py",
    }
    assert files["src/aelfrice/cli.py"] == ["main"]
    assert plan["mutants"] == 6
    assert sum(s["mutants"] for s in plan["shards"]) == plan["mutants"]


def test_plan_stops_when_an_excluded_file_is_gone(sh: Any, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "src" / "aelfrice" / "cli.py").unlink()
    with pytest.raises(sh.PlanError, match="cli.py"):
        sh.plan_tree(root, 2, 100, _units_counter)


# --- applying one shard -----------------------------------------------------


def _plan_for(sh: Any, root: Path) -> dict[str, Any]:
    return sh.plan_tree(root, 1, 100, _units_counter)


def test_apply_pragmas_the_units_outside_the_shard_and_scopes_only_mutate(
    sh: Any, tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    before_a = (root / "src/aelfrice/a.py").read_text(encoding="utf-8")
    sh.apply_shard(root, _plan_for(sh, root), 0, write=True)

    cli = (root / "src/aelfrice/cli.py").read_text(encoding="utf-8")
    assert "def build_parser():  # pragma: no mutate block" in cli
    assert "def main():\n" in cli, "the shard's own unit stays mutable"
    assert (root / "src/aelfrice/a.py").read_text(encoding="utf-8") == before_a

    config = (root / "pyproject.toml").read_text(encoding="utf-8")
    listed = re.findall(r'^    "([^"]+)",$', config, re.MULTILINE)
    assert sorted(listed) == [
        "src/aelfrice/a.py", "src/aelfrice/cli.py", "src/aelfrice/sub/b.py",
    ]


def test_apply_dry_run_writes_nothing(sh: Any, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    sh.apply_shard(root, _plan_for(sh, root), 0, write=False)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == snapshot


def test_apply_refuses_a_unit_the_file_no_longer_has_and_writes_nothing(
    sh: Any, tmp_path: Path,
) -> None:
    """Every edit is computed first: `cli.py`'s pragma, which comes before
    the failing file in the plan, must not be written either."""
    root = _tree(tmp_path)
    plan = _plan_for(sh, root)
    plan["shards"][0]["files"]["src/aelfrice/sub/b.py"] = ["g", "gone"]
    snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(sh.PlanError, match="gone"):
        sh.apply_shard(root, plan, 0, write=True)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == snapshot


def test_apply_names_a_shard_the_plan_does_not_have(sh: Any, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with pytest.raises(sh.PlanError, match="no shard 5"):
        sh.apply_shard(root, _plan_for(sh, root), 5, write=False)


def test_only_mutate_is_never_written_twice(sh: Any) -> None:
    text = "[tool.mutmut]\nonly_mutate = []\n"
    with pytest.raises(sh.PlanError, match="already sets only_mutate"):
        sh.only_mutate_config(text, ["src/aelfrice/a.py"])


def test_apply_refuses_outside_ci(
    sh: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _tree(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan_for(sh, root)), encoding="utf-8")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.chdir(root)
    before = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert sh.main(["apply", "--plan", str(plan), "--shard", "0"]) == 3
    assert (root / "pyproject.toml").read_text(encoding="utf-8") == before


# --- merging the shards' reports ----------------------------------------------


def _report(**statuses: int) -> str:
    lines = ["noise that is not a result line"]
    n = 0
    for status, count in statuses.items():
        for _ in range(count):
            n += 1
            lines.append(f"    aelfrice.m.x_f__mutmut_{n}: {status.replace('_', ' ')}")
    return "\n".join(lines) + "\n"


def _two_shard_plan() -> dict[str, Any]:
    return {
        "version": 1, "cap": 10, "mutants": 8, "excluded": {},
        "shards": [
            {"index": 0, "mutants": 4, "files": {"src/aelfrice/a.py": None}},
            {"index": 1, "mutants": 4, "files": {"src/aelfrice/b.py": None}},
        ],
    }


def test_merge_sums_the_shards_and_passes_a_partial_run(sh: Any) -> None:
    summary, failed = sh.merge_reports(_two_shard_plan(), {
        0: _report(killed=3, survived=1),
        1: _report(killed=1, not_checked=3),
    })
    assert not failed, "a shard that ran out of time partway is a warning, not a failure"
    assert "5 of 8 planned mutants checked: 4 killed, 1 survived." in summary
    assert "| 1 | 4 | 4 | 1 | 0 | 3 |" in summary


def test_merge_fails_when_a_planned_shard_has_no_report(sh: Any) -> None:
    summary, failed = sh.merge_reports(_two_shard_plan(), {0: _report(killed=4)})
    assert failed
    assert "No report from shard 1" in summary


def test_merge_fails_when_no_shard_checked_a_mutant(sh: Any) -> None:
    _, failed = sh.merge_reports(_two_shard_plan(), {
        0: _report(not_checked=4), 1: _report(not_checked=4),
    })
    assert failed


def test_read_reports_maps_artifact_directories_to_shards(sh: Any, tmp_path: Path) -> None:
    (tmp_path / "mutation-report-0").mkdir()
    (tmp_path / "mutation-report-0" / "mutation-report.txt").write_text("x", encoding="utf-8")
    (tmp_path / "mutation-report-1").mkdir()
    got = sh.read_reports([tmp_path / "mutation-report-0", tmp_path / "mutation-report-1"])
    assert got == {0: "x", 1: None}


# --- the workflow -------------------------------------------------------------


def _workflow() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _job(name: str) -> str:
    """The text of one top-level job, from its key to the next job's key."""
    text = _workflow()
    match = re.search(rf"^  {name}:\n(.*?)(?=^  [a-z]+:\n|\Z)", text, re.MULTILINE | re.DOTALL)
    assert match is not None, f"no job {name!r}"
    return match.group(1)


def test_the_merge_pattern_is_the_weekly_guards_pattern(sh: Any) -> None:
    """One definition of a result line, or the merge and the guard drift."""
    patterns = set(re.findall(r"MUTANT_LINE='([^']*)'", _job("mutmut")))
    assert patterns == {sh.MUTANT_LINE.pattern}


def test_the_matrix_is_read_from_the_plan(sh: Any) -> None:
    mutmut = _job("mutmut")
    assert "shard: ${{ fromJSON(needs.plan.outputs.shards) }}" in mutmut
    assert "fail-fast: false" in mutmut
    assert "shards: ${{ steps.plan.outputs.shards }}" in _job("plan")


def test_each_shard_is_time_boxed_inside_its_job_limit() -> None:
    """The time box, not the job limit, must end a long shard.

    A job cancelled at its limit uploads no report, which is how every
    weekly run before #1747 ended. The box sends SIGINT, which mutmut
    handles by keeping each result it has saved.
    """
    mutmut = _job("mutmut")
    box = re.search(r"MUTMUT_TIME_BOX: (\d+)m", mutmut)
    limit = re.search(r"timeout-minutes: (\d+)", mutmut)
    kill = re.search(r"--kill-after=(\d+)m", mutmut)
    assert box and limit and kill
    assert "timeout --signal=INT" in mutmut
    assert int(box.group(1)) + int(kill.group(1)) + 10 <= int(limit.group(1)) <= 360


def test_the_shard_guard_fails_only_on_a_run_that_checked_nothing() -> None:
    mutmut = _job("mutmut")
    guard = mutmut.split("- name: Assert the run actually produced results", 1)[1]
    guard = guard.split("- uses:", 1)[0]
    assert guard.count("exit 1") == 2
    assert '[ "${total}" -eq 0 ]' in guard
    assert '[ "${unchecked}" -eq "${total}" ]' in guard
    assert "::warning title=Partial mutation run::" in guard


def test_the_report_job_runs_after_failed_shards_and_merges_them() -> None:
    report = _job("report")
    # Job level, four spaces in: the upload step has its own `if: always()`.
    assert re.search(r"^    if: always\(\) && ", report, re.MULTILINE)
    assert "needs: [plan, mutmut]" in report
    assert "scripts/mutation_shards.py merge" in report
    assert "pattern: mutation-report-*" in report


# --- the real tree --------------------------------------------------------------


@pytest.mark.source_scan
def test_every_excluded_unit_exists_in_the_real_tree(sh: Any) -> None:
    """Caught here in the ordinary suite, not by the weekly plan failing."""
    for path, units in sh.EXCLUDED.items():
        source = (_REPO / path).read_text(encoding="utf-8")
        names = {
            n.name for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = set(units) - names
        assert not missing, f"{path}: {sorted(missing)} no longer exist"


def test_count_mutants_matches_mutmuts_own_mutant_names(
    sh: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per unit, the planner's count is what mutmut 3.8.0 generates."""
    pytest.importorskip("mutmut")
    from mutmut.mutation.file_mutation import (  # type: ignore[import-not-found]
        combine_mutations_to_source,
        create_mutations,
    )

    source = (
        "def f(a, b):\n    return a + b > 1\n\n\n"
        "class Box:\n    def grow(self, n):\n        return n * 2 - 1\n"
    )
    root = _tree(tmp_path)
    monkeypatch.chdir(root)
    counts = sh.count_mutants("src/aelfrice/a.py", source)
    module, mutations, ic, ifn = create_mutations("src/aelfrice/a.py", source, None, None)
    names = combine_mutations_to_source(module, mutations, ic, ifn).mutant_names
    # `combine` names mutants without the module prefix: `x_f__mutmut_1`.
    assert counts["f"] == sum(1 for n in names if n.startswith("x_f__mutmut_"))
    assert counts["Box.grow"] == sum(
        1 for n in names if n.startswith("xǁBoxǁgrow__mutmut_")
    )
    assert sum(counts.values()) == len(names) > 0
