"""The weekly mutation run is planned into shards that each finish (#1747).

The weekly job ran mutmut over the whole tree in one job and hit its time
limit during mutant generation every week, with no mutant ever run.
`scripts/mutation_shards.py` cuts the tree into shards, scopes one shard's
checkout, and merges the shards' reports; `mutation.yml` runs it as a matrix.
These tests pin the planner, the scoping, the merge, and the workflow wiring.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib.util
import json
import os
import re
import sys
import textwrap
import time
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


def test_merge_warns_when_a_shard_lists_other_than_its_plan(sh: Any) -> None:
    summary, failed = sh.merge_reports(_two_shard_plan(), {
        0: _report(killed=4), 1: _report(killed=3),
    })
    assert not failed
    assert "Shard 1 listed a different number of mutants" in summary
    clean, _ = sh.merge_reports(_two_shard_plan(), {
        0: _report(killed=4), 1: _report(killed=4),
    })
    assert "listed a different number" not in clean


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


def test_merge_does_not_count_statuses_that_judge_nothing(sh: Any) -> None:
    """`no tests` or `suspicious` on every mutant is a run that tested nothing."""
    summary, failed = sh.merge_reports(_two_shard_plan(), {
        0: _report(suspicious=2, no_tests=2), 1: _report(no_tests=4),
    })
    assert failed
    assert "0 of 8 planned mutants checked" in summary
    assert sh.tally(_report(timeout=1, killed=1)).checked == 2


def test_plan_leaves_out_what_do_not_mutate_names(sh: Any, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    config = root / "pyproject.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + 'do_not_mutate = ["src/aelfrice/sub/*"]\n',
        encoding="utf-8",
    )
    plan = sh.plan_tree(root, 2, 100, _units_counter)
    files = {p for s in plan["shards"] for p in s["files"]}
    assert files == {"src/aelfrice/a.py", "src/aelfrice/cli.py"}
    assert plan["mutants"] == 4


def test_plan_refuses_function_name_patterns_it_cannot_model(
    sh: Any, tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    config = root / "pyproject.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + 'do_not_mutate_patterns = ["x_*"]\n',
        encoding="utf-8",
    )
    with pytest.raises(sh.PlanError, match="do_not_mutate_patterns"):
        sh.plan_tree(root, 2, 100, _units_counter)


# --- the time box ---------------------------------------------------------------

# Stands in for mutmut: rewrites its `.meta` in place, non-atomically, the
# way mutmut 3.8.0's `save()` does, until it is stopped.
_WRITER = textwrap.dedent("""\
    import json, signal, sys, time
    meta, mode = sys.argv[1], sys.argv[2]
    if mode == "ignore-int":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if mode == "orphan":
        import subprocess
        child = subprocess.Popen([sys.executable, "-c",
            "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)"])
        with open(meta + ".pid", "w") as f:
            f.write(str(child.pid))
    if mode == "self-kill":
        import os
        os.kill(os.getpid(), signal.SIGTERM)
    if mode == "truncate-on-int":
        def cut(*_):
            with open(meta, "w") as f:
                f.write('{"exit_code_by_key": {"m"')
            sys.exit(1)
        signal.signal(signal.SIGINT, cut)
    n = 0
    while True:
        n += 1
        with open(meta, "w") as f:
            f.write(json.dumps({"exit_code_by_key": {"m": n}}))
        if mode == "finish" and n == 3:
            sys.exit(7)
        time.sleep(0.02)
""")


def _writer(tmp_path: Path, mode: str) -> tuple[list[str], Path]:
    script = tmp_path / "writer.py"
    script.write_text(_WRITER, encoding="utf-8")
    meta_root = tmp_path / "mutants"
    (meta_root / "src").mkdir(parents=True)
    meta = meta_root / "src" / "a.py.meta"
    return [sys.executable, str(script), str(meta), mode], meta_root


def _boxed(sh: Any, tmp_path: Path, mode: str, box_s: float, grace_s: float) -> Any:
    command, meta_root = _writer(tmp_path, mode)
    return sh.run_boxed(
        command, box_s=box_s, grace_s=grace_s, every_s=0.05,
        meta_root=meta_root, store=tmp_path / "copies",
    )


def test_a_run_that_finishes_in_time_keeps_its_own_status(sh: Any, tmp_path: Path) -> None:
    result = _boxed(sh, tmp_path, "finish", box_s=20, grace_s=5)
    assert (result.returncode, result.timed_out, result.killed) == (7, False, False)


def test_the_time_box_stops_the_run_with_sigint(sh: Any, tmp_path: Path) -> None:
    result = _boxed(sh, tmp_path, "run", box_s=0.5, grace_s=10)
    assert result.timed_out and not result.killed
    assert result.removed == []
    json.loads((tmp_path / "mutants/src/a.py.meta").read_text(encoding="utf-8"))


def test_a_meta_the_stop_signal_truncated_is_restored(sh: Any, tmp_path: Path) -> None:
    """The failure GNU `timeout` left behind: a signal mid-save cuts the
    `.meta` short, and `mutmut results` then fails on it."""
    result = _boxed(sh, tmp_path, "truncate-on-int", box_s=0.5, grace_s=10)
    assert result.timed_out
    assert result.restored == ["src/a.py.meta"]
    restored = json.loads((tmp_path / "mutants/src/a.py.meta").read_text(encoding="utf-8"))
    assert restored["exit_code_by_key"]["m"] >= 1


def test_a_run_that_ignores_sigint_is_killed_after_the_grace(sh: Any, tmp_path: Path) -> None:
    result = _boxed(sh, tmp_path, "ignore-int", box_s=0.5, grace_s=0.5)
    assert result.timed_out and result.killed


def test_a_truncated_meta_is_put_back_from_its_last_whole_copy(sh: Any, tmp_path: Path) -> None:
    meta_root = tmp_path / "mutants"
    meta = meta_root / "src" / "a.py.meta"
    meta.parent.mkdir(parents=True)
    meta.write_text('{"exit_code_by_key": {"m": 1}}', encoding="utf-8")
    copies = sh.MetaSnapshots(meta_root, tmp_path / "copies")
    assert copies.take() == 1
    meta.write_text('{"exit_code_by_key": {"m"', encoding="utf-8")  # cut mid-write
    assert copies.take() == 0, "a file that doesn't parse is never copied"
    assert copies.repair() == (["src/a.py.meta"], [])
    assert json.loads(meta.read_text(encoding="utf-8")) == {"exit_code_by_key": {"m": 1}}


def test_a_changed_meta_is_copied_again_so_a_restore_is_recent(
    sh: Any, tmp_path: Path,
) -> None:
    """Copying a file only once would restore it to its first state and
    drop every result saved since."""
    meta_root = tmp_path / "mutants"
    meta_root.mkdir()
    first, second = meta_root / "a.py.meta", meta_root / "b.py.meta"
    first.write_text('{"n": 1}', encoding="utf-8")
    second.write_text('{"n": 1}', encoding="utf-8")
    copies = sh.MetaSnapshots(meta_root, tmp_path / "copies")
    assert copies.take() == 2, "every changed file, in one pass"
    assert copies.take() == 0, "an unchanged file isn't copied again"
    first.write_text('{"n": 22}', encoding="utf-8")  # a different size, so a new key
    assert copies.take() == 1
    first.write_text('{"n"', encoding="utf-8")
    copies.repair()
    assert json.loads(first.read_text(encoding="utf-8")) == {"n": 22}


def test_workers_left_behind_are_killed_with_the_group(sh: Any, tmp_path: Path) -> None:
    result = _boxed(sh, tmp_path, "orphan", box_s=0.5, grace_s=10)
    assert result.timed_out
    pid = int((tmp_path / "mutants/src/a.py.meta.pid").read_text(encoding="utf-8"))
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail(f"worker {pid} outlived the run")


def test_a_run_killed_by_a_signal_exits_128_plus_the_signal(
    sh: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, _ = _writer(tmp_path, "self-kill")
    monkeypatch.chdir(tmp_path)
    assert sh.main(["run", "--time-box-minutes", "1", "--", *command]) == 128 + 15


def test_an_interrupted_supervisor_stops_the_run_and_repairs_first(
    sh: Any, tmp_path: Path,
) -> None:
    """A cancelled job interrupts the supervisor, not mutmut's own group."""
    command, meta_root = _writer(tmp_path, "truncate-on-int")
    calls = iter(range(1000))

    def clock() -> float:
        n = next(calls)
        if n == 20:
            raise KeyboardInterrupt
        return 0.0

    with pytest.raises(KeyboardInterrupt):
        sh.run_boxed(
            command, box_s=60, grace_s=10, every_s=0.05,
            meta_root=meta_root, store=tmp_path / "copies", clock=clock,
        )
    meta = meta_root / "src" / "a.py.meta"
    # Stopped: the writer rewrites the file every 20 ms while it runs, and
    # truncates it on SIGINT, so a whole file that stays put means it was
    # stopped and the truncation put back.
    before = meta.read_bytes()
    time.sleep(0.2)
    assert meta.read_bytes() == before, "the command still runs after the interrupt"
    assert json.loads(before)["exit_code_by_key"]["m"] >= 1


def test_a_truncated_meta_with_no_copy_is_removed_not_left_to_crash_results(
    sh: Any, tmp_path: Path,
) -> None:
    meta_root = tmp_path / "mutants"
    meta = meta_root / "a.py.meta"
    meta_root.mkdir()
    meta.write_text("{", encoding="utf-8")
    assert sh.MetaSnapshots(meta_root, tmp_path / "copies").repair() == ([], ["a.py.meta"])
    assert not meta.exists()


def test_run_exits_124_when_the_time_box_ends_the_run(
    sh: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, _ = _writer(tmp_path, "run")
    monkeypatch.chdir(tmp_path)
    code = sh.main([
        "run", "--time-box-minutes", "0.01", "--snapshot-seconds", "0.05", "--", *command,
    ])
    assert code == sh.TIMED_OUT == 124


def test_run_dry_run_runs_nothing(sh: Any, tmp_path: Path) -> None:
    command, _ = _writer(tmp_path, "finish")
    assert sh.main(["run", "--time-box-minutes", "1", "--dry-run", "--", *command]) == 0
    assert not (tmp_path / "mutants/src/a.py.meta").exists()


# --- the workflow -------------------------------------------------------------


def _workflow() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _job(name: str) -> str:
    """The text of one top-level job, from its key to the next job's key."""
    text = _workflow()
    match = re.search(rf"^  {name}:\n(.*?)(?=^  [a-z]+:\n|\Z)", text, re.MULTILINE | re.DOTALL)
    assert match is not None, f"no job {name!r}"
    return match.group(1)


def _step(job: str, name: str) -> str:
    """One step of `job`, from its `- name:` line to the next step."""
    body = _job(job).split(f"- name: {name}\n", 1)
    assert len(body) == 2, f"no step {name!r} in job {job!r}"
    return re.split(r"^      - ", body[1], maxsplit=1, flags=re.MULTILINE)[0]


def test_the_merge_pattern_is_the_weekly_guards_pattern(sh: Any) -> None:
    """One definition of a result line, or the merge and the guard drift."""
    patterns = set(re.findall(r"MUTANT_LINE='([^']*)'", _job("mutmut")))
    assert patterns == {sh.MUTANT_LINE.pattern}


def test_the_matrix_is_read_from_the_plan() -> None:
    mutmut = _job("mutmut")
    assert "shard: ${{ fromJSON(needs.plan.outputs.shards) }}" in mutmut
    assert "fail-fast: false" in mutmut
    assert "shards: ${{ steps.plan.outputs.shards }}" in _job("plan")


def test_each_shard_scopes_and_uploads_its_own_shard() -> None:
    """Every matrix job applying shard 0, or uploading under one name, would
    still produce a green run with one shard's worth of results."""
    scope = _step("mutmut", "Scope mutmut to this shard")
    assert '--shard "${{ matrix.shard }}"' in scope
    assert "name: mutation-report-${{ matrix.shard }}" in _job("mutmut")


def test_each_shard_runs_under_the_time_box_inside_its_job_limit(sh: Any) -> None:
    """The time box, not the job limit, must end a long shard.

    A job cancelled at its limit uploads no report, which is how every
    weekly run before #1747 ended. The box is `mutation_shards.py run`,
    not GNU `timeout`, which can leave mutmut's results truncated.
    """
    mutmut = _job("mutmut")
    run = _step("mutmut", "Run mutation tests")
    box = re.search(r"MUTMUT_TIME_BOX_MINUTES: (\d+)", run)
    limit = re.search(r"^    timeout-minutes: (\d+)", mutmut, re.MULTILINE)
    assert box and limit
    assert "scripts/mutation_shards.py run" in run
    assert '--time-box-minutes "${MUTMUT_TIME_BOX_MINUTES}" --' in run
    assert "timeout --signal" not in run
    assert "--grace-minutes" not in run, "the job budget below assumes the default"
    budget = int(box.group(1)) + sh.DEFAULT_GRACE_MINUTES + 10
    assert budget <= int(limit.group(1)) <= 360


def test_the_run_step_keeps_going_to_the_report_and_records_its_status() -> None:
    """Without `|| status=$?`, `bash -e` ends the step at the time box's 124,
    before `mutmut results` writes the report the shard uploads."""
    run = _step("mutmut", "Run mutation tests")
    assert re.search(r"\.venv/bin/mutmut run --max-children 4 \|\| status=\$\?$", run, re.MULTILINE)
    assert 'echo "status=${status}" >> "${GITHUB_OUTPUT}"' in run
    assert 'if [ "${status}" -eq 124 ]; then' in run
    assert run.index("|| status=$?") < run.index(".venv/bin/mutmut results --all=true")


def test_the_shard_runs_with_the_project_environment_on_path() -> None:
    """`.venv/bin/mutmut` bypasses `uv run`, which is what puts `aelf` on
    PATH; the first real run failed every shard's stats pass on that."""
    mutmut = _job("mutmut")
    step = _step("mutmut", "Put the project's environment on PATH")
    assert 'echo "${GITHUB_WORKSPACE}/.venv/bin" >> "${GITHUB_PATH}"' in step
    assert 'echo "VIRTUAL_ENV=${GITHUB_WORKSPACE}/.venv" >> "${GITHUB_ENV}"' in step
    assert mutmut.index("Put the project's environment on PATH") < mutmut.index(
        "- name: Run mutation tests",
    )


def test_a_mutmut_failure_partway_fails_the_shard() -> None:
    """mutmut 3.8.0 exits 0 whether or not mutants survive, so any status
    but 0 and the time box's 124 is mutmut failing; the guard alone would
    read the checked part as a partial run."""
    step = _step("mutmut", "Fail when mutmut stopped on an error")
    assert "RUN_STATUS: ${{ steps.run.outputs.status }}" in step
    assert '0|124) echo "mutmut run ended with ${RUN_STATUS}" ;;' in step
    assert "exit 1 ;;" in step
    mutmut = _job("mutmut")
    upload = mutmut[mutmut.index("name: mutation-report-${{ matrix.shard }}") - 200:]
    assert "if: always()" in upload, "the report uploads even when the shard fails"


def test_the_plan_installs_the_pinned_mutmut_for_counting() -> None:
    plan = _step("plan", "Plan the shards")
    assert "uv run --no-project --with 'mutmut==3.8.0'" in plan
    contract = _step("plan", "Check the mutmut contract the plan relies on")
    assert "tests/test_mutation_function_scope_1632.py" in contract


def test_the_shard_guard_fails_only_on_a_run_that_judged_nothing() -> None:
    guard = _step("mutmut", "Assert the run actually produced results")
    assert guard.count("exit 1") == 2
    assert '[ "${total}" -eq 0 ]' in guard
    assert "checked=$(( killed + survived + timedout ))" in guard
    assert "timedout=$(grep -cE ': timeout$' mutation-report.txt || true)" in guard
    assert '[ "${checked}" -eq 0 ]' in guard
    assert "::warning title=Partial mutation run::" in guard


def test_the_plan_job_checks_the_mutmut_contract_before_planning() -> None:
    plan = _job("plan")
    contract = _step("plan", "Check the mutmut contract the plan relies on")
    assert "AELF_REQUIRE_MUTMUT: \"1\"" in contract
    assert "--with 'mutmut==3.8.0' pytest" in contract
    assert "tests/test_mutation_shards_1747.py" in contract
    assert plan.index("Check the mutmut contract") < plan.index("- name: Plan the shards")


def test_the_report_job_runs_after_failed_shards_and_merges_them() -> None:
    report = _job("report")
    # Job level, four spaces in: the upload step has its own `if: always()`.
    assert re.search(
        r"^    if: always\(\) && github.event_name != 'pull_request' "
        r"&& needs.plan.result == 'success'$",
        report, re.MULTILINE,
    )
    assert "needs: [plan, mutmut]" in report
    assert "pattern: mutation-report-*" in report
    assert "path: reports" in report
    merge = _step("report", "Merge the shard reports")
    assert "shopt -s nullglob" in merge
    assert "scripts/mutation_shards.py merge" in merge
    assert "reports/mutation-report-*/" in merge
    assert "|| status=$?" in merge
    assert merge.rstrip().endswith('exit "${status}"'), "the merge's failure must fail the job"
    assert "|| true" not in merge


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


def _require_mutmut() -> None:
    """Skip without mutmut, except where the workflow requires it."""
    if os.environ.get("AELF_REQUIRE_MUTMUT") == "1":
        importlib.import_module("mutmut")
    else:
        pytest.importorskip("mutmut")


def test_count_mutants_matches_mutmuts_own_mutant_names(
    sh: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per unit, the planner's count is what mutmut 3.8.0 generates."""
    _require_mutmut()
    file_mutation: Any = importlib.import_module("mutmut.mutation.file_mutation")

    source = (
        "def f(a, b):\n    return a + b > 1\n\n\n"
        "class Box:\n    def grow(self, n):\n        return n * 2 - 1\n"
    )
    root = _tree(tmp_path)
    monkeypatch.chdir(root)
    counts = sh.count_mutants("src/aelfrice/a.py", source)
    module, mutations, ic, ifn = file_mutation.create_mutations(
        "src/aelfrice/a.py", source, None, None,
    )
    names: list[str] = list(
        file_mutation.combine_mutations_to_source(module, mutations, ic, ifn).mutant_names,
    )
    # `combine` names mutants without the module prefix: `x_f__mutmut_1`.
    assert counts["f"] == sum(1 for n in names if n.startswith("x_f__mutmut_"))
    assert counts["Box.grow"] == sum(
        1 for n in names if n.startswith("xǁBoxǁgrow__mutmut_")
    )
    assert sum(counts.values()) == len(names) > 0


def test_mutmut_skips_what_do_not_mutate_names(
    sh: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan's reading of `do_not_mutate` is mutmut's own."""
    _require_mutmut()
    configuration: Any = importlib.import_module("mutmut.configuration")

    root = _tree(tmp_path)
    (root / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["src/aelfrice"]\n'
        'do_not_mutate = ["src/aelfrice/sub/*"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    monkeypatch.setattr(configuration, "_config", None)
    ignored: Any = configuration.config()._should_ignore_for_mutation
    for path in ("src/aelfrice/a.py", "src/aelfrice/sub/b.py"):
        skipped = any(
            fnmatch.fnmatch(path, p) for p in sh.do_not_mutate(root)
        )
        assert skipped == ignored(path), path
