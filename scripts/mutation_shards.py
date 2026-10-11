#!/usr/bin/env python3
"""#1747 — split the weekly mutation run into shards that each finish.

The weekly job ran mutmut over the whole tree in one job and never got past
mutant generation: every run hit the job's 240-minute limit while mutmut was
still generating, so no mutant was ever executed. Two costs did it.

* **One file.** mutmut 3.8.0 writes every mutant of a function as a full copy
  of that function, then parses the result. `cli.py` produced 17k mutants in
  a 650 MB file, and parsing it had not finished after 23 CPU-minutes. A third
  of those mutants came from `build_parser`, which is argparse help text.
* **The whole tree.** About 61k mutants, at up to a second each with four
  workers, is more than one job's time limit.

So the weekly job runs as a matrix of shards, planned here.

## What a shard is

A *piece* is a set of functions mutmut mutates as a unit (`mutation_scope`
defines units): a whole file, or, for a file with more than `--cap` mutants,
a run of consecutive units in source order. Each piece goes to the shard with
the fewest mutants so far, largest piece first. A shard mutates only its own
pieces: its files go in `only_mutate`, and in a file it holds only part of,
every other unit gets mutmut's `# pragma: no mutate block`. A smaller mutated
file is also what keeps generation fast, because mutmut's parse of the file
it generates grows faster than linearly with the file's size.

The units in `EXCLUDED` are never assigned, so no shard mutates them. A
missing excluded unit stops the plan: a renamed `build_parser` would
otherwise put its mutants back without anyone noticing.

Counts come from mutmut's own `create_mutations`, grouped by unit the way
mutmut's `combine_mutations_to_source` groups them, so the plan balances the
mutants mutmut will actually generate. This is why the workflow pins mutmut
to the version this was verified against.

## Usage

    scripts/mutation_shards.py plan --shards 8 --out plan.json [--summary plan.md]
    scripts/mutation_shards.py plan --shards 8 --dry-run
    scripts/mutation_shards.py apply --plan plan.json --shard 3
    scripts/mutation_shards.py apply --plan plan.json --shard 3 --dry-run
    scripts/mutation_shards.py merge --plan plan.json --summary merged.md DIR...
    scripts/mutation_shards.py run --time-box-minutes 330 -- mutmut run
    scripts/mutation_shards.py run --time-box-minutes 330 --dry-run -- mutmut run

`plan` counts every file under `src/aelfrice/`, except those
`[tool.mutmut] do_not_mutate` names, and writes the plan as JSON.
`--dry-run` prints the plan on stderr and writes nothing.

`apply` writes one shard's pragmas into `src/` and its `only_mutate` into
`pyproject.toml`. It edits the working tree in place, so it is meant for a
throwaway CI checkout: it refuses unless the `CI` environment variable is
`true`, which GitHub Actions sets, or `--allow-src-rewrite` is passed.
`--dry-run` prints what it would write and writes nothing.

`merge` reads each shard's `mutation-report.txt` from the given directories,
writes a Markdown summary of the whole run, and prints the totals.

`run` runs mutmut under the time box and keeps its saved results intact
when the box ends it; `run_boxed` says how. It exits with the command's own
status, or 124 when the time box ended the run.

Exit codes: 0 on success; 1 when `merge` finds a planned shard with no
report, or a run that checked no mutant at all; 2 when a count, a file, or
the plan cannot be read or does not hold; 3 when `apply` is refused; 124
when `run` reached its time box.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mutation_scope import exclude_units, mutation_units  # noqa: E402

#: `source_paths` in `[tool.mutmut]`; mutmut mutates every file below it,
#: subpackages included.
SOURCE_ROOT: Final[str] = "src/aelfrice"

#: The files the weekly job mutates.
SOURCE_GLOB: Final[str] = f"{SOURCE_ROOT}/**/*.py"

#: Units no shard mutates, by file, with the reason the summary prints.
EXCLUDED: Final[Mapping[str, Mapping[str, str]]] = {
    "src/aelfrice/cli.py": {
        "build_parser": (
            "argparse declarations and help text; its mutants made cli.py's "
            "mutant file too large for mutmut to generate within the job "
            "(#1747)"
        ),
    },
}

#: Matrix width the workflow runs. The plan must match it exactly.
DEFAULT_SHARDS: Final[int] = 8

#: Largest piece, in mutants, before a file is split by unit.
DEFAULT_CAP: Final[int] = 4000

#: The weekly job's own pattern for a result line; a test ties the two.
MUTANT_LINE: Final[re.Pattern[str]] = re.compile(r"^ *[^ :]+__mutmut_[0-9]+: ")

#: Statuses that mean a test ran against the mutant and judged it.
CHECKED_STATUSES: Final[tuple[str, ...]] = ("killed", "survived", "timeout")

#: The report file each shard uploads, in its own artifact directory.
REPORT_NAME: Final[str] = "mutation-report.txt"

PLAN_VERSION: Final[int] = 1


class PlanError(Exception):
    """The plan cannot be made, read, or applied as asked."""


# --- counting -----------------------------------------------------------------

#: (path, source) -> mutants per unit qualname, in source order.
Counter = Callable[[str, str], dict[str, int]]


def count_mutants(path: str, source: str) -> dict[str, int]:
    """Mutants mutmut generates for each unit of `source`, in source order.

    Groups `create_mutations` output by top-level function exactly as
    `combine_mutations_to_source` does in mutmut 3.8.0: a module-level
    function, or a method of a module-level class with an indented body.
    A unit with no mutants maps to 0.
    """
    # Imported here, and untyped: only the planning job installs mutmut.
    cst: Any = importlib.import_module("libcst")
    file_mutation: Any = importlib.import_module("mutmut.mutation.file_mutation")

    module, mutations, _, _ = file_mutation.create_mutations(path, source, None, None)
    per_node: dict[int, int] = {}
    for mutation in mutations:
        node = mutation.contained_by_top_level_function
        if node is not None:
            per_node[id(node)] = per_node.get(id(node), 0) + 1
    counts: dict[str, int] = {}
    statement: Any
    method: Any
    for statement in module.body:
        if isinstance(statement, cst.FunctionDef):
            counts[statement.name.value] = per_node.get(id(statement), 0)
        elif isinstance(statement, cst.ClassDef) and isinstance(
            statement.body, cst.IndentedBlock,
        ):
            for method in statement.body.body:
                if isinstance(method, cst.FunctionDef):
                    name = f"{statement.name.value}.{method.name.value}"
                    counts[name] = per_node.get(id(method), 0)
    return counts


# --- planning -----------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """Units of one file that go to the same shard.

    `units` is None when the piece is the whole file and needs no pragma.
    """

    path: str
    units: tuple[str, ...] | None
    mutants: int


def file_pieces(
    path: str,
    counts: Mapping[str, int],
    excluded: Mapping[str, str],
    cap: int,
) -> list[Piece]:
    """Split one file's units into pieces of at most `cap` mutants.

    A file under the cap with nothing excluded is one whole-file piece.
    Otherwise its units are listed, so `apply` can pragma the rest: in runs
    of consecutive units in source order, each closed before it would pass
    the cap. A unit over the cap on its own is a piece by itself. Units with
    no mutants are left out; they need no shard.
    """
    missing = sorted(set(excluded) - set(counts))
    if missing:
        raise PlanError(
            f"{path}: excluded units not found: {', '.join(missing)}. "
            "Update EXCLUDED in scripts/mutation_shards.py.",
        )
    live = [(q, n) for q, n in counts.items() if q not in excluded and n > 0]
    total = sum(n for _, n in live)
    if total == 0:
        return []
    if not excluded and total <= cap:
        return [Piece(path, None, total)]
    pieces: list[Piece] = []
    run: list[str] = []
    size = 0
    for qualname, n in live:
        if run and size + n > cap:
            pieces.append(Piece(path, tuple(run), size))
            run, size = [], 0
        run.append(qualname)
        size += n
    pieces.append(Piece(path, tuple(run), size))
    return pieces


@dataclass
class Shard:
    """One matrix job's share of the mutants."""

    index: int
    pieces: list[Piece] = field(default_factory=lambda: [])

    @property
    def mutants(self) -> int:
        return sum(p.mutants for p in self.pieces)

    def files(self) -> dict[str, list[str] | None]:
        """Each file this shard mutates: its units, or None for all of it.

        Pieces of one file in the same shard merge; their units stay in
        source order because pieces are cut in source order.
        """
        out: dict[str, list[str] | None] = {}
        for piece in sorted(self.pieces, key=lambda p: p.path):
            if piece.units is None:
                out[piece.path] = None
            else:
                kept = out.setdefault(piece.path, [])
                assert kept is not None, "a whole-file piece is never split"
                kept.extend(piece.units)
        return out


def assign(pieces: Sequence[Piece], shards: int) -> list[Shard]:
    """Largest piece first, each to the shard with the fewest mutants.

    Ties go to the lowest shard index, and pieces of equal size are taken
    in path order, so the same tree always gives the same plan.
    """
    if shards < 1:
        raise PlanError(f"--shards must be at least 1, not {shards}")
    out = [Shard(i) for i in range(shards)]
    order = sorted(
        pieces, key=lambda p: (-p.mutants, p.path, p.units or ()),
    )
    for piece in order:
        target = min(out, key=lambda s: (s.mutants, s.index))
        target.pieces.append(piece)
    return out


def do_not_mutate(root: Path) -> list[str]:
    """`[tool.mutmut] do_not_mutate` from `root`'s `pyproject.toml`.

    mutmut 3.8.0 matches each pattern with `fnmatch` against the file's
    path relative to the root, and generates nothing for a match, so the
    plan leaves those files out the same way. Function-name patterns
    (`do_not_mutate_patterns`) aren't modelled, so the plan refuses them
    rather than counting mutants mutmut would skip.
    """
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    config = tomllib.loads(text).get("tool", {}).get("mutmut", {})
    if config.get("do_not_mutate_patterns"):
        raise PlanError("do_not_mutate_patterns is set; the plan doesn't model it")
    patterns = config.get("do_not_mutate", [])
    if not isinstance(patterns, list):
        raise PlanError("[tool.mutmut] do_not_mutate must be a list")
    return [str(p) for p in patterns]  # type: ignore[misc]


def plan_tree(
    root: Path, shards: int, cap: int, counter: Counter = count_mutants,
) -> dict[str, object]:
    """Count every source file under `root` and assign the pieces."""
    if cap < 1:
        raise PlanError(f"--cap must be at least 1, not {cap}")
    pieces: list[Piece] = []
    skipped = do_not_mutate(root)
    paths = sorted(
        path
        for path in (p.relative_to(root).as_posix() for p in root.glob(SOURCE_GLOB))
        if not any(fnmatch.fnmatch(path, pattern) for pattern in skipped)
    )
    if not paths:
        raise PlanError(f"no files match {SOURCE_GLOB} under {root}")
    unknown = sorted(set(EXCLUDED) - set(paths))
    if unknown:
        raise PlanError(f"EXCLUDED names files that do not exist: {unknown}")
    for path in paths:
        source = (root / path).read_text(encoding="utf-8")
        counts = counter(path, source)
        pieces.extend(file_pieces(path, counts, EXCLUDED.get(path, {}), cap))
    assigned = assign(pieces, shards)
    return {
        "version": PLAN_VERSION,
        "cap": cap,
        "mutants": sum(p.mutants for p in pieces),
        "excluded": {path: sorted(units) for path, units in EXCLUDED.items()},
        "shards": [
            {"index": s.index, "mutants": s.mutants, "files": s.files()}
            for s in assigned
        ],
    }


def render_plan(plan: Mapping[str, object]) -> str:
    """The plan as a Markdown table, one row per shard."""
    shards = _shards(plan)
    out = [
        "### Mutation shards",
        "",
        f"{plan['mutants']} mutants in {len(shards)} shards. A file with more "
        f"than {plan['cap']} mutants is split by function.",
        "",
        "| shard | mutants | files |",
        "| --- | --- | --- |",
    ]
    for shard in shards:
        files = cast("dict[str, list[str] | None]", shard["files"])
        # Relative to the package, not the bare name: `lifecycle.py` and
        # `wonder/lifecycle.py` are different files.
        names = ", ".join(
            f"`{Path(p).relative_to(SOURCE_ROOT).as_posix()}`"
            + ("" if units is None else f" ({len(units)} functions)")
            for p, units in files.items()
        )
        out.append(f"| {shard['index']} | {shard['mutants']} | {names} |")
    out += ["", "Not mutated in any shard:", ""]
    for path, units in EXCLUDED.items():
        for unit, reason in units.items():
            out.append(f"- `{path}` `{unit}`: {reason}.")
    return "\n".join(out)


# --- applying -----------------------------------------------------------------


def _shards(plan: Mapping[str, object]) -> list[dict[str, object]]:
    if plan.get("version") != PLAN_VERSION:
        raise PlanError(f"plan version {plan.get('version')!r}, expected {PLAN_VERSION}")
    shards = plan.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PlanError("plan has no shards")
    return shards  # type: ignore[return-value]


def shard_files(plan: Mapping[str, object], index: int) -> dict[str, list[str] | None]:
    """The files shard `index` mutates, from a loaded plan."""
    shards = _shards(plan)
    matches = [s for s in shards if s.get("index") == index]
    if len(matches) != 1:
        raise PlanError(
            f"plan has {len(shards)} shards and no shard {index}; the "
            "workflow matrix and --shards must agree",
        )
    files = matches[0]["files"]
    assert isinstance(files, dict)
    return files  # type: ignore[return-value]


def only_mutate_config(text: str, paths: Sequence[str]) -> str:
    """`pyproject.toml` text with `only_mutate` set to `paths`."""
    marker = "[tool.mutmut]\n"
    if text.count(marker) != 1:
        raise PlanError("pyproject.toml must hold exactly one [tool.mutmut] table")
    if re.search(r"^only_mutate\s*=", text, re.MULTILINE):
        raise PlanError("pyproject.toml already sets only_mutate")
    entries = "".join(f'    "{p}",\n' for p in paths)
    return text.replace(marker, f"{marker}only_mutate = [\n{entries}]\n", 1)


def apply_shard(
    root: Path, plan: Mapping[str, object], index: int, *, write: bool,
) -> list[str]:
    """Pragma every unit outside shard `index` and scope `only_mutate`.

    Returns the lines of a report. Every edit is computed before any file
    is written, so a refusal leaves the tree untouched. A unit whose header
    cannot be annotated stops the run rather than widening the shard: in a
    split file that would mutate a unit another shard also mutates.
    """
    files = shard_files(plan, index)
    if not files:
        raise PlanError(f"shard {index} has no files")
    edits: dict[Path, str] = {}
    lines = [f"shard {index}: {len(files)} files"]
    for path, units in files.items():
        target = root / path
        source = target.read_text(encoding="utf-8")
        if units is None:
            lines.append(f"  {path}: whole file")
            continue
        names = {u.qualname for u in mutation_units(ast.parse(source))}
        unknown = sorted(set(units) - names)
        if unknown:
            raise PlanError(f"{path}: plan names units not in the file: {unknown}")
        rewritten, failed = exclude_units(source, set(units))
        if failed:
            raise PlanError(f"{path}: could not annotate {failed}")
        edits[target] = rewritten
        lines.append(f"  {path}: {len(units)} of {len(names)} functions")
    pyproject = root / "pyproject.toml"
    edits[pyproject] = only_mutate_config(
        pyproject.read_text(encoding="utf-8"), list(files),
    )
    if write:
        for target, text in edits.items():
            target.write_text(text, encoding="utf-8")
    return lines


# --- merging ------------------------------------------------------------------


@dataclass
class Tally:
    """Status counts from one `mutmut results --all=true` report."""

    statuses: dict[str, int] = field(default_factory=lambda: {})

    @property
    def total(self) -> int:
        return sum(self.statuses.values())

    @property
    def checked(self) -> int:
        """Mutants a test ran against and judged.

        `killed`, `survived`, and `timeout` (the tests hung on the mutant,
        which mutmut counts as caught). `no tests`, `suspicious`, `skipped`
        and the rest say nothing about the tests, so a run made only of
        them checked nothing.
        """
        return sum(self.statuses.get(s, 0) for s in CHECKED_STATUSES)

    def get(self, status: str) -> int:
        return self.statuses.get(status, 0)


def tally(report: str) -> Tally:
    """Count each status in a report, by the weekly job's own pattern."""
    out = Tally()
    for line in report.splitlines():
        if MUTANT_LINE.match(line):
            status = line.rsplit(": ", 1)[1].strip()
            out.statuses[status] = out.statuses.get(status, 0) + 1
    return out


def merge_reports(
    plan: Mapping[str, object], reports: Mapping[int, str | None],
) -> tuple[str, bool]:
    """The run's Markdown summary, and whether it failed.

    It fails when a planned shard left no report, or when no shard checked
    a single mutant. A shard that stopped partway is reported, not failed:
    each shard's own job already decided that.
    """
    shards = _shards(plan)
    rows: list[str] = []
    whole = Tally()
    missing: list[int] = []
    for shard in shards:
        index = shard["index"]
        assert isinstance(index, int)
        report = reports.get(index)
        if report is None:
            missing.append(index)
            rows.append(f"| {index} | {shard['mutants']} | no report | | | |")
            continue
        t = tally(report)
        for status, n in t.statuses.items():
            whole.statuses[status] = whole.statuses.get(status, 0) + n
        rows.append(
            f"| {index} | {shard['mutants']} | {t.total} | {t.get('killed')} "
            f"| {t.get('survived')} | {t.get('not checked')} |",
        )
    out = [
        "## Mutation, whole tree",
        "",
        f"{whole.checked} of {plan['mutants']} planned mutants checked: "
        f"{whole.get('killed')} killed, {whole.get('survived')} survived.",
        "",
    ]
    if missing:
        out += [
            "> [!WARNING]",
            f"> No report from shard {', '.join(map(str, missing))}. That "
            "shard's job log shows why.",
            "",
        ]
    out += [
        "| shard | planned | listed | killed | survived | not checked |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "Every status, all shards:",
        "",
        "```",
        *(f"{s}: {n}" for s, n in sorted(whole.statuses.items())),
        "```",
    ]
    failed = bool(missing) or whole.checked == 0
    return "\n".join(out), failed


def read_reports(dirs: Sequence[Path]) -> dict[int, str | None]:
    """Shard index -> report text, from `mutation-report-<index>` dirs."""
    out: dict[int, str | None] = {}
    for directory in dirs:
        match = re.fullmatch(r"mutation-report-(\d+)", directory.name)
        if match is None:
            raise PlanError(f"{directory}: expected a mutation-report-<shard> directory")
        report = directory / REPORT_NAME
        out[int(match.group(1))] = (
            report.read_text(encoding="utf-8") if report.is_file() else None
        )
    return out


# --- running one shard under a time box ---------------------------------------

#: Exit status `run` returns when the time box ended the run, as GNU timeout does.
TIMED_OUT: Final[int] = 124

#: How long mutmut gets to stop after SIGINT before it is killed.
DEFAULT_GRACE_MINUTES: Final[float] = 2.0


class MetaSnapshots:
    """The last copy of each mutmut `.meta` file that parsed.

    mutmut 3.8.0 saves a source file's results by rewriting its `.meta`
    file in place (`SourceFileMutationData.save`), after every mutant. A
    signal that lands mid-write leaves the file truncated, and `mutmut
    results` then fails on it. The time box has to signal mutmut, so it
    keeps copies to put back.
    """

    def __init__(self, meta_root: Path, store: Path) -> None:
        self.meta_root = meta_root
        self.store = store
        self._seen: dict[Path, tuple[int, int]] = {}

    def take(self) -> int:
        """Copy each changed `.meta` that parses. Returns how many it copied.

        A file caught mid-write doesn't parse and is skipped; the previous
        copy stays until a later pass finds the file whole.
        """
        copied = 0
        for meta in sorted(self.meta_root.rglob("*.meta")):
            try:
                stat = meta.stat()
                data = meta.read_bytes()
            except OSError:
                continue
            key = (stat.st_mtime_ns, stat.st_size)
            if self._seen.get(meta) == key or not _parses(data):
                continue
            target = self.store / meta.relative_to(self.meta_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".tmp")
            partial.write_bytes(data)
            os.replace(partial, target)
            self._seen[meta] = key
            copied += 1
        return copied

    def repair(self) -> tuple[list[str], list[str]]:
        """Put back each `.meta` that no longer parses.

        Returns the files restored from a copy and the files removed because
        there was none. A removed file's mutants drop out of `mutmut
        results`, which is reported, rather than crashing it.
        """
        restored: list[str] = []
        removed: list[str] = []
        for meta in sorted(self.meta_root.rglob("*.meta")):
            if _parses(meta.read_bytes()):
                continue
            name = meta.relative_to(self.meta_root).as_posix()
            copy = self.store / meta.relative_to(self.meta_root)
            if copy.is_file():
                partial = meta.with_name(meta.name + ".tmp")
                partial.write_bytes(copy.read_bytes())
                os.replace(partial, meta)
                restored.append(name)
            else:
                meta.unlink()
                removed.append(name)
        return restored, removed


def _parses(data: bytes) -> bool:
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


@dataclass
class RunResult:
    """How a time-boxed run ended."""

    returncode: int
    timed_out: bool
    killed: bool
    restored: list[str]
    removed: list[str]


def run_boxed(
    command: Sequence[str],
    *,
    box_s: float,
    grace_s: float,
    every_s: float,
    meta_root: Path,
    store: Path,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Run `command`, and stop it at `box_s` seconds without losing results.

    The command runs in its own process group, so the stop reaches
    mutmut's workers too. Every `every_s` seconds the `.meta` files are
    copied. At the deadline the group gets SIGINT, which mutmut handles by
    stopping its workers and keeping what it saved, and SIGKILL `grace_s`
    seconds later if it is still running: a SIGINT that lands while
    mutmut forks a worker is swallowed. Then every `.meta` that a signal
    truncated is put back from its last copy.
    """
    snapshots = MetaSnapshots(meta_root, store)
    deadline = clock() + box_s
    proc = subprocess.Popen(list(command), start_new_session=True)
    timed_out = killed = False
    while True:
        try:
            returncode = proc.wait(timeout=max(0.0, min(every_s, deadline - clock())))
            break
        except subprocess.TimeoutExpired:
            snapshots.take()
        if clock() >= deadline:
            timed_out = True
            _signal_group(proc, signal.SIGINT)
            try:
                returncode = proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                killed = True
                _signal_group(proc, signal.SIGKILL)
                returncode = proc.wait()
            # Workers that outlived mutmut would hold the runner open.
            _signal_group(proc, signal.SIGKILL)
            break
    restored, removed = snapshots.repair()
    return RunResult(returncode, timed_out, killed, restored, removed)


def _signal_group(proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass  # the group already exited


# --- command line -------------------------------------------------------------


def _load(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read plan {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="count mutants and assign shards")
    p_plan.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    p_plan.add_argument("--cap", type=int, default=DEFAULT_CAP)
    p_plan.add_argument("--out", type=Path, default=None)
    p_plan.add_argument("--summary", type=Path, default=None)
    p_plan.add_argument("--dry-run", action="store_true",
                        help="print the plan on stderr; write nothing")

    p_apply = sub.add_parser("apply", help="scope the working tree to one shard")
    p_apply.add_argument("--plan", type=Path, required=True)
    p_apply.add_argument("--shard", type=int, required=True)
    p_apply.add_argument("--dry-run", action="store_true",
                         help="print what would change; write nothing")
    p_apply.add_argument("--allow-src-rewrite", action="store_true",
                         help="let apply edit src/ outside CI (CI=true allows it)")

    p_merge = sub.add_parser("merge", help="summarise every shard's report")
    p_merge.add_argument("--plan", type=Path, required=True)
    p_merge.add_argument("--summary", type=Path, default=None)
    p_merge.add_argument("dirs", type=Path, nargs="*")

    p_run = sub.add_parser("run", help="run a command under the time box")
    p_run.add_argument("--time-box-minutes", type=float, required=True)
    p_run.add_argument("--grace-minutes", type=float, default=DEFAULT_GRACE_MINUTES)
    p_run.add_argument("--snapshot-seconds", type=float, default=60.0)
    p_run.add_argument("--meta-root", type=Path, default=Path("mutants"))
    p_run.add_argument("--store", type=Path, default=Path("mutation-meta-copies"))
    p_run.add_argument("--dry-run", action="store_true",
                       help="print what would run; run nothing")
    # Not `command`: that is the subcommand's own `dest`.
    p_run.add_argument("argv", nargs=argparse.REMAINDER,
                       help="the command to run, after --")

    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.command == "run":
        return _run(args, parser)
    try:
        if args.command == "plan":
            if not args.dry_run and args.out is None:
                parser.error("plan needs --out unless --dry-run is given")
            plan = plan_tree(root, args.shards, args.cap)
            summary = render_plan(plan)
            if args.dry_run:
                print(summary, file=sys.stderr)
                return 0
            args.out.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
            if args.summary is not None:
                args.summary.write_text(summary + "\n", encoding="utf-8")
            print(summary)
            return 0
        if args.command == "apply":
            write = not args.dry_run
            if write and os.environ.get("CI") != "true" and not args.allow_src_rewrite:
                print(
                    "mutation_shards: apply edits src/ and pyproject.toml in "
                    "place and is meant for a throwaway CI checkout. Refusing: "
                    "CI is not \"true\". Pass --allow-src-rewrite to run it "
                    "here anyway, or --dry-run.",
                    file=sys.stderr,
                )
                return 3
            lines = apply_shard(root, _load(args.plan), args.shard, write=write)
            print("\n".join(lines))
            return 0
        summary, failed = merge_reports(_load(args.plan), read_reports(args.dirs))
        if args.summary is not None:
            args.summary.write_text(summary + "\n", encoding="utf-8")
        print(summary)
        return 1 if failed else 0
    except (PlanError, OSError, SyntaxError, ValueError) as exc:
        print(f"mutation_shards: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    command = list(args.argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("run needs a command after --")
    box_s = args.time_box_minutes * 60
    if args.dry_run:
        print(f"would run {command} for at most {box_s:.0f} s", file=sys.stderr)
        return 0
    result = run_boxed(
        command,
        box_s=box_s,
        grace_s=args.grace_minutes * 60,
        every_s=args.snapshot_seconds,
        meta_root=args.meta_root,
        store=args.store,
    )
    if result.timed_out:
        how = "killed after the grace period" if result.killed else "stopped by SIGINT"
        print(f"mutation_shards: time box reached; {how}", file=sys.stderr)
    for name in result.restored:
        print(f"mutation_shards: restored {name} from its last whole copy", file=sys.stderr)
    for name in result.removed:
        print(
            f"mutation_shards: removed {name}: truncated, and no whole copy to "
            "restore; its mutants are missing from the results",
            file=sys.stderr,
        )
    if result.timed_out:
        return TIMED_OUT
    if result.returncode < 0:
        return 128 - result.returncode
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
