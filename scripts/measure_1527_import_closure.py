#!/usr/bin/env python3
"""Re-derive the #1527 `aelfrice` module-closure counts, for any git refs.

Usage:

    uv run python scripts/measure_1527_import_closure.py
    uv run python scripts/measure_1527_import_closure.py --ref github/main --ref HEAD
    uv run python scripts/measure_1527_import_closure.py --dry-run

Every count published for #1527 -- in `CHANGELOG/unreleased/`, in the
`hook.py` deferral comment, and as the ceilings pinned by
`tests/test_hook_import_cost_1351.py` -- comes out of this script. Run it to
check any of them. It exits non-zero if a probe fails.

Wall-clock is deliberately absent. Three separate paired cold-subprocess runs
against #1527's two arms returned medians about 10 ms apart, and one of them
reported a bootstrap interval that the other two fell outside — so a
millisecond figure from a loaded developer machine does not reproduce and is
not worth publishing. The module closure is deterministic, so that is what
this script measures.

How it measures: each ref is exported with `git archive` into a temporary
directory, so the tree under test is never the installed package, and every
probe is a fresh interpreter with `PYTHONPATH` pointed at that export. Each
probe also strips every `AELFRICE_*` / `AELF_*` variable and repoints `HOME`,
the dotdir and the DB at a scratch directory before importing anything: the
lane resolvers are env-first, so an ambient variable changes which lanes run
and therefore which modules load, and opening a `MemoryStore` is a write.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# (label, import target). `None` for a module absent from a ref is reported as
# "n/a" rather than as a failure -- `rebuild_log` does not exist before #1527.
_IMPORT_TARGETS: tuple[tuple[str, str], ...] = (
    ("import aelfrice.hook", "aelfrice.hook"),
    ("import aelfrice.context_rebuilder", "aelfrice.context_rebuilder"),
    ("import aelfrice.rebuild_log", "aelfrice.rebuild_log"),
    ("import aelfrice.query_understanding", "aelfrice.query_understanding"),
)

_ENV_PRELUDE = """
import os
import sys

for _k in [k for k in os.environ if k.startswith(("AELFRICE_", "AELF_"))]:
    del os.environ[_k]

_tmp = {tmp!r}
os.environ["HOME"] = _tmp
os.environ["AELFRICE_DOTDIR"] = os.path.join(_tmp, ".aelfrice")
os.environ["AELFRICE_DB"] = os.path.join(_tmp, "memory.db")
os.environ["AELF_NO_UPDATE_CHECK"] = "1"
os.chdir(_tmp)


def _n():
    return len([
        m for m in sys.modules
        if m == "aelfrice" or m.startswith("aelfrice.")
    ])
"""

_IMPORT_PROBE = _ENV_PRELUDE + """
import {target}
print(_n())
"""

_NAMES_PROBE = _ENV_PRELUDE + """
{imports}
print(",".join(sorted(
    m for m in sys.modules if m == "aelfrice" or m.startswith("aelfrice.")
)))
"""

# The one retrieval-adjacent module `hook.py` still imports at module scope,
# and the leaf module that also binds it. `--marginal` measures what deferring
# it would buy, so the two comments about it do not have to guess.
_EAGER_EXCEPTION = "aelfrice.query_understanding"
_EAGER_EXCEPTION_VIA = "aelfrice.rebuild_log"

# The gate-skipped fire. `"ok"` is under `_MIN_PROMPT_LEN`, so the prompt-shape
# gate returns `trivial:short` and no retrieval runs. The gate reason is
# printed so the caller can reject a reading taken on the wrong path.
_FIRE_PROBE = _ENV_PRELUDE + """
import io
import json
import pathlib

import aelfrice.hook as hook

rc = hook.user_prompt_submit(
    stdin=io.StringIO(
        json.dumps({{"prompt": "ok", "session_id": "s1", "cwd": _tmp}})
    ),
    stdout=io.StringIO(),
    stderr=io.StringIO(),
)
gate = ""
for _p in sorted(pathlib.Path(_tmp).rglob("*.jsonl")):
    for _line in _p.read_text(encoding="utf-8").splitlines():
        if not _line.strip():
            continue
        _rec = json.loads(_line)
        if _rec.get("hook") == "user_prompt_submit":
            gate = str(_rec.get("prompt_shape_gate_skip") or "")
print(json.dumps({{"rc": rc, "gate": gate, "n": _n()}}))
"""


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _export(ref: str, root: Path, dest: Path) -> None:
    """Export `ref`'s tree into `dest`."""
    tar = dest / "tree.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(tar), ref],
        cwd=root,
        check=True,
    )
    src = dest / "src-tree"
    src.mkdir()
    subprocess.run(["tar", "-xf", str(tar), "-C", str(src)], check=True)
    tar.unlink()


def _run(src_root: Path, code: str) -> str:
    """Run one cold probe against an exported tree. Returns stdout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root / "src")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        return "!" + proc.stderr.strip().splitlines()[-1]
    return proc.stdout.strip()


def _measure(ref: str, root: Path) -> dict[str, str]:
    """Return every count for one ref."""
    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="i1527-") as td:
        dest = Path(td)
        _export(ref, root, dest)
        tree = dest / "src-tree"
        for label, target in _IMPORT_TARGETS:
            scratch = dest / ("scratch-" + target.replace(".", "-"))
            scratch.mkdir()
            out = _run(
                tree,
                _IMPORT_PROBE.format(tmp=str(scratch), target=target),
            )
            if out.startswith("!"):
                results[label] = (
                    "n/a" if "No module named" in out else out[:60]
                )
            else:
                results[label] = out
        scratch = dest / "scratch-fire"
        scratch.mkdir()
        out = _run(tree, _FIRE_PROBE.format(tmp=str(scratch)))
        if out.startswith("!"):
            results["driven gate-skipped fire"] = out[:60]
        else:
            row = json.loads(out)
            gate = row["gate"]
            if row["rc"] != 0 or gate != "trivial:short":
                results["driven gate-skipped fire"] = (
                    f"UNUSABLE rc={row['rc']} gate={gate!r}"
                )
            else:
                results["driven gate-skipped fire"] = str(row["n"])
    return results


def _module_scope_imports(tree: Path) -> list[str]:
    """Every `aelfrice.*` module `hook.py` imports at module scope.

    Derived from the AST rather than from a hand-kept list, so it cannot go
    stale against the file. Function-scope imports -- the deferred ones -- are
    excluded by not descending into `def` or `class`.
    """
    src = (tree / "src" / "aelfrice" / "hook.py").read_text(encoding="utf-8")
    found: list[str] = []

    def _visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("aelfrice"):
                    found.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("aelfrice"):
                        found.append(alias.name)
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    _visit(inner)  # pyright: ignore[reportUnknownArgumentType]
            for handler in getattr(node, "handlers", []):
                _visit(handler.body)

    _visit(ast.parse(src).body)
    return sorted(set(found))


def _measure_marginal(ref: str, root: Path) -> str:
    """Report what deferring `_EAGER_EXCEPTION` would buy, two ways.

    The binding is one decision made in two files, so there are two honest
    marginals and they differ:

    * from `hook.py` alone -- the closure of every *other* module-scope import
      of `hook.py`, `_EAGER_EXCEPTION_VIA` included, subtracted from it;
    * from both files -- `_EAGER_EXCEPTION_VIA` excluded from that comparison
      set as well, which is the figure `rebuild_log.py`'s docstring quotes.

    The comparison set is read out of `hook.py`'s AST, so neither figure can go
    stale against a changed import block.
    """
    with tempfile.TemporaryDirectory(prefix="i1527m-") as td:
        dest = Path(td)
        _export(ref, root, dest)
        tree = dest / "src-tree"
        eager = _module_scope_imports(tree)
        if _EAGER_EXCEPTION not in eager:
            return f"{_EAGER_EXCEPTION} is not a module-scope import of hook.py"
        others = [m for m in eager if m != _EAGER_EXCEPTION]
        others_no_leaf = [m for m in others if m != _EAGER_EXCEPTION_VIA]

        def _names(mods: list[str], tag: str) -> set[str]:
            scratch = dest / ("scratch-" + tag)
            scratch.mkdir()
            out = _run(
                tree,
                _NAMES_PROBE.format(
                    tmp=str(scratch),
                    imports="\n".join(f"import {m}" for m in mods),
                ),
            )
            if out.startswith("!"):
                raise RuntimeError(out)
            return set(out.split(","))

        own = _names([_EAGER_EXCEPTION], "qu")
        from_hook = sorted(own - _names(others, "others"))
        from_both = sorted(own - _names(others_no_leaf, "others-no-leaf"))
    return (
        f"deferring {_EAGER_EXCEPTION} from hook.py's module scope alone would "
        f"save {len(from_hook)} aelfrice module(s): {from_hook or 'none'}\n"
        f"  ({_EAGER_EXCEPTION_VIA} binds it too, so its closure arrives "
        "either way)\n"
        f"deferring it from both files would save {len(from_both)}: "
        f"{from_both or 'none'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--ref",
        action="append",
        dest="refs",
        help="git ref to measure; repeatable. Default: github/main then HEAD.",
    )
    _ = parser.add_argument(
        "--marginal",
        action="store_true",
        help=(
            "also report what deferring hook.py's one remaining eager "
            "retrieval-adjacent import would save. Uses the last ref given."
        ),
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be measured and exit 0 without running probes.",
    )
    args = parser.parse_args(argv)
    refs: list[str] = args.refs or ["github/main", "HEAD"]
    root = _repo_root()
    labels = [label for label, _ in _IMPORT_TARGETS]
    labels.append("driven gate-skipped fire")

    if args.dry_run:
        print(f"repo: {root}")
        print(f"refs: {', '.join(refs)}")
        for label in labels:
            print(f"  would count aelfrice modules after: {label}")
        if args.marginal:
            print(f"  would report the marginal cost of {_EAGER_EXCEPTION}")
        return 0

    table: dict[str, dict[str, str]] = {}
    for ref in refs:
        table[ref] = _measure(ref, root)

    width = max(len(label) for label in labels)
    header = " " * width + "".join(f"  {ref:>18}" for ref in refs)
    print("aelfrice modules in sys.modules after each of:")
    print(header)
    failed = False
    for label in labels:
        cells = []
        for ref in refs:
            value = table[ref][label]
            if not (value.isdigit() or value == "n/a"):
                failed = True
            cells.append(f"  {value:>18}")
        print(f"{label:<{width}}" + "".join(cells))
    if args.marginal:
        print()
        print(_measure_marginal(refs[-1], root))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
