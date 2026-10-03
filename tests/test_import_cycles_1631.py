"""#1631: `aelfrice.hook` sits in no import cycle.

Asserted over the whole first-party graph by strongly connected component,
not by naming pairs. A pair test passes while a five-module component sits
next to it, which is how `hook -> cli -> doctor -> hook` went unnoticed:
every edge on it was function-local, so nothing failed at import.

The graph comes from `scripts/import_cycles.py`, the same producer the
published before/after figures come from, so the test and the figures cannot
disagree about what an edge is.

The other cycles in the package are not this test's subject. Nothing here
asserts them away; they are dispositioned separately.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "import_cycles.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("import_cycles", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("import_cycles", mod)
    spec.loader.exec_module(mod)
    return mod


_IC = _load_script()


def _graph() -> dict[str, set[str]]:
    return _IC.build_graph(_IC.read_sources_from_tree(_REPO))


@pytest.mark.timeout(60)
def test_hook_is_in_no_import_cycle() -> None:
    """The gate: no component of size two or more contains `aelfrice.hook`.

    Falsifiable by putting back any one edge #1631 removed, for example
    `from aelfrice.hook import memory_block_enabled` inside `doctor.py`, or
    `from aelfrice.hook import _escape_attr` inside `provenance_render.py`.
    """
    graph = _graph()
    assert "aelfrice.hook" in graph, "the scan did not see aelfrice.hook"
    for component in _IC.cycles(graph):
        assert "aelfrice.hook" not in component, (
            "aelfrice.hook is in an import cycle with "
            f"{len(component) - 1} other module(s): {', '.join(component)}. "
            "Re-derive with: uv run python scripts/import_cycles.py"
        )


@pytest.mark.timeout(60)
def test_the_scan_sees_the_hook_edges_it_must_see() -> None:
    """Guard the guard: a scan that misses function-local edges proves nothing.

    `hook -> cli` (#1626) and `cli -> doctor` are both function-local
    imports. If the scan only read module scope, the graph would have no
    path out of `hook` toward `doctor`, and the gate above would pass on the
    unfixed tree.
    """
    graph = _graph()
    assert "aelfrice.cli" in graph["aelfrice.hook"]
    assert "aelfrice.doctor" in graph["aelfrice.cli"]
    assert len(graph) > 100, f"the scan saw only {len(graph)} modules"


def test_the_scc_finder_reports_the_pre_1631_component() -> None:
    """The six edges #1631 measured form one five-module component.

    A synthetic graph, so the finder is checked against a known answer
    rather than against whatever the tree currently says.
    """
    graph: dict[str, set[str]] = {
        "aelfrice.hook": {"aelfrice.cli", "aelfrice.provenance_render"},
        "aelfrice.cli": {"aelfrice.doctor"},
        "aelfrice.doctor": {"aelfrice.hook", "aelfrice.hook_search_tool"},
        "aelfrice.hook_search_tool": {"aelfrice.hook"},
        "aelfrice.provenance_render": {"aelfrice.hook"},
        "aelfrice.leaf": set(),
    }
    assert _IC.cycles(graph) == [
        [
            "aelfrice.cli",
            "aelfrice.doctor",
            "aelfrice.hook",
            "aelfrice.hook_search_tool",
            "aelfrice.provenance_render",
        ]
    ]
    graph["aelfrice.doctor"] = {"aelfrice.hook_search_tool"}
    graph["aelfrice.hook_search_tool"] = set()
    graph["aelfrice.provenance_render"] = set()
    assert _IC.cycles(graph) == []


def test_every_import_form_becomes_an_edge() -> None:
    """Module scope, function scope, `from pkg import module`, and relative."""
    sources = {
        "src/aelfrice/__init__.py": "",
        "src/aelfrice/a.py": "def f():\n    from aelfrice import b\n",
        "src/aelfrice/b.py": "import aelfrice.pkg.c\n",
        "src/aelfrice/pkg/__init__.py": "",
        "src/aelfrice/pkg/c.py": "from . import d\nfrom ..a import f\n",
        "src/aelfrice/pkg/d.py": "from aelfrice.a import f as g\n",
    }
    graph = _IC.build_graph(sources)
    assert graph["aelfrice.a"] == {"aelfrice.b"}
    assert graph["aelfrice.b"] == {"aelfrice.pkg.c"}
    assert graph["aelfrice.pkg.c"] == {"aelfrice.pkg.d", "aelfrice.a"}
    assert graph["aelfrice.pkg.d"] == {"aelfrice.a"}
    assert _IC.cycles(graph) == [
        ["aelfrice.a", "aelfrice.b", "aelfrice.pkg.c", "aelfrice.pkg.d"]
    ]


def test_assert_acyclic_exits_nonzero_for_a_module_in_a_cycle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The script's own gate fails on a module that is in a cycle.

    Uses a module from one of the cycles #1631 leaves alone, so the check
    has a real positive case on the live tree without asserting anything
    about whether that cycle should exist.
    """
    in_cycle = sorted(m for c in _IC.cycles(_graph()) for m in c)
    if not in_cycle:
        pytest.skip("the tree has no import cycle left to probe with")
    assert _IC.main(["--assert-acyclic", in_cycle[0]]) == 1
    assert _IC.main(["--assert-acyclic", "aelfrice.hook"]) == 0
    capsys.readouterr()
