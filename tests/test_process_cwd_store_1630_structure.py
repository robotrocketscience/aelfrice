"""Structural guard: no in-turn consumer resolves the store from the payload cwd (#1630).

`tests/test_process_cwd_store_1630.py` drives the real hook and pins the
consumers that a default-config turn reaches. Many more consumers sit
behind flags, cadence, or sentiment, and those turns never reach them,
so a behavioural test cannot hold all of them. This module reads the
source instead and holds three rules over `hook.py` and the modules it
delegates store or state-path resolution to:

1. Every call to `db_path()` takes no arguments. `db_path()` reads the
   process cwd, and it is the only sanctioned resolver.
2. Nothing redirects resolution to another directory: no `chdir` call
   (the chdir-resolve-chdir-back shape), no `_git_common_dir` call with
   arguments, no `--git-common-dir` literal, and no `AELFRICE_DB` literal
   (setting it would redirect `db_path()`) in these modules.
3. Every `db_path()` call, every `MemoryStore(...)` construction in
   `hook.py`, and every path join that spells the store layout (`.git`,
   `aelfrice`, `memory.db`, or the `db_paths` layout constants) sits in a
   function named in an explicit inventory below. A new consumer, or a
   consumer that builds its own store path from the payload cwd, changes
   the inventory and fails until someone updates it on purpose.

What the rules allow, on purpose: the payload `cwd` is read for config
discovery (`load_user_prompt_submit_config(start=...)`,
`_load_aelfrice_toml(start=...)`), for the `<recent-work>` block, for
category matching, and as the cwd argument of the phantom and git
helpers that do not resolve a store. None of those calls `db_path()` or
`chdir`, or joins a store-layout component, so none is flagged.
`Path.cwd()` is allowed too; it is the same process cwd `db_path()`
reads.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterator
from typing import TypeGuard
from pathlib import Path

import aelfrice

PKG = Path(aelfrice.__file__).resolve().parent

# `hook.py` plus the modules it delegates store or state-path resolution
# to during a turn. A module that only uses a path it is handed
# (`hook_audit`, `session_exclusions`, `lock_gaps`) has nothing to check.
MODULES = (
    "hook.py",
    "session_ring.py",
    "injection_ledger.py",
    "feed_log.py",
    "transcript_logger.py",
    "sidecar_warm.py",
    "context_rebuilder.py",
)

_LAYOUT_LITERALS = frozenset({".git", "aelfrice", "memory.db"})
_LAYOUT_NAMES = frozenset({
    "DEFAULT_DB_DIR",
    "DEFAULT_DB_FILENAME",
    "_REPO_STORE_PARENT_DIRNAME",
})

Site = tuple[str, str]  # (module, qualname of the enclosing def)

# Every `db_path()` call, by enclosing function. A new entry is a new
# store consumer: it must call `db_path()` with no arguments.
DB_PATH_CALLS: dict[Site, int] = {
    ("context_rebuilder.py", "main"): 1,
    ("feed_log.py", "feed_path"): 1,
    ("hook.py", "_build_rebuild_block_from_payload"): 1,
    ("hook.py", "_cadence_resume_cache_path"): 1,
    ("hook.py", "_emit_user_prompt_submit_rebuild_log"): 1,
    ("hook.py", "_load_prior_ups_belief_ids"): 1,
    ("hook.py", "_maybe_log_cadence_shadow_tick"): 1,
    ("hook.py", "_maybe_phantom_opportunity_block"): 1,
    ("hook.py", "_maybe_phantom_promotion_block"): 1,
    ("hook.py", "_open_store"): 1,
    ("hook.py", "_rebuild_and_format"): 1,
    ("hook.py", "_recap_last_ts_path"): 1,
    ("hook.py", "_run_cadence_rebuild"): 1,
    ("hook.py", "_session_state_path"): 1,
    ("hook.py", "_store_handle"): 1,
    ("hook.py", "_write_command_outcome_record"): 1,
    ("hook.py", "_write_hook_audit_record"): 1,
    ("hook.py", "_write_sentiment_feedback_audit"): 1,
    ("hook.py", "_write_telemetry"): 1,
    ("hook.py", "build_lock_gap_notice"): 1,
    ("hook.py", "user_prompt_submit"): 1,
    ("injection_ledger.py", "ledger_path"): 1,
    ("session_ring.py", "_session_ring_path"): 1,
    ("sidecar_warm.py", "_record_warm_outcome"): 1,
    ("sidecar_warm.py", "warm_sidecar"): 1,
    ("transcript_logger.py", "_record_skipped_duplicate"): 1,
}

# Every store opened in `hook.py`. Each opens the path `db_path()` gave
# the same function; a new site has to show it does the same.
MEMORYSTORE_CALLS: dict[Site, int] = {
    ("hook.py", "_maybe_phantom_opportunity_block"): 1,
    ("hook.py", "_maybe_phantom_promotion_block"): 1,
    ("hook.py", "_open_store"): 1,
    ("hook.py", "_store_handle"): 1,
    ("hook.py", "user_prompt_submit"): 1,
}

# Every path join that spells the store layout itself.
LAYOUT_JOINS: dict[Site, int] = {
    # Rooted at `_git_common_dir()` with no arguments: the process cwd.
    ("transcript_logger.py", "transcripts_dir"): 1,
    # KNOWN EXCEPTION, not a store: `find_aelfrice_log(cwd)` reads the
    # transcript log under the directory it is given, and
    # `hook._read_recent_for_pre_compact` and `context_rebuilder`'s own
    # rebuild entry point give it the payload cwd (#1706 tracks both). It is
    # a read of `turns.jsonl`, not of the store or session state, and
    # predates #1630. Changing it is a behaviour change, out of scope here.
    ("context_rebuilder.py", "<module>"): 1,
    ("context_rebuilder.py", "find_aelfrice_log"): 1,
}


def _walk_with_scope(tree: ast.AST) -> Iterator[tuple[ast.AST, str]]:
    """Yield each node with the qualname of its innermost enclosing def."""
    def visit(node: ast.AST, scope: str) -> Iterator[tuple[ast.AST, str]]:
        for child in ast.iter_child_nodes(node):
            yield child, scope
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                inner = child.name if scope == "<module>" else f"{scope}.{child.name}"
                yield from visit(child, inner)
            else:
                yield from visit(child, scope)
    yield from visit(tree, "<module>")


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _db_path_aliases(tree: ast.AST) -> frozenset[str]:
    """Local names bound to `aelfrice.db_paths.db_path` anywhere in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "aelfrice.db_paths":
            names.update(a.asname or a.name for a in node.names if a.name == "db_path")
    return frozenset(names)


def _is_db_path_call(call: ast.Call, aliases: frozenset[str]) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in aliases
    # `db_paths.db_path()` counts; `store.db_path()` is the store's own method.
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "db_path"
        and isinstance(func.value, ast.Name)
        and func.value.id == "db_paths"
    )


def _is_layout_component(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value in _LAYOUT_LITERALS
    if isinstance(node, ast.Name):
        return node.id in _LAYOUT_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _LAYOUT_NAMES
    return False


def _is_div(node: ast.AST) -> TypeGuard[ast.BinOp]:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)


def _div_operands(node: ast.BinOp) -> list[ast.AST]:
    out: list[ast.AST] = []
    for side in (node.left, node.right):
        if _is_div(side):
            out.extend(_div_operands(side))
        else:
            out.append(side)
    return out


class Scan:
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.db_path_calls: Counter[Site] = Counter()
        self.memorystore_calls: Counter[Site] = Counter()
        self.layout_joins: Counter[Site] = Counter()


def scan(sources: dict[str, str]) -> Scan:
    """Apply the three rules to `{module name: source}`."""
    result = Scan()
    for module, source in sources.items():
        tree = ast.parse(source, filename=module)
        aliases = _db_path_aliases(tree)
        # Only the outermost `/` of a chain is a join; `a / b / c` is one.
        inner_divs = {
            id(side)
            for node in ast.walk(tree) if _is_div(node)
            for side in (node.left, node.right) if _is_div(side)
        }
        for node, scope in _walk_with_scope(tree):
            site = (module, scope)
            where = f"{module}:{getattr(node, 'lineno', '?')} in {scope}"
            if (
                _is_div(node)
                and id(node) not in inner_divs
                and any(_is_layout_component(o) for o in _div_operands(node))
            ):
                result.layout_joins[site] += 1
            if isinstance(node, ast.Constant) and node.value == "--git-common-dir":
                result.violations.append(f"{where}: resolves a git dir itself")
            if isinstance(node, ast.Constant) and node.value == "AELFRICE_DB":
                result.violations.append(
                    f"{where}: names AELFRICE_DB, which can redirect db_path()"
                )
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "joinpath" and any(_is_layout_component(a) for a in node.args):
                result.layout_joins[site] += 1
            if _is_db_path_call(node, aliases):
                result.db_path_calls[site] += 1
                if node.args or node.keywords:
                    result.violations.append(f"{where}: db_path() given arguments")
            if name == "chdir":
                result.violations.append(f"{where}: chdir moves the cwd db_path() reads")
            if name == "_git_common_dir" and (node.args or node.keywords):
                result.violations.append(f"{where}: _git_common_dir() given arguments")
            if name == "MemoryStore" and module == "hook.py":
                result.memorystore_calls[site] += 1
    return result


def _real_scan() -> Scan:
    return scan({m: (PKG / m).read_text(encoding="utf-8") for m in MODULES})


def _diff(actual: Counter[Site], expected: dict[Site, int]) -> list[str]:
    keys = sorted(set(actual) | set(expected))
    return [
        f"{k}: found {actual.get(k, 0)}, inventory says {expected.get(k, 0)}"
        for k in keys if actual.get(k, 0) != expected.get(k, 0)
    ]


def test_no_consumer_redirects_store_resolution() -> None:
    assert _real_scan().violations == []


def test_db_path_call_sites_match_the_inventory() -> None:
    assert _diff(_real_scan().db_path_calls, DB_PATH_CALLS) == []


def test_store_constructions_in_hook_match_the_inventory() -> None:
    assert _diff(_real_scan().memorystore_calls, MEMORYSTORE_CALLS) == []


def test_store_layout_joins_match_the_inventory() -> None:
    assert _diff(_real_scan().layout_joins, LAYOUT_JOINS) == []


# The rules above pass vacuously if the scanner cannot see a violation.
# Each case below is one shape a payload-cwd regression could take.

_HEADER = "from aelfrice.db_paths import db_path\nimport os\nfrom pathlib import Path\n"


def test_the_scanner_sees_a_chdir_around_resolution() -> None:
    src = _HEADER + (
        "def consumer(payload_cwd):\n"
        "    old = os.getcwd()\n"
        "    os.chdir(payload_cwd)\n"
        "    try:\n"
        "        return db_path()\n"
        "    finally:\n"
        "        os.chdir(old)\n"
    )
    assert len(scan({"hook.py": src}).violations) == 2


def test_the_scanner_sees_a_payload_cwd_join() -> None:
    src = _HEADER + (
        "def consumer(payload_cwd):\n"
        "    return Path(payload_cwd) / '.git' / 'aelfrice' / 'memory.db'\n"
    )
    assert scan({"hook.py": src}).layout_joins == Counter({("hook.py", "consumer"): 1})


def test_the_scanner_sees_an_aelfrice_db_redirect() -> None:
    src = _HEADER + (
        "def consumer(payload_cwd):\n"
        "    os.environ['AELFRICE_DB'] = str(Path(payload_cwd))\n"
        "    return db_path()\n"
    )
    assert len(scan({"hook.py": src}).violations) == 1


def test_the_scanner_sees_db_path_given_arguments() -> None:
    src = _HEADER + "def consumer(payload_cwd):\n    return db_path(payload_cwd)\n"
    result = scan({"hook.py": src})
    assert result.db_path_calls == Counter({("hook.py", "consumer"): 1})
    assert len(result.violations) == 1


def test_the_scanner_sees_an_aliased_db_path_in_a_nested_def() -> None:
    src = (
        "def outer():\n"
        "    def inner():\n"
        "        from aelfrice.db_paths import db_path as _p\n"
        "        return _p()\n"
        "    return inner\n"
    )
    assert scan({"hook.py": src}).db_path_calls == Counter(
        {("hook.py", "outer.inner"): 1},
    )


def test_the_scanner_sees_a_git_dir_resolved_at_the_payload_cwd() -> None:
    src = (
        "import subprocess\n"
        "from aelfrice.db_paths import _git_common_dir\n"
        "def a(payload_cwd):\n"
        "    return _git_common_dir(payload_cwd)\n"
        "def b(payload_cwd):\n"
        "    return subprocess.run(['rev-parse', '--git-common-dir'], cwd=payload_cwd)\n"
    )
    assert len(scan({"hook.py": src}).violations) == 2


def test_the_scanner_allows_payload_cwd_for_config() -> None:
    src = _HEADER + (
        "def consumer(payload_cwd):\n"
        "    cfg = load_config(start=payload_cwd)\n"
        "    here = Path.cwd()\n"
        "    return cfg, here, db_path().parent / 'state.json'\n"
    )
    result = scan({"hook.py": src})
    assert result.violations == []
    assert not result.layout_joins
