"""#1582: `.aelfrice.toml` discovery is bounded by the project.

`docs/user/CONFIG.md` promises that "there is no global configuration and
no per-user configuration". Before this change the walk ascended to the
filesystem root, so from any directory under `$HOME` it found
`$HOME/.aelfrice.toml` — per-user configuration, delivered by a code path
that documented itself as not having any. On a machine carrying one, two
tests resolved the operator's `posterior_weight` instead of the default
and failed for a reason that was not the code.

These tests pin the bound at its own surface (`discover_config`), at the
surface of the resolver named in the issue (`resolve_posterior_weight`),
and at the population: no module may reintroduce a private walk and so
opt out of the bound without anything reporting it.

Every fixture is planted under `tmp_path`. Nothing here reads, writes or
depends on the real per-user config file.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import aelfrice
from aelfrice import retrieval
from aelfrice.config_discovery import (
    CONFIG_FILENAME,
    WORKTREE_MARKER,
    discover_config,
)

# A weight that is not the default, so "the default came back" and "the
# planted file came back" are distinguishable outcomes.
_PLANTED_WEIGHT = 1.5

_PLANTED_TOML = f"[retrieval]\nposterior_weight = {_PLANTED_WEIGHT}\n"


@pytest.fixture()
def sandboxed_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point `Path.home()` at a directory under `tmp_path`.

    `Path.home()` reads `HOME` on POSIX and `USERPROFILE` on Windows, so
    both are set. Without this the home rule would be asserted against
    the real home directory, which is the one directory these tests must
    never touch.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert Path.home().resolve() == home.resolve(), (
        "Path.home() did not follow the sandboxed environment; the home "
        "bound below would be measured against the real home directory"
    )
    return home


@pytest.fixture()
def no_weight_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AELFRICE_POSTERIOR_WEIGHT` outranks TOML in the resolver.

    With it set in the developer's shell the end-to-end assertion below
    passes without the walk being bounded at all.
    """
    monkeypatch.delenv("AELFRICE_POSTERIOR_WEIGHT", raising=False)


def _worktree(root: Path, *, marker_is_file: bool = False) -> Path:
    """Make `root` look like a git work-tree root."""
    root.mkdir(parents=True, exist_ok=True)
    marker = root / WORKTREE_MARKER
    if marker_is_file:
        marker.write_text("gitdir: /elsewhere/.git/worktrees/w\n")
    else:
        marker.mkdir()
    return root


def test_a_config_above_the_worktree_root_is_invisible(
    tmp_path: Path, sandboxed_home: Path,
) -> None:
    """The defect, as an assertion.

    A `.aelfrice.toml` one directory above the work-tree root is outside
    the project, and discovery must not reach it from inside.
    """
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / CONFIG_FILENAME).write_text(_PLANTED_TOML)
    repo = _worktree(outer / "repo")
    deep = repo / "src" / "pkg"
    deep.mkdir(parents=True)

    assert discover_config(deep) is None
    assert discover_config(repo) is None


def test_a_config_at_the_worktree_root_is_still_read(
    tmp_path: Path, sandboxed_home: Path,
) -> None:
    """The distinguishing case.

    A bound that returned None everywhere would satisfy the test above
    while deleting project configuration outright. The work-tree root is
    inside the project, so its config is the one that applies.
    """
    repo = _worktree(tmp_path / "repo")
    (repo / CONFIG_FILENAME).write_text(_PLANTED_TOML)
    deep = repo / "src" / "pkg"
    deep.mkdir(parents=True)

    assert discover_config(deep) == repo / CONFIG_FILENAME


def test_a_linked_worktree_marker_file_bounds_the_walk(
    tmp_path: Path, sandboxed_home: Path,
) -> None:
    """`git worktree` and submodules write `.git` as a *file*.

    Probing for a directory would leave exactly the checkout shape this
    issue was reported from — a linked work tree — unbounded.
    """
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / CONFIG_FILENAME).write_text(_PLANTED_TOML)
    linked = _worktree(outer / "linked", marker_is_file=True)
    sub = linked / "sub"
    sub.mkdir()

    assert discover_config(sub) is None


def test_the_home_config_is_unreachable_without_a_worktree(
    sandboxed_home: Path,
) -> None:
    """No `.git` anywhere, so only the home rule can stop the walk.

    This is the shape a plain directory under `$HOME` has, and the
    work-tree rule cannot help with it.
    """
    (sandboxed_home / CONFIG_FILENAME).write_text(_PLANTED_TOML)
    work = sandboxed_home / "scratch" / "notes"
    work.mkdir(parents=True)

    assert discover_config(work) is None


def test_the_home_config_is_unreachable_from_home_itself(
    sandboxed_home: Path,
) -> None:
    """Rule 1 fires before the probe, so `$HOME` is not merely last.

    A home directory that is itself a project is the one case where the
    per-user file would otherwise still be read, and reading it is
    exactly what the documented contract denies.
    """
    (sandboxed_home / CONFIG_FILENAME).write_text(_PLANTED_TOML)

    assert discover_config(sandboxed_home) is None


def test_outside_a_worktree_the_walk_still_ascends(
    sandboxed_home: Path,
) -> None:
    """The bound narrows the walk; it does not abolish it.

    "Examine `start` only" would also pass every assertion above while
    silently dropping the config at the top of a non-git project that a
    caller reaches from a subdirectory.
    """
    project = sandboxed_home / "project"
    project.mkdir()
    (project / CONFIG_FILENAME).write_text(_PLANTED_TOML)
    deep = project / "a" / "b"
    deep.mkdir(parents=True)

    assert discover_config(deep) == project / CONFIG_FILENAME


def test_the_home_rule_bounds_the_ancestors_of_start_not_the_machine(
    tmp_path: Path, sandboxed_home: Path,
) -> None:
    """Rule 1 matches `$HOME` itself; it is not a ceiling on the walk.

    `docs/user/CONFIG.md` and `discover_config` both say so, and both
    said the opposite until #1582's review: "it never reaches your home
    directory or anything above it" contradicted rule 4 in the same
    section, and the code delivers rule 4. From a start that has
    `$HOME` nowhere in its ancestry, the walk ascends past the level
    `$HOME` sits at and reads a config it finds there.

    The one thing that stays true in this shape is the contract the fix
    exists for: the user's own `$HOME/.aelfrice.toml` is still never
    read, which the sibling test above pins.
    """
    (tmp_path / CONFIG_FILENAME).write_text(_PLANTED_TOML)
    outside = tmp_path / "elsewhere" / "proj" / "sub"
    outside.mkdir(parents=True)

    found = discover_config(outside)

    assert found == tmp_path / CONFIG_FILENAME
    home = sandboxed_home.resolve()
    assert found.parent.resolve() in home.parents, (
        "the fixture no longer places the planted config above the "
        "sandboxed home directory, so this test would pass without the "
        "walk having crossed that level at all"
    )


def test_resolve_posterior_weight_ignores_a_config_above_the_project(
    tmp_path: Path, sandboxed_home: Path, no_weight_env: None,
) -> None:
    """End to end, through the resolver #1582 was reported against.

    `discover_config` returning None is the mechanism; a resolver
    returning its documented default is the behaviour the suite depends
    on, and the two are worth asserting separately.
    """
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / CONFIG_FILENAME).write_text(_PLANTED_TOML)
    repo = _worktree(outer / "repo")

    assert retrieval.resolve_posterior_weight(start=repo) == (
        retrieval.DEFAULT_POSTERIOR_WEIGHT
    )
    # Distinguishing: the same resolver still reads an in-project file,
    # so the assertion above is about the boundary and not about the
    # resolver having stopped reading TOML at all.
    (repo / CONFIG_FILENAME).write_text(_PLANTED_TOML)
    assert retrieval.resolve_posterior_weight(start=repo) == _PLANTED_WEIGHT


def _private_walkers() -> dict[str, str]:
    """Functions that look for the config filename *and* climb parents.

    Enumerated from the AST rather than listed, because the value of the
    #1304 funnel is that bounding one walk bounds every reader, and a
    module that grows its own walk loop silently leaves the bound
    behind. A function that names the config filename (under either
    spelling, or as a bare literal) and also touches `.parent` /
    `.parents` is doing its own discovery.

    The enumeration recurses into the subpackages. No config reader
    lives in one today, so scanning only the top level would give the
    same empty answer for the wrong reason, and a reader added under
    `wonder/` or `query_understanding/` later would not be seen.
    """
    offenders: dict[str, str] = {}
    package_dir = Path(aelfrice.__file__).parent
    for source_path in _package_sources():
        if source_path.name == "config_discovery.py":
            continue
        dotted = ".".join(
            source_path.relative_to(package_dir).with_suffix("").parts,
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _names_config(node) and _climbs_parents(node):
                offenders[f"{dotted}.{node.name}"] = source_path.name
    return offenders


def _names_config(node: ast.AST) -> bool:
    """True when `node` mentions the config filename under any spelling."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in (
            "CONFIG_FILENAME",
            "_CONFIG_FILENAME",
        ):
            return True
        if isinstance(child, ast.Constant) and child.value == CONFIG_FILENAME:
            return True
    return False


def _climbs_parents(node: ast.AST) -> bool:
    """True when `node` touches `.parent` or `.parents`."""
    return any(
        isinstance(child, ast.Attribute)
        and child.attr in ("parent", "parents")
        for child in ast.walk(node)
    )


def test_no_module_discovers_config_outside_the_shared_walk() -> None:
    """The population guard for acceptance criterion 2.

    Bounding one resolver and leaving the others unbounded would be
    worse than leaving all of them unbounded, because the behaviour
    would differ per key. Every reader funnels through
    `config_discovery.discover_config`, so the bound applies once; this
    fails the moment a module reintroduces a walk of its own.
    """
    offenders = _private_walkers()
    assert offenders == {}, (
        "these functions look for the config file and climb parents "
        "themselves, so they do not inherit the #1582 bound: "
        f"{sorted(offenders)}"
    )


def test_the_scan_would_see_a_private_walk() -> None:
    """Keeps the guard above from passing because it inspects nothing.

    The scan reads real package sources, so a rename or a packaging
    change could make it enumerate zero functions and still assert an
    empty offender set. This runs the same predicate over a module that
    does contain a private walk.
    """
    tree = ast.parse(
        "from pathlib import Path\n"
        "CONFIG_FILENAME = '.aelfrice.toml'\n"
        "def find(start):\n"
        "    for d in [start, *start.parents]:\n"
        "        if (d / CONFIG_FILENAME).is_file():\n"
        "            return d / CONFIG_FILENAME\n"
        "    return None\n",
    )
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert _names_config(func) and _climbs_parents(func), (
        "the predicate behind the population guard does not recognise a "
        "private walk, so the guard is vacuous"
    )


def test_the_scan_reaches_the_subpackages() -> None:
    """The other way the population guard could pass by seeing nothing.

    `pkgutil.iter_modules` does not recurse, so the scan used to skip
    `wonder/`, `query_understanding/`, `slash_commands/` and `data/`
    entirely. It found no offender there because it never looked.
    """
    package_dir = Path(aelfrice.__file__).parent
    nested = {
        source.relative_to(package_dir).parts[0]
        for source in _package_sources()
        if len(source.relative_to(package_dir).parts) > 1
    }
    assert len(nested) >= 2, (
        "the scan enumerates fewer than two subpackages, so the "
        "population guard is not covering them: "
        f"{sorted(nested)}"
    )


# The tables `docs/user/CONFIG.md` tells a user have no per-key
# `AELFRICE_*` override, mapped to the environment names their module
# may name at all. An empty set means "reads no environment"; a
# one-element set means "an on/off switch, and nothing that can carry a
# byte cap, a budget or a threshold".
_ENVLESS_TABLES: dict[str, set[str]] = {
    "noise_filter": set(),
    "dedup": set(),
    "hook_audit": {"AELFRICE_HOOK_AUDIT"},
    "relationship_detector": {"AELFRICE_AUTO_RELATIONSHIPS"},
    # `load_rebuilder_config` lives here (#1527). The one name is the
    # `[rebuild_log]` switch; no `[rebuilder]` budget has an override.
    "rebuild_log": {"AELFRICE_REBUILD_LOG"},
}

# Key fragments that may not appear in any environment name anywhere in
# the package, because CONFIG.md tells a user no such variable exists.
# `TOKEN_BUDGET` is deliberately absent: `[retrieval] token_budget`
# does have `AELFRICE_RETRIEVAL_TOKEN_BUDGET`.
_ABSENT_KEY_FRAGMENTS: tuple[str, ...] = (
    "NOISE",
    "DEDUP",
    "JACCARD",
    "MIN_WORDS",
    "MAX_BYTES",
    "CONFIDENCE",
    "CANDIDATE_PAIRS",
    "THRESHOLD_FRACTION",
    "TURN_WINDOW",
)


def _env_names_in(source_path: Path) -> set[str]:
    """Every `AELFRICE_*` name a module uses as a string literal.

    `fullmatch` rather than a substring search, so prose in a docstring
    that merely *mentions* a variable is not counted as a module
    reading one.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"AELFRICE_[A-Z0-9_]+", node.value)
    }


def _package_sources() -> list[Path]:
    """Every shipped module, including the subpackages."""
    return sorted(Path(aelfrice.__file__).parent.rglob("*.py"))


def test_the_tables_the_doc_calls_envless_really_are() -> None:
    """The remedy in `docs/user/CONFIG.md` has to be reachable.

    That paragraph tells a user who loses `~/.aelfrice.toml` which keys
    they can rescue with an environment variable and which they can
    only rescue by copying into the project file. It said "export the
    matching `AELFRICE_*` environment variable" of every key until
    #1582's review, and for these tables there is no matching variable
    to export: the user follows the instruction, exports a plausible
    name, gets the defaults, and sees no error. Adding a real override
    here is welcome -- it just has to reach the paragraph too.
    """
    package_dir = Path(aelfrice.__file__).parent
    found = {
        module: _env_names_in(package_dir / f"{module}.py")
        for module in _ENVLESS_TABLES
    }
    assert found == _ENVLESS_TABLES, (
        "these modules no longer name exactly the environment variables "
        "docs/user/CONFIG.md says they do, so the remedy paragraph is "
        "now wrong in one direction or the other"
    )


def test_no_module_names_an_override_the_doc_denies() -> None:
    """The same claim across the package, not just the owning module.

    A reader of `[dedup] jaccard_min` does not care which module holds
    the override; they care whether exporting something wins. This
    scans every shipped module, subpackages included, for a name that
    would make the paragraph's denial false.
    """
    offenders: dict[str, list[str]] = {}
    for source_path in _package_sources():
        hits = sorted(
            name
            for name in _env_names_in(source_path)
            if any(frag in name for frag in _ABSENT_KEY_FRAGMENTS)
        )
        if hits:
            offenders[source_path.name] = hits
    assert offenders == {}, (
        "an environment override now exists for a key docs/user/CONFIG.md "
        f"tells users has none: {offenders}"
    )


def test_the_env_scan_sees_names_that_do_exist() -> None:
    """Keeps both guards above from passing because they see nothing.

    Both assert an absence over sources read off disk, so a rename or a
    packaging change could make them scan an empty set of literals and
    still pass. `[cadence]` is the counter-case: every one of its keys
    has a per-key variable, which is why CONFIG.md names that table on
    the other side of the sentence.
    """
    cadence = Path(aelfrice.__file__).parent / "cadence.py"
    cadence_names = _env_names_in(cadence)
    assert len(cadence_names) > 1, (
        "the scan found at most one environment name in cadence.py, "
        "which has nine; the absence assertions above prove nothing"
    )
    assert "AELFRICE_CADENCE_K" in cadence_names
