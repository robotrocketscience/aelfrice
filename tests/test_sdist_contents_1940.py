"""The sdist holds the tracked files plus PKG-INFO, after tests have run (#1940).

`publish.yml` ran pytest and then `uv build` in one checkout. Hypothesis
writes its cache to `.hypothesis/` in the working directory, and hatchling
put every file there into the sdist: the 5.4.0 sdist on PyPI carried 239
cache files. Hatchling reads the root `.gitignore` and not the `.gitignore`
that Hypothesis writes inside the cache folder.

Two lines in the repository keep the cache out, and each one must hold on
its own: the root `.gitignore` entry and the sdist `exclude` in
`pyproject.toml`. The tests share one copy of the tracked files with a
`.hypothesis/` folder shaped like the one Hypothesis writes. Each removes
one of the two lines or neither, builds an sdist through the backend's
PEP 517 hook, and compares the archive with the tracked files.
The control removes both lines and must see the cache in the archive;
without it, a fixture that hatchling skipped for some other reason would
pass every other arm.

The copy opens every tracked `.py` file under `src/aelfrice`, so the module
carries `source_scan` and the mutation run leaves it out. Inside `mutants/`
`git ls-files` lists nothing, and the copy would hold mutmut's files.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.source_scan, pytest.mark.timeout(120)]

_REPO = Path(__file__).resolve().parents[1]

_SDIST_TABLE = "[tool.hatch.build.targets.sdist]"

#: Files shaped like a real cache: Hypothesis writes a `.gitignore` holding
#: `*` at the top of the folder and one file per module under `constants/`.
_CACHE_FILES = {
    ".hypothesis/.gitignore": "# This file was automatically created by Hypothesis\n*\n",
    ".hypothesis/constants/0123456789abcdef": "# file: /ci/runner/src/aelfrice/x.py\n",
    ".hypothesis/examples/fedcba9876543210/0011": "probe\n",
}


def _tracked() -> list[str]:
    out = subprocess.run(  # path-filter: **
        ["git", "ls-files", "-z"],
        cwd=_REPO, check=True, capture_output=True, timeout=60,
    ).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p]


def _copy_tracked(tree: Path) -> set[str]:
    """Copy each tracked file present in the working tree; return the paths."""
    copied: set[str] = set()
    for rel in _tracked():
        src = _REPO / rel  # path-filter: **
        if not src.is_file() and not src.is_symlink():
            continue  # deleted in the working tree, or a submodule
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)
        copied.add(rel)
    assert len(copied) > 100, f"only {len(copied)} tracked files copied"
    return copied


def _add_cache(tree: Path) -> None:
    for rel, text in _CACHE_FILES.items():
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _configure(tree: Path, *, keep_gitignore: bool, keep_exclude: bool) -> None:
    """Rewrite both config files from the repository, minus the named lines.

    The tree is shared by every test in the module, so each test starts
    from the repository's own copies rather than from the last test's.
    """
    gitignore = (_REPO / ".gitignore").read_text(encoding="utf-8")
    if not keep_gitignore:
        gitignore = "".join(
            line for line in gitignore.splitlines(keepends=True)
            if line.strip() not in {".hypothesis", ".hypothesis/", "/.hypothesis", "/.hypothesis/"}
        )
    (tree / ".gitignore").write_text(gitignore, encoding="utf-8")

    out: list[str] = []
    skipping = False
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    for line in pyproject.splitlines(keepends=True):
        if not keep_exclude and line.strip() == _SDIST_TABLE:
            skipping = True
            continue
        if skipping and line.startswith("["):
            skipping = False
        if not skipping:
            out.append(line)
    (tree / "pyproject.toml").write_text("".join(out), encoding="utf-8")


def _build_sdist(tree: Path, out_dir: Path) -> list[str]:
    """Build through hatchling's PEP 517 hook; return the archive's files."""
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("HATCH_")}
    result = subprocess.run(
        [
            sys.executable, "-I", "-c",
            "import sys, hatchling.build as b; print(b.build_sdist(sys.argv[1]))",
            str(out_dir),
        ],
        cwd=tree, env=env, capture_output=True, text=True, timeout=100,
    )
    assert result.returncode == 0, result.stderr
    (archive,) = out_dir.glob("*.tar.gz")
    with tarfile.open(archive) as tar:
        names = [m.name for m in tar.getmembers() if not m.isdir()]
    prefixes = {n.split("/", 1)[0] for n in names}
    assert len(prefixes) == 1, prefixes
    return [n.split("/", 1)[1] for n in names]


@pytest.fixture(scope="module")
def _tree(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, set[str]]:
    """One copy of the tracked files with a cache folder; about 120 MB."""
    tree = tmp_path_factory.mktemp("sdist_tree")
    expected = _copy_tracked(tree) | {"PKG-INFO"}
    _add_cache(tree)
    return tree, expected


def _sdist_for(
    tree_and_expected: tuple[Path, set[str]], out_dir: Path,
    *, keep_gitignore: bool, keep_exclude: bool,
) -> tuple[set[str], list[str]]:
    tree, expected = tree_and_expected
    _configure(tree, keep_gitignore=keep_gitignore, keep_exclude=keep_exclude)
    return expected, _build_sdist(tree, out_dir)


@pytest.mark.parametrize(
    ("keep_gitignore", "keep_exclude"),
    [(True, True), (True, False), (False, True)],
    ids=["both-lines", "gitignore-only", "sdist-exclude-only"],
)
def test_the_sdist_is_the_tracked_files_plus_pkg_info(
    _tree: tuple[Path, set[str]], tmp_path: Path,
    keep_gitignore: bool, keep_exclude: bool,
) -> None:
    expected, names = _sdist_for(
        _tree, tmp_path / "dist",
        keep_gitignore=keep_gitignore, keep_exclude=keep_exclude,
    )
    assert len(names) == len(set(names)), "the archive lists a file twice"
    extra = sorted(set(names) - expected)
    missing = sorted(expected - set(names))
    assert extra == [], f"files in the sdist that git does not track: {extra}"
    assert missing == [], f"tracked files missing from the sdist: {missing}"


def test_the_control_without_either_line_ships_the_cache(
    _tree: tuple[Path, set[str]], tmp_path: Path,
) -> None:
    expected, names = _sdist_for(
        _tree, tmp_path / "dist", keep_gitignore=False, keep_exclude=False,
    )
    extra = set(names) - expected
    assert extra == set(_CACHE_FILES), (
        "with both lines removed the sdist no longer carries the planted "
        f"cache files, so the other arms cannot tell the fix apart: {sorted(extra)}"
    )

