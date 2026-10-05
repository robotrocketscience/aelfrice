"""#1632 — the per-PR mutation job mutates changed functions, not files.

A real change to one function of `cli.py` put the whole 11k-line file in
`only_mutate`, and mutmut spent the job's 60 minutes generating its mutants.
`scripts/mutation_scope.py` now maps the PR's diff hunks to the functions
they touch, and marks every other function with mutmut's
`# pragma: no mutate block` so mutmut never generates their mutants.

These tests drive the real mapping on synthetic source. Each case pairs a
function that must be in scope with one that must not, so a mapper that
selects everything, or nothing, fails.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "mutation_scope.py"
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "mutation_scope_1632", str(_SCRIPT_PATH),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mutation_scope_1632"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ms() -> Any:
    return _load()


# Line numbers matter here: the comments on the right are the head-side
# line numbers the hunks below refer to.
_BASE = '''\
"""A module."""
import functools

LIMIT = 3


def first(a: int) -> int:
    def helper(x: int) -> int:
        return x * 2
    return helper(a) + 1


@functools.cache
def cached(a: int) -> int:
    return a + 1


def second(a: int, b: int) -> int:
    total = a + b
    return total - 1


class Box:
    size = 2

    def grow(self, n: int) -> int:
        return n + 10

    @staticmethod
    def shrink(n: int) -> int:
        return n - 10

    class Inner:
        def deep(self, n: int) -> int:
            return n * 7
'''
#  1 """A module."""
#  7 def first     8-9 helper      10 return
# 13 @functools.cache   14 def cached   15 return
# 18 def second    19 total   20 return
# 23 class Box     24 size
# 26 def grow      27 return
# 29 @staticmethod 30 def shrink   31 return
# 33 class Inner   34 def deep     35 return


def _names(units: list[Any]) -> list[str]:
    return [u.qualname for u in units]


def _scope(ms: Any, after: str, touched: set[int]) -> Any:
    result = ms.classify("mod.py", _BASE, after, touched)
    assert result is not None
    return result


def test_fixture_line_numbers_are_as_annotated() -> None:
    """The other tests address lines by number; pin that they are right."""
    lines = _BASE.splitlines()
    assert lines[6].startswith("def first")
    assert lines[17].startswith("def second")
    assert lines[18].strip() == "total = a + b"
    assert lines[25].strip().startswith("def grow")
    assert lines[34].strip() == "return n * 7"


# --- hunk parsing -----------------------------------------------------------


def test_an_added_hunk_touches_each_added_line(ms: Any) -> None:
    assert ms.touched_lines("@@ -3,0 +4,2 @@\n+a\n+b\n") == {4, 5}


def test_a_one_line_hunk_has_an_implicit_count(ms: Any) -> None:
    """`+10` with no count means one line, not zero."""
    assert ms.touched_lines("@@ -10 +10 @@\n-a\n+b\n") == {10}


def test_a_pure_deletion_touches_both_neighbours(ms: Any) -> None:
    """`+6,0` deletes between head lines 6 and 7; both are touched."""
    assert ms.touched_lines("@@ -7,2 +6,0 @@\n-a\n-b\n") == {6, 7}


def test_non_hunk_lines_are_ignored(ms: Any) -> None:
    diff = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n"
        "@@ -1 +1 @@\n-@@ -9 +9 @@\n++@@ -9 +9 @@\n"
    )
    assert ms.touched_lines(diff) == {1}


# --- hunk to function -------------------------------------------------------


def test_a_modified_line_selects_its_function_only(ms: Any) -> None:
    after = _BASE.replace("total = a + b", "total = a * b")
    result = _scope(ms, after, {19})
    assert _names(result.in_scope) == ["second"]
    assert result.unchanged == []
    assert result.outside == []


def test_an_added_function_is_in_scope(ms: Any) -> None:
    after = _BASE.replace(
        "\n\ndef second", "\n\ndef added(a: int) -> int:\n    return a - 3\n\n\ndef second",
    )
    lines = after.splitlines()
    start = lines.index("def added(a: int) -> int:") + 1
    result = _scope(ms, after, {start, start + 1, start + 2, start + 3})
    assert _names(result.in_scope) == ["added"]


def test_a_deletion_inside_a_function_selects_it(ms: Any) -> None:
    """A removed statement leaves no head line; its neighbours stand in."""
    after = _BASE.replace("    total = a + b\n", "    total = 0\n").replace(
        "    return total - 1\n", "",
    )
    # Deleting head line 20 (`return total - 1`) is `+19,0`: lines 19, 20.
    result = _scope(ms, after, ms.touched_lines("@@ -20 +19,0 @@\n"))
    assert "second" in _names(result.in_scope)
    assert "first" not in _names(result.in_scope)


def test_a_change_in_a_nested_function_selects_the_outer_unit(ms: Any) -> None:
    """mutmut mutates `helper` as part of `first`, so `first` is the unit."""
    after = _BASE.replace("return x * 2", "return x * 3")
    result = _scope(ms, after, {9})
    assert _names(result.in_scope) == ["first"]


def test_a_method_change_selects_the_method_with_its_mangled_key(ms: Any) -> None:
    after = _BASE.replace("return n + 10", "return n + 11")
    result = _scope(ms, after, {27})
    assert _names(result.in_scope) == ["Box.grow"]
    assert result.in_scope[0].key == "xǁBoxǁgrow"
    assert not result.in_scope[0].skipped_by_mutmut


def test_a_module_function_key_is_x_underscore(ms: Any) -> None:
    after = _BASE.replace("total = a + b", "total = a * b")
    assert _scope(ms, after, {19}).in_scope[0].key == "x_second"


def test_a_decorator_line_belongs_to_its_function(ms: Any) -> None:
    after = _BASE.replace("@functools.cache", "@functools.lru_cache")
    result = _scope(ms, after, {13})
    assert _names(result.in_scope) == ["cached"]
    assert result.in_scope[0].skipped_by_mutmut, (
        "mutmut does not mutate a decorated function"
    )
    assert result.mutated == []


def test_a_lone_staticmethod_is_still_mutated(ms: Any) -> None:
    """mutmut's one exception: a single staticmethod/classmethod decorator."""
    after = _BASE.replace("return n - 10", "return n - 11")
    result = _scope(ms, after, {31})
    assert _names(result.mutated) == ["Box.shrink"]


_DUNDER = '''\
class Box:
    def __new__(cls) -> "Box":
        return super().__new__(cls)

    def __setattr__(self, name: str, value: int) -> None:
        super().__setattr__(name, value + 1)

    def __getattribute__(self, name: str) -> int:
        return 1

    def size(self) -> int:
        return 2
'''
#  2 __new__   3 return   5 __setattr__   6 super   8 __getattribute__
#  9 return   11 size   12 return


def test_a_method_mutmut_skips_by_name_is_reported_not_mutated(ms: Any) -> None:
    """mutmut 3.8.0 never mutates these three, so no mutant name exists."""
    after = (
        _DUNDER.replace("value + 1", "value + 2")
        .replace("return super().__new__(cls)", "return object.__new__(cls)")
        .replace("        return 1\n", "        return 3\n")
        .replace("return 2", "return 4")
    )
    result = ms.classify("src/aelfrice/m.py", _DUNDER, after, {3, 6, 9, 12})
    assert result is not None
    assert _names(result.in_scope) == [
        "Box.__new__", "Box.__setattr__", "Box.__getattribute__", "Box.size",
    ]
    assert _names(result.mutated) == ["Box.size"]
    text = ms.render_summary(ms.Report(scopes=[result]))
    for name in ("__new__", "__setattr__", "__getattribute__"):
        assert (
            f"`Box.{name}`: changed, but not mutated: mutmut skips this "
            "method." in text
        )
        assert f"xǁBoxǁ{name}" not in text
    assert "`xǁBoxǁsize__mutmut_*`" in text


def test_a_module_level_edit_is_outside_every_unit(ms: Any) -> None:
    after = _BASE.replace("LIMIT = 3", "LIMIT = 4")
    result = _scope(ms, after, {4})
    assert result.in_scope == []
    assert result.outside == [4]


def test_a_class_body_and_nested_class_edit_are_outside(ms: Any) -> None:
    """mutmut mutates neither class attributes nor nested-class methods."""
    after = _BASE.replace("size = 2", "size = 3").replace(
        "return n * 7", "return n * 8",
    )
    result = _scope(ms, after, {24, 35})
    assert result.in_scope == []
    assert result.outside == [24, 35]


def test_a_touched_blank_or_comment_line_is_not_reported_outside(ms: Any) -> None:
    after = _BASE.replace("LIMIT = 3\n", "LIMIT = 3\n# A note.\n")
    result = _scope(ms, after, {5, 6})
    assert result.outside == []


def test_a_moved_function_is_unchanged_not_in_scope(ms: Any) -> None:
    """Moving code adds lines in the hunk, but carries no new mutant."""
    block = "\n\ndef second(a: int, b: int) -> int:\n    total = a + b\n    return total - 1\n"
    assert block in _BASE
    after = _BASE.replace(block, "").replace(
        "\n\n@functools.cache", block + "\n\n@functools.cache",
    )
    assert after != _BASE
    lines = after.splitlines()
    start = lines.index("def second(a: int, b: int) -> int:") + 1
    result = _scope(ms, after, {start, start + 1, start + 2})
    assert result.in_scope == []
    assert _names(result.unchanged) == ["second"]


def test_a_function_named_like_a_method_is_not_matched_to_it(ms: Any) -> None:
    """Unchanged means the same *qualified* name had the same code."""
    clone = "\n\ndef grow(self, n: int) -> int:\n    return n + 10\n"
    after = _BASE + clone
    start = len(_BASE.splitlines()) + 3
    result = _scope(ms, after, {start, start + 1})
    assert _names(result.in_scope) == ["grow"]
    assert result.unchanged == []


_DUP_BASE = "def dup() -> int:\n    return 1\n\n\ndef dup() -> int:\n    return 2\n"
#  1 def dup   2 return 1   5 def dup   6 return 2


def test_a_duplicated_name_is_never_called_unchanged(ms: Any) -> None:
    """Changing the second `dup` to equal the first is a real change.

    Matching by name against any base body would pair it with the first
    `dup` and report it unchanged, dropping a behavioural change.
    """
    after = _DUP_BASE.replace("return 2", "return 1")
    result = ms.classify("mod.py", _DUP_BASE, after, {6})
    assert result is not None
    assert [(u.qualname, u.start) for u in result.in_scope] == [("dup", 5)]
    assert result.unchanged == []


def test_a_name_duplicated_only_on_the_head_side_stays_in_scope(ms: Any) -> None:
    """Adding a second `dup` identical to the base one is still new code."""
    before = "def dup() -> int:\n    return 1\n"
    after = before + "\n\n" + before
    result = ms.classify("mod.py", before, after, {5, 6})
    assert result is not None
    assert [(u.qualname, u.start) for u in result.in_scope] == [("dup", 5)]
    assert result.unchanged == []


def test_a_comment_edit_inside_a_function_is_unchanged(ms: Any) -> None:
    after = _BASE.replace("    total = a + b\n", "    total = a + b  # Sum.\n")
    result = _scope(ms, after, {19})
    assert result.in_scope == []
    assert _names(result.unchanged) == ["second"]


def test_a_new_file_puts_every_touched_function_in_scope(ms: Any) -> None:
    result = ms.classify("mod.py", None, _BASE, set(range(1, 40)))
    assert result is not None
    assert _names(result.in_scope) == [
        "first", "cached", "second", "Box.grow", "Box.shrink",
    ]


def test_an_unparseable_head_fails_open(ms: Any) -> None:
    assert ms.classify("mod.py", _BASE, "def broken(:\n", {1}) is None


def test_an_unparseable_base_leaves_touched_units_in_scope(ms: Any) -> None:
    result = ms.classify("mod.py", "def broken(:\n", _BASE, {19})
    assert result is not None
    assert _names(result.in_scope) == ["second"]


# --- the pragma writer ------------------------------------------------------


_WRITE = '''\
def kept(a: int) -> int:
    return a + 1


def dropped(
    a: int,
    b: dict[str, int] = {},
) -> int:  # existing note
    return a - 1


def one(a: int) -> int: return a * 2


@functools.cache
def decorated(a: int) -> int:
    return a


class Box:
    @staticmethod
    def stat(n: int) -> int:
        return n + 5

    async def wait(self) -> None:
        await sleep(1)
'''


def test_out_of_scope_headers_get_the_block_pragma(ms: Any) -> None:
    out, failed = ms.exclude_units(_WRITE, {"kept"})
    assert failed == []
    lines = out.splitlines()
    assert lines[0] == "def kept(a: int) -> int:", "an in-scope unit is untouched"
    # The pragma goes on the line holding the header colon, ahead of the
    # comment already there: mutmut reads the first `no mutate` it finds.
    assert lines[7] == ") -> int:  # pragma: no mutate block  # existing note"
    assert lines[4] == "def dropped(", "the `def` line itself is not the header end"
    assert lines[11] == "def one(a: int) -> int: return a * 2  # pragma: no mutate"
    assert lines[21].endswith("def stat(n: int) -> int:  # pragma: no mutate block")
    assert lines[24].endswith("async def wait(self) -> None:  # pragma: no mutate block")


def test_a_function_mutmut_already_skips_is_left_alone(ms: Any) -> None:
    out, _ = ms.exclude_units(_WRITE, set())
    assert "def decorated(a: int) -> int:\n" in out


def test_the_rewrite_changes_no_line_count_and_no_ast(ms: Any) -> None:
    out, _ = ms.exclude_units(_WRITE, set())
    assert out != _WRITE
    assert len(out.splitlines()) == len(_WRITE.splitlines())
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(_WRITE))


def test_a_multi_line_one_liner_is_reported_not_annotated(ms: Any) -> None:
    """No comment position covers it, so it is named rather than dropped."""
    source = "def odd(\n    a: int,\n) -> int: return a\n\n\ndef kept() -> int:\n    return 1\n"
    out, failed = ms.exclude_units(source, {"kept"})
    assert failed == ["odd"]
    assert out == source


def test_a_rewrite_that_changes_the_ast_is_refused(
    ms: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The round-trip check is what licenses editing source at all."""
    monkeypatch.setattr(ms, "PRAGMA_LINE", "+ 1")
    with pytest.raises(ValueError, match="round-trip"):
        ms.exclude_units(_WRITE, {"kept"})


def test_keep_matches_by_qualname_so_line_drift_is_harmless(ms: Any) -> None:
    """A PR checkout is the merge commit, not the head the diff used."""
    shifted = "# Lines added on the base branch.\n\n\n" + _WRITE
    out, _ = ms.exclude_units(shifted, {"Box.stat"})
    assert "def stat(n: int) -> int:\n" in out
    assert "def kept(a: int) -> int:  # pragma: no mutate block\n" in out


# --- the report -------------------------------------------------------------


def test_ranges_compacts_consecutive_lines(ms: Any) -> None:
    assert ms._ranges([3, 4, 5, 9, 11, 12]) == "3-5, 9, 11-12"
    assert ms._ranges([7]) == "7"
    assert ms._ranges([]) == ""


def test_the_summary_names_every_scope_decision_and_its_reason(ms: Any) -> None:
    after = (
        _BASE.replace("total = a + b", "total = a * b")
        .replace("@functools.cache", "@functools.lru_cache")
        .replace("LIMIT = 3", "LIMIT = 4")
        .replace("return x * 2", "return x * 2  # Same.")
    )
    verdict = ms.classify("src/aelfrice/mod.py", _BASE, after, {4, 9, 13, 19})
    report = ms.Report(
        files=["src/aelfrice/mod.py"],
        scopes=[verdict],
        skipped_files=[("src/aelfrice/doc.py", "comments only")],
        whole_files=[("src/aelfrice/bad.py", "head version does not parse")],
        not_excluded=[("src/aelfrice/mod.py", "odd")],
    )
    text = ms.render_summary(report)
    assert "| `src/aelfrice/mod.py` | `second` | `x_second__mutmut_*` |" in text
    assert "`src/aelfrice/mod.py` `first`: not mutated: only moved" in text
    assert (
        "`src/aelfrice/mod.py` `cached`: changed, but not mutated: mutmut "
        "does not mutate decorated functions." in text
    )
    assert "`src/aelfrice/mod.py` lines 4: not mutated" in text
    assert "`src/aelfrice/doc.py`: not mutated: comments only." in text
    assert "`src/aelfrice/bad.py`: mutated whole: head version does not parse." in text
    assert "`src/aelfrice/mod.py` `odd`: mutated although unchanged" in text
    assert "| `src/aelfrice/mod.py` | `first`" not in text


def test_an_empty_scope_says_so(ms: Any) -> None:
    assert "No changed function is mutated." in ms.render_summary(ms.Report())


# --- against mutmut itself, where it is installed ---------------------------


def test_mutmut_generates_mutants_only_for_kept_units(
    ms: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The pragma contract, checked against the generator it relies on.

    mutmut is installed only in the mutation job, not in the test
    environment, so this skips elsewhere; the unit tests above carry the
    mapping. Run from an empty directory so mutmut's config loader reads
    no project file.
    """
    file_mutation = pytest.importorskip("mutmut.mutation.file_mutation")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["src"]\n', encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    configuration = pytest.importorskip("mutmut.configuration")
    configuration.reset_config()
    source = _WRITE.replace("@functools.cache\n", "")
    out, failed = ms.exclude_units(source, {"kept", "Box.wait"})
    assert failed == []
    mutated = file_mutation.mutate_file_contents("src/m.py", out)
    keys = {name.partition("__mutmut_")[0] for name in mutated.mutant_names}
    assert keys == {"x_kept", "xǁBoxǁwait"}
    configuration.reset_config()


def test_mutmut_generates_no_mutant_for_the_named_methods(
    ms: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`MUTMUT_SKIPPED_NAMES` matches what the generator really skips."""
    file_mutation = pytest.importorskip("mutmut.mutation.file_mutation")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["src"]\n', encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    configuration = pytest.importorskip("mutmut.configuration")
    configuration.reset_config()
    mutated = file_mutation.mutate_file_contents("src/m.py", _DUNDER)
    keys = {name.partition("__mutmut_")[0] for name in mutated.mutant_names}
    assert keys == {"xǁBoxǁsize"}
    assert ms.MUTMUT_SKIPPED_NAMES == file_mutation.NEVER_MUTATE_FUNCTION_NAMES
    configuration.reset_config()


# --- end to end over a real git diff ----------------------------------------


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-c", "user.email=test@example.invalid",
            "-c", "user.name=test",
            "-c", "commit.gpgsign=false",
            "-c", "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo, capture_output=True, text=True, check=False, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.fixture
def pr_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """A base and a head commit: one real change, one comment-only file,
    and one module-level-only change."""
    pkg = tmp_path / "src" / "aelfrice"
    pkg.mkdir(parents=True)
    consts = "X = {}\n\n\ndef f() -> int:\n    return X\n"
    (pkg / "mod.py").write_text(_BASE, encoding="utf-8")
    (pkg / "doc.py").write_text("# Old.\nX = 1\n", encoding="utf-8")
    (pkg / "consts.py").write_text(consts.format(1), encoding="utf-8")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (pkg / "mod.py").write_text(
        _BASE.replace("total = a + b", "total = a * b"), encoding="utf-8",
    )
    (pkg / "doc.py").write_text("# New.\nX = 1\n", encoding="utf-8")
    (pkg / "consts.py").write_text(consts.format(2), encoding="utf-8")
    _git(tmp_path, "commit", "-q", "-am", "head")
    head = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.chdir(tmp_path)
    return base, head


@pytest.mark.timeout(60)  # spawns git (#1307)
def test_function_scope_keeps_only_files_with_a_changed_function(
    ms: Any, pr_repo: tuple[str, str],
) -> None:
    base, head = pr_repo
    report = ms.function_scope(base, head, write=False)
    assert report.files == ["src/aelfrice/mod.py"]
    assert dict(report.skipped_files) == {
        "src/aelfrice/doc.py": "comments, docstrings, or formatting only",
        "src/aelfrice/consts.py": "no changed function that mutmut mutates",
    }
    (mod,) = [s for s in report.scopes if s.path == "src/aelfrice/mod.py"]
    assert _names(mod.in_scope) == ["second"]
    (consts,) = [s for s in report.scopes if s.path == "src/aelfrice/consts.py"]
    assert consts.outside == [1]


@pytest.mark.timeout(60)  # spawns git (#1307)
def test_a_change_merged_into_the_base_is_not_this_prs_scope(
    ms: Any, pr_repo: tuple[str, str],
) -> None:
    """The hunks are the merge-base diff, so `main`'s own edits stay out."""
    base, head = pr_repo
    repo = Path.cwd()
    _git(repo, "switch", "-q", "-c", "later", base)
    mod = repo / "src" / "aelfrice" / "mod.py"
    mod.write_text(_BASE.replace("return helper(a) + 1", "return helper(a) + 2"), encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "main moves on")
    advanced = _git(repo, "rev-parse", "HEAD")
    report = ms.function_scope(advanced, head, write=False)
    (scope_,) = [s for s in report.scopes if s.path == "src/aelfrice/mod.py"]
    assert _names(scope_.in_scope) == ["second"]


@pytest.mark.timeout(60)  # spawns git (#1307)
def test_write_annotates_every_other_function_in_the_working_tree(
    ms: Any, pr_repo: tuple[str, str],
) -> None:
    base, head = pr_repo
    before = Path("src/aelfrice/mod.py").read_text(encoding="utf-8")
    ms.function_scope(base, head, write=True)
    after = Path("src/aelfrice/mod.py").read_text(encoding="utf-8")
    assert "def second(a: int, b: int) -> int:\n" in after
    assert "def first(a: int) -> int:  # pragma: no mutate block\n" in after
    assert "    def grow(self, n: int) -> int:  # pragma: no mutate block\n" in after
    assert ast.dump(ast.parse(after)) == ast.dump(ast.parse(before))
    # A file left out of `only_mutate` is never touched.
    consts = Path("src/aelfrice/consts.py").read_text(encoding="utf-8")
    assert "pragma" not in consts


@pytest.mark.timeout(60)  # spawns git (#1307)
def test_dry_run_writes_nothing_and_prints_the_scope(
    ms: Any, pr_repo: tuple[str, str], capsys: pytest.CaptureFixture[str],
) -> None:
    base, head = pr_repo
    before = Path("src/aelfrice/mod.py").read_text(encoding="utf-8")
    assert ms.main(["--base", base, "--head", head, "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "src/aelfrice/mod.py\n"
    assert "`second`" in captured.err
    assert Path("src/aelfrice/mod.py").read_text(encoding="utf-8") == before


@pytest.mark.timeout(60)  # spawns git (#1307)
def test_the_summary_is_written_even_when_nothing_is_in_scope(
    ms: Any, pr_repo: tuple[str, str], capsys: pytest.CaptureFixture[str],
) -> None:
    base, _ = pr_repo
    summary = Path("scope.md")
    args = ["--base", base, "--head", base, "--write-pragmas", "--summary", str(summary)]
    assert ms.main(args) == 0
    assert capsys.readouterr().out == ""
    assert "No changed function is mutated." in summary.read_text(encoding="utf-8")


def _renamed_repo(root: Path, after: str) -> tuple[str, str]:
    """Commit `_BASE` as `b.py`, then move it to `d.py` holding `after`."""
    pkg = root / "src" / "aelfrice"
    pkg.mkdir(parents=True)
    (pkg / "b.py").write_text(_BASE, encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    base = _git(root, "rev-parse", "HEAD")
    _git(root, "mv", "src/aelfrice/b.py", "src/aelfrice/d.py")
    (pkg / "d.py").write_text(after, encoding="utf-8")
    _git(root, "commit", "-q", "-am", "head")
    return base, _git(root, "rev-parse", "HEAD")


@pytest.mark.timeout(60)  # spawns git (#1307)
def test_a_pure_rename_puts_nothing_in_scope(
    ms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base blob is read from the old path, so nothing looks new."""
    base, head = _renamed_repo(tmp_path, _BASE)
    monkeypatch.chdir(tmp_path)
    report = ms.function_scope(base, head, write=False)
    assert report.files == []
    assert report.scopes == []
    assert report.skipped_files == [
        ("src/aelfrice/d.py", "comments, docstrings, or formatting only"),
    ]


@pytest.mark.timeout(60)  # spawns git (#1307)
def test_a_rename_with_one_changed_function_scopes_that_function(
    ms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the real change is in scope; a comment edit is compared
    against the old path's code and stays out."""
    after = _BASE.replace("total = a + b", "total = a * b").replace(
        "return x * 2", "return x * 2  # Same.",
    )
    base, head = _renamed_repo(tmp_path, after)
    monkeypatch.chdir(tmp_path)
    report = ms.function_scope(base, head, write=False)
    assert report.files == ["src/aelfrice/d.py"]
    (scope_,) = report.scopes
    assert _names(scope_.in_scope) == ["second"]
    assert _names(scope_.unchanged) == ["first"]
    # The untouched decorated function is not reported as changed.
    assert "`cached`" not in ms.render_summary(report)


def test_dry_run_and_write_cannot_be_combined(ms: Any) -> None:
    with pytest.raises(SystemExit):
        ms.main(["--base", "a", "--head", "b", "--dry-run", "--write-pragmas"])


# --- the workflow wiring ----------------------------------------------------


def test_the_workflow_writes_pragmas_and_publishes_the_scope() -> None:
    """The script only narrows the run if the job asks it to write."""
    workflow = (
        Path(__file__).resolve().parent.parent
        / ".github" / "workflows" / "mutation.yml"
    ).read_text(encoding="utf-8").replace("\\\n", " ")
    calls = [
        line for line in workflow.splitlines()
        if "scripts/mutation_scope.py" in line
        and not line.lstrip().startswith("#")
    ]
    assert len(calls) == 1
    assert "--write-pragmas" in calls[0]
    assert "--summary mutation-scope.md" in calls[0]
    assert "--dry-run" not in calls[0]
    assert 'cat mutation-scope.md >> "${GITHUB_STEP_SUMMARY}"' in workflow
