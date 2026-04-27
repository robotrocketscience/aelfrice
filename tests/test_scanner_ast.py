"""extract_ast: walk .py files and emit docstring-based candidates.

Tests use tmp_path to write Python source files, then invoke the
extractor and assert one property each — split per the deterministic-
atomic-short policy.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.scanner import SentenceCandidate, extract_ast


# --- Empty / missing inputs ---------------------------------------------


def test_missing_path_returns_empty(tmp_path: Path) -> None:
    bogus = tmp_path / "nope"
    assert extract_ast(bogus) == []


def test_file_path_instead_of_directory_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "module.py"
    f.write_text('"""a module"""', encoding="utf-8")
    assert extract_ast(f) == []


def test_empty_directory_returns_empty(tmp_path: Path) -> None:
    assert extract_ast(tmp_path) == []


def test_directory_with_no_py_files_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")
    assert extract_ast(tmp_path) == []


# --- Module docstrings --------------------------------------------------


def test_module_docstring_yields_candidate(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        '"""the module ships only the regex fallback at v0.5"""\n',
        encoding="utf-8",
    )
    candidates = extract_ast(tmp_path)
    assert len(candidates) == 1


def test_module_docstring_text_is_stripped(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        '"""\n  the module ships only the regex fallback at v0.5  \n"""\n',
        encoding="utf-8",
    )
    c = extract_ast(tmp_path)[0]
    assert c.text == "the module ships only the regex fallback at v0.5"


def test_module_source_is_ast_module(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        '"""docs"""\n',
        encoding="utf-8",
    )
    c = extract_ast(tmp_path)[0]
    assert c.source == "ast:m.py:module"


def test_module_without_docstring_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "x = 1\n",
        encoding="utf-8",
    )
    assert extract_ast(tmp_path) == []


# --- Function docstrings ------------------------------------------------


def test_top_level_function_docstring_yields_candidate(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        'def greet():\n    """says hello to the world"""\n    return "hi"\n',
        encoding="utf-8",
    )
    candidates = extract_ast(tmp_path)
    assert len(candidates) == 1


def test_function_source_format(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        'def greet():\n    """says hello"""\n    return "hi"\n',
        encoding="utf-8",
    )
    c = extract_ast(tmp_path)[0]
    assert c.source == "ast:m.py:func:greet"


def test_async_function_is_extracted(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        'async def fetch():\n    """async fetch from upstream"""\n    return None\n',
        encoding="utf-8",
    )
    c = extract_ast(tmp_path)[0]
    assert c.source == "ast:m.py:func:fetch"


def test_function_without_docstring_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "def greet():\n    return 'hi'\n",
        encoding="utf-8",
    )
    assert extract_ast(tmp_path) == []


def test_nested_function_is_not_extracted(tmp_path: Path) -> None:
    """Top-level functions only; nested defs are skipped per v1.0 'simple AST'."""
    (tmp_path / "m.py").write_text(
        'def outer():\n'
        '    """outer description"""\n'
        '    def inner():\n'
        '        """inner that should not be extracted"""\n'
        '        return 1\n'
        '    return inner\n',
        encoding="utf-8",
    )
    sources = [c.source for c in extract_ast(tmp_path)]
    assert "ast:m.py:func:outer" in sources
    assert not any("inner" in s for s in sources)


# --- Class docstrings ---------------------------------------------------


def test_class_docstring_yields_candidate(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        'class Belief:\n    """a unit of memory"""\n    pass\n',
        encoding="utf-8",
    )
    candidates = extract_ast(tmp_path)
    assert len(candidates) == 1


def test_class_source_format(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        'class Belief:\n    """a unit of memory"""\n    pass\n',
        encoding="utf-8",
    )
    c = extract_ast(tmp_path)[0]
    assert c.source == "ast:m.py:class:Belief"


def test_class_without_docstring_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "class Belief:\n    pass\n",
        encoding="utf-8",
    )
    assert extract_ast(tmp_path) == []


def test_method_docstrings_are_not_extracted(tmp_path: Path) -> None:
    """Methods on top-level classes are skipped per v1.0 'simple AST'."""
    (tmp_path / "m.py").write_text(
        'class Foo:\n'
        '    """class-level docstring"""\n'
        '    def method(self):\n'
        '        """method docstring that should be skipped"""\n'
        '        return 1\n',
        encoding="utf-8",
    )
    sources = [c.source for c in extract_ast(tmp_path)]
    assert "ast:m.py:class:Foo" in sources
    assert not any("method" in s for s in sources)


# --- Multiple symbols per file ------------------------------------------


def test_module_plus_function_plus_class_yield_three_candidates(
    tmp_path: Path,
) -> None:
    (tmp_path / "m.py").write_text(
        '"""the module ships the regex fallback at v0.5"""\n'
        '\n'
        'def greet():\n'
        '    """says hello to the world"""\n'
        '    return "hi"\n'
        '\n'
        'class Belief:\n'
        '    """a unit of memory in the store"""\n'
        '    pass\n',
        encoding="utf-8",
    )
    candidates = extract_ast(tmp_path)
    sources = [c.source for c in candidates]
    assert "ast:m.py:module" in sources
    assert "ast:m.py:func:greet" in sources
    assert "ast:m.py:class:Belief" in sources


# --- Skip-dirs filter ---------------------------------------------------


def test_dot_git_directory_is_skipped(tmp_path: Path) -> None:
    git = tmp_path / ".git"
    git.mkdir()
    (git / "hooks.py").write_text('"""docstring"""\n', encoding="utf-8")
    assert extract_ast(tmp_path) == []


def test_pycache_directory_is_skipped(tmp_path: Path) -> None:
    pc = tmp_path / "__pycache__"
    pc.mkdir()
    (pc / "x.py").write_text('"""docstring"""\n', encoding="utf-8")
    assert extract_ast(tmp_path) == []


def test_venv_directory_is_skipped(tmp_path: Path) -> None:
    v = tmp_path / ".venv"
    v.mkdir()
    (v / "x.py").write_text('"""docstring"""\n', encoding="utf-8")
    assert extract_ast(tmp_path) == []


# --- Recursion + ordering ----------------------------------------------


def test_nested_directories_are_walked(tmp_path: Path) -> None:
    sub = tmp_path / "src" / "pkg"
    sub.mkdir(parents=True)
    (sub / "deep.py").write_text('"""docstring"""\n', encoding="utf-8")
    candidates = extract_ast(tmp_path)
    assert len(candidates) == 1
    assert candidates[0].source == "ast:src/pkg/deep.py:module"


def test_files_returned_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text('"""b docs"""\n', encoding="utf-8")
    (tmp_path / "a.py").write_text('"""a docs"""\n', encoding="utf-8")
    sources = [c.source for c in extract_ast(tmp_path)]
    assert sources == ["ast:a.py:module", "ast:b.py:module"]


def test_repeated_extraction_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        '"""docs"""\n'
        'def foo():\n'
        '    """foo docs"""\n'
        '    pass\n',
        encoding="utf-8",
    )
    one = extract_ast(tmp_path)
    two = extract_ast(tmp_path)
    assert [c.source for c in one] == [c.source for c in two]
    assert [c.text for c in one] == [c.text for c in two]


# --- Robustness ---------------------------------------------------------


def test_syntax_error_file_is_skipped_others_kept(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text('"""good module"""\n', encoding="utf-8")
    (tmp_path / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    sources = [c.source for c in extract_ast(tmp_path)]
    assert "ast:good.py:module" in sources
    assert not any("bad.py" in s for s in sources)


def test_results_are_sentence_candidates(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text('"""docs"""\n', encoding="utf-8")
    candidates = extract_ast(tmp_path)
    assert all(isinstance(c, SentenceCandidate) for c in candidates)
