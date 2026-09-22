"""The posterior-writer manifest must be complete, and must be able to fail.

#1592 AC2 asks which paths can write `beliefs.alpha` / `beliefs.beta`.
`scripts/check_posterior_writers.py` answers it from the AST and gates the
answer, because #1359 established that a one-off enumeration of this shape
decays: it surfaced a new unguarded writer every review round, since nothing
stopped one being added between rounds.

A gate that cannot fail is worth nothing, so the arms below add each writer
shape to a synthetic tree and confirm the check reds. The tree's own writer
set is checked too — that is the arm that fires when someone adds a writer
and does not declare it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_posterior_writers.py"

_spec = importlib.util.spec_from_file_location("_posterior_writers", _SCRIPT)
assert _spec and _spec.loader
# `Any` for the reason `tests/test_soak_producer_figures.py` gives: pyright
# runs `tests/` in strict mode and an implicitly-typed module object turns
# every attribute read into an `Unknown`.
check: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    src = tmp_path / "aelfrice"
    src.mkdir(exist_ok=True)
    (src / name).write_text(body, encoding="utf-8")
    return src


def test_every_writer_in_the_tree_is_declared() -> None:
    """The live arm. Fires when a writer is added without a manifest entry."""
    found = check.scan(check.SRC)
    undeclared = sorted(set(found) - set(check.MANIFEST))
    assert not undeclared, (
        "undeclared posterior writer(s); add each to MANIFEST in "
        f"{_SCRIPT.name} with what it does to the posterior: {undeclared}"
    )


def test_no_declared_writer_has_vanished() -> None:
    """The other half: a stale entry hides that a writer was removed."""
    found = check.scan(check.SRC)
    vanished = sorted(set(check.MANIFEST) - set(found))
    assert not vanished, (
        f"MANIFEST declares writer(s) that no longer exist: {vanished}"
    )


def test_the_checker_passes_on_the_tree_as_it_stands() -> None:
    assert check.main([]) == 0


def test_main_returns_nonzero_when_a_writer_is_undeclared(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """`main` must actually fail, not merely print.

    This is what CI runs, and nothing else asserted its failing exit. An
    adversarial review mutated `main` to `return 0` unconditionally and the
    whole file stayed green — the gate would have been wired into a job
    that could never go red.
    """
    src = _write(
        tmp_path, "sneaky.py",
        'def repair(conn):\n'
        '    conn.execute("UPDATE beliefs SET alpha = 1.0 WHERE id = ?")\n',
    )
    monkeypatch.setattr(check, "SRC", src)

    assert check.main([]) == 1


def test_main_returns_nonzero_when_a_declared_writer_vanishes(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """The other exit path: a stale entry hides that a writer was removed."""
    src = _write(tmp_path, "empty.py", "def nothing():\n    return 1\n")
    monkeypatch.setattr(check, "SRC", src)

    assert check.main([]) == 1


def test_main_reports_a_missing_source_tree(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    monkeypatch.setattr(check, "SRC", tmp_path / "does-not-exist")
    assert check.main([]) == 2


def test_a_new_update_beliefs_statement_is_detected(tmp_path: Path) -> None:
    src = _write(
        tmp_path, "sneaky.py",
        'def repair(conn):\n'
        '    conn.execute("UPDATE beliefs SET alpha = 1.0 WHERE id = ?")\n',
    )
    found = check.scan(src)
    assert "sneaky.py::repair::sql" in found, found


def test_a_new_update_belief_call_is_detected(tmp_path: Path) -> None:
    src = _write(
        tmp_path, "sneaky.py",
        'def retag(store, b):\n'
        '    b.type = "factual"\n'
        '    store.update_belief(b)\n',
    )
    found = check.scan(src)
    assert "sneaky.py::retag::update_belief" in found, found


def test_sql_in_an_fstring_is_detected(tmp_path: Path) -> None:
    """An f-string is a `JoinedStr`, not a `Constant`.

    Reading only `ast.Constant` would miss a writer that interpolates a
    WHERE clause while naming the column in its literal text.
    """
    src = _write(
        tmp_path, "sneaky.py",
        'def repair(conn, where):\n'
        '    conn.execute(f"UPDATE beliefs SET alpha = 1.0 WHERE {where}")\n',
    )
    found = check.scan(src)
    assert "sneaky.py::repair::sql" in found, found


def test_sql_in_a_docstring_is_not_a_writer(tmp_path: Path) -> None:
    """The reason this is an AST walk and not a grep.

    `clamp_ghosts.py` documents the SQL that reverses a clamp inside its
    module docstring. A regex sweep counts it as a writer; declaring it
    would put a path in the manifest that does not exist, and a later
    reader would go looking for it.
    """
    src = _write(
        tmp_path, "documented.py",
        '"""Reverse a clamp with:\n'
        '\n'
        '    UPDATE beliefs SET alpha = alpha + 1 WHERE id = ?\n'
        '"""\n'
        'def nothing():\n'
        '    return 1\n',
    )
    assert check.scan(src) == {}


def test_a_statement_not_touching_the_posterior_is_ignored(
    tmp_path: Path,
) -> None:
    """`UPDATE beliefs SET lock_level = ?` is not a posterior writer.

    Without this the manifest would fill with every column write in the
    store and stop being read.
    """
    src = _write(
        tmp_path, "other.py",
        'def relock(conn):\n'
        '    conn.execute("UPDATE beliefs SET lock_level = ? WHERE id = ?")\n',
    )
    assert check.scan(src) == {}


def test_sql_passed_as_a_keyword_is_detected(tmp_path: Path) -> None:
    """`execute(sql=...)` is a writer too.

    Reading only `node.args` makes the keyword form structurally invisible,
    which is a one-word evasion of the whole gate.
    """
    src = _write(
        tmp_path, "kw.py",
        'def repair(conn):\n'
        '    conn.execute(sql="UPDATE beliefs SET alpha = 1.0")\n',
    )
    assert "kw.py::repair::sql" in check.scan(src)


def test_a_quoted_or_qualified_table_name_is_detected(tmp_path: Path) -> None:
    """SQLite accepts `"beliefs"` and `main.beliefs`; so must the gate.

    `store.py` records that real stores exist carrying a quoted
    `CREATE TABLE IF NOT EXISTS "beliefs"`, so these spellings are not
    hypothetical.
    """
    src = _write(
        tmp_path, "quoted.py",
        'def a(conn):\n'
        '    conn.execute(\'UPDATE "beliefs" SET alpha = 1.0\')\n'
        'def b(conn):\n'
        '    conn.execute("UPDATE main.beliefs SET beta = 1.0")\n',
    )
    found = check.scan(src)
    assert "quoted.py::a::sql" in found, found
    assert "quoted.py::b::sql" in found, found


def test_the_fts_and_rebuild_sibling_tables_are_not_posterior_writers(
    tmp_path: Path,
) -> None:
    """`beliefs_fts` starts with `beliefs` and carries no posterior.

    A substring test on "insert into beliefs" matches it and fills the
    manifest with writes that cannot move a mean.
    """
    src = _write(
        tmp_path, "fts.py",
        'def index(conn, alpha):\n'
        '    conn.execute("INSERT INTO beliefs_fts (id, content) '
        'VALUES (?, ?)", (1, alpha))\n',
    )
    assert check.scan(src) == {}


def test_the_documented_evasions_stay_documented() -> None:
    """The gate is a backstop, not a proof, and must keep saying so.

    An adversarial review enumerated ten ways to write a posterior that
    this check cannot see; nine work. They are named in the module
    docstring so nobody reads a green run as "no writer was added". If
    the caveat is deleted, this fails.
    """
    doc = check.__doc__ or ""
    assert "cannot see" in doc, "the evasion caveat was removed"
    for shape in ("concatenation", "partial", "getattr", "constant"):
        assert shape in doc, f"the {shape} evasion is no longer named"


def test_every_manifest_entry_has_an_effect_and_a_reason() -> None:
    """An entry with no stated effect silences the gate without answering it."""
    valid = {check.CREATES, check.MOVES, check.SUMS, check.IDENTITY,
             check.INERT}
    for key, (effect, why) in check.MANIFEST.items():
        assert effect in valid, f"{key} has unknown effect {effect!r}"
        assert len(why) > 20, f"{key} has no real explanation: {why!r}"


def test_the_undetectable_list_is_not_silently_empty() -> None:
    """The manifest must not read as exhaustive when it is not.

    One writer in the tree builds its column list at runtime, so no static
    check can classify it by content. It is recorded rather than omitted.
    """
    assert check.UNDETECTABLE, (
        "UNDETECTABLE was emptied; if the dynamic-SQL writer really is gone, "
        "say so here rather than deleting the record that it existed"
    )
    for name, effect, why in check.UNDETECTABLE:
        assert "::" in name
        assert effect
        assert len(why) > 20
