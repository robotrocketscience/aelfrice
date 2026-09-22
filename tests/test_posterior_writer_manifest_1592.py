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


def _write(tmp_path: Path, name: str, body: str) -> Path:
    src = tmp_path / "aelfrice"
    src.mkdir(exist_ok=True)
    (src / name).write_text(body, encoding="utf-8")
    return src


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
