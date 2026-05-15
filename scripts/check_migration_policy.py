"""Migration-policy gate for #840 — refuse destructive `_MIGRATIONS`
entries arriving without a paired reader update.

Background — #833 incident. `_MIGRATIONS` in `src/aelfrice/store.py`
gained an unconditional `ALTER TABLE beliefs DROP COLUMN
demotion_pressure` entry while the v3.1.0 reader at
`_row_to_belief` still had an unconditional `row["demotion_pressure"]`
lookup. Any installed v3.1.0 reader opening a DB previously migrated
by post-#814 code crashed with `IndexError: No item with that key`.
The bug class — destructive schema migration shipped without a paired
reader change visible in the same PR — is generic.

This script flags that class deterministically:

  1. Extract `_MIGRATIONS` from the base and head versions of
     `store.py` (AST parse + `ast.literal_eval` of the tuple).
  2. Treat `head - base` as the *added* migration entries.
  3. For each added entry, regex-match the destructive-pattern set
     (DROP COLUMN, DROP TABLE, RENAME COLUMN, RENAME TO).
  4. If any added entry is destructive, assert the same diff also
     changes `_row_to_belief` in `store.py` OR the `Belief` dataclass
     in `models.py`. Either touch satisfies the gate — the assumption
     is that the reader path or the schema row-class on the affected
     table is what protects the older reader.

`ALTER TABLE ADD COLUMN` is non-destructive — old readers don't
reference the new column. Widening retypes (the precedent at
`_maybe_retype_belief_corroborations_belief_id`) live outside the
`_MIGRATIONS` tuple (they're separate methods on `MemoryStore`) so
they pass cleanly here.

Exit codes:
  0 — clean, or no destructive entries added.
  1 — destructive entry added without paired reader update.
  2 — internal error (parse failure, missing file, etc.).
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


# Patterns that, if matched against an added `_MIGRATIONS` entry,
# require a paired reader change in the same diff. Each pattern is
# applied with `re.IGNORECASE` against the SQL statement string.
DESTRUCTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bDROP\s+COLUMN\b", "DROP COLUMN"),
    (r"\bDROP\s+TABLE\b", "DROP TABLE"),
    (r"\bRENAME\s+COLUMN\b", "RENAME COLUMN"),
    # ALTER TABLE ... RENAME TO ... — the old table name is gone.
    # Match must require ALTER TABLE in front so we don't fire on a
    # CREATE TABLE ... AS RENAME-To shape that doesn't exist in
    # SQLite but could appear in fixtures.
    (r"\bALTER\s+TABLE\s+\w+\s+RENAME\s+TO\b", "RENAME TO"),
)

# Tuple constants in store.py whose entries are the SQL ladder.
# `_MIGRATIONS` is the load-bearing list. `_POST_MIGRATION_INDEXES`
# runs after and is structurally non-destructive (CREATE INDEX IF NOT
# EXISTS only), so it is *not* checked — extending the patterns here
# would false-positive on legitimate index additions.
MIGRATION_CONST_NAME = "_MIGRATIONS"


def _extract_tuple_constant(source: str, name: str) -> tuple[str, ...]:
    """Parse `source` as Python and return the string-tuple bound to
    `name` at module scope. Raises ValueError if the assignment is
    missing or the RHS is not a string-only tuple."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id == name:
                value = node.value
                if value is None or not isinstance(value, ast.Tuple):
                    raise ValueError(
                        f"{name} is bound to {type(value).__name__}, "
                        "expected ast.Tuple of string literals"
                    )
                out: list[str] = []
                for elt in value.elts:
                    # ast.literal_eval handles `"a" "b"`-style implicit
                    # string concatenation (a Constant after parse) the
                    # same as a single string.
                    try:
                        v = ast.literal_eval(elt)
                    except (ValueError, SyntaxError) as exc:
                        raise ValueError(
                            f"{name} entry is not a string literal: "
                            f"{ast.dump(elt)}"
                        ) from exc
                    if not isinstance(v, str):
                        raise ValueError(
                            f"{name} entry resolves to {type(v).__name__}, "
                            "expected str"
                        )
                    out.append(v)
                return tuple(out)
    raise ValueError(f"{name} not found at module scope")


def _function_source(source: str, name: str) -> str | None:
    """Return the source segment of a top-level function `name`, or
    None if absent."""
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            seg = ast.get_source_segment(source, node)
            return seg
    return None


def _class_source(source: str, name: str) -> str | None:
    """Return the source segment of a top-level class `name`, or None."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            seg = ast.get_source_segment(source, node)
            return seg
    return None


def _classify_destructive(stmt: str) -> str | None:
    """Return the destructive-pattern label that matches `stmt`, or
    None if `stmt` is non-destructive."""
    for pattern, label in DESTRUCTIVE_PATTERNS:
        if re.search(pattern, stmt, re.IGNORECASE):
            return label
    return None


def check(
    *,
    base_store: str,
    head_store: str,
    base_models: str | None,
    head_models: str | None,
) -> tuple[int, str]:
    """Run the gate. Returns (exit_code, message)."""
    try:
        base_migs = set(
            _extract_tuple_constant(base_store, MIGRATION_CONST_NAME)
        )
    except ValueError as exc:
        # A base that doesn't have the constant (e.g. the very first
        # commit that introduces it) trivially passes — any added
        # entry there is "new", but there's no older reader to
        # protect either. Fail-soft.
        if "not found" in str(exc):
            base_migs = set()
        else:
            return 2, f"base store parse error: {exc}"
    try:
        head_migs = _extract_tuple_constant(head_store, MIGRATION_CONST_NAME)
    except ValueError as exc:
        return 2, f"head store parse error: {exc}"

    added = [s for s in head_migs if s not in base_migs]
    destructive_added = [
        (s, label) for s in added if (label := _classify_destructive(s))
    ]

    if not destructive_added:
        return 0, (
            f"OK: {len(added)} new migration entr{'y' if len(added) == 1 else 'ies'}"
            f", none destructive"
            if added
            else "OK: no new migration entries"
        )

    # Reader-pair check. The destructive entry passes if EITHER
    # `_row_to_belief` changed OR the `Belief` dataclass changed
    # between base and head.
    base_reader = _function_source(base_store, "_row_to_belief")
    head_reader = _function_source(head_store, "_row_to_belief")
    reader_changed = base_reader != head_reader

    belief_changed = False
    if base_models is not None and head_models is not None:
        base_belief = _class_source(base_models, "Belief")
        head_belief = _class_source(head_models, "Belief")
        belief_changed = base_belief != head_belief

    if reader_changed or belief_changed:
        labels = ", ".join(sorted({label for _, label in destructive_added}))
        return 0, (
            f"OK: {len(destructive_added)} destructive entr"
            f"{'y' if len(destructive_added) == 1 else 'ies'} added ({labels}); "
            f"paired reader change present "
            f"(_row_to_belief changed={reader_changed}, "
            f"Belief dataclass changed={belief_changed})"
        )

    # Unpaired destructive — emit one line per entry for clarity.
    lines = [
        "FAIL: destructive _MIGRATIONS entry without paired reader update.",
        "Each destructive entry below requires the same PR to also change",
        "either _row_to_belief (src/aelfrice/store.py) or the Belief",
        "dataclass (src/aelfrice/models.py). See #840 for rationale.",
        "",
    ]
    for stmt, label in destructive_added:
        lines.append(f"  [{label}] {stmt}")
    return 1, "\n".join(lines)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-store", required=True, type=Path)
    parser.add_argument("--head-store", required=True, type=Path)
    parser.add_argument("--base-models", type=Path, default=None)
    parser.add_argument("--head-models", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        base_store = _read(args.base_store)
        head_store = _read(args.head_store)
        base_models = _read(args.base_models) if args.base_models else None
        head_models = _read(args.head_models) if args.head_models else None
    except FileNotFoundError as exc:
        print(f"check_migration_policy: missing input: {exc}", file=sys.stderr)
        return 2

    exit_code, message = check(
        base_store=base_store,
        head_store=head_store,
        base_models=base_models,
        head_models=head_models,
    )
    print(message)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
