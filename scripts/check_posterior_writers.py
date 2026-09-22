#!/usr/bin/env python3
"""Gate the set of code paths that can write `beliefs.alpha` / `beliefs.beta` (#1592).

#1592 AC2 asks which writers move a belief's posterior, and #1359 established
why a one-off answer is worth little: that enumeration surfaced a new unguarded
writer every review round, because nothing in the tree stopped one being added.
This is the stopper. It enumerates the writers from the AST and fails when the
set changes without the manifest changing with it.

Two classes are detected:

1. **SQL** — a string passed to a call (so `conn.execute("UPDATE beliefs ...")`)
   that writes the `beliefs` table and names `alpha` or `beta`.
2. **API** — a call to a store method that carries a posterior:
   `insert_belief`, `update_belief`, `bump_posterior`.

Only **literal** strings in argument or keyword position count, which is the
point of using the AST rather than grep. `clamp_ghosts.py` documents the SQL
that reverses a clamp inside its module docstring; a regex sweep reports it as
a sixth writer, and adding it to the manifest would be recording a writer that
does not exist.

**What this check cannot see.** It is a backstop against accidental additions,
not a proof. An adversarial review enumerated ten evasion shapes and nine are
invisible to it: SQL built by concatenation, `%`, or `.format()`; SQL held in a
module-level constant and passed by name; a call through an alias,
`functools.partial`, `getattr`, or a local holding a bound method; and a
schema-qualified or quoted table name reached by a path the regex below misses.
Three of those shapes already exist in this tree for *other* tables —
`store.py`'s `_BACKFILL_STATEMENTS` constant, `doctor.py`'s concatenated
SELECT, `store.py`'s `" ".join(sql_parts)` — so they are realistic, not
hypothetical. None writes a posterior today; the gate's value is that adding
one the ordinary way now fails CI, and the failure mode to watch for is a
writer added the extraordinary way.

`update_belief` is in class 2 because it is a whole-row write of an in-memory
snapshot: any caller that loads a belief, changes an unrelated field, and writes
it back carries `alpha`/`beta` along. That is not hypothetical — it is the shape
#1168 found losing 180 of 240 concurrent feedback events. Each call site is
declared with whether it can move a posterior.

Usage:
    check_posterior_writers.py [--dry-run] [--list]

    --dry-run  report what would be checked, and exit 0 without failing
    --list     print every detected writer with its manifest entry

Exit codes: 0 clean, 1 manifest drift, 2 bad invocation.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "aelfrice"

#: Store methods whose arguments carry a posterior into the database.
POSTERIOR_API = frozenset({"insert_belief", "update_belief", "bump_posterior"})

#: How a writer affects the posterior. The distinction matters because
#: retrieval blends `log(posterior_mean)`: a writer that only adds mass
#: (dedupe summing `n` copies of one prior) is invisible to ranking, while
#: one that moves the mean is not.
CREATES = "creates"    # writes an insertion prior on a new row
MOVES = "moves"        # can change the posterior mean of an existing belief
SUMS = "sums"          # adds mass, mean unchanged
IDENTITY = "identity"  # copies alpha/beta unchanged (table rebuild, migration)
INERT = "inert"        # carries a posterior but cannot change a live one

#: The declared writer set. A key is `<relpath>::<enclosing def>::<kind>`,
#: keyed on the enclosing function rather than a line number so ordinary
#: edits above a writer do not churn this file.
#:
#: Adding an entry here is a deliberate act. If this gate fails because you
#: added a writer, say in the entry what the writer does to the posterior and
#: why that is correct — do not just silence it.
MANIFEST: dict[str, tuple[str, str]] = {
    # --- the store's own SQL -------------------------------------------
    "store.py::insert_belief::sql": (
        CREATES,
        "the single creation path; alpha/beta come from the Belief, which "
        "carries whatever prior the deriving lane chose",
    ),
    "store.py::update_belief::sql": (
        MOVES,
        "whole-row write of an in-memory snapshot. Every caller is declared "
        "below; a new one is a new posterior writer whether it means to be "
        "or not (#1168)",
    ),
    "store.py::bump_posterior::sql": (
        MOVES,
        "the intended evidence path. `SET alpha = alpha + ?` evaluated by "
        "SQLite, so concurrent writers cannot lose each other's events",
    ),
    "store.py::_maybe_consolidate_content_hash_duplicates::sql": (
        SUMS,
        "collapses a duplicate group onto the canonical row by SUMMING the "
        "group's alphas and betas, so n copies of one prior land at n*prior "
        "— off the insertion grid, n times the mass, mean unchanged. 2,392 "
        "beliefs on the measured host are this",
    ),
    "store.py::insert_or_corroborate::insert_belief": (
        CREATES,
        "delegates to store.py::insert_belief on the miss path, so it is "
        "the same write counted twice — declared because the gate keys on "
        "call sites. A hit records a corroboration and does NOT move the "
        "posterior",
    ),

    # --- outside the store ---------------------------------------------
    "clamp_ghosts.py::clamp_ghost_alphas::sql": (
        MOVES,
        "UPDATE beliefs SET alpha = ? with a LOWER alpha. The only writer "
        "that DECREASES a posterior, so it can push a moved belief back "
        "onto an insertion prior and make it read as never touched. Reached "
        "by `aelf clamp-ghosts --apply`",
    ),
    "feedback.py::apply_feedback::bump_posterior": (
        MOVES,
        "the evidence path. Not the only INTENTIONAL mover — clamp_ghosts "
        "is the other — but the only one that moves a posterior on "
        "evidence. Skipped entirely when update_posterior=False, which "
        "since #1086 is the default for retrieval exposure — the reason "
        "6,220 beliefs carry feedback events and sit on their prior",
    ),

    # --- callers of update_belief: each carries alpha/beta along --------
    "hook.py::_autolock_candidates::update_belief": (
        MOVES,
        "rewrites origin to user_stated and re-locks; the posterior rides "
        "along on the whole-row write. NOT a default path: gated on "
        "AELF_AUTOLOCK_CORRECTIONS, which ships off — the prompt-instead-"
        "of-lock branch is what runs unless the user opts in",
    ),
    "promotion.py::promote::update_belief": (
        MOVES,
        "promotion rewrites origin/lock and carries alpha/beta through",
    ),
    "promotion.py::devalidate::update_belief": (
        MOVES,
        "flips origin user_validated -> agent_inferred and writes the "
        "whole row back. Audits at valence 0.0, which apply_feedback "
        "rejects, so the posterior is carried unchanged rather than moved",
    ),
    "promotion.py::unlock::update_belief": (
        MOVES,
        "clears lock_level/locked_at/lock_expires_at and writes the whole "
        "row back; audits at valence 0.0, so the posterior rides along "
        "unchanged rather than being updated",
    ),
    "review.py::apply_decisions::update_belief": (
        MOVES,
        "carries the snapshot's posterior back on a review decision",
    ),
    "doctor.py::classify_orphans::update_belief": (
        MOVES,
        "orphan reclassification; posterior rides along",
    ),
    "cli.py::_cmd_lock::update_belief": (
        MOVES,
        "four call sites in one function (resolve, tier, two window "
        "writes). The key is per-function, so a fifth added here does NOT "
        "trip this gate — read the function, not just the manifest",
    ),
    "cli.py::_apply_scope_change::update_belief": (
        MOVES,
        "flips the scope field and writes a zero-valence audit row. Does "
        "not touch origin or lock, but the whole-row write still carries "
        "alpha/beta from the snapshot it read",
    ),

    # --- creation and copy paths ---------------------------------------
    "migrate.py::migrate::insert_belief": (
        IDENTITY,
        "copies alpha/beta verbatim from a legacy store. UNBOUNDED: a "
        "migrated store can carry a pair on no insertion-prior grid, which "
        "is why the #1592 sweep reports unknown pairs rather than counting "
        "them as evidence",
    ),
    "wonder/lifecycle.py::wonder_ingest::insert_belief": (
        CREATES,
        "speculative ingest at _INGEST_ALPHA/_INGEST_BETA = (0.3, 1.0) — "
        "the pair #1592 first logged as UNVERIFIED",
    ),

    # --- not production stores -----------------------------------------
    "wonder/simulator.py::populate_store::insert_belief": (
        INERT,
        "wonder simulation fixture; builds a throwaway store and never "
        "touches a production one",
    ),
    "eval_harness.py::build_calibration_store::insert_belief": (
        INERT,
        "builds the calibration fixture store used by the eval harness; "
        "not a production write path",
    ),
    "benchmark.py::seed_corpus::insert_belief": (
        INERT,
        "seeds a synthetic corpus at the Jeffreys prior (1.0, 1.0). "
        "`aelf bench synthetic --db PATH` opens PATH read-write and seeds "
        "it: the help says 'an empty SQLite file' but nothing enforces "
        "that, so this is inert only by convention, not by construction",
    ),
    "benchmark.py::seed_multihop_corpus::insert_belief": (
        INERT,
        "seeds the multihop benchmark corpus at the Jeffreys prior; "
        "never runs against a user store",
    ),
}

#: Writers this gate provably cannot see, recorded so the manifest is not
#: read as exhaustive. `store.py::_rebuild_beliefs_table` issues
#: `f"INSERT INTO beliefs_new ({col_list}) SELECT {col_list} FROM beliefs"`,
#: where the column list is computed at runtime — the literal parts of the
#: f-string never name alpha or beta, so no static check can classify it by
#: content. It is a column-for-column copy during a schema rebuild and so
#: carries the posterior through unchanged.
UNDETECTABLE: tuple[tuple[str, str, str], ...] = (
    (
        "store.py::_rebuild_beliefs_table::copy",
        IDENTITY,
        "f-string with a runtime column list, so the literal text never "
        "names alpha or beta; a column-for-column copy that carries the "
        "posterior through unchanged",
    ),
    (
        "store.py::_rebuild_beliefs_table::create",
        IDENTITY,
        "the same function builds `CREATE TABLE beliefs_new (... alpha REAL "
        "NOT NULL ...)` by concatenation and executes it as a variable, so "
        "the column declaration is invisible here too. Schema DDL, not a "
        "value write",
    ),
)


#: `beliefs` written as SQLite will accept it: bare, quoted three ways, and
#: schema-qualified. A bare `in` test on `"update beliefs"` misses every form
#: but the first, and `store.py` records that real stores exist carrying a
#: quoted `CREATE TABLE IF NOT EXISTS "beliefs"`.
_TABLE = r'(?:(?:main|temp)\s*\.\s*)?["\'`\[]?beliefs["\'`\]]?'
_WRITES_RE = re.compile(
    rf'\b(?:update\s+{_TABLE}|insert\s+(?:or\s+\w+\s+)?into\s+{_TABLE})\b',
    re.IGNORECASE,
)
#: Sibling tables whose names start with `beliefs`. Matching them would fill
#: the manifest with FTS and rebuild writes that carry no posterior.
_NOT_THE_TABLE = re.compile(r'\bbeliefs_(?:fts|new)\b', re.IGNORECASE)


def _writes_beliefs_posterior(sql: str) -> bool:
    """True when `sql` writes the `beliefs` table and names alpha or beta.

    Both halves are required. `UPDATE beliefs SET lock_level = ?` is a write
    to the table and not to a posterior; `INSERT INTO beliefs_fts` names
    neither. Without the column test the manifest would fill with every
    column write in the store and stop being read.
    """
    lowered = " ".join(sql.split())
    if not re.search(r"\b(alpha|beta)\b", lowered, re.IGNORECASE):
        return False
    stripped = _NOT_THE_TABLE.sub(" ", lowered)
    return _WRITES_RE.search(stripped) is not None


def _literal_sql(node: ast.expr) -> str | None:
    """The literal text of a string argument, f-strings included.

    An f-string reaches the AST as a `JoinedStr` whose interpolations are
    opaque here, so only its literal parts are returned. That is enough
    whenever the statement names `alpha` or `beta` in the literal text,
    and not enough when the column list itself is interpolated — see
    `UNDETECTABLE`, which records the one such writer in the tree rather
    than letting the manifest read as exhaustive.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = [
            v.value for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
        return " ".join(parts) if parts else None
    return None


class _Walker(ast.NodeVisitor):
    """Collect posterior writers, tracking the enclosing function name."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.stack: list[str] = []
        self.found: dict[str, int] = {}

    def _enclosing(self) -> str:
        return self.stack[-1] if self.stack else "<module>"

    def _record(self, kind: str, lineno: int) -> None:
        self.found[f"{self.relpath}::{self._enclosing()}::{kind}"] = lineno

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        # Class 2: a store method that carries a posterior.
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in POSTERIOR_API:
                self._record(node.func.attr, node.lineno)
        # Class 1: SQL in argument position only. A module or function
        # docstring is an ast.Expr, never a Call argument, so it cannot
        # reach here — which is the whole reason this is an AST walk.
        # Keywords as well as positional args: `execute(sql=...)` is a
        # structurally invisible writer if only `node.args` is read.
        candidates: list[ast.expr] = list(node.args)
        candidates += [kw.value for kw in node.keywords]
        for arg in candidates:
            sql = _literal_sql(arg)
            if sql is not None and _writes_beliefs_posterior(sql):
                self._record("sql", arg.lineno)
        self.generic_visit(node)


def scan(root: Path) -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        walker = _Walker(rel)
        walker.visit(ast.parse(path.read_text(encoding="utf-8")))
        found.update(walker.found)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if not SRC.is_dir():
        print(f"error: no source tree at {SRC}", file=sys.stderr)
        return 2

    found = scan(SRC)

    if args.list or args.dry_run:
        for key in sorted(found):
            effect, why = MANIFEST.get(key, ("UNDECLARED", ""))
            print(f"{effect:>8}  {key}:{found[key]}")
            if why:
                print(f"          {why}")
        print(f"\n{len(found)} writers detected, {len(MANIFEST)} declared")
        if args.dry_run:
            return 0

    undeclared = sorted(set(found) - set(MANIFEST))
    vanished = sorted(set(MANIFEST) - set(found))

    if undeclared:
        print(
            f"\n{len(undeclared)} UNDECLARED posterior writer(s). Every path "
            "that can write beliefs.alpha or beliefs.beta must be declared "
            "in MANIFEST with what it does to the posterior — #1592 AC2, "
            "and #1359 for why a partial list is worth nothing:",
            file=sys.stderr,
        )
        for key in undeclared:
            print(f"  {key}:{found[key]}", file=sys.stderr)

    if vanished:
        print(
            f"\n{len(vanished)} declared writer(s) no longer found. If one "
            "was removed, delete its MANIFEST entry in the same commit; if "
            "one was renamed, update the key:",
            file=sys.stderr,
        )
        for key in vanished:
            print(f"  {key}", file=sys.stderr)

    if undeclared or vanished:
        return 1
    print(f"{len(found)} posterior writers, all declared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
