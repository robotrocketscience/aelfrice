"""Re-derive the two live-store figure sets #1368 publishes.

Standing rule: a published number ships the script that re-derives it.
`CHANGELOG/v4.md`'s #1368 entry quotes two independent measurements taken
against a live repo-local store, and this is their derivation.

**Arm 1 — the negation right-hand bound (`correction.py`).**
#1159 §5 replaced substring containment with word boundaries. The trailing
spaces in `"not "` / `"no "` were doing two jobs, and only one of them was
the bug: they wrongly bounded the *left* edge (matching inside "pia**no**
is"), but they also bounded the *right* edge, and `\\b` does not bound it the
same way — `\\b` matches before a hyphen where a space did not. So the
straight swap to `\\b` on both sides is not the pure narrowing it looks like.
This arm counts the three-way split: how many beliefs newly fire negation
under plain `\\b`, how many of those are hyphen compounds (false positives),
and how many are real negations the trailing-space form had been missing.

**Arm 2 — the enum member boundary (`value_compare.py`).**
#1159 §13: `"deterministic"` matched inside `"non-deterministic"`, so one
belief tagged both groups of the `determinism` category and `find_conflicts`
short-circuited on its group-disjointness test. Three candidate fixes were
measured before one was chosen; this arm reproduces that table. Two of the
three are *not* what shipped, and they are re-derived here precisely because
the argument for the shipped fix is that the other two cost more than the
defect.

The store is opened `mode=ro` through a bare `sqlite3` connection and never
through `MemoryStore`, whose `__init__` documents that a bare open is a write
(DDL, migrations, a scope-id mint and an expiry sweep). A measurement script
must not mutate the corpus it measures.

**The numbers move with the store.** These are counts over one operator's
live belief set, not a fixed corpus, so they drift as beliefs accumulate.
`n_active` is printed on every run and should be quoted alongside any figure
taken from it. The CHANGELOG figures were taken at n=44,683.

Usage::

    uv run python benchmarks/classifier_boundary_1368.py
    uv run python benchmarks/classifier_boundary_1368.py --db /path/to/memory.db --json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from aelfrice.value_compare import _ENUM_MEMBER_INDEX, _extract_enums

# --- Arm 1: negation ---------------------------------------------------

# The negation vocabulary, with `"stop"` excluded. `"stop"` leaving the
# negation list is #1159 §4's fix and a separate change; including it here
# would fold two effects into one column.
_NEG_TERMS: tuple[str, ...] = ("do not", "don't", "dont", "not", "no more", "no")


def _neg_pattern(right_boundary: str) -> re.Pattern[str]:
    """Compile the negation alternation with a chosen right-hand bound.

    Mirrors `correction._boundary_alternation`. Kept as a local replica
    rather than an import so the three arms can be built side by side; the
    shipped arm is cross-checked against the real `_NEGATION_RE` below.
    """
    parts: list[str] = []
    for term in sorted(_NEG_TERMS, key=len, reverse=True):
        pattern = re.escape(term)
        if term[:1].isalnum() or term[:1] == "_":
            pattern = r"\b" + pattern
        if term[-1:].isalnum() or term[-1:] == "_":
            pattern = pattern + right_boundary
        parts.append(pattern)
    return re.compile("(?:" + "|".join(parts) + ")")


def _pre_fix_pattern() -> re.Pattern[str]:
    """The pre-#1368 form: bare containment, trailing space on bare terms."""
    literal = [t + " " if t in ("not", "no") else t for t in _NEG_TERMS]
    return re.compile("|".join(re.escape(t) for t in literal))


def negation_arm(texts: list[str]) -> dict[str, Any]:
    before = _pre_fix_pattern()
    plain = _neg_pattern(r"\b")
    shipped = _neg_pattern(r"(?![\w-])")

    from aelfrice.correction import _NEGATION_RE

    if shipped.pattern != _NEGATION_RE.pattern:
        raise SystemExit(
            "replica drifted from correction._NEGATION_RE:\n"
            f"  replica: {shipped.pattern}\n  shipped: {_NEGATION_RE.pattern}"
        )

    newly = [t for t in texts if plain.search(t) and not before.search(t)]
    kept = [t for t in newly if shipped.search(t)]
    dropped = [t for t in newly if not shipped.search(t)]

    contexts: dict[str, int] = {}
    for t in dropped:
        m = plain.search(t)
        assert m is not None
        key = f"{m.group(0).lower()}{t[m.end() : m.end() + 1]}"
        contexts[key] = contexts.get(key, 0) + 1

    return {
        "newly_fire_under_plain_b": len(newly),
        "hyphen_false_positives_dropped": len(dropped),
        "real_negations_kept": len(kept),
        "dropped_contexts": dict(
            sorted(contexts.items(), key=lambda kv: -kv[1])[:8]
        ),
        "examples_dropped": [t[:90] for t in dropped[:3]],
        "examples_kept": [t[:90] for t in kept[:3]],
    }


# --- Arm 2: enum member boundary --------------------------------------

# Imported rather than rebuilt: a second copy of the member→(category,
# group) mapping is exactly the drift this script exists to rule out.
_MEMBER_INDEX = _ENUM_MEMBER_INDEX


def _tags(text: str, left: str, right: str) -> set[tuple[str, str]]:
    """Baseline/variant extraction: every member tested independently."""
    lowered = text.lower()
    out: set[tuple[str, str]] = set()
    for member, (category, _group) in _MEMBER_INDEX.items():
        if re.search(rf"(?<!{left}){re.escape(member)}(?!{right})", lowered):
            out.add((category, member))
    return out


def _shipped_tags(text: str) -> set[tuple[str, str]]:
    return {(s.category, s.member) for s in _extract_enums(text)}


def _groups_by_category(
    tags: set[tuple[str, str]],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for category, member in tags:
        out.setdefault(category, set()).add(_MEMBER_INDEX[member][1])
    return out


def enum_arm(texts: list[str]) -> dict[str, Any]:
    plain = r"[A-Za-z0-9_]"
    both = r"[A-Za-z0-9_\-]"

    variants = {
        "hyphen_both_sides": lambda t: _tags(t, both, both),
        "hyphen_left_only": lambda t: _tags(t, both, plain),
        "longest_match_shipped": _shipped_tags,
    }

    baseline = [_tags(t, plain, plain) for t in texts]
    base_groups = [_groups_by_category(b) for b in baseline]

    results: dict[str, Any] = {}
    for name, fn in variants.items():
        changed = destroyed = fixed = 0
        for text, base, bg in zip(texts, baseline, base_groups, strict=True):
            var = fn(text)
            if var == base:
                continue
            changed += 1
            vg = _groups_by_category(var)
            for category, groups in bg.items():
                if category not in vg:
                    # every tag in this category is gone
                    destroyed += 1
                elif len(groups) > 1 and len(vg[category]) == 1:
                    # the category no longer spans two groups off one belief,
                    # which is the defect #1159 §13 names
                    fixed += 1
        results[name] = {
            "beliefs_changed": changed,
            "intra_category_fixes": fixed,
            "whole_category_tags_destroyed": destroyed,
        }
    return results


# --- driver ------------------------------------------------------------


def _default_db() -> Path:
    """The repo-local store, resolved the way the product resolves it.

    `db_paths.db_path()` walks to the git *common* dir, so this is correct
    from inside a linked worktree, where `.git` is a file and the store
    lives under the main checkout. Importing `db_paths` does not open a
    store; only `_open_store()` does, and this script never calls it.
    """
    from aelfrice.db_paths import db_path

    return db_path()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="path to a repo-local memory.db (default: this repo's)",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    db = args.db or _default_db()
    if not db.exists():
        print(f"no store at {db}", file=sys.stderr)
        return 2

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        texts = [
            row[0]
            for row in con.execute(
                "SELECT content FROM beliefs "
                "WHERE valid_to IS NULL AND content IS NOT NULL"
            )
        ]
    finally:
        con.close()

    report = {
        "db": str(db),
        "n_active": len(texts),
        "negation_right_bound": negation_arm(texts),
        "enum_member_boundary": enum_arm(texts),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    neg = report["negation_right_bound"]
    print(f"store: {db}")
    print(f"active beliefs: {report['n_active']}\n")
    print("-- arm 1: negation right-hand bound (#1159 §5) --")
    print(f"  newly fire under plain \\b ....... {neg['newly_fire_under_plain_b']}")
    print(f"    hyphen false positives dropped  {neg['hyphen_false_positives_dropped']}")
    print(f"    real negations kept ..........  {neg['real_negations_kept']}")
    print("  top dropped contexts:")
    for ctx, n in neg["dropped_contexts"].items():
        print(f"    {n:5d}  {ctx!r}")
    print("\n-- arm 2: enum member boundary (#1159 §13) --")
    print(f"  {'variant':24s} {'changed':>8s} {'fixes':>7s} {'destroyed':>10s}")
    for name, row in report["enum_member_boundary"].items():
        print(
            f"  {name:24s} {row['beliefs_changed']:8d} "
            f"{row['intra_category_fixes']:7d} "
            f"{row['whole_category_tags_destroyed']:10d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
