"""#1384 — sweep every codepoint's tokenisation against a real FTS5 oracle.

The #1348 changelog published a codepoint-sweep figure for the diacritic
fold. That figure was measured against the fold rule #1384 replaced, so it
no longer describes anything the code does. Per the project rule that a
published number ships the script that re-derives it, this is that script.

Three arms, all against the same in-memory ``fts5(c, tokenize='porter
unicode61')`` table read back through ``fts5vocab``:

``shipped``
    ``bm25.tokenize_stemmed`` as it stands. This is the number to publish.

``blanket``
    NFD the whole string and strip every combining mark. The implementation
    #1348 rejected, kept here because "the narrow fold is better than the
    blanket one" is the claim the entry makes and it should be checkable.

``no_fold``
    The word-class split and stem guard with no diacritic handling at all,
    which isolates what the fold itself buys.

The oracle is a real table, never a second Python model of unicode61 —
modelling the thing under test is how #1348 happened.

Usage:

    uv run python -m benchmarks.bm25_fold_codepoint_sweep
"""
from __future__ import annotations

import sqlite3
import unicodedata
from typing import Callable

from aelfrice import bm25

# 0x20..0x2FFFF less the surrogates, which are not legal scalar values.
# Fixed so the denominator is stable across runs and comparable to the
# figure this supersedes.
_LO, _HI = 0x20, 0x30000
_SURROGATES = range(0xD800, 0xE000)


def _universe() -> list[int]:
    return [c for c in range(_LO, _HI) if c not in _SURROGATES]


def _oracle(codepoints: list[int]) -> dict[int, set[str]]:
    """Ask a real FTS5 table what each codepoint tokenises to."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE t USING fts5(c, tokenize='porter unicode61')")
    for row, cp in enumerate(codepoints):
        con.execute("INSERT INTO t(rowid, c) VALUES (?, ?)", (row, chr(cp)))
    con.execute("CREATE VIRTUAL TABLE v USING fts5vocab(t, 'instance')")
    out: dict[int, set[str]] = {}
    for term, row in con.execute("SELECT term, doc FROM v"):
        out.setdefault(row, set()).add(term)
    con.close()
    return out


def _blanket(text: str) -> list[str]:
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch)
    )
    return bm25.tokenize_stemmed(stripped)


def _no_fold(text: str) -> list[str]:
    saved = bm25._fold_diacritics
    bm25._fold_diacritics = lambda s: s  # type: ignore[assignment]
    try:
        return bm25.tokenize_stemmed(text)
    finally:
        bm25._fold_diacritics = saved  # type: ignore[assignment]


ARMS: dict[str, Callable[[str], list[str]]] = {
    "shipped": bm25.tokenize_stemmed,
    "blanket": _blanket,
    "no_fold": _no_fold,
}


def main() -> int:
    codepoints = _universe()
    oracle = _oracle(codepoints)
    print(f"universe: {len(codepoints):,} codepoints "
          f"(U+{_LO:04X}..U+{_HI - 1:04X}, surrogates excluded)")
    for name, fn in ARMS.items():
        bad = sum(
            1
            for row, cp in enumerate(codepoints)
            if oracle.get(row, set()) != set(fn(chr(cp)))
        )
        print(f"  {name:<10} {bad:>7,} disagreements  "
              f"({100 * bad / len(codepoints):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
