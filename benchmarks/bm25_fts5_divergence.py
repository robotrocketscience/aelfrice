"""How far the BM25F index vocabulary is from the FTS5 one (#1348, #1158 §2).

``retrieval`` presents the BM25F and FTS5 lanes as interchangeable, and
``AELFRICE_BM25F=0`` is the documented way to debug one against the other.
That only holds if both describe the same corpus. They did not:
``beliefs_fts`` is declared ``tokenize='porter unicode61'`` while the BM25F
index tokenised with ``\\w+`` and an unguarded Porter stemmer.

This script measures the gap instead of asserting it. The oracle is a real
in-memory SQLite ``fts5`` table declared exactly as ``store.py`` declares
``beliefs_fts``, read back through ``fts5vocab(..., 'instance')`` so the
comparison is against terms SQLite actually indexed — not against a second
Python reimplementation of what SQLite is believed to do.

Cumulative arms, one change per step, so every row is attributable to
the single thing that row adds:

``legacy``
    ``\\w+`` + unguarded stem. What shipped before #1348.
``+ word class``
    the unicode61 word class (``_`` is a separator), nothing else.
``+ diacritic fold``
    adds the shipped per-character fold.
``+ byte guard``
    adds SQLite's 3..64-**byte** stemming guards. Isolates the stemmer.
``shipped``
    calls :func:`aelfrice.bm25.tokenize_stemmed` itself, so this arm tracks
    the code rather than a copy of it and goes red if the two drift.

Divergence is reported per **document**, not per vocabulary term. The
distinction is the whole finding: the stemmer disagreement covers a tiny
slice of the vocabulary but the words it covers are the most frequent in
English, so it reaches a third of all documents.

Usage::

    uv run python benchmarks/bm25_fts5_divergence.py \\
        --store .git/aelfrice/memory.db
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import snowballstemmer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Module, not `from ... import`: `arm_shipped` must resolve
# `tokenize_stemmed` through the module every call, or this arm
# snapshots the function object and stops tracking the shipped code.
import aelfrice.bm25 as bm25  # noqa: E402

# The declaration under test. Kept as a literal rather than imported so a
# change to `store.py` shows up here as a measured divergence instead of
# silently moving the oracle with the thing being measured.
FTS5_TOKENIZER = "porter unicode61"

# Documents per fts5vocab round trip. The vocab table is scanned whole, so
# batching keeps that scan bounded on a 45k-belief store.
CHUNK = 5_000

_STEMMER = snowballstemmer.stemmer("porter")

_LEGACY_PATTERN = re.compile(r"\w+", re.UNICODE)
_UNICODE61_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _stem_unguarded(token: str) -> str:
    return _STEMMER.stemWord(token)


def _stem_guarded(token: str) -> str:
    """SQLite's rule: only tokens of 3..64 UTF-8 bytes are stemmed."""
    n_bytes = len(token.encode("utf-8"))
    if n_bytes < 3 or n_bytes > 64:
        return token
    return _STEMMER.stemWord(token)


# The shipped fold, imported rather than re-spelled — a local copy is
# what let a blanket-NFD fold look correct here while it was welding
# Hebrew and Devanagari tokens together in the shipped path.
_fold = bm25._fold_diacritics


def arm_legacy(text: str) -> list[str]:
    return [
        _stem_unguarded(m.group(0).lower())
        for m in _LEGACY_PATTERN.finditer(text)
    ]


def arm_split_no_fold(text: str) -> list[str]:
    """Word class only. No fold, no guard — one change off legacy."""
    return [
        _stem_unguarded(m.group(0).lower())
        for m in _UNICODE61_PATTERN.finditer(text)
    ]


def arm_split_fold(text: str) -> list[str]:
    return [
        _stem_unguarded(m.group(0).lower())
        for m in _UNICODE61_PATTERN.finditer(_fold(text))
    ]


def arm_split_fold_guard(text: str) -> list[str]:
    return [
        _stem_guarded(m.group(0).lower())
        for m in _UNICODE61_PATTERN.finditer(_fold(text))
    ]


def arm_shipped(text: str) -> list[str]:
    """Resolved at call time, not bound at import.

    Reached through the module rather than a name bound at import, so
    rebinding `aelfrice.bm25.tokenize_stemmed` is visible here. Binding
    it would snapshot the function object and this arm would keep
    reporting the old pipeline — the opposite of what it is for.
    """
    return bm25.tokenize_stemmed(text)


# Cumulative, one change per step, so each row is attributable to the
# single thing added on that line. The earlier version folded inside the
# "split only" arm, which silently charged the fold's delta to the word
# class.
ARMS: tuple[tuple[str, Callable[[str], list[str]]], ...] = (
    ("legacy (pre-#1348)", arm_legacy),
    ("+ word class", arm_split_no_fold),
    ("+ diacritic fold", arm_split_fold),
    ("+ byte guard", arm_split_fold_guard),
    ("shipped tokenize_stemmed", arm_shipped),
)


def fts5_terms(
    conn: sqlite3.Connection, docs: dict[int, str],
) -> dict[int, list[str]]:
    """The terms SQLite indexes for each document, in offset order.

    Both virtual tables are created in ``temp`` — the three-argument
    ``fts5vocab`` form only parses there, which is why ``store.py`` does
    the same at ``_ensure_fts5_query_probe``.
    """
    conn.execute("DROP TABLE IF EXISTS temp.probe")
    conn.execute("DROP TABLE IF EXISTS temp.probe_terms")
    conn.execute(
        "CREATE VIRTUAL TABLE temp.probe "
        f"USING fts5(t, tokenize='{FTS5_TOKENIZER}')"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE temp.probe_terms "
        "USING fts5vocab('temp', 'probe', 'instance')"
    )
    conn.executemany(
        "INSERT INTO temp.probe(rowid, t) VALUES (?, ?)", list(docs.items()),
    )
    positioned: dict[int, list[tuple[int, str]]] = {}
    for term, doc, _col, offset in conn.execute(
        "SELECT term, doc, col, offset FROM temp.probe_terms"
    ):
        positioned.setdefault(int(doc), []).append((int(offset), term))
    out = {d: [t for _o, t in sorted(v)] for d, v in positioned.items()}
    for doc in docs:
        out.setdefault(doc, [])
    return out


def load_documents(store: Path, limit: int) -> dict[int, str]:
    """Indexed belief text, straight out of ``beliefs_fts``.

    Opened read-only through a URI: this reads a live store, and a bare
    `MemoryStore` open runs migrations plus the #1314 lock-expiry sweep.
    Measuring a store is not a reason to write to it.
    """
    query = "SELECT rowid, content FROM beliefs_fts ORDER BY rowid"
    params: tuple[int, ...] = ()
    if limit:
        # Bounded in SQL, not after fetchall(): a small --limit is meant
        # to make a trial run cheap, and slicing in Python would still
        # read every row. ORDER BY makes "first N" mean insertion order
        # rather than whatever the scan happens to yield.
        query += " LIMIT ?"
        params = (limit,)
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        rows: Iterable[tuple[int, str | None]] = conn.execute(
            query, params,
        ).fetchall()
    finally:
        conn.close()
    return {int(rowid): (content or "") for rowid, content in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path, required=True,
        help="path to a memory.db (read-only; .git/aelfrice/memory.db)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="only the first N beliefs (0 = all)",
    )
    parser.add_argument(
        "--show-residual", type=int, default=6,
        help="how many residual term disagreements to print per arm",
    )
    args = parser.parse_args(argv)

    docs = load_documents(args.store, args.limit)
    if not docs:
        print("no indexed beliefs in that store", file=sys.stderr)
        return 2

    conn = sqlite3.connect(":memory:")
    try:
        truth: dict[int, list[str]] = {}
        items = list(docs.items())
        for start in range(0, len(items), CHUNK):
            truth.update(fts5_terms(conn, dict(items[start:start + CHUNK])))
    finally:
        conn.close()

    total = len(docs)
    print(f"store      : {args.store}")
    print(f"sqlite     : {sqlite3.sqlite_version}")
    print(f"tokenizer  : {FTS5_TOKENIZER}")
    print(f"documents  : {total}")
    print()
    print("documents whose token list differs from the FTS5 index:")

    # The arms are ordered so `shipped` is last; this keeps its residual.
    shipped_residual: Counter[tuple[str, str]] = Counter()
    for name, tokenizer in ARMS:
        differing = 0
        residual: Counter[tuple[str, str]] = Counter()
        for doc, text in docs.items():
            got = tokenizer(text)
            want = truth[doc]
            if got == want:
                continue
            differing += 1
            only_py = set(got) - set(want)
            only_fts5 = set(want) - set(got)
            for term in sorted(only_py)[:2]:
                residual[("python-only", term)] += 1
            for term in sorted(only_fts5)[:2]:
                residual[("fts5-only", term)] += 1
        pct = 100.0 * differing / total
        print(f"  {name:26s} {differing:7d} / {total}  = {pct:6.2f}%")
        shipped_residual = residual

    if args.show_residual and shipped_residual:
        print()
        print("residual disagreements on the shipped arm:")
        for (side, term), count in shipped_residual.most_common(
            args.show_residual
        ):
            print(f"  {side:12s} {term!r:24s} {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
