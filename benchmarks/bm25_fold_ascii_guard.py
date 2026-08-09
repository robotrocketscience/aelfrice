"""What the ASCII fast path in `_fold_diacritics` costs and saves (#1387).

`bm25._fold_diacritics` is a per-character Python loop calling
``unicodedata.normalize("NFD", ch)`` on every character of every document,
and it runs inside `tokenize_stemmed`, which runs over the whole corpus on
`BM25Index.build`. #1387 adds a one-line ASCII fast path. This re-derives
what that is worth instead of asserting it.

Two arms, one change between them:

``unguarded``
    the loop with no fast path — what shipped between #1348 and #1387.
``shipped``
    `aelfrice.bm25.tokenize_stemmed` as it stands, reached through the
    module so this arm tracks the code rather than a copy of it.

The `unguarded` arm is a local reimplementation for one reason: after the
fix ships there is no other way to re-derive the before-figure. It is
deliberately a duplicate of the loop and **not** a monkeypatched constant —
what it must reproduce is the pre-#1387 behaviour, which no longer exists in
the tree. Its output is checked against the shipped fold on every document
before timing starts, so an arm that has drifted reports a drift rather than
a fabricated speedup.

Timing is **min-of-N**, not mean: this measures a floor for a deterministic
CPU-bound loop, so on a busy machine the mean measures the other work rather
than the fold. `--repeat` controls N.

Three figures are reported because they answer different questions:

``fold``
    `tokenize_stemmed` over every document, arm by arm. The clean
    attribution — nothing else differs between the arms.
``build``
    a real `BM25Index.build` off a read-only handle to the same store.
    In-process and warm.
``cold``
    the same build in a **fresh interpreter**, wall clock from process
    start: interpreter startup, imports, store open, build. This is the
    path the hook budget in `src/aelfrice/data/hook_manifest.json` is
    sized against, and the only one #1387's headroom claim can be made
    about. Off by default because it is the slow arm; `--cold N` runs it.

Report the document share **and** the character share, because they
disagree and the second is the one that predicts the saving. The fast
path is per document but the cost it skips is per character, and on this
corpus non-ASCII beliefs are far longer than ASCII ones — so a reader who
takes the document share as the expected saving will overestimate it.

Usage::

    uv run python benchmarks/bm25_fold_ascii_guard.py \\
        --store .git/aelfrice/memory.db --repeat 10 --cold 5 --hook 5

Writes ``benchmarks/bm25_fold_ascii_guard.json`` beside this file unless
``--out`` says otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Module, not `from ... import`: the `shipped` arm must resolve through the
# module every call, or it snapshots the function and stops tracking main.
import aelfrice.bm25 as bm25  # noqa: E402
from aelfrice.store import MemoryStore  # noqa: E402


def unguarded_fold(text: str) -> str:
    """`_fold_diacritics` as it stood before #1387 — no ASCII fast path.

    `_REMOVED_MARKS` is read through the module because the mark set is
    shared data that #1387 does not touch; only the loop is duplicated.
    """
    out: list[str] = []
    for ch in text:
        if unicodedata.combining(ch):
            if ch not in bm25._REMOVED_MARKS:
                out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        if (
            len(decomposed) == 2
            and decomposed[1] in bm25._REMOVED_MARKS
            and decomposed[0].isascii()
            and decomposed[0].isalpha()
        ):
            out.append(decomposed[0])
        else:
            out.append(ch)
    return "".join(out)


def unguarded_tokenize_stemmed(text: str) -> list[str]:
    """`tokenize_stemmed` with the unguarded fold spliced in.

    Everything after the fold is the shipped code, so the two arms differ
    by exactly the fast path.
    """
    if not text:
        return []
    return [
        bm25._stem(m.group(0).lower())
        for m in bm25._FTS5_TOKEN_PATTERN.finditer(unguarded_fold(text))
    ]


def load_documents(store: Path, limit: int) -> list[tuple[int, str]]:
    """Indexed belief text, straight out of ``beliefs_fts``.

    Opened read-only through a URI: this reads a live store, and a bare
    `MemoryStore` open runs migrations plus the #1314 lock-expiry sweep.
    Measuring a store is not a reason to write to it.
    """
    query = "SELECT rowid, content FROM beliefs_fts ORDER BY rowid"
    params: tuple[int, ...] = ()
    if limit:
        query += " LIMIT ?"
        params = (limit,)
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        rows: Iterable[tuple[int, str | None]] = conn.execute(
            query, params,
        ).fetchall()
    finally:
        conn.close()
    return [(int(rowid), content or "") for rowid, content in rows]


def assert_arms_agree(docs: list[tuple[int, str]]) -> int:
    """Both arms must emit the same tokens before either is timed.

    A speedup between two arms that disagree is not a speedup, and after
    #1387 ships the `unguarded` arm is the only surviving description of
    the old behaviour — so it has to be checked against something, and
    the shipped fold is the only thing available. Returns the number of
    documents on which the ASCII fast path fires, which is what the
    saving is proportional to.
    """
    ascii_docs = 0
    for rowid, text in docs:
        if text.isascii():
            ascii_docs += 1
        shipped = bm25.tokenize_stemmed(text)
        reference = unguarded_tokenize_stemmed(text)
        if shipped != reference:
            raise SystemExit(
                f"arms disagree on belief rowid={rowid}: the unguarded "
                "reference has drifted from the shipped tokeniser, so no "
                "timing below would be attributable to the fast path"
            )
    return ascii_docs


def _child_build(store_path: str, arm: str) -> None:
    """Build the index once, in this fresh interpreter, then exit.

    The parent times the whole process, so everything before this — the
    interpreter start and every import — is inside the measurement. That
    is the point: on the cold path the hook pays those too, and #1161
    sized the budget against the wall clock, not against the build call.
    """
    if arm == "unguarded":
        bm25._fold_diacritics = unguarded_fold  # type: ignore[assignment]
    store = MemoryStore(store_path, read_only=True)
    try:
        bm25.BM25Index.build(store)
    finally:
        store.close()


def time_cold_min(store_path: Path, arm: str, repeat: int) -> float:
    """Min-of-`repeat` wall time for a cold-process build, milliseconds.

    A fresh `sys.executable` per iteration, because the thing being
    measured is a cost that only exists once per process and would be
    invisible to any in-process loop.
    """
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        completed = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--store", str(store_path),
                "--child-arm", arm,
            ],
            capture_output=True,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        if completed.returncode != 0:
            raise SystemExit(
                f"cold child ({arm}) failed:\n"
                f"{completed.stderr.decode('utf-8', 'replace')}"
            )
        best = min(best, elapsed)
    return best


_HOOK_PROMPT = (
    "what did we decide about the retrieval budget and the lock floor?"
)


def time_hook_fire_min(
    store_path: Path, arm: str, repeat: int, workdir: Path,
) -> float:
    """Min-of-`repeat` wall time for a cold `UserPromptSubmit` fire, ms.

    This is the quantity #1161 sized the 15 s budget against — the whole
    hook process with a **stale sidecar**, not `BM25Index.build` alone.
    The build arms above cannot restate that multiple: they exclude the
    retrieval, rendering and store work the hook also pays for, so a
    saving quoted against them would be a saving against the wrong
    denominator.

    Runs against a **copy** of the store. The hook writes — session ring,
    audit rows, the sidecar itself — and measuring a store is not a
    reason to write to it. The sidecar is deleted before each iteration,
    which is what makes the fire cold; the store copy is reused, because
    the hook's own small writes do not change the rebuild being timed.

    ``AELFRICE_DB`` points the child at the copy. Every other
    ``AELFRICE_*``/``AELF_*`` variable is **cleared** rather than
    inherited, so the fire measures shipped defaults instead of whatever
    the invoking shell happens to export.
    """
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("AELFRICE_", "AELF_"))
    }
    env["AELFRICE_DB"] = str(store_path)
    if arm == "unguarded":
        # The child imports `bm25` itself, so the arm has to be applied
        # inside it. `sitecustomize` is the one hook that runs before the
        # entry point without editing the shipped module.
        env["PYTHONPATH"] = f"{workdir}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["AELF_FOLD_BENCH_UNGUARDED"] = "1"

    payload = json.dumps({"prompt": _HOOK_PROMPT, "cwd": str(REPO_ROOT)})
    sidecar = Path(f"{store_path}{bm25._SIDECAR_SUFFIX}")
    best = float("inf")
    for _ in range(repeat):
        sidecar.unlink(missing_ok=True)
        start = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-c",
             "from aelfrice.hook import main; raise SystemExit(main())"],
            input=payload.encode(), capture_output=True, env=env,
            cwd=str(REPO_ROOT),
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        if completed.returncode != 0:
            raise SystemExit(
                f"hook fire ({arm}) failed:\n"
                f"{completed.stderr.decode('utf-8', 'replace')}"
            )
        best = min(best, elapsed)
    return best


_SITECUSTOMIZE = '''\
"""Applies the `unguarded` arm inside the hook child process (#1387).

Only active when AELF_FOLD_BENCH_UNGUARDED is set, and only reachable via
the PYTHONPATH this benchmark injects. Written to a temp dir, never to
the tree.
"""
import os

if os.environ.get("AELF_FOLD_BENCH_UNGUARDED") == "1":
    import unicodedata

    import aelfrice.bm25 as bm25

    def _unguarded_fold(text):
        out = []
        for ch in text:
            if unicodedata.combining(ch):
                if ch not in bm25._REMOVED_MARKS:
                    out.append(ch)
                continue
            decomposed = unicodedata.normalize("NFD", ch)
            if (
                len(decomposed) == 2
                and decomposed[1] in bm25._REMOVED_MARKS
                and decomposed[0].isascii()
                and decomposed[0].isalpha()
            ):
                out.append(decomposed[0])
            else:
                out.append(ch)
        return "".join(out)

    bm25._fold_diacritics = _unguarded_fold
'''


def time_min(fn: object, repeat: int) -> float:
    """Min-of-`repeat` wall time in milliseconds.

    Min rather than mean: the loop is deterministic and CPU-bound, so the
    floor is the property being measured and anything above it is the
    machine's other work. Under any concurrent load a mean stops being
    reproducible, which is the condition these figures have to survive.
    """
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()  # type: ignore[operator]
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def _at_least_one(value: str) -> int:
    """`--repeat` must run the timed body at least once.

    `time_min` starts at `inf` and only ever takes a min, so a repeat of
    zero leaves every figure at infinity — and `json.dumps` writes that
    as the bare token `Infinity`, which is not valid JSON. The committed
    report would then be unparseable by anything but Python.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative(value: str) -> int:
    """Zero disables an optional arm; negative is a typo, not a request."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or more")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--limit", type=_non_negative, default=0)
    parser.add_argument("--repeat", type=_at_least_one, default=6)
    parser.add_argument(
        "--cold", type=_non_negative, default=0,
        metavar="N",
        help="also time N cold-process builds per arm (slow; off by default)",
    )
    parser.add_argument(
        "--hook", type=_non_negative, default=0, metavar="N",
        help=(
            "also time N cold UserPromptSubmit fires per arm against a "
            "COPY of the store with the sidecar deleted (slowest; this is "
            "the path the #1161 hook budget was sized against)"
        ),
    )
    parser.add_argument(
        "--child-arm", choices=("shipped", "unguarded"), default=None,
        help=argparse.SUPPRESS,  # internal: one cold-process iteration
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    args = parser.parse_args(argv)

    if args.child_arm is not None:
        _child_build(str(args.store), args.child_arm)
        return 0

    docs = load_documents(args.store, args.limit)
    if not docs:
        print("no indexed beliefs in that store", file=sys.stderr)
        return 1

    ascii_docs = assert_arms_agree(docs)
    texts = [text for _rowid, text in docs]

    fold_unguarded = time_min(
        lambda: [unguarded_tokenize_stemmed(t) for t in texts], args.repeat,
    )
    fold_shipped = time_min(
        lambda: [bm25.tokenize_stemmed(t) for t in texts], args.repeat,
    )

    # The fold ON ITS OWN. `tokenize_stemmed` also lowercases, regex-scans
    # and stems, so a saving expressed over its total answers a different
    # question from one expressed over the fold. #1387 estimated a share of
    # the *fold*; without this arm that estimate cannot be checked, and the
    # two denominators had already been compared as if they were one.
    fold_only_unguarded = time_min(
        lambda: [unguarded_fold(t) for t in texts], args.repeat,
    )
    fold_only_shipped = time_min(
        lambda: [bm25._fold_diacritics(t) for t in texts], args.repeat,
    )

    # `read_only=True` is not a nicety: a bare open runs migrations and the
    # #1314 lock-expiry sweep, so the first arm would mutate the store the
    # second arm then measures.
    store = MemoryStore(str(args.store), read_only=True)
    try:
        original = bm25._fold_diacritics
        try:
            bm25._fold_diacritics = unguarded_fold  # type: ignore[assignment]
            build_unguarded = time_min(
                lambda: bm25.BM25Index.build(store), args.repeat,
            )
        finally:
            bm25._fold_diacritics = original  # type: ignore[assignment]
        build_shipped = time_min(
            lambda: bm25.BM25Index.build(store), args.repeat,
        )
    finally:
        store.close()

    hook: dict[str, float] = {}
    if args.hook:
        with tempfile.TemporaryDirectory(prefix="aelf-fold-bench-") as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "sitecustomize.py").write_text(_SITECUSTOMIZE)
            store_copy = tmpdir / "memory.db"
            # `sqlite3` backup rather than a file copy: the live store runs
            # in WAL, so copying the .db alone would silently drop whatever
            # is still in the -wal file.
            src = sqlite3.connect(f"file:{args.store}?mode=ro", uri=True)
            dst = sqlite3.connect(str(store_copy))
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            hook_unguarded = time_hook_fire_min(
                store_copy, "unguarded", args.hook, tmpdir,
            )
            hook_shipped = time_hook_fire_min(
                store_copy, "shipped", args.hook, tmpdir,
            )
        hook = {
            "unguarded": round(hook_unguarded, 1),
            "shipped": round(hook_shipped, 1),
            "saved": round(hook_unguarded - hook_shipped, 1),
        }

    cold: dict[str, float] = {}
    if args.cold:
        cold_unguarded = time_cold_min(args.store, "unguarded", args.cold)
        cold_shipped = time_cold_min(args.store, "shipped", args.cold)
        cold = {
            "unguarded": round(cold_unguarded, 1),
            "shipped": round(cold_shipped, 1),
            "saved": round(cold_unguarded - cold_shipped, 1),
        }

    ascii_chars = sum(len(t) for t in texts if t.isascii())
    total_chars = sum(len(t) for t in texts)
    # Relative to the repo when it lives there, bare filename otherwise.
    # The committed report goes to a public repo and an absolute path
    # carries the operator's home directory into it for no benefit — the
    # store is identified well enough by its document count.
    try:
        store_label = str(args.store.resolve().relative_to(REPO_ROOT))
    except ValueError:
        store_label = args.store.name

    report = {
        "store": store_label,
        "documents": len(docs),
        "ascii_documents": ascii_docs,
        "ascii_document_share": round(ascii_docs / len(docs), 6),
        # The share that predicts the saving. The fold is per character,
        # so a document-share reading overestimates it whenever the
        # non-ASCII documents are the longer ones — which they are here.
        "ascii_character_share": round(ascii_chars / max(total_chars, 1), 6),
        "repeat": args.repeat,
        "cold_repeat": args.cold,
        "statistic": "min",
        # Denominator note: `saved / unguarded` here is a share of the
        # FOLD, which is what #1387 estimated. The same saving over
        # `tokenize_stemmed_ms.unguarded` is a smaller number describing a
        # different quantity. Report both rather than picking one.
        "fold_only_ms": {
            "unguarded": round(fold_only_unguarded, 1),
            "shipped": round(fold_only_shipped, 1),
            "saved": round(fold_only_unguarded - fold_only_shipped, 1),
            "saved_share_of_fold": round(
                (fold_only_unguarded - fold_only_shipped)
                / max(fold_only_unguarded, 1e-9), 4,
            ),
        },
        "tokenize_stemmed_ms": {
            "unguarded": round(fold_unguarded, 1),
            "shipped": round(fold_shipped, 1),
            "saved": round(fold_unguarded - fold_shipped, 1),
        },
        "index_build_ms": {
            "unguarded": round(build_unguarded, 1),
            "shipped": round(build_shipped, 1),
            "saved": round(build_unguarded - build_shipped, 1),
        },
    }
    if cold:
        report["cold_build_ms"] = cold
    if hook:
        report["cold_hook_fire_ms"] = hook
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"store            : {store_label}")
    print(f"documents        : {len(docs)}")
    print(f"all-ASCII docs   : {ascii_docs} ({ascii_docs / len(docs):.2%})")
    print(f"all-ASCII chars  : {ascii_chars} "
          f"({ascii_chars / max(total_chars, 1):.2%})")
    print(f"statistic        : min of {args.repeat}")
    print()
    print(f"tokenize_stemmed : {fold_unguarded:9.1f} ms unguarded")
    print(f"                 : {fold_shipped:9.1f} ms shipped")
    print(f"                 : {fold_unguarded - fold_shipped:9.1f} ms saved")
    print()
    print(f"BM25Index.build  : {build_unguarded:9.1f} ms unguarded")
    print(f"                 : {build_shipped:9.1f} ms shipped")
    print(f"                 : {build_unguarded - build_shipped:9.1f} ms saved")
    print()
    if hook:
        print(f"cold hook fire   : {hook['unguarded']:9.1f} ms unguarded")
        print(f"                 : {hook['shipped']:9.1f} ms shipped")
        print(f"                 : {hook['saved']:9.1f} ms saved")
        print()
    if cold:
        print(f"cold build       : {cold['unguarded']:9.1f} ms unguarded")
        print(f"                 : {cold['shipped']:9.1f} ms shipped")
        print(f"                 : {cold['saved']:9.1f} ms saved")
        print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
