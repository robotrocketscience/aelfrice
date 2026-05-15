#!/usr/bin/env python3
"""Authoring-time validator for the #844 A4-fidelity corpus.

The A4 bench gate (``tests/bench_gate/test_compression_a4_fidelity.py``)
asks whether ``use_type_aware_compression`` ON regresses continuation
fidelity vs OFF. The R7-R18 campaign on #769 surfaced a corpus-shape
trap: when a row's ``transcript_pre_clear`` pre-answers the
continuation question, the post-clear answer paraphrases the prior
assistant turn from ``<recent-turns>`` and the compression of
``<retrieved-beliefs>`` is invisible to the scorer. The row scores
high on both arms while measuring nothing.

This script emits a per-row diagnostic so authors can catch that
shape regression before committing rows. Two flags per row:

* ``arm_divergence_flag`` — at least one expected answer has materially
  different token-coverage under OFF vs ON (``|Δ| > arm_delta``,
  default 0.001). Rows where every answer is OFF/ON-identical likely
  flow entirely from ``<recent-turns>``; the compression branch never
  bites.

* ``beliefs_load_bearing_flag`` — at least one expected answer's
  token-coverage drops by more than ``beliefs_delta`` (default 0.05)
  when the ``<retrieved-beliefs>...</retrieved-beliefs>`` section is
  excised from the OFF-arm rebuild block. If False the answer can
  be assembled from ``<recent-turns>`` alone and the row does not
  measure compression of retrieved content. (We strip the section
  in-place rather than emptying the store because ``rebuild_v14``
  returns the empty string when nothing is retrieved AND no locks
  exist — emptying the store would zero out ``<recent-turns>`` too
  and the diagnostic would degenerate to "is the rebuild block
  non-empty?".)

The R18 failure shape is ``beliefs_load_bearing_flag=False``. The
``--strict`` mode exits non-zero on any such row. The diagnostic
itself is JSONL on stdout (or ``--out``), one record per row, in
input order.

Coverage primitive. The validator uses the same whitespace-split,
casefold + NFC token-coverage proxy as the runner's
``_coverage_score`` (``tests/retrieve_uplift_runner.py``) — not the
#138 exact-method scorer. Exact-method requires
``captured_post_clear_answers_{off,on}`` arrays that exist only at
bench-time, after a live model has run. The coverage proxy is a
captures-free heuristic that catches the R18 trap structurally:
if the beliefs are stripped and the rebuild block still carries the
expected tokens, the answer is provably retrievable from
``<recent-turns>``.

Discretion. The validator reads only the public corpus surface
(``tests/corpus/v2_0/compression_a4_fidelity/*.jsonl``) or whatever
``--corpus-root`` is pointed at. Lab-side rows under
``$AELFRICE_CORPUS_ROOT`` are loaded the same way. No
``~/.claude/``-derived content reaches the output; the diagnostic
record carries only the row id, the flag pair, and the per-answer
coverage floats, never the answer text.

Usage::

    python scripts/validate_a4_corpus.py \\
        --corpus-root tests/corpus/v2_0 \\
        [--strict] [--out diagnostic.jsonl]

Exit codes:
    0   diagnostic emitted; no strict violations (or ``--strict``
        omitted)
    1   load/parse error
    2   ``--strict`` violations: at least one row has
        ``beliefs_load_bearing_flag=False``
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"[\w']+")

_RETRIEVED_BELIEFS_RE = re.compile(
    r"<retrieved-beliefs\b[^>]*>.*?</retrieved-beliefs>",
    re.DOTALL,
)


def _strip_retrieved_beliefs(rebuild_block: str) -> str:
    """Remove the ``<retrieved-beliefs>...</retrieved-beliefs>`` section.

    Leaves ``<recent-turns>``, ``<continue/>``, and the
    ``<aelfrice-rebuild>`` envelope intact. The diagnostic asks
    whether the expected answer is still derivable from what
    remains; if so the row is not measuring belief compression.
    """
    return _RETRIEVED_BELIEFS_RE.sub("", rebuild_block)


def _normalize_tokens(text: str) -> list[str]:
    """NFC + casefold + word-token split.

    Mirrors ``tests.retrieve_uplift_runner._normalize_tokens_for_coverage``
    so the validator and the bench runner agree on what "token
    coverage" means.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    return _TOKEN_RE.findall(normalized)


def _coverage(expected: str, rebuild_block: str) -> float:
    """Fraction of expected tokens that appear in the rebuild block.

    Empty expected → 1.0 (vacuous), matching the #138 scorer's
    empty-case convention. Empty rebuild block + non-empty expected
    → 0.0.
    """
    expected_tokens = _normalize_tokens(expected)
    if not expected_tokens:
        return 1.0
    block_tokens = set(_normalize_tokens(rebuild_block))
    hits = sum(1 for t in expected_tokens if t in block_tokens)
    return hits / len(expected_tokens)


@dataclass(frozen=True)
class RowDiagnostic:
    id: str
    arm_divergence_flag: bool
    beliefs_load_bearing_flag: bool
    n_expected_answers: int
    max_arm_delta: float
    max_beliefs_delta: float


def _recent_turns_from_row(row: dict):  # type: ignore[type-arg]
    """Build the rebuilder's ``recent_turns`` list — lazy import."""
    from aelfrice.context_rebuilder import RecentTurn
    turns = []
    for t in row.get("transcript_pre_clear", []):
        turns.append(
            RecentTurn(
                role=str(t.get("role", "user")),
                text=str(t.get("text", "")),
                session_id=t.get("session_id"),
                ts=t.get("ts"),
            )
        )
    return turns


_TS = "2026-05-15T00:00:00+00:00"


def _belief_from_row(b: dict):  # type: ignore[type-arg]
    """Build a ``Belief`` from a corpus-row belief dict.

    Mirrors ``tests.retrieve_uplift_runner._a2_belief_from_row`` so the
    validator and bench-gate agree on belief shape.
    """
    from aelfrice.models import (
        BELIEF_FACTUAL,
        LOCK_NONE,
        LOCK_USER,
        ORIGIN_AGENT_INFERRED,
        RETENTION_UNKNOWN,
        Belief,
    )
    lock = LOCK_USER if str(b.get("lock_level", "none")) == "user" else LOCK_NONE
    return Belief(
        id=b["id"],
        content=b["content"],
        content_hash=f"corpus:{b['id']}",
        alpha=float(b.get("alpha", 1.0)),
        beta=float(b.get("beta", 1.0)),
        type=b.get("type", BELIEF_FACTUAL),
        lock_level=lock,
        locked_at=_TS if lock == LOCK_USER else None,
        created_at=_TS,
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
        retention_class=str(b.get("retention_class", RETENTION_UNKNOWN)),
    )


def _build_block(
    *,
    row: dict,  # type: ignore[type-arg]
    beliefs: list,  # type: ignore[type-arg]
    compress_on: bool,
    budget: int,
    db_path: Path,
) -> str:
    """Run ``rebuild_v14`` for one (arm, beliefs-set) combination.

    Caller owns the temp ``db_path``. The function sets+restores
    ``AELFRICE_TYPE_AWARE_COMPRESSION`` around the call.
    """
    from aelfrice.context_rebuilder import rebuild_v14
    from aelfrice.store import MemoryStore

    store = MemoryStore(str(db_path))
    try:
        for b in beliefs:
            store.insert_belief(_belief_from_row(b))
        prior = os.environ.get("AELFRICE_TYPE_AWARE_COMPRESSION")
        os.environ["AELFRICE_TYPE_AWARE_COMPRESSION"] = "1" if compress_on else "0"
        try:
            return rebuild_v14(
                _recent_turns_from_row(row),
                store,
                token_budget=budget,
            )
        finally:
            if prior is None:
                os.environ.pop("AELFRICE_TYPE_AWARE_COMPRESSION", None)
            else:
                os.environ["AELFRICE_TYPE_AWARE_COMPRESSION"] = prior
    finally:
        store.close()


def diagnose_row(
    row: dict,  # type: ignore[type-arg]
    *,
    tmp_root: Path,
    arm_delta: float,
    beliefs_delta: float,
    db_counter: list,  # type: ignore[type-arg]
) -> RowDiagnostic:
    """Compute the per-row flag pair for a single corpus row.

    Builds two rebuild blocks (OFF, ON) and a third virtual block
    by stripping ``<retrieved-beliefs>`` from the OFF block. Scores
    token coverage per expected answer and reduces to the max
    per-answer delta on each axis.
    """
    from aelfrice.context_rebuilder import DEFAULT_REBUILDER_TOKEN_BUDGET

    budget = int(row.get("rebuilder_token_budget", DEFAULT_REBUILDER_TOKEN_BUDGET))
    expected_answers = [str(a) for a in row.get("expected_post_clear_answers", [])]
    beliefs = list(row.get("beliefs", []))
    row_id = str(row["id"])

    blocks: dict[bool, str] = {}
    for compress_on in (False, True):
        db_counter[0] += 1
        db = (
            tmp_root
            / f"validate_a4_{row_id}_{int(compress_on)}_{db_counter[0]}.db"
        )
        blocks[compress_on] = _build_block(
            row=row,
            beliefs=beliefs,
            compress_on=compress_on,
            budget=budget,
            db_path=db,
        )

    off_no_beliefs = _strip_retrieved_beliefs(blocks[False])

    max_arm = 0.0
    max_beliefs = 0.0
    for answer in expected_answers:
        c_off = _coverage(answer, blocks[False])
        c_on = _coverage(answer, blocks[True])
        c_off_no = _coverage(answer, off_no_beliefs)
        arm = abs(c_off - c_on)
        beliefs_drop = c_off - c_off_no
        if arm > max_arm:
            max_arm = arm
        if beliefs_drop > max_beliefs:
            max_beliefs = beliefs_drop

    return RowDiagnostic(
        id=row_id,
        arm_divergence_flag=max_arm > arm_delta,
        beliefs_load_bearing_flag=max_beliefs > beliefs_delta,
        n_expected_answers=len(expected_answers),
        max_arm_delta=round(max_arm, 6),
        max_beliefs_delta=round(max_beliefs, 6),
    )


def _iter_corpus_rows(corpus_root: Path) -> Iterable[dict]:  # type: ignore[type-arg]
    """Yield rows from ``<corpus_root>/compression_a4_fidelity/*.jsonl``.

    Matches the layout the bench gate loads via
    ``tests.conftest.load_corpus_module``.
    """
    mod_dir = corpus_root / "compression_a4_fidelity"
    if not mod_dir.is_dir():
        return
    for p in sorted(mod_dir.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Per-row diagnostic for the A4-fidelity corpus (#844). "
            "Emits arm-divergence + beliefs-load-bearing flags."
        )
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path(os.environ.get("AELFRICE_CORPUS_ROOT", "tests/corpus/v2_0")),
        help="Corpus root; expects compression_a4_fidelity/*.jsonl under it",
    )
    parser.add_argument(
        "--arm-delta",
        type=float,
        default=0.001,
        help="Per-answer |OFF - ON| coverage threshold for arm-divergence",
    )
    parser.add_argument(
        "--beliefs-delta",
        type=float,
        default=0.05,
        help=(
            "Per-answer coverage drop threshold for beliefs-load-bearing "
            "(coverage_with_beliefs - coverage_without_beliefs)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSONL diagnostic here; stdout if omitted",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit 2 if any row has beliefs_load_bearing_flag=False "
            "(R18-shape regression guard)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        rows = list(_iter_corpus_rows(args.corpus_root))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"validate_a4_corpus: corpus load failed: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(
            "validate_a4_corpus: zero rows under "
            f"{args.corpus_root}/compression_a4_fidelity/",
            file=sys.stderr,
        )
        return 1

    db_counter = [0]
    diagnostics: list[RowDiagnostic] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for row in rows:
            diagnostics.append(
                diagnose_row(
                    row,
                    tmp_root=tmp_root,
                    arm_delta=args.arm_delta,
                    beliefs_delta=args.beliefs_delta,
                    db_counter=db_counter,
                )
            )

    out_handle = open(args.out, "w") if args.out else sys.stdout
    try:
        for d in diagnostics:
            out_handle.write(json.dumps(asdict(d), sort_keys=True) + "\n")
    finally:
        if args.out:
            out_handle.close()

    if args.strict:
        violations = [d for d in diagnostics if not d.beliefs_load_bearing_flag]
        if violations:
            ids = ", ".join(d.id for d in violations[:10])
            tail = f" (and {len(violations) - 10} more)" if len(violations) > 10 else ""
            print(
                "validate_a4_corpus: --strict violations "
                f"(beliefs_load_bearing_flag=False): {ids}{tail}",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
