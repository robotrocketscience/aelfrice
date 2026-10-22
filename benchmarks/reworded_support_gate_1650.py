"""Graded precision gate for the reworded-restatement matcher (#1650).

The log-only pass (`aelfrice.supports_log`) records a reworded
restatement when a sentence the user typed restates a non-user belief
from another session. Before any match writes an edge, the matcher has
to pass the two-grader precision gate in
`docs/design/feature-supports-writer.md`. This is that instrument.

**Sweep.** `sweep` copies the store with SQLite's backup API, opens the
copy read-only, and replays every past user sentence in `ingest_log`
through `supports_log.iter_reworded_candidates`, the same eligibility
rules and matcher the log-only pass runs. A sentence is replayed once,
from the `transcript` row that inserted its belief (the first row that
derived it); later rows only corroborated it, and ingest offers no
supporter for a corroborating sentence. The sweep writes no edge, moves
no posterior, records no evidence row, and never touches the original
store. It replays against the store as it is now, so a target retired
since the sentence was typed is not a candidate.

**Population.** A match counts only when its target is not a phantom
(`type` is not `speculative`). Log-only mode matches every non-user
belief, and edges will later apply to phantoms only, so the gate is
measured on the matches the log already holds. The null arm is pairs the
matcher rejected, drawn from the same candidates and the same non-phantom
targets.

**Sample.** At most `DEFAULT_MAX` matches are sampled with `--seed`, and
as many null pairs, or `--null` of them. Each pair's id hashes its
supporter and target ids, so it says nothing about its arm. `sweep`
writes three files to `--out`:

- `packet.jsonl`: one `{"id", "user_sentence", "target_belief"}` row per
  pair, the two arms mixed in id order. It holds the user sentence and
  the target belief's full text; the matched sentence and the matched
  tokens are left out. Give it, with `instructions.txt`, to each grader.
- `instructions.txt`: what a grader labels.
- `key.jsonl`: each id's arm, supporter and target ids, and, for a
  match, the matched sentence's index and text. Keep it from graders.

The same store and seed give byte-identical files.

**Score.** Two fresh graders label every packet row independently, in
the label format of `benchmarks.graded_gate`. `score` reports each
grader's precision on the matches with its Wilson 95% interval, the
null-arm false-match rate (the share of rejected pairs a grader called a
restatement) with its interval, Cohen's kappa on the pairs both graded,
and the verdict. Unclear labels are left out of precision, rates, and
kappa. The verdict uses the #1677 gate's thresholds: `pass` when each
grader's precision on the matches reaches `PRECISION_BAR`, and
`insufficient` while either grader has graded fewer than `MIN_GRADED`
matches. The #1677 gate pre-registers no kappa or null-arm threshold, so
neither takes part in the verdict here.

The packet and the key hold user text. Keep them, and the label files,
out of the public repository.

Usage::

    uv run python -m benchmarks.reworded_support_gate_1650 sweep --store <copy of memory.db> --seed 1650 --out <dir>
    uv run python -m benchmarks.reworded_support_gate_1650 score --key <dir>/key.jsonl --labels a.jsonl b.jsonl

`sweep --dry-run` prints the counts without writing files. Every
subcommand exits 1 on bad input, and `score` exits 1 unless the verdict
is `pass`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from aelfrice.models import BELIEF_SPECULATIVE, INGEST_SOURCE_TRANSCRIPT
from aelfrice.store import MemoryStore, read_only_uri
from aelfrice.supports_log import iter_reworded_candidates
from benchmarks.graded_gate import (
    BAR,
    MIN_GRADED,
    Labels,
    grader_figures,
    paired_kappa,
    read_labels,
    require_labels,
    verdict,
    wilson_interval,
)

PRECISION_BAR: Final[float] = BAR
"""Precision each grader must reach on the matches: the #1677 gate's bar."""
MIN_GRADED_MATCHES: Final[int] = MIN_GRADED
"""Matches each grader must grade before the verdict can pass (#1677)."""
DEFAULT_MAX: Final[int] = 200
"""Largest match sample the precision gate grades (feature-supports-writer.md)."""
ARM_MATCH: Final[str] = "match"
ARM_NULL: Final[str] = "null"

INSTRUCTIONS: Final[str] = """\
Each row of packet.jsonl pairs a sentence a user typed with a belief stored
earlier. For each row, decide whether the user sentence restates something
the belief says, in the same or other words.

  true   the user sentence says the same thing as some part of the belief
  false  it does not: it says something else, says less than a claim needs,
         reverses or negates it, or only shares words with it
  null   you can't tell

Write one JSON object per line to your own labels file:
  {"id": "<row id>", "correct": true}
Label every row. Don't discuss the rows with the other grader.
"""


@dataclass(frozen=True)
class Pair:
    """One supporter and one non-phantom candidate target."""

    id: str
    arm: str
    supporter_id: str
    target_id: str
    ingest_log_id: str
    user_sentence: str
    target_belief: str
    target_sentence_index: int | None
    matched_span: str | None


def pair_id(supporter_id: str, target_id: str) -> str:
    """Stable id from the two belief ids; the same for either arm."""
    key = f"{supporter_id}\0{target_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def copy_store(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` through a read-only handle and SQLite's backup API.

    The backup reads a consistent snapshot, write-ahead log included,
    and opening `src` with `mode=ro` means the copy can't change it.
    """
    source = sqlite3.connect(read_only_uri(str(src)), uri=True)
    try:
        target = sqlite3.connect(str(dst))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def replay_supporters(db: Path) -> list[tuple[str, str]]:
    """`(belief id, ingest_log id)` for the row that inserted each transcript belief.

    Rows are read in id order, which is insertion order, and only the
    first row naming a belief as its first derived id is kept.
    """
    conn = sqlite3.connect(read_only_uri(str(db)), uri=True)
    try:
        rows = conn.execute(
            "SELECT id, json_extract(derived_belief_ids, '$[0]') FROM ingest_log "
            "WHERE source_kind = ? AND json_valid(derived_belief_ids) ORDER BY id",
            (INGEST_SOURCE_TRANSCRIPT,),
        ).fetchall()
    finally:
        conn.close()
    seen: set[str] = set()
    supporters: list[tuple[str, str]] = []
    for log_id, belief_id in rows:
        if not isinstance(belief_id, str) or belief_id in seen:
            continue
        seen.add(belief_id)
        supporters.append((belief_id, str(log_id)))
    return supporters


def collect_pairs(
    store: MemoryStore, supporters: Iterable[tuple[str, str]],
) -> tuple[list[Pair], list[Pair], int]:
    """`(matches, rejected pairs, phantom candidates skipped)`, each list sorted by id."""
    phantom: dict[str, bool] = {}
    matches: list[Pair] = []
    rejected: list[Pair] = []
    skipped = 0
    for cand in iter_reworded_candidates(store, supporters):
        if cand.target_id not in phantom:
            target = store.get_belief(cand.target_id)
            phantom[cand.target_id] = target is None or target.type == BELIEF_SPECULATIVE
        if phantom[cand.target_id]:
            skipped += 1
            continue
        m = cand.match
        pair = Pair(
            id=pair_id(cand.supporter_id, cand.target_id),
            arm=ARM_MATCH if m is not None else ARM_NULL,
            supporter_id=cand.supporter_id, target_id=cand.target_id,
            ingest_log_id=cand.ingest_log_id,
            user_sentence=cand.supporter_text, target_belief=cand.target_text,
            target_sentence_index=m.target_sentence_index if m is not None else None,
            matched_span=m.target_sentence if m is not None else None,
        )
        (matches if m is not None else rejected).append(pair)
    matches.sort(key=lambda p: p.id)
    rejected.sort(key=lambda p: p.id)
    return matches, rejected, skipped


def draw_sample(
    matches: list[Pair], rejected: list[Pair], seed: int,
    max_n: int = DEFAULT_MAX, null_n: int | None = None,
) -> list[Pair]:
    """Seeded matches plus a null arm, in id order.

    The null arm holds as many rejected pairs as sampled matches, or
    `null_n`, capped by how many exist.
    """
    rng = random.Random(seed)
    real = matches if len(matches) <= max_n else rng.sample(matches, max_n)
    k = min(len(real) if null_n is None else null_n, len(rejected))
    null = rng.sample(rejected, k)
    return sorted([*real, *null], key=lambda p: p.id)


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write_sample(sample: list[Pair], out: Path) -> None:
    """Write the grader packet, the instructions, and the hidden key to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "packet.jsonl").write_text(_jsonl(
        {"id": p.id, "user_sentence": p.user_sentence, "target_belief": p.target_belief}
        for p in sample
    ), encoding="utf-8")
    (out / "instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")
    (out / "key.jsonl").write_text(_jsonl(
        {"id": p.id, "arm": p.arm, "supporter_id": p.supporter_id,
         "target_id": p.target_id, "ingest_log_id": p.ingest_log_id,
         "target_sentence_index": p.target_sentence_index,
         "matched_span": p.matched_span}
        for p in sample
    ), encoding="utf-8")


def read_key(path: Path) -> dict[str, str]:
    """{pair id: arm} from a key file; ValueError on an unknown arm."""
    arms: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("id"), str) or row.get("arm") not in (ARM_MATCH, ARM_NULL):
                raise ValueError(f"{path}:{n}: need an id and arm=match|null")
            arms[row["id"]] = row["arm"]
    return arms


def _with_upper(ids: list[str], labels: Labels) -> dict[str, Any]:
    figures = grader_figures(ids, labels)
    figures["wilson_upper"] = wilson_interval(figures["correct"], figures["graded"])[1]
    return figures


def score(arms: Mapping[str, str], a: Labels, b: Labels) -> dict[str, Any]:
    """Precision on matches, null-arm false-match rate, kappa, and the verdict."""
    ids = sorted(arms)
    require_labels(ids, a, b, "key pairs")
    match_ids = [i for i in ids if arms[i] == ARM_MATCH]
    null_ids = [i for i in ids if arms[i] == ARM_NULL]
    graders = [_with_upper(match_ids, a), _with_upper(match_ids, b)]
    null: list[dict[str, Any]] = []
    for labels in (a, b):
        figures = _with_upper(null_ids, labels)
        null.append({
            "graded": figures["graded"],
            "called_restatement": figures["correct"],
            "false_match_rate": figures["precision"],
            "wilson_lower": figures["wilson_lower"],
            "wilson_upper": figures["wilson_upper"],
        })
    kappa, kappa_n, agreed = paired_kappa(ids, a, b)
    return {
        "pairs": len(ids),
        "matches": len(match_ids),
        "null_pairs": len(null_ids),
        "graders": graders,
        "null_arm": null,
        "kappa": kappa,
        "kappa_n": kappa_n,
        "agreed": agreed,
        "bar": PRECISION_BAR,
        "min_graded": MIN_GRADED_MATCHES,
        "verdict": verdict(graders, PRECISION_BAR, MIN_GRADED_MATCHES),
    }


def sweep(store_path: Path, work: Path) -> tuple[list[Pair], list[Pair], int, int]:
    """`(matches, rejected, phantom skipped, supporters)` from a read-only copy."""
    if not store_path.is_file():
        raise FileNotFoundError(f"no store at {store_path}")
    db = work / "memory.db"
    copy_store(store_path, db)
    supporters = replay_supporters(db)
    store = MemoryStore(str(db), read_only=True)
    try:
        matches, rejected, skipped = collect_pairs(store, supporters)
    finally:
        store.close()
    return matches, rejected, skipped, len(supporters)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--store", type=Path, required=True)
    sw.add_argument("--seed", type=int, required=True)
    sw.add_argument("--max", type=int, default=DEFAULT_MAX)
    sw.add_argument("--null", type=int)
    sw.add_argument("--out", type=Path)
    sw.add_argument("--work-dir", type=Path,
                    help="where the store copy goes while the sweep runs")
    sw.add_argument("--dry-run", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--key", type=Path, required=True)
    sc.add_argument("--labels", type=Path, nargs=2, required=True)
    args = parser.parse_args(argv)

    try:
        if args.cmd == "sweep":
            out: Path | None = args.out
            if out is None and not args.dry_run:
                print("error: sweep needs --out unless --dry-run", file=sys.stderr)
                return 1
            with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
                matches, rejected, skipped, n_sup = sweep(args.store, Path(work))
            sample = draw_sample(matches, rejected, args.seed, args.max, args.null)
            n_real = sum(1 for p in sample if p.arm == ARM_MATCH)
            print(f"supporters={n_sup} matches={len(matches)} rejected={len(rejected)} "
                  f"phantom_skipped={skipped} sample_matches={n_real} "
                  f"sample_null={len(sample) - n_real} seed={args.seed}")
            if args.dry_run or out is None:
                return 0
            write_sample(sample, out)
            print(f"wrote {out}")
            return 0
        report = score(read_key(args.key), read_labels(args.labels[0]),
                       read_labels(args.labels[1]))
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
