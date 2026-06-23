"""#981 HRR vocabulary-bridge expansion-lane ablation (the #977-sweep arm).

Runs LoCoMo under four retrieval configurations to isolate the
``use_hrr_expand`` lane:

    baseline    — BFS off, HRR-expand off  (production-style default)
    +hrr-expand — HRR-expand on,  BFS off
    +bfs        — BFS on,         HRR-expand off
    all-on      — BFS on,         HRR-expand on

Scoring is the LoCoMo adapter's deterministic retrieval-overlap F1
(``score_qa`` over the concatenated retrieved context, incl. the cat-5
"No information available" length heuristic). This is exactly the scorer the
#981 issue's honest-uplift caveat references — no LLM reader is involved, so
the arm comparison is fully reproducible. Absolute F1 is a retrieval-recall
proxy, not reader accuracy; the *relative* arm deltas are the deliverable.

Ingest is arm-independent (the flags only change retrieval), so each
conversation is ingested once, its struct index + neighbour table built once,
then all four arms run against the same DB.

LongMemEval / MAB / StructMemEval are intentionally out of scope here: their
full corpora are not present locally (only micro smoke fixtures) and per the
project's bench notes several are structurally unscorable. The headline
delta this issue addresses (66.1%→40.88%) is LoCoMo, so LoCoMo is the arm.

Usage:
    uv run python -m benchmarks.hrr_expand_ablation \
        --data /tmp/LoCoMo/data/locomo10.json --out /tmp/hrr_expand_ablation.json
    # smoke: --subset-convs 1 --subset-qa 20
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Final

from aelfrice.hrr_expand import precompute_expand_neighbors
from aelfrice.hrr_index import HRRStructIndexCache
from aelfrice.relationship_detector import write_semantic_edges
from aelfrice.retrieval import retrieve_v2
from aelfrice.store import MemoryStore

from benchmarks.locomo_adapter import (
    CATEGORY_NAMES,
    DEFAULT_DATA_PATH,
    ingest_conversation,
    load_locomo,
    score_qa,
)


@dataclass(frozen=True)
class Arm:
    key: str
    use_bfs: bool
    use_hrr_expand: bool


ARMS: Final[tuple[Arm, ...]] = (
    Arm("baseline", use_bfs=False, use_hrr_expand=False),
    Arm("+hrr-expand", use_bfs=False, use_hrr_expand=True),
    Arm("+bfs", use_bfs=True, use_hrr_expand=False),
    Arm("all-on", use_bfs=True, use_hrr_expand=True),
)

# Edge-substrate dimension. The HRR-expand and BFS lanes traverse semantic
# edges; vanilla LoCoMo ingest writes none, so both lanes are inert (#977).
# #988's deterministic relationship detection mints CONTRADICTS edges at
# ingest behind a default-off flag — the substrate these lanes consume. The
# ablation runs the arms under both so the lane's substrate-gating is visible.
SUBSTRATE_NONE: Final[str] = "none"
SUBSTRATE_988: Final[str] = "contradicts-988"
SUBSTRATES: Final[tuple[str, ...]] = (SUBSTRATE_NONE, SUBSTRATE_988)


@dataclass
class ArmScore:
    total_qa: int = 0
    total_f1: float = 0.0
    cat_f1: dict[int, list[float]] = field(default_factory=dict)
    expand_hits: int = 0  # total beliefs the lane merged across questions

    def add(self, category: int, f1: float) -> None:
        self.total_qa += 1
        self.total_f1 += f1
        self.cat_f1.setdefault(category, []).append(f1)

    @property
    def overall(self) -> float:
        return self.total_f1 / self.total_qa if self.total_qa else 0.0


def _context(beliefs: list) -> str:
    return " ".join(b.content for b in beliefs)


def run_arm_on_store(
    store: MemoryStore,
    cache: HRRStructIndexCache,
    qa_pairs: list,
    arm: Arm,
    budget: int,
    score: ArmScore,
) -> None:
    from aelfrice import retrieval as _R

    for qa in qa_pairs:
        result = retrieve_v2(
            store=store,
            query=qa.question,
            budget=budget,
            include_locked=False,
            use_bfs=arm.use_bfs,
            use_hrr_expand=arm.use_hrr_expand,
            hrr_struct_index_cache=cache,
        )
        score.expand_hits += _R._LAST_TELEMETRY.hrr_expand
        prediction = _context(result.beliefs)
        if qa.category == 5 and len(prediction.split()) < 10:
            prediction = "No information available"
        answer = qa.answer if qa.category != 5 else ""
        score.add(qa.category, score_qa(prediction, answer, qa.category))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA_PATH)
    ap.add_argument(
        "--out",
        default=os.path.join(tempfile.gettempdir(), "hrr_expand_ablation.json"),
    )
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--subset-convs", type=int, default=None)
    ap.add_argument("--subset-qa", type=int, default=None)
    args = ap.parse_args()

    # Resolver reads env; clear any ambient overrides so the explicit kwargs
    # this runner passes are authoritative.
    for var in ("AELFRICE_HRR_EXPAND", "AELFRICE_BFS"):
        os.environ.pop(var, None)

    convs = load_locomo(args.data)
    if args.subset_convs is not None:
        convs = convs[: args.subset_convs]

    scores: dict[tuple[str, str], ArmScore] = {
        (sub, a.key): ArmScore() for sub in SUBSTRATES for a in ARMS
    }
    edge_counts: dict[str, int] = {sub: 0 for sub in SUBSTRATES}
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory() as db_dir:
        for ci, conv in enumerate(convs):
            db_path = f"{db_dir}/{conv.sample_id}.db"
            store = MemoryStore(db_path)
            ingest_conversation(store, conv)
            qa_pairs = conv.qa_pairs
            if args.subset_qa is not None:
                qa_pairs = qa_pairs[: args.subset_qa]
            # SUBSTRATE_NONE must run before edges are written (edges only
            # add). For SUBSTRATE_988, mint CONTRADICTS edges, then rebuild a
            # fresh index/cache over the now-edged store.
            for sub in SUBSTRATES:
                if sub == SUBSTRATE_988:
                    write_semantic_edges(store)
                edge_counts[sub] += store._conn.execute(  # noqa: SLF001
                    "SELECT COUNT(*) FROM edges"
                ).fetchone()[0]
                cache = HRRStructIndexCache(store=store, store_path=db_path)
                precompute_expand_neighbors(store, cache.get())
                for arm in ARMS:
                    run_arm_on_store(
                        store, cache, qa_pairs, arm, args.budget,
                        scores[(sub, arm.key)],
                    )
            print(
                f"[{ci + 1}/{len(convs)}] {conv.sample_id}: "
                f"{len(qa_pairs)} QA × {len(ARMS)} arms × {len(SUBSTRATES)} "
                f"substrates done",
                flush=True,
            )

    elapsed = time.monotonic() - t0
    report = _build_report(scores, edge_counts, elapsed, args)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    _print_report(report)
    print(f"\nWrote {args.out}  ({elapsed:.1f}s)")


def _build_report(
    scores: dict[tuple[str, str], ArmScore],
    edge_counts: dict[str, int],
    elapsed: float,
    args: argparse.Namespace,
) -> dict:
    all_cats = sorted({c for s in scores.values() for c in s.cat_f1})
    substrates_out: dict[str, dict] = {}
    for sub in SUBSTRATES:
        base = scores[(sub, "baseline")].overall
        arms_out: dict[str, dict] = {}
        for arm in ARMS:
            s = scores[(sub, arm.key)]
            arms_out[arm.key] = {
                "use_bfs": arm.use_bfs,
                "use_hrr_expand": arm.use_hrr_expand,
                "total_qa": s.total_qa,
                "overall_f1": round(s.overall, 4),
                "delta_vs_baseline": round(s.overall - base, 4),
                "expand_hits": s.expand_hits,
                "category_f1": {
                    str(c): round(sum(s.cat_f1[c]) / len(s.cat_f1[c]), 4)
                    for c in all_cats
                    if s.cat_f1.get(c)
                },
            }
        substrates_out[sub] = {
            "total_edges": edge_counts[sub],
            "arms": arms_out,
        }
    return {
        "benchmark": "locomo",
        "scorer": "retrieval_overlap_f1 (locomo_adapter.score_qa)",
        "issue": 981,
        "note": (
            "Retrieval-overlap F1 (no LLM reader); relative arm deltas are "
            "the deliverable. HRR-expand/BFS traverse semantic edges — "
            "inert without an edge substrate (#977). 'contradicts-988' "
            "enables #988 relationship detection at ingest."
        ),
        "data": args.data,
        "budget": args.budget,
        "elapsed_s": round(elapsed, 1),
        "category_names": CATEGORY_NAMES,
        "substrates": substrates_out,
    }


def _print_report(report: dict) -> None:
    cats = sorted({
        int(c)
        for sub in report["substrates"].values()
        for a in sub["arms"].values()
        for c in a["category_f1"]
    })
    print("\n" + "=" * 78)
    print(f"#981 HRR-expand ablation — LoCoMo ({report['scorer']})")
    print("=" * 78)
    for sub, subdata in report["substrates"].items():
        print(f"\n[substrate={sub}]  total_edges={subdata['total_edges']}")
        header = (
            f"  {'arm':<13}{'overall':>9}{'Δ':>7}{'expand':>8}  "
            + "".join(f"c{c}:{CATEGORY_NAMES.get(c, '?')[:5]:>7}" for c in cats)
        )
        print(header)
        for key, a in subdata["arms"].items():
            row = (
                f"  {key:<13}{a['overall_f1'] * 100:>8.1f}%"
                f"{a['delta_vs_baseline'] * 100:>+6.1f}{a['expand_hits']:>8}  "
            )
            row += "".join(
                f"{a['category_f1'].get(str(c), 0.0) * 100:>13.1f}"
                for c in cats
            )
            print(row)


if __name__ == "__main__":
    main()
