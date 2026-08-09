"""The `scan_repo` admission funnel, measured rather than quoted (#1398).

#1159 gates three open changes on a `90.5% admission / 17.4% correction`
baseline that had no committed derivation. This script is that derivation.

#1159's two apparently-conflicting counts — 8,428 beliefs from 9,265 candidates
in its opening summary, 8,388 from the same 9,265 in its junk-percolation
section — are **both correct**. They are two different rates reported under one
word, and this script's job is to stop that from happening again by always
printing both, named:

``survival``
    candidates that got past both filters, as a fraction of all candidates.
    This is what `9265 - 834 - 3` computes.
``admission``
    distinct beliefs the scan actually inserted, as a fraction of all
    candidates. Lower, because `scan_repo` defers materialisation to
    `derivation_worker.run_worker` and then counts *unique* belief ids: two
    candidates whose `sha256(source\\x00text)[:16]` collide converge onto one
    belief, and one log row can derive several. `scanner.py:325-335` counts the
    second and later arrivals as `skipped_existing`, not as `inserted`.

Report both, always. A single "admission rate" with no definition attached is
what made #1159's two correct numbers look like a contradiction.

**The funnel is a function of the tree scanned, so quote the `sha` with the
number.** Running this script at `2a608b1d` reproduces #1159 to the digit —
9,265 candidates, 834 noise, 3 non-persisting, 8,428 survived, 8,388 inserted,
a 40-belief gap, 90.5% admission, 17.4% `correction`. At `9848be03`, 548 commits
later, the same code gives 10,823 / 91.6% / 19.8%. That is corpus growth, not
drift in the filter: the second run used *current* code against the *old* tree
and still returned 90.5%, so the `noise_filter` and `derivation` changes in
between move this corpus not at all. Two figures from two trees are not a
before-and-after.

The script also scans the repository it lives in, so its own source enters the
corpus: a checkout carrying this file and its test measures higher than a
pristine one by their own line count. **Quote the pristine figure** — 10,823 at
`9848be03` — and point `--root` at a clean worktree to reproduce it. The
self-including count is deliberately not written down here: it moves every time
either file is edited, including by the edit that would record it. `sha` and
`dirty` are printed on every run for exactly this reason.

Noise attribution is reported **first-match-wins**, mirroring `is_noise`'s
short-circuit at `noise_filter.py:447-474` bucket for bucket. Re-testing all
four categories independently would double-count — most heading blocks are also
short enough to be three-word fragments — so the columns would not sum to
`skipped_noise` and the per-rule split would be meaningless. `_noise_bucket`
below is a replica of that dispatch, and every candidate is cross-checked
against the real `is_noise` so the replica cannot silently drift from it.

The store is a fresh temporary directory, created per run and never reused.
That is structural, not a flag: `MemoryStore.__init__` documents that a bare
open is a write, and an admission rate measured against a store that already
holds the corpus reports `skipped_existing`, not `inserted`.

Usage::

    uv run python benchmarks/scan_admission_funnel.py
    uv run python benchmarks/scan_admission_funnel.py --root /path/to/repo --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from aelfrice.noise_filter import (
    NoiseConfig,
    is_checklist_block,
    is_heading_block,
    is_license_boilerplate,
    is_noise,
    is_three_word_fragment,
)
from aelfrice.scanner import (
    SentenceCandidate,
    _build_file_recency_map,
    extract_ast,
    extract_filesystem,
    extract_git_log,
    scan_repo,
)
from aelfrice.store import MemoryStore

# A fixed clock, so two runs at the same SHA produce byte-identical output.
# `scan_repo` derives its session_id from (root, timestamp) and stamps
# `created_at` from it wherever a candidate carries no commit date, so leaving
# this to wall-clock time would make the run non-reproducible by construction.
FIXED_NOW = "2026-01-01T00:00:00Z"

# Buckets in the order `is_noise` tests them. Order is the contract: the
# first match wins, so a reordering changes the split without changing the
# total.
NOISE_BUCKETS = (
    "empty",
    "headings",
    "checklists",
    "fragments",
    "license",
    "exclude_words",
    "exclude_phrases",
)


def _noise_bucket(text: str, cfg: NoiseConfig) -> str | None:
    """The bucket `is_noise` would drop `text` under, or None if it survives.

    A line-for-line replica of `noise_filter.is_noise` (`:447-474`), returning
    which arm fired instead of a bool. Kept in the same order and under the
    same config guards; `_assert_replica_agrees` checks the two against each
    other on every candidate, so drift fails the run rather than quietly
    skewing the split.
    """
    if not text or not text.strip():
        return "empty"
    if cfg.drop_headings and is_heading_block(text, cfg):
        return "headings"
    if cfg.drop_checklists and is_checklist_block(text, cfg):
        return "checklists"
    if cfg.drop_fragments and is_three_word_fragment(text, cfg):
        return "fragments"
    if cfg.drop_license and is_license_boilerplate(text, cfg):
        return "license"
    if cfg._word_patterns and any(
        p.search(text) is not None for p in cfg._word_patterns
    ):
        return "exclude_words"
    if cfg._phrase_lowers:
        lowered = text.lower()
        if any(phr in lowered for phr in cfg._phrase_lowers):
            return "exclude_phrases"
    return None


def _assert_replica_agrees(
    candidates: list[SentenceCandidate], cfg: NoiseConfig
) -> None:
    """Fail loudly if `_noise_bucket` and `is_noise` disagree on any candidate.

    Without this the script still runs and still prints a plausible table when
    `is_noise` grows a fifth category — the new arm's drops would land in the
    survivor column and the noise split would under-report with no symptom.
    """
    for cand in candidates:
        replica = _noise_bucket(cand.text, cfg) is not None
        product = is_noise(cand.text, cfg)
        if replica != product:
            raise SystemExit(
                "_noise_bucket has drifted from noise_filter.is_noise "
                f"(replica={replica}, is_noise={product}) on candidate "
                f"{cand.source!r}. Update _noise_bucket and NOISE_BUCKETS to "
                "match the new dispatch order before trusting any split."
            )


def _git(root: Path, *args: str) -> str:
    """`git -C root ...`, or the empty string if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _corpus_identity(root: Path) -> dict[str, Any]:
    """Everything needed to reproduce this run's inputs.

    A rate without its corpus identity is the failure mode #1398 was filed
    over, so this is printed on every run rather than offered as a flag.
    """
    dirty = _git(root, "status", "--porcelain")
    return {
        "root": str(root.resolve()),
        "sha": _git(root, "rev-parse", "HEAD") or "unknown",
        "dirty": bool(dirty),
        "dirty_paths": len(dirty.splitlines()) if dirty else 0,
        "now": FIXED_NOW,
    }


def measure(root: Path) -> dict[str, Any]:
    """Run the funnel once against a throwaway store and return the counts."""
    cfg = NoiseConfig.discover(root)

    # Extract per-source so the git-log share is attributable: #1159's primary
    # claim is specifically that `extract_git_log` admits commit subjects with
    # no filter of their own.
    recency = _build_file_recency_map(root)
    by_extractor: dict[str, list[SentenceCandidate]] = {
        "filesystem": extract_filesystem(root, recency=recency),
        "git_log": extract_git_log(root),
        "ast": extract_ast(root, recency=recency),
    }
    candidates = [c for group in by_extractor.values() for c in group]
    _assert_replica_agrees(candidates, cfg)

    noise_by_bucket: Counter[str] = Counter()
    noise_by_extractor: Counter[str] = Counter()
    for name, group in by_extractor.items():
        for cand in group:
            bucket = _noise_bucket(cand.text, cfg)
            if bucket is not None:
                noise_by_bucket[bucket] += 1
                noise_by_extractor[name] += 1

    # A store that must not exist beforehand and is deleted afterwards. The
    # ambient development store is unreachable from here by construction.
    tmpdir = Path(tempfile.mkdtemp(prefix="aelf-scan-admission-"))
    db_path = tmpdir / "memory.db"
    if db_path.exists():  # pragma: no cover - mkdtemp guarantees a fresh dir
        raise SystemExit(f"refusing to scan into a pre-existing store: {db_path}")
    store: MemoryStore | None = None
    try:
        store = MemoryStore(str(db_path))
        before = len(store.list_belief_ids())
        if before:
            raise SystemExit(
                f"throwaway store opened with {before} beliefs already present"
            )
        result = scan_repo(store, root, now=FIXED_NOW, noise_config=cfg)
        types: Counter[str] = Counter()
        for bid in store.list_belief_ids():
            belief = store.get_belief(bid)
            if belief is not None:
                types[belief.type] += 1
        stored = sum(types.values())
    finally:
        # Close before unlinking: the connection holds WAL and SHM sidecars,
        # and `MemoryStore.__init__` can raise, so the name may be unbound.
        if store is not None:
            store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    total = result.total_candidates
    survived = total - result.skipped_noise - result.skipped_non_persisting
    return {
        "corpus": _corpus_identity(root),
        "funnel": {
            "total_candidates": total,
            "skipped_noise": result.skipped_noise,
            "skipped_non_persisting": result.skipped_non_persisting,
            "skipped_existing": result.skipped_existing,
            "inserted": result.inserted,
            "beliefs_in_store": stored,
        },
        "rates": {
            "survival": (survived / total) if total else 0.0,
            "admission": (result.inserted / total) if total else 0.0,
            "convergence_gap": survived - result.inserted,
        },
        "candidates_by_extractor": {k: len(v) for k, v in by_extractor.items()},
        "noise_by_extractor": {k: noise_by_extractor[k] for k in by_extractor},
        "noise_by_bucket": {b: noise_by_bucket[b] for b in NOISE_BUCKETS},
        "types": dict(types.most_common()),
        "correction_share": (types.get("correction", 0) / stored) if stored else 0.0,
    }


def _render(m: dict[str, Any]) -> str:
    c, f, r = m["corpus"], m["funnel"], m["rates"]
    lines = [
        "scan_repo admission funnel (#1398)",
        "=" * 62,
        f"corpus     : {c['root']}",
        f"sha        : {c['sha']}"
        + (f"  (WORKING TREE DIRTY: {c['dirty_paths']} paths)" if c["dirty"] else ""),
        f"clock      : {c['now']} (fixed, for reproducibility)",
        "",
        "Funnel",
        "-" * 62,
        f"  total_candidates        {f['total_candidates']:>7}",
        f"  - skipped_noise         {f['skipped_noise']:>7}",
        f"  - skipped_non_persist   {f['skipped_non_persisting']:>7}",
        f"  = survived filters      {f['total_candidates'] - f['skipped_noise'] - f['skipped_non_persisting']:>7}",
        f"  inserted (distinct)     {f['inserted']:>7}",
        f"  skipped_existing        {f['skipped_existing']:>7}",
        f"  beliefs in store        {f['beliefs_in_store']:>7}",
        "",
        "Rates  (two different numbers; state which one you mean)",
        "-" * 62,
        f"  survival  = survived / candidates   {r['survival']:>7.1%}",
        f"  admission = inserted / candidates   {r['admission']:>7.1%}",
        f"  convergence gap (survived - inserted)      {r['convergence_gap']:>4}"
        "   <- candidates that converged onto an already-derived belief id",
        "",
        "Candidates by extractor",
        "-" * 62,
    ]
    for name, n in m["candidates_by_extractor"].items():
        dropped = m["noise_by_extractor"][name]
        lines.append(f"  {name:<12} {n:>7}   noise-dropped {dropped:>6}")
    lines += ["", "Noise by rule (first match wins; sums to skipped_noise)", "-" * 62]
    for bucket, n in m["noise_by_bucket"].items():
        lines.append(f"  {bucket:<16} {n:>7}")
    lines.append(f"  {'TOTAL':<16} {sum(m['noise_by_bucket'].values()):>7}")
    lines += ["", "Belief types admitted", "-" * 62]
    for tname, n in m["types"].items():
        lines.append(f"  {tname:<16} {n:>7}")
    lines.append("")
    lines.append(f"  correction share  {m['correction_share']:.1%}   (#1159 quoted 17.4%)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository to scan (default: this checkout)",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    m = measure(root)
    print(json.dumps(m, indent=2, sort_keys=True) if args.json else _render(m))

    noise_total = sum(m["noise_by_bucket"].values())
    if noise_total != m["funnel"]["skipped_noise"]:
        print(
            f"\nFAIL: per-rule noise columns sum to {noise_total} but scan_repo "
            f"reported skipped_noise={m['funnel']['skipped_noise']}. The "
            "attribution is not reproducing the filter.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
