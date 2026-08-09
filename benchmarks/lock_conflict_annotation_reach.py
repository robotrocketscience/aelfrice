"""#1365 reach measurement: how far does the lock-conflict annotation actually get?

Replaces the AC5 kill experiment, which the operator struck on 2026-08-06 as a
guaranteed null: `benchmarks/longmemeval_adapter.py` runs with
`include_locked=False`, so `lock_consistency` short-circuits on every question
and the flag-on arm is byte-identical to flag-off. A null from an instrument
that cannot fire means "no measurement", not "no effect" — the R3 IDF-clip
failure mode.

What this measures instead: **of the packs aelfrice actually injected, what
fraction would carry at least one `conflicts_with` annotation?**

Method, and why it is not a query replay:

  The obvious harness — replay recorded prompts through `retrieve()` with the
  flag on — is unavailable here. `hook_audit.jsonl` stores only
  `prompt_prefix`, which is truncated at 200 chars, and whose median length IS
  the cap; a prior retrieval A/B that treated it as the query carried a ~4x
  error. So no query is reconstructed at all.

  Instead the audit log records the *belief ids actually injected on each
  fire*, with their lane and lock flag. Those ARE the packs. Reach is computed
  directly on them: for each fire, take the non-locked injected beliefs as
  candidates and the locked ones as the lock set, and run the shipped
  `lock_conflict_annotations` over the pair. That is deterministic, uses the
  real production population, and depends on no reconstructed input.

  Content comes from the live store, opened READ-ONLY: a plain `MemoryStore`
  open is a write (DDL, migrations, scope-id backfill), which a diagnostic must
  not do.

Caveats reported rather than hidden:
  - Beliefs since deleted or retired cannot be scored; they are counted and
    excluded, not silently dropped.
  - The lock set is the one recorded on that fire, not today's — which is the
    honest counterfactual for "would this pack have been annotated".
  - Fires whose recorded lock set is empty are reported separately: the feature
    cannot fire on them by construction, so folding them into the denominator
    would understate reach against the population it can actually reach.

Usage:
    uv run python benchmarks/lock_conflict_annotation_reach.py [--audit PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _audit_default() -> Path:
    from aelfrice.db_paths import _git_common_dir

    git_dir = _git_common_dir()
    if git_dir is None:
        raise SystemExit("not in a git work-tree; pass --audit")
    return git_dir / "aelfrice" / "hook_audit.jsonl"


def _store_default() -> Path:
    from aelfrice.db_paths import _git_common_dir

    git_dir = _git_common_dir()
    if git_dir is None:
        raise SystemExit("not in a git work-tree; pass --store")
    return git_dir / "aelfrice" / "memory.db"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all fires")
    args = ap.parse_args()

    audit = args.audit or _audit_default()
    store_path = args.store or _store_default()
    if not audit.is_file():
        raise SystemExit(f"no audit log at {audit}")

    from aelfrice.lock_consistency import lock_conflict_annotations
    from aelfrice.store import MemoryStore
    from aelfrice.value_compare import extract_values

    # Read-only: opening a store normally runs DDL + migrations, which a
    # measurement must never do to the live store.
    store = MemoryStore(str(store_path), read_only=True)

    content: dict[str, str] = {}

    def _content(bid: str) -> str | None:
        if bid not in content:
            b = store.get_belief(bid)
            content[bid] = b.content if b is not None else ""
        return content[bid] or None

    fires = 0
    skipped_no_locks = 0
    annotated_fires = 0
    missing_beliefs = 0
    total_candidates = 0
    annotated_candidates = 0
    per_lock: Counter[str] = Counter()

    for line in audit.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("hook") != "user_prompt_submit":
            continue
        beliefs = rec.get("beliefs")
        if not isinstance(beliefs, list) or not beliefs:
            continue

        locked_ids = [b["id"] for b in beliefs if b.get("locked")]
        cand_ids = [b["id"] for b in beliefs if not b.get("locked")]
        if not locked_ids:
            skipped_no_locks += 1
            continue
        fires += 1
        if args.limit and fires > args.limit:
            fires -= 1
            break

        locked_pairs = []
        for bid in locked_ids:
            c = _content(bid)
            if c is None:
                missing_beliefs += 1
                continue
            lock = store.get_belief(bid)
            if lock is not None:
                locked_pairs.append((lock, extract_values(c)))

        candidates = []
        for bid in cand_ids:
            c = _content(bid)
            if c is None:
                missing_beliefs += 1
                continue
            candidates.append((bid, extract_values(c)))

        total_candidates += len(candidates)
        if not locked_pairs or not candidates:
            continue
        ann = lock_conflict_annotations(candidates, locked_pairs)
        if ann:
            annotated_fires += 1
            annotated_candidates += len(ann)
            for lock_id in ann.values():
                per_lock[lock_id] += 1

    store.close()

    if fires == 0:
        print("no scorable fires found", file=sys.stderr)
        return 1

    pack_reach = annotated_fires / fires
    cand_rate = annotated_candidates / total_candidates if total_candidates else 0.0

    print("#1365 lock-conflict annotation — reach on the live store")
    print(f"  audit log                      {audit}")
    print(f"  fires with >=1 lock (scored)   {fires}")
    print(f"  fires with no lock (excluded)  {skipped_no_locks}")
    print(f"  beliefs no longer in store     {missing_beliefs}")
    print()
    print(f"  PACK REACH  fires carrying >=1 annotation   "
          f"{annotated_fires}/{fires} = {pack_reach:.2%}")
    print(f"  candidate annotation rate                  "
          f"{annotated_candidates}/{total_candidates} = {cand_rate:.2%}")
    print()
    if per_lock:
        top = per_lock.most_common(5)
        share = top[0][1] / annotated_candidates if annotated_candidates else 0.0
        print(f"  distinct locks implicated      {len(per_lock)}")
        print(f"  top lock's share               {share:.1%}")
        for lock_id, n in top:
            print(f"    {lock_id}  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
