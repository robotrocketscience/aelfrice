"""Replay-soak fixture loader + runner (#403 deliverable A).

Loads `tests/corpus/replay_soak/v0.1/*.jsonl`, replays each row through
`aelfrice.derivation.derive` + `MemoryStore.record_ingest` /
`insert_belief`, then runs `replay_full_equality` and returns the
report.

Used by:

- `tests/test_replay_soak_corpus.py` — per-PR per-run invariant check.
- `.github/workflows/replay-soak.yml` (`scripts/replay_soak_run.py`) —
  daily soak run that appends the report to `.replay-soak-status.json`.

The runner deliberately uses a fixed `ts` so the derived ids and log
rows are reproducible across runs. Determinism is the contract under
test; clock drift would mask it.
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.derivation import DerivationInput, derive
from aelfrice.replay import FullEqualityReport, replay_full_equality
from aelfrice.store import MemoryStore

REPLAY_SOAK_CORPUS_ROOT: Path = (
    Path(__file__).parent / "corpus" / "replay_soak" / "v0.1"
)

# Fixed ts so the harness is deterministic across runs and across
# hosts. Any value before 2027 is fine — `derive` only uses ts to
# stamp the belief's created_at.
_FIXED_TS: str = "2026-05-04T00:00:00+00:00"


def _iter_rows() -> list[dict]:  # type: ignore[type-arg]
    rows: list[dict] = []  # type: ignore[type-arg]
    for path in sorted(REPLAY_SOAK_CORPUS_ROOT.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def run_replay_soak(store: MemoryStore) -> FullEqualityReport:
    """Drive the full v0.1 corpus through the ingest path then probe.

    For each row:

    1. Build a `DerivationInput` from the row fields.
    2. Call `derive(input)` to produce a `DerivationOutput`.
    3. If the output produces a belief (`persist=True` paths), call
       `store.record_ingest` to append the log row, then
       `store.insert_belief` if the belief is not already present.

    After all rows are loaded, return `replay_full_equality(store)`.
    Caller asserts `has_drift is False`.

    The corpus is hand-authored to avoid `persist=False` paths
    (empty / question forms), so every row contributes one belief
    and one log row.
    """
    inserted: set[str] = set()
    for row in _iter_rows():
        inp = DerivationInput(
            raw_text=row["raw_text"],
            source_kind=row["source_kind"],
            source_path=row.get("source_path"),
            raw_meta=row.get("raw_meta"),
            ts=_FIXED_TS,
        )
        out = derive(inp)
        # Public corpus is hand-authored to avoid skip paths.
        assert out.belief is not None, (
            f"row {row['id']!r}: derive returned no belief — "
            f"corpus must avoid persist=False shapes"
        )
        store.record_ingest(
            source_kind=row["source_kind"],
            source_path=row.get("source_path"),
            raw_text=row["raw_text"],
            raw_meta=row.get("raw_meta"),
            derived_belief_ids=[out.belief.id],
            ts=_FIXED_TS,
        )
        if out.belief.id not in inserted:
            store.insert_belief(out.belief)
            inserted.add(out.belief.id)

    return replay_full_equality(store)
