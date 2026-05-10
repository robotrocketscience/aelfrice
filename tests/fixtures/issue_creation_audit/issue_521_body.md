Public-side coding work for **#433** (HRR vocabulary bridge). Filed as a separate sister-claimable issue because #433 carries `bench-gated`, which sister sessions filter out — but the substrate-landing work is independent of the bench evidence and can proceed now. #433 stays open as the bench-gate parent; this issue closes when the substrate ships.

The bench-gate scaffold (`tests/bench_gate/test_vocab_bridge_uplift.py`) and the corpus path (`tests/corpus/v2_0/vocab_bridge/`, lab-side) already exist; what is missing is the production module that the gate's `VocabBridge.rewrite(...)` import will resolve to.

## Acceptance

1. New module `src/aelfrice/vocab_bridge.py` exposing:
    - `class VocabBridge` with at least `rewrite(query: str, store: MemoryStore) -> RewrittenQuery` where `RewrittenQuery` carries the original query plus an ordered list of appended canonical-entity tokens.
    - The bridge harvests anchor surface forms from inbound edges (parity with the BM25F anchor mechanism shipped in #148) and uses HRR bind/probe primitives (#152, #216) to map surface-form-divergent query tokens to canonical entities present in the store.
2. Wiring point in `src/aelfrice/retrieval.py`: a flag-gated call site (default-OFF; flag name `use_hrr_vocab_bridge` to match `use_*` convention in #154 tracker) that runs the bridge before the BM25F + BFS lanes execute, and threads any appended canonicals into the query that downstream lanes see.
3. Spec memo at `docs/feature-hrr-vocab-bridge.md` covering input/output shapes, where the bridge sits relative to BM25F and BFS lanes (§ Position), the bind/probe construction (§ Mechanism), and the bench-gate criterion (§ A2 — references #433's bench-gate spec). The bench-gate scaffold's expected row schema (`query`, `store_beliefs[*].anchors`, `expected_canonicals`) is the contract; document it.
4. Unit tests in `tests/test_vocab_bridge.py` (no lab corpus): synthetic two-anchor-source case demonstrating canonical recovery, plus one negative case (query whose canonicals are not in store → empty append, no exception).
5. Bench-gate scaffold (`tests/bench_gate/test_vocab_bridge_uplift.py`) imports cleanly against the new module — i.e. the lab-side run with `AELFRICE_CORPUS_ROOT` set executes the gate without `ImportError`. Gate evidence is the parent #433's responsibility, not this issue's.

## Out of scope

- Building the labeled `vocab_bridge` corpus (lab-side, blocking the bench gate but not this substrate).
- Bench-evidence sweep itself — that closes #433.
- Default-ON flip — gated on bench evidence; stays default-OFF here.

## Refs

- Parent: #433 (bench gate stays here)
- Composition tracker: #154
- HRR primitives: #152, #216
- Anchor surface-form parity: #148
- Bench-gate scaffold: `tests/bench_gate/test_vocab_bridge_uplift.py`

