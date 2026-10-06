### Added

- **The core admission gate's label cache and `aelf core-gate accept` ([#1638](https://github.com/robotrocketscience/aelfrice/issues/1638)).** Two new tables hold the gate's state. `core_gate_labels` caches one A, B, or C label per belief content hash and classifier version, so a label never applies to changed content or under a changed classifier. `core_gate_batches` records each classifier batch that is emitted, under an id derived from the classifier version, the creation time, and the batch's content hashes. Both tables are additive: a fresh store has them, and an existing store gains them the next time it opens. The hidden `aelf core-gate accept <batch-id>` command reads the host model's reply from stdin, checks it with the strict `core_gate` parser, and caches the labels in one transaction. It exits 1 and writes nothing if the batch is unknown, already accepted, or from an earlier classifier version, or if the reply doesn't label every snippet exactly once. `aelf doctor core-gate --emit` and the Stop hook's session-end batch emit batches.

  The two tables add two statements to every store open. At the repo-store layout, a settled store's writable open now issues 89 statements instead of 87: 58 `CREATE ... IF NOT EXISTS` instead of 56, a DDL battery of 60 instead of 58, and 84 statements that a read-only handle avoids instead of 82. Outside the repo-store layout, the open issues 88 instead of 86. On a store with one expired lock, it issues 97 instead of 95. The [#1561](https://github.com/robotrocketscience/aelfrice/issues/1561) entry in the 5.0.0 notes keeps the counts measured at that release, and the markers that bound those counts to `benchmarks/store_open_cost.py` move here, re-derived.
  <!-- derived: benchmarks/store_open_cost.py#statements = 89 -->
  <!-- derived: benchmarks/store_open_cost.py#creates = 58 -->
  <!-- derived: benchmarks/store_open_cost.py#ddl_statements = 60 -->
  <!-- derived: benchmarks/store_open_cost.py#avoidable_statements = 84 -->
  <!-- derived: benchmarks/store_open_cost.py#no_repo_identity_statements = 88 -->
  <!-- derived: benchmarks/store_open_cost.py#expired_lock_statements = 97 -->
