### Removed

- **New stores no longer get the `hrr_expand_neighbors` table, and the HRR expand lane no longer reads it ([#1658](https://github.com/robotrocketscience/aelfrice/issues/1658)).** The table cached precomputed neighbours for the default-off HRR expand lane (`AELFRICE_HRR_EXPAND`, `[retrieval] use_hrr_expand`). Only the ablation benchmark ever filled it, so every query already fell back to probing the HRR index live. That live probe is now the only path, so the lane returns the same results as before. `hrr_expand.precompute_expand_neighbors` is gone. A store created before this change keeps its `hrr_expand_neighbors` table, which nothing reads; no migration drops it. The lane's determinism tests now check the live probe's neighbour rows byte for byte, across repeated probes and across two independent index builds.

  Dropping the table's `CREATE TABLE IF NOT EXISTS` removes one statement from every store open. At the repo-store layout, a settled store's writable open now issues 88 statements instead of 89: 57 `CREATE ... IF NOT EXISTS` instead of 58, a DDL battery of 59 instead of 60, and 83 statements that a read-only handle avoids instead of 84. Outside the repo-store layout, the open issues 87 instead of 88. On a store with one expired lock, it issues 96 instead of 97. The [#1638](https://github.com/robotrocketscience/aelfrice/issues/1638) entry in the 5.1.0 notes keeps the counts measured at that release, and the markers that bound those counts to `benchmarks/store_open_cost.py` move here, re-derived.
  <!-- derived: benchmarks/store_open_cost.py#statements = 88 -->
  <!-- derived: benchmarks/store_open_cost.py#creates = 57 -->
  <!-- derived: benchmarks/store_open_cost.py#ddl_statements = 59 -->
  <!-- derived: benchmarks/store_open_cost.py#avoidable_statements = 83 -->
  <!-- derived: benchmarks/store_open_cost.py#no_repo_identity_statements = 87 -->
  <!-- derived: benchmarks/store_open_cost.py#expired_lock_statements = 96 -->
