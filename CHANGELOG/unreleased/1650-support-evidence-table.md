### Internal

- **Stores gain a `belief_support_evidence` table for SUPPORTS matches ([#1650](https://github.com/robotrocketscience/aelfrice/issues/1650)).** Each row records one match from a supporting belief to one sentence of a target belief, keyed by supporter, target, and kind. A row also holds the target sentence's index, the supporter's session and `ingest_log` ids, the matched text, an applied flag, and a timestamp. A match that was only logged has the flag set to 0. An index on the target id serves lookups by target. A fresh store has the table, and an existing store gains it the next time it opens. Deleting either belief deletes its rows. Nothing writes the table yet, and nothing changes in retrieval, edges, or confidence.

  The table and its index add two statements to every store open. At the repo-store layout, a settled store's writable open now issues 90 statements instead of 88: 59 `CREATE ... IF NOT EXISTS` instead of 57, a DDL battery of 61 instead of 59, and 85 statements that a read-only handle avoids instead of 83. Outside the repo-store layout, the open issues 89 instead of 87. On a store with one expired lock, it issues 98 instead of 96. The [#1658](https://github.com/robotrocketscience/aelfrice/issues/1658) `hrr_expand_neighbors` entry keeps the counts measured at that change, and the markers that bound those counts to `benchmarks/store_open_cost.py` move here, re-derived.
  <!-- derived: benchmarks/store_open_cost.py#statements = 90 -->
  <!-- derived: benchmarks/store_open_cost.py#creates = 59 -->
  <!-- derived: benchmarks/store_open_cost.py#ddl_statements = 61 -->
  <!-- derived: benchmarks/store_open_cost.py#avoidable_statements = 85 -->
  <!-- derived: benchmarks/store_open_cost.py#no_repo_identity_statements = 89 -->
  <!-- derived: benchmarks/store_open_cost.py#expired_lock_statements = 98 -->
