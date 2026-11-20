### Documentation

- **BENCHMARKS.md lists BFS as off by default ([#1928](https://github.com/robotrocketscience/aelfrice/issues/1928)).** The page listed breadth-first search among the always-on lanes that reflect production. `is_bfs_enabled` resolves false unless `AELFRICE_BFS` or `[retrieval] bfs_enabled` turns it on, and the academic adapters pass `use_bfs=True`, so their numbers don't describe a default install. The page now lists BFS with the off-by-default lanes and says that the expansion gate skips it on broad prompts.
