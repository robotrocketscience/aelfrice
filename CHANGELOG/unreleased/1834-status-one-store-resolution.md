### Fixed

- **`aelf status` reports the counts of the store its `store:` line names ([#1834](https://github.com/robotrocketscience/aelfrice/issues/1834)).** The command resolved the store for the `store:` and `store rule:` lines, resolved it again to open the store for the counts, and resolved it a third time for the `hrr.persist_state` line. When the `[store]` marker that applied changed between those steps, the output named one store and printed the counts of another. `aelf status` now resolves the store once, and every line describes that store.
