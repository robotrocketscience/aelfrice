### Fixed

- **`aelf doctor` reports the count of the store its `store:` line names ([#1843](https://github.com/robotrocketscience/aelfrice/issues/1843)).** The store check resolved the store to open it for the belief count, resolved it again for the path on the `store:` line, and resolved it a third time for the `store rule:` line. When the `[store]` marker that applied changed between those steps, the `store:` line paired one store's count with another store's path. The check now resolves the store once, and its count, path, and rule describe that store.
