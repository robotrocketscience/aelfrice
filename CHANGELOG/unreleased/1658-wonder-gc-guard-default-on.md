### Changed

- **Wonder GC now runs at every SessionStart by default, and it keeps phantoms you are partway to promoting ([#1658](https://github.com/robotrocketscience/aelfrice/issues/1658)).**
  - **Before:** `AELFRICE_WONDER_AUTOGC` was off unless you set it to `1`, so stale phantoms stayed in the store and kept reaching retrieval after their 14-day TTL. When GC did run, it also collected a phantom you had restated or corroborated, because those supports land on twin beliefs and corroboration rows that leave the phantom's posterior unchanged.
  - **Now:** GC skips any phantom with a support that evidence promotion (#1650) counts: a corroboration you spoke, or a complete restatement you typed in another session. Then the SessionStart hook runs GC once per session unless you set `AELFRICE_WONDER_AUTOGC` to `0`, `false`, `no`, or `off`. The manual `aelf wonder --gc` uses the same guard.
  - **Upgrading:** the first session after you upgrade soft-deletes every phantom that is already stale. Each sweep writes a `wonder.gc` feed row and a stderr notice, and `aelf restore <id>` brings a phantom back.
