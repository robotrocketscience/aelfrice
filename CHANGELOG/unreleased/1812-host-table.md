### Internal

- **`aelf setup`, `unsetup`, `uninstall`, and `doctor` read the host choices and the default from `aelfrice.hosts` ([#1812](https://github.com/robotrocketscience/aelfrice/issues/1812), [#1903](https://github.com/robotrocketscience/aelfrice/pull/1903)).** Setup and unsetup pick a per-host executor from a table, and doctor hands off to a per-host scan when the host has one. The search-tool and memory-mirror hooks take a `host` argument, and the installed console scripts leave it at the default. The installed hooks and every `--host` option behave as in 5.4.0.
