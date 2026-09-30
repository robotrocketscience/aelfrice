### Fixed

- **A belief can't be stored with a `created_at` that isn't a timestamp ([#1629](https://github.com/robotrocketscience/aelfrice/issues/1629)).** `insert_belief` now raises `ValueError` for a value that doesn't parse as ISO-8601, so no writer can reintroduce the filename-as-date case. Rows already in a store aren't checked or changed.
