"""Fire census: does each aelfrice feature fire on real stores? (#1887)

One read-only probe per feature on `main`. Each probe counts the rows or
log lines the feature writes when it runs, and reports `fired`, `silent`
or `untraceable` per store, with integer counts only.

Modules:

  * `core` -- the `Result`, `Ctx` and `Probe` types and the read helpers.
  * `probes_<subsystem>` -- the probes, one module per subsystem.
  * `registry` -- every probe, checked for duplicate ids.
  * `__main__` -- the runner (`python -m benchmarks.real_store_fire_census`).
  * `copy_store` -- byte-copies a store folder for the runner to read.

`benchmarks/real_store_fire_census/README.md` says how to run it.
"""
