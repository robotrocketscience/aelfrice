"""Context-rebuilder eval harness scaffolding (#136, v1.4.0).

Importable companion to the directory `benchmarks/context-rebuilder/`
(hyphenated, kept for back-compat with the v1.2.0-shipped skeleton at
`benchmarks/context-rebuilder/eval_harness.py` and the synthetic
fixture corpus at `benchmarks/context-rebuilder/fixtures/synthetic/`).

This package is *scaffolding only*: it loads a transcript fixture,
optionally forces a midpoint context-clear, and measures token-cost
delta + PreCompact hook latency. This package originally shipped as
fidelity-scoring scaffolding only (#136); the continuation-fidelity
scorer landed at #138 and now lives alongside it in ``score.py``
(``score_continuation_fidelity``), wired into ``replay.run()`` via
``score_method`` / ``post_clear_answers``.

Public surface:

  * `replay.run(fixture, ...)` -- end-to-end replay against a
    synthetic `turns.jsonl` fixture; returns a `ReplayResult` with
    per-turn `token_budget_delta` and `hook_latency_ms` fields.
  * `inject.midpoint_clear(...)` -- forces a synthetic context-clear
    at a configurable turn index.
  * `measure.token_budget_delta(...)` / `measure.hook_latency_ms(...)`
    -- the two scaffolding metrics.

Per `docs/design/eval_fixture_policy.md`: synthetic fixtures only on the
public-repo path. Captured fixtures live lab-side and never gate CI.
"""
from __future__ import annotations
