# v2.0 spec: reproducibility harness — `aelf bench all` and the v2.0.0 canonical numbers

Spec for issue [#437](https://github.com/robotrocketscience/aelfrice/issues/437). The ship-gate for v2.0: on a fresh clone, `uv sync && aelf bench all` reproduces every published headline number within documented tolerance bands.

Status: **ratified 2026-05-06**. All eight design asks below resolved against the recommendations except #2 (headline cut), where the operator overrode "sized headline cut" with "full benchmarks (no sized cut)". See § Ratification at the bottom for the resolved set and the implications of the override.

## What's being decided

Six contract calls left open by the issue body:

1. **CLI shape** — `aelf bench` subcommand vs a freestanding `benchmarks/run.py`. Where the dispatcher lives.
2. **What "all" means** — every adapter at full size, or a sized "headline" cut per adapter.
3. **Canonical-numbers schema** — what `benchmarks/results/v2.0.0.json` actually contains, and how it differs from the per-adapter outputs each adapter already emits.
4. **Tolerance bands** — relative (±X%), absolute (±X points), or per-metric.
5. **CI shape** — nightly-only, PR-blocking, or two-tier (nightly canonical + PR smoke).
6. **External-dependency policy** — adapters that pull from HuggingFace, LLM-judge paths, and `/tmp/...` data dirs need a defensible default for CI runs without API keys / network.

After this issue ships, `benchmarks/results/v2.0.0.json` is checked-in canonical state, the README badge "reproducibility: ✅ as of $DATE" is wired against it, and the v2.0 ship-gate is operationally green or red against a single artifact.

## Recommendation

**Ship at v2.0 as a thin `aelf bench` subcommand that dispatches to the existing adapter `main()` functions, with a sized-headline cut, relative-with-floor tolerance bands, and a two-tier CI (nightly canonical + PR smoke).**

### CLI shape

```
aelf bench all      [--adapters mab,locomo,...]  [--out PATH]  [--canonical]
aelf bench <name>   [--subset N]                  [--out PATH]
```

The subcommand lives in `src/aelfrice/cli.py` alongside `_cmd_unlock`, `_cmd_locked`, etc. It dispatches to the existing adapter modules under `benchmarks/` via their `main(argv)` functions — adapters already accept `--subset`, `--retrieve-only`, `--out`, etc. The dispatcher's only job is the loop, the JSON merge, and the canonical-vs-cron filename split.

Reasons:

1. **No duplicate adapter code.** Every adapter already has a `__main__` that does the right thing for that benchmark. The dispatcher is ~80 LOC of fan-out, not a re-implementation.
2. **`aelf bench` is the user-visible surface.** README shows `uv sync && aelf bench all`; nothing about benchmarks lives outside the `aelf` CLI for a fresh-clone user.
3. **Sub-subcommand granularity already exists.** Adapters handle their own flags; the dispatcher passes through `--subset N` per-adapter.

Rejected:

- **Freestanding `python -m benchmarks.run`.** Splits the surface — fresh-clone users would have to learn two entry points (`aelf` for the product, `python -m benchmarks.run` for the harness). One CLI is cleaner.
- **Auto-discover adapters by glob.** Looks magical, fails opaquely. Explicit registry in the dispatcher (a list of `(name, module, default_subset)` tuples) is debuggable.

### What "all" means — sized headline cut

`aelf bench all` runs each adapter at a **fixed headline cut**, not the full benchmark. Headline cuts:

| Adapter | Headline cut | Reason for the cut |
|---|---|---|
| MAB | All four splits, full | Conflict_Resolution / Test_Time_Learning / Long_Range / Accurate_Retrieval; the published number is per-split. |
| LoCoMo | 10 conversations | Matches the published leaderboard size. Full is 10 conversations. |
| LongMemEval | All 6 question types, oracle subset | The cleaned oracle dataset is the headline (~500 Qs). |
| StructMemEval | `--bench small` across all 4 tasks | `big_bench` is ~10× the runtime; spec'd as a follow-up. |
| AMA-Bench | Full 208 episodes | Small enough to run end-to-end. |

`--canonical` flag asserts the run matches the headline cut and writes to `v2.0.0.json` (vs `v2.0.0-cron-<date>.json` or `v2.0.0-local-<date>.json`). Mismatched cut → write refused with a non-zero exit. Prevents accidental overwrite of the canonical artifact with a partial run.

### Canonical-numbers schema

```json
{
  "schema_version": 2,
  "label": "v2.0.0 canonical",
  "captured_at_utc": "2026-MM-DDTHH:MM:SSZ",
  "git_commit": "<sha>",
  "aelfrice_version": "2.0.0",
  "harness_version": "1",
  "headline_cut": {
    "mab": {"splits": ["all"], "subset": null},
    "locomo": {"conversations": 10, "subset": null},
    "longmemeval": {"question_type": null, "subset": null, "oracle": true},
    "structmemeval": {"tasks": ["all"], "bench": "small"},
    "amabench": {"max_episodes": 208, "domains": ["all"]}
  },
  "results": {
    "mab": {
      "Conflict_Resolution": {"f1_avg": 0.NN, "exact_match": 0.NN, "n": NNN, "tolerance_band": {...}},
      "Test_Time_Learning": {...},
      "Long_Range_Understanding": {...},
      "Accurate_Retrieval": {...}
    },
    "locomo": {"f1_avg": 0.NN, "by_category": {"multi-hop": ..., "temporal": ..., ...}, "tolerance_band": {...}},
    "longmemeval": {"by_question_type": {...}, "tolerance_band": {...}},
    "structmemeval": {"by_task": {"location": ..., "accounting": ..., "recommendations": ..., "tree": ...}, "tolerance_band": {...}},
    "amabench": {"by_qa_type": {"A": ..., "B": ..., "C": ..., "D": ...}, "by_domain": {...}, "tolerance_band": {...}}
  }
}
```

Per-metric `tolerance_band` lives next to the value:

```json
"f1_avg": {
  "value": 0.4823,
  "lower": 0.4470,
  "upper": 0.5176,
  "band_kind": "relative",
  "band_pct": 7.3
}
```

`schema_version: 2` distinguishes from `v1.2.0-pre.json` (`schema_version: 1`). The harness validates against schema_v2 on read.

### Tolerance bands — relative-with-floor

Each metric carries `lower` and `upper` bounds. Default policy:

- **Relative band:** ±X% of the canonical value, where X is per-metric (defaults: F1 ±7%, exact-match ±10%, latency ±25%).
- **Absolute floor:** the band never falls below ±2 percentage points (prevents tiny-value flapping; e.g. 0.5% → 0.55% is below numeric noise but +10% relative).
- **Per-metric override:** the canonical JSON can declare wider bands for known-noisy metrics (LLM-judge runs, anything with a non-deterministic ranker tie-break).

The bands are calibrated on the first canonical run by running it ≥3 times and taking the observed range × 1.5. This is the **calibration-pass** the issue body alludes to and is documented as a one-time operation in the harness README. Re-calibration is a deliberate operator action, not automatic on drift.

### CI shape — two-tier

**Nightly cron (`replay-soak-gate.yml`-style new workflow):**
- Runs `aelf bench all`, writes `benchmarks/results/v2.0.0-cron-<date>.json`.
- Compares against `v2.0.0.json` per-metric; any value outside its `tolerance_band` is a CI **fail** (issue auto-opened).
- Soft warnings (drift inside the band but >50% of the band width) emit a workflow notice but pass.
- Commits the cron JSON to a `benchmark-results` branch on PR for diffability; never pushes to main.

**PR smoke (added to existing `ci.yml`):**
- Runs `aelf bench mab --subset 5` + `aelf bench amabench --max-episodes 5`. Catches regressions in the harness itself, not in the adapters' published numbers.
- ≤2 minutes wall-clock. No HF download (cached fixture committed under `tests/fixtures/bench_smoke/`).
- Blocks PR merge on failure. Treats result as a smoke signal only — no tolerance bands at this size.

Reasons:

1. **PR-blocking the full canonical run is too slow.** End-to-end against full LongMemEval + LoCoMo is minutes of network + minutes of LLM-judge calls (when keys are configured); putting that on every PR shifts the merge floor by an unacceptable amount.
2. **Nightly is fast enough to catch real regressions.** A regression that hides for a day before a nightly run is acceptable for a reproducibility harness — that's not a user-visible bug, it's an audit signal.
3. **Two tiers separate "did the harness break" (PR smoke) from "did the published numbers break" (nightly canonical).** They have different failure modes and different escalation paths.

### External-dependency policy

Three classes:

| Class | Adapters | Default behavior |
|---|---|---|
| **HF datasets** (network on first run, cached after) | mab, longmemeval, amabench | CI workflow uses `actions/cache@v4` keyed on `HF_DATASET` constants. Cache miss → fall through to live fetch with a 5-minute timeout. Local runs use `~/.cache/huggingface/`. |
| **`/tmp/...` data dirs** | locomo, structmemeval | Doc surface (`docs/COMMANDS.md`) explains the manual download. Harness skips with a clear "data dir missing at $PATH; see docs" message rather than failing — adapter exits 2 (skip), not 1 (fail). Aggregated result records `"status": "skipped_data_missing"`. |
| **LLM-judge paths** (require API key) | longmemeval, optionally locomo | Off by default in CI. Harness records the structural retrieval result; LLM-scoring is a separate `aelf bench score-judge --in <path>` step that runs locally with a configured key. Canonical numbers come from a manual judge run; CI does not regenerate them. |

This keeps `uv sync && aelf bench all` runnable on a clean machine with no API keys, and degrades to "skipped" rather than "failed" when external data is unavailable. The README badge reflects the cron-canonical state, not the local-machine state.

## Decision asks

- [ ] **Confirm the `aelf bench` CLI shape** vs `python -m benchmarks.run`. Recommendation: `aelf bench` subcommand. Reject only if there's a v3 plan to extract `benchmarks/` into its own package.
- [ ] **Confirm the headline cut table.** Each row is a quantity choice; LoCoMo at 10 conversations and StructMemEval at `small_bench` are the two non-obvious ones. Alternatives: full LoCoMo (10 already is full), `big_bench` for StructMemEval (~10× cost; defer to a follow-up issue).
- [ ] **Confirm `schema_version: 2` for `v2.0.0.json`.** Alternative: `v3` to leave room. Recommendation: `2`; bump on the next breaking schema change.
- [ ] **Confirm the relative-with-floor tolerance policy.** Alternatives: absolute-only (simpler, harder to calibrate), per-adapter custom (more flexible, more places to forget). Recommendation: relative + floor; per-metric overrides allowed.
- [ ] **Confirm two-tier CI shape.** Alternatives: nightly-only (no PR signal), PR-blocking-canonical (too slow). Recommendation: two-tier.
- [ ] **Confirm "skipped" vs "failed" for missing data dirs.** Alternative: hard-fail to force operators to fix data setup. Recommendation: skip, on the grounds that the harness is reproducibility infrastructure, not a setup linter — a missing `/tmp/LoCoMo` is a configuration issue, not a regression.
- [ ] **Confirm canonical numbers come from a manual judge run, not nightly.** Alternative: nightly with a CI-side API key in secrets. Recommendation: manual; LLM-judge cost is non-trivial and the cron should be deterministic.
- [ ] **Confirm README badge wiring.** A static "reproducibility: ✅ as of $DATE" updated by the cron when canonical comparison passes; turns red on a band-busting regression and stays red until acknowledged. Alternative: shields.io with a JSON endpoint. Recommendation: static, committed.

## Why this is judgment-scope

The CLI shape, the tolerance policy, and the CI tiering are interdependent design calls. Picking them in isolation produces a harness that doesn't compose: a PR-blocking canonical run with relative bands is slow AND noisy; a nightly-only run with no PR smoke means harness regressions hide for a day. The recommendation above picks a coherent set; alternative sets exist but should be picked as a set, not piecemeal.

The acceptance items 1, 2, 3 are mostly mechanical once these decisions are settled. Item 4 (tolerance bands) and item 5 (README badge) are the design-loaded ones.

## Downstream impact

- `pyproject.toml`: new console-script entry for `aelf bench` if not already present (the existing `aelf` script will dispatch via subcommand router; verify no naming collision).
- `src/aelfrice/cli.py`: new `_cmd_bench_*` family.
- `benchmarks/run.py` **(new)**: thin dispatcher consumed by `aelf bench`. Adapter modules unchanged.
- `benchmarks/results/v2.0.0.json` **(new)**: committed canonical artifact.
- `benchmarks/results/README.md` **(new or extended)**: describes the schema, the calibration-pass procedure, the cron filename convention.
- `.github/workflows/`: new nightly cron workflow (e.g. `bench-canonical.yml`) + ~30 LOC added to `ci.yml` for PR smoke.
- `tests/fixtures/bench_smoke/`: tiny pinned fixtures (10 LoCoMo turns, 5 MAB QA pairs, etc.) so the PR smoke runs offline. Fixtures are derived from public datasets; license attribution committed alongside.
- `README.md`: badge added; `## Reproducibility` section linking to `docs/COMMANDS.md` and the canonical artifact.
- `docs/COMMANDS.md`: `aelf bench` documented next to `aelf locked`, `aelf doctor`, etc.
- `docs/LIMITATIONS.md`: drops the "no end-to-end reproducibility surface" caveat (or adds it if not yet listed).

Estimated effort: ~250 LOC dispatcher + harness, ~150 LOC tests, ~80 LOC CI workflow, ~200 LOC docs. Adapter code unchanged.

## Out of scope (deferred)

- **Hyperparameter sweeps for the v2.0 numbers themselves.** This issue ships the harness, not the calibration. The first canonical run records whatever numbers v2.0 produces; tuning is separate.
- **Cross-version regression tracking** (`v2.0.0` vs `v2.1.0` deltas). Out of scope until a v2.1 ship-gate exists.
- **`big_bench` StructMemEval coverage.** ~10× runtime; spec'd as a follow-up issue once the small_bench numbers are stable.
- **Public dashboard.** A web surface for the cron history is downstream of the harness existing at all.
- **GPU-required benchmarks.** None of the current adapters need GPU; if a future adapter does, the policy lives in that adapter's add-issue, not this one.

## Provenance

- Source-of-truth: [docs/ROADMAP.md § v2.0.0](ROADMAP.md) — "Reproducibility harness: `benchmarks/results/v2.0.0.json` is canonical; CI runs the academic suite nightly."
- Existing baseline: `benchmarks/results/v1.2.0-pre.json` (schema_v1; supersedes-on-merge).
- Existing adapters: `benchmarks/{mab,locomo,longmemeval,structmemeval,amabench}_adapter.py` — already consumed `aelfrice.ingest.ingest_turn` + `aelfrice.retrieval.retrieve_v2`; no API change needed for the harness.
- Adjacent: #438 (correction-detection eval, nominally one of the suites this harness runs once it spec'd ratifies).

## Ratification

Ratified 2026-05-06 by the operator. Resolved set:

| # | Decision | Choice | vs Recommendation |
|---|----------|--------|-------------------|
| 1 | CLI shape | `aelf bench` subcommand | Same |
| 2 | What "all" means | **Full benchmarks (no sized cut)** | **Override**: spec recommended sized headline cut; operator picked full |
| 3 | Schema version | `schema_version: 2` | Same |
| 4 | Tolerance bands | Relative-with-floor (±X% per metric, ±2pp absolute floor) | Same |
| 5 | CI shape | Two-tier: nightly canonical + PR smoke | Same |
| 6 | Missing data dirs | Skip with `status: skipped_data_missing` (exit 2) | Same |
| 7 | LLM-judge cadence | Manual local run (off by default in CI) | Same |
| 8 | README badge | Static, committed by cron | Same |

### Implications of the #2 override (full benchmarks)

The original headline-cut table is superseded by:

| Adapter | Cut | Notes |
|---------|-----|-------|
| MAB | All four splits, full | Same as spec |
| LoCoMo | Full (all 10 conversations) | Spec said 10; full happens to be 10. No change. |
| LongMemEval | Full dataset (not oracle subset) | **Increased** vs spec (oracle was the recommended cut) |
| StructMemEval | `--bench big` across all 4 tasks | **~10× runtime** vs spec's `small_bench` |
| AMA-Bench | Full 208 episodes | Same as spec |

Cron runtime is multi-hour rather than the spec's tens of minutes. Trade-off: stronger reproducibility claim, no "we only validate a subset" caveat in the README.

The PR-smoke tier (`aelf bench mab --subset 5 + amabench --max-episodes 5`) is unchanged — still ≤2 minutes wall-clock — because the override is about the canonical cron, not the smoke signal.

### Implementation order

1. Capture this ratification (this commit).
2. Dispatcher: `benchmarks/run.py` + `_cmd_bench_all` extension to `src/aelfrice/cli.py`. Subprocess each adapter; never tight-couple to adapter internals.
3. Tolerance module: `benchmarks/tolerance.py` — relative-with-floor, per-metric overrides, in-band/warn/fail classification.
4. Tests for dispatcher and tolerance module.
5. PR smoke: `tests/fixtures/bench_smoke/` pinned fixtures + `ci.yml` smoke job.
6. Nightly cron: `.github/workflows/bench-canonical.yml` — pushes `v2.0.0-cron-<date>.json` to a `benchmark-results` branch (sidesteps the `main` branch ruleset; orthogonal to the cron-push problem in #461 which is for `replay-soak`).
7. Docs: README badge + `## Reproducibility` section, `docs/COMMANDS.md` entry, `docs/LIMITATIONS.md` caveat removal.
8. Skeleton `benchmarks/results/v2.0.0.json` (schema-v2) with TBD numbers; calibration pass (acceptance #2 from the issue body) is a separate operator action — `aelf bench all --canonical` run locally with all data dirs present.
