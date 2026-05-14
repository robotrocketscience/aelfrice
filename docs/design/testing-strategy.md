# Testing strategy

aelfrice ships three layers of automated tests. Each layer answers a different
question; landing a new test in the wrong layer produces either green-but-blind
coverage or slow-and-flaky CI. This page is the contract for where new tests go.

Companion to umbrella issue #334 (E2E job) and follow-up #370 (push:main +
failure surfacing).

## Layer 1 — unit tests (`tests/test_*.py`)

**Question answered:** does this module behave correctly in isolation?

- Run on every PR via the `pytest (3.12)` and `pytest (3.13)` matrix in
  `.github/workflows/ci.yml`.
- In-process imports. Mocks are allowed for store init, network, subprocess.
- Should be fast (< 1s per file typical). The full unit suite is the dev
  inner-loop signal.
- **Land here when:** you are adding or changing a single module's
  behavior and the test can be written against that module's public API
  with the rest of the system stubbed.

## Layer 2 — integration tests (`tests/test_*.py`, real-store)

**Question answered:** do these modules agree on a contract when wired through
a real `BeliefStore`?

- Same workflow file as unit tests; same matrix.
- In-process imports, but **no mocks for the store, schema, or migrations**.
  Use `tmp_path` for the DB.
- Examples in tree: `test_legacy_migration.py`, `test_v1_to_v1x_migration.py`,
  `test_retrieve_v2.py`, `test_replay_full_equality.py`.
- **Land here when:** the test exercises a contract between two or more
  modules (e.g. ingest writes a column that retrieval reads), and the
  contract is enforceable without crossing a process boundary.

## Layer 3 — end-to-end tests (`tests/e2e/test_*.py`)

**Question answered:** does the installed binary, against a real DB, behave
the way an external user would experience it?

- Run by `.github/workflows/e2e.yml`:
  - on PR with the `e2e` label (opt-in for PRs touching cross-module seams),
  - on every push to `main` (post-merge regression catch),
  - across an install-method matrix: `uv-tool`, `pipx`, `venv-pip`.
- **No in-process imports of `aelfrice.*`.** Tests invoke the installed
  `aelf` binary or `python -m aelfrice.mcp_server.serve` via `subprocess.run`
  and assert on its observable output.
- No mocks at all. Fixtures: `tmp_path` DB, the public-safe project under
  `tests/e2e/fixtures/tiny-project/`, and the synthetic v1.4 snapshot at
  `tests/e2e/fixtures/v14-snapshot.db` (rebuildable via
  `tests/e2e/build-v14-snapshot.sh`).
- Failure surfacing (#370): a push:main failure opens an
  `attn:e2e-failure` issue; a PR failure adds the `attn:e2e-failure` label.
  Both surface in `aelf-scan §1`.
- **Land here when:** you are catching a regression class that crossed a
  module *seam* and was missed by unit/integration tests, or that depends
  on how `aelf` is installed.

### Class of bug each E2E scenario catches

The seed scenarios are deliberately picked to cover the regression classes
that unit tests structurally cannot catch:

| Scenario | File | Class caught |
|---|---|---|
| install → onboard → search | `test_install_onboard_search.py` | install-time wiring; fixture-project loadable |
| hook → inject roundtrip | `test_hook_inject_roundtrip.py` | hook ↔ ingest ↔ rebuild ↔ injection seam |
| source-kind discrimination | `test_source_type_discrimination.py` | the #190 R1 class — constant defined in module A, never recorded by path B |
| v1.4 → current migration | `test_migration_v14_to_current.py` | migration regressions on real DB shapes |

Scenarios 4 (lock-survives-reopen) and 6 (`aelf upgrade-advice`) are listed in
#334 as additive follow-ups, not part of the issue-close gate.

## Choosing a layer — quick decision

```
Is the test behavior observable only when the binary is installed
(install-time wiring, install-method-specific behavior, real subprocess)?
  → Layer 3 (E2E).

Does the test require a real DB and exercises a contract between
two or more modules, but stays in-process?
  → Layer 2 (integration), in tests/.

Otherwise — single module, mocks acceptable?
  → Layer 1 (unit), in tests/.
```

When in doubt, prefer the cheapest layer that can deterministically catch
the regression you have in mind. Cost climbs sharply: unit (~ms), integration
(~10ms–1s), E2E (~30s–8min).

## Bench / regression gates

- E2E wall time: ≤ 8 min p95 over the first 10 main-branch runs (per #334).
  Hard cap is the workflow's `timeout-minutes: 8`.
- Flake budget: zero. A flake is quarantined within one business day and
  tagged `attn:e2e-flake`.
- Unit and integration runtime: untouched by E2E work — the install-matrix
  cost is paid only on push:main and labeled PRs.

## Out of scope

- Cross-OS matrix (macOS / Windows). Linux-only for v1.
- UI / frontend testing.
- Mutation testing (#325 bundle workflow handles this separately).
- Performance / benchmark regression gates (`bench-gated` v2.0 work).
