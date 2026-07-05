# In-code documentation audit — `tests/` + `benchmarks/` — 2026-07-04

> **Third companion to `DOCS-AUDIT-2026-07-04.md`.** The first pass audited the
> doc-file surface (README, `docs/**`, slash-command docs, benchmark READMEs).
> `DOCS-AUDIT-2026-07-04-incode.md` then audited the in-code documentation layer
> of every `.py` under `src/aelfrice/` (103 files) and **explicitly deferred**
> the `tests/` and `benchmarks/` `.py` docstrings to a tracked follow-up (its
> § Scope table: "recorded, **not** line-audited this pass"). This record is
> that follow-up — it closes the in-code layer for the #1075 v4.0.0 release gate.

## Scope

| | Surface | Disposition |
|---|---|---|
| ✅ | Every `.py` under `tests/` — **387 files**, in-code docs only | audited, this record |
| ✅ | Every `.py` under `benchmarks/` — **48 files**, in-code docs only | audited, this record |

**435 files total.** Pinned at `github/main` HEAD `a74fbee5`. In-code
documentation only: module docstrings, class/function/method docstrings,
argparse `help=`/`description=` strings, and inline doc comments that make a
checkable factual claim. Runtime/test logic is out of scope except where a
docstring's claim had to be checked against the assertion it describes.

## Method

15-batch multi-agent audit: files were partitioned into 15 doc-weight-balanced
batches (~29 files each), one reader per batch verifying every doc-bearing
construct against the code it describes — following claims across modules into
`src/aelfrice/**` (e.g. a test docstring asserting "`retrieve_v2` combines
L0+L1+L2.5" was checked against both the test's assertions and `retrieve_v2`'s
real behavior). **Every CRITICAL/HIGH finding was independently re-verified by
direct code inspection** (default-reject on uncertainty) before any fix was
written; 1 HIGH was dropped there. Fixes are surgical, doc-text-only, one
atomic commit per file. No code behavior changed.

## Coverage

- **Files audited:** 435 / 435 (100%)
- **Doc-bearing constructs (docstrings + doc comments):** ~2,917
- **Findings:** 2 CRITICAL + 33 HIGH + 10 MEDIUM + 7 LOW = **52**
  - **49 fixed in place** (this PR) — 1 CRITICAL + 32 HIGH + 10 MEDIUM + 6 LOW
  - **2 deferred** — out of doc-only scope (code changes; see below)
  - **1 dropped** on re-verify — claim not confidently reproducible

Absence of a file from a fix is a recorded disposition (audited-clean), not a
silent skip. The overwhelming majority of the 435 files had no checkable
factual drift — test docstrings mostly restate what a test verifies, and those
that made specific claims (constants, defaults, issue/PR numbers, API names,
version gates) were checked and held.

## Recurring drift patterns

Most findings clustered into a few stale-reference classes, useful signal for
future audits:

- **`#1031` PreCompact → SessionStart(compact) move.** Several test/e2e
  docstrings still describe the rebuild block as emitted by the PreCompact hook;
  since #1031 the harness rejects `additionalContext` from PreCompact and the
  block is emitted by `SessionStart` when `source == "compact"`.
- **Shipped-but-doc'd-as-placeholder flags.** `use_heat_kernel` (#150/#154) and
  `use_hrr_structural` (#152) shipped and left `PLACEHOLDER_FLAGS`, but several
  docstrings still list them as unwired placeholders.
- **Default-flip drift.** `use_intentional_clustering` (#436) and the heat-kernel
  flag (#154) flipped to default-**on**; docstrings still said default-off.
- **Stale skip messages.** `bench_gate` skip strings cite issues (`#201`, `#228`,
  `#229`) as "not yet implemented" though the modules shipped (or shipped under a
  different design) and those issues are CLOSED.
- **`docs/design/` → `docs/design/historical/` relocations** (commit `096b01c3`):
  `relevance_floor.md`, `lru_query_cache.md`, `belief_retention_class.md`.
- **Stale line-number pointers.** The four `bench_gate/test_bfs_multihop_*.py`
  scaffolds all cited `bfs_multihop.py:155-160` for the zero-weight skip logic,
  which now lives at `bfs_multihop.py:212-213`.

## CRITICAL — fixed in place (1)

| # | File:line | Claim (doc said) | Reality (code does) | Fix |
|---|---|---|---|---|
| 1 | `tests/test_bm25_index.py`:182 (+ module docstring :17, `pytest_addoption` stub :38) | Perf test is opt-in via `pytest --run-perf`. | `--run-perf` is registered nowhere: `tests/conftest.py` has no `pytest_addoption`, and the in-file stub's body is `pass` (pytest does not collect `pytest_addoption` from a plain test module). `pytest --run-perf` errors with "unrecognized arguments"; the test is permanently skipped unless the `_has_run_perf` guard is edited. | Docstrings rewritten to describe reality (no working `--run-perf` flag; bypass the guard to run locally). |

## HIGH — fixed in place (32)

Materially misleading: wrong defaults/counts/versions, wrong API/flag names,
wrong issue references, `unwired/unimplemented` claims false at HEAD, or a
docstring describing an assertion the test no longer makes. Representative
entries (full list = the 32 `docs(...)` commits on this branch):

| File:line | Was | Now |
|---|---|---|
| `tests/test_heat_kernel.py`:15 | AC7 flag default **False** | default **True** (flipped #154 after #437 11/11 gate) |
| `tests/test_hrr.py`:34,90 | noise-floor figures cited at **dim=2048** | test uses `DEFAULT_DIM` = **512** (`hrr.py:37`); `1/√512 ≈ 0.044` |
| `tests/test_insert_belief_gate.py`:6 | `INSERT_BELIEF_ALLOWLIST` = 4 modules | 5 modules — adds `wonder.lifecycle` (#548) |
| `tests/test_doctor_classify_orphans.py`:149 | `alpha+beta < 2` | `alpha+beta <= 2` (`store.py:3892`, inclusive) |
| `tests/test_cli.py`:551 | `aelf health` "is gone" | retained as deprecated `argparse.SUPPRESS` alias of `aelf doctor graph` |
| `tests/test_mcp_server.py`:377 | `edges` "removed in v1.2.0" | never removed; both keys emitted indefinitely (`mcp_server.py:812`) |
| `tests/test_composition_tracker.py`:5 | 4 placeholder flags | 2 (`use_signed_laplacian`, `use_posterior_ranking`); heat-kernel/HRR shipped |
| `tests/test_clustering_integration.py`:3 | byte-identical **default-OFF** | default-**ON** (#436) |
| `tests/test_hook_project_context_filter.py`:111 | `scope='user'` is a taxonomy member | no `'user'` scope in #688 taxonomy; user-promotion is `lock_level==LOCK_USER` |
| `tests/e2e/test_source_type_discrimination.py`:85 | records `source_type` in `belief_corroborations` | reads `source_kind` from `ingest_log` |
| `tests/regression/test_commit_ingest_latency.py`:11 | 200 ms p95 envelope | `P95_BUDGET_MS = 1500.0` |
| `tests/test_cli_eval.py`:138 | "future R5 CI status-check surface" | R5 shipped: `.github/workflows/eval-calibration.yml` |
| `tests/test_context_rebuilder_hook.py`:5 / `tests/e2e/test_hook_inject_roundtrip.py`:3 | PreCompact emits the block | SessionStart(compact) emits it; PreCompact neutered (#1031) |
| `tests/test_derivation_worker_route_overrides.py`:9,12 | `if was_inserted` guard in `scanner.py:301-310` | relocated to `derivation_worker.py` (#265 PR-B) |
| `tests/bench_gate/test_{contradiction,wonder_consolidation}.py`:18 | modules "not yet implemented (#201/#228)" | modules ship; issues CLOSED |
| `tests/bench_gate/test_dedup.py`:4 | skips until `#197` detector ships | `aelfrice.dedup` ships but exposes no `classify()`; gate errors, not skips |
| `tests/bench_gate/test_intentional_clustering.py`:9 | retrieval wiring is a pending follow-up | wiring shipped default-on (#436 R6) |
| `tests/retrieve_uplift_runner.py`:17,72,804 | `use_hrr_structural` placeholder; clustering "mutually exclusive with compression" | HRR shipped (not exposed on `retrieve()`); clustering composable since #878 |
| `benchmarks/context_rebuilder/__init__.py`:10 | fidelity scoring "tracked at #138" (undone) | #138 shipped; `score_continuation_fidelity` lives in `score.py` |
| `benchmarks/context_rebuilder/__main__.py`:23 | `replay.py` has no `__main__`; routes via alias | `replay.py` has its own `__main__` using `runpy.run_module` |
| `benchmarks/mab_reader.py`:4 | writes file "compatible with `exp5_score.py`" | `exp5_score.py` never existed; scorer is `mab_adapter.py` |
| `benchmarks/bfs_latency_v3.py`:30,55 | "~25k edges"; `is_bfs_enabled` line 1144 | ~20.2k edges (10k×2 SUPPORTS + 200 CITES); `is_bfs_enabled` at line 2385 |
| `tests/test_slash_commands.py`:216 | all `HIDDEN_SUBCOMMANDS` hidden from `--help` | `mcp` is visible in `--help` (real `help=`); "hidden" means "no slash file" |

## MEDIUM (10) + LOW (6) — fixed in place

Stale doc-path pointers (`docs/design/` → `historical/`; `docs/specs/` →
`docs/design/`), stale line-number references (the `bfs_multihop.py:155-160`
family → `212-213`), version-attribution slips (`v0.5.0` → `v0.6.0`), and
arithmetic/count typos in explanatory comments (`Three` → `Four`;
`20%` → `~16.7%`; `60%` → `~66%`; `bm25.py line 243` → `244`;
`dynamic_probe` fire threshold `median` → `1.5× median`). Each is one of the
`docs(...)` commits on this branch.

## Deferred — NOT fixed here (2, require code changes)

These surfaced through the doc audit but are code defects, out of scope for a
doc-text-only PR. Recorded here for coverage honesty; route to a code-fix issue.

| File:line | Issue | Why deferred |
|---|---|---|
| `benchmarks/longmemeval_budget_sweep.py`:87-94 | The documented `Usage:` recipe crashes: the script calls `retrieve_v2(..., top_k=top_k, ...)` but `retrieve_v2` (`retrieval.py:3366`) takes `l1_limit`, not `top_k`, and has no `**kwargs` — `TypeError` on the first sweep iteration. | Fix is a code change (rename the call-site kwarg to `l1_limit`), not a docstring edit. A doc note alone would leave a broken benchmark. |
| `tests/test_contradiction.py`:478 | `test_class_names_cover_all_three_classes` asserts over `CLASS_NAMES`, which now has **six** entries (`contradiction.py:97-104`), not three; the name implies full coverage it does not provide. | Fixing requires renaming the test or extending its assertion (a code change), not a doc edit. |

## Dropped on re-verify (1)

| File:line | Finding | Why dropped |
|---|---|---|
| `tests/test_value_compare.py`:89 | Claim that the `\b`-boundary test cannot distinguish correct word-boundary matching from a substring match (because the two tokens share a `group_id`). | The docstring names `asynchrony`; the refutation reasoned about `asynchronous`. The exact `ENUM_VOCAB` group membership needed to rewrite the docstring correctly could not be pinned with confidence — default-reject rather than ship a possibly-wrong "correction." |

## Reversibility

Every commit on `docs/issue-1075-tests-bench-incode` is doc-text-only and
independently revertable (`git revert <sha>`); no code, tests, or fixtures
changed behavior. `git revert github/main..HEAD` undoes the whole pass.
