# DOCS-AUDIT-2026-07-04-docfiles

**Scope:** tranche 2 of the #1075 v4.0.0 documentation audit — the remaining
doc-file buckets deferred by `DOCS-AUDIT-2026-07-04.md` (doc-file surface) and
the in-code ledgers (`-incode.md`, `-tests-bench.md`).

**Audited against:** code at `github/main` HEAD `13f7d0b4` (v3.8.0 + post-release main).

**Method:** 8-batch multi-agent fan-out, every line of every in-scope file read
and checked; each CRITICAL/HIGH finding independently re-verified against the
code before its fix was written; source line-pointer corrections re-grepped at
HEAD before landing. Disposition per this issue's rules: current-behavior docs
fixed in place; frozen snapshots (`docs/design/historical/`, `docs/adr/`,
`docs/experiments/`, prior `docs/audits/` records) corrected via
banner/annotation only; CHANGELOG entries verified against code at the
corresponding tags; code-side drift filed separately, not papered over in docs.

## Coverage

**95 / 95 files audited (100%).** Every file appears below as a finding or in
the verified-clean list; absence from the ledger would mean unaudited, not clean.

| Bucket | Files |
|---|---|
| `CHANGELOG.md` + `CHANGELOG/v0–v3.md` | 5 |
| `docs/design/*` (current-behavior) | 58 |
| `docs/design/historical/*` | 13 |
| `docs/adr/*` | 5 |
| `docs/audits/*` (prior + current records) | 7 |
| `docs/experiments/*` | 1 |
| `tests/` READMEs + fixture | 4 |
| `.github/pull_request_template.md` | 1 |
| `CITATION.cff` | 1 |
| **Total** | **95** |

## Tally

**24 findings: 0 CRITICAL, 2 HIGH, 14 MEDIUM, 8 LOW.** 24 fixed in place /
annotated (0 deferred, 0 code-side). 71 files verified clean.

### HIGH

1. **`docs/audits/README.md`** — the audit index omitted
   `DOCS-AUDIT-2026-07-04-tests-bench.md`, a record present on disk (the
   tests/+benchmarks in-code tranche). A present record missing from the index
   fails the coverage rule. → index entry added.
2. **`docs/design/context_rebuilder.md`** — the delivery section documented the
   rebuilder as a PreCompact hook emitting `additionalContext` via
   `emit_pre_compact_envelope()`; superseded by #1031 (the harness rejects
   PreCompact `additionalContext`; the block now ships on `session_start()` with
   `source=="compact"`). → delivery-channel banner added; the original design
   narrative and the `rebuild_v14` code path are unchanged.

### MEDIUM

- **`CHANGELOG/v3.md`** — `[Unreleased]` compare link read `v3.7.0...HEAD`
  (double-counts everything shipped in v3.8.0) → `v3.8.0...HEAD`.
- **`CITATION.cff`** — stale fixed operation count ("eight") vs the current
  CLI / MCP / slash-command surface → version-agnostic phrasing.
- **`docs/design/bfs_multihop.md`** (×2) — `BFS_EDGE_WEIGHTS` located in
  `retrieval.py` (actually `bfs_multihop.py:61`, contradicting the doc's own
  code block); function named `bfs_expand` (shipped as `expand_bfs`,
  `bfs_multihop.py:118`).
- **`docs/design/feature-aelf-confirm-cli.md`** — sample output showed
  `mean 0.500->0.667`; shipped `_cmd_confirm` emits a single post-update mean
  (`cli.py:2803-2807`).
- **`docs/design/feature-temporal-spine.md`** — retrieval no-op guard is
  `store.has_edge_type(EDGE_TEMPORAL_NEXT)` (`retrieval.py:3285`), not
  `count_edges_by_type()`.
- **`docs/design/historical/v2_derivation_worker.md`** — frozen spec cites
  `tests/test_concurrency.py`, renamed post-ship to
  `tests/test_worktree_concurrency.py` → audit-note annotation (body left as-authored).
- **`docs/design/hook_hardening.md`** — `_format_hits` line pointer 419 → 1787.
- **`docs/design/phantom_trigger_generation.md`** (×2) — `<cadence-checkpoint>`
  pointer `hook.py:807–826` → `853–877`; `is_bfs_enabled` pointer
  `retrieval.py:2258` → `2385`.
- **`docs/design/v2_close_the_loop.md`** — dangling link
  `docs/no_embeddings_first.md` (never existed) → `docs/concepts/PHILOSOPHY.md`
  § "Determinism is the property".
- **`tests/corpus/replay_soak/README.md`** (×3) — corpus grew a 7th ingest
  source kind (`claude_memory`, `INGEST_SOURCE_CLAUDE_MEMORY`): layout tree,
  row total (60→66), and the "6 non-`legacy_unknown` kinds" note all reconciled
  to 7 kinds / 66 rows.

### LOW

- **`CITATION.cff`** — added `version: 3.8.0` + `date-released` (CFF 1.2.0 recommends both).
- **`docs/design/bfs_multihop.md`** — stale `store.py:617` / `store.py:559,576,584`
  line pointers dropped in favour of symbol names (`propagate_valence`,
  `_fire_invalidation`).
- **`docs/design/context_rebuilder.md`** — turn-window default narrative said 10;
  shipped `DEFAULT_TURN_WINDOW_N = 50`.
- **`docs/design/feature-correction-detection-eval.md`** — `relationship_detector.py`
  pointers `:73-78, :322` → `:75-80, :344`.
- **`docs/design/feature-type-aware-compression.md`** — `resolve_use_type_aware_compression`
  pointer `retrieval.py:1703-1731` → `2013-2040`.
- **`docs/design/phantom_trigger_generation.md`** — `hook_manifest.json` count 8 → 10.
- **`docs/design/relevance-signal.md`** — deferred-note "PreCompact rebuilder
  injects" reworded to the SessionStart(compact) reality (#1031).
- **`docs/design/v3_relatedness_philosophy.md`** — `aelf doctor dedup` →
  `aelf doctor --dedup` (flag, not subcommand).

## Recurring themes

- **#1031 PreCompact→SessionStart(compact) move** recurs across three docs
  (`context_rebuilder`, `relevance-signal`, and — fixed in prior tranches —
  the hook/test docstrings). The PreCompact hook no longer injects.
- **Post-ship default flips / renames** under-documented: `claude_memory`
  ingest source kind (replay_soak corpus), `expand_bfs` symbol, temporal-spine
  guard.
- **Source line-pointer drift** is the dominant LOW class; where a pointer was
  purely positional it was dropped in favour of the symbol name to resist
  re-drift.

## Verification limits

- **Lab-sourced measurement numbers** (benchmark F1 / coverage figures cited in
  design docs and CHANGELOG) cannot be re-derived from the public tree; checked
  only for internal consistency and treated as externally sourced, per the
  issue's sole permitted verification limit.
- **`tests/fixtures/issue_creation_audit/issue_521_body.md`** is a verbatim
  fixture copy of a past issue body used as test input; verified-clean as a
  frozen fixture, not audited against current code (it must stay byte-identical
  for the test).

## Remaining on #1075 after this tranche

Only the **v4.0.0 release cut** to PyPI per `docs/concepts/RELEASING.md`. The
documentation surface — doc-file (`-07-04.md`), in-code `src/` (`-incode.md`),
in-code `tests/`+`benchmarks/` (`-tests-bench.md`), and these remaining
doc-file buckets — is now fully in the ledger set.
