# Docs audit — 2026-05-26

> Full audit of 133 markdown files against current HEAD (`4a49c60d`, v3.3.0). Five batches dispatched in parallel; reports synthesized here. Goal: every claim in our docs backed by code or git history. This document is the audit landing point; per-file findings drive the atomic fix commits that follow.

## Scope

| Batch | Files | Audited |
|---|---|---|
| A — entry + concepts | README, CHANGELOG (+v0–v3), CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, docs/README, docs/concepts/* | 15 |
| B — user docs + features | docs/user/* (9) + docs/feature-*.md (4) | 13 |
| C — slash command docs | src/aelfrice/slash_commands/*.md | 21 |
| D — design docs | docs/design/*.md | 66 |
| E — ADRs / audits / experiments / aux | adr/, audits/, experiments/, benchmarks/**, tests/**/README, PR template | 15 |
| **Total** | | **130** (3 trivial non-audit files: `docs/adr/template.md`, `tests/fixtures/issue_creation_audit/issue_521_body.md`, `.pytest_cache/README.md`) |

## TL;DR

The codebase has shipped fast — three major versions (v1, v2, v3) — and the docs reflect a project that thinks it's still pre-v1.0 in several places. The substantive findings cluster into seven cross-cutting themes:

1. **The v3.1 #814 contradict-auto-demote removal is half-propagated.** README, PHILOSOPHY, QUICKSTART, INSTALL, LIMITATIONS, and `slash_commands/lock.md` still describe the "5 contradicting feedback events → auto-demote a lock" mechanism that `store.py:600` explicitly dropped. COMMANDS.md correctly documents the removal — the doc set is internally inconsistent.
2. **The v3.0.1 #730 uv-only install collapse is half-propagated.** INSTALL.md correctly says aelfrice is uv-only since v3.0.x; QUICKSTART, CONFIG, MCP, LIMITATIONS, and COMMANDS still recommend `pip install aelfrice`. The `aelf mcp` error string in code still emits the pip hint.
3. **`slash_commands/reason.md` instructs the agent to use a CLI flag that doesn't exist.** It says `aelf feedback <id> --direction help|reject`; the actual subparser takes a positional `signal` with `choices=["used", "harmful"]`. Live agents reading the doc will fail at the CLI.
4. **CLI / slash surface area has outgrown the docs.** `aelf graph` (#629), `aelf scope-out` (#856), `aelf export-canvas`, `aelf export-obsidian`, `aelf cadence-score`, `aelf label` are not in COMMANDS.md. SLASH_COMMANDS.md says "Nineteen markdown files" — actual is 21 (`graph.md`, `scope-out.md` missing from the reference table). LIMITATIONS.md still says "No graph viz" even though `aelf graph` ships DOT+JSON.
5. **MCP tool count is stale in two places.** COMMANDS.md L60 + MCP.md L117 say 12; actual is 15; even `mcp_server.py` docstrings say 14. The intro of MCP.md gets 15 right — internal contradiction.
6. **Governance docs are framed pre-v1.0.** CONTRIBUTING.md, RELEASING.md, and ROADMAP.md all treat v1.0 as forthcoming or recently shipped. ROADMAP's versions-at-a-glance table stops at v3.0.1 (no v3.1, v3.2, v3.3 rows). RELEASING says "PyPI publish gated until v1.0.0" — package has been on PyPI for many releases.
7. **Nine design docs carry "Status: spec, no implementation" banners against shipped reality.** `docs/design/README.md` disclaims the directory as historical, but a reader landing from a search who follows the banner could implement a feature that already shipped.

There are no audit-blocking factual errors in the design-doc bodies (Batch D); the discussions are correct historical record. The user-facing surfaces (Batches A–C) are where the load-bearing fixes live.

## Severity legend

- **CRITICAL** — load-bearing claim on a high-traffic user-facing surface, contradicted by current code. Fix before anything else.
- **HIGH** — factual error or stale governance/positioning claim that a reader at HEAD will encounter and act on incorrectly.
- **MEDIUM** — stale banner, internal inconsistency, or out-of-date layout/list that mostly mis-frames context without immediately breaking action.
- **LOW** — cosmetic: stale source-file line refs, minor narrative drift.

## Recommended fix order

The order below clusters fixes by file/scope so the work splits cleanly into atomic commits. Each entry is sized as one commit (or a small group of trivially-related commits).

### CRITICAL — propagate v3.1 #814 contradict-auto-demote removal

| Commit | File(s) | Change |
|---|---|---|
| F1 | `README.md` L130 | Drop or rewrite the "five independent harmfuls auto-demote" paragraph |
| F2 | `docs/concepts/PHILOSOPHY.md` L76, L141 | Rewrite "Locks, not just decay" — describe the `aelf unlock` / `aelf delete` posture per COMMANDS.md L28 |
| F3 | `docs/user/QUICKSTART.md` L80 | Drop the auto-demote bullet |
| F4 | `docs/user/INSTALL.md` L78 | Drop the auto-demote bullet |
| F5 | `docs/user/LIMITATIONS.md` L38, L39 | Drop the auto-demote bullet; rewrite the "`CONTRADICTS` edges drive demotion pressure" sentence |
| F6 | `src/aelfrice/slash_commands/lock.md` L11 | Drop the "resist demotion until 5+ contradicting feedback events" claim |

### CRITICAL — propagate v3.0.1 #730 uv-only install collapse

| Commit | File(s) | Change |
|---|---|---|
| F7 | `docs/user/QUICKSTART.md` L3, L6 | Replace `pip install aelfrice` → `uv tool install aelfrice` |
| F8 | `docs/user/CONFIG.md` L3 | Same swap in the lead paragraph |
| F9 | `docs/user/MCP.md` L13, L26, L31 | Replace `pip install "aelfrice[mcp]"` → `uv tool install --with fastmcp aelfrice` (or `uv tool install "aelfrice[mcp]"`); update the doc's reproduction of the `aelf mcp` error string |
| F10 | `docs/user/LIMITATIONS.md` L79 | Drop the "`pip` works" qualifier |
| F11 | `docs/user/COMMANDS.md` L60 | Replace `pip install 'aelfrice[mcp]'` |
| F12 | `src/aelfrice/mcp_server.py` (code, not doc) | If the error string at runtime still emits the pip hint, update there (defer to operator — code-side fix outside docs-audit scope, but worth flagging). |

### CRITICAL — slash command factual errors

| Commit | File(s) | Change |
|---|---|---|
| F13 | `src/aelfrice/slash_commands/reason.md` L33, L46 | Replace `aelf feedback <id> --direction help\|reject` → `aelf feedback <id> used` / `aelf feedback <id> harmful`; the rolling fork-resolver instructions must match the actual subparser |
| F14 | `src/aelfrice/slash_commands/setup.md` L10–L11, L17–L23 | Rewrite to reflect that setup installs ~7 hook surfaces by default (transcript-ingest, commit-ingest, session-start, stop-hook, search-tool, search-tool-bash, plus statusline), not a single UserPromptSubmit entry |

### HIGH — origin count, MCP tool count, surface drift

| Commit | File(s) | Change |
|---|---|---|
| F15 | `README.md` L135, L165 | Origin tier values are 9 (added `user_transcript` at v3.3+); bump "eight" → "nine" in both the prose and the Lin's-pillar table |
| F16 | `docs/concepts/ARCHITECTURE.md` L32 | Default `token_budget=2000` → `token_budget=2400` (match `retrieval.py:119`) |
| F17 | `docs/concepts/ARCHITECTURE.md` L154–162 | Refresh the hooks table to cover v3.x default-on additions: `aelf-stop-hook`, `aelf-search-tool`, `aelf-search-tool-bash`, cadence Stop/UPS dispatch |
| F18 | `docs/concepts/ARCHITECTURE.md` L171–190 | Drop or update the "manual is the v1.4 ship default" line — `DEFAULT_TRIGGER_MODE` flipped to `"threshold"` at v3.1.0 (#746) |
| F19 | `docs/concepts/ROADMAP.md` L39 | Token budget 2000→2400 in the same row |
| F20 | `docs/user/COMMANDS.md` L60 | "12 memory tools" → "15 memory tools" |
| F21 | `docs/user/MCP.md` L117 | "twelve pure handlers" → "fifteen pure handlers" |
| F22 | `src/aelfrice/mcp_server.py` docstrings (code-side note) | `start_mcp_server` docstring + module docstring say "14"; should be 15 |
| F23 | `docs/user/COMMANDS.md` | Add rows for `aelf graph`, `aelf scope-out`; either document or correctly mark-hidden `export-canvas`, `export-obsidian`, `cadence-score`, `label`; fix L39 `status` semantics (aliases `stats`, not `health`) |
| F24 | `docs/user/SLASH_COMMANDS.md` L3 + reference table | "Nineteen" → "Twenty-one"; add `/aelf:graph` and `/aelf:scope-out` rows |
| F25 | `docs/user/LIMITATIONS.md` L47 | Drop the "No graph viz" bullet; `aelf graph` emits DOT + JSON with color-coded edges |
| F26 | `docs/user/INSTALL.md` L71 | Clarify `aelf status` is the counts replacement (aliases `stats`); structural-auditor replacement is `aelf doctor graph` |

### HIGH — p99 latency reconciliation (operator decision needed)

| Commit | File(s) | Change |
|---|---|---|
| F27 | `README.md` L58, `docs/concepts/ROADMAP.md` L156, `docs/user/CONFIG.md` L107 | Three places render the `stack-r1-r3` p99 differently: 4.5 ms / ~0.65 ms / 0.328 ms. **Operator: which number is the canonical bench-anchored one?** Whichever is right, propagate to the other two; flag the others STALE-pre-source-of-truth |

### HIGH — HARNESS_INTEGRATION.md needs a rewrite

| Commit | File(s) | Change |
|---|---|---|
| F28 | `docs/concepts/HARNESS_INTEGRATION.md` L19, L43, L50 | Fix DB path: `<git-common-dir>/aelfrice/memory.db` (drop the spurious `/.git/` segment — `git rev-parse --git-common-dir` already returns the `.git/` dir itself) |
| F29 | `docs/concepts/HARNESS_INTEGRATION.md` L21, L103, L167 | Remove or rewrite references to `aelf remember` / `aelf:remember` — neither command exists. Replace with `aelf lock` / `aelf onboard` per actual surface |
| F30 | `docs/concepts/HARNESS_INTEGRATION.md` L151, L169 | `aelf:validate` slash form does not exist (renamed `/aelf:promote`); replace, or remove the validate reference if not applicable |
| F31 | `docs/concepts/HARNESS_INTEGRATION.md` L29–35, L52, L183 | Refresh v1.2 / v1.3 framing — hooks are default-on since v2.1.0 (#529); federation is read-only since v3.0 (#650/#655/#688) |

### HIGH — governance docs framed pre-v1.0

| Commit | File(s) | Change |
|---|---|---|
| F32 | `CONTRIBUTING.md` L7, L20, L75–83, L77, L83, L94 | Replace pre-v1.0 framing throughout. The "What's likely to land where" section (v1.0.1 / v1.1.0 / v1.2.0 / v1.3 / v2.0 plan) is wholly stale — every named version has shipped. "Once contributions are open" framing is stale (project actively accepts contributions). |
| F33 | `CHANGELOG/v3.md` | OK overall — `## [Unreleased]` section is by-design per the contributing-changelog protocol; flagged only for reader-context, not a defect |
| F34 | `docs/concepts/RELEASING.md` L7–8, L42, L76, L84 | Pre-v1.0 versioning policy (`0.x.y` milestones + "semver in force at 1.0.0") is stale. "PyPI publish gated until v1.0.0" is wrong — package live. Pre-release `v0.9.0-rc1` example pre-dates v1.0. Branch-protection "after v1.0 flip" framing is two majors stale. |
| F35 | `docs/concepts/ROADMAP.md` L1–12, L16–33, L113–131, L133–144, L225–231, L242–244 | Rewrite the Origin section, refresh the versions-at-a-glance table to v3.3.0 (currently stops at v3.0.1), drop/refresh the v1.x "Planned" section, fix Compatibility section semantics (no v2/v3 mentioned), update Validation tense framing |
| F36 | `docs/concepts/PHILOSOPHY.md` L62, L66, L68, L115, L132, L142, L147 | Refresh v1.x forward-looking framing — most of these describe behavior that has shipped (v1.3 posterior reranking, v1.6 ingest_log table, v2.0 contracts). Either tense-shift to past or rewrite to current state |

### HIGH — SECURITY.md

| Commit | File(s) | Change |
|---|---|---|
| F37 | `SECURITY.md` L11 | `aelf --version` IS a registered flag — drop the "indirect" framing |
| F38 | `SECURITY.md` L20 | "(once `v1.0.0` ships)" — package published; remove the qualifier |
| F39 | `SECURITY.md` L36 | DB path: present per-project (`<git-common-dir>/aelfrice/memory.db`) as the primary; `~/.aelfrice/memory.db` is the legacy fallback per `db_paths.py:57–74`, not the canonical location |

### MEDIUM — design-doc status banners (12 files)

One commit per file is overkill; group as one commit with a uniform banner addition pattern.

| Commit | File(s) | Change |
|---|---|---|
| F40 | `docs/design/substrate_decision.md`, `v2_view_flip.md`, `v2_derivation_worker.md`, `v2_replay.md`, `v2_relationship_detector.md`, `v2_phantom_promotion_trigger.md`, `rebuild_silent_vs_always.md`, `relevance_floor.md`, `query_understanding.md`, `belief_retention_class.md`, `rebuild_eval_harness.md`, `lru_query_cache.md` | Replace `Status: spec, no implementation` (or equivalent) with `Status: shipped per #N — historical design record`. List of shipping issues and modules per individual file in Batch D report. Don't rewrite bodies. |

### MEDIUM — benchmark + experiment docs

| Commit | File(s) | Change |
|---|---|---|
| F41 | `benchmarks/README.md` | Refresh the v1.0.x framing to v3.3.0; drop "mab_entity_index_adapter.py not yet present" (it exists); fix `docs/BENCHMARKS.md` → `docs/concepts/BENCHMARKS.md`; resolve the `benchmarks/datasets.toml` reference (either ship it or drop the reference); re-verify the "`use_hrr` raises TypeError" claim |
| F42 | `docs/experiments/EXP-001-bm25-params.md` L39, L98–103 | Phase 1 (Python BM25 reranker) shipped — `src/aelfrice/bm25.py` is in tree as a v1.5.0-era BM25F module. Update from "not implemented" to "shipped" with reference to that file. Drop the `aelfrice/retrieval/bm25.py (does not exist)` comment. |
| F43 | `benchmarks/context-rebuilder/README.md` L8–12, L316, L380, L437–453 | Refresh v1.2.0/v1.4.0 framing; fix the two `docs/BENCHMARKS.md` mentions to `docs/concepts/BENCHMARKS.md` |
| F44 | `tests/corpus/v2_0/README.md` L21–60 | Layout tree omits `implements_edge/` and lists `retrieve_uplift/` + `bfs_potentially_stale/` which don't exist. Fix the tree against actual disk state |
| F45 | `docs/audits/CLI_SURFACE_AUDIT.md` | Already self-disclaimed as v1.3-era historical; the gap has roughly doubled (current ~45 subparsers vs memo's 22). Either refresh in place or supersede with this audit's data |

### LOW — stale source line refs (one commit)

| Commit | File(s) | Change |
|---|---|---|
| F46 | `docs/design/feature-aelf-confirm-cli.md` L4 (`mcp_server.py:432-476` → `:748`), `docs/design/feature-aelf-delete-cli.md` L5 (`store.py:1315` → `:1934`), `docs/design/feature-doc-linker.md` L5 (`derivation.py:64` → `:106`), `docs/design/v2_view_flip_scanner_call_site.md` L13, L20 (multiple scanner.py refs drifted) | One-line edits — re-grep and update line numbers |

## What's OK

For completeness — these files were audited and pass:

- All four CHANGELOG/v*.md files (immutable historical record; spot-checked).
- `CODE_OF_CONDUCT.md`, `docs/README.md`, `docs/concepts/README.md`, `docs/concepts/BENCHMARKS.md`.
- 18 of 21 slash command docs (`doctor`, `delete`, `confirm`, `unlock`, `promote`, `wonder`, `graph`, `eval`, `tail`, `rebuild`, `scope-out`, `core`, `uninstall`, `upgrade`, `onboard`, `search`, `locked`, `status`).
- All 4 top-level `docs/feature-*.md` (hot-path, posterior-temperature, rerank-relevance-corpus, zeta-posterior-rerank) — well-aligned with code.
- 53 of 66 design docs (41 OK_CURRENT + 12 explicitly OK_HISTORICAL).
- `docs/adr/0001`, `docs/adr/0002`, `docs/adr/README.md` (template not audited).
- `benchmarks/uri_baki_retest/RESULTS.md`, `benchmarks/context-rebuilder/fixtures/README.md`, `tests/corpus/replay_soak/README.md`, `tests/fixtures/bench_smoke/README.md`, `.github/pull_request_template.md`.
- `docs/audits/README.md`.

## What needs operator input

1. **p99 latency reconciliation (F27).** Three numbers (4.5 ms / ~0.65 ms / 0.328 ms) cannot all be true. Which bench artifact is the source of truth? Same question for the README marketing claim "+0.2851 absolute NDCG@k (+94.8%) versus the v1.4 raw-BM25 baseline" — what was the bench run and where is the artifact? (Not flagged by auditors as factual error; flagged here for operator anchoring.)
2. **Scope of the marketing refresh.** Several governance docs (CONTRIBUTING, RELEASING, ROADMAP) read pre-v1.0. Full rewrite or surgical refresh? Suggest: surgical for this PR, full rewrite (or replacement with newer roadmap) as a separate planned phase.
3. **Code-side fix (F12, F22).** The `aelf mcp` error string in `src/aelfrice/mcp_server.py` still emits a `pip install` hint. `mcp_server.py` docstrings say "14 tools". These are code-changes, technically outside docs scope — fix in this PR or surface as a follow-up?
4. **Design-doc status banner format (F40).** Bulk add a single-line addendum per file (`Update YYYY-MM-DD: shipped per #N — historical design record`) or rewrite the existing "Status:" lines? Adding an addendum preserves the original write-time honesty.
5. **CLI_SURFACE_AUDIT.md (F45).** Refresh in place or supersede with this docs-audit document and let CLI_SURFACE_AUDIT remain frozen-historical?

## Estimated PR shape

- ~46 logical changes (commits), grouped per file as listed above.
- Mostly small line edits; F32–F36 (governance refresh) are the largest by line count.
- No code changes required for the docs fixes themselves (F12 and F22 are flagged as out-of-scope code-side issues — operator call).
- Discretion grep should be clean — these changes are pure factual corrections; no introduction of session-name / model-name content.

## Reports

Per-batch reports retained at:

- `/tmp/audit-batch-A.md`
- `/tmp/audit-batch-B.md`
- `/tmp/audit-batch-C.md`
- `/tmp/audit-batch-D.md`
- `/tmp/audit-batch-E.md`

These are ephemeral; this synthesized audit document is the durable record.
