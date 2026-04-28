# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-`1.0.0` releases are atomic milestones building toward the first
installable release; see the roadmap in [README.md](README.md).

## [Unreleased]

### Fixed

- **`aelf doctor` now flags hooks that wrap a script in `2>/dev/null || true`** ([#113](https://github.com/robotrocketscience/aelfrice/issues/113), [#114](https://github.com/robotrocketscience/aelfrice/issues/114)). Previously the shell-meta heuristic short-circuited before extracting the script path, so a stale `bash /abs/path.sh 2>/dev/null || true` hook entry — like the one #113 reproduces against the long-deleted `hook-aelf-search.sh` — was reported as `skipped` instead of `broken`. The check now extracts the script path even with a trailing wrapper and reports it as broken when the file is missing. Hooks using the silent-failure pattern are additionally surfaced as a soft warning regardless of whether the underlying script resolves, with a fix-hint pointing at `~/.aelfrice/logs/hook-failures.log`. Doctor also now tails that log when present, so future bash hook wrappers that redirect stderr into it have a discovery path back to the user.

### Added (toward v1.2.0 final)

- **Ingest enrichment schema** ([docs/ingest_enrichment.md](docs/ingest_enrichment.md)). Three coupled additions: `DERIVED_FROM` edge type (valence 0.5, mirrors `CITES`); `anchor_text TEXT` column on `edges` carrying the citing belief's own phrasing of the relationship, capped at 1000 characters at the dataclass boundary; `session_id TEXT` column on `beliefs` plus a real `sessions` table (`MemoryStore.create_session` / `complete_session` go from no-op stubs to persisting). `ingest_turn` now writes `session_id` end-to-end. Forward-compatible with v1.0 stores: `ALTER TABLE` adds the columns on first open and is idempotent on re-open. The `idx_beliefs_session` index is sequenced after the migration so a v1.0 store can open at all. Producers that populate the new fields densely are the v1.2.0 transcript-ingest, commit-ingest, and triple-extraction paths.
- **Commit-ingest `PostToolUse` hook** ([docs/commit_ingest_hook.md](docs/commit_ingest_hook.md)). New `aelfrice.hook_commit_ingest:main` entry point fires after every successful `git commit` Bash call, parses `[branch shorthash]` from the commit's stdout, runs the triple extractor on the full commit message body (fetched via one `git log -1 --format=%B <hash>` subprocess), and inserts beliefs and edges under a deterministic session id `sha256(branch + ":" + commit_hash)[:16]`. The first ingest path that densely populates `Edge.anchor_text`, `Belief.session_id`, and `DERIVED_FROM` edges in production data, closing the v1.0 limitation that the graph only grew on explicit `aelf onboard` / `aelf remember` calls. Lazy imports keep cold-start cost off the latency budget; commit-message bodies are capped at 4 KB before extraction. Non-blocking on every failure path — a hook problem may NEVER cause `git commit` to feel broken. Wired via `aelf setup --commit-ingest` (opt-in at v1.2; default flip pending production-corpus latency telemetry); `aelf unsetup --commit-ingest` strips it. New `aelf-commit-ingest` project script registered in `pyproject.toml`. Setup machinery generalises `_get_event_list(data, event, ...)` so the same builders cover UserPromptSubmit, Stop, and PostToolUse-with-matcher entries; `_build_entry` gains an optional `matcher` field for PreToolUse / PostToolUse shapes. 15 unit tests + 1 latency regression test (in-process p95 < 200 ms across 20 iterations on a fixture commit).
- **Triple extractor** ([docs/triple_extractor.md](docs/triple_extractor.md)). New `aelfrice.triple_extractor` module with `extract_triples(text) -> list[Triple]` (pure regex over a fixed pattern bank — six relation families covering `SUPPORTS`, `CITES`, `CONTRADICTS`, `SUPERSEDES`, `RELATES_TO`, `DERIVED_FROM`, with active and passive forms) and `ingest_triples(store, triples, session_id=None) -> IngestResult` (resolves subject/object beliefs by content-hash lookup, creating with the project default prior when missing; inserts edges with `anchor_text` populated from the citing prose; idempotent on re-run; self-edges dropped). All triple-derived beliefs share a canonical id space via the `triple` source label, so the same noun phrase resolves to the same belief id across every extraction call site. Stdlib-only; no LLM, no POS tagger, no embedding. Reusable by every v1.x prose-ingesting caller — the v1.2.0 commit-ingest hook, transcript-ingest, manual `aelf remember` calls, and the v1.3.0 entity-index path. 38 tests cover the per-pattern templates, negative cases, anchor-text substring guarantee, idempotency, session_id propagation, and self-edge handling.
- **Transcript ingest** ([docs/transcript_ingest.md](docs/transcript_ingest.md)). New `aelfrice.transcript_logger:main` entry point handles four Claude Code hook events through one dispatch: `UserPromptSubmit` → append `{"role": "user", ...}`; `Stop` → read the last assistant message from the transcript path and append `{"role": "assistant", ...}`; `PreCompact` → write a `compaction_start` marker, rotate `turns.jsonl` into `archive/turns-<ts>.jsonl`, spawn `aelf ingest-transcript` detached; `PostCompact` → write a `compaction_complete` marker. JSONL lives at `<git-common-dir>/aelfrice/transcripts/turns.jsonl` (or `~/.aelfrice/transcripts/turns.jsonl` outside a git tree); `$AELFRICE_TRANSCRIPTS_DIR` overrides. Per-turn budget is sub-10ms p99 by skipping `git rev-parse` / `symbolic-ref` on the hot path; `cwd` and Claude Code's own `session_id` from the payload stay on the line. Non-blocking contract: every failure returns exit 0 with a stack trace on stderr.
- **`aelfrice.ingest.ingest_jsonl()`** reads a `turns.jsonl` archive and lowers each line through `ingest_turn`. Within a session, consecutive turns get a `DERIVED_FROM` edge linking turn N+1's last belief to turn N's last belief with `anchor_text` set to turn N's literal text. Idempotent. Returns `IngestJsonlResult(lines_read, turns_ingested, beliefs_inserted, edges_inserted, skipped_lines)`. Compaction markers and malformed lines count under `skipped_lines` and never raise.
- **`aelf ingest-transcript PATH` CLI** prints the `IngestJsonlResult` summary. Used as the detached spawn target of the `PreCompact` hook on rotation; also runnable manually for replay or recovery on archived transcripts.
- **`aelf setup --transcript-ingest`** wires the four logger hook events to the new `aelf-transcript-logger` entry point. Idempotent. `aelf unsetup --transcript-ingest` strips them. The flag is opt-in at v1.2.0.
- **`aelf-transcript-logger` project script** registered in `pyproject.toml`.
- Slash command `aelf:ingest-transcript`.
- **`agent_inferred → user_validated` promotion** ([docs/promotion_path.md](docs/promotion_path.md)). Implements the v1.1.0-designed promotion path. New `Belief.origin` schema column (`TEXT NOT NULL DEFAULT 'unknown'`) tags each belief with one of `user_stated`, `user_corrected`, `user_validated`, `agent_inferred`, `agent_remembered`, `document_recent`, `unknown`. Forward-compatible with v1.0/v1.1 stores: `ALTER TABLE` adds the column on first open, then a one-shot backfill flips locked rows to `user_stated` and correction rows to `user_corrected` (everything else stays `unknown` rather than retroactively claim `agent_inferred`). Producers tag origin explicitly: `scan_repo` writes `agent_inferred` on every onboard belief; `aelf lock` and the MCP `aelf:lock` write `user_stated`. New `aelfrice.promotion` module exposes `promote(store, belief_id)` and `devalidate(store, belief_id)` — provenance flip only, no math change (alpha/beta/lock_level/type unchanged). Each call writes one zero-valence audit row to `feedback_history` tagged `promotion:user_validated` or `promotion:revert_to_agent_inferred`. Idempotent; refuses locked beliefs and `user_stated` rows with a clear "demote first" message. New `aelf validate <belief_id> [--source LABEL]` CLI subcommand and `aelf:validate` MCP tool. `aelf demote` extended one-tier-per-call: drops a lock if locked, else flips `user_validated` → `agent_inferred`. The contradiction tie-breaker grows from three to five precedence classes (`user_stated > user_corrected > user_validated > document_recent > agent_inferred`); `precedence_class()` reads `belief.origin` first, with `lock_level=user` short-circuiting to `user_stated` regardless of origin so the v1.0.1 lock-always-wins invariant holds. `unknown` and `agent_remembered` fall through to `document_recent` (honest unknown bucket). 33 new tests across `tests/test_promotion.py`, `tests/test_contradiction.py`, `tests/test_cli.py`, and `tests/test_mcp_server.py`.
- **`docs/HARNESS_INTEGRATION.md`** ([docs/HARNESS_INTEGRATION.md](docs/HARNESS_INTEGRATION.md)). User-facing operational guide for running aelfrice alongside Claude Code's auto-memory. Documents three coexistence modes (default coexistence after `aelf setup --transcript-ingest`; aelfrice-canonical with a `~/.claude/CLAUDE.md` edit; aelfrice-only after disabling auto-memory) plus a migration recipe (`aelf onboard ~/.claude/projects/<slug>/memory`) for moving accumulated `.md` content into aelfrice as `agent_inferred` beliefs. Rewrites [docs/LIMITATIONS.md § harness conflict](docs/LIMITATIONS.md) to point at the v1.2.0 hook mitigation rather than the v1.0/v1.1 manual `CLAUDE.md` edit.
- **SessionStart hook**. New `aelfrice.hook.session_start()` entry point reads the SessionStart JSON payload (drained for protocol; no fields used at MVP), retrieves L0 locked beliefs via `retrieve()` with empty query, and emits an `<aelfrice-baseline>` block to stdout. Fires once per Claude Code session, before any user message, so the agent enters the session with durable user-asserted ground truth already in context. Distinct tags from UserPromptSubmit's `<aelfrice-memory>` so the model can tell channels apart: baseline = "stuff that's always true," memory = "stuff related to this prompt." Honors the non-blocking hook contract (returns 0 on every failure path; empty store / no locked beliefs / empty stdin / malformed payload all silent). Wires `install_session_start_hook` / `uninstall_session_start_hook` / `resolve_session_start_hook_command` in `aelfrice.setup` mirroring the UserPromptSubmit pair (idempotent, atomic-write, basename-match). Three event channels (UserPromptSubmit, transcript-ingest, SessionStart) coexist in the same `settings.json` without disturbing each other. New `--session-start` flag on `aelf setup` / `aelf unsetup` and new `aelf-session-start-hook` console script.

## [1.2.0a0] - 2026-04-27

Alpha pre-release. **Context rebuilder MVP** lands as an opt-in
`PreCompact` hook that surfaces aelfrice retrieval results before
Claude Code runs its default summarization. Augment-mode only --
the harness still runs its own compaction afterward, so this is
additive context, not a replacement.

This is a vertical slice. Several quality dimensions are deferred
until the supporting infrastructure ships (transcript ingest at
v1.2.0, posterior-weighted ranking at v1.3+, triple-extracted
queries, session-scoped retrieval). The alpha is for measurement
and feedback, not for setting the v1.x default.

### Added

- `aelfrice.context_rebuilder.rebuild()`: pure function taking a
  list of recent turns and an open `MemoryStore`, returns the
  formatted `<aelfrice-rebuild>` XML block with L0 locked beliefs
  and L1 BM25 hits. Per-token union retrieval works around the
  public store's AND-only FTS5 semantics. Deterministic given
  identical inputs (eval-harness contract).
- Two transcript adapters: `read_recent_turns_aelfrice` for the
  canonical `<root>/.git/aelfrice/transcripts/turns.jsonl` log
  (per-turn ingest format from the v1.2.0 transcript-ingest spec)
  and `read_recent_turns_claude_transcript` as a best-effort
  fallback for Claude Code's internal per-session JSONL.
- `pre_compact()` hook entry-point in `aelfrice.hook`: reads the
  PreCompact JSON payload, locates a transcript (canonical
  preferred, Claude-internal fallback), invokes `rebuild()`,
  writes the block to stdout. Honors the non-blocking hook
  contract (returns 0 on every failure path).
- `aelf-pre-compact-hook` console script entry registered for
  `settings.json` hook entries.
- `install_pre_compact_hook` / `uninstall_pre_compact_hook` /
  `resolve_pre_compact_hook_command`: same idempotency,
  atomic-write, and basename-match semantics as the existing
  UserPromptSubmit pair. Two events coexist in the same
  `settings.json` without disturbing each other.
- `aelf setup --rebuilder` flag: also installs the PreCompact hook
  alongside the UserPromptSubmit hook.
- `aelf rebuild` CLI subcommand (with `--transcript`, `--n`,
  `--budget`): manually emits the rebuild block to stdout for
  inspection. Required surface for spec acceptance criterion 5
  (manual mode ships before threshold-mode auto-triggers).
- Matching `rebuild.md` slash command shipped in
  `src/aelfrice/slash_commands/`.

### Deferred (named here so users know they are coming)

- **Session-scoped retrieval.** The schema's `session_id` field is
  not yet populated end-to-end. The MVP retrieves globally.
- **Triple-extractor query construction.** The MVP concatenates
  recent turn text and per-token-unions over the result. A
  triple-extracted query will replace this once the extractor
  ships.
- **Posterior-weighted ranking.** The MVP uses the public BM25
  ranker. The Bayesian-weighted ranker is a v1.3+ candidate.
- **Suppress-mode coordination with the harness.** The MVP runs in
  augment mode only. Suppress-mode lands once eval-harness fidelity
  calibration justifies it.
- **Default trigger threshold.** Augment-mode means the trigger is
  whatever Claude Code's default PreCompact firing is; the rebuilder
  emits on every PreCompact. Per-session-state trigger tuning is
  v1.3+.

## [1.1.0] - 2026-04-27

Minor release: project identity, structural auditor, onboarding
git-recency, legacy-DB migration tool, design memo for the v1.2.0
promotion path, worktree concurrency tests, and the `edges → threads`
user-facing rename. Eight PRs landed between v1.0.3 and v1.1.0.

### Added

- **Per-project DB resolution** ([#88](https://github.com/robotrocketscience/aelfrice/issues/88), PR [#96](https://github.com/robotrocketscience/aelfrice/pull/96)). v1.0.x stored everything in a single global `~/.aelfrice/memory.db`. v1.1.0 introduces a resolution chain in `cli.db_path()`: `$AELFRICE_DB` (override) → `<git-common-dir>/aelfrice/memory.db` (when `cwd` is in a git work-tree; resolved via `git rev-parse --path-format=absolute --git-common-dir`, so worktrees of one repo share one DB) → `~/.aelfrice/memory.db` (legacy fallback for non-git dirs). `.git/` is not git-tracked — the brain graph never crosses the git boundary. New `_git_common_dir()` helper shells out to `git rev-parse` and returns `None` gracefully when git is unavailable. No new runtime dependencies.
- **`aelfrice.auditor` module + rewritten `aelf health`** ([#90](https://github.com/robotrocketscience/aelfrice/issues/90), PR [#100](https://github.com/robotrocketscience/aelfrice/pull/100)). `audit(store) -> AuditReport` runs three mechanical structural checks: `orphan_threads` (edges whose src/dst no longer exists), `fts_sync` (`beliefs_fts` row count vs `beliefs`), `locked_contradicts` (pairs of locked beliefs joined by CONTRADICTS). `aelf health` exits 1 if any check fails. Informational metrics (counts, average confidence, credal gap, thread counts by type) print alongside but don't affect exit. The v1.0 regime classifier is preserved as a separate `aelf regime` command (always exit 0).
- **`aelf status`** ([#90](https://github.com/robotrocketscience/aelfrice/issues/90)). Alias for `aelf health` per lab `COMMAND_DESIGN.md`. Same handler, same exit codes.
- **`aelf regime`** ([#90](https://github.com/robotrocketscience/aelfrice/issues/90)). New command preserving the v1.0 regime classifier output (`supersede` / `ignore` / `mixed` / `insufficient_data`).
- **Store API additions** for the auditor: `count_orphan_edges()`, `count_fts_rows()`, `list_locked_contradicts_pairs()`, `count_edges_by_type()`. Read-only.
- **`aelf migrate`** ([#93](https://github.com/robotrocketscience/aelfrice/issues/93), PR [#104](https://github.com/robotrocketscience/aelfrice/pull/104)). One-shot copy from the legacy global `~/.aelfrice/memory.db` into the active project's resolved DB. Reads source via SQLite `mode=ro` URI (rejects writes at the SQLite layer). Default project-mention filter (belief content references the absolute project root); `--all` overrides. Edges follow only when both endpoints land. Dry-run by default; `--apply` writes. `--from PATH` overrides the source. Idempotent. Never deletes from the source. New module `aelfrice.migrate` with `default_legacy_db_path()` and `migrate(...) -> MigrateReport`.
- **`PRAGMA busy_timeout=5000`** in `MemoryStore.__init__` ([#89](https://github.com/robotrocketscience/aelfrice/issues/89), PR [#102](https://github.com/robotrocketscience/aelfrice/pull/102)). Required for safe multi-worktree access where two processes share one `.git/aelfrice/memory.db`. Without it, the second writer hits `database is locked` immediately instead of waiting.
- **Worktree concurrency test suite** at `tests/test_worktree_concurrency.py` ([#89](https://github.com/robotrocketscience/aelfrice/issues/89)). Five tests: WAL-on regression, busy_timeout regression, two-worktree DB-path identity, concurrent multi-process write correctness (40 beliefs across 2 spawn processes), and a three-round repeat to catch flakes.
- **Onboard git-recency weighting (Tier 2)** ([#94](https://github.com/robotrocketscience/aelfrice/issues/94), PR [#103](https://github.com/robotrocketscience/aelfrice/pull/103)). Scanner records the most-recent author date of the commit that touched each source file, then `scan_repo` writes that as `belief.created_at` so the existing decay mechanism (already in `scoring.py` at v1.0) penalises stale prose. New `_build_file_recency_map(root)` makes ONE `git log --name-only --pretty=format:%aI` call per scan and returns `{relative-path: most-recent-iso-date}`. `SentenceCandidate` gains an optional `commit_date: str | None = None` field. `extract_filesystem` and `extract_ast` accept the recency map and stamp candidates from recency-known files; `extract_git_log` carries each commit's own author date. Files outside git, untracked files, and the entire fallback when git is unavailable continue to use wall-clock `now`. No new ranking math.
- **`docs/promotion_path.md`** ([#95](https://github.com/robotrocketscience/aelfrice/issues/95), PR [#101](https://github.com/robotrocketscience/aelfrice/pull/101)). Design memo for the v1.2.0 `agent_inferred → user_validated` promotion path. Recommends adding `origin TEXT NOT NULL DEFAULT 'unknown'` to `beliefs` (v1.1.0 schema bump deferred to v1.2.0 implementation alongside the command), conservative backfill, flag-only flip mechanism (preserve `α/β` and `lock_level`), `aelf validate <belief_id>` CLI surface mirroring `aelf demote`, tie-breaker slot between `user_corrected` and `document_recent`, zero-valence audit row tagged `source='promotion:user_validated'`. 10 open TBDs explicitly enumerated.
- **`aelf` CLI grew from 12 to 19 subcommands.** Added: `health` (rewritten), `status` (alias), `regime`, `migrate`. Slash commands: `health.md` rewritten, new `status.md`, `regime.md`, `migrate.md`.
- **MCP `aelf:stats` returns `threads` key.** Same integer value as `edges`. `edges` retained for one minor.
- **MCP `aelf:health.features.thread_per_belief` field.** Same value as `edge_per_belief`. `edge_per_belief` retained for one minor.

### Changed

- **`edges` → `threads` user-facing rename** ([#92](https://github.com/robotrocketscience/aelfrice/issues/92), PR [#105](https://github.com/robotrocketscience/aelfrice/pull/105)). All user-facing surfaces — CLI output labels (`aelf stats`, `aelf health`, `aelf migrate`), slash command descriptions, COMMANDS.md / MCP.md prose, auditor finding strings — now use "threads". The internal SQLite `edges` table, the `Edge` Python dataclass, and the `EDGE_*` type constants are unchanged. The auditor's `metrics` dict keys (`threads`, `threads_supports`, `threads_contradicts`, …) follow the rename. Schema docs (ARCHITECTURE.md, design memos) keep `edge` / `Edge` because they describe internals.
- **Project-identity narrative reconciled** ([#99](https://github.com/robotrocketscience/aelfrice/pull/99)). Re-applies the framing fix dropped from #87's squash and sweeps post-#96 docs: ARCHITECTURE.md / COMMANDS.md / INSTALL.md § Database / PRIVACY.md / LIMITATIONS § Project identity all describe the new resolution chain. PRIVACY.md drops "to share across machines: sync the file" and links to LIMITATIONS § Sharing or sync of brain-graph content. INSTALL.md uninstall section uses "resolved DB path" language and a `cli.db_path()`-discovering verification command.

### Deprecated

- **MCP `aelf:stats` JSON: `edges` key.** v1.1.0 emits both `edges` and `threads` with the same integer value. **`edges` is removed in v1.2.0.** Clients should migrate to `threads` now.
- **MCP `aelf:health.features.edge_per_belief` key.** v1.1.0 emits both `edge_per_belief` and `thread_per_belief` with the same value. `edge_per_belief` is removed in v1.2.0.
- **`aelfrice.auditor.CHECK_ORPHAN_EDGES` constant.** Kept as a deprecated alias of `CHECK_ORPHAN_THREADS` for v1.0 importer compatibility. Removed in v1.2.0.

### Documentation

- **Local-only contract locked** ([#87](https://github.com/robotrocketscience/aelfrice/pull/87)). LIMITATIONS.md gains "Sharing or sync of brain-graph content" under "Out of scope by design" with privacy/determinism/audit rationale. ROADMAP.md "Non-goals" adds "Brain-graph sharing, sync, or export between users / machines / projects." `.aelfrice/seed.md`, `.aelfrice.toml` cross-machine identity, and v2.0 cross-project shared store are removed from the roadmap. Bootstrap recipe for new clones is `aelf onboard .`.
- **`docs/promotion_path.md`** linked from ROADMAP § v1.2.0 and LIMITATIONS § Onboarding.

## [1.0.3] - 2026-04-27

Patch release: contradiction tie-breaker, `aelf resolve` CLI, onboard
performance regression test, and a power-user CONFIG.md for the
`.aelfrice.toml` schema. Three feature PRs and a docs PR landed
between v1.0.2 and v1.0.3 — this release surfaces them to PyPI.

### Added

- `aelfrice.contradiction` module: `resolve_contradiction`,
  `find_unresolved_contradictions`, `auto_resolve_all_contradictions`.
  When the graph holds a CONTRADICTS edge, the tie-breaker picks a
  winner via precedence (`user_stated > user_corrected >
  document_recent`; ties broken by recency, then by id), creates a
  SUPERSEDES edge from winner to loser, and writes a
  `feedback_history` audit row tagged
  `source='contradiction_tiebreaker:<rule>'`. v1.0.x collapses to
  three precedence classes (the fourth `agent_inferred` class needs
  a `Belief.origin` field landing in v1.1.0) (#75).
- `aelf resolve` CLI subcommand (12th) — sweeps unresolved
  CONTRADICTS edges in the store and runs the tie-breaker on each.
  Matching `slash_commands/resolve.md`. The 1:1 CLI ↔ slash
  invariant is preserved at 12/12 (#75).
- `tests/regression/test_onboard_perf_50k_loc.py`: regression benchmark
  asserting `scan_repo` finishes in under 60s on a synthetic ~55k-LOC
  project (250 .py + 60 doc files). Marked `regression`. Held against
  the `:memory:` store. Current measured time ~0.8s on Apple Silicon;
  the 60s budget is a regression alarm, not a target (#76).
- `docs/CONFIG.md`: power-user reference for `.aelfrice.toml` —
  full schema, worked examples, and what each setting affects. Linked
  from the noise-filter LIMITATIONS entry, the onboard CLI help, and
  ARCHITECTURE.md (#74).

## [1.0.2] - 2026-04-27

Patch release: per-project install routing, `aelf doctor` settings
linter, and CI guardrails for release docs. Closes the v1.0.1 gap
where one machine couldn't cleanly route per-project venv hooks
alongside a global `pipx` install, and the README roadmap drifted out
of sync minutes after the wheel was on PyPI.

### Added

- `aelf doctor`: scan user-scope and project-scope Claude Code
  `settings.json` for hook + statusline commands whose program token
  doesn't resolve. Catches dangling absolute paths, bare names not on
  `$PATH`, and missing scripts under `bash /…` interpreter wrappers.
  Exits `1` on any broken finding so it can gate CI (#81).
- `staging-gate.yml` `release-docs-check` job: when a PR bumps
  `pyproject.toml` `version`, enforce `CHANGELOG.md` has a matching
  `## [X.Y.Z]` section + compare-link footnote, and that `README.md`
  has no roadmap row marking the released version as `next` /
  `planned`. No-op on non-release PRs (#80).
- `post-release-docs-issue.yml`: on `release.published`, opens a
  tracking issue `docs sweep for vX.Y.Z` with a per-doc checklist
  for the second-order docs the gate can't verify automatically
  (RELEASING.md test counts, ROADMAP.md narrative, etc.) (#80).

### Changed

- `aelf setup` no longer requires `--scope`. Default is auto-detect:
  `project` (writes `<cwd>/.claude/settings.json`) when `cwd/.venv`
  matches the active interpreter's `sys.prefix`, else `user` (writes
  `~/.claude/settings.json`). Explicit `--scope` still wins (#81).
- `aelf setup --command` defaults to an absolute path: project scope
  -> the active venv's `aelf-hook`; user scope -> the first
  `aelf-hook` on `$PATH` (typically a `pipx`-installed
  `~/.local/bin/aelf-hook`). Lets one machine route per-project venvs
  to their own hook AND fall back to a global pipx install outside
  any project, without bare-name `$PATH` collisions (#81).
- `aelf setup` now silently removes legacy dangling
  `/usr/local/bin/aelf{,-hook}` symlinks if their target no longer
  exists. Real files and live symlinks are never touched (#81).
- `aelf unsetup` defaults to basename-match cleanup (every entry
  whose program basename is `aelf-hook`), so an install written with
  the new auto-resolved absolute path can be torn down by bare
  `aelf unsetup` without specifying the path (#81).

## [1.0.1] - 2026-04-27

Patch release: power-user ergonomics + retrieval-side feedback loop.
Closes the v1.0.0 gap where hook retrievals never wrote audit rows,
adds a no-config noise filter (with a TOML escape hatch), and ships
`aelf --version` as a first-class CLI flag.

### Added

- `aelfrice.hook_search` module: `search_for_prompt(store, prompt, ...)`
  and `record_retrieval(store, beliefs, ...)`. Closes the v1.0.1
  retrieval-side feedback-loop gap: every UserPromptSubmit hook
  retrieval now writes one `feedback_history` row per returned belief,
  tagged `source='hook'` with `HOOK_RETRIEVAL_VALENCE` (0.1) positive
  valence (#70).
- `propagate: bool = True` kwarg on `aelfrice.feedback.apply_feedback`.
  Default preserves the corrective-feedback contract (positive signal
  on a contradictor pressures the contradicted user lock); pass `False`
  for non-corrective signals (hook-driven retrievals) to update the
  posterior without pressure-walking locked beliefs (#70).
- `aelf --version` flag prints `aelf <__version__>` and exits 0.
  Closes the long-standing argparse error users hit when probing the
  installed version (#71).
- `aelfrice.noise_filter` module: pure-stdlib pure-function predicate
  `is_noise(text, config)` that drops candidate paragraphs matching
  one of four well-known non-belief shapes — markdown heading blocks,
  checklist blocks, three-word fragments, or license-header
  boilerplate (seven canonical signatures). Wired into
  `scanner.scan_repo` before classification. New
  `ScanResult.skipped_noise` counter (default 0 for back-compat)
  reports how many candidates were filtered per scan (#72).
- Power-user configuration via a single `.aelfrice.toml` at the
  project root (or any ancestor); discovered automatically by
  `scan_repo` walking up from the scan root. The `[noise]` table
  exposes: `disable` (subset of `headings`, `checklists`,
  `fragments`, `license`), `min_words` (override fragment
  threshold), `exclude_words` (whole-word case-insensitive — "jso"
  drops "jso" but never "json"), and `exclude_phrases` (literal
  substring case-insensitive). No regex in the user-facing schema.
  Library callers can pass an explicit `NoiseConfig` to
  `scan_repo(..., noise_config=...)` to bypass file discovery (#72).
- `aelf upgrade` subcommand: prints the right pip-upgrade command for
  the user's install context (venv / pipx / system), includes the
  published wheel SHA-256 + PyPI URL for hash-pinned installs (#73).
- `aelf uninstall` subcommand with mutually-exclusive `--keep-db` /
  `--purge` / `--archive PATH` disposition flags. `--purge` requires
  three redundant gates (explicit flag, typed `PURGE`, final `[y/N]`).
  `--archive` encrypts the DB with AES-128-CBC + HMAC via Fernet,
  scrypt-derived key from a user password (Salt embedded; password is
  the only secret needed for recovery). Public `decrypt_archive()` API
  for later recovery (#73).
- `aelf statusline` subcommand: emits an orange-coloured one-line
  update-banner prefix snippet for Claude Code's statusbar (and any
  shell-driven status bar). Empty output when no update is pending.
  Truecolor → 256-color → basic-yellow fallback, NO_COLOR honoured (#73).
- Two-component update notifier ported from GSD's pattern: detached
  background PyPI check writes a JSON cache at
  `~/.cache/aelfrice/update_check.json`; statusline + post-command
  banners read that cache only. Cache TTL: 24h. Opt out via
  `AELF_NO_UPDATE_CHECK=1` (#73).
- `aelf setup` auto-wires the statusline alongside the hook by default
  (`--no-statusline` opts out). Composes deterministically with an
  existing `statusLine`: empty slot → install; already ours → no-op;
  simple existing → wrap with `; aelf statusline 2>/dev/null`; complex
  (shell metacharacters) → leave alone with a one-line hint (#73).
- `aelf unsetup` reverses the statusline composition surgically (#73).
- New optional extra: `pip install 'aelfrice[archive]'` adds the
  `cryptography` dep that `aelf uninstall --archive` needs (#73).
- Slash commands: `/aelf:upgrade`, `/aelf:uninstall`, `/aelf:statusline` (#73).

### Changed

- `aelfrice.hook.user_prompt_submit` now routes through
  `aelfrice.hook_search.search_for_prompt` instead of calling
  `aelfrice.retrieval.retrieve()` directly. Same non-blocking contract,
  same OPEN_TAG/CLOSE_TAG output envelope, same payload schema; the
  behavioural difference is the audit-row write per returned belief
  (#70).
- `aelf setup` output now shows two lines (hook + statusline) on a
  fresh install. `aelf unsetup` shows the matching teardown lines (#73).
- README "Install" section rewritten with explicit verification
  commands; new "Upgrade" and "Uninstall" sections (#73).
- `docs/INSTALL.md` rewritten with explicit Statusline composition
  table, full uninstall reference (including archive recovery), and
  Update notifier opt-out instructions (#73).

## [1.0.0] - 2026-04-27

First installable PyPI release. The full v1.0 surface is the cumulative
shipped contents of v0.1.0–v0.9.0rc0; this release tags it, builds the
sdist + wheel, and publishes via the GitHub Actions Trusted Publisher
workflow.

### Added

- PyPI publication: `pip install aelfrice` (`pip install "aelfrice[mcp]"`
  for the MCP server extra).

### Changed

- Trove `Development Status` classifier promoted from `4 - Beta` to
  `5 - Production/Stable`.
- README headline and tagline rewritten for a post-rebuild audience:
  install instruction front-and-centre, "Status: under active rebuild"
  warning removed, "What works today" section retitled to v1.0.0.

## [0.9.0rc0] - 2026-04-26

Benchmark-harness milestone — final gate before `v1.0.0`. The `v0.9.0-rc`
roadmap row is shipped as PEP 440 release candidate `0.9.0rc0`.

### Added

- `aelfrice.benchmark` module — deterministic 16-belief × 16-query
  synthetic harness. Public surface: `BENCHMARK_NAME`,
  `seed_corpus(store)`, `run_benchmark(store, *, aelfrice_version,
  top_k=5)`, frozen `BenchmarkReport` dataclass with `hit_at_1`,
  `hit_at_3`, `hit_at_5`, `mrr`, `p50_latency_ms`, `p99_latency_ms`
  (#50).
- `aelf bench` CLI subcommand — prints the report as a single JSON
  document. Defaults to in-memory store for full reproducibility;
  `--db PATH` for an explicit on-disk SQLite file; `--top-k N` to
  override hit@k accounting depth (#50).
- `slash_commands/bench.md` keeping the CLI ↔ slash 1:1 invariant
  green at 11 commands (#50).
- `tests/regression/test_benchmark_cli_end_to_end.py` —
  `@pytest.mark.regression` end-to-end coverage of `aelf bench`
  default invocation, `--db PATH`, and `--top-k` override (#50).

### Notes

- The harness is the **measurement instrument**, not proof of the
  central feedback claim. Retrieval ranking in the v1.0 line is
  BM25-only (`store.search_beliefs` orders by `bm25(beliefs_fts)`,
  posterior `alpha`/`beta` are not consumed), so `apply_feedback`
  does not currently move benchmark scores. A v1.x retrieval
  upgrade that consumes posterior is the precondition for using
  this harness to claim feedback drives accuracy.

## [0.8.0] - 2026-04-26

Project-metadata milestone — everything required for PyPI publish (gated
until v1.0.0) is now present and verified.

### Added

- `LICENSE` file at repo root (MIT) matching the `pyproject.toml` license
  declaration. `uv build` bundles it into both wheel and sdist (#45).
- `CHANGELOG.md` following the [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
  format with retroactive sections for v0.1.0–v0.7.0 (#46).
- `docs/INSTALL.md` — install, configure, quickstart, Claude Code wiring,
  development workflow, uninstall (#47).
- `docs/ARCHITECTURE.md` — design principles, module map, data model,
  Bayesian update path, retrieval flow with ASCII diagram, Claude Code
  integration diagram, test layers, explicit "out of scope through v1.0.0"
  list (#47).
- README `## docs` section linking to both new files (#47).
- `pyproject.toml` PyPI-ready metadata pass: sharpened `description`,
  `authors` with email, explicit `license-files = ["LICENSE"]`, ten
  `keywords`, thirteen Trove `classifiers` (Beta dev status, Python 3 /
  3.12 / 3.13, MIT, Topic / Audience / `Typing :: Typed`), and
  `[project.urls]` (Homepage, Repository, Documentation, Changelog,
  Issues) (#48).

## [0.7.0] - 2026-04-26

Claude Code wiring milestone — aelfrice retrieval can now be installed
as a `UserPromptSubmit` hook with a single `aelf setup`.

### Added

- `aelfrice.setup` module: idempotent `install_user_prompt_submit_hook` /
  `uninstall_user_prompt_submit_hook` / `default_settings_path`
  functions that mutate a Claude Code `settings.json`. Atomic on-disk
  write via sibling tempfile + `os.replace` (#39).
- `aelfrice.hook` module: `aelfrice.hook:main` reads the
  `UserPromptSubmit` JSON payload from stdin, runs aelfrice retrieval,
  and writes an `<aelfrice-memory>...</aelfrice-memory>` block to stdout.
  Non-blocking by contract — every failure mode (empty stdin, malformed
  JSON, missing/blank/wrong-type prompt field, retrieval exceptions)
  exits 0 with no stdout (#40).
- `aelf setup` and `aelf unsetup` CLI subcommands wrapping the install /
  uninstall functions, with `--scope user|project`, `--project-root`,
  `--settings-path`, `--command`, `--timeout`, `--status-message`. CLI
  surface grows from 8 to 10 commands; matching `setup.md` and
  `unsetup.md` slash commands ship in `src/aelfrice/slash_commands/`
  (#41).
- `aelf-hook = "aelfrice.hook:main"` script in `[project.scripts]`. CLI
  default `--command` switches to `aelf-hook` (#42).
- End-to-end regression test in `tests/regression/` exercising
  `aelf setup` → real subprocess spawn of the recorded hook command →
  verify retrieval output → `aelf unsetup` (#43).

### Changed

- `aelfrice.__version__` and `uv.lock` synced to `0.6.0` after v0.6.0
  shipped (#38).

## [0.6.0] - 2026-04-26

CLI / MCP / slash-commands milestone — the user-facing surface.

### Added

- `aelfrice.cli` with eight subcommands (`onboard`, `search`, `lock`,
  `locked`, `demote`, `feedback`, `stats`, `health`) and the `aelf`
  console script in `[project.scripts]`. Folds `config.py` into the CLI
  and reorganises `scoring.py` / `store.py` (#32).
- `aelfrice.mcp_server` with eight FastMCP tools mirroring the CLI
  surface; `pip install aelfrice[mcp]` extra adds the `fastmcp`
  dependency (#35).
- `src/aelfrice/slash_commands/` directory with eight markdown slash
  commands matched 1:1 to CLI subcommands; an invariant test enforces
  the correspondence (#36).
- `aelfrice.health` module with regime classifier
  (insufficient-data / supersede / ignore / balanced) backed by
  confidence, lock density, and edge density features (#31).
- Polymorphic onboard state machine in `aelfrice.classification` (#34).
- `onboard_sessions` schema + CRUD helpers in `aelfrice.store` (#33).

### Fixed

- FTS5 query special characters are now escaped in `search_beliefs`
  (#30).

## [0.5.0] - 2026-04-26

Scanner milestone — onboarding from a project directory.

### Added

- `aelfrice.scanner` package with `scan_repo` orchestrator combining
  three extractors (filesystem walk, git log, AST) with the
  classification module and the store (#28).
- Filesystem-walk extractor (#25), git-log extractor (#26), and AST
  extractor (#27).
- `aelfrice.classification` with `TYPE_PRIORS` and a regex fallback
  (#24).
- End-to-end regression test for the onboarding flow (#29).

## [0.4.0] - 2026-04-26

Feedback-loop milestone — the central `apply_feedback` endpoint.

### Added

- `apply_feedback` endpoint in `aelfrice.feedback` performing
  Beta-Bernoulli updates and writing to the `feedback_history` table
  (#19).
- `feedback_history` table + `Store` helpers (#18).
- Demotion-pressure-on-contradiction-edge: contradicting feedback
  against a locked belief now increments `demotion_pressure` (#20).
- Auto-demote locked belief when `demotion_pressure` crosses threshold
  (#21).
- No-LLM correction detector in `aelfrice.correction` (#22).
- End-to-end regression test for the feedback loop (#23).

## [0.3.0] - 2026-04-26

Retrieval milestone — two-layer L0 locked + L1 FTS5 BM25.

### Added

- `aelfrice.retrieval` module with L0 locked-belief auto-load and L1
  FTS5 BM25 keyword search. Token-budgeted output. No HRR, no BFS
  multi-hop, no entity-index in the v1.0 line (#14).
- Lock test enforcing L0-before-L1 ordering invariant in retrieval
  output (#15).
- Property test: token-budget invariant holds across budget magnitudes
  (#17).

### Changed

- Pytest configured with `pytest-timeout` (5s default) and registered
  markers; subprocess-driven tests (v0.6.0+) override per-test (#16).

## [0.2.0] - 2026-04-26

Scoring milestone — Beta-Bernoulli posterior + decay.

### Added

- `aelfrice.scoring` with posterior mean, type-specific half-life decay
  (with lock-floor short-circuit), and a basic relevance combiner; plus
  the `test_bayesian_inertia` property test (#8).
- `test_decay_required` property test confirming decay actually moves
  posteriors over time (#9).
- `test_lock_floor_sharp` property test confirming a `user`-locked
  belief does not decay (#10).

### Changed

- Pyright strict mode enabled across `src/aelfrice` and `tests` (#12).

### Fixed

- `requirement` belief half-life corrected to 720h per spec (#11).

### Documentation

- README test-count corrected (#13).

## [0.1.0] - 2026-04-26

Foundation milestone — store, models, config.

### Added

- `aelfrice.models`: `Belief` / `Edge` dataclasses plus enum-style
  module-level constants for belief / edge / lock types; `aelfrice.config`
  with type half-lives and broker-attenuation parameters (#4).
- `aelfrice.store` with SQLite WAL journaling, FTS5 full-text search,
  CRUD for beliefs and edges, broker-confidence-attenuated
  `propagate_valence`, and `demotion_pressure` read/write (#5).
- Property test: `propagate_valence` broker-attenuation invariant (#6).
- Round-trip test for `demotion_pressure` reads + writes (#7).
- Initial repo scaffold: pyproject, README, GitHub Actions workflows,
  scan configs (commit `67b4343`).

[Unreleased]: https://github.com/robotrocketscience/aelfrice/compare/v1.2.0a0...HEAD
[1.2.0a0]: https://github.com/robotrocketscience/aelfrice/compare/v1.1.0...v1.2.0a0
[1.1.0]: https://github.com/robotrocketscience/aelfrice/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/robotrocketscience/aelfrice/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/robotrocketscience/aelfrice/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/robotrocketscience/aelfrice/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.9.0rc0...v1.0.0
[0.9.0rc0]: https://github.com/robotrocketscience/aelfrice/compare/v0.8.0...v0.9.0rc0
[0.8.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robotrocketscience/aelfrice/releases/tag/v0.1.0
