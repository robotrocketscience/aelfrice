# In-code documentation audit — 2026-07-04

> **Companion to `DOCS-AUDIT-2026-07-04.md`.** That pass audited the doc-file
> surface (README, `docs/**`, slash-command docs, benchmark READMEs) and
> **deferred the in-code documentation layer** — Python docstrings, argparse
> `--help`/`description=` strings, MCP tool descriptions + schemas, and hook
> self-descriptions — to a tracked follow-up (#1075 § Deferred scope, matching
> the #959/#964 precedent). This record is that pass.

## Scope

| | Surface | Disposition |
|---|---|---|
| ✅ | Every `.py` file under `src/aelfrice/` — **103 files**, in-code docs only | audited, this record |
| — | `tests/` docstrings (361 `.py`) | recorded, **not** line-audited this pass — low user-facing value; separate follow-up |
| — | `benchmarks/` `.py` docstrings (41 `.py`) | recorded, follow-up; the benchmark **READMEs** were covered by the doc-file pass |

Pinned at HEAD `github/main`. The audit ran against `33b7ddd5`; all 103 target
files are byte-identical through `2fc04ffe` (only doc files moved in between),
so every finding holds at HEAD. This boundary (in-code layer only; `tests/` +
`benchmarks/` `.py` deferred) is stated explicitly so coverage accounting is
honest — absence of a file from a fix is a recorded disposition, not a silent skip.

## Method

17-batch multi-agent audit: one reader per file-group verified every
doc-bearing construct against the code it describes (following constants and
functions across modules). **Every CRITICAL/HIGH finding was independently
re-verified by a second adversarial pass** (default-reject on uncertainty)
before any fix was written; 1 finding was rejected there. Fixes were applied
centrally (surgical, doc-text-only) and syntax-checked.

## Coverage

- **Files audited:** 103 / 103 (100%)
- **Doc-bearing constructs checked:** 1,238
- **Findings:** 4 CRITICAL + 43 HIGH = **47 fixed in place** (this PR);
  44 MEDIUM + 4 LOW = **48 confirmed, deferred** to a follow-up fix PR;
  **1 rejected** on adversarial re-verify.

## CRITICAL — fixed in place (4)

These are in-code docs a user or agent would *act on* and be broken by (a
suggested command that errors, a reversibility recipe that won't run).

| # | File:line | Claim (doc said) | Reality (code does) | Fix |
|---|---|---|---|---|
| 1 | `cli.py`:1567 | `_cmd_wonder_gc`'s docstring states the GC age-threshold candidate rule as: "older than ``--ttl-days`` days". | The registered argparse flag for this threshold is `--gc-ttl-days` (dest=`gc_ttl_days`, default 14) — see line 6578 … | older than ``--gc-ttl-days`` days, and have unchanged Bayesian priors + |
| 2 | `hook.py`:2980 | `_format_stop_prompt` prints a pre-filled suggested command for each unlocked correction candidate: `aelf lock --statement '<belief text>'` … | `aelf lock`'s CLI parser defines `statement` as a positional argument, not a `--statement` option: `p_lock.add_argument("statement", … | lines.append(f" aelf lock {_shell_quote(b.content)}") |
| 3 | `hook.py`:4183 | `build_session_start_recap_line` returns: `f"aelfrice: {count} beliefs written since last session — \`aelf:feed -n {count}\` to review."` — … | `aelf feed`'s row-limit flag is `--limit` (`p_feed.add_argument("--limit", type=int, default=0, help="show only the last N rows (0 = all; … | f" — `aelf:feed --limit {count}` to review." |
| 4 | `clamp_ghosts.py`:34 | Documented "Reversibility" SQL to undo a clamp: `UPDATE beliefs SET alpha = alpha + (-fh.valence) WHERE id IN (SELECT belief_id FROM … | `fh` is never defined as an alias — the subquery selects from `feedback_history` (no `AS fh`), so `fh.valence` in the outer SET clause is … | UPDATE beliefs SET alpha = alpha + ( SELECT -fh.valence FROM feedback_history fh WHERE fh.belief_id = beliefs.id AND … |

## HIGH — fixed in place (43)

Materially misleading: wrong defaults, flags, versions, counts, signatures,
or "unimplemented/unused" claims that are false at HEAD.

| # | File:line | Claim (doc said) | Reality (code does) | Fix |
|---|---|---|---|---|
| 5 | `cli.py`:6347 | `aelf search --budget` help text: "output token budget (default 2000)". | The argparse default is `DEFAULT_TOKEN_BUDGET`, imported from `aelfrice.retrieval` (cli.py:130), which is defined as `DEFAULT_TOKEN_BUDGET: … | help="output token budget (default 2400)", |
| 6 | `cli.py`:6343 | `aelf search` subcommand help text: "L0 locked + L1 FTS5 retrieval" — describes only two retrieval lanes. | `_cmd_search` (cli.py:828) calls `retrieve(store, args.query, token_budget=args.budget)` with `entity_index_enabled` left at its default … | help="L0 locked + L2.5 entity-index + L1 FTS5 retrieval" |
| 7 | `store.py`:3922 | `set_retention_class`'s docstring states: "Phase-3 promotion writes through this rather than `update_belief` because `update_belief` … | `update_belief` (store.py:2122-2179) DOES carry `retention_class` in its UPDATE SET clause (`retention_class = ?,` around line 2151) and … | Phase-3 promotion writes through this rather than ``update_belief`` to avoid the broader FTS5/entity-index rewrite that … |
| 8 | `hook.py`:91 | `DEFAULT_HOOK_TOKEN_BUDGET`'s docstring: "Below the CLI default (2000) to leave headroom for the user's prompt..." | The CLI's `--budget` default is `retrieval.DEFAULT_TOKEN_BUDGET`, which is 2400, not 2000 (`DEFAULT_TOKEN_BUDGET: Final[int] = 2400`, … | Below the CLI default (2400) to leave headroom for the user's |
| 9 | `retrieval.py`:1191 | In `decode_bfs_depth_budget`'s docstring: "`v=0.0` → 1, `v=1.0` → 6, `v=0.5` → 2 (rounded from ~2.45, one hop below … | `BFS_DEFAULT_MAX_DEPTH` (imported from `bfs_multihop.DEFAULT_MAX_DEPTH`) is 2 — the exact same value the decode produces at v=0.5, not one … | `v=0.0` → 1, `v=1.0` → 6, `v=0.5` → 2 (rounded from ~2.45, matching :data:`BFS_DEFAULT_MAX_DEPTH` exactly, so a … |
| 10 | `mcp_server.py`:1782 | aelf_wonder_persist's `budget` field is documented as "BFS expansion budget (total nodes, default 24)." | tool_wonder_persist (mcp_server.py:891-976) accepts `budget: int = 24` but never reads it anywhere in the function body. The BFS call at … | description="Currently unused by wonder_persist — the effective BFS node budget is `top * 2` (see the `top` parameter). … |
| 11 | `mcp_server.py`:1487 | aelf_promote docstring: "Returns: see aelf_validate. When to_scope is supplied, the payload includes an additional 'scope' key containing … | tool_validate (shared by aelf_promote) returns the scope-change result dict directly and unwrapped — with kind one of scope.invalid / … | Returns: When to_scope resolves to scope.updated or scope.unchanged, see aelf_validate — the payload is a validate.* … |
| 12 | `mcp_server.py`:91 | _SERVER_INSTRUCTIONS (passed as the FastMCP `instructions=` at server construction, line 1086): "TIER (demote): the only … | Three tools carry `"destructiveHint": True` in their registration annotations: aelf_demote (line 1340), aelf_wonder_persist (line 1759), … | - TIER (demote): drops a lock or devalidates a belief one tier; reversible only by re-locking with fresh evidence. Not … |
| 13 | `doctor.py`:1610 | `_belief_with_type` docstring: "Return a copy of `b` (a Belief) with `type` and `origin` replaced. Uses dataclasses.replace-style … | The function does not call `dataclasses.replace` at all; it manually constructs a new `Belief(...)` passing only 12 of the dataclass's 21 … | """Return a modified copy of `b` (a Belief) with `type` and `origin` replaced. CAUTION: this is NOT a full-fidelity … |
| 14 | `doctor.py`:1141 | `_format_missing_auto_capture_section` fix text: "fix: re-run 'aelf setup' to wire transcript-ingest, commit-ingest, session-start, and … | `aelf-stop-hook` (#582) was added as the fourth default-on auto-capture hook in v3.0.0, not v2.1.0. CHANGELOG/v3.md:468,528 dates the #582 … | "fix: re-run 'aelf setup' to wire transcript-ingest, " "commit-ingest, and session-start (default-on since v2.1 / " … |
| 15 | `context_rebuilder.py`:4 | "When the harness's context window approaches its compaction threshold, the harness fires a PreCompact hook. This module is the script-side … | Per #1031, the harness rejects `additionalContext` emitted from a PreCompact hook, so `aelfrice.hook.pre_compact()` now "Always returns 0, … | When the harness compacts context, the harness fires SessionStart with source=="compact" immediately afterward (a … |
| 16 | `context_rebuilder.py`:194 | "`threshold`: PreCompact hook fires when called by the harness's harness; the harness's own threshold gating is the trigger." | The PreCompact hook never fires/emits the rebuild block in any mode — `hook.pre_compact()` always returns 0 and writes nothing to stdout … | `threshold`: the rebuild block fires on the SessionStart hook when `source == "compact"` (i.e., right after the … |
| 17 | `context_rebuilder.py`:1631 | "PreCompact hook entry point for v1.4. Reads the the harness PreCompact JSON payload from stdin, locates a transcript log..., runs the v1.4 … | This function is not wired to any console-script entry point. pyproject.toml maps `aelf-pre-compact-hook` to … | Legacy v1.4 PreCompact entry point, retained for tests/benchmarks only — NOT the live hook. Since #1031 the harness … |
| 18 | `context_rebuilder.py`:105 | "Legacy default kept for v1.2.0a0 callers (`rebuild()` and `aelf rebuild --budget`). The v1.4 hook path uses … | `aelf rebuild --budget` (src/aelfrice/cli.py `_cmd_rebuild`) calls `rebuild_v14()`, not `rebuild()`; its budget resolves via … | Legacy default kept for v1.2.0a0 callers of the plain `rebuild()` function (exercised only by the unit-test suite). … |
| 19 | `setup.py`:319 | "Mirrors install_user_prompt_submit_hook for the PreCompact event, which the harness fires before its default context compaction. The … | Since #1031, the PreCompact-wired command (`aelf-pre-compact-hook` -> `aelfrice.hook.pre_compact()`) never emits the rebuild block: it … | Mirrors install_user_prompt_submit_hook for the PreCompact event, which the harness fires before its default context … |
| 20 | `cadence.py`:526 | resolve_cadence_policy(): "Currently recognised policies: ``off``, ``p1_every_k_turns``, ``p2_ctx_threshold``. P3 (turn-density-aware) per … | `_VALID_POLICIES` (lines 90-96) already includes `POLICY_P3_VELOCITY` ("p3_velocity") and `POLICY_P3_SUBSTANTIVE` ("p3_substantive"), both … | Currently recognised policies: ``off``, ``p1_every_k_turns``, ``p2_ctx_threshold``, ``p3_velocity``, ``p3_substantive`` … |
| 21 | `cadence.py`:215 | CadenceConfig: "Six fields. ... Defaults are off / off / 15 / 0.50 / 600000 / off." (lines 215, 226) | The dataclass has nine fields, not six: `enabled`, `policy`, `k`, `ctx_threshold`, `ctx_byte_window`, `shadow_mode_enabled`, … | Nine fields. ``enabled`` gates the whole feature ... Defaults are off / off / 15 / 0.50 / 600000 / off / 3000 / 10 / … |
| 22 | `hook_search_tool.py`:49 | INJECTED_TOKEN_BUDGET's docstring: "Token budget for retrieve() — half the user-facing default." | INJECTED_TOKEN_BUDGET = 600, but the user-facing default token budget (DEFAULT_TOKEN_BUDGET, used by `aelf search --budget` and as … | """Token budget for retrieve() — one-quarter of the user-facing default (2400). Auxiliary context should not crowd out … |
| 23 | `benchmark.py`:9 | "This harness is the measurement instrument, not a proof of the central claim. It does not currently differentiate 'with-feedback' vs … | The harness calls aelfrice.retrieval.retrieve() (imported at benchmark.py:51), not store.search_beliefs. Since v1.3.0, retrieve() combines … | "This harness is the measurement instrument, not a proof of the central claim. It does not currently differentiate … |
| 24 | `benchmark.py`:64 | "Higher than retrieval's default 2000 so the budget doesn't cap the harness at fewer than top_k results when belief contents are long." | aelfrice.retrieval.DEFAULT_TOKEN_BUDGET is 2400, not 2000. | "Higher than retrieval's default 2400 so the budget doesn't cap the harness at fewer than top_k results when belief … |
| 25 | `session_ring.py`:351 | "Empty ids is a no-op and returns the current next_fire_idx (or 0 if the ring was just created)." | append_ids has no early-return / no-op branch for an empty ids list. It always reads fire_idx = data['next_fire_idx'], unconditionally sets … | "ids may be empty; the call still bumps and persists next_fire_idx by one (it is not a no-op) and returns fire_idx + 1 … |
| 26 | `models.py`:469 | "The public v1.0.0 schema does not persist sessions; this dataclass exists so academic-suite benchmark adapters ... have a stable handle to … | The public schema DOES persist sessions, and has since v1.2.0. store.py defines `CREATE TABLE IF NOT EXISTS sessions (id, started_at, … | Persisted since v1.2.0 in the `sessions` table (`MemoryStore.create_session` / `complete_session` write real rows, not … |
| 27 | `reason.py`:218 | classify()'s docstring for the `paths` (R2, #658) argument: "...emits a ``TIE`` impasse when two CONTRADICTS-forked paths share a common … | The fork-TIE check at line 296-298 calls `_compound_paths_tie(a.compound_confidence, b.compound_confidence)`, which tests … | "...emits a ``TIE`` impasse when two CONTRADICTS-forked paths share a common parent and have compound-confidence values … |
| 28 | `telemetry.py`:454 | emit_session_delta()'s docstring: "``store``: open MemoryStore to read from. When None, the caller's environment variable / git-common-dir … | The code (lines 478-480) does `from aelfrice.db_paths import _open_store` then `store = _open_store()` — a different function, from a … | "``store``: open MemoryStore to read from. When None, the caller's environment variable / git-common-dir resolution is … |
| 29 | `auto_install.py`:5 | "...so any default-on hook added in a new release (e.g. `aelf-stop-hook` shipped in v2.1) is missing for users who never re-run setup." | `aelf-stop-hook` shipped in v3.0.0, not v2.1.0. CHANGELOG/v3.md documents "Session-end Stop hook prompts to lock session corrections (#582) … | "...so any default-on hook added in a new release (e.g. `aelf-stop-hook` shipped in v3.0) is missing for users who … |
| 30 | `pre_issue_create_hook.py`:11 | Hook contract states: "stdout — human-readable additionalContext block (on PASS, may be empty)." | `run_guard` never writes to stdout on any PASS path — every PASS branch is a bare `return 0`. The only `print(...)` call in the whole … | * stdout — unused; every PASS path produces no output at all (no additionalContext envelope is emitted by this hook). |
| 31 | `replay.py`:8 | "Reachability (cheap): every belief in the canonical store has at least one ingest_log row that references its id ... This is the v2.0 … | `check_log_reachability` / `ReachabilityReport` are defined in replay.py but are never imported or called anywhere else in the codebase … | This is the v2.0 contract guarantee — no orphan beliefs. Not currently wired into any CLI entry point; call … |
| 32 | `replay.py`:88 | `FullEqualityReport`'s documented "Shape-equality contract (ratified 2026-04-29)" lists a fourth criterion: "the deterministic edge set … | `replay_full_equality` computes `matched`/`mismatched` purely from `content_hash_match`, `type_match`, and `origin_match` (three checks). … | Shape-equality contract (ratified 2026-04-29; edge-set check not yet implemented): - content_hash matches, AND - type … |
| 33 | `contradiction.py`:9 | "## Precedence (v1.2+, five classes)" followed by a numbered list of exactly 5 classes: user_stated, user_corrected, user_validated, … | The precedence system has six classes, not five: `PRECEDENCE_USER_TRANSCRIPT` ("user_transcript", value 3, added per v3.x #888) sits … | ## Precedence (v3.x #888, six classes) 1. **`user_stated`** — ... 2. **`user_corrected`** — ... 3. **`user_validated`** … |
| 34 | `value_compare.py`:16 | Numeric slots: `(key_token, value, unit?)`. The key is the nearest alphabetic token preceding the number; the value is parsed as float; the … | `NumericSlot` (lines 151-166) has exactly two fields, `key: str` and `value: float` -- there is no unit field. `_NUMERIC_RE` (lines 62-74) … | Numeric slots: `(key_token, value)`. The key is the nearest alphabetic token preceding the number; the value is parsed … |
| 35 | `value_compare.py`:275 | Numeric conflict: same `(key, unit)` with values outside the relative-tolerance band. Unit-mismatch is silent -- different units mean … | `find_conflicts` groups numeric slots by `key` alone (`a_num_by_key: dict[str, list[NumericSlot]]`, keyed on `s.key`, lines 286-291) and … | Numeric conflict: same `key` with values outside the relative-tolerance band. Units are not extracted or compared -- … |
| 36 | `derivation.py`:157 | Shared scheme with `ingest._belief_id` and `scanner._derive_belief_id` so the same (text, source) pair always resolves to the same id … | Neither function exists at HEAD. `ingest.py` imports `run_worker` from `derivation_worker` and never defines a local `_belief_id`; … | Shared scheme with the derivation-worker path used by `ingest.py` and `scanner.py` (both route belief creation through … |
| 37 | `derivation.py`:168 | Matches `mcp_server._lock_id_for` and `cli._lock_id_for`. | Neither function exists at HEAD. `mcp_server.py` calls `derive()` directly and reads `lock_bid = derived.belief.id` (mcp_server.py:382); … | Matches the id `derive()` itself assigns on the lock/remember path; `mcp_server.py` and `cli.py` call `derive()` … |
| 38 | `derivation.py`:177 | Matches `triple_extractor._belief_id_for_phrase`. (Same broken cross-reference recurs at line 195 as `triple_extractor._content_hash` and … | Neither `_belief_id_for_phrase` nor `_content_hash` exists in `triple_extractor.py` at HEAD (confirmed via `git grep` across src/aelfrice). … | Line 177: "Deterministic id for triple-extracted noun-phrase beliefs, produced when `triple_extractor.ingest_triples` … |
| 39 | `meta_beliefs.py`:40 | `update_meta_belief` calls for a class not in the subscription list are recorded as no-op audit events rather than silently dropped. | `MemoryStore.update_meta_belief` (src/aelfrice/store.py:4189-4221) returns False and writes nothing when `signal_class not in weights` … | calls for a class not in the subscription list are silently dropped (no audit row is written in this commit; audit … |
| 40 | `entity_extractor.py`:207 | "Pure regex over `text`. No side effects, no I/O, no exceptions (an empty / non-string-shaped input returns [])." | Only falsy input (None, "", 0, empty list, etc.) is caught by the `if not text or max_entities <= 0: return []` guard (line 216). A truthy … | Pure regex over `text`. No side effects, no I/O; falsy input (None, "", 0, []) returns []. Non-string, non-empty input … |
| 41 | `hook_commit_ingest.py`:6 | "Closes the v1.0 limitation that the belief graph only grows on explicit `aelf onboard` / `aelf remember` calls." | There is no `aelf remember` subcommand registered anywhere in cli.py (grep for sub.add_parser / "remember" finds none). The only … | Closes the v1.0 limitation that the belief graph only grows on explicit `aelf onboard` / `aelf lock` calls. |
| 42 | `clamp_ghosts.py`:81 | "`matched` and `scanned` are equal under the current implementation (no per-row skip logic beyond the SQL filter). The split is kept for … | ClampResult has no `scanned` field at all — its fields (lines 85-91) are matched, clamped, skipped, dry_run, threshold_alpha, target_alpha, … | The count fields satisfy `matched == clamped + skipped`. `matched` is the number of rows selected by the SQL filter, … |
| 43 | `classification_core.py`:111 | "pending_classification: ... Always True in v1.0; the polymorphic-host-handshake path that flips this to False lands in v0.6.0." | pyproject.toml pins the package at v3.8.0. All six return sites in classify_sentence (lines 202, 213, 233, 248, 259, 269) still set … | pending_classification: ... Always True as of v3.8.0; no code path currently flips it to False — the … |
| 44 | `wonder/evaluator.py`:23 | "Output is a single BakeoffResult dataclass with one StrategyMetrics per strategy plus the cross-strategy Jaccard matrix and the … | BakeoffResult (defined here, exported in __all__) is never instantiated anywhere in src/ or tests/ (grep for BakeoffResult( returns … | BakeoffResult is an unused convenience dataclass — defined and exported here but never instantiated in src/ or tests/. … |
| 45 | `wonder/evaluator.py`:167 | "3. Single-strategy ship if exactly one strategy clears the floor with retrieval-per-cost within 25% of best, and the others are not … | cost_window (the "retrieval-per-cost within 25% of best" set, computed at lines 188-191) never gates a return value: len(clearing) == 1 … | 3. Single-strategy ship — the default outcome once at least one strategy clears the floor and the top two are not … |
| 46 | `wonder/result.py`:50 | `phantoms_created` (on `WonderResult`) is "Sourced from the `inserted` count of the ingest path when `--persist-docs FILE` is used." | `_cmd_wonder_persist_docs` (cli.py, the `--persist-docs FILE` handler) never constructs a `WonderResult` at all — it prints `inserted=N … | phantoms_created: Count of phantoms created (axes mode only); ``0`` in graph-walk mode. Currently always ``0`` — the … |
| 47 | `wonder/__init__.py`:8 | The harness is a research surface: nothing here writes to a live ``Store`` outside the bake-off. The chosen-strategy production wiring is a … | The `aelfrice.wonder` package now contains `wonder/lifecycle.py`, which defines `wonder_ingest()` (inserts `Belief` rows with … | The generation strategies in this module (`random_walk`, `triangle_closure`, `span_topic_sampling`) are research-only … |

## MEDIUM / LOW — confirmed, deferred to follow-up fix PR (48)

Stale-but-harmless or typo-level in-code doc drift. Recorded here for 100%
coverage; a follow-up fix PR lands these so this PR stays reviewable.

| # | File:line | Drift |
|---|---|---|
| 1 | `cli.py`:7701 | `aelf setup --claude-memory-mirror` help text: "wire the PostToolUse:Write\|Edit hook that mirrors the upstream auto-memory tool memory writes into the belief graph (#985)." |
| 2 | `store.py`:1086 | `_bump_belief_version`'s docstring states: "Idempotent INSERT OR REPLACE on the composite PK keeps the row count bounded by `n_beliefs * n_scopes`." |
| 3 | `store.py`:2367 | `_write_belief_entities`'s docstring states: "Lazy import of `aelfrice.entity_extractor` to keep the import graph acyclic (entity_extractor only imports triple_extractor, not store)..." |
| 4 | `store.py`:3591 | `record_touch`'s docstring states: "Raises `ValueError` on empty `belief_id` / `session_id` or non-positive `fire_idx` / `event_kind`." |
| 5 | `store.py`:3022 | `record_corroboration`'s docstring states the idempotency key as: "#1020: idempotent per `(belief_id, session_id, source_path_hash)` via `INSERT OR IGNORE` against the … |
| 6 | `hook.py`:1455 | `_record_injection_events`'s docstring: "Fires from the UPS hook after retrieval has decided which beliefs will appear in the rendered ``<aelfrice-rebuild>`` block." |
| 7 | `hook.py`:2617 | `_read_recent_for_pre_compact`'s resolution-order docstring: the canonical `turns.jsonl` is "written by the per-turn UserPromptSubmit/Stop hooks once transcript_ingest ships", and the … |
| 8 | `retrieval.py`:3658 | `RetrievalCache` class docstring: "Cache key includes the entity-index flag (v1.3.0 default-on), the BFS flag (v1.3.0 default-off), and `posterior_weight` (v1.3.0 default 0.5, rounded to … |
| 9 | `mcp_server.py`:15 | "Tool surface (all under the `aelf:` namespace at the host):" followed by a list using colon-separated names (aelf:onboard, aelf:search, aelf:lock, aelf:locked, aelf:demote, aelf:validate, … |
| 10 | `mcp_server.py`:1621 | aelf_stats docstring's documented return shape: {"kind": "stats.snapshot", "beliefs": int, "edges": int, "threads": int, "locked": int, "feedback_events": int, "onboard_sessions_total": int} |
| 11 | `doctor.py`:354 | `DoctorReport.missing_auto_capture_hooks` field comment: "Default-on auto-capture hook basenames (since v2.1, #529) that are absent from every scanned settings.json." |
| 12 | `doctor.py`:609 | `_scan_orphan_slash_commands` docstring: "Ignores `aelf-*` files that wrap meta commands (none ship today but reserved)." |
| 13 | `context_rebuilder.py`:608 | "Wrap a rebuild block in the the harness PreCompact JSON envelope. The harness expects `additionalContext` under `hookSpecificOutput`." |
| 14 | `context_rebuilder.py`:292 | "Legacy v1.2.0a0 entry point. Preserved for backwards compatibility with the `aelf rebuild` CLI and the eval harness." |
| 15 | `context_rebuilder.py`:861 | "Legacy v1.2.0a0 path; preserved so the existing `aelf rebuild` CLI behaves byte-identical to the alpha." |
| 16 | `cadence.py`:10 | "This module implements two of the policies pre-registered by #749: * P1 every-K-turns ... * P2 ctx-threshold + phase-boundary ..." |
| 17 | `scanner.py`:110 | LLMRouter docstring: "Production callers pass `aelfrice.llm_classifier.ScannerRouter` (defined in cli.py to keep the SDK lazily-imported)" |
| 18 | `hook_search_tool.py`:785 | _db_path_accepts_cwd docstring: "Best-effort detection: does aelfrice.cli.db_path() accept a cwd kw?" |
| 19 | `scanner.py`:172 | scan_repo docstring: "Existing beliefs are detected via `MemoryStore.get_belief(id)` and skipped." |
| 20 | `ingest.py`:124 | "Idempotent on (source, sentence) pairs: re-ingesting the same turn triggers `INSERT OR IGNORE` semantics in the store (belief id derives from the sha256 of source + sentence)." |
| 21 | `bm25.py`:419 | Documented Format:: byte layout for serialize()/deserialize() lists, in order: magic, version, anchor_weight, k1/b/avgdl, n_docs/n_terms, belief_ids, vocabulary terms, dl, idf, tf.indptr, tf.indices, … |
| 22 | `session_ring.py`:777 | "Read-only. Surface for aelf doctor telemetry — callers should treat the dict as opaque and read ring length, ring_max, and evicted_total for display." |
| 23 | `ingest.py`:432 | "Within a session, consecutive turns are linked with DERIVED_FROM edges from turn N+1's last belief back to turn N's last belief, anchor_text set to the prior turn's text (truncated to … |
| 24 | `reason.py`:210 | classify()'s docstring: "Determinism: impasse list order is fixed by construction order in this function (``NO_CHANGE`` first, then ``CONSTRAINT_FAILURE``, then ``TIE``, then ``GAP``); within each … |
| 25 | `models.py`:3 | "Load-bearing fields only. 4 belief types, 5 edge types, 2 lock levels." |
| 26 | `models.py`:372 | Belief class docstring: "`lock_tier` (v3.7 #1016-B) is the layered-lock tier — 'frozen' (always injected verbatim) or 'reference' (bounded, manifest-only)." |
| 27 | `pre_issue_create_hook.py`:38 | "Body-file safety" section implies body content is normally used for scoring and is only skipped/fell back to title-only when the `--body-file` path is unsafe (under `~/.claude/`): "the body read is … |
| 28 | `project_warm.py`:39 | "Override via `~/.aelfrice/config.json`'s `project_warm.deny_globs` key (list of strings; `~` expanded at load time)." |
| 29 | `pre_issue_create_hook.py`:63 | `TOP_K_QUERY_TOKENS` docstring: "Number of highest-value query tokens forwarded to gh/git searches." |
| 30 | `derivation.py`:209 | Dispatch rules (in evaluation order): 1. Lock/remember paths... 2. Triple-extraction path... 3. All other paths (filesystem, python_ast, feedback_loop_synthesis, legacy_unknown): run … |
| 31 | `triple_extractor.py`:4 | Reusable by every v1.x ingest caller that has prose at hand: the v1.2.0 commit-ingest hook, transcript-ingest, manual `aelf remember` calls, and the v1.3.0 entity-index path. |
| 32 | `claude_memory.py`:279 | Default: False -- the mirror is opt-in, consistent with the narrow-surface PHILOSOPHY (#605) and the opt-in default posture ratified for new behaviours (ADR 0003 decision 4, #606). |
| 33 | `review.py`:241 | remove → soft_delete_belief (existing primitive; writes audit via feedback_history) |
| 34 | `review.py`:17 | MalformedRowError — raised by parse_review_file when a row's ID is missing |
| 35 | `expansion_gate.py`:4 | Cheap deterministic prompt-shape gate that decides whether to run the expensive expansion layers (BFS multi-hop, HRR structural) for a given query. |
| 36 | `expansion_gate.py`:26 | no file paths (``src/...``, ``tests/...``, ``docs/...``, ``benchmarks/...``) |
| 37 | `derivation_worker.py`:35 | Belief id is sha256(source + text)[:16] |
| 38 | `canvas_export.py`:23 | Nodes: ``type="text"``, ``text = belief.content[:NODE_TEXT_MAX]``. |
| 39 | `hook_tail.py`:137 | Header line: `<HH:MM:SS> <hook> <tokens>tok <latency>ms L0×N L1×M`. |
| 40 | `cadence_score.py`:7 | "...and a 2x2 agreement matrix between the two currently-implemented policies (P1 vs P2)." |
| 41 | `entity_extractor.py`:178 | "span_start / span_end are byte offsets into the source text" |
| 42 | `auditor.py`:3 | "Three mechanical checks in v1.1.0. Each fires severity='fail' when the invariant is violated..." |
| 43 | `auditor.py`:28 | "CHECK_ORPHAN_EDGES is kept as a deprecated alias for the constant (same value as the new CHECK_ORPHAN_THREADS) for v1.0 importer compatibility; remove in v1.2.0." |
| 44 | `calibration_metrics.py`:3 | "Pure-stdlib leaf module. Three functions: precision_at_k, roc_auc, spearman_rho." |
| 45 | `wonder/strategies.py`:75 | "Seed selection: any belief whose uncertainty_score(alpha, beta) exceeds uncertainty_floor is eligible." |
| 46 | `db_paths.py`:14 | "Imports here must stay leaf-side: `aelfrice.store` (which itself only imports `models` + `ulid`) is the only intra-package dep." |
| 47 | `hook_search.py`:45 | "`source` value written into feedback_history for every hook-driven retrieval row. LIMITATIONS.md commits this string publicly; downstream analysis ... depends on it." |
| 48 | `doc_linker_types.py`:7 | Mirrors the leaf-module pattern used in #500 for `np_pattern` / `classification_core` / `db_paths`. |

## Rejected on adversarial re-verify (1)

- `classification_core.py`:187 — reader claimed the "v0.6.0 host-handshake
  ship promise never materialized" was a defect; the adversarial pass found the
  docstring's description is accurate at HEAD. Not filed. (A *separate*, real
  drift in the same file's class docstring at line 111 — the absurd "lands in
  v0.6.0" forward-reference while the package is at v3.8.0 — was confirmed and
  is fixed above.)

## Code-side drift resolved

- **#1077 (finding C1 from the doc-file pass):** `cli.py` `aelf search --budget`
  help said `default 2000`; the real default is `DEFAULT_TOKEN_BUDGET = 2400`.
  Fixed in this PR (plus the same stale `2000` in `hook.py`, `benchmark.py`,
  and `hook_search_tool.py`'s "half the default" → "one-quarter"). **Closes #1077.**

## Traps correctly avoided (checked, not filed)

- The BM25F `-raw` sign at the scorer call site is **correct** (scorers negate
  internally); #978 covers the real gap. Readers were instructed not to re-file
  it, and did not.

## Verification limit — externally-sourced numbers

Per #1075's disposition rules, lab/bench-sourced numeric claims that cannot be
re-derived from this repo were checked only for internal consistency and left
un-flagged. Examples surfaced by the readers:

- b0: Audited every argparse help=/description= string (all ~40 subparsers, ~145 add_argument calls) and every module/class/function docstring in src/aelfrice/cli.py containing a checkable fact (env var, default, file path, version gate, or …
- b2: Audited the full 4208-line src/aelfrice/hook.py at github/main HEAD (fetched via `git fetch github main`), cross-checking every constant/env-var/path/default claim against its defining module (retrieval.py, cli.py, setup.py, …
- b3: Audited the entire 3751-line src/aelfrice/retrieval.py against github/main HEAD. This module has no argparse help=/description= strings, no MCP tool descriptions, and no hook self-description strings — it is a pure retrieval-logic …
- b4: All claims in this file are checkable against in-repo code (no external/lab-sourced numbers). One finding (module-docstring 'aelf:' colon namespace) required confirming FastMCP's default tool-naming behavior; verified directly against …
- b5: Covered the entire file (1847 lines of github/main content) across module docstring, all dataclass docstrings/field comments, and all function docstrings. Cross-checked constants against their defining modules: HRR dim default (hrr.py …
- b6: Root cause behind most findings: PR merging #1031 (commit 6e65d365, 'fix(hook): inject rebuild block at SessionStart(source=compact), not PreCompact') rewired the live delivery of the context-rebuilder block from the PreCompact hook to …
- b7: Scope: audited src/aelfrice/cadence.py (1261 lines), src/aelfrice/llm_classifier.py (1190 lines), src/aelfrice/lifecycle.py (877 lines) at github/main HEAD (fetched via `git fetch github main`; all content read via `git show …
- b8: Coverage: read the full HEAD content of all 4 files via `git show github/main:<path>` and checked every argparse/MCP/hook-description/docstring construct that stated a checkable fact (default value, env var, file path, function …

## Deferred — discretion-sensitive vocabulary (1)

The repo's `pre-push` discretion policy (`BANNED_VOCAB`) bars **newly added**
lines carrying certain harness/model literals; pre-existing occurrences are
grandfathered. These confirmed finding(s) sit in a docstring so entangled with
those literals that any changed line re-adds a grandfathered token. Rather than
add banned vocab, they are deferred to a follow-up that rewrites the docstring
in neutral terms:

- `llm_classifier.py`:1 (HIGH) — """the LLM classifier onboard classifier (opt-in, v1.3.0). ... Default-OFF in every code path." (lines 1 and 4)

## Still open for the v4.0.0 gate (this record)

- [ ] MEDIUM/LOW in-code fixes (48) — follow-up fix PR.
- [ ] Discretion-sensitive docstring(s) (1) — neutral-vocab rewrite.
- [ ] `tests/` + `benchmarks/` `.py` docstring layer — follow-up pass.
