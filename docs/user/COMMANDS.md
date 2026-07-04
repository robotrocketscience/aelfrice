# Commands

Multi-subcommand CLI; full set listed via `aelf --help --advanced`. The retrieval/feedback ones are also exposed as MCP tools (see [MCP](MCP.md)) and slash commands (see [SLASH_COMMANDS](SLASH_COMMANDS.md)). Lifecycle commands (`setup`, `doctor`, `migrate`, `upgrade`, `uninstall`, etc.) are CLI-only.

```
aelf <subcommand> [args] [options]
aelf --help
aelf --version
```

DB resolves from `$AELFRICE_DB`, then `<git-common-dir>/aelfrice/memory.db` when `cwd` is in a git work-tree, then `~/.aelfrice/memory.db` as the non-git fallback.

## Memory operations

| Command | What it does |
|---|---|
| `onboard <path>` | Walk filesystem, git log, Python AST. Classify candidates, insert non-duplicates. Tunable via `.aelfrice.toml` — see [CONFIG](CONFIG.md). v1.5.1+ default-on host-driven LLM classification (#238): `[onboard.llm].enabled = true` by default, routed through the host model's Task tool — no API key required. Soft-fallback to the deterministic regex classifier when no host Task tool is reachable. Direct-API path: `--llm-classify` (requires the API-key install extra; retains legacy fail-fast install-hint). Other flags: `--emit-candidates` / `--accept-classifications` (low-level handshake driving the host-driven classifier; documented in [llm_classifier.md](../design/llm_classifier.md)), `--dry-run` (preview candidates without inserting), `--revoke-consent` (remove the stored consent sentinel and exit), `--check` (read-only pre-scan: report already-present vs to-classify counts and exit, #761), `--force` (bypass the rejection ledger so previously-rejected candidates are re-emitted, #801). |
| `search <query> [--budget N]` | L0 locked + L2.5 entity-index (v1.3+) + L1 FTS5 BM25, token-budgeted (default 2,400 at v1.3+, 2,000 prior). L2.5 default-on; disable via `[retrieval] entity_index_enabled = false` in `.aelfrice.toml` or `AELFRICE_ENTITY_INDEX=0` in the env. Distinguishes "store empty" from "no match". |
| `lock <statement>` | Insert at `(α, β) = (9.0, 0.5)` with `lock_level=user`. Idempotent — re-lock upgrades existing. |
| `locked` | List locks. |
| `core [--json] [--limit N] [--min-corroboration N] [--min-posterior FLOAT] [--min-alpha-beta N] [--locked-only] [--no-locked]` | (v2.0+, #439) Surface load-bearing beliefs: locked ∪ {corroboration ≥ 2} ∪ {posterior ≥ 2/3 with α+β ≥ 4}. Read-only. |
| `unlock <belief_id>` | Drop a user-lock without changing origin. Idempotent. Writes a `lock:unlock` audit row. |
| `delete <belief_id> [--yes] [--force]` | Hard-delete a belief: removes the belief row, FTS entry, edges (src and dst), and entity index rows. Writes one audit row to `feedback_history` (valence=-1.0, source=`user_deleted`) before the cascade so the forensic record survives. Confirmation prompt by default — prints belief content and requires the user to type the first 8 characters of the id; `--yes` skips the prompt. Refuses locked (`lock_level=user`) beliefs without `--force`; with `--force` the audit source becomes `user_deleted_force`. Exit 0 on success; exit 1 on not-found, locked-without-force, or prompt mismatch. |
| `demote <belief_id> [--to-scope SCOPE]` | Drop a user lock (one tier per call: lock first, then user_validated). Delegates to `unlock` for the lock-drop path so an audit row is always written. v3.0+ `--to-scope` ([#689](https://github.com/robotrocketscience/aelfrice/issues/689)) flips federation visibility back to `project` (or any valid scope); when `--to-scope` is the only flag supplied, tier demotion is skipped — scope-only. Writes a `scope:<old>-><new>` audit row. |
| `promote <belief_id> [--source user_validated] [--to-scope SCOPE]` | Promote an `agent_inferred` belief to `user_validated`. Alias of `validate` for the promotion semantics; unlike `validate`, it also accepts `--to-scope`. v3.0+ `--to-scope` ([#689](https://github.com/robotrocketscience/aelfrice/issues/689)) flips the belief's federation visibility (`project` / `global` / `shared:<name>`) in the same call; writes a zero-valence `scope:<old>-><new>` audit row. The scope flip works on `agent_inferred` (combined with the origin flip) and on already-`user_validated` rows (scope-only — origin path is idempotent). Foreign IDs raise `ForeignBeliefError`. Speculative beliefs (`origin=speculative`) promote to `user_validated` per Surface A ([#550](https://github.com/robotrocketscience/aelfrice/issues/550)). |
| `validate <belief_id> [--source user_validated]` | Promote an `agent_inferred` belief to a user-validated origin (v1.2+). |
| `confirm <belief_id> [--source S] [--note TEXT]` | Explicit user affirmation: α += 1.0 with source `user_confirmed` (default). Writes to `feedback_history`. Note is printed on success but not persisted. Distinct from `lock` (no freeze) and from implicit retrieval feedback (`used` signal). MCP sibling: `aelf_confirm` (#390). |
| `feedback <belief_id> <used\|harmful> [--source S]` | `used` ⇒ α += 1; `harmful` ⇒ β += 1. Contradiction-driven lock demotion was removed under [#814](https://github.com/robotrocketscience/aelfrice/issues/814); lock correction now goes through `aelf lock` overwriting per PHILOSOPHY [#605](https://github.com/robotrocketscience/aelfrice/issues/605). |
| `resolve` | Sweep unresolved `CONTRADICTS` threads. Picks a winner per precedence (`user_stated > user_corrected > document_recent`) and inserts a `SUPERSEDES` thread. Idempotent. |
| `reason <query> [--seed-id ID]... [--k N] [--depth N] [--budget N] [--fanout N] [--json]` | (v2.0+, [#389](https://github.com/robotrocketscience/aelfrice/issues/389)) Surface a reasoning chain over the belief graph. Default seeds: top-3 BM25 hits over `<query>`; `--seed-id` overrides (repeatable). Walks `expand_bfs` with terminal-tight defaults (depth=2, budget=10, fanout=8). Default output is an indented hop tree with edge-type breadcrumbs and path-scores; `--json` for tooling. Read-only over the graph. **v3.0+ ([#645](https://github.com/robotrocketscience/aelfrice/issues/645), [#658](https://github.com/robotrocketscience/aelfrice/issues/658), [#690](https://github.com/robotrocketscience/aelfrice/issues/690), [#713](https://github.com/robotrocketscience/aelfrice/issues/713)):** `--json` payload gains `verdict` + `impasses` (Verdict/ImpasseKind classifiers), `paths` (ConsequencePath fork on `CONTRADICTS`), `dispatch` (one row per impasse → Verifier / Gap-filler / Fork-resolver), and `suggested_updates` (belief-id direction notes for `feedback` close-the-loop). Peer-aware: hops and seeds in foreign scopes are annotated `[scope:<name>]` in human output and carry `owning_scope` in JSON; `--seed-id` resolves through `find_foreign_owner` if the id is foreign. |
| `review [--generate] [--apply] [--out PATH] [--json]` | (v3.5+, [#936](https://github.com/robotrocketscience/aelfrice/issues/936)) Weekly belief-review checkpoint. **`--generate`** (default when no flag): writes `.aelfrice/review.md` with up to 10 oldest-unconfirmed beliefs as a markdown checkbox list sorted by `last_confirmed_at NULLS FIRST`, then `last_retrieved_at NULLS FIRST`, then `created_at ASC`. **`--apply`**: parses verdicts from the file (`keep` → updates `last_confirmed_at`; `remove` → soft-delete with audit row; `lock` → promotes to `lock_level=user`; blank → skip). Rejects any row with two or more checked boxes (fail-closed — no partial apply). `--out PATH` overrides the default `.aelfrice/review.md` for both modes. `--json` (apply only) emits `ApplyReport` as JSON. Shares its NULLS-FIRST age/recency ordering with `aelf stale` (#933), which shipped alongside it in v3.5.0. |
| `wonder [QUERY] [--axes QUERY] [--axes-agents N] [--seed ID] [--top N] [--emit-phantoms] [--persist] [--persist-docs FILE] [--gc] [--gc-dry-run] [--gc-ttl-days N] [--json]` | (v2.0+, [#389](https://github.com/robotrocketscience/aelfrice/issues/389)) Two modes selected by argument shape. **No-arg graph-walk mode (v2.0)**: surface consolidation candidates with `--seed` (default: highest-degree non-locked belief, id-asc tiebreak). Combines BFS path-score with `wonder_consolidation.score` token-overlap to rank candidates; suggested actions in `{merge, supersede, contradict, relate}`. **Positional `QUERY` / `--axes QUERY` axes mode (v3.0+, [#542](https://github.com/robotrocketscience/aelfrice/issues/542) / [#551](https://github.com/robotrocketscience/aelfrice/issues/551) / [#552](https://github.com/robotrocketscience/aelfrice/issues/552) / [#645](https://github.com/robotrocketscience/aelfrice/issues/645))**: runs `analyze_gaps()` + `generate_research_axes()` and emits 2-6 orthogonal research axes for skill-layer subagent dispatch. Recognises agent-count shorthand `quick N-agent` / `deep N-agent` / `N-agent`. **`--persist`** writes BFS phantom candidates through `wonder_ingest` (mutually exclusive with `--axes` and the positional QUERY → exit 2). **`--persist-docs FILE`** ingests subagent research documents (JSONL: `{axis_name, content, anchor_ids}`) via `wonder_ingest` per #644 v2 dedup key. **`--gc [--gc-dry-run] [--gc-ttl-days N]`** runs `wonder_gc` (v3.0 flag form replaces the v2.1 `wonder gc` sub-subcommand); manual by default — opt into automatic per-session GC on the SessionStart hook with `AELFRICE_WONDER_AUTOGC=1` (default-off, [#980](https://github.com/robotrocketscience/aelfrice/issues/980); `AELFRICE_WONDER_AUTOGC_TTL_DAYS=<days>` overrides the 14-day threshold; sweeps that delete anything emit a `wonder.gc` feed row + an `aelf-hook:` stderr notice). `--emit-phantoms` prints `Phantom` JSON objects for offline review. `--json` emits the `WonderResult` dataclass ([#656](https://github.com/robotrocketscience/aelfrice/issues/656)) with `mode` (`graph_walk` / `axes`), clamped `coverage ∈ [0.0, 1.0]` ([#667](https://github.com/robotrocketscience/aelfrice/issues/667)), `phantoms_created`, etc. |
| `stale [--older-than N] [--cold-for N] [--locked-only] [--limit N] [--json]` | (v3.5+, [#933](https://github.com/robotrocketscience/aelfrice/issues/933)) List beliefs with `created_at` older than N days (default 30) AND `last_retrieved_at` NULL or older than M days (default 14). No decay model — plain windows. `--limit` default 20. |
| `feed [--since DUR] [--limit N] [--json]` | (v3.5+, [#931](https://github.com/robotrocketscience/aelfrice/issues/931)) Read the belief-write event log at `<git-common-dir>/aelfrice/feed.jsonl` (lock / onboard / wonder-promote / feedback rows). `--since` takes durations like `5m` / `2h` / `1d`; `--limit 0` (default) shows all. |
| `speculative [--origin TAG] [--limit N] [--json]` | (v3.5+, [#937](https://github.com/robotrocketscience/aelfrice/issues/937)) List non-user-locked active beliefs sorted by α descending — the agent-inferred / ingested / wonder-generated layer. `--limit` default 20. |
| `audit-claude-memory [--project PATH] [--json]` | (v3.5+, [#935](https://github.com/robotrocketscience/aelfrice/issues/935)) Read-only cross-store dedup audit between locked aelfrice beliefs and the host's auto-memory `MEMORY.md`. Reports potential duplicates, contradictions, and store-exclusive entries. |
| `export-obsidian <vault> [--scope all\|recent\|query] [--query TEXT] [--neighborhood-hops N] [--k N] [--max-notes N] [--force]` | (v3.0+) One-way DB→vault export: one Markdown note per belief under `<vault>/aelfrice/`, typed edges in YAML front-matter (Dataview-queryable) plus body wikilinks. Default cap 500 notes, hard ceiling 5000 without `--force`. |

## Diagnostics

| Command | What it does |
|---|---|
| `stats` | Belief / thread / lock / feedback counts. |
| `health [--json]` | Structural auditor: orphan threads, FTS5 sync, locked contradictions, corpus volume. Includes a per-edge-type count breakdown (sorted by count desc, then alphabetically); empty store prints `no edges yet`. `--json` emits `{"audit": {...}, "features": {"edges_by_type": {...}}}`. Exits 1 on structural failure; corpus-volume warnings are informational. |
| `status` | Alias for `stats` — belief / thread / lock / feedback counts. v3.0+ adds an `hrr.persist_state` summary line — see *HRR persistence reporter* below. The structural auditor is `aelf doctor graph` (the `health` verb is hidden). |
| `graph [anchor] [--seed-id ID]... [--hops N] [--budget N] [--fanout N] [--edge-types CSV] [--format dot\|json] [--preview-chars N] [--out PATH]` | (v3.3+, [#629](https://github.com/robotrocketscience/aelfrice/issues/629)) Emit a BFS subgraph anchored on a belief. `anchor` is tried as a literal belief id first, falling back to top BM25 hit; `--seed-id` (repeatable) overrides. Color-coded edges (all 11 edge types; legend in `--help`), nodes shaded by lock status and posterior bucket. `--format dot` (default) for Graphviz; `--format json` for tooling. Defaults `--hops 2`, `--preview-chars 80`. |
| `scope-out <pattern> [--list] [--clear]` | (v3.3+, [#856](https://github.com/robotrocketscience/aelfrice/issues/856)) Session-scoped retrieval exclusion. Positional `pattern` adds a case-insensitive literal substring; beliefs whose content contains it (locked included) are dropped from hook injection for the rest of the active session, and the list auto-clears when a new session starts. `--list` shows active patterns; `--clear` empties them. (Federation visibility is `promote`/`demote --to-scope`, not this.) |
| `regime` | The v1.0 regime classifier output (`supersede` / `ignore` / `mixed` / `insufficient_data`). Informational; always exits 0. |
| `doctor` | Verify hook + statusline commands resolve. Inspects `bash <script>` wrappers, flags `2>/dev/null \|\| true` patterns. Surfaces empty-store warning. Exits 1 on broken hooks. v1.6+ flags: `--gc-orphan-feedback` (delete `feedback_history` rows whose `belief_id` no longer exists, #223); `--promote-retention` (one-shot reclassification pass over low-prior beliefs based on accumulated retrieval / corroboration evidence, #290 phase-3). v3.0+ adds three HRR persistence rows (`hrr.persist_enabled`, `hrr.on_disk_bytes`, `hrr.last_build_seconds`) — see *HRR persistence reporter* below. |
| `bench [--top-k N]` | Run the deterministic 16-belief × 16-query benchmark. Prints a JSON `BenchmarkReport`. |
| `bench all --out PATH [--canonical] [--adapters CSV] [--smoke]` | (v2.0+, #437) Reproducibility harness — subprocess each academic-suite adapter (mab, locomo, longmemeval, structmemeval, amabench) at the canonical headline cut and merge into one schema-v2 JSON. `--canonical` asserts the run matches `CANONICAL_INVOCATIONS` (full benchmarks per the 2026-05-06 ratification) and refuses if the cut differs. `--smoke` runs the small SMOKE_INVOCATIONS subset. `--adapters` filters; combined with `--canonical` this refuses (cut mismatch). Returns 0 ok / 1 any error / 2 any skipped_data_missing. |
| `tail [--since DUR] [--filter key=value]... [--no-blob] [--no-follow]` | (v1.6+) Live-tail the per-turn hook audit log at `<git-common-dir>/aelfrice/hook_audit.jsonl`. Per fire: a header line (time, hook, tokens, latency, `L0×N L1×M`) plus one indented snippet line per injected belief. `--no-blob` suppresses snippet bodies; `--no-follow` dumps and exits; `--filter` matches fields like `hook=user_prompt_submit` / `lane=L0` (repeatable). See [hook-injection-audit.md](../design/hook-injection-audit.md). |
| `sweep-feedback` | (v1.6+) Run the deferred-feedback sweeper once (#191). Observes which retrieved beliefs are referenced by the host's continuation and emits implicit posterior-feedback events into `feedback_history`. Default-on background path; this verb forces a one-shot pass. |
| `eval [--corpus PATH]` | Run the relevance-calibration harness (P@K / ROC-AUC / Spearman ρ) on a synthetic corpus. Prints the calibration block; exit 0 on success. |
| `clamp-ghosts [--threshold F] [--target F] [--apply] [--limit N]` | **Hidden / advanced.** One-shot repair tool for stores migrated from pre-v1.0 schemas. Identifies belief rows whose α is inflated above prior yet have zero `feedback_history` and zero `belief_corroborations` entries (audit-trail-less ghosts; α floor `--threshold`, default 4.0). `--target` (default 4.0, must be ≤ threshold) is the α value clamped down to; `--limit` caps rows per call. Default dry-run; `--apply` writes the UPDATE plus a negative-valence `feedback_history` row inside one transaction so the clamp is reversible and idempotent. |
| `scan-derivation --reference PATH [--threshold F] [--n N] [PATH ...]` | (v3.0+, [#681](https://github.com/robotrocketscience/aelfrice/issues/681)) N-gram Jaccard similarity gate against a reference document. Reads each PATH (or stdin via `-` / no args), prints `MATCH [score] label: excerpt` or `clean [score] label`, exits 0 (all clean) / 1 (one or more matched) / 2 (reference unreadable). Designed to drop into a git pre-commit / pre-push hook. Defaults: 3-gram windows, threshold 0.6. |

## Lifecycle

| Command | What it does |
|---|---|
| `setup` | Install the full default hook set: `UserPromptSubmit` retrieval hook + statusline notifier, plus — each default ON with a `--no-*` opt-out — transcript-ingest (four-event loggers), commit-ingest (`PostToolUse:Bash`), session-start (L0 injection), stop-hook (lock prompt), sessionstart-recap, search-tool (`PreToolUse:Grep\|Glob`), search-tool-bash, pre-issue-guard, claude-memory-mirror (`PostToolUse:Write\|Edit\|MultiEdit`, v3.7.0+; inert until `AELFRICE_MIRROR_CLAUDE_MEMORY` is set, #985), and agent-context (`PreToolUse:Agent`; dispatched subagents inherit locked + task-relevant beliefs, #1068). `--rebuilder` is the one opt-in lane (`PreCompact`). Auto-detects scope (`project` if `cwd/.venv` matches the active interpreter, else `user`). Idempotent + atomic. See [INSTALL § default-on hooks](INSTALL.md). |
| `unsetup` | Remove the hook and our statusline contribution. Composed statuslines are surgically unwrapped. Mirrors `setup` flags. |
| `upgrade-cmd [--check]` | Print the install-method-aware upgrade command (uv tool / pipx / venv / system). Includes wheel SHA-256 for hash-pinned installs. Does not run the upgrade itself — replacing the running interpreter mid-process is unreliable. (Bare `upgrade` remains as a hidden deprecated alias that prints a stderr notice before delegating to `upgrade-cmd`.) |
| `uninstall (--keep-db \| --archive PATH \| --purge)` | Tear down aelfrice. One disposition flag required. `--purge` has three confirmation gates. `--archive` writes a Fernet-encrypted file then deletes the original. |
| `migrate [--from P] [--apply] [--all]` | Port beliefs from the legacy global DB into the active project's per-project DB. Dry-run by default. Read-only on the source. |
| `statusline` | Emit the update-banner snippet (or empty). Reads cache only, no network. |
| `mcp` | Start the FastMCP stdio server exposing the 15 memory tools. Requires the `[mcp]` extra (`uv tool install "aelfrice[mcp]"`). Blocks; SIGINT exits cleanly. Hosts can also use `python -m aelfrice.mcp_server`. See [MCP](MCP.md). |
| `ingest-transcript [PATH \| --batch DIR] [--since DATE]` | Ingest one `turns.jsonl` file or batch-walk a directory. Auto-detects aelfrice and Claude Code formats. Idempotent. |
| `rebuild [--transcript PATH] [--n N] [--budget N]` | Manual context-rebuilder run (alpha; normally fires on `PreCompact`). Prints the rebuild block to stdout. |
| `project-warm <path> [--debounce N]` | CwdChanged hook entry point. Resolves `<path>` to a project root (git work-tree or `~/.aelfrice/projects/<id>/`-provisioned ancestor), pre-loads the SQLite + OS page cache, and writes a sentinel under `~/.aelfrice/projects/<id>/.last_warm`. Silent no-op for unknown paths, denied paths (default deny: `/tmp/**`, `/var/folders/**`, `~/Downloads/**`, `~/Desktop/**` — override via `~/.aelfrice/config.json` `project_warm.deny_globs`), and any call inside the 60-second debounce window. Always exits 0; never writes to stdout. |
| `session-delta [--id ID] [--telemetry-path PATH]` | **Advanced/hidden.** SessionEnd hook entry point. Computes per-session deltas (beliefs created, corrections detected, feedback given, velocity) from beliefs tagged with `--id` in the active store, combines with a current store snapshot (beliefs/graph blocks) and rolling-window rollups from the existing `telemetry.jsonl`, and appends one v=1 JSON row to `PATH` (default `~/.aelfrice/telemetry.jsonl`). Missing or empty `--id` is a silent no-op (stderr warning, exit 0). Idle sessions with zero beliefs still emit a row so `len(telemetry.jsonl)` equals session count. Not shown in `aelf --help`. |

## HRR persistence reporter

(v3.0+, #696) `aelf doctor` and `aelf status` report whether the HRR structural-index is persisted to disk, how large the blob is, and how long the most recent rebuild took. The rows let an operator confirm warm-load is actually firing instead of silently rebuilding every cold start.

### `aelf doctor`

Three rows under the `HRR` block:

| Row | Type | Meaning |
|---|---|---|
| `hrr.persist_enabled` | `true` / `false` | Result of `HRRStructIndexCache._resolve_persist_dir()`. `true` when the resolver returns a non-`None` path — captures `store_path` set + `AELFRICE_HRR_PERSIST != "0"` + (#695) ephemeral-path not auto-disabled. |
| `hrr.on_disk_bytes` | int | `os.path.getsize(<persist_dir>/struct.npy) + os.path.getsize(<persist_dir>/meta.npz)` when both files exist; `0` otherwise. |
| `hrr.last_build_seconds` | float / `n/a` | Wall-clock of the most recent `HRRStructIndex.build()` call this process. `n/a` if no build has fired since process start. |

On `aelf doctor`, `--json` only affects `--meta-beliefs`; the hooks/graph report (including the HRR rows) is text-only today. When persistence is off, the `reason` string (`no store path`, `AELFRICE_HRR_PERSIST=0`, `ephemeral path`) is appended to the `hrr.persist_enabled: false` row.

### `aelf status`

One summary line:

```
hrr.persist_state: on <N> bytes, last build <X>s
hrr.persist_state: off (<reason>)
```

Where `<reason>` is one of `no store path`, `AELFRICE_HRR_PERSIST=0`, `ephemeral path`, or `unknown (probe error)`.

See [CONFIG § `hrr_persist`](CONFIG.md) for the underlying flag, [`docs/design/feature-hrr-integration.md`](../design/feature-hrr-integration.md) for the substrate spec, and `tests/test_hrr_struct_index.py` for the matrix of observable states.

## Help flags

`aelf --help` shows the everyday surface (visible subcommands). `aelf --help --advanced` (or `aelf --advanced`) shows the full surface including hidden subcommands (`bench`, `cadence-score`, `clamp-ghosts`, `demote`, `export-canvas`, `feedback`, `gate`, `health`, `ingest-transcript`, `label`, `project-warm`, `regime`, `resolve`, `session-delta`, `spine`, `stats`, `statusline`, `uninstall`, `unsetup`, `upgrade`, `upgrade-cmd`, `validate`). The `--advanced` flag was wired in v1.4 (PR #174). `export-obsidian` is visible in the everyday `--help`.

## Output and exit codes

- Human-readable on stdout. Errors on stderr.
- `bench` prints JSON.
- Exit codes: `0` = success or no-op, `1` = bad argument / belief not found / structural audit failure, `>1` = uncaught exception.

## Defaults that matter

| Constant | Value |
|---|---|
| Lock initial prior | `(α, β) = (9.0, 0.5)` |
| Decay target | Jeffreys prior `(0.5, 0.5)` |
| Half-lives | factual 14d, requirement 30d, preference 12w, correction 24w |
| Retrieval token budget | 2,400 (`DEFAULT_TOKEN_BUDGET` in `aelfrice.retrieval`; was 2,000 prior to v1.3) |
| Valence propagation | BFS, max 3 hops, threshold 0.05; fires on every feedback event (off: `AELFRICE_VALENCE_PROPAGATION=0`) |
| Benchmark hit-depth | top-5 |

Half-lives and the decay target live in `aelfrice.scoring`; the token budget in `aelfrice.retrieval` (`DEFAULT_TOKEN_BUDGET`); the lock prior in `aelfrice.derivation` / `aelfrice.classification_core`; valence-propagation defaults on `MemoryStore.propagate_valence`; the benchmark hit-depth in `aelfrice.benchmark` (`DEFAULT_TOP_K`). See [ARCHITECTURE](../concepts/ARCHITECTURE.md).
