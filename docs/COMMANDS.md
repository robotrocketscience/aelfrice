# Commands

Twenty-four CLI subcommands. The retrieval/feedback ones are also exposed as MCP tools (see [MCP](MCP.md)) and slash commands (see [SLASH_COMMANDS](SLASH_COMMANDS.md)). Lifecycle commands (`setup`, `doctor`, `migrate`, `upgrade`, `uninstall`, etc.) are CLI-only.

```
aelf <subcommand> [args] [options]
aelf --help
aelf --version
```

DB resolves from `$AELFRICE_DB`, then `<git-common-dir>/aelfrice/memory.db` when `cwd` is in a git work-tree, then `~/.aelfrice/memory.db` as the non-git fallback.

## Memory operations

| Command | What it does |
|---|---|
| `onboard <path>` | Walk filesystem, git log, Python AST. Classify candidates, insert non-duplicates. Tunable via `.aelfrice.toml` — see [CONFIG](CONFIG.md). Optional flags (v1.3+): `--llm-classify` (route through Haiku classifier; default-off, requires `ANTHROPIC_API_KEY`), `--dry-run` (preview candidates without inserting; requires `--llm-classify`), `--revoke-consent` (remove the stored consent sentinel and exit). |
| `search <query> [--budget N]` | L0 locked + L2.5 entity-index (v1.3+) + L1 FTS5 BM25, token-budgeted (default 2,400 at v1.3+, 2,000 prior). L2.5 default-on; disable via `[retrieval] entity_index_enabled = false` in `.aelfrice.toml` or `AELFRICE_ENTITY_INDEX=0` in the env. Distinguishes "store empty" from "no match". |
| `lock <statement>` | Insert at `(α, β) = (9.0, 0.5)` with `lock_level=user`. Idempotent — re-lock upgrades existing. |
| `locked [--pressured]` | List locks. With `--pressured`, only those with `demotion_pressure > 0`. |
| `demote <belief_id>` | Drop a user lock. Belief itself remains. |
| `validate <belief_id> [--source user_validated]` | Promote an `agent_inferred` belief to a user-validated origin (v1.2+). |
| `feedback <belief_id> <used\|harmful> [--source S]` | `used` ⇒ α += 1; `harmful` ⇒ β += 1. Harmful feedback through outbound `CONTRADICTS` threads to user-locks bumps their demotion_pressure; ≥5 ⇒ auto-demote. |
| `resolve` | Sweep unresolved `CONTRADICTS` threads. Picks a winner per precedence (`user_stated > user_corrected > document_recent`) and inserts a `SUPERSEDES` thread. Idempotent. |

## Diagnostics

| Command | What it does |
|---|---|
| `stats` | Belief / thread / lock / feedback counts. |
| `health` | Structural auditor: orphan threads, FTS5 sync, locked contradictions, corpus volume. Exits 1 on structural failure; corpus-volume warnings are informational. |
| `status` | Alias for `health`. |
| `regime` | The v1.0 regime classifier output (`supersede` / `ignore` / `mixed` / `insufficient_data`). Informational; always exits 0. |
| `doctor` | Verify hook + statusline commands resolve. Inspects `bash <script>` wrappers, flags `2>/dev/null \|\| true` patterns. Surfaces empty-store warning. Exits 1 on broken hooks. |
| `bench [--top-k N]` | Run the deterministic 16-belief × 16-query benchmark. Prints a JSON `BenchmarkReport`. |

## Lifecycle

| Command | What it does |
|---|---|
| `setup` | Install the `UserPromptSubmit` hook + statusline notifier. Auto-detects scope (`project` if `cwd/.venv` matches the active interpreter, else `user`). Idempotent + atomic. Optional flags: `--transcript-ingest`, `--commit-ingest`, `--session-start`, `--rebuilder`. |
| `unsetup` | Remove the hook and our statusline contribution. Composed statuslines are surgically unwrapped. Mirrors `setup` flags. |
| `upgrade [--check]` | Print the pip-upgrade command for the active env (venv / pipx / system). Includes wheel SHA-256 for hash-pinned installs. Does not run pip. |
| `uninstall (--keep-db \| --archive PATH \| --purge)` | Tear down aelfrice. One disposition flag required. `--purge` has three confirmation gates. `--archive` writes a Fernet-encrypted file then deletes the original. |
| `migrate [--from P] [--apply] [--all]` | Port beliefs from the legacy global DB into the active project's per-project DB. Dry-run by default. Read-only on the source. |
| `statusline` | Emit the update-banner snippet (or empty). Reads cache only, no network. |
| `ingest-transcript [PATH \| --batch DIR] [--since DATE]` | Ingest one `turns.jsonl` file or batch-walk a directory. Auto-detects aelfrice and Claude Code formats. Idempotent. |
| `rebuild [--transcript PATH] [--n N] [--budget N]` | Manual context-rebuilder run (alpha; normally fires on `PreCompact`). Prints the rebuild block to stdout. |
| `project-warm <path> [--debounce N]` | CwdChanged hook entry point. Resolves `<path>` to a project root (git work-tree or `~/.aelfrice/projects/<id>/`-provisioned ancestor), pre-loads the SQLite + OS page cache, and writes a sentinel under `~/.aelfrice/projects/<id>/.last_warm`. Silent no-op for unknown paths, denied paths (default deny: `/tmp/**`, `/var/folders/**`, `~/Downloads/**`, `~/Desktop/**` — override via `~/.aelfrice/config.json` `project_warm.deny_globs`), and any call inside the 60-second debounce window. Always exits 0; never writes to stdout. |
| `session-delta [--id ID] [--telemetry-path PATH]` | **Advanced/hidden.** SessionEnd hook entry point. Computes per-session deltas (beliefs created, corrections detected, feedback given, velocity) from beliefs tagged with `--id` in the active store, combines with a current store snapshot (beliefs/graph blocks) and rolling-window rollups from the existing `telemetry.jsonl`, and appends one v=1 JSON row to `PATH` (default `~/.aelfrice/telemetry.jsonl`). Missing or empty `--id` is a silent no-op (stderr warning, exit 0). Idle sessions with zero beliefs still emit a row so `len(telemetry.jsonl)` equals session count. Not shown in `aelf --help`. |

## Help flags

`aelf --help` shows the everyday surface (visible subcommands). `aelf --help --advanced` (or `aelf --advanced`) shows the full surface including hidden subcommands (`bench`, `feedback`, `health`, `migrate`, `project-warm`, `regime`, `session-delta`, `stats`, `statusline`, `unsetup`). The `--advanced` flag was wired in v1.4 (PR #174).

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
| Demotion threshold | 5 contradicting events |
| Retrieval token budget | 2,000 |
| Valence propagation | BFS, max 3 hops, threshold 0.05 |
| Benchmark hit-depth | top-5 |

All exposed as module-level constants in `aelfrice.feedback`, `aelfrice.scoring`, `aelfrice.retrieval`. See [ARCHITECTURE](ARCHITECTURE.md).
