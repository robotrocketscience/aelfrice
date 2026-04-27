# CLI Reference

Fifteen subcommands. The first eight (retrieval/feedback) are also available as MCP tools and Claude Code slash commands. The lifecycle commands (`setup`/`unsetup`/`upgrade`/`uninstall`/`statusline`/`doctor`) and `bench` are CLI-only.

DB resolves from `$AELFRICE_DB` or `~/.aelfrice/memory.db`.

```
aelf <subcommand> [args] [options]
```

## Reference

| Command | Args | Behaviour |
|---|---|---|
| `onboard <path>` | `path` | Walk filesystem (`.md/.rst/.txt/.adoc`), git log, Python AST. Classify candidates, insert non-duplicates. Power users can tune the noise filter via a `.aelfrice.toml` at the project root — see [CONFIG.md](CONFIG.md). |
| `search <query> [--budget N]` | `query`, `--budget` (default 2000) | L0 locked + L1 FTS5 BM25, token-budgeted. |
| `lock <statement>` | `statement` | Insert at `(α, β) = (9.0, 0.5)` with `lock_level=user`. Idempotent — re-lock of identical text upgrades existing. |
| `locked [--pressured]` | `--pressured` | List locks. With flag, only those with `demotion_pressure > 0`. |
| `demote <belief_id>` | `belief_id` | `lock_level → none`, `locked_at → null`, `demotion_pressure → 0`. Belief itself remains. |
| `resolve` | — | Sweep unresolved CONTRADICTS edges. For each pair, pick a winner per precedence (`user_stated > user_corrected > document_recent`; ties broken by recency, then by id) and create a SUPERSEDES edge from winner to loser. Writes one `feedback_history` row per resolution with `source='contradiction_tiebreaker:<rule>'`. Idempotent. |
| `feedback <belief_id> <used\|harmful> [--source S]` | id, signal, optional source | `used` ⇒ α += 1. `harmful` ⇒ β += 1. Positive feedback flowing through outbound CONTRADICTS edges to user-locks bumps their `demotion_pressure`; ≥ 5 ⇒ auto-demote. |
| `stats` | — | beliefs / edges / locked / feedback_events counts. |
| `health` | — | Regime classifier: one of `early-onboarding`, `steady`, `lock-heavy`, `over-confident`, `insufficient-data`. |
| `doctor [--user-settings P] [--project-root D]` | — | Verify hook + statusline commands in user and project `settings.json` resolve to executables. Reports broken absolute paths and bare names not on `$PATH`. Special-cases `bash /script.sh` to inspect the script. Exits `1` on any broken finding. |
| `setup [--scope user\|project] [--project-root D] [--settings-path P] [--command C] [--timeout N] [--status-message M] [--no-statusline]` | various | Install `UserPromptSubmit` hook + `statusLine` notifier in Claude Code `settings.json`. Defaults: `--scope` auto-detects `project` if `cwd/.venv` matches `sys.prefix` else `user`; `--command` auto-resolves to absolute `aelf-hook` (project venv for project scope, `$PATH` for user scope). Also silently removes legacy dangling `/usr/local/bin/aelf{,-hook}` symlinks. Idempotent + atomic. |
| `unsetup` (same scope flags, `--command`) | — | Remove the hook + our statusline contribution. Default `--command` matches every entry whose program basename is `aelf-hook`, so a bare-name install and an absolute-path install are both cleaned by the same call. Composed statuslines are surgically unwrapped to restore the original command. |
| `upgrade [--check]` | — | Print the right pip-upgrade command for the running env (venv → `pip install --upgrade`, pipx → `pipx upgrade`, system → `pip install --user --upgrade`). When an update is available, also prints the wheel SHA-256 + PyPI release URL for hash-pinned installs. `--check` suppresses the command line. |
| `uninstall (--keep-db \| --purge \| --archive PATH) [--password-stdin] [--yes] [--keep-hook] [--settings-path P]` | one disposition flag required | Tear down aelfrice. Disposition modes are mutually exclusive. `--purge` requires typing `PURGE` then `[y/N]` unless `--yes` is passed. `--archive` requires `pip install 'aelfrice[archive]'`. By default also runs `unsetup`; `--keep-hook` opts out. Tail message points at `pip uninstall aelfrice` for wheel removal. |
| `statusline` | — | Emit the orange update-banner snippet (or empty when no update is pending). Reads cache only, no network. Composes onto an existing statusline via shell `;`. Color: truecolor → 256-color → basic, NO_COLOR honoured. |
| `bench [--db PATH] [--top-k N]` | — | Run the deterministic 16-belief × 16-query benchmark. Print a single JSON `BenchmarkReport`: `hit_at_1` / `hit_at_3` / `hit_at_5` / `mrr` + `p50_latency_ms` / `p99_latency_ms`. |
| `--version` (root flag) | — | Print `aelfrice X.Y.Z` and exit. |

## Output format

Human-readable on stdout (CLI), JSON on stdout (`bench`). Errors on stderr.

Exit codes: `0` = success or no-op, `1` = invalid argument or belief not found, `>1` = uncaught exception.

## Defaults that matter

- Lock initial prior: `(9.0, 0.5)`
- Decay target: Jeffreys prior `(0.5, 0.5)`
- Half-lives: factual 14d / requirement 30d / preference 12w / correction 24w
- Demotion threshold: 5 contradicting events
- Search token budget: 2000
- Valence propagation: BFS, max 3 hops, threshold 0.05
- Benchmark hit-depth: top-5 (override with `--top-k`)

All exposed as module-level constants in `aelfrice.feedback`, `aelfrice.scoring`, `aelfrice.retrieval`. See [ARCHITECTURE](ARCHITECTURE.md).
