# CLI Reference

Eleven subcommands. The first eight (retrieval/feedback) are also available as MCP tools and Claude Code slash commands. `setup`/`unsetup`/`bench` are CLI-only.

DB resolves from `$AELFRICE_DB` or `~/.aelfrice/memory.db`.

```
aelf <subcommand> [args] [options]
```

## Reference

| Command | Args | Behaviour |
|---|---|---|
| `onboard <path>` | `path` | Walk filesystem (`.md/.rst/.txt/.adoc`), git log, Python AST. Classify candidates, insert non-duplicates. |
| `search <query> [--budget N]` | `query`, `--budget` (default 2000) | L0 locked + L1 FTS5 BM25, token-budgeted. |
| `lock <statement>` | `statement` | Insert at `(α, β) = (9.0, 0.5)` with `lock_level=user`. Idempotent — re-lock of identical text upgrades existing. |
| `locked [--pressured]` | `--pressured` | List locks. With flag, only those with `demotion_pressure > 0`. |
| `demote <belief_id>` | `belief_id` | `lock_level → none`, `locked_at → null`, `demotion_pressure → 0`. Belief itself remains. |
| `feedback <belief_id> <used\|harmful> [--source S]` | id, signal, optional source | `used` ⇒ α += 1. `harmful` ⇒ β += 1. Positive feedback flowing through outbound CONTRADICTS edges to user-locks bumps their `demotion_pressure`; ≥ 5 ⇒ auto-demote. |
| `stats` | — | beliefs / edges / locked / feedback_events counts. |
| `health` | — | Regime classifier: one of `early-onboarding`, `steady`, `lock-heavy`, `over-confident`, `insufficient-data`. |
| `setup [--scope user\|project] [--project-root D] [--settings-path P] [--command C] [--timeout N] [--status-message M]` | various | Install `UserPromptSubmit` hook in Claude Code `settings.json`. Default command: `aelf-hook`. Idempotent + atomic. |
| `unsetup` (same scope flags) | — | Remove the hook. Match by command string. |
| `bench [--db PATH] [--top-k N]` | — | Run the deterministic 16-belief × 16-query benchmark. Print a single JSON `BenchmarkReport`: `hit_at_1` / `hit_at_3` / `hit_at_5` / `mrr` + `p50_latency_ms` / `p99_latency_ms`. |

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
