# CLI surface audit

> **Historical (v1.3-era design memo).** This memo was written ahead of the v1.3.0 surface consolidation and is now several releases stale. The CLI has continued to evolve (v1.5.x added `aelf tail`, `sweep-feedback`, host-driven onboard classification; v1.6.0 finalised the hardening pass). For the current surface see [COMMANDS.md](../user/COMMANDS.md). Kept here as the rationale trail behind the v1.3 visible/hidden split, not as a current spec.

Status: design memo. No code changes implied by merging this file.
Target milestone: v1.3.0 (consolidation; backwards-compat aliases for one minor).

The CLI surface grew from 11 subcommands at v1.0.0 to 22 at v1.2.0 final. This memo inventories every command, classifies it as user-facing or invisible, and proposes a consolidation that drops the count without dropping any capability.

## Inventory at v1.2.0

| # | Command | Today's purpose | Caller | Verdict |
|---|---|---|---|---|
| 1 | `onboard` | Scan a project root and ingest sentence-level beliefs from the filesystem, git log, and Python ASTs. | Human (one-shot per project). | **Keep, user-facing.** |
| 2 | `search` | L0 locked + L1 FTS5 retrieval against the active store. | Human; also exercised by hooks via `aelfrice.retrieval.retrieve()`, not the CLI. | **Keep, user-facing.** |
| 3 | `rebuild` | v1.2.0 alpha PreCompact context-rebuilder driver. Reads recent transcript turns, calls `rebuild()`, prints the rebuild block. | Hook (`aelf-pre-compact-hook`); rarely a human. | **Demote to invisible** — keep the entry point, drop the slash file, hide from `--help`. |
| 4 | `lock` | Insert or upgrade a belief to `lock_level=user`, `origin=user_stated`. | Human. | **Keep, user-facing.** |
| 5 | `locked` | List locked beliefs (optionally filter to pressured). | Human. | **Keep, user-facing.** |
| 6 | `demote` | One-tier-per-call: drop a lock, or flip `user_validated → agent_inferred`. | Human. | **Keep, user-facing.** |
| 7 | `validate` | v1.2.0 promotion: flip `agent_inferred → user_validated`. | Human. | **Keep, user-facing.** |
| 8 | `resolve` | Run the contradiction tie-breaker on every unresolved CONTRADICTS edge. | Human; could be a periodic hook. | **Keep, user-facing.** |
| 9 | `feedback` | Apply one positive/negative feedback event to a belief. | Human; the UserPromptSubmit hook calls `apply_feedback()` directly, not through this CLI. | **Keep, user-facing.** |
| 10 | `stats` | One-screen "what's in this store" snapshot — counts of beliefs / edges / locks / feedback events. | Human (quick glance). | **Keep, user-facing.** Rename to **`status`** (see below). |
| 11 | `health` | Structural auditor: orphan threads, FTS5 desync, locked-pair contradictions. Exits 1 on failure. | Human (rare); CI gate. | **Fold into `doctor graph`.** |
| 12 | `status` | Pure alias for `health` (literally the same handler). | Human (typo of `stats`?). | **Drop the alias.** Re-purpose `status` as the new name for `stats`. |
| 13 | `regime` | v1.0 regime classifier: supersede / ignore / mixed / insufficient_data based on five anchor features. | Human; preserved from v1.0 mostly for back-compat. | **Demote to invisible.** Research output, not a daily verb. |
| 14 | `migrate` | One-shot copy from legacy `~/.aelfrice/memory.db` into the per-project `.git/aelfrice/memory.db`. | Human (one-time, v1.0 → v1.1 path). | **Demote to invisible** post-v1.3. The migration era will be over; keep the code, hide from `--help`. |
| 15 | `ingest-transcript` | Lower a `turns.jsonl` archive (or directory of historical session JSONLs via `--batch`) into beliefs and edges. | Hook (PreCompact rotation spawn target); human (manual replay or `--batch` historical backfill). | **Keep, user-facing** for `--batch`; the no-arg hook spawn path stays internal. |
| 16 | `doctor` | Settings linter: walks `settings.json`, validates hook commands resolve, flags silent-failure wrappers, surfaces orphan slash files, tails the hook-failures log. Exits 1 on broken hooks. | Human; CI gate. | **Keep, expand to `doctor [hooks\|graph]`.** |
| 17 | `setup` | Idempotent install of UserPromptSubmit + statusline + opt-in transcript-ingest / commit-ingest / SessionStart / rebuilder hooks into Claude Code's `settings.json`. | Human (once per project). | **Keep, user-facing.** |
| 18 | `unsetup` | Strip aelfrice hook entries from `settings.json`. | Human (rare). | **Keep, user-facing** but consider folding into `setup --uninstall` so the verb count drops. |
| 19 | `uninstall` | Tear down the brain-graph DB (`--keep-db` / `--purge` / `--archive PATH`) plus run unsetup unless `--keep-hook`. | Human (one-shot, end-of-life). | **Keep, user-facing.** |
| 20 | `statusline` | Emit one-line statusline prefix (orange "upgrade available" banner or empty). | Statusline command in `settings.json`; never a human directly. | **Demote to invisible.** Keep the entry point; the slash file already isn't a verb users invoke. |
| 21 | `upgrade` | Print the `pip install --upgrade aelfrice` command for the current install context. | Human. | **Keep, user-facing.** Pure shim, but it correctly differentiates pipx/venv/system. |
| 22 | `bench` | Deterministic synthetic benchmark (16 beliefs, 16 queries) → hit@k / MRR / latency JSON. | CI (regression detector); rarely a human. | **Demote to invisible.** Rename to `benchmark` if it stays user-callable, or move to `python -m aelfrice.benchmark`. |

## Proposed consolidation

### User-facing commands (10)

The set every aelfrice user should know:

```
aelf onboard [PATH]              # populate the store from the filesystem
aelf search QUERY                # retrieve relevant beliefs
aelf lock STATEMENT              # assert ground truth
aelf locked                      # show what is locked
aelf demote BELIEF_ID            # drop one tier (lock or validation)
aelf validate BELIEF_ID          # promote agent_inferred -> user_validated
aelf resolve                     # run the contradiction tie-breaker
aelf feedback BELIEF_ID SIGNAL   # apply a feedback event
aelf status                      # one-screen counts (was: aelf stats)
aelf doctor [hooks|graph]        # validate hooks (was: aelf doctor) +
                                 # graph audit (was: aelf health)
aelf setup [--<feature>]*        # wire hooks into Claude Code
aelf uninstall                   # tear down DB + hooks
aelf upgrade                     # print upgrade command for this install
aelf ingest-transcript --batch DIR  # backfill historical sessions
```

That's 14 verbs; the four "rare" verbs (uninstall, upgrade, ingest-transcript, validate) sit just below the daily four (onboard, search, lock, status) but are still user-facing.

### Invisible commands (8)

Subcommands the parser still accepts but `--help` does not list. The harness, hooks, CI, and migration scripts use them; humans rarely do. Implementation: pass `help=argparse.SUPPRESS` on the subparser.

```
aelf rebuild        # PreCompact hook driver
aelf statusline     # statusline command target
aelf bench          # CI regression target
aelf regime         # v1.0 classifier (research)
aelf migrate        # v1.0 → v1.1 one-shot import
aelf health         # alias of `doctor graph` (back-compat for one minor)
aelf stats          # alias of `status` (back-compat for one minor)
aelf unsetup        # callable for scripts; humans use `setup --uninstall`
```

### Net change

| | v1.2.0 | v1.3.0 proposed | Δ |
|---|---|---|---|
| Total subcommands | 22 | 22 | 0 (we keep them all callable) |
| Listed in `--help` | 22 | 14 | **−8** |
| Distinct verbs a new user must learn | 22 | 10 | **−12** |

Capability is preserved; the surface a user sees on first contact drops from "wall of 22 commands" to "10 verbs that map to thinkable actions."

## The store/graph terminology decision

**Resolved: use "graph" externally, "store" internally.**

Today the codebase uses both terms interchangeably. Going forward:

- `aelf doctor graph` — the user-facing audit verb. "Graph" reads as the conceptual model (typed-edge belief graph) the user is reasoning about.
- `MemoryStore`, `store.db`, `db_path()` — internal naming for the SQLite implementation. Users don't see this unless they're reading code.
- `aelf status` reports counts ("beliefs", "edges", "locks") — same conceptual frame as `doctor graph`. No "store" in user-visible output.

This collapses the "what's the difference between store and graph?" question to: there isn't one user-facing. They are two names for the same thing seen from different distances.

## `doctor [hooks|graph]` shape

```
aelf doctor                  # default: run both hooks and graph
aelf doctor hooks            # settings.json + slash files + hook-failures log
aelf doctor graph            # orphan threads + FTS5 desync + locked contradictions
aelf doctor --json           # machine-readable shape for CI
```

Exit codes:

- `0` if both subchecks pass (or only the requested subcheck passes).
- `1` if any structural failure fires (broken hook **or** orphan thread / FTS5 desync / locked contradiction).
- `2` argparse usage error.

Argparse implementation: positional optional `{hooks,graph}` choice, defaults to running both. The existing `_cmd_health` and `_cmd_doctor` handlers move into `aelfrice.diagnostic.run_graph()` and `aelfrice.diagnostic.run_hooks()` — pure functions returning a structured result that `_cmd_doctor` formats. Re-using the existing `DoctorReport` machinery avoids a rewrite.

## Migration plan

One minor's worth of overlap, then deletion. Aliases live for v1.3.x; removed at v1.4.0.

1. **v1.3.0** — Land the `doctor [hooks|graph]` shape. `health`, `status`, `stats` become hidden aliases (continue to work, no `--help` listing, no slash command). Rename `stats` → `status` for the user-facing snapshot verb. Add a deprecation notice on the aliased commands.
2. **v1.3.0** — Hide `rebuild`, `regime`, `migrate`, `bench`, `statusline`, `unsetup` from `--help` via `help=argparse.SUPPRESS`. Keep callable.
3. **v1.4.0** — Delete the `health` and old `stats` aliases; keep `status`. Aliases live exactly one minor.
4. **v1.4.0** — Optional: fold `unsetup` into `setup --uninstall` and remove the standalone command. Decide based on usage telemetry from v1.3 (do users actually invoke `unsetup` standalone?).

## Slash commands

The slash command surface mirrors the `--help` listing: only user-facing commands ship as `aelf:<verb>` slash files. The eight invisible commands lose their slash files at v1.3.0:

- Drop: `aelf:rebuild.md`, `aelf:regime.md`, `aelf:migrate.md`, `aelf:bench.md`, `aelf:statusline.md`, `aelf:unsetup.md`, `aelf:health.md`, `aelf:stats.md`.
- Add: `aelf:status.md` (replaces `aelf:stats.md` and `aelf:health.md`).
- Keep: every user-facing entry from § "User-facing commands."

The orphan-slash-file check in `aelf doctor` (issue #115, shipped v1.2.0) will catch any leftover slash files automatically — the v1.3.0 deletion is documented but the check is what enforces it on installed users.

## Open questions

- **TBD: `aelf:remember`.** The MCP exposes `aelf:remember`; the CLI does not (`aelf lock` is the closest verb but stronger). Is `aelf remember` a future user-facing CLI verb, or is the MCP the canonical write path for "save this thought"? Affects whether v1.3 grows the count back toward 11.
- **TBD: rename `bench` → `benchmark`.** Three extra characters; reads better in CI logs. Rejected if `bench` is too entrenched.
- **TBD: should `migrate` get a lifetime-bound flag** like `aelf migrate --legacy-v1.0` so when v2.0 introduces another migration the verb is reusable rather than version-pinned?
- **TBD: machine-readable output across the surface.** Today `--json` exists on `health` only. v1.3 should land it on `status`, `doctor`, `locked`, `stats`-equivalent for consistency.
- **TBD: `aelf` (no subcommand)** currently prints argparse help. Could become a one-screen "what is aelfrice + what should I run first" landing page tuned for new users (similar to `git` with no args). Out of scope for the audit but flagged for the v1.3 polish window.
