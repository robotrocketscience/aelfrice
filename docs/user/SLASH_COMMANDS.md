# Slash Commands

Nineteen markdown files in `src/aelfrice/slash_commands/`, tracking the v1.2.0 CLI consolidation, the v1.4.0 `rebuild` promotion, the v2.0 reasoning surfaces, and the v2.x `eval` calibration surface. After `aelf setup`, they appear as `/aelf:*` in the host. Each is a thin wrapper over the CLI — `/aelf:foo` invokes `aelf foo` against the active project's DB.

Slash files cover the everyday user-facing surface, plus a handful of operator workflows where one keystroke matters (`/aelf:uninstall`, `/aelf:upgrade`). Hidden CLI subcommands (`bench`, `clamp-ghosts`, `demote`, `feedback`, `gate`, `health`, `ingest-transcript`, `project-warm`, `regime`, `resolve`, `session-delta`, `stats`, `statusline`, `unsetup`, `upgrade-cmd`, `validate`) and the per-hook entry points stay callable from the CLI for scripting and back-compat — they're not surfaced as slashes. The visible CLI verbs `migrate`, `mcp`, `sweep-feedback`, and `scan-derivation` are likewise CLI-only because they're operator / scripting flows rather than per-turn agent surface.

`aelf setup` installs all slash-command files automatically into `~/.claude/commands/aelf/` and prunes any stale files left behind by renames (e.g. `stats.md` after the v1.2.0 rename to `status.md`). Re-running `aelf setup` after an upgrade is sufficient to keep the set current.

## Reference

| Slash | Argument hint |
|---|---|
| `/aelf:onboard` | path to project dir |
| `/aelf:search` | keyword query |
| `/aelf:lock` | statement to lock |
| `/aelf:unlock` | belief id — drops the lock without changing origin tier |
| `/aelf:locked` | (none) |
| `/aelf:confirm` | belief id — bumps posterior without freezing |
| `/aelf:promote` | belief id — promote `agent_inferred` to `user_validated`. v3.0+ accepts `--to-scope SCOPE` to flip federation visibility in the same call (#689). |
| `/aelf:delete` | belief id (locked beliefs require `--force`; `--yes` skips prompt) |
| `/aelf:core` | optional `--json`, `--locked-only` — surfaces load-bearing beliefs |
| `/aelf:status` | (none) — belief / lock / history counts (renamed from `stats` at v1.2.0) |
| `/aelf:doctor` | optional `[hooks\|graph]`, `--user-settings`, `--project-root` |
| `/aelf:tail` | optional `--filter`, `--since`, `--no-follow` — live-tail the hook injection audit log |
| `/aelf:setup` | optional `--scope`, `--command`, `--transcript-ingest`, etc. |
| `/aelf:upgrade` | (none) — imperative upgrade. Detects install context, runs `uv tool upgrade aelfrice` (or pipx/pip equivalent) in Bash (separate process, no mid-process interpreter replacement), then `aelf setup` to refresh the slash-command bundle, then clears the stale update-banner cache. The advisory `aelf upgrade-cmd` CLI verb still exists for scripted use. |
| `/aelf:uninstall` | one of `--keep-db`, `--archive`, `--purge` |
| `/aelf:rebuild` | optional `--n N`, `--budget T`, `--transcript PATH` — manually fire the context rebuilder; v1.4.0+ |
| `/aelf:reason` | keyword query — v2.0+ (#389) walks the belief graph from BM25-seeded starting points; v3.0+ ([#645](https://github.com/robotrocketscience/aelfrice/issues/645) R3, [#690](https://github.com/robotrocketscience/aelfrice/issues/690), [#713](https://github.com/robotrocketscience/aelfrice/issues/713)) expands into a three-step orchestrator: (1) run `aelf reason --json` and print the chain; (2) fan out one Task subagent per `payload.dispatch[i]` with a role-tagged prompt (Verifier / Gap-filler / Fork-resolver derived from VERDICT + ImpasseKind); (3) print a `SUGGESTED UPDATES` section that maps `payload.suggested_updates[*]` to `aelf feedback` close-the-loop directions. Peer hops in foreign scopes are annotated `[scope:<name>]`. |
| `/aelf:wonder` | two modes (v3.0+, [#542](https://github.com/robotrocketscience/aelfrice/issues/542) / [#552](https://github.com/robotrocketscience/aelfrice/issues/552)). **No-arg / `--seed`** runs the v2.0 graph-walk consolidation (`--top N`, `--emit-phantoms`). **Positional `QUERY` / `--axes QUERY`** runs `aelf wonder --axes QUERY`, fans out one Task subagent per research axis with the axis name + search hints + gap context, collects each subagent's research document into a JSONL file with `{axis_name, content, anchor_ids}` rows, and hands the file to `aelf wonder --persist-docs FILE` which materialises one phantom per axis via `wonder_ingest`. Agent-count shorthand `quick N-agent` / `deep N-agent` recognised. Phantom-store integration shipped at v3.0; pre-v3 the slash only emitted candidates. |
| `/aelf:eval` | optional `--corpus PATH`, `--k N`, `--seed N`, `--json` — runs the relevance-calibration harness (P@K / ROC-AUC / Spearman ρ) ratified at #365 |

Behaviour matches the CLI exactly — see [COMMANDS](COMMANDS.md). The v1.1.0 `edges` → `threads` user-facing rename does not surface here; the program name remains `aelf`.

## Pick a surface

| Caller | Use |
|---|---|
| You, in Claude Code | `/aelf:*` slash command |
| The agent, mid-turn | `aelf:*` MCP tool — see [MCP](MCP.md) |
| Shell or script | `aelf` CLI — see [COMMANDS](COMMANDS.md) |
| Tests / embedded | `tool_*` handlers from `aelfrice.mcp_server` |

Remove with `rm -rf ~/.claude/commands/aelf/` plus `aelf unsetup`. The two are independent registrations.

## `/aelf:upgrade` orchestrator flow

The `upgrade` slash file is the only `/aelf:*` command that does not pass straight through to a single CLI verb. It orchestrates four steps in sequence; the underlying upgrade itself runs in a Bash subprocess separate from the running `aelf` interpreter, so there is no mid-process interpreter replacement to worry about.

```mermaid
sequenceDiagram
    actor User
    participant SlashHost as Slash host
    participant Slash as /aelf:upgrade
    participant CLI as aelf CLI
    participant Bash

    User->>SlashHost: invoke /aelf:upgrade
    SlashHost->>Slash: load upgrade slash script

    Note over Slash: Step 1 — detect install context
    Slash->>CLI: aelf upgrade-cmd
    CLI-->>Slash: prints "run: <command>" or "up to date"
    Slash->>Slash: parse printed line
    Slash->>User: if up-to-date, print message and stop

    Note over Slash,Bash: Step 2 — execute upgrade via Bash
    Slash->>Bash: run <command> in subprocess
    Bash-->>User: stream stdout/stderr
    Bash-->>Slash: exit code
    Slash->>User: if non-zero, print captured output and stop

    Note over Slash,CLI: Step 3 — refresh slash bundle
    Slash->>CLI: aelf setup
    CLI-->>User: deploy/prune /aelf:* bundle

    Note over Slash,CLI: Step 4 — clear upgrade banner cache
    Slash->>CLI: aelf upgrade-cmd
    CLI-->>User: refresh cache so banner disappears
```

## `detect_reachable_installs()` — running-venv suppression

Exposes every `aelf` install on the user's PATH, suppressing the venv that hosts the running interpreter (otherwise `uv run` produces a spurious "multiple installs detected" warning when there's actually only one persistent install on the user's shell PATH).

```mermaid
flowchart TD
    A[detect_reachable_installs] --> B[init empty sites list]
    B --> C[compute uv_root]
    C --> D[compute pipx_root]
    D --> E[resolve all `aelf` on PATH → path_aelf_resolved]
    E --> F[_running_interpreter_aelf]

    F --> G{base_prefix != sys.prefix?}
    G -->|no| H[return None]
    G -->|yes| I[candidate = sys.prefix/bin/aelf]
    I --> J{candidate exists?}
    J -->|no| H
    J -->|yes| K[candidate.resolve]
    K -->|error| H
    K -->|ok| L[return resolved Path]

    H --> M[running_aelf = None]
    L --> M[running_aelf = resolved Path]

    M --> N[if uv_root: append InstallSite kind=uv_tool]
    N --> O[if pipx_root: append InstallSite kind=pipx]
    O --> P[iterate exe in path_aelf_resolved]

    P --> Q{exe under uv_root or pipx_root?}
    Q -->|yes| P
    Q -->|no| R{running_aelf is set AND exe == running_aelf?}
    R -->|yes| P
    R -->|no| S[append InstallSite kind=user_local_bin path=exe]
    S --> P

    P -->|done| T[return sites]
```

Source: `src/aelfrice/lifecycle.py`. Diagrams generated by Sourcery for PR #513.
