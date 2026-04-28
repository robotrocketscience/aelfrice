# Claude Code Slash Commands

Twenty-one markdown files in `src/aelfrice/slash_commands/`. After `aelf setup`, they appear as `/aelf:*` in Claude Code. The v1.1.0 user-facing rename of `edges` → `threads` does not surface here — slash commands are 1:1 wrappers over the CLI, which keeps the term unchanged.

Manual install:

```bash
mkdir -p ~/.claude/commands/aelf
cp src/aelfrice/slash_commands/*.md ~/.claude/commands/aelf/
```

Each is a thin shell over the CLI — runs `aelf <subcommand>` against the same DB.

| Slash | Argument hint |
|---|---|
| `/aelf:onboard` | path to project dir |
| `/aelf:search` | keyword query |
| `/aelf:lock` | statement to lock |
| `/aelf:locked` | optional `--pressured` |
| `/aelf:demote` | belief id (16-hex) |
| `/aelf:feedback` | `<belief_id> <used\|harmful>` |
| `/aelf:stats` | (none) |
| `/aelf:health` | (none) |
| `/aelf:status` | (none) — alias for `health` |
| `/aelf:regime` | (none) — v1.0 regime classifier |
| `/aelf:resolve` | (none) — sweep CONTRADICTS via tie-breaker |
| `/aelf:doctor` | optional `--user-settings`, `--project-root` |
| `/aelf:migrate` | optional `--apply`, `--all`, `--from` |
| `/aelf:setup` | optional `--scope`, `--command`, … |
| `/aelf:unsetup` | same flags as `setup` |
| `/aelf:statusline` | (none) — emit Claude Code statusline |
| `/aelf:upgrade` | (none) — print pip-upgrade hint |
| `/aelf:uninstall` | optional `--keep-db` |
| `/aelf:rebuild` | optional `--transcript PATH` (v1.1 alpha rebuilder) |
| `/aelf:ingest-transcript` | path to `turns.jsonl` |
| `/aelf:bench` | optional `--db PATH`, `--top-k N` |

Behaviour matches the CLI exactly — see [COMMANDS](COMMANDS.md). The MCP onboard is polymorphic; the slash version is the synchronous regex path (faster, lower-quality classification).

## Pick a surface

| Caller | Use |
|---|---|
| You, in Claude Code | `/aelf:*` slash command |
| The agent, mid-turn | `aelf:*` MCP tool |
| Shell or script | `aelf` CLI |
| Tests / embedded | `tool_*` handlers from `aelfrice.mcp_server` |

Remove with `rm -rf ~/.claude/commands/aelf/` plus `aelf unsetup` (independent — slash commands and the MCP hook are separate registrations).
