# Slash Commands

Twenty-two markdown files in `src/aelfrice/slash_commands/`. After `aelf setup`, they appear as `/aelf:*` in Claude Code. Each is a thin wrapper over the CLI — `/aelf:foo` invokes `aelf foo` against the active project's DB.

Manual install:

```bash
mkdir -p ~/.claude/commands/aelf
cp src/aelfrice/slash_commands/*.md ~/.claude/commands/aelf/
```

## Reference

| Slash | Argument hint |
|---|---|
| `/aelf:onboard` | path to project dir |
| `/aelf:search` | keyword query |
| `/aelf:lock` | statement to lock |
| `/aelf:locked` | optional `--pressured` |
| `/aelf:demote` | belief id |
| `/aelf:validate` | belief id |
| `/aelf:feedback` | `<belief_id> <used\|harmful>` |
| `/aelf:resolve` | (none) — sweep CONTRADICTS via tie-breaker |
| `/aelf:stats` | (none) |
| `/aelf:health` | (none) |
| `/aelf:status` | (none) — alias for `health` |
| `/aelf:regime` | (none) — v1.0 regime classifier |
| `/aelf:doctor` | optional `--user-settings`, `--project-root` |
| `/aelf:migrate` | optional `--apply`, `--all`, `--from` |
| `/aelf:ingest-transcript` | path or `--batch DIR` |
| `/aelf:rebuild` | optional `--transcript PATH` (alpha) |
| `/aelf:setup` | optional `--scope`, `--command`, `--transcript-ingest`, etc. |
| `/aelf:unsetup` | same flags as `setup` |
| `/aelf:statusline` | (none) — emit Claude Code statusline |
| `/aelf:upgrade` | (none) — print pip-upgrade hint |
| `/aelf:uninstall` | one of `--keep-db`, `--archive`, `--purge` |
| `/aelf:bench` | optional `--top-k N` |

Behaviour matches the CLI exactly — see [COMMANDS](COMMANDS.md). The v1.1.0 `edges` → `threads` user-facing rename does not surface here; the program name remains `aelf`.

## Pick a surface

| Caller | Use |
|---|---|
| You, in Claude Code | `/aelf:*` slash command |
| The agent, mid-turn | `aelf:*` MCP tool — see [MCP](MCP.md) |
| Shell or script | `aelf` CLI — see [COMMANDS](COMMANDS.md) |
| Tests / embedded | `tool_*` handlers from `aelfrice.mcp_server` |

Remove with `rm -rf ~/.claude/commands/aelf/` plus `aelf unsetup`. The two are independent registrations.
