# Slash Commands

Fifteen markdown files in `src/aelfrice/slash_commands/`, tracking the v1.2.0 CLI consolidation plus the v1.4.0 `rebuild` promotion. After `aelf setup`, they appear as `/aelf:*` in Claude Code. Each is a thin wrapper over the CLI — `/aelf:foo` invokes `aelf foo` against the active project's DB.

Slash files are not shipped for hidden CLI subcommands (`bench`, `feedback`, `health`, `migrate`, `project-warm`, `regime`, `session-delta`, `stats`, `statusline`, `unsetup`). Those subcommands stay callable from the CLI for scripting, hook entry-points, and back-compat aliases — they're just not surfaced as slashes.

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
| `/aelf:status` | (none) — belief / lock / history counts (renamed from `stats` at v1.2.0) |
| `/aelf:doctor` | optional `[hooks\|graph]`, `--user-settings`, `--project-root` |
| `/aelf:ingest-transcript` | path or `--batch DIR` |
| `/aelf:setup` | optional `--scope`, `--command`, `--transcript-ingest`, etc. |
| `/aelf:upgrade` | (none) — print pip-upgrade hint |
| `/aelf:uninstall` | one of `--keep-db`, `--archive`, `--purge` |
| `/aelf:rebuild` | (none) — manually fire the context rebuilder; v1.4.0+ |

Behaviour matches the CLI exactly — see [COMMANDS](COMMANDS.md). The v1.1.0 `edges` → `threads` user-facing rename does not surface here; the program name remains `aelf`.

## Pick a surface

| Caller | Use |
|---|---|
| You, in Claude Code | `/aelf:*` slash command |
| The agent, mid-turn | `aelf:*` MCP tool — see [MCP](MCP.md) |
| Shell or script | `aelf` CLI — see [COMMANDS](COMMANDS.md) |
| Tests / embedded | `tool_*` handlers from `aelfrice.mcp_server` |

Remove with `rm -rf ~/.claude/commands/aelf/` plus `aelf unsetup`. The two are independent registrations.
