# Claude Code Slash Commands

Eleven markdown files in `src/aelfrice/slash_commands/`. After `aelf setup`, they appear as `/aelf:*` in Claude Code.

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
| `/aelf:setup` | optional `--scope`, `--command`, … |
| `/aelf:unsetup` | same flags as `setup` |
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
