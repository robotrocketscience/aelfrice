# MCP Tool Reference

aelfrice exposes the same eight retrieval/feedback operations as the CLI through a [Model Context Protocol](https://modelcontextprotocol.io) server. `setup`/`unsetup` are CLI-only.

```bash
uv sync --extra mcp
uv run python -m aelfrice.mcp_server
```

Host config (Claude Code, Codex, any MCP host):

```json
{
  "mcpServers": {
    "aelfrice": {
      "command": "uv",
      "args": ["run", "--project", "/abs/path/to/aelfrice", "python", "-m", "aelfrice.mcp_server"]
    }
  }
}
```

Tools register under the `aelf:` namespace.

## Tools

| Tool | Required | Optional | Returns |
|---|---|---|---|
| `aelf:onboard` | — | `path`, `session_id`, `classifications` | polymorphic — see below |
| `aelf:search` | `query` | `budget` (default 2000) | `{kind, n_hits, hits[]}` |
| `aelf:lock` | `statement` | — | `{kind, id, action}` |
| `aelf:locked` | — | `pressured` | `{kind, n, locked[]}` |
| `aelf:demote` | `belief_id` | — | `{kind, id, demoted}` |
| `aelf:feedback` | `belief_id`, `signal` | `source` | `{kind, id, signal, prior_alpha, new_alpha, prior_beta, new_beta, pressured_locks, demoted_locks}` |
| `aelf:stats` | — | — | `{kind, beliefs, edges, locked, feedback_events, onboard_sessions_total}` |
| `aelf:health` | — | — | `{kind, regime, description, classification_confidence?, features?}` |

## `aelf:onboard` polymorphism

Three input shapes dispatched by which fields the caller supplied:

| Input | Phase | Returns |
|---|---|---|
| `{path}` | start | `{kind: "onboard.session_started", session_id, sentences[]}` for the host LLM to classify |
| `{session_id, classifications}` | finish | `{kind: "onboard.session_completed", inserted, skipped_*}` after host posts back classifications |
| `{}` | status | `{kind: "onboard.status", n_pending, pending_session_ids}` |

Unlike the CLI's regex-based `scan_repo`, the MCP onboard uses the host LLM for higher-quality classification.

## Pure handlers (testing)

```python
from aelfrice.store import Store
from aelfrice.mcp_server import tool_search, tool_lock

store = Store(":memory:")
tool_lock(store, statement="Never push directly to main")
tool_search(store, query="push", budget=500)
```

No `fastmcp` dependency needed for handler-level use.
