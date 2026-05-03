# MCP

aelfrice exposes eleven memory tools through a [Model Context Protocol](https://modelcontextprotocol.io) server. The agent calls them mid-turn; you don't have to invoke them yourself.

Lifecycle commands (`setup`, `unsetup`, `migrate`, `doctor`, `upgrade`, `uninstall`) are CLI-only.

## Install + run

```bash
pip install "aelfrice[mcp]"
uv run python -m aelfrice.mcp_server     # or just `aelf-mcp` after install
```

Host config — Claude Code, Codex, any MCP-capable host:

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
| `aelf:search` | `query` | `budget` (default 2,000) | `{kind, n_hits, hits[]}` |
| `aelf:lock` | `statement` | — | `{kind, id, action}` |
| `aelf:locked` | — | `pressured` | `{kind, n, locked[]}` |
| `aelf:demote` | `belief_id` | — | `{kind, id, demoted}` |
| `aelf:unlock` | `belief_id` | — | `{kind, id, unlocked, audit_event_id?}` |
| `aelf:validate` | `belief_id` | `source` (default `user_validated`) | `{kind, id, prior_origin, new_origin, audit_event_id?}` |
| `aelf:promote` | `belief_id` | `source` (default `user_validated`) | `{kind, id, prior_origin, new_origin, audit_event_id?}` |
| `aelf:feedback` | `belief_id`, `signal` | `source` | `{kind, id, signal, prior_alpha, new_alpha, prior_beta, new_beta, pressured_locks, demoted_locks}` |
| `aelf:stats` | — | — | `{kind, beliefs, threads, locked, feedback_events, ...}` |
| `aelf:health` | — | — | `{kind, regime, description, classification_confidence?, features?}` |

`signal` is `"used"` or `"harmful"`. `aelf:unlock` drops a user-lock without touching origin and always writes a `lock:unlock` audit row; idempotent on already-unlocked beliefs. `aelf:promote` is a first-class alias of `aelf:validate` — identical semantics and return shape. Both `aelf:validate` and `aelf:promote` promote an `agent_inferred` belief to a user-validated origin tier (v1.2+).

## `aelf:onboard` polymorphism

Three input shapes, dispatched by which fields the caller supplied:

| Input | Phase | Returns |
|---|---|---|
| `{path}` | start | `{kind: "onboard.session_started", session_id, sentences[]}` for the host LLM to classify |
| `{session_id, classifications}` | finish | `{kind: "onboard.session_completed", inserted, skipped_*}` |
| `{}` | status | `{kind: "onboard.status", n_pending, pending_session_ids}` |

Unlike the CLI's synchronous regex `scan_repo`, the MCP onboard uses the host LLM for higher-quality classification.

## Pure handlers

Every tool is a pure function `(store, **kwargs) -> dict`. You can call them in tests without `fastmcp`:

```python
from aelfrice.store import MemoryStore
from aelfrice.mcp_server import tool_search, tool_lock

store = MemoryStore(":memory:")
tool_lock(store, statement="never push directly to main")
tool_search(store, query="push", budget=500)
```

## Backward compatibility

`aelf:stats` and `aelf:health` emit both `edges` (the v1.0 schema name) and `threads` (the v1.1.0 user-facing name) with the same integer value during the v1.1.0 deprecation window. `edges` and `edge_per_belief` are removed in v1.2.0; clients should migrate to `threads` and `thread_per_belief`.
