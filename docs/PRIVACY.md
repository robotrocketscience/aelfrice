# Privacy

Verifiable properties of the codebase, not marketing claims.

## Local only

The store, retrieval, scoring, scanner, and feedback paths run locally against SQLite. No network code in `store.py`, `retrieval.py`, `scoring.py`, `feedback.py`, `scanner.py`, or `cli.py`. Confirm:

```bash
grep -rE "requests|httpx|urllib|aiohttp|socket\.|http\." src/aelfrice/
```

The optional `[mcp]` extra (`fastmcp`) speaks MCP over **stdio** to the host on the same machine. No remote sockets.

## No telemetry

There is no usage tracking. No analytics. No phone-home. Not opt-out — **the capability does not exist in the shipped package.** No conditional import, no commented-out endpoint, no env-var toggle. Confirm by reading `pyproject.toml` — only `[mcp]` adds anything beyond stdlib.

## No accounts

No sign-in, no API key, no sync server. Everything is the file at `~/.aelfrice/memory.db` (or `$AELFRICE_DB`). To back up: copy the file. To share across machines: sync the file. It's plain SQLite.

## Data location

Default `~/.aelfrice/memory.db` — single SQLite file with WAL journaling. Six tables: `beliefs`, `beliefs_fts`, `edges`, `feedback_history`, `onboard_sessions`, `schema_meta`.

Override with `AELFRICE_DB`. Use `:memory:` for ephemeral.

## What aelfrice doesn't control

The cloud LLM at the other end of your prompt sees whatever aelfrice injects. That's inherent to using a cloud LLM. Mitigations:

- **Per-query token budget** (default 2000). The full memory is never injected.
- **L0/L1 ordering** surfaces locks + today's relevance, not a memory dump.

If a fact must never leave your machine, don't store it.

## Reproducible from source

All beliefs come from files you already have: code, docs, git history. After `rm ~/.aelfrice/memory.db`, re-running onboard is deterministic up to the classifier. The state of the world is your codebase, not the memory.

## Reporting

See [SECURITY.md](../SECURITY.md). Privacy issues are treated as security issues.
