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

No sign-in, no API key, no sync server. Everything is one local SQLite file. To back up: copy the file. aelfrice ships no mechanism for sharing memory contents between users, machines, or projects — see [LIMITATIONS § Sharing or sync of brain-graph content](LIMITATIONS.md#sharing-or-sync-of-brain-graph-content).

## Data location

A single SQLite file with WAL journaling. Six tables: `beliefs`, `beliefs_fts`, `edges`, `feedback_history`, `onboard_sessions`, `schema_meta`.

Resolution order (v1.1.0):

1. `$AELFRICE_DB` if set (override; use `:memory:` for ephemeral).
2. `<git-common-dir>/aelfrice/memory.db` when `cwd` is inside a git work-tree. `.git/` is not git-tracked — the brain graph never crosses the git boundary.
3. `~/.aelfrice/memory.db` for non-git directories.

## What aelfrice doesn't control

The cloud LLM at the other end of your prompt sees whatever aelfrice injects. That's inherent to using a cloud LLM. Mitigations:

- **Per-query token budget** (default 2000). The full memory is never injected.
- **L0/L1 ordering** surfaces locks + today's relevance, not a memory dump.

If a fact must never leave your machine, don't store it.

## Per-file opt-out: `INEDIBLE` marker (v1.3+)

Any file whose basename contains the literal string `INEDIBLE` (case-sensitive, all caps, anywhere in the basename) is unconditionally skipped by every aelfrice ingest path:

- `aelf onboard` filesystem walk and AST walk.
- `aelf ingest-transcript` (single-file invocation).
- `aelf ingest-transcript --batch DIR` recursive scan.

Examples that match: `INEDIBLE.md`, `INEDIBLE_secrets.txt`, `notes_INEDIBLE.txt`, `partINEDIBLEpart.py`. Examples that do not: `inedible.md`, `Inedible.md`. Case sensitivity is intentional — the marker should be unmistakable in directory listings.

Directory basenames also count: a directory named `INEDIBLE/` (or `INEDIBLE_drafts/`) is not descended at all, so its contents never reach a classifier or an extractor regardless of file names underneath.

The check is on the basename, not the content. When `is_inedible(path)` returns True, aelfrice does not open, read, or hash the file. The check happens before any classification, before any tokenization, before any noise filter — earlier in the pipeline than any other exclusion in the codebase.

Mechanism: see [`src/aelfrice/inedible.py`](../src/aelfrice/inedible.py). The predicate is the only opt-out aelfrice respects deterministically across every ingest path; reproduce with `python3 -c "from aelfrice.inedible import is_inedible; print(is_inedible('your/path.md'))"`.

## Reproducible from source

All beliefs come from files you already have: code, docs, git history. After `rm <resolved-db-path>`, re-running `aelf onboard .` is deterministic up to the classifier. The state of the world is your codebase, not the memory.

## Reporting

See [SECURITY.md](../SECURITY.md). Privacy issues are treated as security issues.
