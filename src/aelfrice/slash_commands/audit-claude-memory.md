---
name: aelf:audit-claude-memory
description: Cross-store dedup audit between locked aelfrice beliefs and the claude-memory MEMORY.md index — reports duplicates, contradictions, and store-exclusive entries.
allowed-tools:
  - Bash
---
<objective>
Compare the locked aelfrice belief store against the claude-memory
MEMORY.md index for the current project and surface discrepancies.
No writes are made to either store; this is a pure read-only audit.

The report has four buckets:
- **Potential duplicates** — same slot key and value in both stores
- **Potential contradictions** — same slot key, different value
- **aelfrice-only** — locked belief not present in claude-memory
- **claude-memory-only** — bullet not present in aelfrice
</objective>

<process>
Run: `uv run aelf audit-claude-memory`

Optionally:
- `uv run aelf audit-claude-memory --project /abs/path/to/project` to audit
  a specific project directory (default: current working directory).
- `uv run aelf audit-claude-memory --json` for a single machine-readable
  JSON object.

Display the output verbatim. Do not add commentary or act on the report
unless the user explicitly asks for a follow-up action.
</process>
