---
name: aelf:reason
description: Walk the belief graph from BM25-seeded starting points to surface a reasoning chain over a query.
argument-hint: Keyword query (e.g. "session id resolution")
allowed-tools:
  - Bash
---
<objective>
Surface a reasoning chain over the aelfrice belief graph for the given query. Seeds come from top-3 BM25 hits; expansion walks outbound edges with terminal-tight defaults.
</objective>

<process>
Run: `uv run aelf reason "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
