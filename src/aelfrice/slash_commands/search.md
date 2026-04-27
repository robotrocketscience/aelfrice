---
name: aelf:search
description: Retrieve beliefs by keyword. Locked beliefs come first (L0), then FTS5 BM25 hits (L1).
argument-hint: Keyword query (e.g. "deployment process")
allowed-tools:
  - Bash
---
<objective>
Search the aelfrice memory store for beliefs matching the given query.
</objective>

<process>
Run: `uv run aelf search "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
