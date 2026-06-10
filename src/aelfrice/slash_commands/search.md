---
name: aelf:search
description: Retrieve beliefs by keyword. Locked beliefs come first (L0), then entity-index hits (L2.5) and FTS5 BM25 hits (L1); peer-scope hits, if federated, are listed last.
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
