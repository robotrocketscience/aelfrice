---
name: aelf:stale
description: List beliefs by age + retrieval recency (--older-than DAYS, --cold-for DAYS).
allowed-tools:
  - Bash
---
<objective>
Surface beliefs that look stale by two deterministic thresholds:
created_at older than `--older-than` days AND last_retrieved_at NULL
or older than `--cold-for` days. No decay model — the user supplies
the windows.
</objective>

<process>
Run: `uv run aelf stale "$@"`
Display the output verbatim. Do not add commentary.
</process>
