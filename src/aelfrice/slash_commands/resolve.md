---
name: aelf:resolve
description: Resolve unresolved CONTRADICTS threads via the v1.0.1 tie-breaker.
allowed-tools:
  - Bash
---
<objective>
Sweep the belief graph for unresolved CONTRADICTS threads and run the
v1.0.1 tie-breaker on each. The tie-breaker picks a winner per
precedence (user_stated > user_corrected > document_recent; ties
broken by recency, then by id) and creates a SUPERSEDES thread from
winner to loser. Each resolution writes an audit row to
feedback_history with source='contradiction_tiebreaker:&lt;rule&gt;'.
</objective>

<process>
Run: `uv run aelf resolve`
Display the output verbatim. Do not add commentary.
</process>
