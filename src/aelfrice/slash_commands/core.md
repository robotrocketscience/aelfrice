---
name: aelf:core
description: Surface load-bearing beliefs — locked ∪ corroborated (re-asserted ≥2 times after first ingest) ∪ high-posterior (μ ≥ 2/3, α+β ≥ 4). Read-only.
argument-hint: Optional flags, e.g. --json or --locked-only
allowed-tools:
  - Bash
---
<objective>
List the beliefs that anchor the store: any belief that is user-locked,
re-asserted at least twice after the original ingest (corroboration_count ≥ 2; re-ingest events are counted, not distinct sources), or has a strong
multi-event positive posterior. This is the operator's first-look lens
when checking whether the store foundation is healthy.
</objective>

<process>
Run: `uv run aelf core $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
