---
name: aelf:core
description: Surface load-bearing beliefs — locked ∪ corroborated (≥2 sources) ∪ high-posterior (μ ≥ 2/3, α+β ≥ 4). Read-only.
argument-hint: Optional flags, e.g. --json or --locked-only
allowed-tools:
  - Bash
---
<objective>
List the beliefs that anchor the store: any belief that is user-locked,
independently corroborated from at least two sources, or has a strong
multi-event positive posterior. This is the operator's first-look lens
when checking whether the store foundation is healthy.
</objective>

<process>
Run: `uv run aelf core $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
