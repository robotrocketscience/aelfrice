---
name: aelf:feedback
description: Apply one Bayesian feedback event to a belief. Signal must be 'used' or 'harmful'.
argument-hint: <belief_id> <used|harmful>
allowed-tools:
  - Bash
---
<objective>
Record one feedback event. `used` increments alpha; `harmful` increments
beta and contributes demotion pressure to any locked beliefs reached via
CONTRADICTS threads.
</objective>

<process>
Run: `uv run aelf feedback $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
