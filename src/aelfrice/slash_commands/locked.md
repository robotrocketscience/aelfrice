---
name: aelf:locked
description: List user-locked beliefs. Pass --pressured to show only locks with nonzero demotion pressure.
argument-hint: (optional) --pressured
allowed-tools:
  - Bash
---
<objective>
Inspect the locked-belief tier. With no arguments, lists every user
lock; with `--pressured`, lists only locks that have accumulated
demotion pressure from contradicting feedback.
</objective>

<process>
Run: `uv run aelf locked $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
