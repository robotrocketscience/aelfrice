---
name: aelf:demote
description: Manually demote a locked belief one tier (user -> none).
argument-hint: The belief id (16-hex prefix as shown by aelf:locked)
allowed-tools:
  - Bash
---
<objective>
Manually drop a user lock. Use this when a previously-locked
constraint no longer holds and you want it removed from the L0 layer.
</objective>

<process>
Run: `uv run aelf demote "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
