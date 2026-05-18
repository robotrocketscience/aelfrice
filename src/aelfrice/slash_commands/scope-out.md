---
name: aelf:scope-out
description: Suppress beliefs matching a substring from this session's memory retrieval.
allowed-tools:
  - Bash
---
<objective>
Stop the UserPromptSubmit memory hook from re-injecting beliefs that
match a substring for the rest of the active session. Use when the user
has asked the agent to drop a topic that keeps resurfacing despite
explicit instruction (#856).
</objective>

<process>
Run: `uv run aelf scope-out $ARGUMENTS`
Display the output verbatim. Do not add commentary.

`$ARGUMENTS` may be:
- a substring (case-insensitive) to add to the exclusion list
- `--list` to show active exclusions
- `--clear` to empty the list

The exclusion list is scoped to the current session; it is
cleared automatically when a new session starts (session_id mismatch).
</process>
