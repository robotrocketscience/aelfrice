---
name: aelf:feed
description: Read the belief-write event log — see what aelfrice has been recording.
allowed-tools:
  - Bash
---
<objective>
Surface what aelfrice has been writing — one JSONL row per belief
lock / onboard / wonder-promote / feedback event. The log lives at
`<git-common-dir>/aelfrice/feed.jsonl` (sibling of memory.db).
</objective>

<process>
Run: `uv run aelf feed $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
