---
name: aelf:rebuild
description: v1.1 alpha — emit the context-rebuild block (what the PreCompact hook would produce) for inspection.
argument-hint: optional --transcript PATH, --n N, --budget N
allowed-tools:
  - Bash
---
<objective>
Manually invoke the context rebuilder and print the rebuild block to
the conversation. Useful for inspecting what aelfrice would inject if
PreCompact fired right now without actually triggering compaction.
</objective>

<process>
Run: `uv run aelf rebuild $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
