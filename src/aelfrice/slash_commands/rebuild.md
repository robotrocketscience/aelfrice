---
name: aelf:rebuild
description: Manually fire the context rebuilder. Prints the rebuild block (locked + session-scoped + L2.5/L1 hits) for the most recent transcript turns.
argument-hint: (optional) `--n N` recent turns, `--budget T` token budget, `--transcript PATH` Claude Code session JSONL.
allowed-tools:
  - Bash
---
<objective>
Manually trigger the v1.4 context rebuilder. Same code path as the
PreCompact hook in `trigger_mode = "threshold"`, but explicit and
unconditional — useful for inspecting what the hook would emit, for
the eval harness, and for users running `trigger_mode = "manual"`
(the v1.4.0 default) who want to fire the rebuild on demand.
</objective>

<process>
Run: `uv run aelf rebuild $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
