---
name: aelf:doctor
description: Diagnose hooks + brain-graph health. Exits 1 on any structural failure.
argument-hint: optional — `hooks` or `graph` to limit to one check
allowed-tools:
  - Bash
---
<objective>
Two-axis diagnostic. Hooks side scans ~/.claude/settings.json (and
the project-scope .claude/settings.json under cwd if present) for
hook / statusline commands whose program token does not resolve, and
surfaces orphan slash files. Graph side runs the structural auditor:
orphan threads, FTS5 desync, locked-pair contradictions, plus
informational metrics (credal gap, thread counts, feedback coverage).

With no argument, runs both checks. Pass `hooks` or `graph` to limit.
</objective>

<process>
Run: `uv run aelf doctor $ARGUMENTS`
Display the output verbatim. Do not add commentary. Exit code is 1
when at least one structural failure fires in either subcheck, so a
wrapping CI step can gate on it.
</process>
