---
name: aelf:tail
description: Live-tail the per-turn hook injection audit log; pretty-print what the UserPromptSubmit and SessionStart hooks injected on each fire.
argument-hint: (optional) --filter hook=user_prompt_submit | --filter lane=L0 | --since 5m | --no-blob | --no-follow
allowed-tools:
  - Bash
---
<objective>
Stream the per-turn hook audit log so the operator can see exactly
which beliefs each UserPromptSubmit / SessionStart fire injected — id,
lane (L0 locked / L1 retrieved), token count, latency, and a snippet.
By default tails forever; pass `--no-follow` for a one-shot dump.
</objective>

<process>
Run: `uv run aelf tail $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
