---
name: aelf:onboard
description: Scan a project directory and ingest beliefs into aelfrice memory.
argument-hint: Path to the project directory (e.g. . or ~/projects/myapp)
allowed-tools:
  - Bash
---
<objective>
Onboard a project into aelfrice using the synchronous regex-classifier
pipeline. The polymorphic host-LLM handshake is reached via the MCP
tool surface; this slash command uses the same code path as the CLI.
</objective>

<process>
Run: `uv run aelf onboard "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
