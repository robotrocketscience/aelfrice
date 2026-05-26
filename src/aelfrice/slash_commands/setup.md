---
name: aelf:setup
description: Install the aelfrice hooks (UserPromptSubmit, Stop, PreToolUse, PostToolUse, SessionStart) in Claude Code's settings.json and the statusline snippet.
allowed-tools:
  - Bash
---
<objective>
Wire aelfrice into Claude Code so that every user prompt is augmented
with the most relevant locked beliefs and FTS5 hits from the local
memory store, and so that session activity flows back into the belief
graph without manual `aelf` invocations. By default `setup` installs
the full default-on hook bundle, not just one entry:

- `UserPromptSubmit` — the retrieval-injection hook (`aelf-hook`).
- `UserPromptSubmit` / `Stop` / `PreCompact` / `PostCompact` —
  transcript-ingest (`aelf ingest-transcript`).
- `PostToolUse:Bash` — commit-ingest (`aelf ingest-commits`).
- `SessionStart` — session-warm + `<recent-work>` block (#887).
- `Stop` — stop-hook cadence dispatch (#749 / #871 / #876).
- `PreToolUse:Grep|Glob` — `aelf-search-tool` (#674, default-on
  since v3.0.1 #738).
- `PreToolUse:Bash` — `aelf-search-tool-bash` (same wave).
- `--rebuilder` (opt-in) — PreCompact rebuilder hook.

Plus the statusline snippet (skip with `--no-statusline`). The 19-file
slash bundle under `~/.claude/commands/aelf/` is also written. All
defaults are opt-out via `--no-<lane>` flags; see `aelf setup --help`.
</objective>

<process>
Run: `uv run aelf setup`

By default the hook bundle is installed user-wide
(`~/.claude/settings.json`) and each recorded command is the
absolute path of the corresponding script entry-point (project venv
for `--scope project`, `$PATH` resolution for user scope). Pass
`--scope project` for a project-scoped install, or `--settings-path
PATH` to write to an explicit location. The command is idempotent:
running it twice results in exactly one matching entry per lane.

Display the output verbatim. Do not add commentary.
</process>
