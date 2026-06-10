---
name: aelf:setup
description: Install the aelfrice hooks (UserPromptSubmit, Stop, PreToolUse, PostToolUse, PreCompact, PostCompact, SessionStart) in the host's settings.json and the statusline snippet.
allowed-tools:
  - Bash
---
<objective>
Wire aelfrice into the host so that every user prompt is augmented
with the most relevant locked beliefs and FTS5 hits from the local
memory store, and so that session activity flows back into the belief
graph without manual `aelf` invocations. By default `setup` installs
the full default-on hook bundle, not just one entry:

- `UserPromptSubmit` — the retrieval-injection hook (`aelf-hook`).
- `UserPromptSubmit` / `Stop` / `PreCompact` / `PostCompact` —
  transcript-ingest (`aelf-transcript-logger`).
- `PostToolUse:Bash` — commit-ingest (`aelf-commit-ingest`).
- `SessionStart` — session-warm + `<recent-work>` block (#887).
- `Stop` — stop-hook cadence dispatch (#749 / #871 / #876).
- `PreToolUse:Grep|Glob` — `aelf-search-tool-hook` (#134, default-on
  since v3.0.1 #738).
- `PreToolUse:Bash` — the same `aelf-search-tool-hook` script with a Bash matcher (dispatches internally on tool_name; same wave).
- `PreToolUse:Bash` — pre-issue duplicate guard (`aelf-pre-issue-hook`): blocks `gh issue create` on Jaccard ≥ 0.5 overlap with existing issues/commits. Opt out with `--no-pre-issue-guard`.
- `--rebuilder` (opt-in) — PreCompact rebuilder hook.

Plus the statusline snippet (skip with `--no-statusline`). The bundled
slash bundle under `~/.claude/commands/aelf/` is also written. All
defaults are opt-out via `--no-<lane>` flags; see `aelf setup --help`.
</objective>

<process>
Run: `uv run aelf setup`

By default the install scope is auto-detected: `project` (writing
`<root>/.claude/settings.json`) if the current directory has a `.venv`
matching the active interpreter, else `user` (`~/.claude/settings.json`).
Each recorded command is the absolute path of the corresponding script
entry-point (project venv for project scope, `$PATH` resolution for
user scope). Pass `--scope user` or `--scope project` to force a
scope, or `--settings-path PATH` to write to an explicit location. The command is idempotent:
running it twice results in exactly one matching entry per lane.

Display the output verbatim. Do not add commentary.
</process>
