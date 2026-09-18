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
- `PreToolUse:Bash` — pre-issue duplicate guard (`aelf-pre-issue-hook`): blocks `gh issue create` on Jaccard ≥ 0.5 overlap with existing issues/commits.
- `PreCompact` — the context rebuilder. Opt-in, not part of the default bundle.

The install also writes the statusline snippet and the bundled slash
bundle under `~/.claude/commands/aelf/`.

On the `claude` host, each default-on lane is opt-out through its own
`--no-<lane>` flag — `--no-pre-issue-guard`, `--no-statusline`, and the
rest — and you opt the rebuilder in with `--rebuilder`. Run
`aelf setup --help` for the full claude-host flag list.

On the `codex` host, `aelf setup --host codex` accepts only `--force` and
`--codex-skills` / `--no-codex-skills`. It reads none of the claude-host
flags above, so it refuses any of them with exit 2 and changes nothing
(#1429). `aelf unsetup --host codex` accepts no flag but `--host`. Run
both commands with no other options on this host.
</objective>

<process>
Run: `uv run aelf setup`

On the `claude` host, the install scope is auto-detected by default:
`project` (writing `<root>/.claude/settings.json`) if the current
directory has a `.venv` matching the active interpreter, else `user`
(`~/.claude/settings.json`). Each recorded command is the absolute path
of the corresponding script entry-point (project venv for project scope,
`$PATH` resolution for user scope). To force a scope, pass `--scope user`
or `--scope project`; to write to an explicit location, pass
`--settings-path PATH`. These are claude-host flags: the `codex` host
refuses them (see the objective above). The command is idempotent:
running it twice results in exactly one matching entry per lane.

Display the output verbatim. Do not add commentary.
</process>
