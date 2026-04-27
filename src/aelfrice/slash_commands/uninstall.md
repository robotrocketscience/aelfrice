---
name: aelf:uninstall
description: Tear down aelfrice locally with explicit handling for the brain-graph DB.
allowed-tools:
  - Bash
---
<objective>
Remove aelfrice's local footprint -- the Claude Code hook + statusline
wiring, plus an explicit choice for the brain-graph SQLite DB at
`~/.aelfrice/memory.db`. The user must pick exactly one of:

  --keep-db       leave the DB in place (safe default for review)
  --purge         permanently delete the DB (redundant gates: type
                  'PURGE' verbatim, then [y/N], unless --yes)
  --archive PATH  encrypt the DB to PATH with a password, then delete
                  the original (requires `pip install 'aelfrice[archive]'`)

After data disposition, runs `unsetup` to remove the hook + statusline
unless `--keep-hook` is passed. Tail message points the user at
`pip uninstall aelfrice` for the final wheel removal.
</objective>

<process>
Run: `uv run aelf uninstall <flags>` with the flags the user requested.

If the user did not specify a disposition flag, ASK THEM which mode
they want before invoking the command. Never invoke `--purge` without
explicit user confirmation; the redundant CLI gates are the last line
of defence, not the only one.

Display the output verbatim. Do not add commentary.
</process>
