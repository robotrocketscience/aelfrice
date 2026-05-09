---
name: aelf:upgrade
description: Upgrade aelfrice to the latest published version using the right tool for this install context.
allowed-tools:
  - Bash
---
<objective>
Imperative upgrade. Detects the install method (uv tool / pipx / venv /
system pip) and runs the upgrade directly via Bash, then re-runs
`aelf setup` so any slash-command bundle changes that shipped in the new
release land cleanly (orphan-pruned, new files written).

Why Bash and not `aelf`: replacing the running package mid-process is
unreliable on Windows and can leave a broken interpreter. The slash
command's Bash block runs `uv tool upgrade` (or equivalent) in a
subprocess separate from the running `aelf` — no mid-process replacement.
</objective>

<process>
Step 1 — detect install context and pick the upgrade command.

Run `aelf upgrade-cmd --check` to ask aelfrice itself what command would
upgrade the active install. Parse the printed `run: <command>` line; if
no update is available, print the "up to date" line verbatim and stop.

Step 2 — execute the upgrade.

Run the command from step 1. Stream stdout/stderr to the user. If the
command fails (non-zero exit), print the captured output and stop —
do not proceed to step 3.

Step 3 — refresh installed slash commands and hooks.

Run `aelf setup` to redeploy the bundled `/aelf:*` slash commands. The
setup machinery is idempotent: identical files are skipped, missing
files are written, and bundle-orphans (e.g. an old `upgrade-cmd.md` on
disk after a rename) are pruned. Display its output.

Step 4 — clear the stale upgrade banner.

Run `aelf upgrade-cmd` (no flags) once more so aelfrice sees the new
version, clears its update cache, and the orange `⬆ aelfrice X.Y.Z`
statusline banner disappears immediately.

Display each step's output verbatim. Do not add commentary between
steps. If any step fails, stop and print the captured output.
</process>
