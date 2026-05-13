---
name: aelf:upgrade
description: Upgrade aelfrice to the latest published version using uv.
allowed-tools:
  - Bash
---
<objective>
Imperative upgrade. This slash is the *end-to-end* upgrade flow: ask
the CLI for the canonical command, **run the upgrade itself**, then
re-run `aelf setup` so any slash-command bundle changes that shipped
in the new release land cleanly (orphan-pruned, new files written).

Per #730 aelfrice is supported on a single install channel: `uv tool`.
If the active install was set up via a different installer (pipx, pip,
system), `aelf upgrade-cmd` emits a migration command (uninstall the
old + `uv tool install aelfrice`) — execute it verbatim, same as a
plain upgrade.

The user invoked this slash because they want aelfrice on the new
version when it's done. Producing the detection output and stopping
is the wrong outcome — that is what `aelf upgrade-cmd` does on its
own. This slash exists to take the next step.

Why Bash and not in-process: replacing the running package
mid-process is unreliable on Windows and can leave a broken
interpreter. The Bash block runs the `run:` command in a subprocess
separate from the running `aelf` — no mid-process replacement.
</objective>

<process>
Step 1 — fetch the canonical upgrade command.

Run `aelf upgrade-cmd` (no flags) to ask aelfrice itself what command
would upgrade the active install. Use the no-flag form deliberately:
the `--check` form short-circuited the `run:` line on aelfrice ≤2.0.1
(#522 lands the symmetry fix in ≥2.0.2), so users who first hit
`/aelf:upgrade` against a pre-fix CLI got no parseable command and
the slash silently did nothing (#530). The no-flag form has emitted
`run:` on every released CLI since the subcommand existed.

**Capture two things from the output:**

- If the output contains `aelfrice is up to date`, print that line
  verbatim and stop. There is nothing to do.
- Otherwise the output contains a line of the form `run: <command>`.
  Extract `<command>` exactly — everything after `run: `. This is the
  canonical command for this install. It is one of:
    - `uv tool upgrade aelfrice` (uv-managed install, in-place upgrade)
    - `pipx uninstall aelfrice && uv tool install aelfrice` (pipx install, migrate to uv)
    - `pip uninstall -y aelfrice && uv tool install aelfrice` (pip/venv/system, migrate to uv)
  **You must use it verbatim** in step 2.

Any parenthetical note in the output (such as
`(installed via uv tool — use uv to upgrade)` or
`(aelfrice is supported via 'uv tool' only (#730) …)`) is hint text
for humans reading the CLI directly. It is NOT a substitute for step
2. The `run:` line is the canonical instruction; the parenthetical is
not.

Step 2 — execute the upgrade. This is the step the user is paying
for; do not skip it.

Run `<command>` from step 1 verbatim, via Bash, and stream
stdout/stderr to the user. This is not advisory; it is the operation
the slash was invoked to perform. The user has not upgraded yet
when step 1 finishes — they have only been told what *would*
upgrade. Step 2 is what makes the upgrade happen.

If `<command>` exits non-zero, print the captured output and stop —
do not proceed to step 3. (Common cause: the install method changed
since the cached detection; rerun the slash. Or: the migration
chain's first half failed — for example pipx wasn't on PATH when the
command tried to run it.)

Step 3 — refresh installed slash commands and hooks.

Run `aelf setup` to redeploy the bundled `/aelf:*` slash commands.
The setup machinery is idempotent: identical files are skipped,
missing files are written, and bundle-orphans (e.g. an old
`upgrade-cmd.md` on disk after a rename) are pruned. Display its
output.

Step 4 — clear the stale upgrade banner.

Run `aelf upgrade-cmd` (no flags) once more so aelfrice sees the new
version, clears its update cache, and the orange `⬆ aelfrice X.Y.Z`
statusline banner disappears immediately. The expected output is the
`aelfrice is up to date` line; if it still shows an update available,
something is wrong (probably another aelfrice install is on PATH —
see the multi-install warning earlier in the output).

Display each step's output verbatim. Do not add commentary between
steps. The only acceptable early stops are: (a) step 1 reports "up
to date" (nothing to do), or (b) step 2 fails non-zero (cannot
proceed safely).
</process>

<failure_modes>
**Anti-pattern: stopping after step 1.** Some agents read the
`(installed via uv tool — use uv to upgrade)` advisory (or the
`(aelfrice is supported via 'uv tool' only …)` migration note) and
interpret it as "the system has told the user what to do, my job is
done." This is wrong. The slash is `/aelf:upgrade`, not
`/aelf:show-me-the-upgrade-command`. The user clicked it because they
wanted the upgrade to happen. If you stop here, the user is no
further along than when they started, and they will reasonably file
a bug
([#611](https://github.com/robotrocketscience/aelfrice/issues/611)
is the original report of exactly this failure).

**Anti-pattern: substituting a "simpler" command.** Step 2's command
is whatever `run: <command>` produced in step 1. Do not substitute
`uv run aelf upgrade` or `pip install -U aelfrice` or any other
guess; the CLI knows which install method is active and emitted the
right command. Running a different command can install a *second*
copy of aelfrice that shadows or competes with the original.
</failure_modes>
