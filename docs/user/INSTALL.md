# Install

## Prerequisites

- Python 3.12 or 3.13. `uv` manages the Python version for you, so you don't have to install Python separately.
- [`uv`](https://docs.astral.sh/uv/), the supported install channel (#730). If you don't have `uv`, run `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code), or any agent that can spawn a hook on `UserPromptSubmit`.
- Linux, macOS, or Windows. Linux runs the full test suite on every pull request that changes code, a smoke job covers Windows, and a nightly job runs the full test suite on macOS as a report that blocks nothing. For what the smoke job asserts and what it doesn't, see [the compatibility notes in the limitations list](LIMITATIONS.md#compatibility).

## 1. Install the package

```bash
uv tool install aelfrice                # core (deps: numpy, scipy, snowballstemmer — local-only)
uv tool install --python 3.12 "aelfrice[archive]"     # add the encrypted-archive uninstall path
```

If `uv` warns that a folder isn't on your `PATH`, run `uv tool update-shell`, then open a new terminal.

Or install from the source repository, which is the developer install:

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice && uv sync
```

This command installs eleven console scripts. The first is `aelf`, the command-line interface (CLI). The other ten are hook entry points that the host spawns:

- `aelf-hook` does the per-prompt retrieval
- `aelf-transcript-logger`
- `aelf-pre-compact-hook`
- `aelf-commit-ingest`
- `aelf-search-tool-hook`
- `aelf-agent-context-hook`
- `aelf-session-start-hook`
- `aelf-stop-hook`
- `aelf-pre-issue-hook`
- `aelf-claude-memory-mirror`

> **Migration from pipx or pip.** As of v3.0.x, aelfrice supports only uv (#730). If you installed aelfrice with pipx, run `pipx uninstall aelfrice && uv tool install aelfrice` once. If you installed aelfrice with pip, run `pip uninstall -y aelfrice && uv tool install aelfrice`. At the next upgrade check, `aelf upgrade-cmd` gives you the same migration line.

To verify the install, run:

```bash
aelf --version       # aelf X.Y.Z
which aelf           # which env owns the binary
```

## 2. Wire it into Claude Code

```bash
aelf setup
```

`aelf setup` is idempotent. Run it again whenever you change Python environments, and again whenever you move projects. The command writes:

1. **A `UserPromptSubmit` hook** in `settings.json`. Every prompt then goes through `aelf-hook` for retrieval before the agent receives it.
2. **A `statusLine` notifier**. The notifier shows a one-line update banner only when a new release is available, and the banner is empty the rest of the time.
3. **The full default-on auto-capture hook set and the bundled `/aelf:*` slash commands**. See [Hooks installed by `aelf setup`](#hooks-installed-by-aelf-setup) later on this page.

Automatic detection selects the scope and the command path:

| Run from… | `--scope` | `--command` |
|---|---|---|
| inside a project venv | `project` (writes `<project>/.claude/settings.json`) | `<project>/.venv/bin/aelf-hook`, or `<project>\.venv\Scripts\aelf-hook.exe` on Windows |
| a `uv tool`-installed `aelf` outside any venv | `user` (writes `~/.claude/settings.json`) | first `aelf-hook` on `$PATH` |
| a venv unrelated to `cwd` | `user` | first `aelf-hook` on `$PATH`, falls back to the active venv |

To set these values yourself, override the detection with `--scope user|project` and `--command /abs/path/aelf-hook`.

### Codex host

If you run Codex CLI, add `--host codex` to the same lifecycle verbs:

```bash
aelf setup --host codex      # hooks into $CODEX_HOME/hooks.json (else ~/.codex) + $aelf-* skills
aelf doctor --host codex     # verify wiring; reports the installed skill count
aelf unsetup --host codex    # remove the aelfrice hooks and skills together
```

`aelf setup --host codex` writes the hook set to the Codex configuration home (#1052). The same command installs the `/aelf:*` slash-command bundle as `$aelf-*` agent skills under `~/.agents/skills/` (v4.1.0+). aelfrice generates both surfaces from one source bundle, so the two never differ. To install the hooks only, pass `--no-codex-skills`. The default is `--codex-skills`.

`aelf setup --host codex` accepts only `--force`, `--codex-skills` / `--no-codex-skills`, and `--status-message`, and `aelf unsetup --host codex` accepts no option but `--host`. The Codex path never reads the other options the two commands share with the `claude` host, such as `--scope`, `--settings-path`, `--no-statusline`, `--rebuilder`, and the per-lane opt-outs. Passing one of them exits 2 and changes nothing, rather than reporting success for an instruction that was discarded (#1429).

The skill install is idempotent and removes orphan skills. It touches only the skills aelfrice generated, because an `AELFRICE-CODEX-SKILL` marker controls replacement and removal. For the full detail — the invocation, the generation transform, and the approval warnings specific to Codex — see [the Codex host section of the slash-command reference](SLASH_COMMANDS.md#codex-host-aelf--skills).

#### Signs that aelfrice ran (unreleased)

Codex adds the text a hook prints to the model's context and doesn't show it to you, and its status line takes only built-in items. aelfrice offers two ways to see it run (#1812):

- `aelf setup --host codex --status-message "aelfrice memory"` writes the text as the Codex `statusMessage` field of the retrieval, SessionStart, and stop handlers. Codex shows it only while the hook runs, so it can disappear within a fraction of a second. Run setup without the option to remove it.
- The `[hook] summary_line` key, or `AELFRICE_HOOK_SUMMARY_LINE=1`, makes the SessionStart hook add one line that Codex shows, such as `aelfrice: 37 beliefs injected (37 locked)`. It's off by default. For the details, see [`[hook] summary_line`](CONFIG.md#hook-summary_line-unreleased).

`aelf setup --host codex` also writes `"additionalContextLimit": 9500` on the SessionStart, retrieval, and search-tool handlers, the three that return context to the model. Codex saves hook output over its limit, about 2,500 tokens by default, to a file and gives the model only a preview ([Codex hooks](https://developers.openai.com/codex/hooks)). aelfrice caps each hook's output at 9,500 characters, so a limit of 9,500 tokens keeps the whole block in the model's context.

Setup writes the SessionStart matcher `startup|resume|clear|compact|fork`, so a thread that you fork from another one gets the baseline block too. Earlier releases wrote `startup|resume|clear|compact`, which leaves out `fork`.

If you set up Codex with an earlier release, run `aelf setup --host codex` again to get these fields, the `--host codex` argument that the summary line needs, and the SessionStart matcher with `fork`. The same run refreshes the `$aelf-setup` skill, whose text now names `--status-message`. Then open `/hooks` in a Codex session and approve the changed entries, because Codex skips a hook whose definition changed until you approve it again. `aelf doctor --host codex` accepts both the earlier and the new handler shapes, but until you rerun setup it exits 1. It names `aelf-setup` among the installed skills that don't match this build, and it reports the SessionStart group that runs aelfrice under the earlier matcher as a stale install that a setup rerun fixes.

##### Move off a SessionStart wrapper

Some setups replace the SessionStart handler with a wrapper script that runs `aelf-session-start-hook` and adds a `systemMessage`. The summary line makes that wrapper unnecessary. Setup decides which handlers it owns by the program each command runs, so it treats a wrapper started by `python3` as one of your own handlers, keeps it, and adds its own SessionStart handler next to it. Codex then starts two commands that each run the session-start hook, so it can inject the baseline twice. `aelf doctor --host codex` checks each event for this state:

- When two or more handlers in one event have a command that is byte-identical to a command `aelf setup --host codex` writes for the same aelfrice program, and a plain `codex` launch starts each of them, doctor reports a failure that names the event and the program, and exits 1. Codex starts a handler in `$CODEX_HOME` only when no `[hooks.state]` entry disables it and you have approved it with `/hooks` ([`discovery.rs`, lines 917-940](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L917-L940)). The approval is the `trusted_hash` of the handler's `[hooks.state]` entry in `config.toml`. It holds only while it equals the hash that Codex computes for the handler now, from the event, the group's matcher, and the handler's command, `timeout`, `async`, `statusMessage`, and `additionalContextLimit` ([`discovery.rs`, lines 972-1017](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L972-L1017)). Doctor computes the hash the same way and counts a handler only when its approval matches. A copy that you haven't approved, or that changed after you approved it, counts no run. When one fire runs it beside another copy, it gets a `[warn]` line that says Codex starts the command twice once both copies are approved. Doctor also counts no run when it can't compute the hash with certainty, such as for a `statusMessage` written as a TOML date. Doctor doesn't see `--dangerously-bypass-hook-trust`, which starts unapproved handlers for one launch. Doctor compares each command with the commands setup would write now, character for character. It doesn't normalize whitespace or read shell syntax, so a copy of setup's own handler counts at any install path. This holds for handlers in `hooks.json` and in the `[hooks]` table of `config.toml` alike, so two approved copies that only `config.toml` holds fail doctor even when `hooks.json` is missing or holds no aelfrice handler.
  - Doctor compares the field that Codex runs. On Windows, Codex runs a handler's `commandWindows` field, which it also reads as `command_windows`, when the handler has one, and its `command` field otherwise ([`hooks.rs`, lines 150-155](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/protocol/src/hooks.rs#L150-L155), and [`discovery.rs`, lines 660-664](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L660-L664)). On other systems, Codex runs `command`.
  - Doctor compares only handlers whose `type` is `command`. Codex skips a `prompt` or `agent` handler ([`discovery.rs`, lines 782-800](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L782-L800)), and an `mcp_tool` handler calls a tool instead of a command.
- Doctor gives every other handler that contains the name of an aelfrice program, such as `aelf-session-start-hook`, in any of its strings, the `[warn]` line described later in this list, and the duplicate check doesn't fail because of it. A separate check still fails doctor when an aelfrice handler in `hooks.json` runs a program path that isn't on disk, as the bullet about older installs describes. This applies to a handler of any `type`, to a field that Codex doesn't run, such as `statusMessage`, and to a handler or a group written as an array, which Codex reads by position and in which doctor counts no run. Doctor finds the name in any case, anywhere in the string, on every system. Before it looks, it deletes each backslash-newline pair, which a shell joins into one line, and then each `'`, `"`, `\`, `` ` ``, and `^`: the quote, escape, and command substitution characters of a shell, and the escape character of `cmd.exe`. It reads no other shell syntax. So `/opt/bin/aelf-session-start-hook`, `--cmd=aelf-session-start-hook`, `env -Saelf-session-start-hook`, `${CMD:-aelf-session-start-hook}`, `aelf-session-start-hook.py`, `Aelf-Session-Start-Hook`, `aelf-sess''ion-start-hook`, and `aelf-"session"-start-hook` all count. The shortest name, `aelf-hook`, also counts inside a longer word, such as `my-aelf-hooks`.
  - The warning covers a wrapper such as `nice`, `env`, `exec`, or `python3 -c`, a `VAR=value` word, a command substitution, a redirect, a compound command, quoting, quotes or a backslash inside the name, a name split across lines by a backslash-newline, an escape such as `\n` or `\t` right before the name, and a copy of setup's command with different spacing.
  - It also covers a handler that only prints the name, such as `echo aelf-session-start-hook`, or that only checks for the program with `command -v` or `test -x`.
  - It doesn't cover a handler that builds the name at run time, such as `aelf-session-start-$X`, or that reaches the program through a link with another name.
  - By default, Codex runs each hook command as `SHELL -lc COMMAND`, where `SHELL` is the value of the `SHELL` environment variable, or `/bin/sh` when that variable isn't set ([`command_runner.rs`, lines 450-470](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/command_runner.rs#L450-L470), and [lines 499-504](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/command_runner.rs#L499-L504)). On Windows, it runs the program that the `COMSPEC` environment variable names, or `cmd.exe`, with `/C` and the command. A handler with the warning can start the program a second time while doctor exits 0.
- Two handlers count as two runs only when one fire runs both, and doctor counts a run only in a group whose matcher it reads. Doctor reads only matchers that mean the same thing to Rust's `regex` crate, which Codex compiles matchers with, and to Python's `re` module:
  - An empty, missing, or `*` matcher, which Codex reads as no matcher, and `.*`. Each matches every fire.
  - A matcher of only letters, digits, `_`, and `|`, which Codex reads as a list of exact names, so `start` matches no session source.
  - Names of letters, digits, and `_` joined by `|`, each with an optional `^` before it and an optional `$` after it, such as `^res|fork$`. A name with both matches only that value, a name with `^` matches a value that starts with it, a name with `$` matches a value that ends with it, and a name with neither matches a value that holds it.
- How doctor applies a matcher depends on the event:
  - In `SessionStart`, `PreCompact`, and `PostCompact`, the matcher matches the session source or the compaction trigger. Setup's own SessionStart handler has the matcher `startup|resume|clear|compact|fork`, so a copy of it in a group whose matcher is `fork`, `resume`, or `^res` counts beside it.
  - In `PreToolUse` and `PostToolUse`, the matcher matches a tool call by the tool's name or one of its aliases: `apply_patch` also matches `Write` and `Edit`, and `spawn_agent` also matches `Agent`. Tool names are an open set, so doctor reads a regular expression here only when each of its names has both `^` and `$`, such as `^Bash$|^Read$`. Two handlers count only when one tool call matches both, so a copy under `WebSearch` doesn't count beside setup's `Bash` group, and copies under `Write` and under `Edit` count as two runs. A group with no matcher, `*`, or `.*` counts beside every other group.
  - In the two events that Codex fires when an agent thread starts and stops, Codex reads the matcher too ([`common.rs`, lines 129-149](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/events/common.rs#L129-L149)), and doctor doesn't list the names it's tested against. Doctor reads a matcher here as it does in the tool events: groups that list exact names count together only when they share a name, and a group with no matcher, `*`, or `.*` counts beside every other group.
  - In `SessionEnd` and `PermissionRequest`, Codex reads the matcher against values that doctor doesn't list, so doctor counts no run in these events.
  - In `UserPromptSubmit` and `Stop`, Codex ignores the matcher. In those events and in the other events where Codex ignores the matcher, doctor counts every handler.
- In a group where doctor counts no run, because doctor doesn't read its matcher or doesn't list its event's values, each handler that names an aelfrice program gets a `[warn]` line that says so and why, and doctor exits 0. This covers matchers such as `^[[:alpha:]]+$`, `[a-z&&[^q]]+`, `(?i)fork`, and `\p{L}+`, which Rust's `regex` crate reads and Python's `re` module reads differently or not at all. It also covers a matcher that Codex can't compile, such as `(?=resume)` or `x{`. Codex skips such a group, logs `invalid matcher`, and runs none of its handlers. Two copies of setup's command in such groups don't fail doctor, even when Codex runs both.
  - Codex's TOML parser reads a TOML date or time, such as `matcher = 1979-05-27`, as its text, which matches no session source or tool name. Doctor counts no run in such a group, and each handler there that names an aelfrice program gets a `[warn]` line.
  - On Windows, Codex runs a `commandWindows` written as a TOML date or time as its text. Doctor counts no run for such a handler and doesn't read its `command` in that field's place.
- Doctor counts handlers only under the event names that Codex reads, the fields of `HookEventsToml` ([`hooks.rs`, lines 8-38](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/protocol/src/hooks.rs#L8-L38)). Codex skips any other key under `hooks`, such as `Notification` or `sessionstart`, so it never runs a handler there. Doctor gives a handler in a list of groups under such a key that names an aelfrice program a `[warn]` line that says Codex never runs it, and the exit code doesn't change because of it.
- Codex loads the `[hooks]` table in `$CODEX_HOME/config.toml` as well as `hooks.json`, and runs the handlers in both ([`discovery.rs`, lines 171-215](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L171-L215)). Doctor reads both for the checks in this list, so a copy of setup's handler in `config.toml` beside setup's handler in `hooks.json` counts as two runs. Doctor then adds a `[warn]` line that names the table. Doctor checks for missing aelfrice handlers, missing programs, and the earlier SessionStart matcher in `hooks.json` only, because setup writes only that file.
- Doctor scores a plain `codex` launch, which reads `config.toml` and no profile file ([`loader/mod.rs`, lines 293-341](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/config/src/loader/mod.rs#L293-L341)). Codex starts none of your handlers while its `hooks` feature is off ([`mod.rs`, lines 290-325](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/mod.rs#L290-L325)). A plain launch reads the feature, and its earlier name `codex_hooks`, from `[features]` and not from a `[profiles.NAME.features]` table ([`config/mod.rs`, lines 3559-3568](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/core/src/config/mod.rs#L3559-L3568)). Of the configuration files that set it, doctor reads only `config.toml` in `$CODEX_HOME`, as a later item in this list describes. Codex refuses to start when `config.toml` has a top-level `profile` key, which it no longer supports ([`config/mod.rs`, lines 3545-3552](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/core/src/config/mod.rs#L3545-L3552)), or a `[features]` table, at the top level or in a `[profiles.NAME]` table, that `FeaturesToml` can't read ([`lib.rs`, lines 959-999](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/features/src/lib.rs#L959-L999)):
  - `FeaturesToml` reads most keys as `true` or `false`, so a key set to another value makes Codex refuse to start.
  - It reads `tool_registry` as a table only, so `tool_registry = true` makes Codex refuse to start too ([`feature_configs.rs`, lines 25-34](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/features/src/feature_configs.rs#L25-L34)).
  - It reads each other key that it reads into a field of its own, such as `code_mode` and `network_proxy`, as `true`, `false`, or a table of that feature's settings ([`lib.rs`, lines 1094-1099](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/features/src/lib.rs#L1094-L1099)). Doctor checks `true` and `false` against these types and checks no table. When one of these keys, or `tool_registry`, holds a table or any other value that isn't `true` or `false`, such as `[features.code_mode]` or `token_budget = 100000`, doctor can't tell whether Codex starts, so it counts no duplicate run and says so in a `[warn]` line.

  When the feature is off or Codex refuses to start, doctor counts no duplicate run and says why in a `[warn]` line. Each handler that names an aelfrice program, other than setup's own handler in the group setup writes, gets a `[warn]` line too.
  - `codex --profile NAME` also reads the profile file `$CODEX_HOME/NAME.config.toml`, and refuses to start while `config.toml` has a `[profiles.NAME]` table. A profile file whose `[features]`, `[hooks.state]`, or `[hooks]` changes which hooks that launch starts, whose `[features]` holds a value that doctor doesn't check, or that doctor can't read, gets a `[warn]` line that names the profile. So does a `[profiles.NAME.features]` table that sets the `hooks` feature, which no launch reads. These lines don't change the exit code, and a failure for the plain launch stays. When doctor can't list the files in `$CODEX_HOME`, it reads no profile file and says so in a `[warn]` line.
- Codex skips a handler whose `[hooks.state]` entry sets `enabled = false` ([`discovery.rs`, lines 1019-1021](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L1019-L1021)). Codex keys the entry `FILE:EVENT:GROUP:HANDLER`, such as `/home/me/.codex/hooks.json:session_start:1:0`, where `FILE` is the absolute path of `hooks.json` or `config.toml`, `EVENT` is the event name in snake case, and `GROUP` and `HANDLER` count from 0 ([`lib.rs`, lines 125-135](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/lib.rs#L125-L135)). Doctor reads these entries in `config.toml`, the file a plain launch reads. It trims Unicode whitespace from each end of the key, as Rust's `str::trim` does, which keeps the control characters U+001C to U+001F. Codex spells the path of the Codex home in the key one way ([`lib.rs`, lines 13-62](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/utils/home-dir/src/lib.rs#L13-L62)). When `CODEX_HOME` is set, the path is canonical, as `realpath(3)` gives it, which on macOS also changes the letter case of each folder name to the case on disk. When `CODEX_HOME` isn't set or is empty, the path is `~/.codex` as written, made absolute with `.` and `..` removed. Doctor reads `enabled` and the approval under that spelling only, and on macOS calls `realpath(3)` as Codex does. When that call fails, doctor uses Python's `os.path.realpath`, which keeps the letter case of the `CODEX_HOME` value, and says so in a `[warn]` line. So when `CODEX_HOME` isn't set and `~/.codex` is a symbolic link, the key starts with the `~/.codex` path. When `CODEX_HOME` names a folder under `/tmp` on macOS, the key starts with `/private/tmp`, and when it spells a folder `/Users/me/Codex` as `/Users/me/codex` on a macOS volume that ignores letter case, the key starts with `/Users/me/Codex`. Codex doesn't expand a `~` at the start of `CODEX_HOME`: it reads the value as a path under the folder you start Codex in, and refuses to start where no such folder exists ([`lib.rs`, lines 24-50](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/utils/home-dir/src/lib.rs#L24-L50)). aelfrice expands it, so for such a value doctor counts no duplicate run and says why in a `[warn]` line. An entry under another spelling disables nothing and approves nothing. When two entries trim to the same key, the later `enabled` and the later `trusted_hash` win, as in Codex. A disabled handler that names an aelfrice program gets a `[warn]` line, and so does a copy of setup's command that no entry approves, with the key Codex reads its approval from.
- Doctor reads only the user layer of Codex's configuration, `hooks.json` and `config.toml` in `$CODEX_HOME`, and the profile files there. Codex merges other layers into the configuration that it loads ([`config/mod.rs`, lines 2295-2296](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/core/src/config/mod.rs#L2295-L2296)), and it loads the hooks of each layer ([`discovery.rs`, lines 171-215](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L171-L215)). The system `/etc/codex/config.toml` is one layer ([`loader/mod.rs`, lines 273-291](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/config/src/loader/mod.rs#L273-L291)). The `.codex/` folder of a trusted project that you start Codex in is another ([`loader/mod.rs`, lines 412-441](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/config/src/loader/mod.rs#L412-L441)). A `--config` override and managed requirements change the configuration too. Any of these can add handlers or turn hooks off without doctor seeing it, so what Codex starts in a folder can differ from what doctor counts.
- Codex rejects all of `hooks.json`, or all of the `[hooks]` table in `config.toml`, when one part of it has a shape that Codex can't read, and then it loads no hook from that file ([`discovery.rs`, lines 390-451](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L390-L451)). For example, Codex rejects a handler with no `type`, or with a `type` other than `command`, `mcp_tool`, `prompt`, or `agent`, a `command` handler with no `command`, a handler with both `commandWindows` and `command_windows`, a `matcher` that isn't a string, a field written twice, and a top-level key in `hooks.json` other than `description` and `hooks`. Doctor counts no handler in such a file, prints a `[warn]` line that says Codex loads no hook from it and why, and exits 1, as it does for a `hooks.json` that isn't valid JSON. Doctor doesn't check the shape of a group or a handler written as an array, so it can't tell whether Codex loads a file that holds one under an event name. In such a file, doctor counts no duplicate run, prints a `[warn]` line that names the first array form, and its last check warns about each program name in the file, setup's own commands included.
  - In `config.toml`, Codex reads the `hooks` value as a table of events beside a `state` table of approvals ([`hook_config.rs`, lines 21-38](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/config/src/hook_config.rs#L21-L38)). So it also rejects a `hooks` value that isn't a table, such as one written as an array, and a `[hooks.state]` entry that isn't a table, has an `enabled` that isn't `true` or `false`, or has a `trusted_hash` or `trusted_content_hash` that isn't a string. Doctor exits 1 on these as well.
  - serde_json, which Codex reads `hooks.json` with, rejects the file when an array or object nests 128 levels deep, counting the top-level object as level 1 ([serde_json 1.0.149 `src/de.rs`](https://docs.rs/crate/serde_json/1.0.149/source/src/de.rs), `check_recursion`). It counts no depth in a value that Codex skips, the value of a key under `hooks` that isn't an event name or of a group key other than `matcher` and `hooks`, so such a value loads at any depth. Doctor exits 1 on the first and reads the second, minus the part past level 127, which it names in a `[warn]` line.
  - serde_json also rejects a lone UTF-16 surrogate escape, such as `"\ud800"`, in any key or value that Codex reads as text, and accepts one in a value that Codex skips. Doctor exits 1 on the first and reads the second.
  - The TOML parser that Codex reads `config.toml` with rejects the whole file when arrays and inline tables nest more than 80 levels deep, or when a key, in a table header or anywhere else, has more than 80 dotted parts ([toml 0.9.11 `src/de/parser/mod.rs`](https://docs.rs/crate/toml/0.9.11/source/src/de/parser/mod.rs), `LIMIT`). It also rejects the whole file over an integer that a 64-bit signed integer can't hold, such as `9223372036854775808` or `0xffffffffffffffff`, and over a float that overflows, such as `1e400`, while it reads `inf` ([`src/value.rs`](https://docs.rs/crate/toml/0.9.11/source/src/value.rs), `visit_u64`, and [`src/de/parser/devalue.rs`](https://docs.rs/crate/toml/0.9.11/source/src/de/parser/devalue.rs), `DeFloat::to_f64`). Codex then refuses to start, so doctor counts no duplicate run in any file, says so in a `[warn]` line, and exits 1.
- Each `[warn]` line about a handler names the event, the handler's path in its file, such as `hooks.SessionStart[1].hooks[0]`, and the handler's command. For a handler in a group where doctor counts runs, the line also names the command setup writes for the program. When one fire, read by the matcher rule above, runs both that handler and another handler of the program, the line says that the handler may also run the program, and tells you to keep setup's handler and delete this one. Otherwise, the line says that doctor can't tell whether the handler runs the program, and that setup's own handler is the supported form. The lines for a handler under a key that isn't an event name, in a group whose matcher doctor doesn't read, in an event whose values doctor doesn't list, while Codex starts no hook, and for a disabled handler say why doctor counts no run there instead, and don't name setup's command.
- After these checks, doctor reads every string in `hooks.json`, in all of `config.toml`, and in each profile file: keys and values, in any object or array, to a depth of 256 levels and up to 100,000 values in each file, and it says so in a `[warn]` line when it stops before the end. Strings outside the `[hooks]` table count too, such as `notify`, a command that Codex runs after each turn ([`config_toml.rs`, lines 246-248](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/config/src/config_toml.rs#L246-L248)), `profiles.NAME.notify`, and `mcp_servers.NAME.command` and `args`. That includes a `hooks` value written as an array, which Codex reads by position, a value that isn't a list under a key that isn't an event name, the first value of a field that Codex doesn't read and that the file writes twice, and `description`. Each string that names an aelfrice program gets a `[warn]` line that names the file and the string's path, such as `hooks.Bogus.hooks[0].command`, unless it's the command of setup's own handler in the group setup writes, or a failure or another `[warn]` line already covers it. The exit code doesn't change because of it.
- A copy of a command that setup writes, under an event where setup writes no handler of that program, such as setup's `aelf-hook` command under `SessionStart`, counts as a run of the program in that event. Doctor also gives it a `[warn]` line that says setup writes no such handler there.
- A handler that an older install wrote, at another install path or without an argument that a later release added, isn't byte-identical to a command setup writes now, so it gets one of these `[warn]` lines instead of counting as a run. When that handler is in `hooks.json`, under an event that setup writes, the line also says that a rerun of `aelf setup --host codex` replaces it with setup's handler. When its program path isn't on disk any more, for example after you delete the older install's virtual environment, doctor also reports a failure that names the missing program and exits 1. That check is separate from the duplicate check, and a rerun of setup fixes both.
- Doctor reads `hooks.json` with Python's `json` module and `config.toml` with Python's `tomllib` module, which read some files differently from Codex:
  - Codex reads TOML 1.1 syntax, such as a trailing comma in an inline table (`x = { a = 1, }`), which `tomllib` rejects. Doctor then prints a `[warn]` line that says it couldn't parse `config.toml`, and counts no duplicate run in any file, because it can't read the approvals in that file. It gives each line of the file that names an aelfrice program a `[warn]` line with the line number, and finds the names as it does in a string, after it deletes each backslash-newline pair and each `'`, `"`, `\`, `` ` ``, and `^`. It reads a profile file that `tomllib` rejects the same way.
  - Codex reads each of these files as UTF-8 text and doesn't translate line endings ([`discovery.rs`, lines 390-419](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/hooks/src/engine/discovery.rs#L390-L419), and [`lib.rs`, lines 684-695](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/file-system/src/lib.rs#L684-L695)), and so does doctor. A `hooks.json` that isn't UTF-8 loads no hook, so doctor names the first byte that isn't UTF-8 in a `[warn]` line and exits 1, as it does for a `hooks.json` that isn't valid JSON. A `config.toml` that isn't UTF-8 makes Codex refuse to start, so doctor counts no duplicate run, says so in a `[warn]` line, and exits 1. A profile file that isn't UTF-8 gets a `[warn]` line that names the profile.
  - Python's `json` module reads `NaN`, `Infinity`, and `-Infinity`, which serde_json rejects at any place in `hooks.json`, a value that Codex skips included ([serde_json 1.0.149 `src/de.rs`](https://docs.rs/crate/serde_json/1.0.149/source/src/de.rs), `ignore_value`). Doctor finds them at any place too, in a value nested past level 127 included, names the place in a `[warn]` line that says Codex loads no hook from the file, counts nothing in the file, and exits 1.
  - Python's `json` module reads a number that a 64-bit float can't hold, such as `1e400`, or an integer of 309 digits or more, as infinity or as an integer. serde_json rejects the whole file over such a number in a value that Codex reads, such as a handler's field, and loads the file when the number is in a value that Codex skips (`parse_long_integer`, `f64_from_parts`, and `ignore_value` in the same file). So doctor can't tell whether Codex loads a `hooks.json` with such a number. It counts no duplicate run in that file, and its `[warn]` line names the place of the first such number. Doctor doesn't raise on an integer of more than 4,300 digits, which Python's `int` doesn't read. In `config.toml` or a profile file, doctor reads such an integer as out of range, which makes Codex reject the file.
  - Codex reads the `[hooks]` table as one field of its whole configuration ([`config_toml.rs`, line 500](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/config/src/config_toml.rs#L500)), and its configuration load returns the error when that field doesn't parse ([`config/mod.rs`, line 2296](https://github.com/openai/codex/blob/69231e119bfca0b14453515e97d99688ce1c47c9/codex-rs/core/src/config/mod.rs#L2296)). So a `[hooks]` table that Codex rejects makes Codex refuse to start. Doctor's `[warn]` line says so, and doctor counts no duplicate run in any file.

To move off the wrapper:

1. Turn on the summary line: add `summary_line = true` under `[hook]` in the project's `.aelfrice.toml`, or set `AELFRICE_HOOK_SUMMARY_LINE=1` in the environment that starts Codex.
2. Run `aelf setup --host codex`.
3. In `$CODEX_HOME/hooks.json`, under `SessionStart`, delete the handler whose `command` runs the wrapper. Keep the handler whose command ends in `aelf-session-start-hook --host codex`.
4. Delete the wrapper script.
5. Open `/hooks` in a Codex session and approve the changed entries.
6. Run `aelf doctor --host codex` to confirm the wiring. Check that it exits 0 and that no `[warn]` line names a `SessionStart` handler. Then start a session and check for the `aelfrice:` line.

#### `$CODEX_HOME` (unreleased)

Codex reads its configuration from `$CODEX_HOME` when that variable has a non-empty value, and from `~/.codex` otherwise. All three verbs resolve the home directory the same way (#1427), so each of these configurations is wired where that Codex reads:

- an isolated profile
- a continuous integration (CI) runner
- a portable install
- two side-by-side Codex configurations

`aelf doctor --host codex` prints the resolved home on its first line, so you can see which directory the command inspected.

A `$CODEX_HOME` that doesn't exist is an error, not a condition for a fallback, and a `$CODEX_HOME` that names something other than a directory, or that aelfrice can't read, such as one under a folder you can't search, is an error too. In each case, setup, doctor, and unsetup refuse the operation, create nothing, and leave `~/.codex` unchanged.

Both refusals match how Codex behaves. `CODEX_HOME=/nowhere codex mcp list` reports `CODEX_HOME points to "/nowhere", but that path does not exist` and then stops.

aelfrice doesn't create the directory for you. Codex refuses to start against such a configuration home, so a report of "setup succeeded" would be false. Create the directory, correct the variable, or unset the variable.

`$CODEX_HOME` does **not** move the `$aelf-*` agent skills. `~/.agents/skills/` is a standard path across agents, and other agents keep their skills in the same directory.

#### Shared `hooks.json` (unreleased)

`hooks.json` belongs to Codex, not to aelfrice. One document holds your own entries, another installer's entries, and aelfrice's.

Setup and unsetup do three things to that document:

1. They take an advisory lock on a sibling `hooks.json.lock`.
2. They re-check the content hash of the file immediately before they commit.
3. They replace the file atomically (#1428).

An aelfrice run therefore merges a concurrent edit instead of overwriting it.

`unsetup` does none of this against a home that has no `hooks.json`; run `unsetup` there and it creates nothing, not even the lock file.

Where a `hooks.json` was present, the zero-byte `hooks.json.lock` stays on disk. That's intentional. If another process holds that lock, deleting the lock puts two writers on two inodes with no mutual exclusion at all.

If the file keeps changing across three attempts, the command refuses and writes nothing. Run the command again.

Two refusals are new and intentional. aelfrice leaves a non-object `hooks` value byte-for-byte unchanged and reports it, and it does the same with a non-list value on an event it installs into. A `null` value gets the same treatment, because `null` is a value at that position and not an absence.

aelfrice can merge only into the documented `{"hooks": {"<Event>": [...]}}` shape. Merging into any other would delete structure it didn't write. Correct the file, or pass `--force` to replace it.

One window stays open. A process that takes no lock can still replace the file in the instant between the final check of aelfrice and the rename. Closing that window needs a mutation protocol from Codex, and Codex doesn't offer one today.

## 3. Verify wiring

```bash
aelf doctor
```

`aelf doctor` runs two checks, one after the other:

1. Hook resolution. The check compares every `command` in `settings.json` against `$PATH`. It also reports the stale `bash <missing>.sh 2>/dev/null || true` wrappers left by older installs.
2. The structural graph audit. The audit covers orphan threads, full-text search version 5 (FTS5) sync, contradictions between locked beliefs, and corpus volume.

`aelf doctor` exits 1 on any structural failure, so CI can gate on the command. An empty store on a new project is normal. The corpus-volume warning fires only when the project is at least 7 days old.

To run one half only, use one of these commands:

```bash
aelf doctor hooks      # hook resolution only
aelf doctor graph      # structural auditor only
```

### Duplicate hook entries (v4.2.0+)

`settings.json` is a shared global configuration file with several
independent writers: the harness itself, `aelf setup`, auto-install,
and your own edits. A migration of the host settings, a merge of
dotfiles, or an edit by hand can leave two entries for the same
aelfrice hook. Both entries fire on every event, and the only symptom
is that prompts get slower for no visible reason. Nothing is broken:
both paths resolve, and every check passes.

`aelf doctor` now reports these entries, and `--fix` collapses them:

```bash
aelf doctor hooks              # reports duplicates (read-only)
aelf doctor hooks --fix        # collapses them, keeping the first of each
```

The collapse touches only the `aelf-*` entries. The hooks that you
wrote yourself stay exactly as they are, including a hook you listed
twice on purpose. aelfrice counts the duplicates separately from the
stale-path prune, because the two repairs answer different questions.
A pruned entry pointed at a venv that no longer exists; a collapsed
entry resolved correctly and was installed twice.

### Hook timeouts

Every hook that aelfrice installs has an explicit `timeout` (v4.2.0+).
The bundled manifest declares the timeout for each hook. A hook that
gates a user-visible action gets 15s. The bulk-ingest hooks get 30s.
The timeout is what bounds the "a hook must never block your prompt"
contract at the host level. Without the timeout, a hook can
wait for the SQLite write lock of another session, and then stall for
the default of the host rather than the budget of aelfrice.

The budgets have headroom over the real worst case, rather than
sitting close to it. A cold-start retrieval on a store of approximately 46k beliefs
measured less than 3s. That retrieval had to rebuild its Best Matching
25 (BM25) sidecar from the start. If your store is very large, or if
your disk is slow, override the budget with
`aelf setup --timeout <seconds>`.

`aelf health` and `aelf stats` are still callable as back-compatible aliases. The default `--help` output hides them, and `aelf --help --advanced` lists them. The canonical replacement for `health` is `aelf doctor graph`, the structural auditor. The canonical replacement for `stats` is `aelf status`. Run `aelf status` for the counts.

## 4. Onboard a project

Start your agent in the project's folder, and then run this command in the agent:

```text
/aelf:onboard .
```

On Codex, run `$aelf-onboard .` instead. Both commands classify with your agent's model and need no API key. When the store is empty, `aelf setup` prints the command for the host you set up. If you run `aelf setup` in a folder that has no git work tree and no `[store]` marker, it tells you to open a project first. A folder with a `[store]` marker has its own store, so `aelf setup` treats it as a project without git.

To scan from a terminal without your agent's model, run the command from the project root:

```bash
cd <project-root>
aelf onboard .
```

`aelf onboard .` walks the project and ingests the structural facts as candidate beliefs. The command reads the filesystem, the git log, and the Python abstract syntax tree (AST). It classifies with the regular-expression classifier unless you configure the direct-API classifier. It typically takes less than one second on a project of 50k lines of code. A second run is idempotent, because the command dedupes on `(source, sentence)`.

## 5. Lock the rules you care about

```bash
aelf lock "never push to main; use scripts/publish.sh"
aelf lock "all commits SSH-signed with ~/.ssh/id_rrs"
aelf locked                          # list what's locked
```

Locked beliefs are the beliefs that survive: aelfrice always returns them at L0, and passive feedback doesn't move a locked posterior.

Restart Claude Code. The next prompt that mentions "push" arrives with your rules already injected.

## Database

The store is a SQLite database. The path resolution order is:

1. `$AELFRICE_DB` — the explicit override. aelfrice honors `:memory:`, which is useful for tests.
2. The nearest `[store]` table in an `.aelfrice.toml` at or above `cwd`, up to the root of the git work tree ([#1796](https://github.com/robotrocketscience/aelfrice/issues/1796)). Use it to give a folder of a repository its own store. See [`[store]`](CONFIG.md#store-1796).
3. `<git-common-dir>/aelfrice/memory.db` — used when `cwd` is inside a git work-tree (v1.1.0+). The worktrees of one repo share a single DB through `--git-common-dir`. Git doesn't track `.git/`, so the brain graph never crosses the git boundary.
4. `~/.aelfrice/memory.db` — the fallback for a directory that isn't in a git work-tree.

To see which store applies and which rule chose it, run `aelf status`.

To pin a project, run `export AELFRICE_DB=/abs/path/.aelfrice.db`. This method works well with `direnv`. A `[store]` table needs no environment, so it also applies to sessions you didn't start from that folder.

### Migrating from v1.0.x

v1.0.x kept a single global DB at `~/.aelfrice/memory.db`. v1.1.0 resolves the path for each project. To port the beliefs forward, use these commands:

```bash
cd <project-root>
aelf migrate                # dry-run; reports what would copy
aelf migrate --apply        # actually copy filtered beliefs
aelf migrate --apply --all  # copy every belief from the legacy DB
aelf migrate --from /alt/path/memory.db --apply
```

`aelf migrate` opens the source DB read-only with the SQLite `mode=ro` URI. The project-mention filter is the default, and it restricts the copy to the beliefs that name the active project. `--all` skips the filter. `aelf migrate` is idempotent on a second run.

### Batch ingest of historical sessions

If you have prior Claude Code sessions sitting at `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`, you can backfill them:

```bash
aelf ingest-transcript --batch ~/.claude/projects/
aelf ingest-transcript --batch ~/.claude/projects/ --since 2026-01-01
```

The command detects the JSONL format line by line, so it handles both the transcript-logger output of aelfrice and the internal session shape of Claude Code. It's idempotent on re-run.

> **Privacy.** A session JSONL file can hold pasted secrets, customer data, or any other text that you typed in the chat. Batch ingestion brings all of that content into the local belief graph. The v1.2 ingest path has no scrubber for personally identifiable information (PII). Review the sessions before you backfill them.

## Hooks installed by `aelf setup`

Bare `aelf setup` installs the v1.2.0 auto-capture pipeline, together with the read-side `UserPromptSubmit` retrieval hook:

| Hook | Event(s) | Default | What it does |
|---|---|---|---|
| UserPromptSubmit retrieval | `UserPromptSubmit` | always | injects the matched beliefs as an `<aelfrice-memory>` block |
| transcript-ingest | `UserPromptSubmit` + `Stop` + `PreCompact` + `PostCompact` | **on** | logs every turn to a per-project JSONL file. Once 12 new lines have been logged, `Stop` ingests the sentences of your new prompts into beliefs and edges, with code blocks removed. A sentence keeps its inline code and URLs, but they don't affect whether aelfrice stores the sentence or which type it gets. It skips noise, hedged ideas such as "maybe we could", sentences that end in a question mark, sentences under 10 characters once their inline code and URLs are removed, such as ``Use `widget.py`!``, and the agent's replies. Some shapes that commonly hold secrets stay removed, and that isn't a general scanner; see [Secrets in your prompts](PRIVACY.md#secrets-in-your-prompts). Your prompts and the agent's replies each count as a line, so that's about every six prompts, and the count spans every session in the project (`AELFRICE_INGEST_STOP_FLUSH_TURNS` changes it). PreCompact rotates the file, then ingests it too |
| commit-ingest | git `post-commit` in the current repository | **on** | after each new commit, the triple extractor runs on its message in the background. On a lowercase phrase such as "X supersedes Y", with a word on each side of it on the same line, aelfrice stores up to six words from each side, from that line only, as a belief, with an edge between them. "SUPERSEDES" doesn't count. A side made only of function words stores nothing for that match: determiners, pronouns, relative words, quantifiers, conjunctions, prepositions, auxiliaries and modals, negators, and adverbs such as "also" and "just". A word of two or more letters in all capitals, such as "IT" or "WHO", counts as a name, so it stores. A capitalized word that matches one of these, such as the code names "None" and "Any" or the name "Will", counts as a function word, so it doesn't store. A word with a hyphen or underscore inside it, such as "one-shot" or "all_of", counts as one word and stores. A side can't start after an apostrophe, so "it's" doesn't store "s". Those words might not form a complete phrase, such as "the old", and a phrase that wraps to the next line is cut at the line break. The edge also keeps about 80 characters of the message, which can include other lines. From a message with no such phrase, aelfrice stores nothing. Commits replayed by `git rebase`, `git revert` commits, and commits from `git merge` or `git pull` are skipped. Since #1698 this is a git hook, not a `PostToolUse:Bash` entry, so run `aelf setup` inside each repository whose commit messages you want read. Setup won't edit a `post-commit` it can't restore exactly, or one in a shared or tracked `core.hooksPath`; it prints the lines to add instead. Removing aelfrice takes the block out of the current repository only, so remove it from other repositories by hand |
| session-start | `SessionStart` | **on** | new sessions open with the L0 locked beliefs already injected |
| stop-lock-prompt | `Stop` | **on** | prompts you to lock this session's correction-class (#582) and directive (#1315) beliefs. It also labels the session's new core candidates for the core admission gate (#1638) in a detached `claude -p` run, with nothing added to your conversation; see [`[core_gate]`](CONFIG.md#core_gate-1638) to turn that off (#1741) |
| search-tool | `PreToolUse:Grep` / `Glob` / `WebSearch` / `WebFetch` | **on** (v3.0.1+) | checks the belief store before the agent's own Grep, Glob, WebSearch, or WebFetch call runs |
| search-tool-bash | `PreToolUse:Bash` | **on** (v3.0.1+) | checks the belief store before a shell grep, rg, find, fd, or ack fires |
| pre-issue-guard | `PreToolUse:Bash` | **on** (v3.4.0+) | blocks `gh issue create` when the title overlaps an existing issue or a shipped commit at 0.5 Jaccard or above (#941) |
| claude-memory-mirror | `PostToolUse:Write` / `Edit` / `MultiEdit` | **on** (v3.7.0+) | mirrors the host claude-memory fact-file writes one-way into the belief graph (#985). Either `AELFRICE_MIRROR_CLAUDE_MEMORY` or `[memory] mirror_claude_memory` enables the hook, and since v4.0 (#1089) so does the per-project consent sentinel, which the one-shot reconcile writes at the first `aelf setup` if the project already has claude-memory files. A project that's set up after its first memory file therefore mirrors by default. If setup ran before any memory existed, or the store directory was removed, the hook stays installed but the mirror stays off; `aelf doctor` warns, and `aelf reconcile-claude-memory` records consent. Setup and reconcile both work from a linked worktree, which uses the main checkout's memory (#1657). An explicit env `0` or TOML `false` always wins over the sentinel; that's the opt-out |
| agent-context | `PreToolUse:Agent` / `Task` | **on** | dispatched subagents inherit the L0 locked beliefs and the task-relevant ones through prompt injection. The hook is a fail-open passthrough. Set `AELFRICE_AGENT_CONTEXT=0` to disable the injection, or pass `--no-agent-context` to opt out (#1068) |
| rebuilder | `PreCompact` (installed) — block ships on `SessionStart(source="compact")` | off | the retrieval-curated context rebuilder (augment-mode, v1.4 alpha). `--rebuilder` installs a `PreCompact` entry that does the trigger-mode bookkeeping only and injects nothing. The block itself ships *after* the compaction, on the default-on `SessionStart` hook that's already present ([#1031](https://github.com/robotrocketscience/aelfrice/issues/1031)) |

To opt out of one hook, use the matching option. The opt-out persists across upgrades in `~/.aelfrice/opt-out-hooks.json`.

```bash
aelf setup --no-transcript-ingest      # skip the four transcript-logger hooks
aelf setup --no-commit-ingest          # skip the commit-message git hook
aelf setup --no-session-start          # skip the SessionStart locked-belief injection
aelf setup --no-stop-hook              # skip the Stop lock-prompt hook
aelf setup --no-search-tool            # skip the PreToolUse:Grep|Glob|WebSearch|WebFetch hook
aelf setup --no-search-tool-bash       # skip the PreToolUse:Bash hook
aelf setup --no-pre-issue-guard        # skip the issue-dup detection guard
aelf setup --no-claude-memory-mirror   # skip the claude-memory → belief-graph mirror hook
aelf setup --no-agent-context          # skip the PreToolUse:Agent worker-context injection
```

The SessionStart recap line ("N beliefs written since last session", v3.5+, #934) has no manifest entry of its own. The SessionStart hook prints the recap line. `aelf setup --no-sessionstart-recap` suppresses the recap line at install time, and `opt-out-hooks.json` doesn't persist that choice.

To opt in to the hooks that are off by default, use this command:

```bash
aelf setup --rebuilder                 # post-compaction context rebuilder (alpha)
```

`aelf unsetup` mirrors `aelf setup`. A bare invocation removes every default-on hook. The `--no-*` flags suppress the removal of the matching hook.

All hooks fail open. Every failure path returns exit 0, because a hook problem must never break a prompt or a commit. The pre-issue-guard is the one intentional exception to the non-blocking behavior: it exits 2 to block a duplicate `gh issue create`. To bypass the guard, set `ALLOW_DUP_ISSUE=1` in the host's environment. The failure paths of the guard still exit 0.

### Self-installing hook manifest (v3.0+)

`src/aelfrice/data/hook_manifest.json` declares the list of default-on hooks above, and that file ships in the wheel. The first `aelf <cmd>` invocation after a fresh install does two things, and so does the first `aelf <cmd>` invocation after a bare `uv tool upgrade aelfrice`:

1. It reconciles the installed manifest version against `~/.aelfrice/installed-manifest-version`.
2. It merges any new entries into `~/.claude/settings.json` automatically.

You no longer have to remember to run `aelf setup` again after a bare upgrade with the package manager. The hooks that a newer release adds arrive without that step.

What auto-install does:

* In the usual path the stamp equals the installed version. That path is one stat and one short file read. It runs no JSON parse. It does not read settings.json.
* On a mismatch it takes an `fcntl` exclusive lock on `~/.aelfrice/.auto-install.lock`, so concurrent `aelf` invocations can't race on the merge.
* It reuses the same install primitives that `aelf setup` calls. The shape of settings.json on disk is byte-identical.
* It adds only the entries that the manifest claims by basename. It preserves everything that you added to settings.json by hand.
* It respects the opt-outs. If you ever ran `aelf setup --no-transcript-ingest`, `~/.aelfrice/opt-out-hooks.json` persists that choice, and the choice survives upgrades. To rescind the opt-out, run `aelf setup` again without the `--no-*` flag.
* It prints a single stderr line when it added entries: `aelfrice: hooks updated to v3.5.0 (was v3.4.0) — added: pre_issue_guard`.

Opt-out controls:

```bash
export AELFRICE_NO_AUTO_INSTALL=1   # power user: I manage settings.json by hand
aelf setup --no-stop-hook           # disable one hook; persists across upgrades
```

`aelf doctor` flags a hook command that is broken or that doesn't resolve. `aelf doctor` also reports when one of the v2.1 auto-capture hooks is missing from settings.json: transcript-ingest, session-start, and stop-hook. It doesn't check the commit-ingest git hook; look for the `aelfrice commit-ingest` block in the repository's `post-commit` hook.

`aelf doctor` doesn't reconcile settings.json against the manifest. That's why `aelf doctor` doesn't flag the newer manifest hooks, search-tool and pre-issue-guard, when they're absent. The auto-installer is the reconciliation path.

> **Privacy note.** transcript-ingest is on by default, so every turn that you type goes into the per-project SQLite DB, at `Stop` after every 12 logged lines (about six prompts) and at the `PreCompact` rotation. The DB is local-only: it uses no network and sends no telemetry. See § "What you get for free" in the README and [the privacy documentation](PRIVACY.md).
>
> The JSONL file has no PII scrubber. If you paste secrets, customer data, or anything else that you don't want indexed in the chat, opt out with `--no-transcript-ingest`. Then use `aelf lock` or `aelf onboard` for explicit ingestion only.

### Legacy-schema detection + auto-migrate (`aelf doctor`, v3.0+)

`aelf doctor` scans every per-project DB under `~/.aelfrice/projects/*/memory.db`. Where a DB uses the pre-v1.x schema, `aelf doctor` migrates it in place. A DB on that schema has no `origin` column on the `beliefs` table.

A DB on the old schema can't participate in the v2.x and v3.0 lifecycle, because the columns that track the lifecycle are absent. That lifecycle includes `agent_remembered`, `user_validated`, the calibrated weights, `aelf:promote`, and the federation `scope`.

Auto-migrate (v3.0+, [#593](https://github.com/robotrocketscience/aelfrice/issues/593)) works like this. For each legacy DB that it detects, doctor does three things:

1. doctor renames `memory.db` to `memory.db.pre-v1x.bak` with an atomic POSIX rename.
2. doctor runs the existing `migrate()` core with `copy_all=True, apply=True` against the backup.
3. doctor writes a fresh modern-schema DB back at the original path.

Each copied belief gets `origin=ORIGIN_UNKNOWN`. doctor preserves the backup verbatim and never overwrites it. If a stale `<path>.pre-v1x.bak` from an earlier failed run is already present, the migration stops for that DB, and `report.failed_migrate_dbs` surfaces the failure. doctor doesn't overwrite the recoverable state.

The success path emits one line per DB:

```
migrated ~/.aelfrice/projects/2e7ed55e017a/memory.db: 35,332 beliefs, 412ms (backup at ~/.aelfrice/projects/2e7ed55e017a/memory.db.pre-v1x.bak)
```

doctor skips an empty DB (zero rows) silently. A failure prints a residual warning that points at the manual `aelf migrate --from <path> --apply` path for that specific DB.

For a manual migration — a DB outside the standard `~/.aelfrice/projects/` tree, for example — use these commands:

```bash
aelf migrate --from /alt/path/memory.db          # dry-run
aelf migrate --from /alt/path/memory.db --apply  # write
```

### Concurrent installs (v4.2.0+)

`settings.json` has more than one writer:

- `aelf setup`
- `aelf unsetup`
- `aelf doctor --fix`
- the automatic hook merge that runs on CLI invocations
- the host itself, each time you approve a permission or change a setting

Since [#1161](https://github.com/robotrocketscience/aelfrice/issues/1161) every aelfrice mutation of that file takes an exclusive lock on a sibling `settings.json.lock`, and then writes once. aelfrice's own writers can no longer overwrite each other. You might see two consequences:

```
setup aborted: another aelfrice process is writing settings.json
(could not acquire ~/.claude/settings.json.lock within 10.0s). Nothing
was changed; re-run `aelf setup`.
```

Another aelfrice process held the lock longer than the wait. aelfrice wrote nothing. Run the command again.

```
setup aborted: ~/.claude/settings.json was modified by another process
while aelfrice was writing it; no changes were made. Re-run the command.
```

Something that doesn't take the aelfrice lock replaced the file during the install. In practice that something is the host. aelfrice doesn't commit over the new file, because a commit would discard what the host just wrote. Run the command again. Every change that aelfrice makes to this file is convergent, so a second run is safe.

You can't make the host take the aelfrice lock, so this check is a detector and not a cure. Before the check, a setting was lost silently. The check now gives you a message that tells you to try again.

### Incomplete store migrations (v4.2.0+)

Opening the store runs a set of one-shot schema migrations. Each migration is stamped as complete only after it finishes its work.

Before [#1161](https://github.com/robotrocketscience/aelfrice/issues/1161) a migration that raised an error took the whole store with it. The completion marker was never written, so the next open re-ran the same pass and failed the same way. Every entry point opens the store: the CLI, the hooks, and MCP. The only way back was to edit the SQLite database by hand.

A migration that can't finish no longer stops the store from opening. aelfrice records the failure and skips the pass. The store keeps its pre-migration shape, and reads and writes work. The completion marker stays unset, so aelfrice retries the migration on every later open. An upgrade to a build that fixes the migration thus repairs the store automatically.

`aelf doctor` is where that state surfaces:

```
store migration(s) INCOMPLETE — the store opens and is usable, but one or more one-shot migrations could not finish:
  _maybe_consolidate_content_hash_duplicates: IntegrityError('UNIQUE constraint failed: edges.src, edges.dst, edges.type')
fix: these retry automatically on every open, so upgrading (`aelf upgrade`) is the first thing to try. If the same
migration keeps failing, report the error above — the store will keep working in the meantime.
```

Try `aelf upgrade` first. If the same migration keeps failing, include the error text above in a report. Nothing urgent is required, because the store stays usable in the meantime.

### Pruning dormant per-project DBs (`aelf doctor --prune-dormant`, v3.0+)

Some per-project DBs hold beliefs from projects that you worked on briefly with an older aelfrice version and then abandoned. Nothing migrates those DBs. Nothing touches them. They stay on disk. `aelf doctor --prune-dormant` lists the DBs whose `memory.db` mtime is older than `--idle-days`, which defaults to 30 days. The command lets you delete those DBs one at a time.

```bash
aelf doctor --prune-dormant                # dry-run: list dormant DBs only
aelf doctor --prune-dormant --idle-days 90 # tighter idle threshold
aelf doctor --prune-dormant --apply        # prompts [y/N] per DB; deletes on 'y'
```

Sample output:

```
found 2 dormant per-project DB(s) (idle >= 30d, 41,615 beliefs, 18,432.0 KiB total):
  ~/.aelfrice/projects/2e7ed55e017a/memory.db (35,332 beliefs, 15,120.4 KiB, idle 35d)
  ~/.aelfrice/projects/18a856c7a96b/memory.db (6,283 beliefs, 3,311.6 KiB, idle 32d)

dry-run only. re-run with `--apply` to be prompted [y/N] per DB.
```

`--apply` prompts you for each DB. The default at the prompt is N. Any answer other than `y` or `yes` preserves the file, and a bare Enter and a pipe from `/dev/null` are such answers.

There is no `--yes` shortcut, because a destructive deletion is always explicit and for one DB at a time. Unlike `aelf migrate`, this command doesn't move beliefs anywhere. The command only removes the DB file.

The dormant scan doesn't examine the schema. The scan flags a pre-v1.x DB and a modern-schema DB equally when the DB is idle. If you still want to migrate a DB, run `aelf migrate --from <path> --apply` (described earlier) before you prune the DB, not after.

## Update notifier

```bash
aelf upgrade-cmd          # prints the canonical upgrade command (`run: …`) when an
                          # update is available; otherwise `aelfrice is up to date`
aelf upgrade-cmd --check  # same output, no behaviour difference
/aelf:upgrade             # imperative slash: detect + run + re-setup
```

`aelf upgrade-cmd` emits `run: uv tool upgrade aelfrice` on a uv-managed install. If another tool (pipx, pip, system) installed aelfrice, the `run:` line is the migration chain instead. That chain is `pipx uninstall aelfrice && uv tool install aelfrice`, or the pip equivalent. uv is the single supported install channel (#730).

The CLI does not run the upgrade itself, because replacing the running interpreter mid-process is unreliable on Windows.

The orange statusline banner appears automatically when an update is available. To disable the banner, run `export AELF_NO_UPDATE_CHECK=1`.

## Uninstall

Pick a disposition for the DB:

```bash
aelf uninstall --keep-db              # leave the DB in place (safe default)
aelf uninstall --archive backup.aenc  # encrypt to file then delete
aelf uninstall --purge                # permanently delete (three confirmation gates)
uv tool uninstall aelfrice            # finally remove the wheel
```

`--archive` uses Fernet with AES-128-CBC, HMAC, and a scrypt-derived key. To recover the DB later, use this code:

```python
from pathlib import Path
from aelfrice.lifecycle import decrypt_archive
open("out.db","wb").write(decrypt_archive(Path("backup.aenc"), "password"))
```

The `--archive` option and this recovery code both require the `[archive]` extra.

### What gets removed

The store is not a single file. Alongside `memory.db` the package
writes these items:

- the SQLite sidecars (`-wal`, `-shm`)
- the fielded Best Matching 25 (BM25F) index
- any backup DBs made by past migrations
- the per-turn log of the hook injection
- the feed log of the belief writes
- the `transcripts/`, `rebuild_logs/`, and `telemetry/` directories

Several of those items hold belief content verbatim.

`--purge` and `--archive` both operate on that whole set. Each one
prints the manifest before it destroys anything. The manifest gives
every path and its size, so you can check the total against what you
expect.

`--archive` encrypts **the belief database only**. Every other item in
the set is either derived from the database or a rolling capture
buffer. The derived items are the BM25F index, the injection log, the
feed log, and the telemetry; the rolling capture buffer is
`transcripts/`. The command deletes those items rather than adding
them to the archive.

The command reports this behavior, and asks for your confirmation
before it takes your password. An encrypted archive next to plaintext
copies of the same content is not a guarantee.

Before v4.2.0 both flags deleted only `memory.db`. `--archive` also
didn't checkpoint the write-ahead log (WAL) first. On a store held
open by a running hook that meant the archive captured a stale
database. In the worst case the archive captured an empty database.
The real content stayed on disk in plaintext in `memory.db-wal`
(#1173).

If you archived a store under an earlier version, that archive might
be incomplete. Check the archive with `decrypt_archive` before you
rely on it.

aelfrice creates a `.git/aelfrice/` directory or a `~/.aelfrice/`
directory. If `$AELFRICE_DB` points somewhere other than such a
directory, aelfrice can't safely attribute the generically named
artifacts to itself. `$HOME/transcripts/` can be your own directory. aelfrice
therefore lists those artifacts for you to remove by hand instead of
deleting them.

### `~/.aelfrice/`

That directory is a second location with different ownership, and it
belongs to no single store. So uninstall removes what aelfrice put
there **by name**, and never sweeps the directory.

The install state goes in **every** mode, `--keep-db` included. Each
of those files records that a step already happened. A file that
survived would make a reinstall read a stale decision as current. The
install state is:

- the **LLM classification consent sentinel**. This one is
  load-bearing. The sentinel used to outlive the uninstall, so a later
  reinstall read the grant as still valid and never re-prompted
  (#1186). The sentinel now goes. A reinstall asks again.
- the manifest-version stamp and the uv-migration stamp. A reinstall
  of the same version therefore re-merges the hooks. A stamp that
  survives the uninstall makes the reinstall short-circuit into a
  state that is installed but inert.
- the temporal-spine backfill sentinel, the auto-install lock, and the
  `claude-memory` reconcile sentinel.
- `logs/hook-failures.log`, and `logs/` itself once it is empty.

The captured data goes only when you asked for data to go. That is
under `--purge` and `--archive`, but not under `--keep-db`. The
captured data is `telemetry.jsonl` and a legacy `transcripts/`
directory left by pre-repo-store versions.

aelfrice keeps these items in every mode:

- `projects/` holds **other projects' stores**. There is one
  `memory.db` for each project-id slug. `aelf uninstall` disposes of
  one store, the store that `db_path()` resolves to. Removing
  `projects/` would destroy belief corpora that you didn't ask
  `aelf uninstall` to touch. Dispose of those stores separately.
- `shared/` is the conventional home for read-only **federation
  peers** (`knowledge_deps.json`). A federation peer is the corpus of
  another store by another name.
- `user/` holds the **user lock store**, `user/memory.db`, which
  `aelf lock --user` writes. Every project reads it, so uninstalling
  from one project leaves it in place. To remove a single user lock,
  run `aelf unlock --user <id>`.
- `config.json` is your configuration. aelfrice keeps it for the same
  reason as `opt-out-hooks.json`. `opt-out-hooks.json` records your
  decision that a hook should not be installed, and that decision
  should survive a reinstall.

aelfrice **lists and leaves alone** anything else found in
`~/.aelfrice/`. The directory can hold files that aelfrice never wrote. The
destructive modes name every path they are about to remove before they
remove it.

To remove the directory entirely, first check
`ls ~/.aelfrice/projects/`, `ls ~/.aelfrice/shared/`, and
`aelf locked`, which lists your user locks. Then run `uninstall`. Then
run `rm -rf ~/.aelfrice/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aelf: command not found` | Confirm that `~/.local/bin`, the uv tool shim directory, is on `$PATH`. `uv tool update-shell` adds it for you. |
| Hook fires but no `<aelfrice-memory>` block appears | Run `aelf doctor`. Usually the hook command points at a deleted script. |
| `aelf doctor` says "skipped (shell metacharacters)" on a hook line | The install is stale. `aelf setup` rewrites the hook in place. |
| Two worktrees of the same repo see the same beliefs | This is the designed behavior. The two worktrees share `--git-common-dir`. Pin one of them with `AELFRICE_DB`, or give a folder a [`[store]`](CONFIG.md#store-1796) table with a relative `path`, which gives each worktree its own store. |
| Two folders of one repo see each other's beliefs | They share the repository's store. Give each folder a [`[store]`](CONFIG.md#store-1796) table in its own `.aelfrice.toml`. |
| `aelf search` returns "store is empty" | Run `aelf onboard .` from the project root. |
| `SQLite database is locked` under heavy concurrent writes | v1.1.0+ uses WAL and `busy_timeout=5000`. If you still see this error, file an issue with the repro. |
