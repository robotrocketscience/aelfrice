# Antigravity CLI hook surface, as observed

Observed behavior of Antigravity CLI (`agy`) hooks, recorded with
`scripts/agy_hook_probe.py`. `host_antigravity.py` and `agy_hook.py` are
built from this document. Where this document and the Antigravity hooks reference
disagree, this document is right for the version it was recorded against.

- Antigravity CLI version: 1.5.7 (`agy --version`), model
  `gemini-3.8-flash-high`
- Recorded: 2026-11-12, in an empty scratch workspace, with the probe
  installed in the workspace's `.agents/hooks.json`. No global config changed.
- Runs: ten headless `agy -p` runs, one per probe scenario, plus one resume
  run with `agy -c -p`. Each prompt asked the model to run `echo hello` in the
  shell and quote every `AELFPROBE` token it could see.

References compared: https://antigravity.google/docs/hooks/ and the hooks
guide bundled with the CLI at
`~/.gemini/antigravity-cli/builtin/skills/agy-customizations/docs/hooks.md`.
The bundled guide is newer than the website (it documents `hookEventName`,
`cwd` and `continuationActive`).

## Events

| Event | Fired | Payload keys beyond the base set | Output that reaches the model | Consequence for aelfrice |
|---|---|---|---|---|
| `SessionStart` | once per new conversation, with the first prompt. Not on `agy -c` resume. | none | `injectSteps` `systemMessage` and `userMessage`: yes | Session-start baseline injection works. A resumed conversation gets no baseline. |
| `PreInvocation` | before every model call. `invocationNum` restarts at 0 on each user turn, including after `-c` resume. | `invocationNum`, `initialNumSteps` | `injectSteps` `systemMessage` and `userMessage`: yes, and they persist for the rest of the turn. `ephemeralMessage`: seen only on the call it was injected for. | `invocationNum == 0` is the per-prompt point. Retrieval injection and user-turn capture go here, as `systemMessage`. |
| `PostInvocation` | after every model call | same as `PreInvocation` | `systemMessage` injected after a tool call: yes, on the next call. After the final call: written to the transcript, never seen. | Unused by aelfrice. |
| `PreToolUse` | before each tool call, matcher `*` | `toolCall` (`id`, `name`, `args`), `stepIdx` | `reason` on `allow`: no | No tool-time context. See Output channels. |
| `PostToolUse` | after each tool call | `toolCall`, `stepIdx`, `error` (empty string on success) | `injectSteps`: no | No tool output in the payload. The output is on disk at `<artifactDirectoryPath>/.system_generated/steps/<stepIdx>/output.txt`. |
| `Stop` | once per turn, at loop end | `executionNum`, `terminationReason` (`NO_TOOL_CALL` on a normal turn), `error`, `fullyIdle`, `continuationActive` | `decision: "continue"` re-enters the loop with `reason` injected | Equivalent to Claude's `Stop`. See the loop-guard note below. |
| `UserPromptSubmit`, `PreCompact`, `PostCompact`, `SessionEnd`, `Notification` | never | n/a | n/a | Accepted silently in `hooks.json` and never fire. Registering them does not break the file. |

Base payload keys on every event: `conversationId`, `workspacePaths`,
`transcriptPath`, `artifactDirectoryPath`, `modelName`, `lastUserInput`,
`hookEventName`, `cwd`. All keys are camelCase.

### Stop loop guard

`continuationActive` stayed `false` on all 11 `Stop` fires of a run whose
hook answered `continue` every time, which contradicts the bundled guide.
`executionNum` counted 0 to 10. After 10 forced continuations agy stopped on
its own and still reported `terminationReason: NO_TOOL_CALL`. A hook that may
answer `continue` must guard on `executionNum > 0`, not on
`continuationActive`.

## Tools

Tool names are snake_case: the shell tool is `run_command`, with the command
in `toolCall.args.CommandLine` (other args: `WaitMsBeforeAsync`,
`toolSummary`). Claude names such as `Bash` never match. The Codex host's
`Bash` matchers translate to `run_command`, and `tool_input.command` to
`toolCall.args.CommandLine`.

## Output channels

| Output | Result on 1.5.7 |
|---|---|
| `PreToolUse` stdout `{"decision":"allow"}` | tool runs, no permission prompt in `-p` mode |
| `PreToolUse` stdout empty, exit 0 | tool runs |
| `PreToolUse` stdout `{}` | tool denied |
| `PreToolUse` exit 2 (stdout `deny`) | tool blocked; agy reports `JSON hook "jsonhook__aelf-probe_PreToolUse_0_0" failed: command failed: exit status 2` |
| `PreInvocation` exit 1 with valid `injectSteps` | injection dropped, turn continues |
| `PreToolUse` `reason` with `allow` | not visible to the model, not in the transcript |
| `PostToolUse` `injectSteps` | not visible to the model, not in the transcript |

Consequences:

- A `PreToolUse` hook that has nothing to say must print nothing or
  `{"decision":"allow"}`. Printing `{}` denies the tool. The search-tool
  hook prints a Claude `hookSpecificOutput` object with no `decision` key,
  which agy would read as a deny (UNVERIFIED: only `{}` was tested), so it
  needs an adapter before it runs under agy.
- A non-zero exit from a `PreToolUse` hook blocks the tool. A crash in an
  aelfrice tool hook would block every shell command, so the adapter must
  catch everything and exit 0. The bundled guide says hooks fail open; that
  holds for `PreInvocation`, not for `PreToolUse`.
- Tool-time context (Claude's `PreToolUse`/`PostToolUse` `additionalContext`,
  used by the search-tool hooks) has no direct channel. The nearest channel is
  the next `PreInvocation` (`invocationNum > 0`), which fires after the tool
  call and before the model reads the result.

## Hook process environment

- Working directory: the folder holding `hooks.json` (for the global file,
  `~/.gemini/config`), never the user's project. aelfrice must resolve the
  project from the payload's `cwd`, not from `os.getcwd()`.
- Host detection: the process gets `ANTIGRAVITY_CONVERSATION_ID`. No other
  `ANTIGRAVITY_*`, `AGY_*` or `GEMINI_*` variable is set.
- Commands run through `sh -c`, so `~` and absolute paths both work.

## Transcripts

`transcriptPath` points at
`~/.gemini/antigravity-cli/brain/<conversationId>/.system_generated/logs/transcript_full.jsonl`
(the bundled guide's example says `transcript.jsonl`, which also exists).
Each line is one step with snake_case keys: `step_index`, `source`
(`USER_EXPLICIT`, `MODEL`, `SYSTEM_SDK`), `type` (`USER_INPUT`,
`PLANNER_RESPONSE`, `SYSTEM_MESSAGE`, `GENERIC`), `status`, `created_at`,
`content`, and on model steps `thinking`, `tool_calls` and token counts.
The user prompt arrives wrapped in `<USER_REQUEST>` tags. Injected system
messages are recorded as `SYSTEM_MESSAGE` steps.

## Mapping

Every row runs through one command, `aelf-agy-hook`, which dispatches on
`hookEventName`.

| aelfrice hook (Claude event) | agy event | Notes |
|---|---|---|
| retrieval injection (`UserPromptSubmit`) | `PreInvocation`, only when `invocationNum == 0` | output as `injectSteps` `systemMessage` |
| transcript logger, user half (`UserPromptSubmit`) | `PreInvocation`, `invocationNum == 0` | prompt from `lastUserInput` |
| transcript logger, assistant half and stop hook (`Stop`) | `Stop` | assistant text from the last `PLANNER_RESPONSE` in `transcriptPath`; a continuation only when `executionNum == 0` |
| session-start baseline (`SessionStart`) | `SessionStart`; on resume, the first `PreInvocation` of a conversation with no baseline marker | markers in `~/.aelfrice/antigravity/baseline/` |
| search-tool hook on Bash (`PreToolUse`) | `PreToolUse` on `run_command`, injected at the next `PreInvocation` | queued per conversation in `~/.aelfrice/antigravity/pending/` |
| commit ingest (`PostToolUse` on `Bash`) | `PostToolUse`, matcher `run_command` | output read from `steps/<stepIdx>/output.txt` |
| pre-issue guard (`PreToolUse` on `Bash`) | `PreToolUse`, matcher `run_command` | prints `{"decision":"deny","reason":...}` on a block and nothing on a pass |
| agent context (`PreToolUse` on `Agent`) | `PreToolUse`, matcher `invoke_subagent` | rewrites `args.Prompt` through `overwrite` |
| `PreCompact`, `PostCompact`, statusline, claude-memory mirror | none | listed by `aelf doctor --host antigravity` |

## Reproducing

```bash
P=scripts/agy_hook_probe.py
python3 -I $P --self-test
python3 -I $P --install <ws> --state <state> --decoys
python3 -I $P --scenario inject-system --state <state>
(cd <ws> && agy -p '<prompt>' > <state>/out-inject-system.txt)
python3 -I $P --report --state <state>
python3 -I $P --uninstall <ws>
```
