# Hook-activity schema

**Status:** schema reservation.
**Target milestone:** v1.2.x patch (companion to the
[search-tool hook](search_tool_hook.md); part of the Claude Code Hook
Observability v1 series tracked in
[issue #135](https://github.com/robotrocketscience/aelfrice/issues/135)).
**Dependencies:** none on the aelfrice side. The producing hook lives
in the user's HOME repo (`~/.claude/hooks/log-tool-failure.py`) and
is tracked separately from this codebase.
**Risk:** low. This document reserves an event-name namespace; it does
not introduce new code. The companion regression test
([`tests/test_hook_activity_schema.py`](../tests/test_hook_activity_schema.py))
guards against future aelfrice-side writers colliding on the namespace.

## Summary

Reserves the `PostToolUseFailure:<tool_name>` event-name namespace
inside `~/.aelfrice/hook-activity.jsonl` for raw tool-failure
observation. Pure observation; no consumer (pattern → belief bridge)
ships in this release.

The producing hook captures every Claude Code `PostToolUseFailure`
matcher firing, reads the standard hook stdin payload, appends one
JSONL line, and exits 0 unconditionally. The hook never blocks
Claude. The aelfrice side guarantees that no writer in
`src/aelfrice/` emits a literal `PostToolUseFailure` event-name
string, so the namespace is safe for the HOME-side hook to own.

## Field schema

One JSONL line per failure event, appended to
`~/.aelfrice/hook-activity.jsonl`:

| field | type | value |
|---|---|---|
| `event` | string | `PostToolUseFailure:<tool_name>` (e.g., `PostToolUseFailure:Bash`) |
| `action` | string | `observe` |
| `content` | string | `tool=<name>; err=<first 200 chars>; input=<truncated keys>` |
| `latency_ms` | float | `0.0` (the hook does not measure tool latency) |
| `cwd` | string | working directory from the hook payload |
| `session` | string | `session_id` from the hook payload |
| `ts` | string | ISO-8601 UTC timestamp at append time |

### Field-size cap

**Every field is capped at 500 characters.** This bounds JSONL line
size — `Bash` failures in particular can carry multi-line stderr
that easily exceeds a sensible per-line budget. The cap is enforced
by the producing hook, not by aelfrice.

The `content` sub-budget reserves room for all three components:
`tool=<name>` (≤32), `err=<first 200 chars>` (≤208 with prefix),
`input=<truncated keys>` (≤60 with prefix). Total stays inside the
500-char field cap.

### Standard fields

`cwd`, `session`, and `ts` follow the existing hook-activity row
shape. They are not specified further here — the producing hook
matches whatever shape other rows in `hook-activity.jsonl` already
use.

## Reserved namespace

The string prefix `PostToolUseFailure:` (note the trailing colon) is
**reserved for the failure-signal capture hook only**. No aelfrice
module, slash command, hook entrypoint, or test fixture emits this
prefix as an event name in `hook-activity.jsonl` or any other
JSONL surface.

The companion test
([`tests/test_hook_activity_schema.py`](../tests/test_hook_activity_schema.py))
greps `src/aelfrice/` for the literal string and fails with a
descriptive message if any future writer collides.

## Consumer-side warnings

### Dedupe by fingerprint before treating as signal

Raw `PostToolUseFailure:*` rows are **noisy**. Transient failures
(network blips, file locks, user-interrupted tools, retry storms)
will be over-represented in the raw count. A consumer that treats
every row as an independent signal will weight transients the same
as systematic problems.

Any future consumer (the deferred pattern → belief bridge) MUST:

1. **Bin by time window** before counting. A 5-minute bucket
   collapses a retry storm into a single observation.
2. **Dedupe by fingerprint** `(tool, error-fingerprint)`. The
   fingerprint is whatever stable hash of `err=...` the consumer
   chooses (first 200 chars is the source material; a normalisation
   step is the consumer's call).
3. **Apply a confidence floor** before promoting a pattern to a
   belief. Single-observation failures should not become beliefs.

Do not act on raw counts. The schema is intentionally minimal so
the consumer owns these decisions.

### `latency_ms = 0.0` is a sentinel

The producing hook fires *after* the tool has already failed. It
does not measure how long the failed tool ran. The literal `0.0`
makes that fact unambiguous in the row.

## Acceptance criteria (schema reservation)

1. `docs/hook_activity_schema.md` documents the
   `PostToolUseFailure:<tool_name>` event-name namespace, the field
   schema, the 500-char field cap, and the consumer-side dedupe
   warning.
2. No file under `src/aelfrice/` emits the literal string
   `PostToolUseFailure` as an event-name in any JSONL output.
3. A regression test asserts (2) deterministically and runs in
   well under 1 second.

## Out of scope

- The hook script itself. Lives in the user's HOME repo at
  `~/.claude/hooks/log-tool-failure.py`; tracked separately.
- Pattern → belief bridge (the consumer). Deferred to a future
  issue once production data shows whether the dedupe + binning
  approach above is sufficient.
- Triggering `apply_feedback` from these events. Strictly
  consumer-layer concern.
- Aelfrice-side aggregation, decay, or storage of the JSONL
  rows. The file is owned by the HOME-side hook; aelfrice does
  not read it in v1.2.x.

## What unblocks when this lands

The schema reservation is the structural guarantee that lets the
HOME-side hook ship without coordination risk: no future aelfrice
release can accidentally write rows in the same namespace. Once
production data accumulates under this schema, a follow-up issue
will scope the consumer (pattern → belief bridge), at which point
the dedupe-by-fingerprint warning above becomes the consumer's
acceptance criterion.
