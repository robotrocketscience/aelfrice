---
name: aelf:promote
description: Promote an agent_inferred belief to user_validated origin. Optionally flip the belief's visibility scope with --to-scope. Alias of aelf:validate with identical semantics.
argument-hint: The belief ID to promote, optionally followed by --to-scope <scope>
allowed-tools:
  - Bash
---
<objective>
Promote an agent_inferred belief to the user_validated origin tier. This is
a provenance change only — alpha/beta posteriors are unchanged. Idempotent:
promoting an already-validated belief exits 0 with an "already validated"
message. This command is a first-class alias of `aelf validate`.

Optionally, flip the belief's visibility scope with `--to-scope`. Valid scope
values are `project` (local-only, default), `global` (any dependent peer), or
`shared:<name>` (named peer group, e.g. `shared:team-a`). The scope flip is
orthogonal to the origin promotion: both may occur in one call when the belief
is `agent_inferred`; only the scope changes when the belief is already
`user_validated`. Each scope change writes a zero-valence audit row tagged
`scope:<old>-><new>` to feedback_history. Foreign belief IDs are rejected.
</objective>

<flags>
--to-scope <scope>   Flip the belief's visibility scope. One of:
                       project   — local-only (default)
                       global    — visible to any dependent peer
                       shared:<name>  — named peer group
                     Writes audit row 'scope:<old>-><new>'. Rejected for
                     foreign belief ids. Invalid scope values are rejected at
                     parse time.
</flags>

<process>
Run: `uv run aelf promote "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
