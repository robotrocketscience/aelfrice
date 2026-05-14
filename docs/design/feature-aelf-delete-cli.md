# Feature spec: `aelf delete` CLI (#440)

**Status:** implementation spec
**Issue:** #440
**Sibling:** `aelf unlock` (`src/aelfrice/cli.py:_cmd_unlock`)
**Storage primitive:** `MemoryStore.delete_belief` (`src/aelfrice/store.py:1315`)

---

## Purpose

Explicit, opt-in removal of one belief from the store. Default user
posture in v1/v2.0 is to **decay** unwanted beliefs (retention class
#290 + Beta-Bernoulli posterior + demotion pressure) rather than delete
them — `aelf delete` is the escape hatch for cases where decay is the
wrong tool: hand-entered duplicates, beliefs that capture session-local
state and were promoted in error, beliefs the user wants gone for
privacy reasons.

The verb is `aelf delete` (sibling of `aelf unlock`). It is hard-delete:
the belief row, FTS entry, edges (both `src` and `dst`), and entity
index rows are removed by `MemoryStore.delete_belief`. One audit row
is written to `feedback_history` *before* the cascade with
`valence = -1.0` and `source = "user_deleted"`, so the forensic record
of "belief X existed and was deleted at T" survives the cascade
(`feedback_history` has no FK back to `beliefs` — orphans are tolerated;
`aelf doctor` already gardens them via `delete_orphan_feedback_events`).

---

## Contract

```
aelf delete <belief-id> [--yes] [--force]
```

| Argument | Required | Default | Notes |
|---|---|---|---|
| `belief_id` | yes | — | The ID of the belief to delete. |
| `--yes` | no | off | Skip the interactive confirmation prompt. |
| `--force` | no | off | Allow deletion of locked beliefs (`lock_level=user`). Does **not** skip the confirmation prompt; pair with `--yes` to delete a locked belief non-interactively. |

Exit codes:

- `0` — deleted successfully.
- `1` — unknown belief ID, locked without `--force`, prompt mismatch, or store write error.

---

## Confirmation prompt

When `--yes` is not passed, the command prints the belief content to stderr
and asks the user to type the first 8 characters of the belief id to
confirm. This is the same idiom `terraform destroy` and `gh repo delete`
use — typed prefix beats `y/N` for muscle-memory safety on a destructive
operation.

```
$ aelf delete a7b23f48d191
about to delete belief a7b23f48d191:
  "The system uses SQLite for storage"
type the first 8 characters of the id to confirm: a7b23f48
deleted: a7b23f48d191
```

Mismatch (or empty input) → exit 1 with `aborted: confirmation did not match`.

`--yes` skips the prompt entirely:

```
$ aelf delete a7b23f48d191 --yes
deleted: a7b23f48d191
```

---

## Locked-belief refusal

By default, beliefs with `lock_level == LOCK_USER` (= the user has run
`aelf lock` on them) are protected. The CLI prints to stderr and exits 1:

```
$ aelf delete a7b23f48d191
belief is locked (lock_level=user); use --force to delete anyway
```

`--force` overrides the lock check:

```
$ aelf delete a7b23f48d191 --force --yes
deleted: a7b23f48d191
```

The audit row written before the cascade carries
`source = "user_deleted_force"` when `--force` was used, so the
forensic record distinguishes ordinary deletes from lock-overrides.

---

## Output

Success: `deleted: <belief-id>` on stdout, exit 0.

Errors (all to stderr, exit 1):

- `belief not found: <belief-id>`
- `belief is locked (lock_level=user); use --force to delete anyway`
- `aborted: confirmation did not match`

---

## Audit trail

One row is written to `feedback_history` immediately before the
`store.delete_belief` cascade:

```python
store.insert_feedback_event(
    belief_id=belief_id,
    valence=-1.0,
    source="user_deleted" if not args.force else "user_deleted_force",
    created_at=_utc_now_iso(),
)
```

The row survives the cascade because `feedback_history` has no FK to
`beliefs`. It is reachable forever via `belief_id`, and `aelf status`
counts it under "feedback events." Eventually it is gardened by
`aelf doctor --gc-orphan-feedback` (already wired); operators who care
about strict audit retention should not run that GC.

`valence = -1.0` is required because `apply_feedback` (the higher-level
path that also updates the Beta posterior) rejects `valence == 0`, and
in any case there is no posterior to update on a belief that is about
to be deleted. The value is symbolic, not numerically consumed.

---

## Semantics — how `delete` differs from related verbs

### vs. `aelf unlock`

`unlock` clears the user-lock on a belief but leaves the belief itself
intact and retrievable. `delete` removes the belief outright. Use
`unlock` when you want the belief to remain available but not
ground-truth; use `delete` when you want it gone.

### vs. `aelf feedback <id> harmful`

`feedback harmful` writes one negative-valence row and lets the
Beta-Bernoulli posterior + demotion-pressure walk demote the belief
gradually. The belief stays in the store and can be rescued by later
positive feedback. `delete` is the terminal action: no recovery.

### vs. retention class (#290)

The retention class makes beliefs decay automatically based on type
(e.g. `session_local` beliefs expire after N days). `delete` is the
manual escape hatch for cases the retention class does not cover.

---

## Implementation notes

- `_cmd_delete` in `src/aelfrice/cli.py` mirrors `_cmd_unlock` in
  structure: open store, resolve belief, gate, act, print, close.
- Argparse subparser registered visible (listed in `--help`) as a
  user-facing verb, consistent with `unlock` and `lock`.
- The interactive prompt reads from `sys.stdin` via `input()`. Tests
  pass a stub stream via `monkeypatch` (same pattern as the
  `aelf onboard` interactive tests in `tests/test_onboard.py`).
- Slash command `/aelf:delete` mirrors
  `src/aelfrice/slash_commands/unlock.md`. **Important safety note in
  the slash file:** the slash form does *not* imply `--yes`. The
  invoking surface must let the user respond to the prompt.
- New entry in `EXPECTED_COMMANDS` (`tests/test_slash_commands.py`).
- New entry in `docs/user/COMMANDS.md`.

---

## Out of scope (deferred to v2.x or beyond)

- **MCP `aelf_delete` tool port.** Per #382 Track E ratification (A6,
  2026-05-04): only `confirm` (#390) and `unlock`/`promote`/`demote`
  (#391) ship in v2.0. `delete` defers until filed user demand AND
  demonstrated bench impact.
- **Soft-delete (`deleted_at` column + `SUPERSEDES` edge).** The issue
  body lists soft-delete as a design option. Hard-delete is chosen for
  v2.0 because:
  1. Soft-delete requires a schema migration and rewrites every
     retrieval path to filter `deleted_at IS NULL`. That diff is large
     and the test surface is subtle — the kind of change that wants
     its own bench-gated issue, not a bundled-in addition to the CLI
     port.
  2. There is no consumer for the `SUPERSEDES` edge today. Adding the
     edge without consumers means writing dead structure.
  3. The audit-row-in-`feedback_history` approach (above) preserves
     "X existed and was deleted at T" for forensic purposes, which is
     the actual user need behind soft-delete in this codebase.
  If a future use case demands recoverable delete (e.g. an "undelete"
  command, or graph-walks across deleted beliefs), file a separate
  issue with that consumer named.
- **Bulk delete (`--all-where <predicate>`).** No filed demand; risk
  of a typo destroying many beliefs is unbounded. Defer.

---

## Acceptance checklist (mirrors issue #440)

1. ✅ Spec memo (this document) — picks hard-delete + audit-row-in-`feedback_history`, justifies vs. soft-delete.
2. ✅ Confirmation prompt by default; `--yes` to bypass.
3. ✅ Refuses to delete locked beliefs without `--force`.
4. ✅ Unit tests + integration tests (covering: not-found, locked-without-force, prompt-mismatch, prompt-match, --yes path, --force path, audit-row written, cascade through edges).
5. ✅ Doc: `docs/user/COMMANDS.md` entry.
6. ✅ Slash command `/aelf:delete`.
7. ✅ Registration in `EXPECTED_COMMANDS`.
