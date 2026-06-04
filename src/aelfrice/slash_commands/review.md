---
name: aelf:review
description: Generate a weekly checkpoint file of beliefs to keep, remove, or lock, then apply your verdicts.
allowed-tools:
  - Bash
  - Read
  - Edit
---
<objective>
Run a periodic belief-review cycle: surface the oldest-unconfirmed beliefs
as a checkbox file, let the user edit verdicts, then apply them to the store.
</objective>

<process>
## Generate a review file

Run: `uv run aelf review --generate`

This writes `.aelfrice/review.md` with up to 10 beliefs sorted by how long
they have gone without a keep/remove/lock verdict. Display the path and
candidate count from the output line.

## Edit the review file

Open `.aelfrice/review.md` for the user to review. Each row looks like:

```
- [ ] keep   [ ] remove   [ ] lock   | <id> (<age>d old, <cold>d cold) — <snippet>
```

The user marks exactly one checkbox per belief:
- `[x] keep` — confirms the belief is still accurate (updates last_confirmed_at)
- `[x] remove` — soft-deletes the belief from active retrieval
- `[x] lock` — promotes the belief to user-locked ground truth
- All boxes empty — skips this belief (it stays in the next review cycle)

**Do NOT auto-apply.** Present the file, explain the format, and ask the
user to fill in their checkboxes before proceeding to the apply step.

## Apply verdicts

Only after the user confirms they have finished editing:

Run: `uv run aelf review --apply`

Display the summary line (kept / removed / locked / skipped counts).
If `--apply` exits non-zero, show the error and do not retry automatically.

## Flags

- `--out PATH` — use a custom path for the review file (both generate and apply).
- `--json` — emit the apply report as JSON (machine-readable).
</process>
