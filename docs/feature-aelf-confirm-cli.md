# Feature spec: `aelf confirm` CLI (#441)

**Status:** implementation spec  
**Issue:** #441  
**MCP sibling:** `aelf_confirm` / `tool_confirm` — shipped in #390  
**`tool_confirm` location:** `src/aelfrice/mcp_server.py:432-476`

---

## Purpose

Explicit user affirmation of an existing belief. `aelf confirm <belief-id>` tells
the system "I've checked this belief and it is correct." The signal is stronger
than the implicit "got used" signal emitted by retrieval hooks, because it carries
the user's intent rather than being inferred from continuation behaviour.

---

## Contract

```
aelf confirm <belief-id> [--source SRC] [--note TEXT]
```

| Argument | Required | Default | Notes |
|---|---|---|---|
| `belief_id` | yes | — | The ID of the belief to affirm. |
| `--source` | no | `user_confirmed` | Written to `feedback_history.source`. Override to tag automated scripts. |
| `--note` | no | `""` | Free-text annotation; appears in stdout on success but is **not persisted** to the store. |

Exit codes:

- `0` — applied successfully.
- `1` — unknown belief ID, or store write error.

---

## Output (stdout on success)

```
confirmed <belief-id>: alpha 1.000->2.000, mean 0.500->0.667
```

Fields:

- `prior_alpha -> new_alpha` — raw Beta parameter before and after the update.
- `mean` — `new_alpha / (new_alpha + new_beta)` rounded to 3 d.p., so the
  posterior direction is immediately readable without mental arithmetic.

On unknown belief, writes to **stderr** and exits 1:

```
confirm error: unknown belief: <belief-id>
```

---

## Semantics — how `confirm` differs from related verbs

### vs. `aelf feedback <id> used`

`feedback used` is the implicit signal emitted when retrieval hooks observe a
belief was retrieved and then the continuation referenced it. `confirm` is an
*explicit* user affirmation carrying `source="user_confirmed"`, which is
distinguishable in `feedback_history` queries and in the `aelf status` counts.
Same Beta-Bernoulli mechanic (`α += 1.0`); different intent and source label.

### vs. `aelf lock <statement>`

`lock` freezes the belief as user-asserted ground truth (`lock_level=user`),
giving it maximum retrieval priority and protecting it from demotion pressure
until `aelf unlock` is called. `confirm` applies one unit of positive feedback
to the Beta posterior without freezing the belief. Use `lock` when you want the
belief treated as canonical; use `confirm` when you want to nudge the posterior
without the commitment of a ground-truth freeze.

---

## Storage layer

`confirm` calls `apply_feedback(store, belief_id=..., valence=1.0, source=...)`,
which writes one row to **`feedback_history`**. This matches the MCP sibling
`tool_confirm` exactly.

### Reconciliation with issue #441 body

Issue #441 acceptance criterion #2 reads: *"Writes a row to the
`belief_corroborations` table (#190)."* This is incorrect / out of date.

The shipped `tool_confirm` (MCP, #390) writes to `feedback_history` via
`apply_feedback`, not to `belief_corroborations`. The `belief_corroborations`
table tracks *duplicate re-ingests* of the same content-hash from different
sources — a structural dedup concern, not a user-affirmation signal. The CLI
implementation follows the MCP — `feedback_history` only.

---

## Implementation notes

- `_cmd_confirm` in `src/aelfrice/cli.py` mirrors `_cmd_unlock` in structure.
- Calls `tool_confirm` imported from `aelfrice.mcp_server`; the wrapper is the
  business logic. No refactor of `mcp_server.py` beyond the import.
- Argparse subparser registered visible (listed in `--help`) as a user-facing
  verb, consistent with `unlock` and `promote`.
- Slash command `/aelf:confirm` mirrors `src/aelfrice/slash_commands/unlock.md`.

---

## Provenance refs

- #441 — this issue (CLI port)
- #390 — shipped MCP `aelf_confirm` / `tool_confirm`
- #190 — `belief_corroborations` table (out of scope for confirm)
