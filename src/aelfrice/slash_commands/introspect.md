---
name: aelf:introspect
description: Read-only honest-signal view over stored beliefs. Groups active beliefs by session or project and surfaces posterior μ, recurrence, grounding, floated-vs-decided status, and stranded-capture noise together — the native answer to "analyse the beliefs it extracted."
argument-hint: Optional flags — --by session|project, --session ID, --project CTX, --only-noise, --limit N, --json
allowed-tools:
  - Bash
---
<objective>
Answer "look at my conversations and analyse the beliefs it extracted"
natively. `introspect` groups the active beliefs (by ingest session, or by
project with `--by project`) and, for each, shows the signals that already
live in the store but are never displayed together:

- **μ / n** — posterior mean and its evidence weight (α+β).
- **recur** — how many times the belief was re-asserted. This is recurrence,
  NOT truth: a junk line captured every session scores high on recur alone.
- **grounding** — durable (grounds to file paths / error codes / symbols),
  ephemeral (grounds to version / branch / bare issue-number chatter), or
  neutral (prose with no grounding signal). The standalone-vs-context-bound
  axis.
- **status** — floated vs decided, from RESOLVES / POTENTIALLY_STALE edges.
- **NOISE** — stranded-capture scaffolding (orphan headers, shell echoes).
  These float to the top of each group as the prime retire candidates.

The view is read-only. Curate with the verbs it points at:
`aelf retire <id>` (reversible), `aelf lock <id>`, `aelf resolve`.

Useful flags: `--session <id>` / `--project <ctx>` to scope, `--only-noise`
for the retire shortlist, `--limit N` (0 = no cap), `--json` for structured
output.
</objective>

<process>
Run: `uv run aelf introspect $ARGUMENTS`
Display the output verbatim, then, if the user wants to curate, offer the
retire/lock/resolve verbs from the footer.
</process>
