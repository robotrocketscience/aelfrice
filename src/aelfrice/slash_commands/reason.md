---
name: aelf:reason
description: Walk the belief graph for a query, then dispatch role-assigned subagents per impasse and emit SUGGESTED UPDATES.
argument-hint: Keyword query (e.g. "session id resolution")
allowed-tools:
  - Bash
  - Task
---
<objective>
Surface a reasoning chain over the aelfrice belief graph for the given query, then act on what the verdict says is missing.

Seeds come from top-3 BM25 hits; expansion walks outbound edges with terminal-tight defaults. R1 derives a verdict (SUFFICIENT / PARTIAL / UNCERTAIN / INSUFFICIENT / CONTRADICTORY) and a list of impasses (TIE / GAP / CONSTRAINT_FAILURE / NO_CHANGE). R3 (this skill) reads that derivation and:

1. Fans out one subagent per impasse, role-tagged by impasse kind.
2. Emits a `SUGGESTED UPDATES` section so the caller can pipe `(belief_id, direction)` rows into `aelf feedback`.

The Python side owns the contract; this markdown is the dispatch script.
</objective>

<process>
Run: `uv run aelf reason "$ARGUMENTS" --json`

Parse the payload. It has these top-level keys: `query`, `seeds`, `hops`, `paths`, `verdict`, `impasses`, `dispatch`, `suggested_updates`.

**Step 1 — present the chain.** For every verdict except `INSUFFICIENT`, print the seeds and hop tree from `payload.hops` in indented form so the operator can read the answer. If the verdict is `INSUFFICIENT`, print only the seeds and the impasse summary.

**Step 2 — dispatch subagents on each `payload.dispatch[i]`.** The CLI already mapped impasse kind to subagent role (`Verifier` / `Gap-filler` / `Fork-resolver`). For each row, spawn one `Task` subagent with the matching role-prompt scaffold below. Pass the row's `belief_ids` and `note` as input. Run all subagents in parallel — they are independent. If `payload.dispatch` is empty (verdict is `SUFFICIENT`), skip step 2.

**Role prompts** (use verbatim, substituting `{belief_ids}` and `{note}`):

- **Verifier** — "A locked belief at IDs {belief_ids} is on a CONTRADICTS edge: {note}. Verify whether the locked belief is still accurate against authoritative sources; if not, propose `aelf unlock` + a replacement lock. Do not run the commands yourself; surface them as suggested follow-ups."
- **Gap-filler** — "The reasoning walk dead-ended at belief(s) {belief_ids} ({note}). Research what's missing: extract entities, find authoritative sources, summarize what should be ingested. Surface suggested `aelf onboard` / `aelf` commands; do not run them."
- **Fork-resolver** — "Two beliefs at IDs {belief_ids} contradict each other with similar posterior strength ({note}). Find evidence that adjudicates; propose `aelf feedback <id> used` (for the supported side) and `aelf feedback <id> harmful` (for the rejected side). Do not run them yourself."

**Step 3 — emit SUGGESTED UPDATES.** After step 2 returns, print exactly:

```
SUGGESTED UPDATES
  - <belief_id>  <direction>  (<note>)
  - <belief_id>  <direction>  (<note>)
  ...
```

…with one row per element in `payload.suggested_updates`. If the list is empty, print `SUGGESTED UPDATES (none)`. Direction values are `+1` (helpful — on the answer chain), `?` (uncertain — on an impasse), or `-1` (rejected — never emitted by the current surface; R2's fork-path data has shipped (#658) but the `-1` derivation over fork-losing paths is deferred to a follow-up).

The caller (or the operator, depending on policy) pipes the `+1` rows into `aelf feedback <belief_id> used` to bump the Beta-Bernoulli posterior (and `-1` rows into `aelf feedback <belief_id> harmful` if a policy uses them). `?` rows are surfaced for manual review and never auto-piped. Skip any row whose `owning_scope` is non-null — that belief lives in a read-only federation peer store and `aelf feedback` rejects mutations on foreign ids (ForeignBeliefError, exit 1); only local rows are pipeable.

Do not add commentary outside the steps above. The CLI text-mode output is for human reading; this `--json`-driven path is the agent-side acting protocol.
</process>
