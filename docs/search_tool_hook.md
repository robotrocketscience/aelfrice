# Search-tool hook

**Status:** spec.
**Target milestone:** v1.3.0 (named on the public roadmap as
"search-tool `PreToolUse` hook for memory-first context retrieval").
**Dependencies:** stdlib only. Consumes the v1.0 retrieval pipeline
([`aelfrice.retrieval.retrieve`](../src/aelfrice/retrieval.py)) and
the v1.1.0 per-project DB resolution
([`aelfrice.cli.db_path`](../src/aelfrice/cli.py)).
**Risk:** medium. Hook fires on every Grep/Glob tool call, so the
latency budget must be tight enough that the user does not perceive
the hook as making search "feel slow."

## Summary

A Claude Code `PreToolUse` hook that fires before a `Grep` or `Glob`
tool call, lifts the agent's search query out of the tool input,
runs the same query against the per-project belief store, and emits
the results back as `additionalContext` so the agent sees them
*before* the tool runs. If memory already has the answer the agent
can skip or refine the tool call; if not, the agent uses the tool to
fill in gaps the memory does not cover.

```
Claude wants to Grep "directive-gate" in the project
        ↓
  PreToolUse hook fires (tool_name in {Grep, Glob})
        ↓
  extract query tokens from tool_input.pattern
        ↓
  retrieve(store, query, token_budget=...) → beliefs
        ↓
  emit additionalContext = "<aelfrice-search query=...>{results}</aelfrice-search>"
        ↓
  Claude reads context AND runs Grep
```

## Motivation

Every retrieval-shaped tool call is an opportunity to surface
already-stored context. Without this hook, the only retrieval
trigger today is `UserPromptSubmit` — which fires once per user
turn, against the user's natural-language prompt. That is a
different and complementary surface from the agent's *own*
retrieval intent.

When Claude reaches for `Grep` in the middle of a multi-tool turn,
the search query is a precise, agent-formulated probe. Two payoffs:

1. **Skip-or-pivot.** If the project's belief store already
   contains the answer (a prior decision, a locked correction, a
   relevant past finding), the agent sees it before the
   filesystem-walking Grep returns. The agent can then skip the
   Grep entirely, or use it for follow-up rather than discovery.
2. **Fill-the-gap.** When the store has nothing relevant, the
   hook surfaces "no matching beliefs" explicitly, distinguishing
   *consulted-and-empty* from *not-consulted*. This signals that
   the Grep result is the new substrate to learn from, and removes
   the ambiguity the agent currently faces ("did the memory check
   not happen, or was there no match?").

This hook is the inverse of the v1.2.0 commit-ingest hook. That one
*writes* the graph after the agent acts. This one *reads* the graph
before the agent acts. Together they close the loop: information
generated during normal session activity becomes context for the
next session's tool-decision points.

## Design

### Hook registration

A new entry point `aelfrice.hook_search_tool:main`, registered as
a Claude Code `PreToolUse` hook with matcher `Grep|Glob`. The hook
configuration matches on tool calls and emits a JSON object with
`hookSpecificOutput.additionalContext` to inject results.

Configuration lives under the user's `~/.claude/settings.json`. The
opt-in surface is `aelf setup --search-tool`, mirroring `aelf setup
--commit-ingest` from v1.2.0.

### Hook contract (PreToolUse)

stdin (one JSON object):

```json
{
  "hook_event_name": "PreToolUse",
  "tool_name": "Grep" | "Glob",
  "tool_input": { "pattern": "...", ... },
  "cwd": "...",
  "session_id": "..."
}
```

stdout (one JSON object on success; empty on no-op skip):

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "additionalContext": "<aelfrice-search query=\"...\">...</aelfrice-search>"
  }
}
```

Exit code: always 0. The hook is non-blocking. Any failure path
(empty query, store missing, retrieval exception) returns silently
without an `additionalContext` block.

### Query extraction

Both `Grep` and `Glob` tool inputs use the field `tool_input.pattern`.
Patterns range from raw regexes (`r"^\\s*git\\s+commit\\b"`) to
glob expressions (`"src/**/*.py"`) to plain phrases
(`"directive-gate"`). The hook extracts alphanumeric word tokens
of length ≥ 3 from the pattern and joins the first 5 with FTS5
`OR` to form a search query:

```
pattern = "experiments/*/HYPOTHESES.md"
tokens  = ["experiments", "HYPOTHESES"]   (3+ chars, alphanumeric)
query   = "experiments OR HYPOTHESES"
```

```
pattern = "src/aelfrice/hook_*.py"
tokens  = ["src", "aelfrice", "hook"]
query   = "src OR aelfrice OR hook"
```

Empty token sets (e.g., a pure-character glob like `"**/*.rs"`)
produce no query and the hook returns silently. The 5-token cap
prevents pathological queries from blowing FTS5 budget. The 3-char
minimum filters single-letter regex anchors and short noise.

### Budget framing

The hook calls `retrieve(store, query, token_budget=600,
l1_limit=10)`. Token budget is half the default to keep the
injection light — this is auxiliary context, not the user's
turn-level retrieval. L1 limit is also lower for the same reason.

The output block uses an XML-shaped envelope keyed on the *query
that was run*, so multiple search-tool injections in the same
turn can be distinguished:

```
<aelfrice-search query="...">{l0+l1 results, one per line}</aelfrice-search>
```

When `retrieve()` returns the L0-only set (empty query) or zero
results, the hook emits an explicit "no matching beliefs in store"
sentinel so the agent learns *the check ran*. Empty stdout would be
ambiguous with hook-skipped (e.g., empty-query path).

### Latency budget

The hook runs synchronously before every Grep and Glob. User-facing
latency matters even more than commit-ingest — search tools are
tight loops, and an unbudgeted hook would feel like the search
itself slowed down. Budget: **median ≤ 50 ms, p95 ≤ 200 ms** on a
populated store (~10k beliefs) for a 5-token query.

Tactics:

1. **Lazy import** of `aelfrice.retrieval` and `aelfrice.store`.
   Cold-start a Python interpreter dominates on most systems.
2. **Skip work on empty token sets.** Pure-glob patterns and
   single-character regexes exit immediately.
3. **Cap token count and per-token length.** First 5 tokens, ≥ 3
   chars each. Bounds FTS5 query complexity.
4. **Reduced retrieval budget.** `token_budget=600`, `l1_limit=10` —
   well below the user-facing default 2000 / 50.
5. **Read-only access.** No write paths — the hook only reads the
   FTS5 index and the locked-beliefs table.

A regression test verifies the budget on a fixture store with
representative volume.

### What gets emitted

- Always: an `additionalContext` block keyed by the extracted query.
- On match: one line per belief in `<aelfrice-search>...</aelfrice-search>`,
  format `[L0|L1] {id-prefix}: {content}` truncated to 200 chars
  per line.
- On no-match: a single sentinel line explaining "no matching
  beliefs in store; the tool result will fill the gap."

What does NOT get emitted:

- The full Grep/Glob result (the tool runs after the hook; its
  output is the agent's responsibility).
- Beliefs above the per-line truncation cap.
- Confidence scores (these are L0 locks + L1 BM25; ranking is
  preserved by ordering, not by numeric annotation).

### Failure modes

- **Empty token set.** Hook exits with no `additionalContext`. Tool
  runs as if the hook were absent.
- **Store missing or locked.** Hook returns silently with one log
  line. Tool runs unaffected.
- **`retrieve()` raises.** Caught at the entry-point boundary;
  traceback printed to stderr; hook returns 0. Tool runs unaffected.
- **JSON encode error on output.** Should not happen — payload is
  small, ASCII-safe with `\u`-escaping. Caught at boundary.

The principle, mirroring commit-ingest: the hook may NEVER cause a
`Grep` or `Glob` to feel broken. Worst case is "no aelfrice context
this call," same as if the hook were uninstalled.

### Opt-in surface

`aelf setup --search-tool` writes the hook configuration. Default
on fresh install: opt-in (consistency with v1.2.0 hook surface).

`aelf setup --no-search-tool` removes it.

## Acceptance criteria

1. The hook fires on `Grep` and `Glob` tool calls and does NOT fire
   on other tool calls (`Read`, `Bash`, `Edit`, etc.).
2. A pattern containing extractable tokens produces an
   `additionalContext` block with the query and at least the L0
   locked beliefs.
3. A pattern with no extractable tokens (pure glob like
   `"**/*.rs"`, single-letter regex like `"\\b"`) is a no-op; the
   hook emits no `additionalContext`.
4. Re-running the hook with the same payload produces the same
   `additionalContext` block (deterministic given store state).
5. With an empty store: hook emits the "no matching beliefs"
   sentinel rather than empty stdout.
6. Median latency on a 10k-belief store, 5-token query ≤ 50 ms;
   p95 ≤ 200 ms. Verified by a regression test.
7. Store-missing / locked / `retrieve()`-raises conditions exit
   silently with a single stderr line; the tool call itself is not
   affected.
8. `aelf setup --search-tool` writes the hook config and
   `aelf setup --no-search-tool` removes it. Idempotent in both
   directions.

## Test plan

- `tests/test_search_tool_hook.py` covering criteria 1–5 + 7–8.
- `tests/regression/test_search_tool_latency.py` for criterion 6.
  Uses `time.perf_counter`, runs N=20, asserts median + p95 against
  a fixture store with 10k beliefs.
- All deterministic. Tests use `:memory:` store with synthetic
  beliefs; the regression test uses a temp on-disk store.
- Wall-clock budget: < 2 s per test (the regression test is the
  longest at ~1 s for N=20 retrievals).

## Out of scope

- LLM-augmented query expansion. v1.3.0 ships the mechanical
  token-OR-join only. Smarter query rewriting (synonyms, entity
  extraction) is a v1.4+ candidate.
- Read tool / WebFetch / WebSearch matching. v1.3.0 ships
  `Grep|Glob` only — those are the two tools whose `tool_input`
  cleanly maps to a search query. Other matchers can be added once
  the hook surface is validated in production.
- Hook-side caching. The retrieval layer's existing `RetrievalCache`
  is opt-in for callers; the hook does not maintain its own.
  Repeated identical queries within a session would benefit from
  cache reuse, but cache lifetime across hook invocations is non-
  trivial and is deferred until production data shows a hit rate
  worth pursuing.
- Confidence threshold filtering. v1.3.0 emits all matched beliefs
  (subject to `l1_limit=10`); a min-confidence floor is a v1.4+
  candidate once the format is validated.

## What unblocks when this lands

This is the first retrieval-shaped hook on the agent's *own* tool
intent. Together with the v1.0.1 `UserPromptSubmit` retrieval hook
(which fires on user turns), it covers the two natural retrieval
trigger points: user-initiated and agent-initiated. Future
retrieval improvements (BM25F augmentation, HRR, entity-index)
benefit from both surfaces.

It also closes a CS-class observability gap: when the agent runs
`Grep "X"` and the project's belief store *does* contain a relevant
prior decision about X, the absence of that context in the agent's
view today is exactly the kind of "memory said something but it
didn't reach the action path" failure the case-studies series
documents (see CS-028 for the strongest version of this problem).
The hook closes the gap on the agent-search code path.

## Open questions

- Should `aelf setup --search-tool` be the default on install at
  v1.3.0, or stay opt-in like v1.2.0's commit-ingest? Recommendation:
  opt-in at v1.3.0; flip to default-on at v1.4.0 once a representative
  corpus shows the latency budget holds in the wild.
- Should the matcher set extend to `Read` (file path → search the
  store for beliefs about that file)? The `tool_input.file_path`
  surface differs enough that v1.3.0 keeps `Grep|Glob` only and
  defers the Read variant. Worth reserving the design space.
- Token-budget tuning: 600 was picked to roughly half the default
  user-prompt budget. Worth measuring real-world injection size
  before defaulting; could be 400 or 800 depending on observed
  signal-to-noise.
