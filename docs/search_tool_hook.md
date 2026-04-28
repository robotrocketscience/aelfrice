# Search-tool hook

**Status:** spec.
**Target milestone:** v1.2.x patch (pulled forward from v1.3.0 — it ships
independently of the v1.3 retrieval wave and validates the `PreToolUse`
retrieval surface ahead of the bigger work). Default-on candidate at v1.3.0.
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

- LLM-augmented query expansion. v1.2.x ships the mechanical
  token-OR-join only. Smarter query rewriting (synonyms, entity
  extraction) is a v1.3+ candidate (entity-index lands in v1.3.0).
- Read tool / WebFetch / WebSearch matching. v1.2.x ships
  `Grep|Glob` only — those are the two tools whose `tool_input`
  cleanly maps to a search query. Other matchers can be added once
  the hook surface is validated in production.
- Hook-side caching. The retrieval layer's existing `RetrievalCache`
  is opt-in for callers; the hook does not maintain its own.
  Repeated identical queries within a session would benefit from
  cache reuse, but cache lifetime across hook invocations is non-
  trivial and is deferred until production data shows a hit rate
  worth pursuing.
- Confidence threshold filtering. v1.2.x emits all matched beliefs
  (subject to `l1_limit=10`); a min-confidence floor is a v1.3+
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
  v1.2.x, or stay opt-in like v1.2.0's commit-ingest? Recommendation:
  opt-in at v1.2.x; flip to default-on at v1.3.0 once a representative
  corpus shows the latency budget holds in the wild.
- Should the matcher set extend to `Read` (file path → search the
  store for beliefs about that file)? The `tool_input.file_path`
  surface differs enough that v1.2.x keeps `Grep|Glob` only and
  defers the Read variant. Worth reserving the design space.
- Token-budget tuning: 600 was picked to roughly half the default
  user-prompt budget. Worth measuring real-world injection size
  before defaulting; could be 400 or 800 depending on observed
  signal-to-noise.

---

## Bash extension (v1.5.0, #155)

**Status:** spec.
**Target milestone:** v1.5.0.
**Dependencies:** v1.2.x search-tool hook (this document § Design).
Reuses the `aelfrice.hook_search_tool` entry point, the
`retrieve()` plumbing it already calls, and the v1.5.0 BM25F lane
(opt-in via `bm25f_enabled`; #148).
**Risk:** medium. Bash hooks fire many times per turn; a poorly
budgeted matcher would make the agent feel slow without the user
ever invoking a search tool directly.

### Decision summary

The v1.2.x hook ships matcher `Grep|Glob` only (this doc §
Design). The carryover question from the 2026-04-27 design
discussion was whether to extend to `Bash` for shell commands
that carry the same search intent (`grep`, `rg`, `find`, `fd`).

**Decision: extend to a narrow allowlist of search-shaped Bash
commands behind a separate opt-in flag.** Default-OFF at v1.5.0;
default-on flip is gated on telemetry showing the latency and
injection-noise budgets hold (§ AC3 below).

### Allowlist

The matcher fires only when `tool_name == "Bash"` AND the parsed
command matches one of:

| command | maps to | query field |
| --- | --- | --- |
| `grep` / `egrep` / `fgrep` | Grep | first non-flag positional after the pattern recogniser |
| `rg` / `ripgrep` | Grep | first non-flag positional after the pattern recogniser |
| `ack` | Grep | first non-flag positional after the pattern recogniser |
| `find` | Glob | argument to `-name` / `-iname` if present, else skip |
| `fd` / `fdfind` | Glob | first non-flag positional after the pattern recogniser |

`ls <path>` is **explicitly excluded.** The "what's here" reflex
fires too frequently and the path token rarely carries belief-
worthy intent. Reconsider after telemetry on the v1.5.0 surface.

`cd`, `cat`, `head`, `tail`, `wc`, `sort`, `uniq`, and any pipe
to such are **excluded.** They are state changes, output
manipulation, or aggregation — none carry retrieval intent.

Anything not in the allowlist silent-skips. There is no
fall-through to "match all Bash."

### Per-command parsing

Each allowlisted command is its own micro-parser. The parser:

1. Tokenises `tool_input.command` on whitespace (no shell
   evaluation; the hook never executes the command).
2. Skips the leading prefix (e.g., `cd foo &&`, `nohup`, env
   assignments like `RUST_LOG=trace rg ...`) until the first
   token matches an allowlisted command name (or its absolute /
   relative path basename — `/usr/bin/grep` matches `grep`).
3. Walks remaining tokens; flag tokens (leading `-`) and their
   parameter values (where the flag takes one) are skipped per a
   per-command flag-value table.
4. Returns the first remaining positional as the query, or `None`
   if no positional remains.

For `find`, the rule is stricter: the parser ONLY emits a query
when an `-name` or `-iname` argument is present. `find . -type f`
contributes no signal and is silent-skipped.

Pipelines, command substitutions, redirections, here-docs, and
shell control flow (`for`, `while`, `if`) abort the parser:
returning `None` on any unrecognised structural token. The
parser is intentionally narrow; failure to parse must
silent-skip, never fire on a wrong query.

The flag-value table per command lives in
`aelfrice.hook_search_tool` as a module-level `Final` mapping;
each entry is a tuple `(takes_arg: bool, ...)`. The table is
unit-tested per-command (§ Test plan).

### Per-turn fire cap

The Bash matcher introduces a fire cap of **3 fires per session
turn** to prevent pipeline storms (e.g., a `for` loop that runs
`rg` ten times). State is stored in a per-process counter keyed
by `session_id` from the hook payload. Once the cap is reached
the matcher silent-skips for the rest of the turn.

The 3-cap is conservative; can be raised after telemetry. The
cap does NOT apply to the existing `Grep|Glob` matcher (which
fires once per direct tool call and rarely loops).

### Token budget

Bash matches are auxiliary signals — lower confidence than a
direct `Grep` / `Glob` invocation, where the agent has already
formulated the query as the tool input. Bash extraction is one
parse hop further from the agent's intent, so the budget shrinks
correspondingly:

| matcher | `token_budget` | `l1_limit` |
| --- | --- | --- |
| `Grep|Glob` (v1.2.x) | 600 | 10 |
| `Bash` allowlist (v1.5.0) | **300** | **5** |

Half the v1.2.x figures. Tunable per `[search_tool_hook]
bash_token_budget` / `bash_l1_limit` keys in `.aelfrice.toml`
once production data lands.

### BM25F interaction (v1.5.0 #148)

The hook continues to call `retrieve()`. When the project has
opted into BM25F via `[retrieval] bm25f_enabled = true` (or the
`AELFRICE_BM25F` env), the Bash-matcher fires use the same lane
— no separate plumb-through. Default-off at v1.5.0 so the
combined surface is conservative until telemetry validates both
levers.

### Hook payload extension

The Bash-matcher path keys its emitted `additionalContext` block
by the parsed query AND the original command (truncated to 80
chars), so the agent can tell which Bash invocation triggered
the injection:

```
<aelfrice-search query="..." source="bash:rg" cmd="rg -t py foo src/">
  ...results...
</aelfrice-search>
```

The `Grep|Glob` matcher emits no `source` attribute (preserving
v1.2.x output for unchanged callers).

### Telemetry

The default-on flip is gated on two metrics, captured by the
hook to a per-project ring buffer at
`<project>/.git/aelfrice/telemetry/search_tool_hook.jsonl`
(append-only, capped at 1000 entries, oldest evicted):

1. **Latency p95.** Per-fire wall-clock from hook entry to
   stdout flush. Budget: same 50 ms median / 200 ms p95 contract
   as the v1.2.x matcher.
2. **Injection-noise rate.** Fraction of Bash-matcher fires
   that produced no L0 hits AND no L1 hits at confidence ≥ a
   documented floor. Budget: ≤ 30 % at the default-on flip
   threshold. Higher means the matcher is dragging the agent's
   context window with low-signal injections.

The telemetry surface lands as a small `aelfrice.telemetry`
module addition; reuse the existing per-project DB locator
(`aelfrice.cli.db_path`) so the file lives next to the brain
graph and inherits its gitignore boundary.

`aelf doctor` gains a `search_tool_hook telemetry` section that
prints the rolling p50 / p95 latency and the noise rate.

### Failure modes

Same contract as the v1.2.x hook: any failure path
(unrecognised command, parser error, fire-cap reached, store
missing, retrieval exception) returns silently with no
`additionalContext`. The Bash tool runs unaffected. The
principle is unchanged: **the hook may NEVER cause a Bash
command to feel broken.**

### Opt-in surface

`aelf setup --search-tool-bash` writes the Bash matcher
configuration. Default on fresh install: opt-in at v1.5.0,
mirroring the v1.2.x `--search-tool` rollout.

`aelf setup --no-search-tool-bash` removes it. Independent of
`--search-tool` (the v1.2.x Grep|Glob path) so a user can run
either, both, or neither.

### Acceptance criteria

1. The Bash matcher fires only on allowlisted commands and
   silent-skips on every other Bash invocation. Verified by a
   parser-level unit test per command in the allowlist plus a
   property test that randomly generated non-allowlisted
   commands never fire.
2. Per-command query extraction is exact for the documented
   shape (e.g., `grep -r foo src/` → `"foo"`). Each command has
   ≥ 5 unit tests in `tests/test_search_tool_hook_bash.py`
   covering: bare invocation, with flags, with flag values, with
   pipelines (must skip), with command substitution (must skip).
3. Default-on flip is gated on telemetry showing latency p95
   ≤ 200 ms AND injection-noise rate ≤ 30 % over a documented
   sample size (≥ 200 fires from a representative corpus).
   Until both clear, default stays OFF.
4. Allowlist is narrow: no fall-through to "match all Bash". A
   regression test asserts that an unmodified `bash -c 'something'`
   payload never produces an `additionalContext` block.
5. Per-turn fire cap (3) holds: a payload that triggers four
   allowlisted fires within one `session_id` produces three
   `additionalContext` blocks and one silent skip.
6. Hook output for the Bash matcher carries `source="bash:<cmd>"`
   and the truncated `cmd` attribute. Hook output for the
   v1.2.x Grep|Glob matcher is unchanged (no `source`
   attribute).
7. `aelf setup --search-tool-bash` writes the hook config and
   `aelf setup --no-search-tool-bash` removes it. Idempotent in
   both directions, independent of the v1.2.x flag.
8. `aelf doctor` surfaces the rolling latency and noise-rate
   telemetry summary. Empty / missing telemetry file prints
   "no fires recorded" rather than raising.

### Test plan

- `tests/test_search_tool_hook_bash.py`: per-command parser
  unit tests (criteria 1–2, 4–5).
- `tests/test_search_tool_hook_bash_integration.py`: end-to-end
  hook invocation against a `:memory:` store, asserting the
  emitted block shape and the `source` attribute (criterion 6).
- `tests/test_aelf_setup_search_tool_bash.py`: idempotent
  install + uninstall (criterion 7).
- `tests/test_aelf_doctor_telemetry.py`: doctor surface
  (criterion 8).
- `tests/regression/test_search_tool_bash_latency.py`: per-fire
  p95 budget on a 10k-belief store. Reuses the v1.2.x latency
  fixture conventions.

All deterministic, in-memory store, < 200 ms each except the
latency regression. Wall-clock cap matches the existing test
suite policy.

### Out of scope

- Detecting search intent in arbitrary shell pipelines (`grep
  foo file | sed ... | head`). The parser is allowlist-only;
  pipelines abort. Reconsider after the v1.5.x telemetry
  pass.
- LLM-augmented query rewriting on the parsed query. The Bash
  matcher reuses the same mechanical token-OR-join the v1.2.x
  hook uses; smarter expansion lands jointly with v2.0 HRR.
- Writing belief observations from Bash hook fires. The hook
  is read-only, same contract as v1.2.x.
- Cross-tool deduplication. A turn that fires the v1.2.x
  matcher AND the Bash matcher emits two blocks; the agent
  reads both. Dedup is a v1.6.x candidate after telemetry on
  the actual collision rate.

### What unblocks when this lands

The Bash matcher closes the v1.2.x gap where the agent reaches
for `rg` or `find` instead of `Grep` / `Glob` (e.g., when an
agent has been trained on shell habits, or when the user's
prompt style steers toward bash commands). With both matchers
running, the agent's search-shaped tool intent is covered
regardless of which surface it picks.

This is also the v1.5.x prerequisite for the v1.6+ proposal to
consolidate the hooks on a single intent-extraction layer (one
hook, multiple tool-name → query-field maps), which only makes
sense once the per-command-per-tool surface has been validated
in production.

### Open questions deferred to implementation

- Should the per-turn fire cap be configurable, or is 3 a
  permanent invariant? Lean toward configurable
  (`bash_fire_cap_per_turn` in `.aelfrice.toml`) so users can
  tighten or loosen without a code change.
- Do we ship the telemetry file at v1.5.0 or wait for v1.5.x?
  Recommendation: ship at v1.5.0. The default-on flip is the
  whole point of the gate, and it can't happen without
  telemetry data flowing.
- Should `aelf setup --search-tool` imply `--search-tool-bash`?
  Recommendation: no at v1.5.0. Keep the surfaces independent
  until production data shows they should rise/fall together.
