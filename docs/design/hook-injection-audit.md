# Hook injection audit

**Issue:** [#220](https://github.com/robotrocketscience/aelfrice/issues/220)
**Scope:** audit and characterisation only — no code changes.
**Status:** investigation complete; recommendations bounded to tuning.
**Date:** 2026-04-28

---

## Overview

Two distinct hook paths inject aelfrice belief context into live
sessions:

1. **UserPromptSubmit — `<aelfrice-memory>` block.** Fires on every
   user message. Runs a full four-layer retrieval against the raw
   user prompt and injects a ranked belief block at the top of the
   turn.

2. **PreToolUse on Grep/Glob — `<aelfrice-search>` block.** Fires
   before every `Grep` and `Glob` tool call. Lifts the search pattern
   from `tool_input.pattern`, strips metacharacters, and runs a
   reduced-budget retrieval against the extracted token set.

Both hooks share the same underlying retrieval stack
(`aelfrice.retrieval.retrieve`), the same DB path resolution
(`aelfrice.cli.db_path`), and the same SQLite FTS5 index. They differ
in trigger surface, query construction, and injected token budget.

The audit also found a **query-construction mismatch** (§ Relevance
filter audit) that affects both hooks and explains the cross-domain
contamination reported in the issue.

---

## Hook 1: UserPromptSubmit — `<aelfrice-memory>`

### Entry point

| Item | Detail |
|---|---|
| Settings key | `UserPromptSubmit` → `command: /Users/thelorax/.local/bin/aelf-hook` |
| Config file | `~/.claude/settings.json` |
| Python entry | `aelfrice.hook:main` → `user_prompt_submit()` |
| Source | `site-packages/aelfrice/hook.py` line 75 |

The `~/.local/bin/aelf-hook` shim is a one-liner that calls
`aelfrice.hook:main`. The hook receives the full `UserPromptSubmit`
JSON payload on stdin from the harness.

### Behavior

1. Reads and JSON-parses the `UserPromptSubmit` payload from stdin
   (`hook.py` lines 101–133).
2. Extracts `payload["prompt"]` — the raw user message exactly as
   typed (`hook.py:_extract_prompt`, line 118).
3. Calls `hook_search.search_for_prompt(store, prompt,
   token_budget=1500)` (`hook.py` line 139).
   - `search_for_prompt` calls `retrieval.retrieve()` then writes
     a `feedback_history` row per hit with `valence=0.1,
     propagate=False` (`hook_search.py` lines 79–80).
4. Formats non-empty results as:
   ```
   <aelfrice-memory>
   [locked] <id>: <content>
           <id>: <content>
   ...
   </aelfrice-memory>
   ```
   (`hook.py:_format_hits`, line 147).
5. Writes the block to stdout; the harness prepends it to the turn.

**Corpus:** `db_path()` in `cli.py` (line 214) resolves as:
`$AELFRICE_DB` → `<git-common-dir>/aelfrice/memory.db` → `~/.aelfrice/memory.db`.
When the cwd is inside any git work-tree, the DB lives in that repo's
`.git/` directory. The hook changes directory to the cwd from the
payload before calling `db_path()`, so per-project DBs are used in
multi-project workflows. However, when the cwd is not inside a git
repo, or when `$AELFRICE_DB` is set, **all sessions share a single
global DB**, which is the primary cross-domain contamination vector.

**Retrieval layers invoked at default settings:**
- L0: all locked beliefs (always returned, never trimmed).
- L2.5: entity-index lookup (`entity_index_enabled=True` by default).
- L1: FTS5 BM25 search, up to 50 results, budget-trimmed from tail.
- L3 BFS: default OFF.
- Posterior rerank: `posterior_weight=0.5` by default.

**Token budget:** `DEFAULT_HOOK_TOKEN_BUDGET = 1500` (`hook.py` line 48).
This is below the user-facing CLI default (2000) to leave headroom for
other concurrent UserPromptSubmit hooks, but well above the PreToolUse
hook budget.

### Relevance filter

The raw user prompt is passed through `_escape_fts5_query()` in
`store.py` (line 219) before the FTS5 query runs. That function:
1. Splits the string on whitespace.
2. Wraps each token in double quotes.
3. Joins with spaces (FTS5 implicit AND).

A three-word prompt `"fix the hook"` becomes `"fix" "the" "hook"` —
**an AND query requiring all three tokens to appear in a single
belief**. Common short words such as "the", "is", "in", "a" are not
stripped, so they contribute AND constraints that match a wide swathe
of the corpus. In a large global store this produces many off-domain
hits ranked by BM25 proximity, not by topic.

**Observed gap:** prompts that contain generic English words (e.g.,
"add a new feature", "check the status", "run the tests") will match
beliefs that happen to contain those same common words, regardless of
domain. With a 20 k-belief global store the probability of a random
belief containing "the" approaches 1.0; the effective filter is
reduced to the less-common words in the prompt.

### Overlap with `aelf rebuild`

`aelf rebuild` (PreCompact hook) runs `rebuild_v14()` against a
transcript window. It is a distinct code path: rebuild queries by
session-scoped turn content, not by the live user prompt. The two
hooks are complementary:

- UserPromptSubmit = per-prompt, query = live prompt text, budget = 1500.
- PreCompact rebuild = per-compaction, query = recent turn text, budget = 4000 (configurable).

There is no de-duplication between them. A belief can appear in both
the per-prompt `<aelfrice-memory>` block and the compaction rebuild
block during the same session. Whether this is redundant depends on
whether the compaction has fired for this session.

---

## Hook 2: PreToolUse on Grep/Glob — `<aelfrice-search>`

### Entry point

| Item | Detail |
|---|---|
| Settings key | `PreToolUse` → matcher `Grep\|Glob` → `command: /Users/thelorax/.claude/hooks/aelfrice-search-tool-inject.sh` |
| Config file | `~/.claude/settings.json` |
| Shell script | `~/.claude/hooks/aelfrice-search-tool-inject.sh` |
| Python entry | `aelfrice.hook_search_tool:main` |
| Source | `site-packages/aelfrice/hook_search_tool.py` line 705 |

The shell script reads the payload from stdin, extracts `session_id`
and `cwd` for logging, then calls `aelf search "$QUERY"` via the
`$AELF_BIN` binary (`aelfrice-search-tool-inject.sh` lines 57–58).
The Python module `hook_search_tool.py` is the canonical
implementation; the shell wrapper is kept for latency-logging and
fallback purposes.

**Note:** The settings.json at `~/.claude/settings.json` wires the
**shell script**, not the Python entry point directly. The shell
script calls `aelf search` (the CLI subcommand), not the Python hook
module. The Python module `hook_search_tool.py` implements the same
logic and is registered as a console script entry point for direct
invocation; whether the shell wrapper or the Python module is the
actual runtime path depends on which is resolved by `$AELF_BIN`.

### Behavior

1. Receives the `PreToolUse` JSON payload (stdin).
2. Checks `tool_name in {"Grep", "Glob"}` (`hook_search_tool.py`
   line 491).
3. Extracts `tool_input.pattern` (`hook_search_tool.py` line 505).
4. Tokenises with `r"[A-Za-z][A-Za-z0-9_-]{2,}"` — alphanumeric
   tokens of 3+ chars (`hook_search_tool.py` lines 59–61).
5. Takes the first 5 tokens and joins them with ` OR `
   (`hook_search_tool.py` line 514):
   ```python
   return " OR ".join(tokens[:QUERY_TOKEN_LIMIT])
   ```
6. Calls `retrieve(store, query, token_budget=600, l1_limit=10)`.
7. Emits an `additionalContext` JSON block with the `<aelfrice-search>`
   envelope.

**Explicit no-match sentinel:** when `retrieve()` returns no results,
the hook emits:
```
<aelfrice-search query="...">no matching beliefs in store; the tool result will fill the gap</aelfrice-search>
```
This fires unconditionally on every Grep/Glob call with extractable
tokens — the "no-op path" in the issue is a real emission, not
silence.

**Token budget:** `INJECTED_TOKEN_BUDGET = 600`, `INJECTED_L1_LIMIT = 10`
(`hook_search_tool.py` lines 48–51). Roughly half the UserPromptSubmit
budget, reflecting that this is auxiliary context.

**v1.5.0 Bash extension:** the same module also handles `PreToolUse`
on `Bash` for allowlisted search commands (`grep`, `rg`, `find`,
`fd`), with a halved budget (300 tokens, 5 results) and a per-turn
fire cap of 3. The Bash matcher is wired separately; the currently
deployed `settings.json` only registers `Grep|Glob` at
`~/.claude/settings.json`.

### Relevance filter

The hook generates a multi-token query string such as
`"hook OR injection OR payload"` and passes it to `retrieve()`, which
ultimately calls `store.search_beliefs(query)` →
`_escape_fts5_query(query)`.

**Critical mismatch:** `_escape_fts5_query()` (`store.py` line 219)
tokenises on whitespace and quote-wraps each token. The string
`"hook OR injection OR payload"` splits into five whitespace tokens:
`["hook", "OR", "injection", "OR", "payload"]`. Each gets wrapped:
`"hook" "OR" "injection" "OR" "payload"`. In FTS5, whitespace between
phrase-quoted tokens is implicit AND, so the effective query is:

```
beliefs_fts MATCH '"hook" "OR" "injection" "OR" "payload"'
```

This requires a belief to contain **all five terms including the
literal word "OR" twice**. Since very few beliefs contain the word
"OR" as a standalone token (it is typically part of code or prose but
rarely a standalone English word in belief content), most multi-token
hook queries produce **zero L1 hits from the FTS5 layer**. The L0
locked beliefs, which bypass the FTS5 search entirely, and any L2.5
entity-index hits are still returned.

The end result is the inverse of the intended OR semantics: the hook
intends "find beliefs mentioning any of these tokens" but the store
layer interprets "find beliefs containing all of these tokens plus the
word OR". This explains both the no-results path and the cross-domain
behaviour:

- For typical code-search patterns, the OR mismatch likely means only
  L0 locked beliefs surface (which are unconditionally returned).
- L0 locked beliefs are global (not scoped to the current query
  topic), so a session working on aelfrice code will surface every
  locked belief regardless of the search pattern.

The shell script path (`aelfrice-search-tool-inject.sh`) calls
`aelf search "$QUERY"` with the same OR-joined string, hitting the
same `_escape_fts5_query` mismatch.

### Fan-out on multi-tool turns

A single investigation pass that fires 10 Grep calls also fires 10
PreToolUse hook invocations. Each invocation:
- Spawns a Python interpreter (cold-start cost ~20–80 ms).
- Opens the SQLite DB.
- Runs the FTS5 query.
- Emits an `additionalContext` block (empty or non-empty).

Even on the no-match path, the sentinel string
`"no matching beliefs in store; your query is not in the indexed graph"`
is emitted. With 10 tool calls that is 10 × (sentinel string ≈ 80
chars ≈ 20 tokens), so ~200 tokens of overhead per investigation pass
even when the store has nothing relevant.

---

## Token budget characterisation

### Per-turn cost structure

| Hook | Trigger | Budget | L0 always included? | Typical L1 hits |
|---|---|---|---|---|
| UserPromptSubmit | Every user message | 1500 tokens | Yes | 0–20 (AND filter; common words bloat) |
| PreToolUse Grep/Glob | Every Grep/Glob tool call | 600 tokens | Yes | 0 (OR mismatch makes L1 effectively dead) |

### Combined turn cost

A representative aelfrice-internal coding turn with:
- 1 user message → 1 UserPromptSubmit injection
- 8 Grep/Glob calls → 8 PreToolUse injections

Worst-case (all inject at budget cap):

| Hook | Invocations | Max tokens each | Max total |
|---|---|---|---|
| UserPromptSubmit | 1 | 1500 | 1500 |
| PreToolUse | 8 | 600 | 4800 |
| **Combined** | | | **6300** |

In practice, the OR mismatch means PreToolUse L1 hits are near-zero
and actual injection is dominated by L0 locked beliefs (typically 200–800
tokens depending on how many beliefs the user has locked). On a store
with 20 k beliefs and the observed cross-domain behaviour, the
UserPromptSubmit hook's AND query likely hits 0–5 L1 results for
tightly-scoped coding prompts and more for generic-vocabulary prompts.

### No-op cost

Even the no-match path is not free. The PreToolUse hook emits a
sentinel `<aelfrice-search>` block on every call, and each block
occupies ~20 tokens. On a 10-Grep investigation that is ~200 tokens
consumed for "no match" feedback — structurally necessary for the
"consulted but empty" signal, but unbounded as the Grep count grows.

---

## Recommended tunings

The following are bounded, targeted changes to the existing
implementation. No design rewrites; no public CLI surface changes.

### 1. Fix the OR / AND mismatch in `hook_search_tool.py`

**File:** `src/aelfrice/hook_search_tool.py`, `_extract_query()`
(line 496 in installed package) and `_extract_bash_query()` (line
312).

**Problem:** the hook builds `" OR ".join(tokens)` and passes it to
`retrieve()`, which calls `store.search_beliefs()` →
`_escape_fts5_query()`. The OR keyword is treated as a literal
whitespace-separated token, so the FTS5 query ANDs all terms
including the literal string "OR". Intended: OR semantics. Actual:
requires all tokens plus "OR" to be present.

**Fix:** either (a) bypass `_escape_fts5_query` by using FTS5 raw
syntax (quote each token, join with ` OR `, no outer escaper), or
(b) pass each token as a separate retrieve call and union the results.
The simplest change is to keep the five-token extraction but join with
a single space (implicit FTS5 AND on the real tokens, no OR keyword)
— this changes semantics from "match any" to "match all" but at least
removes the literal "OR" requirement. A proper fix is to have
`store.search_beliefs` accept an `operator` argument or expose a raw
FTS5 query path so callers that want OR semantics can ask for them.

**Impact:** removes the single largest driver of the no-match path on
multi-token patterns. Will surface more L1 results per Grep/Glob call,
potentially increasing per-call injection volume. Should be paired
with recommendation 3 (min-score threshold).

### 2. Add a domain filter or project-scope guard to `hook.py:_retrieve_and_format`

**File:** `src/aelfrice/hook.py`, `_retrieve_and_format()` (line
136) / `_open_store()` (line 157).

**Problem:** when the cwd is not inside a git repo (or when
`$AELFRICE_DB` is set to a global path), `db_path()` returns the
global `~/.aelfrice/memory.db`. All sessions across all projects then
share the same corpus, which explains the cross-domain contamination
(unrelated-project tables, trivia snippets surfacing on aelfrice coding
tasks).

**Fix:** expose the `cwd` from the `UserPromptSubmit` payload and pass
it to `db_path(cwd=...)` when available (the `hook_search_tool.py`
code at line 617 already does this for PreToolUse). The UserPromptSubmit
hook currently does not use `cwd` from the payload — adding it would
scope per-prompt retrieval to the project-local DB the same way the
PreToolUse hook already does.

**Specific lines to change:** `hook.py:_retrieve_and_format` (line
136) would need to receive the `cwd` extracted from the payload and
pass it through to `_open_store(cwd)`. `_open_store` (line 157) would
need to accept and pass `cwd` to `db_path`.

**Impact:** for single-project sessions this is a no-op. For users
with a large global `~/.aelfrice/memory.db` accumulated from multiple
projects, it would eliminate inter-project crosstalk in sessions where
the cwd is inside a git repo.

### 3. Add a minimum BM25 score threshold to the L1 filter in `retrieval.py`

**File:** `src/aelfrice/retrieval.py`, `_l1_hits()` (line 646).

**Problem:** the current retrieval returns up to 50 L1 beliefs trimmed
only by token budget, not by a relevance floor. On a 20 k-belief store
the bottom of the ranked list may have very low BM25 scores (i.e.,
beliefs that share only common words with the query). These are the
most likely source of cross-domain unrelated-project hits surfacing
on unrelated coding tasks.

**Fix:** add an optional `min_bm25_score: float | None = None`
parameter to `retrieve()` (and `_l1_hits()`). When set, beliefs whose
BM25 score exceeds `min_bm25_score` (FTS5 BM25 is non-positive; the
threshold would be a negative number like `-5.0`, where values closer
to 0.0 are more relevant) are dropped before budget accounting. A
conservative default of `None` preserves existing behaviour; users
can opt in via `.aelfrice.toml` `[retrieval] min_bm25_score = -3.0`.

**Specific lines to change:** `_l1_hits()` in `retrieval.py` (line
646) already has access to the BM25 scores via
`store.search_beliefs_scored()` (called when `posterior_weight > 0`).
The filter would be a one-line `if raw > min_bm25_score: continue`
after line 694 (the `for b, raw in beliefs` loop). For the non-scored
path (`posterior_weight == 0.0`) the scored variant would need to be
used, or the no-score path can remain unfiltered as it is today.

**Impact:** reduces L1 payload size and cross-domain noise on general
prompts. Threshold calibration requires looking at real BM25 scores in
the store; `aelf doctor` or a one-time `aelf search --scored` run can
expose the score distribution.

### 4. Gate PreToolUse injection on minimum pattern length

**File:** `~/.claude/hooks/aelfrice-search-tool-inject.sh` (line 36)
and `src/aelfrice/hook_search_tool.py:_extract_query` (line 496).

**Problem:** patterns like `"*.py"` or `"**/*.rs"` produce an empty
token set (correct no-op) but patterns like `"src"` (3-char minimum
passes) produce a single-token query that is unlikely to be
meaningfully scoped. The hook fires, runs retrieval, and emits a block
for every such call.

**Fix:** require at least two extractable tokens before running
retrieval. A single 3-char token is too broad a signal on a 20 k
corpus. This is a one-line change: `if len(tokens) < 2: return None`
after line 511 in `hook_search_tool.py`.

**Impact:** reduces hook firings on short or file-extension-only
patterns. Expected to cut ~20–30% of invocations on typical coding
sessions where Grep is used with short pattern fragments.

### 5. Cap the no-match sentinel injection

**File:** `src/aelfrice/hook_search_tool.py:_format_results` (line
517).

**Problem:** the no-match path emits a sentinel string on every call
regardless of result count. On a 10-Grep investigation pass with no
relevant beliefs, this adds ~200 tokens of boilerplate context.

**Fix:** make the no-match sentinel opt-in via a `emit_no_match:
bool = True` kwarg on `_format_results`, defaulting to `False` for
the PreToolUse hook path. The sentinel is useful for the agent
("consulted but empty") but its per-call frequency is hard to bound
without a cap. Alternatively, suppress the sentinel after N no-match
fires per session turn (paralleling the Bash per-turn fire cap).

**Specific lines to change:** `_format_results()` line 555 in
`hook_search_tool.py` — the `if not lines:` branch that returns the
sentinel string.

**Impact:** reduces per-call overhead on the no-match path. Estimated
savings: 20–80 tokens per no-match invocation depending on the query
string length in the `query=` attribute.

---

## Out of scope

Per the issue:

- **Disabling the hooks.** The injection is confirmed wanted by the
  user. This audit debugs and tunes, does not remove the hooks.
- **Re-architecting the brain-graph schema.** The SQLite FTS5 table
  structure and belief-entity schema are unchanged.
- **Changes to the public CLI surface.** `aelf search`, `aelf lock`,
  `aelf rebuild`, and related commands are unchanged. Recommendations
  are internal to the hook and retrieval layer.

---

## References

- `~/.claude/settings.json` — hook registrations for UserPromptSubmit
  and PreToolUse.
- `~/.claude/hooks/aelfrice-search-tool-inject.sh` — PreToolUse shell
  wrapper.
- `~/.local/bin/aelf-hook` — UserPromptSubmit shim.
- `site-packages/aelfrice/hook.py` — UserPromptSubmit Python
  implementation.
- `site-packages/aelfrice/hook_search_tool.py` — PreToolUse Python
  implementation (canonical).
- `site-packages/aelfrice/hook_search.py` — feedback audit wrapper
  used by UserPromptSubmit.
- `site-packages/aelfrice/retrieval.py` — four-layer retrieval stack
  (L0/L2.5/L1/L3).
- `site-packages/aelfrice/store.py:_escape_fts5_query` (line 219) —
  FTS5 query construction where the OR mismatch originates.
- `site-packages/aelfrice/cli.py:db_path` (line 214) — DB path
  resolution (per-project vs. global).
- `docs/design/search_tool_hook.md` — PreToolUse hook design spec.
- `docs/design/context_rebuilder.md` — PreCompact hook design spec.
