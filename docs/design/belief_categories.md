# Belief categories — keyword-triggered rule grouping (#1126)

Umbrella: [#1126](https://github.com/robotrocketscience/aelfrice/issues/1126). Status: v1 shipped (soft `UserPromptSubmit` injection lane).

## Problem

Rule-shaped beliefs (repo conventions, git-push rules, prose-style rules) are enforced today only by (a) being *locked*, which injects them unconditionally on every turn, or (b) living in a static `CLAUDE.md` / `AGENTS.md`. Neither injects **the right rules at the right moment**. A user wants to group beliefs into categories and have a category's rules surface *when they are relevant* — e.g. the git-workflow rules when the prompt says "commit and push".

## Prior-art constraint

The enforcement triad ([#199](https://github.com/robotrocketscience/aelfrice/issues/199), `v2_enforcement.md`) already litigated hook-level enforcement:

- **H1** (infer rule-shaped directives from prose) — deferred, bench-gated. `directive_detector.py` exists but is unwired.
- **H2** (a hook that runs a predicate to enforce) — dropped on shell-injection grounds (predicate text derives from belief content).
- **H3** (score locked rules vs the prompt) — superseded by [#379](https://github.com/robotrocketscience/aelfrice/issues/379): locks are the always-injected pool (L0).

Every aelfrice hook is fail-open / exit-0 and none emit `deny`/`ask`. This feature is therefore **advisory injection, not enforcement**. A hard-block lane is explicitly out of v1 scope.

## v1 design

### Categories (starter set)

`aelf category init` seeds five: `repo-rules` (always-on / locked), `git-workflow` (keyword / locked), `secrets-and-safety` (tool-match + always-on / locked), `prose-and-docs` (keyword / advisory), `testing` (file-glob or keyword / mixed). Design invariant used as a review check: always-on categories are almost entirely locked; keyword/tool categories hold the mix.

### Data model (two additive tables)

Additive `CREATE TABLE IF NOT EXISTS` in `store._SCHEMA` — invisible to the destructive-only `migration-policy-check` gate, present on every fresh store.

```sql
categories(name PK, always_on INT, trigger_json TEXT, default_lock TEXT, created_at TEXT)
belief_categories(belief_id → beliefs.id, category_name → categories.name,
                  created_at, PK(belief_id, category_name))  -- FK CASCADE both sides
```

`trigger_json` is one read-once JSON blob per category — `{keywords, tool_globs, file_globs}` — the same pattern as `onboard_sessions.candidates_json`. Membership is a many-to-many join modeled on `belief_documents` (#435); a belief may belong to several categories. There is **no** `category` column on `Belief` — the dataclass is untouched.

### Matching (`aelfrice.category`)

Pure, deterministic, stdlib-only (#605): keyword phrases compile to a case-insensitive, word-boundary alternation (internal whitespace relaxed to `\s+`); tool/file globs use `fnmatch`. `match_prompt(prompt, categories)` returns the categories that are `always_on` or keyword-matched, de-duplicated by name, in name-ASC order. No embeddings, no model call, no clock.

### Rerank-on-trigger lane (`hook._apply_category_boost`)

On `UserPromptSubmit`, when `category.is_enabled(toml)` (default-off; env `AELFRICE_BELIEF_CATEGORIES` > `[belief_categories] enabled` > `false`), a fired category does **not** emit a separate block. Instead it **reranks the single retrieval output**: its member beliefs are lifted to the top of the `<aelfrice-memory>` hits, a bounded set (`CATEGORY_BOOST_MAX_EXTRA`) of members retrieval missed is pulled in, and a one-line `<category-focus>` note is prepended naming the fired categories. Deterministic: categories name-ASC (`match_prompt`), members in stable store order, de-duplicated by belief id; the un-promoted remainder keeps retrieval order. Fail-soft: on disable / no-fire / error the hits pass through unchanged.

This design is the direct result of the R&D below — a separate injected block double-injected whatever retrieval already returns, so reranking the one block is the correct mechanism.

Only the **prompt keyword** lane (plus `always_on`) is wired in v1. `tool_globs` / `file_globs` are parsed, stored, and matched (`command_hit` / `paths_hit`, unit-tested) but the `PreToolUse` wiring that consumes them is a follow-up.

### R&D that shaped the design

The first cut emitted a separate `<belief-category-rules>` block ahead of the retrieval body. Empirical rounds against the real hook / a copy of a 45k-belief store refuted that approach:

- **R1 — double-injection (confirmed).** A locked rule in a fired keyword category appeared **twice**: once in the category block, once in the L0 `<aelfrice-memory>` block. All seed categories default `locked`, so every fired seed category duplicated content L0 already injects.
- **R2 — the block is redundant (confirmed).** Across every configuration tested — fresh/aged belief, tiny/45k store, token budget 200→2400 — a category member was **already present in the retrieval output**. No realistic case surfaced net-new content. (A methodological note: small synthetic stores over-retrieve, and a freshly-inserted belief ranks #0 for *any* query — both masked the effect until corrected for.) So the block's value is prioritising + labelling the one retrieval block, not adding a second.
- **R3 — keyword recall (quantified).** Literal-phrase matching missed ~39% of natural phrasings ("ship it", "get this merged", "land this change"). Expanding the `git-workflow` seed keyword set cut this to ~17%; the residual is the inherent floor of deterministic literal matching (chosen over fuzzy matching for determinism, #605).

The rerank-on-trigger design resolves R1 and R2 (one injection, no duplicate) and delivers the original "right rule at the right moment" intent by *ordering + labelling* rather than *adding*.

### CLI

`aelf category init|add|list|show|set-trigger|assign|unassign|delete` (visible verb, nested actions) and `aelf lock "<rule>" --category <name>`. Membership is user-driven — **no auto-classification** in v1.

## Non-goals (v1)

- **No hard `deny`/block lane.** A false-positive block wedges a real tool call; the codebase has never emitted deny. A strict-mode block is a separate future issue, default-off, with a false-positive escape hatch, and even then fail-**open** on hook error.
- **No auto-classification** of beliefs into categories.
- **No inference of rule-shaped directives from prose** (the bench-gated H1 path). Triggers are user-declared, not inferred.

## Follow-ups

- `PreToolUse:Bash` consumption of `tool_globs` (inject-on-command-match, still exit-0, no deny).
- File-glob lane wired to touched paths.
- Optional strict "block" mode behind its own default-off flag + escape hatch.
