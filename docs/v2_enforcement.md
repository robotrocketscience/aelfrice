# v2.0 evaluation: enforcement triad

Spec for issue [#199](https://github.com/robotrocketscience/aelfrice/issues/199). Substrate-cascade addendum to [`substrate_decision.md`](substrate_decision.md) (#196 ratified Option B).

Status: H3 implemented at v2.0 ([#373](https://github.com/robotrocketscience/aelfrice/issues/373)); H1 split + deferred + bench-gated ([#374](https://github.com/robotrocketscience/aelfrice/issues/374)); H2 dropped per § H2 below.

**Reading 2 picked for the SessionStart-without-prompt ambiguity (#373):** locked-belief injection moves out of SessionStart entirely and into the per-turn UserPromptSubmit hook, where a real prompt exists to score against. The SessionStart payload contains no user prompt, so any session-boot relevance signal would be a guess. The legacy all-locked SessionStart behaviour is preserved behind `[user_prompt_submit_hook] inject_all_locked = true` in `.aelfrice.toml` — a single switch that also disables UPS L0 trimming, restoring the v1.x injection contract end-to-end with no data loss.

## What's being decided

The research-line `enforcement.py` (~347-373 LOC) is three independent submodules with different risk profiles. Issue #199 lumps them as a unit. They should be specced and decided separately.

- **H1 — Directive detection.** Regex over 29 imperative verbs ("never", "always", "must", "don't", ...) + question/hedging filters. Auto-creates TODO-tagged beliefs.
- **H2 — Compliance audit.** `check_compliance(store)` evaluates predicate strings on locked beliefs: `file_contains:path:content`, `config_value:path:key:expected`, `command_succeeds:cmd`. **`command_succeeds` runs arbitrary shell.**
- **H3 — Selective injection.** Scores locked beliefs against the *current* prompt; injects top-K instead of all-locked at SessionStart.

## Substrate dependency

None for any of the three. Substrate-neutral.

## Recommendation per submodule

### H3 selective injection — ship at v2.0

1. **Highest leverage, lowest risk.** Currently every locked belief is unconditionally injected at SessionStart (`hook.py:329-398`). Users with growing locked sets see context bloat that's mostly irrelevant to the current task.
2. **Substrate-neutral scoring.** Top-K by query overlap × `posterior_mean` works on scalar Beta-Bernoulli (Option B). Locked beliefs have saturated posteriors anyway, so the scoring degenerates to query overlap — clean.
3. **Reversible.** If selective injection misses a locked belief the user expected to see, fallback is a `--inject-all` flag. No data loss.
4. **Effort: small.** ~80 LOC of the ~347 total in `enforcement.py`. Self-contained.

### H1 directive detection — defer to v2.x with benchmark gate

1. **High false-positive risk on coding prompts.** "never push directly to main" is a directive; "I never push to main when I'm tired" is not. The 29-verb regex doesn't disambiguate. Auto-creating TODO beliefs from every imperative-sounding sentence will pollute the belief graph.
2. **No benchmark backs the precision/recall tradeoff.** Research line shipped without published numbers on directive-detection precision. Per `V2_REENTRY_QUEUE.md` decision #1 (strict bench gate), this defers until evidence lands.
3. **Effort: medium.** ~150 LOC + the TODO lifecycle (escalation, `detect_repetition`, `check_escalation`) is meaningful surface.

### H2 compliance audit — drop

1. **`command_succeeds:cmd` runs arbitrary shell on the user's machine.** Predicate strings come from belief content. Belief content comes from prompts, scanner output, and onboard. Any path that can write a belief can therefore inject a shell command.
2. **No security model for the predicate language.** The research line stored predicates as `TEXT` and evaluated them. There's no allowlist, no sandbox, no provenance check on the predicate's origin.
3. **No benchmark backs the capability.** "These compliance checks would be useful" is editorial; "this audit moves a measured number" is not demonstrated.
4. **`file_contains` and `config_value` alone aren't worth the schema cost.** A subset port that excludes `command_succeeds` is mechanically possible but loses most of the original value, and the schema (predicate column on locked beliefs) still has to ship.

If a future user actually needs compliance auditing, the right path is a separate, opt-in module with: an explicit allowlist of predicate types, a security review, and a benchmark. Not a quiet port at v2.0.

## Decision asks

- [ ] **Confirm split.** Issue #199 becomes three sub-issues: H3 (ship), H1 (defer + benchmark gate), H2 (drop).
- [ ] **H3 scoring.** Query overlap × `posterior_mean` is the recommended formula. Confirm, or specify alternative? Locked beliefs have saturated posteriors so the second factor is approximately constant — could simplify to query overlap alone.
- [ ] **H3 max_k default.** Research line used 5. Ship 5? Configurable?
- [ ] **H1 deferral language.** What evidence reopens H1? (Recommendation: ≥80% precision + ≥60% recall on a labeled sample of 200 coding prompts. If that bar isn't met, H1 stays deferred.)
- [ ] **H2 drop confirmation.** Final decision: drop, not defer. Removing the temptation to revisit "just port the safe predicates" prevents incremental erosion of the security stance.

## Downstream impact (as shipped)

- **H3 shipped at v2.0 (#373).** UserPromptSubmit applies top-K (default `locked_max_k = 5`) by query overlap × `posterior_mean(alpha, beta)`. SessionStart no longer emits a `<aelfrice-baseline>` block under the default; both halves of the legacy contract revive together via `[user_prompt_submit_hook] inject_all_locked = true`. Per-session injection counts are surfaced by `aelf tail` (live-tail of the per-turn audit log; each record carries `n_beliefs` and `n_locked`).
- **H1 split + deferred (#374).** Bench gate: ≥80% precision + ≥60% recall on a labeled sample of 200 coding prompts before reopening.
- **H2 dropped.** v2.0 ships without a compliance-audit predicate language; the research line's `enforcement.py` H2 is not ported. The `command_succeeds:cmd` predicate would have executed arbitrary shell from belief content, with no allowlist or sandbox.

## Provenance

Research-line module: `agentmemory/enforcement.py` (~347-373 LOC depending on extension surface).
Lab parity audit: `aelfrice-lab/docs/agentmemory-parity-audit-2026-04-28.md` § 14.
Substrate ratification: [substrate_decision.md](substrate_decision.md) (Option B). Substrate-neutral; H3 scoring works on scalar `posterior_mean`.
