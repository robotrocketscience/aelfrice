# v2.0 evaluation: sentiment-from-prose feedback

Spec for issue [#193](https://github.com/robotrocketscience/aelfrice/issues/193). Substrate-cascade addendum to [`substrate_decision.md`](substrate_decision.md) (#196 ratified Option B).

Status: spec, no implementation. **Recommendation: ship at v2.0, opt-in via `.aelfrice.toml`, off by default.**

## What's being decided

Whether to port the research-line `sentiment_feedback.py` (~174 LOC, stdlib-only) to v2.0. The module reads each user prompt, regex-matches against ~24 patterns ("ok good", "yes", "perfect" / "no thats wrong", "fix it", "i told you"), and distributes the resulting `SentimentSignal` across the previous turn's retrieved beliefs as `feedback_history` rows. v1.x requires explicit `aelf feedback <id> used|harmful`.

## Substrate dependency

None. The module emits scalar `(used|harmful, weight)` events compatible with v1.x's existing `apply_feedback` path. Substrate Option B is exactly what this module wants.

## Recommendation

**Ship at v2.0. Opt-in only. Default off.**

Three reasons:

1. **Real UX gap.** Explicit `aelf feedback` calls are the right primitive but practically no one uses them mid-session. The implicit feedback loop is what makes the feedback mechanism actually closed in chat-driven workflows. Without it, `apply_feedback` is plumbing nobody exercises (the existing concern flagged in #127's history).
2. **Cheap and deterministic.** Pure regex over text. No LLM, no outbound calls. Compatible with PHILOSOPHY's determinism + local-first properties.
3. **Privacy concern is real but addressable.** The hook reads every prompt and silently mutates belief confidence. That's a meaningful change in what aelfrice does with user prose. Solution: opt-in via `.aelfrice.toml`, explicit `aelf health` surfacing showing it's enabled, and a PRIVACY.md entry covering the prose-inspection surface. Off by default means existing users see no behavior change.

## Decision asks

- [ ] **Confirm opt-in port at v2.0.** If yes: ~174 LOC port + `.aelfrice.toml` flag (`[feedback] sentiment_from_prose = false`) + PRIVACY.md entry + `aelf health` surfacing.
- [ ] **Pattern set.** Research line shipped 12 positive + 12 negative patterns. Ship those verbatim, or curate? (Recommendation: ship verbatim to minimize port surface; users can override via config in v2.x if needed.)
- [ ] **`detect_correction_frequency` (the ≥40% recent-turns escalator).** Ship with the base module, or defer? (Recommendation: ship together; it's the strongest signal the regex-only path produces and it's stateless beyond a sliding window.)
- [ ] **Distribution target.** Apply the signal to *all* previous-turn retrieved beliefs equally, or scale by retrieval rank? (Recommendation: equally — matches the research-line behavior; ranked distribution adds a knob without an evidence-gate.)
- [ ] **Audit row schema.** `feedback_history` already has the columns. Confirm the new event tag (`origin = sentiment_inferred`) doesn't collide with #224's `origin always 'unknown'` fix path.

## Privacy review (the v2.0 decision the title asks for)

- **What's read:** every user prompt the hook sees (already true today for retrieval — the new behavior is regex matching, not new data access).
- **What's stored:** an audit row in `feedback_history` per matched pattern, with the matched substring + pattern ID + affected belief IDs. No raw prompt text is stored.
- **What leaves the machine:** nothing. Stdlib regex; no outbound calls.
- **Determinism:** same prompt, same matches, same updates. Reproducible.

The privacy story is "we are now doing more with prompts you already submitted to the hook" — not "we are seeing new content." That distinction belongs in PRIVACY.md as the second-paragraph clarification.

## Downstream impact

- New `.aelfrice.toml` section: `[feedback] sentiment_from_prose = false`.
- New `aelf health` line: `Sentiment-from-prose feedback: enabled / disabled` and pattern-match count if enabled.
- New PRIVACY.md paragraph under "Optional outbound": clarify that this feature is *inbound* prose inspection, no outbound traffic.
- Tied to #224 (origin tagging) — sentiment-inferred events need a stable origin tag that survives the #224 fix.

## Provenance

Research-line module: `agentmemory/sentiment_feedback.py` (~174 LOC, stdlib-only).
Lab parity audit: `aelfrice-lab/docs/agentmemory-parity-audit-2026-04-28.md` § 9.
Substrate ratification: [substrate_decision.md](substrate_decision.md) (Option B). Substrate-neutral; this module emits scalar feedback events.
