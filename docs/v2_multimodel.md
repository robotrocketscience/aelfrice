# v2.0 evaluation: multi-LLM consensus module

Spec for issue [#198](https://github.com/robotrocketscience/aelfrice/issues/198). Substrate-cascade addendum to [`substrate_decision.md`](substrate_decision.md) (#196 ratified Option B).

Status: spec, no implementation. **Recommendation: defer to v2.x with a strict evidence-gate.**

## What's being decided

Whether to port the research-line `multimodel.py` (~143-189 LOC) to v2.0. The module runs the same prompt across multiple providers (Claude + GPT + local), takes a majority vote, and tags low-consensus beliefs `origin = multi_model_disagreement`. v1.x routes `--llm-classify` through Claude Haiku alone.

## Substrate dependency

None mechanically. But under Option B (substrate_decision.md ratified), per-axis uncertainty was deliberately dropped — and the research line used cross-model classification primarily to populate per-aspect `(α_i, β_i)` from prose. With single-axis substrate, multimodel's strongest historical justification disappears.

## Recommendation

**Defer to v2.x. Ship only with a benchmark.**

Five reasons:

1. **Privacy regression.** PRIVACY.md caps optional outbound at "Claude Haiku via `ANTHROPIC_API_KEY` for `--llm-classify`." Multi-provider consensus expands the trust surface to *N* third parties. That requires a deliberate PRIVACY.md rewrite, not a quiet ship. Some users running aelfrice precisely *because* it's local-first would treat this as a regression.
2. **Determinism regression.** Multi-model consensus is non-deterministic by construction. Same prompt, different runs, different outputs. PHILOSOPHY's determinism property breaks.
3. **Cost.** *N×* the per-onboard token spend. Onboard is already the most expensive operation in the system.
4. **No benchmark backs cross-model improvement.** The research line shipped multimodel without publishing evidence that consensus measurably improves classification quality over single-model. `V2_REENTRY_QUEUE.md` decision #1 (strict bench-impact gate) applies. Until consensus moves an `aelf bench` number ≥ 3pp on classification accuracy or onboard quality, the cost isn't justified.
5. **Substrate decision removed the strongest use case.** Under single-axis substrate, the per-aspect classification that multimodel was originally meant to support doesn't exist.

## Decision asks

- [ ] **Confirm defer.** If yes, this issue moves from `v2.0` to `v2.x` and gets a `bench-gated` label. No port until evidence lands.
- [ ] **If override (ship at v2.0):** what evidence justifies it? Without a benchmark, the privacy + determinism + cost trio outweighs the speculative gain.
- [ ] **Alternative:** ship `multimodel` as opt-in via `.aelfrice.toml` with explicit user consent and a flag in `aelf health` showing it's enabled. Reduces the surprise factor but doesn't fix the determinism break for users who turn it on.

## Downstream impact (defer path)

- No code change at v2.0.
- `V2_REENTRY_QUEUE.md` row stays "defer" with the new gate language: "≥3pp classification accuracy improvement on `aelf bench` AND a privacy-doc rewrite."
- Issue #198 keeps `v2.0` label removed; gains `bench-gated` label.

## Provenance

Research-line module: `agentmemory/multimodel.py` (~143-189 LOC depending on extension surface).
Lab parity audit: `aelfrice-lab/docs/agentmemory-parity-audit-2026-04-28.md` § 5.
Substrate ratification: [substrate_decision.md](substrate_decision.md) (Option B). Note: the substrate decision removed multimodel's strongest historical justification.
