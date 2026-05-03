# V2 re-entry queue

Index of v2.0-deferred capabilities and the evidence required to reopen each. Referenced by `docs/v2_*.md` spec memos as the durable home for "deferred, bench-gated" decisions.

This file records what was deferred, why, and what evidence reopens the decision. It does not promise re-entry. A capability stays out until its row's gate is met *and* a fresh ratification ships.

## Decision #1 — strict bench-impact gate

The default disposition for any v2.0 capability whose spec lacks a published benchmark is **defer**. A deferred capability re-enters the v2.x scope only when:

1. A labeled benchmark exists against which the capability can be measured. The benchmark itself ships first (typically a `benchmarks/<name>/` directory with fixtures and a runner) and is reviewable independently of the capability port.
2. The capability moves a measured number on that benchmark by the threshold listed in the row below — not "feels better in spot checks."
3. Any privacy / determinism / cost regression introduced by the capability has an explicit user-facing surface (PRIVACY.md update, opt-in flag, or LIMITATIONS § Sharp edges entry). A capability that quietly broadens the trust surface does not re-enter on bench numbers alone.

The gate is intentionally strict. The research-line codebase shipped most of these capabilities without published evidence; v2.0's premise is that we don't repeat that.

## Rows

Each row references its public spec memo and the GitHub issue that tracks re-entry.

### Multi-LLM consensus (multimodel) — issue #198

- **Spec:** [`docs/v2_multimodel.md`](v2_multimodel.md)
- **Disposition:** defer to v2.x.
- **Gate:** ≥3pp improvement on a classification-accuracy or onboard-quality `aelf bench` number, **and** a PRIVACY.md rewrite documenting per-provider trust surface and consent gate.
- **Why deferred:** privacy regression (outbound to *N* providers vs. 1), determinism regression (non-deterministic by construction), *N×* per-onboard token cost, no published evidence that cross-model consensus measurably improves classification quality. Substrate-decision (#196 Option B) removed the strongest historical justification.

### Directive detection (enforcement H1) — issue #374

- **Spec:** [`docs/v2_enforcement.md`](v2_enforcement.md) § H1
- **Disposition:** defer to v2.x. (H3 selective injection ships at v2.0; H2 compliance audit is dropped, not deferred.)
- **Gate:** ≥80% precision and ≥60% recall on a labeled sample of 200 coding prompts. Below that bar, the 29-verb regex's false-positive rate (e.g., "I never push to main when I'm tired" misclassified as a directive) pollutes the belief graph faster than the captured directives help.
- **Why deferred:** high false-positive risk on coding prompts; no published precision/recall numbers from the research line; ~150 LOC + TODO lifecycle is meaningful surface to ship without evidence.

### Phantom promotion-trigger rule — issue #229

- **Spec:** [`docs/v2_phantom_promotion_trigger.md`](v2_phantom_promotion_trigger.md)
- **Disposition:** bench-gated for v2.0 ship.
- **Gate:** ≥90% precision and ≥70% recall on the labeled corpus described in the spec's § Labeled-corpus benchmark. The rule itself is binary (`aelf:validate` typed, or content-matching `aelf:lock` typed) and has no tunable threshold; the benchmark verifies that the rule has acceptable recall against phantoms users would validate if asked, and acceptable precision against phantoms users would reject.
- **Why bench-gated:** three naive triggers (N positive feedback, retrieval count, user_corrected adjacency) were rejected for conflating posterior movement with explicit validation, selection bias on user query distribution, and adjacency ≠ acknowledgment respectively. Without the benchmark, there is no way to verify the explicit-acknowledgment rule has acceptable recall — and a low-recall promotion path is silently equivalent to "no promotion path" for most users.
- **Failure mode:** if the gate is missed, #229 stays open and the wonder line item ships without auto-promotion (phantoms remain `origin = speculative` indefinitely until the user manually validates).

## Out of queue (resolved by spec, not deferred)

For reference. These were considered for the queue and routed elsewhere:

- **Compliance audit (enforcement H2)** — [`v2_enforcement.md`](v2_enforcement.md) § H2. **Dropped, not deferred.** Removing the temptation to revisit "just port the safe predicates" prevents incremental erosion of the security stance. A future user actually needing this should propose a separate, opt-in module with an allowlist and a security review.
- **Deduplication (#197)** — [`v2_dedup.md`](v2_dedup.md). Ships at v2.0 per spec; the spec references this queue's decision #1 only as the gate language for `aelf:dedup`-as-MCP-tool, which is itself out of scope for v2.0.
- **Semantic contradiction detector (#201)** — [`v2_relationship_detector.md`](v2_relationship_detector.md). Ships at v2.0 per spec.

## Adding a row

When a v2 spec memo recommends defer, add a row here in the same PR that lands the spec. Each row needs: spec link, disposition, gate (concrete numbers or a binary criterion), and why-deferred. A row without a measurable gate is not a re-entry row — it's a wishlist, and belongs in a separate doc (or nowhere).

When a row's gate is met, the re-entry PR removes the row in the same commit that lands the implementation. The git history of this file is the audit trail.
