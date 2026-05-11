# v3.0 scope memo: multimodel

Spec for issue [#607](https://github.com/robotrocketscience/aelfrice/issues/607). Successor to [`v2_multimodel.md`](v2_multimodel.md), which deferred [#198](https://github.com/robotrocketscience/aelfrice/issues/198) (multi-LLM consensus) to v2.x with a strict bench gate. This memo ratifies the v3.0 disposition of that deferral.

## Decision

**Subsumed.** Multi-LLM consensus is not in v3.0 scope and does not move to a v3.x bench-gated queue. The "different models surface different gaps" use case is covered by wonder-dispatch ([#542](https://github.com/robotrocketscience/aelfrice/issues/542) umbrella) plus the research-agent dispatch surface that shipped at v2.1 ([#551](https://github.com/robotrocketscience/aelfrice/issues/551), closed 2026-05-10).

No new module. No `anthropic` / `openai` / multi-provider client surface added to aelfrice. The polymorphic pattern established by `/aelf:onboard` and `aelf:wonder` — aelfrice writes the structured request, the host harness runs the model call(s) with its own credentials, aelfrice ingests the structured response — is the v3 answer to the question multimodel was meant to answer.

## Why

The five reasons in [`v2_multimodel.md`](v2_multimodel.md) § *Recommendation* (privacy, determinism, cost, no-benchmark, substrate-removed-justification) are unchanged at v3.0. None of v2.1's defaults flips weakened any of them: PRIVACY.md still caps optional outbound at single-provider Claude Haiku for `--llm-classify`, the determinism property has if anything *hardened* under [#605](https://github.com/robotrocketscience/aelfrice/issues/605)'s relatedness ratification (see [`v3_relatedness_philosophy.md`](v3_relatedness_philosophy.md)), and no bench result has landed showing cross-model voting moves an `aelf bench` number.

What changed is the alternative surface. At v2.0 the only way to get "this question deserves more than one angle" was multi-provider voting on a single prompt. At v2.1 the wonder-dispatch surface ships:

- `wonder.dispatch.analyze_gaps(query, …)` — pure function, fixed-store snapshot in, named gap set out. No LLM, no randomness. ([`src/aelfrice/wonder/dispatch.py`](../src/aelfrice/wonder/dispatch.py))
- `wonder.dispatch.generate_research_axes(gap_analysis, agent_count)` — 2–6 orthogonal axes per query, deterministic given the analysis.
- MCP `wonder()` tool + `aelf wonder <query> --axes` CLI — both wrap the same two pure functions.
- Skill-layer integration ([#552](https://github.com/robotrocketscience/aelfrice/issues/552) in flight) — the host harness fans out one research agent per axis, each agent runs its own model call with its own credentials, results pipe back through `wonder_ingest`.

The polymorphic split delivers what multi-provider voting was *for* — surfacing the questions one model misses — without paying the multi-provider cost trio. Each axis is investigated by an agent in the host's own dispatch surface; the host can route different axes to different models if the user has those credentials configured, and aelfrice neither knows nor cares. The "multiple models surface different gaps" property holds at the dispatch layer instead of the consensus layer, and it composes with the determinism property because aelfrice's analysis half stays pure.

Multi-provider voting on a single prompt is a narrower mechanism than research-axis dispatch. Voting collapses N answers to one with no audit trail of *why* they disagreed; axis dispatch produces N investigations whose disagreements are first-class observable rows. The voting shape is the one v2 deferred; the dispatch shape is the one v2.1 shipped. There is no remaining gap that voting would fill that dispatch does not.

## What this rules out, explicitly

- A `multimodel.py` module in `src/aelfrice/`. Not at v3.0, not as opt-in via `.aelfrice.toml`, not as a `[multimodel]` extra. The research-line module of that name (`agentmemory/multimodel.py`) does not port.
- Direct `anthropic`, `openai`, or any other model-provider SDK dependency in aelfrice's runtime install footprint. The existing default-path-reach guard test stays load-bearing.
- A v3.x successor ticket of the form `feat/issue-NNN-multimodel-consensus-with-bench-gate`. Filed as a non-issue: the deferral path is closed, not bumped. Any future re-open requires the amendment gate in *What this leaves open* below.
- An `origin = multi_model_disagreement` belief tag or any persistence surface that assumes cross-provider voting at write time. The store schema does not gain a column for it.

## What this leaves open

- **Host-harness routing of dispatch axes to different models.** If a user configures their host harness to send axis-1 research to Claude and axis-2 to GPT-4o, that's a host configuration question, not an aelfrice change. The aelfrice surface stays provider-agnostic; the structured axis payload travels through the host's existing dispatch mechanism.
- **Lab-side bench experiments** in `~/projects/aelfrice-lab` that measure whether dispatched research-axis disagreement *between providers* yields different `wonder_ingest` outcomes than single-provider dispatch. The lab is the right place to characterize that; nothing in this memo constrains lab investigation.
- **Re-opening this memo at v3.x.** If a future bench (lab or otherwise) lands evidence that single-prompt cross-provider voting produces a result that cannot be reproduced by research-axis dispatch — i.e., a real capability gap, not a stylistic difference — file a new issue citing this memo, naming the specific capability gap, and proposing the minimum amendment to `PHILOSOPHY.md` § *Determinism is the property* and `PRIVACY.md` § *Optional outbound* required to admit the change. The amendment is the gate, not the SDK.

## Acceptance

- [x] Decision recorded: subsumed.
- [x] Subsumption rationale grounded in the v2.1 wonder-dispatch surface (`analyze_gaps`, `generate_research_axes`, skill-layer dispatch).
- [x] Five v2 deferral reasons reaffirmed unchanged at v3.0.
- [x] Explicit exclusions listed so future tickets don't have to re-litigate the boundary.
- [x] v2 memo gets a one-line successor pointer (separate commit).

## Out of scope

- Implementing wonder-dispatch's skill-layer integration. That is [#552](https://github.com/robotrocketscience/aelfrice/issues/552), in-flight, tracked separately on the v3.0 milestone.
- Editing `PHILOSOPHY.md` § *Determinism* or `PRIVACY.md` § *Optional outbound*. The decision is to leave them as-is; no doc surgery beyond cross-references.
- Closing [#198](https://github.com/robotrocketscience/aelfrice/issues/198). Already CLOSED COMPLETED 2026-05-02 under the v2 deferral; this memo is the v3 successor decision, not a re-close.

## Refs

- [`v2_multimodel.md`](v2_multimodel.md) — predecessor deferral memo (five reasons reaffirmed)
- [`substrate_decision.md`](substrate_decision.md) — Option B ratified; removes multimodel's strongest historical justification (per-aspect classification)
- [`v3_relatedness_philosophy.md`](v3_relatedness_philosophy.md) — sibling v3 design-cut memo; same determinism-property posture
- [`PHILOSOPHY.md`](PHILOSOPHY.md) § *Determinism is the property*
- [`PRIVACY.md`](PRIVACY.md) § *Optional outbound*
- [`src/aelfrice/wonder/dispatch.py`](../src/aelfrice/wonder/dispatch.py) — subsuming surface
- [#198](https://github.com/robotrocketscience/aelfrice/issues/198) — multi-LLM consensus (CLOSED COMPLETED, deferred to v2.x; superseded here)
- [#542](https://github.com/robotrocketscience/aelfrice/issues/542) — wonder-consolidation umbrella
- [#551](https://github.com/robotrocketscience/aelfrice/issues/551) — research-agent dispatch core (shipped 2026-05-10)
- [#552](https://github.com/robotrocketscience/aelfrice/issues/552) — wonder-dispatch skill-layer integration (in flight)
- [#608](https://github.com/robotrocketscience/aelfrice/issues/608) — v3.0 milestone tracker
