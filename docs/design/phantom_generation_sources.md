# Broadening phantom-generation sources — R&D memo

**Status:** R&D complete, 2026-07-20. Recommendation: **park all proposed new
generation sources; the binding constraint is downstream, not supply.**
**Tracking issue:** [#1125](https://github.com/robotrocketscience/aelfrice/issues/1125)
(R&D: broaden phantom-belief generation sources beyond `wonder_ingest`).
**Prior art (do not re-litigate):**
[#980](https://github.com/robotrocketscience/aelfrice/issues/980) /
`docs/design/phantom_trigger_generation.md` (RATIFIED 2026-06-23),
[#605](https://github.com/robotrocketscience/aelfrice/issues/605) PHILOSOPHY
lock.

## 0. TL;DR

The issue asks whether aelfrice should grow additional sources of speculative
("phantom") beliefs beyond the single `wonder_ingest` writer. Four research
dives plus two empirical rounds against seven real stores say **no** for every
concrete candidate, for two independent reasons:

1. **The premise mis-locates the bottleneck.** Phantom *supply* is thin, but
   supply is not what's limiting the feature. Phantoms are barely consumed and
   have **never once been promoted** in any real store, even though the
   promotion path is fully wired. More generation feeds rows into a pipeline
   whose downstream throughput is measured at zero. (§3)
2. **The load-bearing constraint (#980 §2) forces every "new source" into one
   of two shapes**, and both are already accounted for: a *detector* (respects
   §2, but is just an extension of the shipped #980 trigger) or a *mechanical
   generator* (violates §2's spirit and produces malformed non-beliefs — proven
   in R2). There is no third shape. (§4–5)

The one defensible, in-scope action is not a new source at all: exercise and
close the **already-built** lifecycle (detector → dispatch → promotion → GC),
whose every stage is gated behind a manual command nobody runs. (§6)

## 1. What "generate a phantom" actually requires

Every phantom is a `type='speculative'` / `origin='speculative'` belief, and
the only writer is `wonder_ingest()` (`wonder/lifecycle.py`), reachable through
three "wonder" front doors (`aelf wonder --persist-docs`, `aelf wonder
--persist`, MCP `wonder_persist`). `phantom_trigger.py` (#980) is **not** a
generator — by contract it only *detects that a generation opportunity exists*
and emits an `<aelfrice-phantom-opportunity>` note; synthesis + persistence are
deferred to an explicit wonder invocation.

The reason the writer is single and explicit is **#980 §2**, the non-negotiable
boundary:

> aelfrice's Python never calls an LLM or the network. aelfrice can decide
> *when* a phantom-generation opportunity exists, but it cannot perform the
> generation. Generation is an LLM act and must run under the host agent's
> credentials.

A phantom is not a raw text fragment. It is a *synthesized speculative claim*
that connects existing beliefs — the content is the distillation. Producing
that distillation is the LLM act §2 forbids aelfrice from performing. This is
the fact that every candidate source below runs into.

## 2. Method

- **R0 — census** across all seven organic per-project stores on the dev
  machine (`.git/aelfrice/memory.db`), plus the loaded bench store. Measures
  prevalence, retrieval exposure, feedback, and promotion of existing phantoms.
- **R2 — mechanical-quality** experiment: deterministically (seeded, no
  wall-clock — #605) sample 200 conversation turns from the largest real
  transcripts, treat each as a candidate "mechanically generated" phantom, and
  compare (a) near-duplicate rate against the existing belief store and (b)
  signal density against the 39 real LLM-synthesized phantoms in the aelfrice
  store.
- Volume/GC steady-state is modelled analytically from the R0 feedback rate and
  the shipped `wonder_gc` eligibility rule.

(R1/R3/R4 collapsed into R0/R2 once the census made the premise-level finding
decisive; recording the numbers, not the dead branches.)

## 3. Evidence: supply is not the binding constraint

**R0 census.** Speculative beliefs across the seven organic stores:

| Signal | Result |
|---|---|
| Prevalence | 0.02 %–0.9 % of beliefs per store; several real stores have **0** |
| Total phantoms observed | ~101 |
| Ever retrieved (`last_retrieved_at` set) | ~21 % |
| Ever earned feedback (priors moved off ingest defaults) | ~5 % |
| Ever **promoted** (`origin` flipped off `speculative`) | **0** |
| Ever **GC'd** (`wonder_gc` soft-delete) | **0** (corroborates the #980 audit) |

The promotion path is not missing — `promote()` for phantoms is wired
(`cli.py`, "#550 Surface B: promote any speculative phantom"), and
`find_promotable_snapshots` exists. It has simply **never fired** in a real
store, because it (like generation and GC) is gated behind an explicit manual
command. The entire phantom lifecycle — generate → promote → GC — is
manual-invocation-only, and in practice only the first stage is ever invoked,
rarely.

**Implication.** Adding generation sources increases inflow to a pipeline whose
promotion throughput is exactly zero and whose GC is default-off. That is the
issue's own stated guardrail ("unbounded generation without a paired GC/quality
gate is a regression risk") turned from a hypothetical into a measurement. The
lever labelled "more sources" pushes on the stage that is *not* limiting.

## 4. Evidence: mechanical generation produces non-beliefs

The marquee candidate ("random sampling from jsonls") is the only proposal that
lets aelfrice mint a phantom *without* an LLM — it would use the sampled record
as the belief content directly. R2 tested that content.

**R2 results** (200 seeded-sampled transcript turns vs store):

| Metric | Sampled raw turns | LLM-synthesized phantoms |
|---|---|---|
| Carry a declarative-claim marker | **40 %** | **97 %** |
| Median length | **142 chars** | **1 758 chars** |
| Near-duplicate of an existing belief (Jaccard ≥ 0.5) | 5.5 % | — |

Two findings, one of which *refuted* a prior hypothesis:

- **Redundancy was NOT the problem** (hypothesis refuted). I expected sampled
  turns to mostly duplicate existing beliefs, since auto-capture already
  ingests transcripts. They don't (5.5 % near-dup, median best-Jaccard 0.21) —
  because auto-capture *distills* beliefs rather than copying turns verbatim, so
  raw turns keep a distinct surface form.
- **Malformation is the problem.** Raw turns are conversational fragments —
  under half carry any declarative content, a median 142 characters, frequently
  mid-thought slices that are not standalone claims of any kind. They are
  dialogue, not beliefs. The 12× length gap and the 40 %→97 % claim-marker gap
  quantify what §1 predicts: **the value of a phantom is the LLM distillation,
  and mechanical sampling skips exactly that step.**

Routing the sampled record *through* an LLM to distill it into a real belief is
possible — but that is no longer "a new source," it is the existing
`/aelf:wonder` dispatch with a sampled seed topic, and it inherits every
property (and cost) of wonder.

## 5. Candidate sources — determinism / volume-GC / quality / verdict

Every candidate collapses onto the §1 dichotomy: it is either a **detector**
(class A — deterministic, emits an opportunity note, synthesis stays in the LLM
dispatch; this *is* the shipped #980 mechanism) or a **mechanical generator**
(class B — mints belief content without an LLM; R2 shows this is malformed).

| Candidate source | Class | Determinism | Volume / GC | Quality gate | Verdict |
|---|---|---|---|---|---|
| **Random / stratified JSONL sampling** | B (or A if LLM-distilled) | Seedable ✓ | Unbounded inflow; no paired GC | R2: fragments, 40 % claim-bearing | **PARK.** Mechanical form fails R2; LLM-distilled form is just wonder-with-a-seed-topic. |
| **Passive background consolidation** (close the detect→generate gap) | Needs an LLM to close ⇒ A via `auto_dispatch` | Detection ✓; synthesis non-det (host LLM) | Bounded by #980 per-session budget | Inherits wonder quality | **ALREADY SHIPPED.** The gap is closed by the #980 `auto_dispatch` sub-flag (host agent acts on the note), not by aelfrice generating. No new code; a docs clarification at most. |
| **Contradiction-pair driven** | A (signal *c* already exists) | ✓ | Bounded | **No substrate** — CONTRADICTS edges are #988-only and rare; the pair-detector measured 0 % precision on real pairs | **PARK.** Inert; nothing to drive it. |
| **Entity-cluster driven** | A (extends signal *b*) | ✓ | Bounded | Marginal over the shipped new-entity signal | **PARK.** At best a thin follow-up to #980 signal *b*; consumption (§3), not detector coverage, is the limit. |
| **Cold-store seeding** | B (no source material) | n/a | n/a | Cannot synthesize from an empty store without an LLM | **PARK.** A cold store has nothing to wonder *about*; seeding it mechanically is §4 again. |

**Volume/GC steady-state** (for any source generating R phantoms/session, ~3
sessions/day): with GC default-off (the shipped default, and the #980 audit's
observed reality of 0 GC'd) accumulation is linear and unbounded — ~270 rows at
R = 1, ~5 400 at R = 20, over 90 days. Turning GC on bounds the ~95 %
GC-eligible fraction at the 14-day window, but the ~5 % that earn feedback
escape GC forever and grow without bound. **No proposed source ships a paired
default-on GC**, so each one is a monotonic-accumulation regression against the
issue's own guardrail.

## 6. Recommendation

**Park all five candidate generation sources.** None survives both filters:
the mechanical ones (JSONL sampling, cold-store seeding) fail R2; the detector
ones (contradiction, entity-cluster) either lack substrate or are marginal
extensions of the already-shipped #980 trigger; and passive consolidation is
already the shipped `auto_dispatch` flag, not new code.

The evidence redirects effort from *supply* to the **downstream stages that are
built but never exercised**:

1. **Promotion is wired but has fired zero times.** The highest-value follow-up
   is understanding why (`promote()` for phantoms exists; nothing calls it) and
   whether promotion should have an automatic, deterministic trigger analogous
   to the #980 detector — an opportunity note, not an auto-write. Tracked under
   the #980 umbrella's promotion-wiring item.
2. **GC is default-off, so any future inflow accumulates.** If generation is
   ever broadened, a paired default-on GC is a hard precondition, not a later
   slice. This memo is the audit trail for that precondition.
3. **Exercise the shipped #980 detector** (default-off today; R0 shows it
   produces ~0 phantoms in practice — the observed phantoms are all explicit
   wonder research docs). Broadening *detection* is only worthwhile once
   consumption/promotion demonstrably converts phantoms into value; today it
   does not.

If the operator still wants a broadened *source* despite §3, the single least
objectionable option is **JSONL-sampling as a detector, not a generator**: emit
a `<aelfrice-phantom-opportunity>` note naming a deterministically-sampled topic
and let the existing dispatch distill it under host credentials. That respects
§2, adds no low-quality rows (it emits notes, not phantoms), and is a ~30-line
extension of `phantom_trigger.py`. It is offered as the fallback, not the
recommendation — §3 says the graph does not yet need more phantoms.

## 7. References

- #1125 — this R&D issue.
- #980 / `docs/design/phantom_trigger_generation.md` — the RATIFIED
  detect/flag/synthesize/persist architecture and §2 load-bearing constraint.
- #605 — PHILOSOPHY: deterministic, narrow surface; non-deterministic judgment
  lives in the consuming agent (locked).
- #988 — semantic-edge substrate (CONTRADICTS-only; default-off) — why the
  contradiction-driven source has no substrate.
- `wonder/lifecycle.py` (`wonder_ingest`, `wonder_gc`), `promotion.py`
  (`promote`), `store.py` (`find_promotable_snapshots`) — the built-but-unexercised
  lifecycle.
